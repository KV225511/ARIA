from __future__ import annotations

import asyncio
from datetime import timedelta
import hashlib
import re
import uuid

import fitz
from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import request_hash
from backend.models import BackgroundJob, Document, utcnow
from backend.settings import AppSettings
from backend.storage import LocalPrivateStorage


EXTRACTION_VERSION = "pymupdf-v1"


def safe_filename(value: str | None) -> str:
    name = (value or "document.pdf").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w .()\-]", "_", name, flags=re.UNICODE).strip(" .")
    return (name or "document.pdf")[:180]


async def read_pdf_upload(upload: UploadFile, settings: AppSettings) -> bytes:
    content = await upload.read(settings.max_pdf_bytes + 1)
    if len(content) > settings.max_pdf_bytes:
        raise HTTPException(413, "PDF exceeds the 10 MiB limit")
    if not content:
        raise HTTPException(422, "PDF is empty")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(415, "Only valid PDF files are accepted")
    return content


async def reserve_document(
    db: AsyncSession,
    owner_id: uuid.UUID,
    kind: str,
    upload: UploadFile,
    client_request_id: uuid.UUID,
    settings: AppSettings,
) -> tuple[Document, bool]:
    if kind not in {"resume", "job_description"}:
        raise HTTPException(422, "Document kind must be resume or job_description")
    content = await read_pdf_upload(upload, settings)
    checksum = hashlib.sha256(content).hexdigest()
    fingerprint = request_hash(kind, checksum, str(len(content)))
    existing = (
        await db.execute(
            select(Document).where(
                Document.owner_id == owner_id,
                Document.client_request_id == client_request_id,
            )
        )
    ).scalar_one_or_none()
    if existing:
        if existing.request_hash != fingerprint:
            raise HTTPException(409, "Idempotency key was already used for another upload")
        return existing, False
    document_id = uuid.uuid4()
    storage = LocalPrivateStorage(settings)
    object_key = storage.object_key(owner_id, document_id)
    document = Document(
        id=document_id,
        owner_id=owner_id,
        kind=kind,
        original_filename=safe_filename(upload.filename),
        storage_backend="local",
        object_key=object_key,
        sha256=checksum,
        size_bytes=len(content),
        media_type="application/pdf",
        status="uploading",
        client_request_id=client_request_id,
        request_hash=fingerprint,
    )
    db.add(document)
    await db.flush()
    try:
        await storage.put(object_key, content)
    except Exception:
        await db.rollback()
        raise HTTPException(503, "Private document storage is unavailable")
    document.status = "processing"
    db.add(
        BackgroundJob(
            owner_id=owner_id,
            document_id=document.id,
            kind="prepare_document",
            deduplication_key=f"prepare-document:{document.id}",
            payload={"schema_version": 1, "document_id": str(document.id)},
        )
    )
    await db.commit()
    return document, True


def extract_pdf(content: bytes, settings: AppSettings) -> tuple[str, int]:
    try:
        pdf = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("PDF_INVALID") from exc
    try:
        if pdf.needs_pass:
            raise ValueError("PDF_ENCRYPTED")
        if pdf.page_count < 1:
            raise ValueError("PDF_EMPTY")
        if pdf.page_count > settings.max_pdf_pages:
            raise ValueError("PDF_TOO_MANY_PAGES")
        text = "\n".join(page.get_text() for page in pdf).strip()
        if not text:
            raise ValueError("PDF_NO_EXTRACTABLE_TEXT")
        if len(text) > settings.max_extracted_characters:
            raise ValueError("PDF_TEXT_TOO_LARGE")
        return text, pdf.page_count
    finally:
        pdf.close()


async def process_document_job(
    db: AsyncSession, job: BackgroundJob, settings: AppSettings
) -> None:
    document = await db.get(Document, job.document_id, with_for_update=True)
    if not document or document.status in {"ready", "deleting"}:
        job.status = "succeeded" if document else "cancelled"
        await db.commit()
        return
    job.status = "running"
    job.attempt_count += 1
    job.lease_token = uuid.uuid4()
    job.leased_until = utcnow() + timedelta(minutes=2)
    document.status = "processing"
    await db.commit()
    try:
        content = await LocalPrivateStorage(settings).read(document.object_key)
        text, pages = await asyncio.wait_for(
            asyncio.to_thread(extract_pdf, content, settings), timeout=30
        )
        await db.refresh(job, with_for_update=True)
        document = await db.get(Document, job.document_id, with_for_update=True)
        if not document or document.status == "deleting":
            job.status = "cancelled"
        else:
            document.extracted_text = text
            document.page_count = pages
            document.extraction_version = EXTRACTION_VERSION
            document.error_code = None
            document.status = "ready"
            job.status = "succeeded"
    except (ValueError, asyncio.TimeoutError) as exc:
        await db.rollback()
        job = await db.get(BackgroundJob, job.id, with_for_update=True)
        document = await db.get(Document, job.document_id, with_for_update=True)
        code = str(exc) if isinstance(exc, ValueError) else "PDF_EXTRACTION_TIMEOUT"
        if document:
            document.status = "failed"
            document.error_code = code[:80]
        job.status = "failed"
        job.error_code = code[:80]
    except Exception:
        await db.rollback()
        job = await db.get(BackgroundJob, job.id, with_for_update=True)
        document = await db.get(Document, job.document_id, with_for_update=True)
        if document:
            document.status = "failed"
            document.error_code = "DOCUMENT_PROCESSING_FAILED"
        job.status = "failed"
        job.error_code = "DOCUMENT_PROCESSING_FAILED"
    job.leased_until = None
    job.lease_token = None
    await db.commit()
