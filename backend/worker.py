from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from sqlalchemy import select

from backend.db import SessionFactory
from backend.documents import process_document_job
from backend.interviews import prepare_interview_job
from backend.models import (
    Answer,
    AuthSession,
    BackgroundJob,
    Document,
    GenerationAttempt,
    InterviewConnectionLease,
    InterviewSession,
    OAuthIdentity,
    Question,
    RefreshToken,
    SessionCheckpoint,
    User,
    utcnow,
)
from backend.settings import get_settings
from backend.storage import LocalPrivateStorage


def append_deletion_ledger(operation_id: uuid.UUID, kind: str, target_id: uuid.UUID) -> None:
    settings = get_settings()
    settings.deletion_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps(
        {
            "operation_id": str(operation_id),
            "kind": kind,
            "target_id": str(target_id),
            "recorded_at": utcnow().isoformat(),
        },
        separators=(",", ":"),
    )
    with settings.deletion_ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(record + "\n")


async def purge_document(db, job: BackgroundJob) -> None:
    document = await db.get(Document, job.document_id, with_for_update=True)
    if not document:
        await db.delete(job)
        await db.commit()
        return
    append_deletion_ledger(job.id, "document", document.id)
    await LocalPrivateStorage(get_settings()).delete(document.object_key)
    await db.delete(document)
    await db.commit()


async def purge_session(db, job: BackgroundJob) -> None:
    interview = await db.get(InterviewSession, job.session_id, with_for_update=True)
    if not interview:
        await db.delete(job)
        await db.commit()
        return
    append_deletion_ledger(job.id, "interview", interview.id)
    document_ids = (interview.resume_document_id, interview.jd_document_id)
    await db.delete(interview)
    await db.flush()
    for document_id in document_ids:
        document = await db.get(Document, document_id, with_for_update=True)
        if document:
            references = (
                await db.execute(
                    select(InterviewSession.id).where(
                        (InterviewSession.resume_document_id == document_id)
                        | (InterviewSession.jd_document_id == document_id)
                    ).limit(1)
                )
            ).scalar_one_or_none()
            if not references:
                document.unreferenced_since = utcnow()
    await db.commit()


async def purge_account(db, job: BackgroundJob) -> None:
    user_id = job.owner_id
    user = await db.get(User, user_id, with_for_update=True) if user_id else None
    if not user:
        await db.delete(job)
        await db.commit()
        return
    append_deletion_ledger(job.id, "account", user.id)
    documents = (
        await db.execute(select(Document).where(Document.owner_id == user.id))
    ).scalars().all()
    for document in documents:
        await LocalPrivateStorage(get_settings()).delete(document.object_key)
    interviews = (
        await db.execute(select(InterviewSession).where(InterviewSession.owner_id == user.id))
    ).scalars().all()
    for interview in interviews:
        await db.delete(interview)
    await db.flush()
    for document in documents:
        await db.delete(document)
    identities = (
        await db.execute(select(OAuthIdentity).where(OAuthIdentity.user_id == user.id))
    ).scalars().all()
    sessions = (
        await db.execute(select(AuthSession).where(AuthSession.user_id == user.id))
    ).scalars().all()
    for identity in identities:
        await db.delete(identity)
    for session in sessions:
        await db.delete(session)
    other_jobs = (
        await db.execute(
            select(BackgroundJob).where(BackgroundJob.owner_id == user.id, BackgroundJob.id != job.id)
        )
    ).scalars().all()
    for other in other_jobs:
        await db.delete(other)
    await db.flush()
    await db.delete(job)
    await db.delete(user)
    await db.commit()


async def run_once() -> bool:
    async with SessionFactory() as db:
        job = (
            await db.execute(
                select(BackgroundJob)
                .where(
                    BackgroundJob.status == "queued",
                    BackgroundJob.available_at <= utcnow(),
                )
                .order_by(BackgroundJob.available_at, BackgroundJob.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
        ).scalar_one_or_none()
        if not job:
            return False
        if job.kind == "prepare_document":
            await process_document_job(db, job, get_settings())
        elif job.kind == "prepare_session":
            await prepare_interview_job(db, job)
        elif job.kind == "purge_document":
            await purge_document(db, job)
        elif job.kind == "purge_session":
            await purge_session(db, job)
        elif job.kind == "purge_account":
            await purge_account(db, job)
        else:
            job.status = "failed"
            job.error_code = "UNSUPPORTED_JOB_KIND"
            await db.commit()
        return True


async def run_forever() -> None:
    while True:
        if not await run_once():
            await asyncio.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process durable ARIA background jobs")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_once() if args.once else run_forever())


if __name__ == "__main__":
    main()
