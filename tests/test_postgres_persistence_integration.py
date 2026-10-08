from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from backend.db import SessionFactory
from backend.models import Document, InterviewSession, User


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("ARIA_RUN_POSTGRES_TESTS") != "1",
        reason="set ARIA_RUN_POSTGRES_TESTS=1 for the isolated PostgreSQL contract test",
    ),
]


def test_postgres_enforces_document_kind_and_owner_for_interview_inputs():
    async def scenario():
        now = datetime.now(timezone.utc)
        request_ids = [uuid.uuid4() for _ in range(5)]
        run_id = uuid.uuid4().hex
        async with SessionFactory() as db:
            owner = User(email=f"owner-{run_id}@example.invalid", last_login_at=now)
            other = User(email=f"other-{run_id}@example.invalid", last_login_at=now)
            db.add_all([owner, other])
            await db.flush()
            resume = Document(
                owner_id=owner.id,
                kind="resume",
                original_filename="resume.pdf",
                object_key=f"{owner.id}/{uuid.uuid4()}.pdf",
                sha256="a" * 64,
                size_bytes=100,
                media_type="application/pdf",
                status="ready",
                extracted_text="Synthetic résumé",
                extraction_version="test",
                page_count=1,
                client_request_id=request_ids[0],
                request_hash="b" * 64,
            )
            jd = Document(
                owner_id=owner.id,
                kind="job_description",
                original_filename="jd.pdf",
                object_key=f"{owner.id}/{uuid.uuid4()}.pdf",
                sha256="c" * 64,
                size_bytes=100,
                media_type="application/pdf",
                status="ready",
                extracted_text="Synthetic job description",
                extraction_version="test",
                page_count=1,
                client_request_id=request_ids[1],
                request_hash="d" * 64,
            )
            db.add_all([resume, jd])
            await db.flush()
            valid = InterviewSession(
                owner_id=owner.id,
                resume_document_id=resume.id,
                jd_document_id=jd.id,
                requested_role="backend_developer",
                client_request_id=request_ids[2],
                request_hash="e" * 64,
                consent_version="test",
                consented_at=now,
                provenance={"schema_version": 1},
            )
            db.add(valid)
            await db.commit()
            valid_id = valid.id
            resume_id = resume.id
            jd_id = jd.id
            owner_id = owner.id
            other_id = other.id

            invalid = InterviewSession(
                owner_id=other.id,
                resume_document_id=resume.id,
                jd_document_id=jd.id,
                requested_role="backend_developer",
                client_request_id=request_ids[3],
                request_hash="f" * 64,
                consent_version="test",
                consented_at=now,
                provenance={"schema_version": 1},
            )
            db.add(invalid)
            with pytest.raises(IntegrityError):
                await db.commit()
            await db.rollback()

            valid = await db.get(InterviewSession, valid_id)
            resume = await db.get(Document, resume_id)
            jd = await db.get(Document, jd_id)
            owner = await db.get(User, owner_id)
            other = await db.get(User, other_id)
            await db.delete(valid)
            await db.flush()
            await db.delete(resume)
            await db.delete(jd)
            await db.delete(owner)
            await db.delete(other)
            await db.commit()

    asyncio.run(scenario())
