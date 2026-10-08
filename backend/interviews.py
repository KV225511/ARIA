from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import hashlib
import uuid

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import request_hash
from backend.models import (
    Answer,
    BackgroundJob,
    Document,
    GenerationAttempt,
    InterviewConnectionLease,
    InterviewSession,
    Question,
    SessionCheckpoint,
    User,
    utcnow,
)
from backend.settings import AppSettings
from modules.module_05_ontology.graph import SkillOntologyGraph
from modules.module_05_ontology.grounding import (
    grounding_packet,
    normalize_generated_question,
    validate_grounded_question,
)
from modules.module_06_belief.belief_state import BeliefStateUpdater
from modules.module_08_llm.generator import build_question_retry_correction


TERMINAL_STATUSES = {"completed", "cancelled", "failed", "expired"}
ACTION_CYCLE = ("probe_foundation", "increase_difficulty", "switch_topic")


def _jsonable(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


async def prepare_interview_job(db: AsyncSession, job: BackgroundJob) -> None:
    interview = await db.get(InterviewSession, job.session_id, with_for_update=True)
    if not interview or interview.status != "preparing":
        job.status = "cancelled"
        await db.commit()
        return
    docs = (
        await db.execute(
            select(Document).where(
                Document.owner_id == interview.owner_id,
                Document.id.in_([interview.resume_document_id, interview.jd_document_id]),
            )
        )
    ).scalars().all()
    by_id = {doc.id: doc for doc in docs}
    resume = by_id.get(interview.resume_document_id)
    jd = by_id.get(interview.jd_document_id)
    job.status = "running"
    job.attempt_count += 1
    await db.commit()
    try:
        if not resume or not jd or resume.status != "ready" or jd.status != "ready":
            raise ValueError("DOCUMENTS_NOT_READY")
        ontology = SkillOntologyGraph(role_name=interview.requested_role)
        if not ontology.adapt_to_candidate(jd.extracted_text or "", resume.extracted_text or ""):
            raise ValueError("ROLE_PROFILE_FAILED")
        belief = BeliefStateUpdater(ontology.get_all_skills())
        interview = await db.get(InterviewSession, interview.id, with_for_update=True)
        job = await db.get(BackgroundJob, job.id, with_for_update=True)
        interview.role_profile = ontology.role_profile.to_dict()
        interview.inferred_role = ontology.inferred_role
        interview.inferred_experience = ontology.inferred_experience
        interview.status = "ready"
        interview.provenance = {
            "schema_version": 1,
            "runtime_version": interview.runtime_version,
            "policy_mode": interview.policy_mode,
            "grounding_profile_hash": ontology.role_profile.profile_hash,
            "claim": "placeholder evidence and fixed three-action cycle",
        }
        checkpoint = SessionCheckpoint(
            session_id=interview.id,
            owner_id=interview.owner_id,
            revision=0,
            schema_version=1,
            state=checkpoint_state(interview, ontology, belief, None, None, 0),
            runtime_version=interview.runtime_version,
            provenance=interview.provenance,
        )
        db.add(checkpoint)
        job.status = "succeeded"
        job.error_code = None
    except Exception as exc:
        await db.rollback()
        interview = await db.get(InterviewSession, job.session_id, with_for_update=True)
        job = await db.get(BackgroundJob, job.id, with_for_update=True)
        code = str(exc) if isinstance(exc, ValueError) else "SESSION_PREPARATION_FAILED"
        interview.status = "failed"
        interview.last_error_code = code[:80]
        interview.ended_at = utcnow()
        job.status = "failed"
        job.error_code = code[:80]
    await db.commit()


def checkpoint_state(
    interview: InterviewSession,
    ontology: SkillOntologyGraph,
    belief: BeliefStateUpdater,
    last_answer_id: uuid.UUID | None,
    current_question_id: uuid.UUID | None,
    last_turn: int,
) -> dict:
    target = getattr(ontology, "_restored_current_target", None)
    return {
        "schema_version": 1,
        "last_applied_answer_id": str(last_answer_id) if last_answer_id else None,
        "last_applied_turn_index": last_turn,
        "current_question_id": str(current_question_id) if current_question_id else None,
        "current_target_skill_id": target.skill_id if target else None,
        "ontology": interview.role_profile,
        "belief": belief.to_checkpoint(),
        "policy": {
            "mode": interview.policy_mode,
            "serializer_version": 1,
            "turn_counter": interview.accepted_answer_count,
        },
        "provenance": interview.provenance,
    }


async def restore_runtime(db: AsyncSession, interview: InterviewSession) -> tuple[dict, SkillOntologyGraph, BeliefStateUpdater, list[dict]]:
    checkpoint = (
        await db.execute(
            select(SessionCheckpoint)
            .where(SessionCheckpoint.session_id == interview.id)
            .order_by(SessionCheckpoint.revision.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if not checkpoint or checkpoint.schema_version != 1 or not interview.role_profile:
        raise HTTPException(409, "Interview state is unavailable or incompatible")
    state = checkpoint.state
    ontology = SkillOntologyGraph(role_name=interview.requested_role)
    ontology.restore_profile(interview.role_profile, interview.inferred_experience or "Mid-Level")
    target_id = state.get("current_target_skill_id")
    ontology._restored_current_target = ontology.role_profile.skill(target_id) if target_id else None
    belief = BeliefStateUpdater.from_checkpoint(state["belief"])
    rows = (
        await db.execute(
            select(Question, Answer)
            .outerjoin(Answer, (Answer.question_id == Question.id) & (Answer.status == "accepted"))
            .where(Question.session_id == interview.id)
            .order_by(Question.turn_index)
        )
    ).all()
    history = [{"q": q.text, "a": a.answer_text} for q, a in rows if a is not None]
    return state, ontology, belief, history


async def create_interview(
    db: AsyncSession,
    owner: User,
    resume_id: uuid.UUID,
    jd_id: uuid.UUID,
    role: str,
    client_request_id: uuid.UUID,
) -> tuple[InterviewSession, bool]:
    fingerprint = request_hash(str(resume_id), str(jd_id), role)
    existing = (
        await db.execute(
            select(InterviewSession).where(
                InterviewSession.owner_id == owner.id,
                InterviewSession.client_request_id == client_request_id,
            )
        )
    ).scalar_one_or_none()
    if existing:
        if existing.request_hash != fingerprint:
            raise HTTPException(409, "Idempotency key was already used for another interview")
        return existing, False
    if not owner.consent_version or not owner.consented_at:
        raise HTTPException(409, "Current privacy consent is required")
    docs = (
        await db.execute(
            select(Document)
            .where(Document.owner_id == owner.id, Document.id.in_([resume_id, jd_id]))
            .with_for_update()
        )
    ).scalars().all()
    by_id = {doc.id: doc for doc in docs}
    resume, jd = by_id.get(resume_id), by_id.get(jd_id)
    if not resume or not jd:
        raise HTTPException(404, "Document not found")
    if resume.kind != "resume" or jd.kind != "job_description":
        raise HTTPException(422, "Interview requires one résumé and one job description")
    if resume.status != "ready" or jd.status != "ready":
        raise HTTPException(409, "Both documents must finish processing")
    interview = InterviewSession(
        owner_id=owner.id,
        resume_document_id=resume.id,
        jd_document_id=jd.id,
        requested_role=role,
        client_request_id=client_request_id,
        request_hash=fingerprint,
        consent_version=owner.consent_version,
        consented_at=owner.consented_at,
        provenance={"schema_version": 1, "claim": "preparation pending"},
    )
    db.add(interview)
    await db.flush()
    resume.unreferenced_since = None
    jd.unreferenced_since = None
    db.add(
        BackgroundJob(
            owner_id=owner.id,
            session_id=interview.id,
            kind="prepare_session",
            deduplication_key=f"prepare-session:{interview.id}",
            payload={"schema_version": 1, "session_id": str(interview.id)},
        )
    )
    await db.commit()
    return interview, True


async def current_question(db: AsyncSession, interview_id: uuid.UUID) -> Question | None:
    return (
        await db.execute(
            select(Question)
            .outerjoin(Answer, (Answer.question_id == Question.id) & (Answer.status == "accepted"))
            .where(Question.session_id == interview_id, Answer.id.is_(None))
            .order_by(Question.turn_index.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def generate_and_commit_question(db: AsyncSession, interview_id: uuid.UUID, llm, send_event=None) -> Question:
    interview = await db.get(InterviewSession, interview_id, with_for_update=True)
    if not interview:
        raise HTTPException(404, "Interview not found")
    outstanding = await current_question(db, interview.id)
    if outstanding:
        return outstanding
    if interview.status in TERMINAL_STATUSES or interview.deletion_requested_at:
        raise HTTPException(409, "Interview is read-only")
    if interview.status not in {"ready", "active", "paused"}:
        raise HTTPException(409, "Interview is not ready")
    interview.status = "active"
    interview.started_at = interview.started_at or utcnow()
    interview.last_activity_at = utcnow()
    expected_revision = interview.revision
    turn_index = interview.accepted_answer_count + 1
    action = "switch_topic" if turn_index == 1 else ACTION_CYCLE[(turn_index - 2) % len(ACTION_CYCLE)]
    deduplication_key = f"generate:{interview.id}:{turn_index}"
    job = (
        await db.execute(
            select(BackgroundJob).where(BackgroundJob.deduplication_key == deduplication_key)
        )
    ).scalar_one_or_none()
    if job and job.status not in {"failed", "cancelled"}:
        raise HTTPException(409, "Question generation is already committed or in progress")
    if job:
        job.status = "running"
        job.error_code = None
        job.attempt_count += 1
        job.payload = {"schema_version": 1, "expected_revision": expected_revision, "turn_index": turn_index}
    else:
        job = BackgroundJob(
            owner_id=interview.owner_id,
            session_id=interview.id,
            kind="generate_question",
            deduplication_key=deduplication_key,
            payload={"schema_version": 1, "expected_revision": expected_revision, "turn_index": turn_index},
            status="running",
            attempt_count=1,
        )
        db.add(job)
    await db.commit()
    state, ontology, belief, history = await restore_runtime(db, interview)
    attempt_base = (
        await db.execute(
            select(func.coalesce(func.max(GenerationAttempt.attempt_index), 0)).where(
                GenerationAttempt.job_id == job.id
            )
        )
    ).scalar_one()
    current = ontology._restored_current_target
    candidates = []
    if current and action == "increase_difficulty":
        candidates = ontology.get_advanced(current.canonical_name)
    elif current and action in {"decrease_difficulty", "probe_foundation"}:
        candidates = ontology.get_prerequisites(current.canonical_name)
    elif current and action != "switch_topic":
        candidates = [current.canonical_name]
    if not candidates:
        candidates = [
            skill.canonical_name for skill in ontology.role_profile.skills
            if current is None or skill.skill_id != current.skill_id
        ] or [current.canonical_name]
    target = min(
        (ontology.get_skill_metadata(name) for name in candidates),
        key=lambda skill: (skill.selection_priority, belief.get_evidence_count(skill.canonical_name), skill.skill_id),
    )
    ontology._restored_current_target = target
    context = grounding_packet(ontology.role_profile, target)
    rejected, rejected_outputs = [], []
    for retry_index in range(1, 4):
        attempt_index = attempt_base + retry_index
        attempt = GenerationAttempt(
            session_id=interview.id,
            owner_id=interview.owner_id,
            job_id=job.id,
            turn_index=turn_index,
            attempt_index=attempt_index,
            status="running",
            model_identifier=getattr(llm, "model", "ollama-configured"),
            prompt_version="aria-grounded-v2",
            generation_config={"temperature": 0.3 + 0.15 * (retry_index - 1)},
        )
        db.add(attempt)
        await db.commit()
        correction = (
            build_question_retry_correction(rejected[-1], rejected_outputs[-1], retry_index)
            if rejected else None
        )
        args = dict(
            action=action,
            belief_state={key: value.tolist() for key, value in belief.beliefs.items()},
            resume=(await db.get(Document, interview.resume_document_id)).extracted_text,
            history=history,
            role=interview.inferred_role,
            experience=interview.inferred_experience,
            target_skill=target.canonical_name,
            grounding_context=context,
            correction=correction,
            temperature=0.3 + 0.15 * (retry_index - 1),
        )
        try:
            if send_event and hasattr(llm, "generate_question_stream"):
                await send_event({"type": "aria_stream_start", "job_id": str(job.id), "attempt_id": str(attempt.id), "action": action})
                chunks = []
                sequence = 0
                async for chunk in llm.generate_question_stream(**args):
                    if chunk:
                        chunks.append(chunk)
                        sequence += 1
                        await send_event({"type": "aria_chunk", "job_id": str(job.id), "attempt_id": str(attempt.id), "sequence": sequence, "text": chunk})
                raw = "".join(chunks)
            else:
                raw = await llm.generate_question(**args)
        except Exception as exc:
            await db.rollback()
            failed_attempt = await db.get(GenerationAttempt, attempt.id, with_for_update=True)
            failed_job = await db.get(BackgroundJob, job.id, with_for_update=True)
            failed_attempt.status = "failed"
            failed_attempt.error_code = "QUESTION_GENERATION_FAILED"
            failed_attempt.finished_at = utcnow()
            failed_job.status = "failed"
            failed_job.error_code = "QUESTION_GENERATION_FAILED"
            await db.commit()
            raise HTTPException(503, "Question generation failed") from exc
        normalized = normalize_generated_question(raw)
        result = validate_grounded_question(normalized, context, history)
        attempt = await db.get(GenerationAttempt, attempt.id, with_for_update=True)
        attempt.finished_at = utcnow()
        attempt.validation_reasons = result["reasons"]
        if not result["valid"]:
            attempt.status = "rejected"
            await db.commit()
            rejected.append(result["reasons"])
            rejected_outputs.append(normalized)
            if send_event:
                await send_event({"type": "aria_stream_reset", "job_id": str(job.id), "attempt_id": str(attempt.id)})
            continue
        interview = await db.get(InterviewSession, interview.id, with_for_update=True)
        if interview.status in TERMINAL_STATUSES or interview.revision != expected_revision:
            attempt.status = "interrupted"
            job = await db.get(BackgroundJob, job.id, with_for_update=True)
            job.status = "cancelled"
            await db.commit()
            raise HTTPException(409, "Interview changed while the question was generated")
        attempt.status = "accepted"
        question = Question(
            session_id=interview.id,
            owner_id=interview.owner_id,
            turn_index=turn_index,
            text=normalized,
            action=action,
            target_skill_id=target.skill_id,
            generation_attempt_id=attempt.id,
            grounding_context=_jsonable(context),
            model_identifier=attempt.model_identifier,
            prompt_version=attempt.prompt_version,
            generation_config=attempt.generation_config,
        )
        db.add(question)
        await db.flush()
        interview.revision += 1
        interview.last_activity_at = utcnow()
        checkpoint = SessionCheckpoint(
            session_id=interview.id,
            owner_id=interview.owner_id,
            revision=interview.revision,
            schema_version=1,
            state=checkpoint_state(
                interview,
                ontology,
                belief,
                uuid.UUID(state["last_applied_answer_id"]) if state.get("last_applied_answer_id") else None,
                question.id,
                int(state.get("last_applied_turn_index", 0)),
            ),
            runtime_version=interview.runtime_version,
            provenance=interview.provenance,
        )
        db.add(checkpoint)
        job = await db.get(BackgroundJob, job.id, with_for_update=True)
        job.status = "succeeded"
        await db.commit()
        return question
    job = await db.get(BackgroundJob, job.id, with_for_update=True)
    job.status = "failed"
    job.error_code = "QUESTION_GROUNDING_FAILED"
    interview = await db.get(InterviewSession, interview.id, with_for_update=True)
    interview.last_error_code = job.error_code
    await db.commit()
    raise HTTPException(503, "ARIA could not produce a grounded question")


async def accept_text_answer(
    db: AsyncSession,
    interview_id: uuid.UUID,
    owner_id: uuid.UUID,
    question_id: uuid.UUID,
    submission_id: uuid.UUID,
    answer_text: str,
    input_mode: str = "text",
    transcriber_model: str | None = None,
) -> tuple[Answer, bool]:
    canonical = answer_text.strip()
    if not canonical:
        raise HTTPException(422, "Answer cannot be empty")
    fingerprint = request_hash(str(question_id), input_mode, canonical)
    existing = (
        await db.execute(
            select(Answer).where(
                Answer.session_id == interview_id,
                Answer.client_submission_id == submission_id,
            )
        )
    ).scalar_one_or_none()
    if existing:
        if existing.request_hash != fingerprint:
            raise HTTPException(409, "Submission ID was already used for another answer")
        return existing, False
    interview = (
        await db.execute(
            select(InterviewSession)
            .where(InterviewSession.id == interview_id, InterviewSession.owner_id == owner_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if not interview:
        raise HTTPException(404, "Interview not found")
    if interview.status in TERMINAL_STATUSES or interview.deletion_requested_at:
        raise HTTPException(409, "Interview is read-only")
    outstanding = await current_question(db, interview.id)
    if not outstanding or outstanding.id != question_id:
        raise HTTPException(409, "Question is no longer accepting an answer")
    _, ontology, belief, _ = await restore_runtime(db, interview)
    target = ontology.role_profile.skill(outstanding.target_skill_id)
    if target:
        belief.update_belief(
            target.canonical_name,
            semantic_score=0.9,
            cognitive_load="low",
            behavior_score=0.9,
        )
        ontology._restored_current_target = target
    answer = Answer(
        session_id=interview.id,
        owner_id=owner_id,
        question_id=question_id,
        client_submission_id=submission_id,
        request_hash=fingerprint,
        input_mode=input_mode,
        status="accepted",
        answer_text=canonical,
        transcriber_model=transcriber_model,
        transcriber_version="faster-whisper-v1" if input_mode == "audio" else None,
        accepted_at=utcnow(),
    )
    db.add(answer)
    await db.flush()
    interview.accepted_answer_count += 1
    interview.revision += 1
    interview.last_activity_at = utcnow()
    db.add(
        SessionCheckpoint(
            session_id=interview.id,
            owner_id=owner_id,
            revision=interview.revision,
            schema_version=1,
            state=checkpoint_state(
                interview, ontology, belief, answer.id, outstanding.id, outstanding.turn_index
            ),
            runtime_version=interview.runtime_version,
            provenance=interview.provenance,
        )
    )
    await db.commit()
    return answer, True


async def acquire_connection_lease(
    db: AsyncSession,
    interview: InterviewSession,
    auth_session_id: uuid.UUID,
    connection_id: uuid.UUID,
    settings: AppSettings,
    takeover: bool = False,
) -> InterviewConnectionLease:
    lease = await db.get(InterviewConnectionLease, interview.id, with_for_update=True)
    now = utcnow()
    if lease and lease.expires_at > now and lease.connection_id != connection_id and not takeover:
        raise HTTPException(409, "Another tab controls this interview")
    epoch = (lease.epoch + 1) if lease else 1
    if not lease:
        lease = InterviewConnectionLease(
            session_id=interview.id,
            owner_id=interview.owner_id,
            auth_session_id=auth_session_id,
            connection_id=connection_id,
            epoch=epoch,
            expires_at=now + timedelta(seconds=settings.connection_lease_seconds),
        )
        db.add(lease)
    else:
        lease.auth_session_id = auth_session_id
        lease.connection_id = connection_id
        lease.epoch = epoch
        lease.expires_at = now + timedelta(seconds=settings.connection_lease_seconds)
    await db.commit()
    return lease


async def serialize_interview(db: AsyncSession, interview: InterviewSession) -> dict:
    outstanding = await current_question(db, interview.id)
    return {
        "id": str(interview.id),
        "status": interview.status,
        "requested_role": interview.requested_role,
        "inferred_role": interview.inferred_role,
        "inferred_experience": interview.inferred_experience,
        "revision": interview.revision,
        "accepted_answer_count": interview.accepted_answer_count,
        "last_error_code": interview.last_error_code,
        "created_at": interview.created_at.isoformat(),
        "current_question": serialize_question(outstanding) if outstanding else None,
        "provenance": interview.provenance,
    }


def serialize_question(question: Question) -> dict:
    return {
        "id": str(question.id),
        "turn_index": question.turn_index,
        "text": question.text,
        "action": question.action,
        "target_skill_id": question.target_skill_id,
        "created_at": question.created_at.isoformat(),
    }
