from __future__ import annotations

import asyncio
import base64
import binascii
from datetime import timedelta
import json
import logging
import os
import secrets
import tempfile
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import ACCESS_COOKIE, OAUTH_BINDING_COOKIE, REFRESH_COOKIE, Principal, clear_auth_cookies, consume_oauth_transaction, create_oauth_transaction, decode_access_token, digest, establish_identity, exchange_google_code, get_principal, google_authorization_url, issue_access_token, require_csrf, rotate_refresh_token, set_auth_cookies, token_urlsafe
from backend.db import SessionFactory, get_db
from backend.documents import reserve_document
from backend.interviews import TERMINAL_STATUSES, accept_text_answer, acquire_connection_lease, create_interview, generate_and_commit_question, serialize_interview, serialize_question
from backend.models import Answer, AuthSession, BackgroundJob, Document, InterviewConnectionLease, InterviewSession, Question, RefreshToken, User, utcnow
from backend.settings import get_settings
from backend.storage import LocalPrivateStorage
from modules.module_01_stt.transcriber import transcribe
from modules.module_08_llm.generator import LLMQuestionGenerator


logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title="ARIA Orchestrator API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins), allow_credentials=True, allow_methods=["GET", "POST", "DELETE", "OPTIONS"], allow_headers=["Content-Type", "Idempotency-Key", "X-CSRF-Token"])

try:
    llm_gen = LLMQuestionGenerator()
except Exception as exc:
    logger.error("Failed to load live question generator: %s", exc)
    llm_gen = None


class ConsentInput(BaseModel):
    version: str


class InterviewInput(BaseModel):
    resume_document_id: uuid.UUID
    jd_document_id: uuid.UUID
    requested_role: str = Field(default="backend_developer", min_length=1, max_length=120)


class RevisionInput(BaseModel):
    expected_revision: int = Field(ge=0)


class TextAnswerInput(BaseModel):
    question_id: uuid.UUID
    submission_id: uuid.UUID
    text: str = Field(min_length=1, max_length=settings.max_answer_characters)


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    code = str(exc.detail).upper().replace(" ", "_")[:80]
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": code, "message": str(exc.detail), "request_id": request.headers.get("x-request-id", str(uuid.uuid4())), "retryable": exc.status_code in {429, 503}}, "detail": str(exc.detail)})


@app.get("/api/health")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(select(1))
        database = "ready"
    except Exception:
        database = "unavailable"
    return {"status": "ok" if database == "ready" else "degraded", "database": database}


@app.get("/api/auth/google/start")
async def auth_google_start(return_path: str = Query("/"), db: AsyncSession = Depends(get_db)):
    state, nonce, binding, challenge = await create_oauth_transaction(db, return_path, settings)
    response = RedirectResponse(google_authorization_url(state, nonce, challenge, settings), 302)
    response.set_cookie(OAUTH_BINDING_COOKIE, f"{binding}.{nonce}", max_age=settings.oauth_ttl_seconds, httponly=True, secure=settings.cookie_secure, samesite="lax", path="/api/auth/google/callback")
    return response


@app.get("/api/auth/google/callback")
async def auth_google_callback(request: Request, code: str, state: str, db: AsyncSession = Depends(get_db)):
    try:
        binding, nonce = request.cookies.get(OAUTH_BINDING_COOKIE, "").split(".", 1)
    except ValueError as exc:
        raise HTTPException(401, "Google login browser binding is missing") from exc
    tx, verifier, return_path = await consume_oauth_transaction(db, state, binding, settings)
    if not secrets.compare_digest(tx.nonce_digest, digest(nonce)):
        raise HTTPException(401, "Google login nonce is invalid")
    claims = await exchange_google_code(code, verifier, nonce, settings)
    user, auth_session, refresh = await establish_identity(db, claims, settings)
    response = RedirectResponse(f"{settings.public_origin}{return_path}", 302)
    set_auth_cookies(response, issue_access_token(user.id, auth_session.id, settings), refresh, settings)
    response.delete_cookie(OAUTH_BINDING_COOKIE, path="/api/auth/google/callback", secure=settings.cookie_secure, samesite="lax")
    return response


@app.get("/api/auth/me")
async def auth_me(principal: Principal = Depends(get_principal), db: AsyncSession = Depends(get_db)):
    user = await db.get(User, principal.user_id)
    return {"id": str(user.id), "email": user.email, "display_name": user.display_name, "consent_version": user.consent_version, "required_consent_version": settings.consent_version}


@app.get("/api/auth/csrf")
async def auth_csrf(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    auth_session = None
    access = request.cookies.get(ACCESS_COOKIE)
    if access:
        try:
            user_id, auth_session_id = decode_access_token(access, settings)
            auth_session = await db.get(AuthSession, auth_session_id, with_for_update=True)
            if not auth_session or auth_session.user_id != user_id:
                auth_session = None
        except HTTPException:
            auth_session = None
    if auth_session is None:
        refresh = request.cookies.get(REFRESH_COOKIE)
        token_row = (await db.execute(select(RefreshToken).where(RefreshToken.token_digest == digest(refresh or "")))).scalar_one_or_none()
        if token_row:
            auth_session = await db.get(AuthSession, token_row.auth_session_id, with_for_update=True)
    user = await db.get(User, auth_session.user_id) if auth_session else None
    if not auth_session or auth_session.revoked_at or auth_session.expires_at <= utcnow() or not user or user.status != "active":
        raise HTTPException(401, "Authentication required")
    raw = token_urlsafe()
    auth_session.csrf_digest = digest(raw)
    await db.commit()
    response.headers["Cache-Control"] = "no-store"
    return {"csrf_token": raw}


@app.post("/api/auth/refresh", status_code=204)
async def auth_refresh(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    if request.headers.get("origin") not in settings.allowed_origins:
        raise HTTPException(403, "Request origin is not allowed")
    raw = request.cookies.get(REFRESH_COOKIE)
    supplied = request.headers.get("x-csrf-token", "")
    if not raw:
        raise HTTPException(401, "Refresh token is required")
    row = (await db.execute(select(RefreshToken).where(RefreshToken.token_digest == digest(raw)))).scalar_one_or_none()
    auth_session = await db.get(AuthSession, row.auth_session_id) if row else None
    if not auth_session or not supplied or not auth_session.csrf_digest or not secrets.compare_digest(auth_session.csrf_digest, digest(supplied)):
        raise HTTPException(403, "CSRF token is invalid")
    user, auth_session, replacement = await rotate_refresh_token(db, raw, settings)
    set_auth_cookies(response, issue_access_token(user.id, auth_session.id, settings), replacement, settings)


@app.post("/api/auth/logout", status_code=204)
async def auth_logout(response: Response, principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    row = await db.get(AuthSession, principal.auth_session_id, with_for_update=True)
    row.revoked_at, row.revocation_reason = utcnow(), "logout"
    await db.commit()
    clear_auth_cookies(response, settings)


@app.post("/api/auth/logout-all", status_code=204)
async def auth_logout_all(response: Response, principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(AuthSession).where(AuthSession.user_id == principal.user_id, AuthSession.revoked_at.is_(None)).with_for_update())).scalars()
    for row in rows:
        row.revoked_at, row.revocation_reason = utcnow(), "logout_all"
    await db.commit()
    clear_auth_cookies(response, settings)


@app.post("/api/me/consent")
async def record_consent(payload: ConsentInput, principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    if payload.version != settings.consent_version:
        raise HTTPException(409, "Consent version is not current")
    user = await db.get(User, principal.user_id, with_for_update=True)
    user.consent_version, user.consented_at = payload.version, utcnow()
    await db.commit()
    return {"consent_version": user.consent_version, "consented_at": user.consented_at}


def parse_idempotency(value: str | None) -> uuid.UUID:
    try:
        return uuid.UUID(value or "")
    except ValueError as exc:
        raise HTTPException(422, "Idempotency-Key must be a UUID") from exc


def document_json(document: Document) -> dict:
    return {"id": str(document.id), "kind": document.kind, "original_filename": document.original_filename, "size_bytes": document.size_bytes, "status": document.status, "page_count": document.page_count, "error_code": document.error_code, "created_at": document.created_at.isoformat()}


async def owned_document(db: AsyncSession, owner_id: uuid.UUID, document_id: uuid.UUID) -> Document:
    row = (await db.execute(select(Document).where(Document.id == document_id, Document.owner_id == owner_id, Document.deletion_requested_at.is_(None)))).scalar_one_or_none()
    if not row:
        raise HTTPException(404, "Document not found")
    return row


@app.post("/api/documents", status_code=202)
async def upload_document(kind: str = Form(...), file: UploadFile = File(...), idempotency_key: str | None = Header(None, alias="Idempotency-Key"), principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    user = await db.get(User, principal.user_id)
    if user.consent_version != settings.consent_version:
        raise HTTPException(409, "Current privacy consent is required")
    document, created = await reserve_document(db, principal.user_id, kind, file, parse_idempotency(idempotency_key), settings)
    return {**document_json(document), "created": created}


@app.get("/api/documents")
async def list_documents(kind: str | None = None, limit: int = Query(20, ge=1, le=100), principal: Principal = Depends(get_principal), db: AsyncSession = Depends(get_db)):
    query = select(Document).where(Document.owner_id == principal.user_id, Document.deletion_requested_at.is_(None))
    if kind:
        query = query.where(Document.kind == kind)
    rows = (await db.execute(query.order_by(Document.created_at.desc(), Document.id.desc()).limit(limit))).scalars()
    return {"items": [document_json(row) for row in rows]}


@app.get("/api/documents/{document_id}")
async def get_document(document_id: uuid.UUID, principal: Principal = Depends(get_principal), db: AsyncSession = Depends(get_db)):
    return document_json(await owned_document(db, principal.user_id, document_id))


@app.get("/api/documents/{document_id}/download")
async def download_document(document_id: uuid.UUID, principal: Principal = Depends(get_principal), db: AsyncSession = Depends(get_db)):
    document = await owned_document(db, principal.user_id, document_id)
    path = LocalPrivateStorage(settings).path(document.object_key)
    if not path.exists():
        raise HTTPException(503, "Private document storage is unavailable")
    return FileResponse(path, media_type="application/pdf", filename=document.original_filename, headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"})


@app.delete("/api/documents/{document_id}", status_code=202)
async def delete_document(document_id: uuid.UUID, principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    document = await owned_document(db, principal.user_id, document_id)
    count = (await db.execute(select(func.count()).select_from(InterviewSession).where(InterviewSession.owner_id == principal.user_id, InterviewSession.deletion_requested_at.is_(None), (InterviewSession.resume_document_id == document.id) | (InterviewSession.jd_document_id == document.id)))).scalar_one()
    if count:
        raise HTTPException(409, "DOCUMENT_IN_USE")
    document.status, document.deletion_requested_at = "deleting", utcnow()
    db.add(BackgroundJob(owner_id=principal.user_id, document_id=document.id, kind="purge_document", deduplication_key=f"purge-document:{document.id}", payload={"schema_version": 1, "object_key": document.object_key}))
    await db.commit()
    return {"status": "deleting"}


async def owned_interview(db: AsyncSession, owner_id: uuid.UUID, interview_id: uuid.UUID) -> InterviewSession:
    row = (await db.execute(select(InterviewSession).where(InterviewSession.id == interview_id, InterviewSession.owner_id == owner_id, InterviewSession.deletion_requested_at.is_(None)))).scalar_one_or_none()
    if not row:
        raise HTTPException(404, "Interview not found")
    return row


@app.post("/api/interviews", status_code=202)
async def create_interview_route(payload: InterviewInput, idempotency_key: str | None = Header(None, alias="Idempotency-Key"), principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    interview, created = await create_interview(db, await db.get(User, principal.user_id), payload.resume_document_id, payload.jd_document_id, payload.requested_role, parse_idempotency(idempotency_key))
    result = await serialize_interview(db, interview)
    return {**result, "created": created}


@app.get("/api/interviews")
async def list_interviews(limit: int = Query(20, ge=1, le=100), principal: Principal = Depends(get_principal), db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(InterviewSession).where(InterviewSession.owner_id == principal.user_id, InterviewSession.deletion_requested_at.is_(None)).order_by(InterviewSession.created_at.desc(), InterviewSession.id.desc()).limit(limit))).scalars().all()
    return {"items": [{"id": str(row.id), "status": row.status, "requested_role": row.requested_role, "inferred_role": row.inferred_role, "accepted_answer_count": row.accepted_answer_count, "created_at": row.created_at.isoformat()} for row in rows]}


@app.get("/api/interviews/{interview_id}")
async def get_interview(interview_id: uuid.UUID, principal: Principal = Depends(get_principal), db: AsyncSession = Depends(get_db)):
    return await serialize_interview(db, await owned_interview(db, principal.user_id, interview_id))


@app.get("/api/interviews/{interview_id}/turns")
async def get_turns(interview_id: uuid.UUID, principal: Principal = Depends(get_principal), db: AsyncSession = Depends(get_db)):
    await owned_interview(db, principal.user_id, interview_id)
    rows = (await db.execute(select(Question, Answer).outerjoin(Answer, (Answer.question_id == Question.id) & (Answer.status == "accepted")).where(Question.session_id == interview_id, Question.owner_id == principal.user_id).order_by(Question.turn_index))).all()
    return {"items": [{"question": serialize_question(question), "answer": ({"id": str(answer.id), "text": answer.answer_text, "input_mode": answer.input_mode, "accepted_at": answer.accepted_at.isoformat()} if answer else None)} for question, answer in rows]}


@app.post("/api/interviews/{interview_id}/answers")
async def post_answer(interview_id: uuid.UUID, payload: TextAnswerInput, principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    answer, created = await accept_text_answer(db, interview_id, principal.user_id, payload.question_id, payload.submission_id, payload.text)
    interview = await db.get(InterviewSession, interview_id)
    return {"id": str(answer.id), "status": answer.status, "created": created, "revision": interview.revision}


@app.post("/api/interviews/{interview_id}/resume")
async def resume_interview(interview_id: uuid.UUID, payload: RevisionInput, principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    interview = await owned_interview(db, principal.user_id, interview_id)
    await db.refresh(interview, with_for_update=True)
    if interview.revision != payload.expected_revision:
        raise HTTPException(409, "Interview revision is stale")
    if interview.status not in {"ready", "active", "paused"}:
        raise HTTPException(409, "Interview cannot be resumed")
    if interview.status != "active":
        interview.status = "active"
        interview.revision += 1
        interview.last_activity_at = utcnow()
        await db.commit()
    return await serialize_interview(db, interview)


@app.post("/api/interviews/{interview_id}/retry-generation")
async def retry_generation(interview_id: uuid.UUID, payload: RevisionInput, principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    interview = await owned_interview(db, principal.user_id, interview_id)
    if interview.revision != payload.expected_revision:
        raise HTTPException(409, "Interview revision is stale")
    if not llm_gen:
        raise HTTPException(503, "Question generation is unavailable")
    question = await generate_and_commit_question(db, interview.id, llm_gen)
    interview = await db.get(InterviewSession, interview.id)
    return {"question": serialize_question(question), "revision": interview.revision}


async def terminal_transition(db: AsyncSession, owner_id: uuid.UUID, interview_id: uuid.UUID, expected_revision: int, target: str) -> InterviewSession:
    interview = await owned_interview(db, owner_id, interview_id)
    await db.refresh(interview, with_for_update=True)
    if interview.revision != expected_revision:
        raise HTTPException(409, "Interview revision is stale")
    if interview.status in TERMINAL_STATUSES:
        if interview.status == target:
            return interview
        raise HTTPException(409, "Interview is already terminal")
    interview.status, interview.ended_at = target, utcnow()
    interview.revision += 1
    lease = await db.get(InterviewConnectionLease, interview.id)
    if lease:
        await db.delete(lease)
    jobs = (await db.execute(select(BackgroundJob).where(BackgroundJob.session_id == interview.id, BackgroundJob.status.in_(["queued", "running"])))).scalars()
    for job in jobs:
        job.status = "cancelled"
    await db.commit()
    return interview


@app.post("/api/interviews/{interview_id}/complete")
async def complete_interview(interview_id: uuid.UUID, payload: RevisionInput, principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    return await serialize_interview(db, await terminal_transition(db, principal.user_id, interview_id, payload.expected_revision, "completed"))


@app.post("/api/interviews/{interview_id}/cancel")
async def cancel_interview(interview_id: uuid.UUID, payload: RevisionInput, principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    return await serialize_interview(db, await terminal_transition(db, principal.user_id, interview_id, payload.expected_revision, "cancelled"))


@app.delete("/api/interviews/{interview_id}", status_code=202)
async def delete_interview(interview_id: uuid.UUID, principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    interview = await owned_interview(db, principal.user_id, interview_id)
    interview.deletion_requested_at = utcnow()
    if interview.status not in TERMINAL_STATUSES:
        interview.status, interview.ended_at = "cancelled", utcnow()
    db.add(BackgroundJob(owner_id=principal.user_id, session_id=interview.id, kind="purge_session", deduplication_key=f"purge-session:{interview.id}", payload={"schema_version": 1}))
    await db.commit()
    return {"status": "deleting"}


@app.delete("/api/me", status_code=202)
async def delete_account(response: Response, principal: Principal = Depends(require_csrf), db: AsyncSession = Depends(get_db)):
    auth_session = await db.get(AuthSession, principal.auth_session_id, with_for_update=True)
    if not auth_session or auth_session.created_at < utcnow() - timedelta(minutes=10):
        raise HTTPException(403, "Sign in again before deleting the account")
    user = await db.get(User, principal.user_id, with_for_update=True)
    user.status = "deleting"
    user.deletion_requested_at = utcnow()
    sessions = (await db.execute(select(AuthSession).where(AuthSession.user_id == user.id).with_for_update())).scalars()
    for session in sessions:
        session.revoked_at = utcnow()
        session.revocation_reason = "account_deletion"
    pending_jobs = (await db.execute(select(BackgroundJob).where(BackgroundJob.owner_id == user.id, BackgroundJob.status.in_(["queued", "running"])))).scalars()
    for pending in pending_jobs:
        pending.status = "cancelled"
    db.add(BackgroundJob(owner_id=user.id, kind="purge_account", deduplication_key=f"purge-account:{user.id}", payload={"schema_version": 1}))
    await db.commit()
    clear_auth_cookies(response, settings)
    return {"status": "deleting"}


@app.post("/api/start-session", status_code=410)
async def legacy_start_session():
    raise HTTPException(410, "Use authenticated document and interview endpoints")


async def transcribe_candidate_audio(audio_base64: str) -> str:
    if not isinstance(audio_base64, str):
        raise ValueError("The recording was invalid. Please record again.")
    encoded = audio_base64.split("base64,", 1)[-1]
    if len(encoded) > settings.max_audio_bytes * 4 // 3 + 4096:
        raise ValueError("The recording was too large. Please record a shorter answer.")
    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The recording could not be decoded. Please record again.") from exc
    if not audio_bytes:
        raise ValueError("The recording was empty. Please record again.")
    if len(audio_bytes) > settings.max_audio_bytes:
        raise ValueError("The recording was too large. Please record a shorter answer.")
    from faster_whisper.audio import decode_audio
    with tempfile.TemporaryDirectory(prefix="aria_audio_") as temp_dir:
        path = os.path.join(temp_dir, "answer.webm")
        with open(path, "wb") as handle:
            handle.write(audio_bytes)
        audio = await asyncio.to_thread(decode_audio, path, 16000)
        if len(audio) > 16000 * 300:
            raise ValueError("The recording was longer than five minutes.")
        result = await transcribe(audio)
    return result.get("transcript", "").strip()


async def websocket_principal(websocket: WebSocket, db: AsyncSession) -> Principal:
    if websocket.headers.get("origin") not in settings.allowed_origins:
        raise HTTPException(403, "WebSocket origin is not allowed")
    token = websocket.cookies.get(ACCESS_COOKIE)
    if not token:
        raise HTTPException(401, "Authentication required")
    user_id, auth_session_id = decode_access_token(token, settings)
    row = (await db.execute(select(User, AuthSession).join(AuthSession, AuthSession.user_id == User.id).where(User.id == user_id, User.status == "active", AuthSession.id == auth_session_id, AuthSession.revoked_at.is_(None), AuthSession.expires_at > utcnow()))).one_or_none()
    if not row:
        raise HTTPException(401, "Authentication required")
    user, _ = row
    return Principal(user.id, auth_session_id, user.email, user.display_name)


@app.websocket("/ws/interview/{interview_id}")
async def websocket_endpoint(websocket: WebSocket, interview_id: uuid.UUID):
    connection_id, lease_epoch = uuid.uuid4(), None
    async with SessionFactory() as db:
        try:
            principal = await websocket_principal(websocket, db)
            interview = await owned_interview(db, principal.user_id, interview_id)
        except HTTPException as exc:
            await websocket.close(code=4401 if exc.status_code == 401 else 4404)
            return
        await websocket.accept()

        async def send_event(payload: dict):
            await websocket.send_json({"protocol_version": 2, "session_id": str(interview_id), **payload})

        try:
            if interview.status in TERMINAL_STATUSES:
                writable = False
                await send_event({"type": "control_lost", "message": "This interview is complete and read-only"})
            else:
                try:
                    lease = await acquire_connection_lease(db, interview, principal.auth_session_id, connection_id, settings)
                    lease_epoch, writable = lease.epoch, True
                    await send_event({"type": "control_granted", "connection_id": str(connection_id), "epoch": lease.epoch, "expires_at": lease.expires_at.isoformat()})
                except HTTPException:
                    writable = False
                    await send_event({"type": "control_lost", "message": "Another tab controls this interview"})
            snapshot = await serialize_interview(db, interview)
            turns = await get_turns(interview_id, principal, db)
            await send_event({"type": "session_snapshot", **snapshot, "turns": turns["items"], "writable": writable})
            if writable and interview.status in {"ready", "active", "paused"} and not snapshot["current_question"]:
                if not llm_gen:
                    await send_event({"type": "operation_error", "code": "LLM_UNAVAILABLE", "message": "Question generation is unavailable", "retryable": True})
                else:
                    question = await generate_and_commit_question(db, interview.id, llm_gen, send_event)
                    interview = await db.get(InterviewSession, interview.id)
                    await send_event({"type": "aria_question", **serialize_question(question), "revision": interview.revision})
            while True:
                payload = json.loads(await websocket.receive_text())
                kind = payload.get("type")
                if kind == "acquire_control":
                    lease = await acquire_connection_lease(db, interview, principal.auth_session_id, connection_id, settings, takeover=bool(payload.get("takeover")))
                    lease_epoch, writable = lease.epoch, True
                    await send_event({"type": "control_granted", "connection_id": str(connection_id), "epoch": lease.epoch, "expires_at": lease.expires_at.isoformat()})
                    continue
                if kind == "heartbeat" and writable:
                    auth_session = await db.get(AuthSession, principal.auth_session_id)
                    if not auth_session or auth_session.revoked_at or auth_session.expires_at <= utcnow():
                        await websocket.close(code=4401)
                        return
                    lease = await db.get(InterviewConnectionLease, interview.id, with_for_update=True)
                    if not lease or lease.connection_id != connection_id or lease.epoch != lease_epoch:
                        writable = False
                        await send_event({"type": "control_lost", "message": "Interview control moved to another tab"})
                    else:
                        lease.expires_at = utcnow() + timedelta(seconds=settings.connection_lease_seconds)
                        await db.commit()
                    continue
                if not writable:
                    await send_event({"type": "operation_error", "code": "READ_ONLY_CONNECTION", "message": "Take control before submitting an answer", "retryable": True})
                    continue
                lease = await db.get(InterviewConnectionLease, interview.id)
                if not lease or lease.connection_id != connection_id or lease.epoch != lease_epoch or lease.expires_at <= utcnow():
                    writable = False
                    await send_event({"type": "control_lost", "message": "Interview control expired"})
                    continue
                if kind not in {"candidate_answer", "candidate_audio"}:
                    continue
                try:
                    question_id = uuid.UUID(payload.get("question_id", ""))
                    submission_id = uuid.UUID(payload.get("submission_id", ""))
                except ValueError:
                    await send_event({"type": "operation_error", "code": "INVALID_SUBMISSION", "message": "Question and submission IDs are required", "retryable": False})
                    continue
                if kind == "candidate_audio":
                    await send_event({"type": "answer_pending", "submission_id": str(submission_id)})
                    try:
                        candidate_text = await transcribe_candidate_audio(payload.get("audio_base64", ""))
                    except Exception as exc:
                        logger.info("Audio transcription failed for interview %s: %s", interview_id, exc)
                        await send_event({"type": "audio_error", "submission_id": str(submission_id), "message": str(exc) if isinstance(exc, ValueError) else "Audio could not be transcribed. Please try again or type your answer.", "retryable": True})
                        continue
                    if not candidate_text:
                        await send_event({"type": "audio_error", "submission_id": str(submission_id), "message": "No speech was detected. Please try again or type your answer.", "retryable": True})
                        continue
                    await send_event({"type": "transcription_result", "submission_id": str(submission_id), "text": candidate_text})
                    mode = "audio"
                else:
                    candidate_text, mode = str(payload.get("text", "")), "text"
                try:
                    answer, _ = await accept_text_answer(db, interview.id, principal.user_id, question_id, submission_id, candidate_text, input_mode=mode, transcriber_model="distil-large-v3" if mode == "audio" else None)
                    interview = await db.get(InterviewSession, interview.id)
                    await send_event({"type": "answer_accepted", "id": str(answer.id), "question_id": str(question_id), "submission_id": str(submission_id), "text": answer.answer_text, "input_mode": answer.input_mode, "revision": interview.revision})
                    if not llm_gen:
                        await send_event({"type": "operation_error", "code": "LLM_UNAVAILABLE", "message": "Your answer was saved, but question generation is unavailable", "retryable": True})
                        continue
                    question = await generate_and_commit_question(db, interview.id, llm_gen, send_event)
                    interview = await db.get(InterviewSession, interview.id)
                    await send_event({"type": "aria_question", **serialize_question(question), "revision": interview.revision})
                except HTTPException as exc:
                    await send_event({"type": "operation_error", "code": "INTERVIEW_CONFLICT", "message": str(exc.detail), "retryable": exc.status_code >= 500})
        except WebSocketDisconnect:
            logger.info("Interview %s disconnected", interview_id)
        except Exception:
            logger.exception("Interview WebSocket failed")
            try:
                await websocket.close(code=1011)
            except Exception:
                pass
        finally:
            if lease_epoch is not None:
                lease = await db.get(InterviewConnectionLease, interview_id, with_for_update=True)
                if lease and lease.connection_id == connection_id and lease.epoch == lease_epoch:
                    lease.expires_at = utcnow()
                    current = await db.get(InterviewSession, interview_id, with_for_update=True)
                    if current and current.status == "active":
                        current.status, current.revision = "paused", current.revision + 1
                    await db.commit()
