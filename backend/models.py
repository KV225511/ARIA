from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now().astimezone()


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(Text)
    display_name: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="active")
    consent_version: Mapped[str | None] = mapped_column(Text)
    consented_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_login_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deletion_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint("status IN ('active','disabled','deleting')", name="ck_users_status"),
        CheckConstraint(
            "(consent_version IS NULL) = (consented_at IS NULL)", name="ck_users_consent_pair"
        ),
    )


class OAuthIdentity(TimestampMixin, Base):
    __tablename__ = "oauth_identities"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    provider: Mapped[str] = mapped_column(String(24))
    issuer: Mapped[str] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(Text)
    provider_email: Mapped[str] = mapped_column(Text)
    email_verified: Mapped[bool] = mapped_column(Boolean)
    __table_args__ = (
        UniqueConstraint("issuer", "subject", name="uq_oauth_identity_subject"),
        CheckConstraint("provider = 'google'", name="ck_oauth_provider_google"),
    )


class AuthSession(TimestampMixin, Base):
    __tablename__ = "auth_sessions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    csrf_digest: Mapped[bytes | None] = mapped_column(LargeBinary)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revocation_reason: Mapped[str | None] = mapped_column(Text)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        UniqueConstraint("id", "user_id", name="uq_auth_session_owner"),
        Index("ix_auth_sessions_user_revoked", "user_id", "revoked_at"),
        Index("ix_auth_sessions_expires", "expires_at"),
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    auth_session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("auth_sessions.id", ondelete="CASCADE"), index=True
    )
    token_digest: Mapped[bytes] = mapped_column(LargeBinary, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        Index(
            "uq_refresh_one_active",
            "auth_session_id",
            unique=True,
            postgresql_where=text("consumed_at IS NULL"),
        ),
    )


class OAuthTransaction(Base):
    __tablename__ = "oauth_transactions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    state_digest: Mapped[bytes] = mapped_column(LargeBinary, unique=True)
    nonce_digest: Mapped[bytes] = mapped_column(LargeBinary)
    browser_binding_digest: Mapped[bytes] = mapped_column(LargeBinary)
    pkce_verifier_ciphertext: Mapped[bytes] = mapped_column(LargeBinary)
    return_path: Mapped[str] = mapped_column(Text, default="/")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    kind: Mapped[str] = mapped_column(String(32))
    original_filename: Mapped[str] = mapped_column(Text)
    storage_backend: Mapped[str] = mapped_column(String(16), default="local")
    object_key: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    media_type: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="uploading")
    extracted_text: Mapped[str | None] = mapped_column(Text)
    extraction_version: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(Text)
    client_request_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    request_hash: Mapped[str] = mapped_column(String(64))
    deletion_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    unreferenced_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        CheckConstraint("kind IN ('resume','job_description')", name="ck_documents_kind"),
        CheckConstraint("storage_backend IN ('local','s3')", name="ck_documents_storage"),
        CheckConstraint(
            "status IN ('uploading','processing','ready','failed','deleting')",
            name="ck_documents_status",
        ),
        CheckConstraint("size_bytes > 0", name="ck_documents_size"),
        UniqueConstraint("storage_backend", "object_key", name="uq_documents_object"),
        UniqueConstraint("owner_id", "client_request_id", name="uq_documents_request"),
        UniqueConstraint("id", "owner_id", name="uq_documents_owner"),
        UniqueConstraint("id", "owner_id", "kind", name="uq_documents_owner_kind"),
        Index("ix_documents_owner_kind_created", "owner_id", "kind", "created_at", "id"),
    )


class InterviewSession(TimestampMixin, Base):
    __tablename__ = "interview_sessions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    resume_document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    jd_document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    resume_kind: Mapped[str] = mapped_column(String(32), default="resume")
    jd_kind: Mapped[str] = mapped_column(String(32), default="job_description")
    status: Mapped[str] = mapped_column(String(16), default="preparing")
    requested_role: Mapped[str] = mapped_column(Text)
    inferred_role: Mapped[str | None] = mapped_column(Text)
    inferred_experience: Mapped[str | None] = mapped_column(Text)
    role_profile: Mapped[dict | None] = mapped_column(JSONB)
    runtime_version: Mapped[str] = mapped_column(Text, default="aria-live-v1")
    policy_mode: Mapped[str] = mapped_column(Text, default="placeholder-cycle")
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict)
    revision: Mapped[int] = mapped_column(BigInteger, default=0)
    accepted_answer_count: Mapped[int] = mapped_column(Integer, default=0)
    client_request_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    request_hash: Mapped[str] = mapped_column(String(64))
    consent_version: Mapped[str] = mapped_column(Text)
    consented_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_error_code: Mapped[str | None] = mapped_column(Text)
    deletion_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        ForeignKeyConstraint(
            ["resume_document_id", "owner_id", "resume_kind"],
            ["documents.id", "documents.owner_id", "documents.kind"],
            ondelete="RESTRICT",
            name="fk_interviews_resume_owner_kind",
        ),
        ForeignKeyConstraint(
            ["jd_document_id", "owner_id", "jd_kind"],
            ["documents.id", "documents.owner_id", "documents.kind"],
            ondelete="RESTRICT",
            name="fk_interviews_jd_owner_kind",
        ),
        CheckConstraint("resume_document_id <> jd_document_id", name="ck_interviews_distinct_docs"),
        CheckConstraint("resume_kind = 'resume'", name="ck_interviews_resume_kind"),
        CheckConstraint("jd_kind = 'job_description'", name="ck_interviews_jd_kind"),
        CheckConstraint(
            "status IN ('preparing','ready','active','paused','completed','cancelled','failed','expired')",
            name="ck_interviews_status",
        ),
        UniqueConstraint("id", "owner_id", name="uq_interviews_owner"),
        UniqueConstraint("owner_id", "client_request_id", name="uq_interviews_request"),
        Index("ix_interviews_owner_created", "owner_id", "created_at", "id"),
        Index("ix_interviews_owner_status_activity", "owner_id", "status", "last_activity_at"),
    )


class BackgroundJob(TimestampMixin, Base):
    __tablename__ = "background_jobs"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    document_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    kind: Mapped[str] = mapped_column(String(32))
    deduplication_key: Mapped[str] = mapped_column(Text, unique=True)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    lease_token: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    leased_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "owner_id"],
            ["interview_sessions.id", "interview_sessions.owner_id"],
            ondelete="CASCADE",
            name="fk_jobs_session_owner",
        ),
        ForeignKeyConstraint(
            ["document_id", "owner_id"],
            ["documents.id", "documents.owner_id"],
            ondelete="CASCADE",
            name="fk_jobs_document_owner",
        ),
        CheckConstraint(
            "status IN ('queued','running','succeeded','failed','cancelled')",
            name="ck_jobs_status",
        ),
        Index(
            "ix_jobs_queued_available",
            "available_at",
            "id",
            postgresql_where=text("status = 'queued'"),
        ),
        Index("ix_jobs_running_lease", "leased_until"),
    )


class GenerationAttempt(Base):
    __tablename__ = "generation_attempts"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("background_jobs.id", ondelete="CASCADE"))
    turn_index: Mapped[int] = mapped_column(Integer)
    attempt_index: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    model_identifier: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(Text)
    generation_config: Mapped[dict] = mapped_column(JSONB, default=dict)
    validation_reasons: Mapped[list] = mapped_column(JSONB, default=list)
    error_code: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "owner_id"],
            ["interview_sessions.id", "interview_sessions.owner_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "status IN ('running','accepted','rejected','failed','interrupted')",
            name="ck_generation_attempt_status",
        ),
        UniqueConstraint("job_id", "attempt_index", name="uq_generation_attempt_job_index"),
        UniqueConstraint("id", "session_id", "owner_id", name="uq_generation_attempt_owner"),
        Index(
            "uq_generation_one_accepted_turn",
            "session_id",
            "turn_index",
            unique=True,
            postgresql_where=text("status = 'accepted'"),
        ),
    )


class Question(Base):
    __tablename__ = "questions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    turn_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    target_skill_id: Mapped[str] = mapped_column(Text)
    generation_attempt_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True)
    grounding_context: Mapped[dict] = mapped_column(JSONB)
    model_identifier: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(Text)
    generation_config: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "owner_id"],
            ["interview_sessions.id", "interview_sessions.owner_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["generation_attempt_id", "session_id", "owner_id"],
            ["generation_attempts.id", "generation_attempts.session_id", "generation_attempts.owner_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("turn_index > 0", name="ck_questions_turn_positive"),
        CheckConstraint("length(btrim(text)) > 0", name="ck_questions_text_nonempty"),
        UniqueConstraint("session_id", "turn_index", name="uq_questions_session_turn"),
        UniqueConstraint("id", "session_id", "owner_id", name="uq_questions_owner"),
    )


class Answer(TimestampMixin, Base):
    __tablename__ = "answers"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    question_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    client_submission_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    request_hash: Mapped[str] = mapped_column(String(64))
    input_mode: Mapped[str] = mapped_column(String(12))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    answer_text: Mapped[str | None] = mapped_column(Text)
    transcriber_model: Mapped[str | None] = mapped_column(Text)
    transcriber_version: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(Text)
    audio_duration_ms: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(Text)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        ForeignKeyConstraint(
            ["question_id", "session_id", "owner_id"],
            ["questions.id", "questions.session_id", "questions.owner_id"],
            ondelete="CASCADE",
        ),
        CheckConstraint("input_mode IN ('text','audio')", name="ck_answers_input_mode"),
        CheckConstraint("status IN ('pending','accepted','failed')", name="ck_answers_status"),
        UniqueConstraint("session_id", "client_submission_id", name="uq_answers_submission"),
        Index(
            "uq_answers_one_current_question",
            "question_id",
            unique=True,
            postgresql_where=text("status IN ('pending','accepted')"),
        ),
    )


class SessionCheckpoint(Base):
    __tablename__ = "session_checkpoints"
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    revision: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer)
    state: Mapped[dict] = mapped_column(JSONB)
    runtime_version: Mapped[str] = mapped_column(Text)
    provenance: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "owner_id"],
            ["interview_sessions.id", "interview_sessions.owner_id"],
            ondelete="CASCADE",
        ),
    )


class InterviewConnectionLease(TimestampMixin, Base):
    __tablename__ = "interview_connection_leases"
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    auth_session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    connection_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    epoch: Mapped[int] = mapped_column(BigInteger, default=1)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "owner_id"],
            ["interview_sessions.id", "interview_sessions.owner_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["auth_session_id", "owner_id"],
            ["auth_sessions.id", "auth_sessions.user_id"],
            ondelete="CASCADE",
        ),
    )


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    event_type: Mapped[str] = mapped_column(Text)
    resource_type: Mapped[str | None] = mapped_column(Text)
    resource_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    request_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    metadata_json: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        Index("ix_audit_actor_created", "actor_user_id", "created_at"),
        Index("ix_audit_created", "created_at"),
    )
