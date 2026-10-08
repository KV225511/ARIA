from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import uuid

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
import fitz
import pytest

from backend.api import app
from backend.auth import decode_access_token, issue_access_token
from backend.documents import extract_pdf, safe_filename
from backend.models import Base, InterviewSession
from backend.settings import AppSettings
from backend.storage import LocalPrivateStorage
from modules.module_05_ontology.graph import SkillOntologyGraph
from modules.module_06_belief.belief_state import BeliefStateUpdater


def jwt_settings(tmp_path: Path) -> AppSettings:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return replace(
        AppSettings(),
        jwt_private_key_path=private_path,
        jwt_public_key_path=public_path,
        storage_root=tmp_path / "documents",
    )


def test_access_jwt_is_bound_to_user_and_auth_session(tmp_path):
    settings = jwt_settings(tmp_path)
    user_id, session_id = uuid.uuid4(), uuid.uuid4()
    token = issue_access_token(user_id, session_id, settings)
    assert decode_access_token(token, settings) == (user_id, session_id)


def test_database_contract_has_owner_scoped_interview_inputs():
    tables = Base.metadata.tables
    assert {"users", "documents", "interview_sessions", "questions", "answers"} <= set(tables)
    constraints = {constraint.name for constraint in InterviewSession.__table__.constraints}
    assert "fk_interviews_resume_owner_kind" in constraints
    assert "fk_interviews_jd_owner_kind" in constraints
    assert "uq_interviews_request" in constraints


def test_pdf_validation_and_private_storage_boundaries(tmp_path):
    settings = replace(AppSettings(), storage_root=tmp_path / "documents")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Private synthetic résumé")
    content = pdf.tobytes()
    pdf.close()
    text, pages = extract_pdf(content, settings)
    assert pages == 1
    assert "Private synthetic résumé" in text
    storage = LocalPrivateStorage(settings)
    assert safe_filename("../../candidate?.pdf") == "candidate_.pdf"
    with pytest.raises(ValueError):
        storage.resolve("../../outside.pdf")


def test_checkpoint_round_trip_preserves_belief_and_ontology():
    jd = (
        "Backend Developer role requiring Python, REST API, SQL, database design, "
        "Docker, Linux, authentication, Kubernetes, CI/CD, system design, testing, "
        "and diagnosis of production failures."
    )
    ontology = SkillOntologyGraph()
    assert ontology.adapt_to_candidate(jd, "Three years building Python services.")
    restored = SkillOntologyGraph()
    restored.restore_profile(ontology.role_profile.to_dict(), ontology.inferred_experience)
    assert restored.role_profile.profile_hash == ontology.role_profile.profile_hash
    belief = BeliefStateUpdater(restored.get_all_skills())
    skill = restored.get_all_skills()[0]
    belief.update_belief(skill, 0.8, "low", 0.7)
    replayed = BeliefStateUpdater.from_checkpoint(belief.to_checkpoint())
    assert replayed.evidence_counts == belief.evidence_counts
    assert replayed.get_belief(skill).tolist() == pytest.approx(belief.get_belief(skill).tolist())


def test_legacy_anonymous_session_creation_is_closed():
    with TestClient(app) as client:
        response = client.post("/api/start-session")
    assert response.status_code == 410
