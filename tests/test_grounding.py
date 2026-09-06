import pytest

from modules.module_05_ontology.grounding import (
    build_pairing_record,
    build_role_profile,
    grounding_packet,
    grounding_contract_hash,
    normalized_text_hash,
    validate_grounded_question,
    validate_role_profile,
)


JD = (
    "Job Title: Battery Validation Engineer\n"
    "Required qualifications include battery pack testing, BMS validation, "
    "DCIR, thermal testing, mechanical testing, and root cause analysis. "
    "The engineer documents test plans and validation results."
)


def test_document_hash_normalizes_newlines_and_spacing():
    assert normalized_text_hash("A  B\r\nC") == normalized_text_hash("A B\nC")
    assert len(grounding_contract_hash()) == 64


def test_profile_has_verifiable_evidence_and_domain_resolution():
    profile = build_role_profile(JD, "Engineer experienced in battery testing and DCIR.")
    validate_role_profile(profile, JD)
    assert profile.role_domain == "mechanical_automotive"
    assert profile.acronym_resolutions[0]["expansion"] == "Battery Management System"
    assert all(skill.jd_evidence for skill in profile.skills)


def test_resume_only_skill_does_not_enter_ontology():
    profile = build_role_profile(JD, "Python Docker Kubernetes specialist.")
    assert "docker" not in {skill.skill_id for skill in profile.skills}


def test_pairing_record_is_descriptive_not_a_filter():
    profile = build_role_profile(JD, "Battery testing and thermal testing experience.")
    record = build_pairing_record(profile)
    assert record["pairing_class"] == "evidence_overlap"
    assert 0 < record["resume_overlap_ratio"] <= 1


def test_question_validator_rejects_wrong_bms_expansion_and_duplicates():
    profile = build_role_profile(JD, "Battery testing experience.")
    skill = profile.skill("battery-management-system")
    packet = grounding_packet(profile, skill)
    wrong = validate_grounded_question(
        "How would you validate a Building Management System (BMS)?", packet, []
    )
    assert wrong["valid"] is False
    assert any("wrong domain" in reason for reason in wrong["reasons"])

    question = "How would you validate a battery management system?"
    duplicate = validate_grounded_question(question, packet, [{"q": question}])
    assert duplicate["valid"] is False
    assert any("duplicates" in reason for reason in duplicate["reasons"])


def test_question_validator_rejects_off_profile_technology():
    profile = build_role_profile(JD, "Battery testing experience.")
    packet = grounding_packet(profile, profile.skill("battery-management-system"))
    result = validate_grounded_question(
        "How would you use Kubernetes to validate a battery management system?",
        packet,
        [],
    )
    assert result["valid"] is False
    assert result["off_profile_skill_ids"] == ["kubernetes"]


def test_short_jd_fails_closed():
    with pytest.raises(ValueError, match="at least 100"):
        build_role_profile("Python required")
