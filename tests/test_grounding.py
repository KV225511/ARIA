import pytest

from modules.module_05_ontology.grounding import (
    build_pairing_record,
    build_role_profile,
    grounding_packet,
    grounding_contract_hash,
    normalize_generated_question,
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('  "Question: How does BMS protect a battery pack?"  ',
         "How does BMS protect a battery pack?"),
        ("```text\nHow does BMS protect a battery pack?\n```",
         "How does BMS protect a battery pack?"),
        ("Interview Question:   How does BMS protect a battery pack?   ",
         "How does BMS protect a battery pack?"),
    ],
)
def test_question_normalization_removes_only_harmless_wrappers(raw, expected):
    assert normalize_generated_question(raw) == expected


def test_profile_has_verifiable_evidence_and_domain_resolution():
    profile = build_role_profile(JD, "Engineer experienced in battery testing and DCIR.")
    validate_role_profile(profile, JD)
    assert profile.role_domain == "mechanical_automotive"
    assert profile.acronym_resolutions[0]["expansion"] == "Battery Management System"
    assert all(skill.jd_evidence for skill in profile.skills)
    packet = grounding_packet(profile, profile.skills[0])
    assert packet["grounding_policy_version"] == "aria-grounding-policy-v2"
    assert packet["target_aliases"]


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


def test_question_validator_accepts_inflected_alias_and_short_technical_token():
    packet = {
        "role_profile_hash": "engineering-profile",
        "target_skill_id": "engineering-drawings",
        "target_skill": "Engineering Drawings",
        "target_aliases": ["2D drawings", "technical drawings"],
        "target_definition": (
            "Creation and interpretation of dimensioned technical drawings and "
            "assembly documentation."
        ),
        "jd_evidence": ["2D drawings", "engineering drawings"],
        "allowed_skill_ids": ["engineering-drawings"],
        "acronym_resolutions": [],
    }
    result = validate_grounded_question(
        "How would you create a 2D drawing for a complex mechanical component?",
        packet,
        [],
    )
    assert result["valid"] is True
    assert {"2d", "drawing"} <= set(result["supporting_terms"])
    assert result["target_skill_id"] == "engineering-drawings"
    assert result["role_profile_hash"] == "engineering-profile"


def test_question_validator_accepts_definition_vocabulary():
    packet = {
        "target_skill_id": "electrical-circuits",
        "target_skill": "Electrical Circuits",
        "target_aliases": ["circuits", "ohm's law"],
        "target_definition": (
            "Circuit analysis using voltage, current, resistance, and network laws."
        ),
        "jd_evidence": ["schematic"],
        "allowed_skill_ids": ["electrical-circuits"],
        "acronym_resolutions": [],
    }
    result = validate_grounded_question(
        "How are voltage and current related through resistance?", packet, []
    )
    assert result["valid"] is True
    assert "voltage" in result["supporting_terms"]

    generic = validate_grounded_question(
        "How would this system work?", packet, []
    )
    assert generic["valid"] is False


def test_question_validator_handles_imperative_and_trailing_text_deliberately():
    profile = build_role_profile(JD, "Battery testing experience.")
    packet = grounding_packet(profile, profile.skill("battery-testing"))
    imperative = validate_grounded_question(
        "Given an intermittent pack failure, describe how you would test the battery.",
        packet,
        [],
    )
    assert imperative["valid"] is True

    compound = validate_grounded_question(
        "How would you test the battery? Describe a second independent procedure.",
        packet,
        [],
    )
    assert compound["valid"] is False
    assert any("trailing text" in reason for reason in compound["reasons"])


def test_related_off_profile_term_is_context_not_a_competency_switch():
    packet = {
        "target_skill_id": "pcb-design",
        "target_skill": "PCB Design",
        "target_aliases": ["pcb", "board layout"],
        "target_definition": (
            "Schematic capture, board layout, signal integrity, and design validation."
        ),
        "jd_evidence": ["schematic"],
        "allowed_skill_ids": ["pcb-design"],
        "acronym_resolutions": [],
    }
    result = validate_grounded_question(
        "How would you design a PCB layout to limit EMI while preserving signal integrity?",
        packet,
        [],
    )
    assert result["valid"] is True
    assert result["off_profile_skill_ids"] == []
    assert result["contextual_skill_ids"] == ["emi-emc"]


def test_same_domain_deployment_detail_is_context_for_ci_cd():
    packet = {
        "target_skill_id": "ci-cd",
        "target_skill": "CI/CD",
        "target_aliases": [
            "continuous integration", "continuous deployment", "jenkins", "gitlab ci",
        ],
        "target_definition": "Automated build, test, release, and deployment workflows.",
        "jd_evidence": ["Jenkins"],
        "allowed_skill_ids": ["ci-cd"],
        "acronym_resolutions": [],
    }
    result = validate_grounded_question(
        "How would you handle database migrations in your CI/CD pipeline to ensure "
        "that the database schema is updated correctly during deployment, and what "
        "steps would you include in Jenkins to automate this task?",
        packet,
        [],
    )
    assert result["valid"] is True
    assert result["off_profile_skill_ids"] == []
    assert result["contextual_skill_ids"] == ["database-design"]


def test_cross_domain_competency_still_fails_closed():
    packet = {
        "target_skill_id": "battery-testing",
        "target_skill": "Battery Testing",
        "target_aliases": ["battery", "battery pack"],
        "target_definition": "Electrical and thermal performance validation of batteries.",
        "jd_evidence": ["battery pack testing"],
        "allowed_skill_ids": ["battery-testing"],
        "acronym_resolutions": [],
    }
    result = validate_grounded_question(
        "How would you use Kubernetes while testing a battery pack?", packet, []
    )
    assert result["valid"] is False
    assert result["off_profile_skill_ids"] == ["kubernetes"]


def test_short_jd_fails_closed():
    with pytest.raises(ValueError, match="at least 100"):
        build_role_profile("Python required")
