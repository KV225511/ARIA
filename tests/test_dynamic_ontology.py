import pytest

from modules.module_05_ontology.graph import SkillOntologyGraph


@pytest.fixture
def ontology():
    return SkillOntologyGraph("backend_developer")


def test_adaptation_fails_closed_for_unsupported_jd(ontology):
    baseline = set(ontology.get_all_skills())
    success = ontology.adapt_to_candidate(
        "We need a specialist for a confidential discipline. " * 4,
        "Candidate profile with sufficient length. " * 4,
    )
    assert success is False
    assert ontology.role_profile is None
    assert set(ontology.get_all_skills()) == baseline


def test_adaptation_builds_evidence_backed_graph(ontology):
    jd = (
        "Job Title: Backend Platform Engineer\n"
        "Required qualifications include Python, REST API development, SQL, "
        "Docker, Kubernetes, authentication, system design, and microservices. "
        "The engineer owns reliable production services and validation."
    )
    resume = (
        "Software engineer with Python, REST API, SQL, Docker and Kubernetes "
        "experience across production platform teams."
    )
    assert ontology.adapt_to_candidate(jd, resume) is True
    skills = set(ontology.get_all_skills())
    assert {"Python", "REST API", "SQL", "Docker", "Kubernetes"} <= skills
    assert ontology.inferred_role == "Backend Platform Engineer"
    assert ontology.role_profile is not None
    assert ontology.get_skill_metadata("Python").jd_evidence
    assert "Docker" in ontology.get_prerequisites("Kubernetes")


def test_bms_is_resolved_in_battery_domain(ontology):
    jd = (
        "Position: Battery Validation Engineer\n"
        "Required work includes battery pack testing, BMS validation, DCIR, "
        "thermal testing, mechanical testing, and root cause analysis for "
        "automotive battery systems."
    )
    assert ontology.adapt_to_candidate(jd, "Battery test engineer resume." * 8)
    resolution = ontology.role_profile.acronym_resolutions[0]
    assert resolution["expansion"] == "Battery Management System"
    assert "Battery Management System" in ontology.get_all_skills()


def test_ambiguous_bms_fails_closed(ontology):
    jd = (
        "Job Title: Facilities Engineer\n"
        "The engineer operates BMS controls and coordinates facilities work "
        "with vendors, technicians, maintenance teams, and site stakeholders."
    )
    assert ontology.adapt_to_candidate(jd, "Facilities engineer." * 10) is False
    assert ontology.role_profile is None
