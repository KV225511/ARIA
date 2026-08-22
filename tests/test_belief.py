import pytest
import numpy as np
from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_06_belief.belief_state import BeliefStateUpdater

@pytest.fixture
def nodes():
    return ["REST API", "SQL", "Docker"]

@pytest.fixture
def updater(nodes):
    return BeliefStateUpdater(nodes)

def test_initial_belief_uniform(updater, nodes):
    """Test that all nodes start with uniform distributions."""
    for node in nodes:
        belief = updater.get_belief(node)
        assert np.allclose(belief, [1/3, 1/3, 1/3], atol=1e-6)
        assert np.isclose(np.sum(belief), 1.0)

def test_initial_global_entropy(updater):
    """Test that global entropy starts around 1.0986 (ln 3)."""
    entropy = updater.get_global_entropy()
    assert np.isclose(entropy, 1.0986, atol=1e-3)

def test_get_belief_unknown_node(updater):
    """Test getting belief for a node not in the graph."""
    belief = updater.get_belief("Unknown Node")
    assert np.allclose(belief, [1/3, 1/3, 1/3], atol=1e-6)

def test_update_belief_unknown_node(updater):
    """Test updating belief for an unknown node (should not crash or modify others)."""
    updater.update_belief("Unknown Node", semantic_score=0.9, cognitive_load="low", behavior_score=0.9)
    assert np.allclose(updater.get_belief("Unknown Node"), [1/3, 1/3, 1/3], atol=1e-6)

def test_update_belief_strong_performance(updater):
    """Test that strong semantic + strong behavior + low load shifts belief towards expert."""
    updater.update_belief("SQL", semantic_score=0.9, cognitive_load="low", behavior_score=0.9)
    belief = updater.get_belief("SQL")
    assert belief[2] > belief[0] # Expert > Beginner
    assert belief[2] > 0.5 # Should have high confidence in expert

def test_update_belief_anxiety_overrides_poor_behavior(updater):
    """Test that anxiety load weighs semantic score higher than behavior score."""
    updater.update_belief("REST API", semantic_score=0.9, cognitive_load="anxiety", behavior_score=0.1)
    belief = updater.get_belief("REST API")
    # Even though behavior was bad, they knew the answer and were anxious.
    # Evidence score should be (0.9*0.9) + (0.1*0.1) = 0.82 -> Expert likelihood
    assert belief[2] > belief[0]
    assert belief[2] > 0.5

def test_update_belief_ignorance_penalizes(updater):
    """Test that ignorance load with poor semantics shifts belief towards beginner."""
    updater.update_belief("Docker", semantic_score=0.1, cognitive_load="ignorance", behavior_score=0.1)
    belief = updater.get_belief("Docker")
    assert belief[0] > belief[2] # Beginner > Expert
    assert belief[0] > 0.5

def test_entropy_decreases_with_certainty(updater):
    """Test that entropy drops as belief shifts strongly to one side."""
    initial_entropy = updater.get_global_entropy()
    updater.update_belief("SQL", semantic_score=0.9, cognitive_load="low", behavior_score=0.9)
    new_entropy = updater.get_global_entropy()
    assert new_entropy < initial_entropy

def test_zero_vector_normalization(updater):
    """Test that passing a zero vector to _normalize returns uniform."""
    dist = np.array([0.0, 0.0, 0.0])
    norm = updater._normalize(dist)
    assert np.allclose(norm, [1/3, 1/3, 1/3], atol=1e-3)


def test_aggregate_belief_ignores_unvisited_skills(updater):
    updater.update_belief(
        "SQL", semantic_score=0.9, cognitive_load="low", behavior_score=0.9
    )
    aggregate = updater.get_aggregate_belief()
    assert np.allclose(aggregate, updater.get_belief("SQL"))
    assert updater.get_visited_skills() == ["SQL"]


def test_aggregate_assessment_tracks_evidence_counts(updater):
    updater.update_belief(
        "SQL", semantic_score=0.8, cognitive_load="low", behavior_score=0.8
    )
    updater.update_belief(
        "Docker", semantic_score=0.2, cognitive_load="low", behavior_score=0.2
    )
    assessment = updater.get_aggregate_assessment()
    assert assessment["evidence_counts"]["SQL"] == 1
    assert assessment["evidence_counts"]["Docker"] == 1
    assert set(assessment["visited_skills"]) == {"SQL", "Docker"}
    assert np.isclose(np.sum(assessment["belief"]), 1.0)


def test_low_confidence_evidence_moves_belief_less(updater):
    low_confidence = BeliefStateUpdater(["SQL"])
    high_confidence = BeliefStateUpdater(["SQL"])
    low_confidence.update_belief(
        "SQL", 0.9, "low", 0.9, evidence_confidence=0.2
    )
    high_confidence.update_belief(
        "SQL", 0.9, "low", 0.9, evidence_confidence=1.0
    )
    assert high_confidence.get_belief("SQL")[2] > low_confidence.get_belief("SQL")[2]
    assert low_confidence.evidence_strengths["SQL"] == pytest.approx(0.2)


def test_behavior_does_not_change_competency_posterior():
    low_behavior = BeliefStateUpdater(["SQL"])
    high_behavior = BeliefStateUpdater(["SQL"])
    low_behavior.update_belief("SQL", 0.7, "low", behavior_score=0.0)
    high_behavior.update_belief("SQL", 0.7, "low", behavior_score=1.0)
    assert np.allclose(
        low_behavior.get_belief("SQL"), high_behavior.get_belief("SQL")
    )


def test_repeated_evidence_has_sublinear_effective_weight():
    updater = BeliefStateUpdater(["SQL"])
    updater.update_belief("SQL", 0.8, "low", evidence_confidence=1.0)
    first_ess = updater.get_effective_sample_size("SQL")
    updater.update_belief("SQL", 0.8, "low", evidence_confidence=1.0)
    second_increment = updater.get_effective_sample_size("SQL") - first_ess
    assert first_ess == pytest.approx(1.0)
    assert second_increment < first_ess


def test_duplicate_question_receives_extra_discount():
    unique = BeliefStateUpdater(["SQL"])
    duplicate = BeliefStateUpdater(["SQL"])
    unique.update_belief("SQL", 0.8, question_fingerprint="a")
    unique.update_belief("SQL", 0.8, question_fingerprint="b")
    duplicate.update_belief("SQL", 0.8, question_fingerprint="a")
    duplicate.update_belief("SQL", 0.8, question_fingerprint="a")
    assert duplicate.get_effective_sample_size("SQL") < unique.get_effective_sample_size("SQL")


def test_non_finite_evidence_is_rejected_without_state_change():
    updater = BeliefStateUpdater(["SQL"])
    before = updater.get_belief("SQL")
    with pytest.raises(ValueError):
        updater.update_belief("SQL", float("nan"))
    assert np.allclose(before, updater.get_belief("SQL"))
    assert updater.get_evidence_count("SQL") == 0


def test_assessment_abstains_when_configured_evidence_is_missing():
    config = BeliefModelConfig(
        minimum_assessment_confidence=0.8,
        minimum_effective_evidence=2.0,
        minimum_skill_coverage=2,
    )
    updater = BeliefStateUpdater(["SQL", "Docker"], config=config)
    assessment = updater.get_aggregate_assessment()
    assert assessment["label"] is None
    assert assessment["status"] == "insufficient_evidence"
