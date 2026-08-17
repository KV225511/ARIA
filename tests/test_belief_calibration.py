import pytest

from modules.module_07_rl.belief_calibration import (
    calibrate_likelihood_sigma,
    evaluate_likelihood_sigma,
)


def _episode(episode_id, label, score):
    return [
        {
            "episode_id": episode_id,
            "target_skill": "Python",
            "semantic_score": score,
            "behavior_score": score,
            "cognitive_load": "low",
            "evaluator_confidence": 1.0,
            "evaluation_valid": True,
            "true_label": label,
            "done": turn == 2,
        }
        for turn in range(3)
    ]


@pytest.fixture
def separable_validation_data():
    transitions = []
    for index in range(3):
        transitions.extend(_episode(f"beginner-{index}", 0, 0.1))
        transitions.extend(_episode(f"mid-{index}", 1, 0.5))
        transitions.extend(_episode(f"expert-{index}", 2, 0.9))
    return transitions


def test_replay_evaluation_recovers_separable_classes(separable_validation_data):
    result = evaluate_likelihood_sigma(separable_validation_data, 0.22)
    assert result["micro_f1"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["prediction_counts"] == {0: 3, 1: 3, 2: 3}


def test_calibration_selects_from_candidates(separable_validation_data):
    report = calibrate_likelihood_sigma(
        separable_validation_data, candidates=(0.16, 0.22, 0.30)
    )
    assert report["best"]["likelihood_sigma"] in {0.16, 0.22, 0.30}
    assert report["best"]["micro_f1"] == 1.0
    assert len(report["candidates"]) == 3
