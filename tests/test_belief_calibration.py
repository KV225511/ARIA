import inspect

import pytest

from modules.module_07_rl.belief_calibration import (
    calibrate_belief_model,
    fit_emission_config,
    calibrate_likelihood_sigma,
    evaluate_likelihood_sigma,
)


def _episode(episode_id, label, score):
    return [
        {
            "episode_id": episode_id,
            "target_skill": f"Skill-{turn}",
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


def test_emission_fit_is_episode_balanced():
    transitions = []
    transitions.extend(_episode("beginner-short", 0, 0.1)[:1])
    transitions.extend(_episode("mid", 1, 0.5))
    transitions.extend(_episode("expert", 2, 0.9))
    # A long Beginner episode must not dominate solely because it has more turns.
    long_episode = _episode("beginner-long", 0, 0.3) * 10
    for index, transition in enumerate(long_episode):
        transition = dict(transition)
        transition["episode_id"] = "beginner-long"
        transition["done"] = index == len(long_episode) - 1
        transitions.append(transition)
    config = fit_emission_config(transitions)
    assert 0.1 <= config.class_centers[0] <= 0.3
    assert config.class_centers[0] <= config.class_centers[1] <= config.class_centers[2]


def test_emission_fit_api_cannot_receive_validation_or_test_labels(separable_validation_data):
    first = fit_emission_config(separable_validation_data)
    second = fit_emission_config(separable_validation_data)
    assert first.class_centers == second.class_centers
    assert first.class_scales == second.class_scales
    fit_parameters = set(inspect.signature(fit_emission_config).parameters)
    assert not {"validation_transitions", "test_transitions"} & fit_parameters
    calibration_parameters = set(inspect.signature(calibrate_belief_model).parameters)
    assert "training_transitions" in calibration_parameters
    assert "validation_transitions" in calibration_parameters
    assert "test_transitions" not in calibration_parameters
