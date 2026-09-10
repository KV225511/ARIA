import inspect

import pytest
from unittest.mock import patch

from modules.module_06_belief.belief_config import BeliefModelConfig

from modules.module_07_rl.belief_calibration import (
    calibrate_belief_model,
    fit_emission_config,
    calibrate_likelihood_sigma,
    evaluate_likelihood_sigma,
    apply_scale_shrinkage,
    build_grouped_cv_folds,
    paired_component_bootstrap,
    select_training_only_calibration,
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


def _grouped_training():
    rows = []
    for index in range(9):
        episode = _episode(f"train-{index}", index % 3, (0.1, 0.5, 0.9)[index % 3])
        for row in episode:
            row.update({
                "dataset_split": "train",
                "resume_content_hash": f"resume-{index}",
                "jd_content_hash": f"jd-{index}",
            })
        rows.extend(episode)
    return rows


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


def test_grouped_cv_is_deterministic_complete_and_identity_safe():
    transitions = _grouped_training()
    first = build_grouped_cv_folds(transitions)
    second = build_grouped_cv_folds(list(reversed(transitions)))
    assert first["fold_count"] == 3
    assert first["assignments"] == second["assignments"]
    assert sorted(first["assignments"].values()).count(0) > 0
    assert set(first["assignments"].values()) == {0, 1, 2}
    assert len(first["assignments"]) == 9
    resume_folds, jd_folds = {}, {}
    for component in first["components"]:
        for identity in component["resume_identities"]:
            resume_folds.setdefault(identity, set()).add(component["fold"])
        for identity in component["jd_identities"]:
            jd_folds.setdefault(identity, set()).add(component["fold"])
    assert all(len(value) == 1 for value in resume_folds.values())
    assert all(len(value) == 1 for value in jd_folds.values())
    with pytest.raises(ValueError, match="training transitions only"):
        build_grouped_cv_folds([{**transitions[0], "dataset_split": "validation"}])


def test_scale_shrinkage_formula_and_clipping():
    config = BeliefModelConfig(
        class_scales=(0.03, 0.20, 0.35),
        fit_metadata={"pooled_scale": 0.50},
    )
    adjusted = apply_scale_shrinkage(config, 0.5)
    assert adjusted.class_scales == pytest.approx((0.265, 0.35, 0.35))
    assert adjusted.fit_metadata["scale_shrinkage"] == 0.5


def _fake_candidate(parameters, eligible):
    values = {
        "scale_shrinkage": 0.0, "aggregation_temperature": 1.0,
        "minimum_assessment_confidence": 0.5, "repeat_discount_power": 0.5,
        "max_skill_effective_sample_size": 5.0,
    }
    values.update(parameters)
    metrics = {
        "overall_accuracy": 0.8, "macro_f1": 0.8,
        "minimum_class_recall": 0.7,
        "maximum_classified_prediction_share": 0.4,
        "abstention_rate": 0.0, "expected_calibration_error": 0.1,
        "ordinal_mae": 0.2, "true_label_counts": {0: 3, 1: 3, 2: 3},
        "decision_prediction_counts": {0: 3, 1: 3, 2: 3},
    }
    if not eligible:
        metrics["overall_accuracy"] = 0.0
    return {
        "parameters": values,
        "parameter_tuple": [values[name] for name in (
            "repeat_discount_power", "max_skill_effective_sample_size",
            "aggregation_temperature", "minimum_assessment_confidence", "scale_shrinkage",
        )],
        "eligible": eligible,
        "rejection_reasons": [] if eligible else ["overall_accuracy_below_minimum"],
        "metrics": metrics,
    }


@pytest.mark.parametrize(
    ("passing_stage", "expected_count", "expected_stages"),
    [("A", 1, ["A"]), ("B", 6, ["A", "B"]),
     ("C", 36, ["A", "B", "C"]), ("D", 45, ["A", "B", "C", "D"])],
)
def test_sequential_search_only_runs_needed_stages(monkeypatch, passing_stage, expected_count, expected_stages):
    calls = []

    def fake(_training, parameters, _folds):
        calls.append(dict(parameters))
        call_number = len(calls)
        stage = "A" if call_number == 1 else "B" if call_number <= 6 else "C" if call_number <= 36 else "D"
        return _fake_candidate(parameters, stage == passing_stage)

    monkeypatch.setattr(
        "modules.module_07_rl.belief_calibration.evaluate_candidate_cross_validated",
        fake,
    )
    report = select_training_only_calibration(_grouped_training(), "raw", "split", "protocol")
    assert report["candidate_count"] == expected_count
    assert report["candidate_count"] <= 45
    assert [item["stage"] for item in report["stages"]] == expected_stages
    assert report["validation_used_for_selection"] is False


def test_paired_component_bootstrap_is_deterministic_and_paired():
    transitions = []
    for index in range(6):
        transitions.append({
            "episode_id": f"v-{index}", "resume_file": f"r-{index}",
            "jd_file": f"j-{index}", "true_label": index % 3, "done": True,
        })
    truth = [index % 3 for index in range(6)]
    probabilities = [[1.0 if label == index % 3 else 0.0 for label in range(3)] for index in range(6)]
    first = paired_component_bootstrap(
        transitions, truth, probabilities, truth, probabilities, samples=20, seed=42,
    )
    second = paired_component_bootstrap(
        transitions, truth, probabilities, truth, probabilities, samples=20, seed=42,
    )
    assert first == second
    assert all(item["point_difference"] == 0.0 for item in first["intervals"].values())
    assert first["diagnostic_only"] is True
    assert len(first["leave_one_component_out"]) == 6
