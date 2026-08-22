from modules.module_07_rl.dataset_audit import (
    audit_calibration_validation,
    audit_learned_policy_evaluation,
    audit_belief_predictions,
    audit_dataset,
    audit_raw_evidence,
)


def _transition(episode, label, prediction, done, reward, model_pair=("candidate", "judge")):
    return {
        "episode_id": episode,
        "true_label": label,
        "aria_label": prediction,
        "done": done,
        "reward": reward,
        "semantic_score": 0.2 + 0.3 * label,
        "skills_covered": 5,
        "evaluation_valid": True,
        "action_idx": label,
        "behavior_policy": "coverage_heuristic",
        "candidate_model": model_pair[0],
        "evaluator_model": model_pair[1],
        "resume_file": f"resume-{episode}",
        "jd_file": f"jd-{episode}",
    }


def test_audit_reports_balanced_terminal_metrics():
    transitions = []
    for label in range(3):
        transitions.extend([
            _transition(f"ep-{label}", label, label, False, -0.1 - label * 0.1),
            _transition(f"ep-{label}", label, label, True, 0.2 + label * 0.1),
        ])

    report = audit_dataset(transitions, min_episodes=3)
    assert report["num_episodes"] == 3
    assert report["terminal_micro_f1"] == 1.0
    assert report["terminal_true_label_counts"] == {0: 1, 1: 1, 2: 1}
    assert report["invalid_evaluations"] == 0
    assert report["passes_quality_gates"] is True


def test_validation_and_policy_gates_remain_distinct():
    transitions = []
    for label in range(3):
        transitions.append(_transition(f"ep-{label}", label, label, True, 0.2))
    validation = audit_calibration_validation(transitions)
    assert validation["gate"] == "calibration_validation"
    assert "ordinal_mae" in validation
    assert "confusion_matrix" in validation

    fixed_offline_report = {
        "evaluation_type": "stored_belief_verdict",
        "evaluates_learned_policy": False,
        "num_episodes": 3,
    }
    assert not audit_learned_policy_evaluation(fixed_offline_report)[
        "passes_quality_gates"
    ]
    rollout_report = {
        "evaluation_type": "learned_policy_rollout",
        "fresh_rollouts": True,
        "checkpoint_hash": "abc123",
        "num_episodes": 3,
    }
    assert audit_learned_policy_evaluation(rollout_report)["passes_quality_gates"]


def test_audit_flags_prediction_collapse_and_model_overlap():
    transitions = [
        _transition(f"ep-{index}", index % 3, 0, True, -0.1, ("same", "same"))
        for index in range(9)
    ]
    report = audit_dataset(transitions, min_episodes=3)
    assert report["passes_quality_gates"] is False
    assert any("collapse" in warning for warning in report["warnings"])
    assert any("overlap" in warning for warning in report["warnings"])


def test_audit_requires_enough_episodes_for_reliable_evaluation():
    transitions = [
        _transition(f"ep-{index}", index % 3, index % 3, True, 0.1)
        for index in range(9)
    ]
    report = audit_dataset(transitions, min_episodes=10)
    assert report["passes_quality_gates"] is False
    assert any("at least 10" in warning for warning in report["warnings"])


def test_raw_gate_does_not_fail_only_because_stored_beliefs_collapse():
    transitions = [
        _transition(f"ep-{index}", index % 3, 1, True, 0.1)
        for index in range(9)
    ]
    raw = audit_raw_evidence(
        transitions, min_episodes=3, min_independent_components=3
    )
    belief = audit_belief_predictions(transitions)
    assert raw["passes_quality_gates"] is True
    assert belief["passes_quality_gates"] is False


def test_content_hash_detects_renamed_duplicate_and_cross_split_leakage():
    transitions = []
    for index, split in enumerate(("train", "validation", "test")):
        item = _transition(f"ep-{index}", index, index, True, 0.1)
        item["dataset_split"] = split
        item["resume_content_hash"] = "same-content" if index < 2 else "other"
        item["resume_file"] = f"renamed-{index}.pdf"
        item["jd_content_hash"] = f"jd-{index}"
        transitions.append(item)
    report = audit_raw_evidence(
        transitions, min_episodes=3, min_independent_components=1
    )
    assert report["split_leaking_resumes"] == ["same-content"]
    assert report["renamed_duplicate_resumes"]
