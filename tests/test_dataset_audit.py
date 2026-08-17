from modules.module_07_rl.dataset_audit import audit_dataset


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

    report = audit_dataset(transitions)
    assert report["num_episodes"] == 3
    assert report["terminal_micro_f1"] == 1.0
    assert report["terminal_true_label_counts"] == {0: 1, 1: 1, 2: 1}
    assert report["invalid_evaluations"] == 0
    assert report["passes_quality_gates"] is True


def test_audit_flags_prediction_collapse_and_model_overlap():
    transitions = [
        _transition(f"ep-{index}", index % 3, 0, True, -0.1, ("same", "same"))
        for index in range(9)
    ]
    report = audit_dataset(transitions)
    assert report["passes_quality_gates"] is False
    assert any("collapse" in warning for warning in report["warnings"])
    assert any("overlap" in warning for warning in report["warnings"])
