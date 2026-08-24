from modules.module_07_rl.metrics import build_belief_report


def test_belief_report_is_not_labeled_as_policy_evaluation():
    dataset = [
        {"true_label": 0, "aria_label": 0, "action_idx": 1, "reward": 0.1, "done": True},
        {"true_label": 1, "aria_label": 1, "action_idx": 2, "reward": 0.2, "done": True},
        {"true_label": 2, "aria_label": 2, "action_idx": 3, "reward": 0.3, "done": True},
    ]
    report = build_belief_report(dataset)
    assert report["evaluation_type"] == "stored_belief_verdict"
    assert report["evaluates_learned_policy"] is False
    assert report["rl_metrics"]["evaluates_learned_policy"] is False
    assert "logged_action_entropy" in report["rl_metrics"]
    assert "policy_entropy" not in report["rl_metrics"]
    assert report["belief_verdict_metrics"]["micro_f1"] == 1.0
    assert report["belief_verdict_metrics"]["ordinal_mae"] == 0.0
    assert report["evaluates_learned_policy"] is False


def test_belief_report_handles_abstention_and_calibration_metrics():
    dataset = [
        {
            "episode_id": "a",
            "resume_file": "r1",
            "jd_file": "j1",
            "true_label": 0,
            "aria_label": None,
            "aggregate_belief": [0.4, 0.35, 0.25],
            "action_idx": 1,
            "reward": 0.0,
            "done": True,
        },
        {
            "episode_id": "b",
            "resume_file": "r2",
            "jd_file": "j2",
            "true_label": 2,
            "aria_label": 2,
            "aggregate_belief": [0.05, 0.1, 0.85],
            "action_idx": 2,
            "reward": 0.0,
            "done": True,
        },
    ]
    report = build_belief_report(dataset)
    metrics = report["belief_verdict_metrics"]
    assert metrics["abstention_count"] == 1
    assert metrics["brier_score"] is not None
    assert metrics["expected_calibration_error"] is not None
