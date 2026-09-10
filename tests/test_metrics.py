from modules.module_07_rl.metrics import build_belief_report, compute_classification_metrics


def test_canonical_metrics_use_classified_denominator_and_abstention_column():
    truths = [0] * 15 + [1] * 35 + [2] * 23 + [0] * 12 + [2] * 2
    predictions = [0] * 3 + [1] * 12 + [1] * 35 + [1] * 4 + [2] * 19 + [None] * 14
    metrics = compute_classification_metrics(truths, predictions)
    assert metrics["num_examples"] == 87
    assert metrics["num_classified"] == 73
    assert metrics["num_abstained"] == 14
    assert metrics["maximum_classified_prediction_share"] == 51 / 73
    assert metrics["micro_f1"] == metrics["overall_accuracy"]
    assert [sum(row) for row in metrics["confusion_matrix"]] == [27, 35, 25]
    assert metrics["confusion_matrix"][0][3] == 12
    assert metrics["confusion_matrix"][2][3] == 2


def test_ece_is_independent_of_decision_abstention():
    metrics = compute_classification_metrics(
        [0, 1], [None, 1], [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]
    )
    assert metrics["expected_calibration_error"] < 0.2
    assert metrics["abstention_rate"] == 0.5


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
