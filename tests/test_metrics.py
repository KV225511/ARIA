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
    assert report["belief_verdict_metrics"]["micro_f1"] == 1.0
