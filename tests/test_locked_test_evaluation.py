import json

import pytest

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.evaluate_locked_test import evaluate_locked_test
from modules.module_07_rl.replay_dataset import replay_dataset
from modules.module_07_rl.rl_spec import ACTION_SCHEMA_VERSION
from modules.module_07_rl.transition_schema import TRANSITION_SCHEMA_VERSION


def _raw_dataset():
    transitions = []
    for index in range(9):
        label = index % 3
        for turn in range(2):
            transitions.append({
                "episode_id": f"episode-{index}",
                "resume_file": f"resume-{index}.pdf",
                "jd_file": f"jd-{index}.pdf",
                "true_label": label,
                "aria_label": 1,
                "target_skill": "Python",
                "semantic_score": (0.1, 0.5, 0.9)[label],
                "behavior_score": 0.5,
                "cognitive_load": "low",
                "evaluator_confidence": 1.0,
                "evaluation_valid": True,
                "action_idx": 3,
                "action": [0, 0, 0, 1, 0, 0, 0, 0],
                "reward": -0.1,
                "done": turn == 1,
                "question": f"Question {turn}",
                "transition_schema_version": TRANSITION_SCHEMA_VERSION,
                "action_schema_version": ACTION_SCHEMA_VERSION,
                "action_mask_before": [1.0] * 7 + [0.0],
                "behavior_action_probs": [1.0 / 7.0] * 7 + [0.0],
                "behavior_action_probability": 1.0 / 7.0,
            })
    return transitions


def test_locked_test_requires_explicit_freeze_and_never_overwrites(tmp_path):
    config = BeliefModelConfig()
    replayed, _, _ = replay_dataset(_raw_dataset(), config)
    test = [item for item in replayed if item["dataset_split"] == "test"]
    test_file = tmp_path / "test.json"
    config_file = tmp_path / "belief.json"
    report_file = tmp_path / "locked-test-report.json"
    test_file.write_text(json.dumps(test), encoding="utf-8")
    config.save(config_file)

    with pytest.raises(ValueError, match="config-freeze confirmation"):
        evaluate_locked_test(test_file, config_file, report_file)

    report = evaluate_locked_test(
        test_file,
        config_file,
        report_file,
        confirm_config_frozen=True,
    )
    assert report["locked_test_gate"]["gate"] == "locked_test"
    assert report["evaluates_learned_policy"] is False
    assert report["test_metrics_unlocked"] is True

    with pytest.raises(FileExistsError):
        evaluate_locked_test(
            test_file,
            config_file,
            report_file,
            confirm_config_frozen=True,
        )
