import copy

import numpy as np
import pytest

from modules.module_07_rl.dataset_audit import audit_learned_policy_evaluation
from modules.module_07_rl.learned_policy_rollout import (
    ROLLOUT_SCHEMA_VERSION,
    evaluate_fresh_rollouts,
    rollout_episode,
)
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE
from modules.module_07_rl.state_builder import STATE_DIM


class FakePolicy:
    checkpoint_sha256 = "c" * 64

    def __init__(self):
        self.calls = 0

    def select_action(self, observation, mask, **kwargs):
        action = 0 if self.calls == 0 else 7
        self.calls += 1
        probabilities = np.zeros(8)
        probabilities[action] = 1.0
        return {"action_idx": action, "action_probabilities": probabilities.tolist(),
                "selected_action_probability": 1.0}


class FakeEnv:
    def reset(self, seed=None):
        self.turn = 0
        return np.zeros(STATE_DIM), {"seed": seed}

    def get_action_mask(self):
        return np.ones(8, dtype=int)

    def select_target_skill(self, action_idx):
        return "Python"

    def step_with_scores(self, action_idx, semantic, behavior, cognitive):
        self.turn += 1
        done = action_idx == 7
        return (np.full(STATE_DIM, self.turn, dtype=float), 1.5 if not done else 0.25,
                done, False, {"termination_reason": "policy_conclusion"} if done else {})


def _evidence(**kwargs):
    return {"semantic_score": 0.8, "behavior_score": 0.7, "cognitive_load": "low"}


def test_fresh_rollout_uses_policy_actions_and_produces_auditable_report(tmp_path):
    episode = rollout_episode(FakeEnv(), FakePolicy(), _evidence, episode_id="fresh-1", seed=42)
    assert [row["action_idx"] for row in episode["transitions"]] == [0, 7]
    assert all(row["action_selection_source"] == "learned_iql_policy" for row in episode["transitions"])
    output = tmp_path / "report.json"
    report = evaluate_fresh_rollouts(
        [episode], checkpoint_sha256="c" * 64, protocol_hash="p" * 64,
        belief_config_hash="b" * 64, output_file=output,
    )
    assert output.exists()
    assert report["num_transitions"] == 2
    assert report["episode_reward"]["mean"] == pytest.approx(1.75)
    assert report["illegal_action_count"] == 0
    assert audit_learned_policy_evaluation(report)["passes_quality_gates"]


def test_report_is_deterministic_and_rejects_nonfresh_or_mixed_policy_input():
    episode = rollout_episode(FakeEnv(), FakePolicy(), _evidence, episode_id="fresh-1")
    kwargs = dict(checkpoint_sha256="c" * 64, protocol_hash="p" * 64,
                  belief_config_hash="b" * 64)
    assert evaluate_fresh_rollouts([episode], **kwargs) == evaluate_fresh_rollouts([episode], **kwargs)
    contaminated = copy.deepcopy(episode)
    contaminated["transitions"][0]["source_split"] = "test"
    with pytest.raises(ValueError, match="rejects dataset split"):
        evaluate_fresh_rollouts([contaminated], **kwargs)
    contaminated = copy.deepcopy(episode)
    contaminated["transitions"][0]["action_selection_source"] = "behavior_policy"
    with pytest.raises(ValueError, match="not selected"):
        evaluate_fresh_rollouts([contaminated], **kwargs)


def test_illegal_policy_action_is_rejected_before_environment_step():
    class IllegalPolicy(FakePolicy):
        def select_action(self, observation, mask, **kwargs):
            probabilities = np.zeros(len(RL_ACTION_SPACE)); probabilities[7] = 1.0
            return {"action_idx": 7, "action_probabilities": probabilities.tolist(),
                    "selected_action_probability": 1.0}

    class MaskedEnv(FakeEnv):
        def get_action_mask(self):
            mask = np.ones(8, dtype=int); mask[7] = 0
            return mask

    with pytest.raises(ValueError, match="illegal action"):
        rollout_episode(MaskedEnv(), IllegalPolicy(), _evidence, episode_id="bad")


def test_audit_rejects_invalid_or_locked_test_reports():
    report = {
        "evaluation_type": "learned_policy_rollout", "fresh_rollouts": True,
        "evaluates_learned_policy": True, "uses_locked_test": True,
        "checkpoint_hash": "c" * 64, "num_episodes": 1,
        "illegal_action_count": 1, "invalid_probability_row_count": 0,
        "completion_rate": 1.0,
    }
    audit = audit_learned_policy_evaluation(report)
    assert not audit["passes_quality_gates"]
    assert any("locked test" in warning for warning in audit["warnings"])
