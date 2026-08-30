from copy import deepcopy
import json
import pytest

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.replay_dataset import replay_dataset
from modules.module_07_rl.rl_spec import ACTION_SCHEMA_VERSION
from modules.module_07_rl.transition_schema import TRANSITION_SCHEMA_VERSION
from modules.module_07_rl.state_builder import (
    STATE_DIM,
    STATE_FEATURE_NAMES,
    STATE_SCHEMA_VERSION,
)


def _raw_episode(index, label, score):
    return [
        {
            "episode_id": f"episode-{index}",
            "resume_file": f"resume-{index}.pdf",
            "jd_file": f"jd-{index}.pdf",
            "true_label": label,
            "aria_label": 1,
            "target_skill": "Python",
            "semantic_score": score,
            "behavior_score": 1.0 - score,
            "cognitive_load": "low",
            "evaluator_confidence": 1.0,
            "evaluation_valid": True,
            "action_idx": 3,
            "action": [0, 0, 0, 1, 0, 0, 0, 0],
            "reward": -0.1,
            "obs": [1 / 3] * 6 + [0.0] * 144 + [1.0, 0.0],
            "next_obs": [1 / 3] * 6 + [0.0] * 144 + [1.0, 0.1],
            "done": turn == 1,
            "question": f"Question {turn}",
            "transition_schema_version": TRANSITION_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "action_mask_before": [1.0] * 7 + [0.0],
            "behavior_action_probs": [1.0 / 7.0] * 7 + [0.0],
            "behavior_action_probability": 1.0 / 7.0,
        }
        for turn in range(2)
    ]


def _raw_dataset():
    transitions = []
    for index in range(9):
        label = index % 3
        transitions.extend(_raw_episode(index, label, (0.1, 0.5, 0.9)[label]))
    return transitions


def test_replay_is_deterministic_and_does_not_mutate_raw_data():
    raw = _raw_dataset()
    snapshot = json.dumps(raw, sort_keys=True)
    first, manifest, _ = replay_dataset(raw, BeliefModelConfig())
    second, _, _ = replay_dataset(raw, BeliefModelConfig(), manifest)
    assert first == second
    assert json.dumps(raw, sort_keys=True) == snapshot


def test_replay_rejects_a_tampered_locked_split_manifest():
    raw = _raw_dataset()
    _, manifest, _ = replay_dataset(raw, BeliefModelConfig())
    tampered = deepcopy(manifest)
    episode_id = next(iter(tampered["assignments"]))
    tampered["assignments"][episode_id] = "test"
    with pytest.raises(ValueError, match="manifest_hash"):
        replay_dataset(raw, BeliefModelConfig(), tampered)


def test_replay_builds_versioned_fixed_states_and_preserves_raw_fields():
    replayed, _, report = replay_dataset(_raw_dataset(), BeliefModelConfig())
    assert replayed
    assert all(len(item["obs"]) == STATE_DIM for item in replayed)
    assert all(item["state_schema_version"] == STATE_SCHEMA_VERSION for item in replayed)
    assert all(item["state_feature_names"] == list(STATE_FEATURE_NAMES) for item in replayed)
    assert all("raw_obs" in item and "raw_reward" in item for item in replayed)
    assert report["derived_transitions"] == len(replayed)


def test_first_state_does_not_include_future_evidence():
    raw = _raw_dataset()
    changed = deepcopy(raw)
    changed[1]["semantic_score"] = 0.0
    first, _, _ = replay_dataset(raw, BeliefModelConfig())
    second, _, _ = replay_dataset(changed, BeliefModelConfig())
    assert first[0]["obs"] == second[0]["obs"]


def test_stable_state_feature_names_match_vector_positions():
    replayed, _, _ = replay_dataset(_raw_dataset(), BeliefModelConfig())
    first = dict(zip(STATE_FEATURE_NAMES, replayed[0]["obs"]))
    second = dict(zip(STATE_FEATURE_NAMES, replayed[0]["next_obs"]))
    assert first["semantic_available"] == 0.0
    assert second["semantic_available"] == 1.0
    assert abs(second["previous_semantic_score"] - 0.1) < 1e-6
    assert "action_mask" not in STATE_FEATURE_NAMES
