import json
from unittest.mock import patch

import torch

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.state_builder import (
    STATE_DIM,
    STATE_FEATURE_NAMES,
    STATE_SCHEMA_VERSION,
)
from modules.module_07_rl.train import train_iql_policy, validate_replayed_dataset
from modules.module_07_rl.rl_spec import ACTION_SCHEMA_VERSION
from modules.module_07_rl.reward_model import REWARD_SCHEMA_VERSION
from modules.module_07_rl.transition_schema import TRANSITION_SCHEMA_VERSION


def _transition(index, split):
    label = index % 3
    action = index % 8
    is_stop = action == 7
    state = [0.0] * STATE_DIM
    state[label] = 1.0
    return {
        "episode_id": f"{split}-{index}",
        "resume_file": f"{split}-resume-{index}.pdf",
        "jd_file": f"{split}-jd-{index}.pdf",
        "dataset_split": split,
        "true_label": label,
        "aria_label": label,
        "semantic_score": None if is_stop else (0.1, 0.5, 0.9)[label],
        "behavior_score": None if is_stop else 0.5,
        "cognitive_load": None if is_stop else "low",
        "evaluator_confidence": None if is_stop else 1.0,
        "evaluation_valid": None if is_stop else True,
        "action_idx": action,
        "obs": state,
        "next_obs": state,
        "reward": 0.1 * action,
        "done": True,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "belief_config_hash": BeliefModelConfig().config_hash,
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "transition_kind": "stop" if is_stop else "question",
        "action_mask_before": [1.0] * 8,
        "behavior_action_probs": [1.0 / 8.0] * 8,
        "behavior_action_probability": 1.0 / 8.0,
    }


def test_dataset_validation_rejects_wrong_schema():
    transition = _transition(0, "train")
    transition["state_schema_version"] = "legacy"
    try:
        validate_replayed_dataset([transition], BeliefModelConfig(), "train")
    except ValueError as error:
        assert "incompatible state schema" in str(error)
    else:
        raise AssertionError("Expected incompatible schema rejection")


def test_training_saves_versioned_best_checkpoint_without_test_input(tmp_path):
    config = BeliefModelConfig()
    config_file = tmp_path / "belief.json"
    config.save(config_file)
    train = [_transition(index, "train") for index in range(160)]
    validation = [_transition(index, "validation") for index in range(24)]
    train_file = tmp_path / "train.json"
    validation_file = tmp_path / "validation.json"
    train_file.write_text(json.dumps(train), encoding="utf-8")
    validation_file.write_text(json.dumps(validation), encoding="utf-8")
    checkpoint_file = tmp_path / "checkpoint.pth"

    result = train_iql_policy(
        train_file=train_file,
        validation_file=validation_file,
        belief_config_file=config_file,
        output_file=checkpoint_file,
        total_epochs=1,
        batch_size=64,
        seed=7,
    )
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_schema_version"] == "aria-iql-checkpoint-v3"
    assert checkpoint["state_schema_version"] == STATE_SCHEMA_VERSION
    assert checkpoint["belief_config_hash"] == config.config_hash
    assert result["evaluates_learned_policy"] is False
    assert result["epochs_completed"] == 1
    assert result["stopped_early"] is False


def test_training_stops_after_validation_patience(tmp_path):
    config = BeliefModelConfig()
    config_file = tmp_path / "belief.json"
    config.save(config_file)
    train_file = tmp_path / "train.json"
    validation_file = tmp_path / "validation.json"
    train_file.write_text(
        json.dumps([_transition(index, "train") for index in range(160)]),
        encoding="utf-8",
    )
    validation_file.write_text(
        json.dumps([_transition(index, "validation") for index in range(24)]),
        encoding="utf-8",
    )

    with patch(
        "modules.module_07_rl.train._validation_objective",
        return_value=3.0,
    ):
        result = train_iql_policy(
            train_file=train_file,
            validation_file=validation_file,
            belief_config_file=config_file,
            output_file=tmp_path / "checkpoint.pth",
            total_epochs=20,
            batch_size=160,
            seed=7,
            early_stopping_patience=2,
            early_stopping_min_delta=1e-4,
        )

    assert result["best_epoch"] == 1
    assert result["epochs_completed"] == 3
    assert result["stopped_early"] is True
