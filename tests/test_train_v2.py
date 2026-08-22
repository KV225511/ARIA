import json

import torch

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.state_builder import (
    STATE_DIM,
    STATE_FEATURE_NAMES,
    STATE_SCHEMA_VERSION,
)
from modules.module_07_rl.train import train_iql_policy, validate_replayed_dataset


def _transition(index, split):
    label = index % 3
    action = index % 8
    state = [0.0] * STATE_DIM
    state[label] = 1.0
    return {
        "episode_id": f"{split}-{index}",
        "resume_file": f"{split}-resume-{index}.pdf",
        "jd_file": f"{split}-jd-{index}.pdf",
        "dataset_split": split,
        "true_label": label,
        "aria_label": label,
        "semantic_score": (0.1, 0.5, 0.9)[label],
        "behavior_score": 0.5,
        "cognitive_load": "low",
        "evaluator_confidence": 1.0,
        "evaluation_valid": True,
        "action_idx": action,
        "obs": state,
        "next_obs": state,
        "reward": 0.1 * action,
        "done": True,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "belief_config_hash": BeliefModelConfig().config_hash,
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
    assert checkpoint["checkpoint_schema_version"] == "aria-iql-checkpoint-v2"
    assert checkpoint["state_schema_version"] == STATE_SCHEMA_VERSION
    assert checkpoint["belief_config_hash"] == config.config_hash
    assert result["evaluates_learned_policy"] is False
