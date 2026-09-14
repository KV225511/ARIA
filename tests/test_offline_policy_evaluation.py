import json

import pytest
import torch

from modules.module_07_rl.calibration_protocol import canonical_json_hash
from modules.module_07_rl.offline_policy_evaluation import evaluate_logged_policy
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE
from modules.module_07_rl.state_builder import STATE_DIM, STATE_SCHEMA_VERSION
from modules.module_07_rl.train import IQLNetworks


def _fixture(tmp_path, split="train"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "checkpoint.pth"
    nets = IQLNetworks(STATE_DIM, len(RL_ACTION_SPACE))
    torch.save({
        "state_schema_version": STATE_SCHEMA_VERSION,
        "model_state_dict": nets.state_dict(),
    }, checkpoint)
    rows = []
    for index in range(4):
        mask = [1.0] * len(RL_ACTION_SPACE)
        probabilities = [1.0 / len(RL_ACTION_SPACE)] * len(RL_ACTION_SPACE)
        rows.append({
            "dataset_split": split, "obs": [0.0] * STATE_DIM,
            "action_mask_before": mask, "action_idx": index % len(RL_ACTION_SPACE),
            "behavior_action_probability": probabilities[index % len(RL_ACTION_SPACE)],
            "reward": float(index),
        })
    replayed = tmp_path / "replayed.json"
    replayed.write_text(json.dumps(rows), encoding="utf-8")
    return checkpoint, replayed


def test_ope_is_diagnostic_deterministic_and_hashed(tmp_path):
    checkpoint, replayed = _fixture(tmp_path)
    first = evaluate_logged_policy(checkpoint, replayed)
    second = evaluate_logged_policy(checkpoint, replayed)
    assert first == second
    assert first["release_gate"] is False
    assert first["evaluates_learned_policy"] is True
    assert first["report_hash"] == canonical_json_hash({
        key: value for key, value in first.items() if key != "report_hash"
    })


def test_ope_rejects_test_data_and_partial_provenance(tmp_path):
    checkpoint, replayed = _fixture(tmp_path, split="test")
    with pytest.raises(ValueError, match="non-test"):
        evaluate_logged_policy(checkpoint, replayed)
    _, train = _fixture(tmp_path / "train")
    with pytest.raises(ValueError, match="requires protocol"):
        evaluate_logged_policy(checkpoint, train, protocol_file="protocol.json")
