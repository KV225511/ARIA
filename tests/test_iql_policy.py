import numpy as np
import pytest
import torch

from modules.module_07_rl.iql_policy import IQLPolicy
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE
from modules.module_07_rl.state_builder import STATE_DIM, STATE_FEATURE_NAMES, STATE_SCHEMA_VERSION
from modules.module_07_rl.train import CHECKPOINT_SCHEMA_VERSION, IQLNetworks


def _checkpoint(path, *, feature_names=None):
    nets = IQLNetworks(STATE_DIM, len(RL_ACTION_SPACE))
    with torch.no_grad():
        for parameter in nets.parameters():
            parameter.zero_()
        nets.policy_net[-1].bias.copy_(torch.arange(len(RL_ACTION_SPACE), dtype=torch.float32))
    torch.save({
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "supported_consumer_versions": [CHECKPOINT_SCHEMA_VERSION],
        "state_schema_version": STATE_SCHEMA_VERSION,
        "state_feature_names": list(feature_names or STATE_FEATURE_NAMES),
        "model_state_dict": nets.state_dict(),
    }, path)
    return path


def test_masked_policy_distribution_and_deterministic_selection(tmp_path):
    policy = IQLPolicy.load(_checkpoint(tmp_path / "model.pth"), require_validated=False)
    observation = np.zeros(STATE_DIM, dtype=np.float32)
    mask = np.ones(len(RL_ACTION_SPACE), dtype=int)
    mask[-1] = 0
    distribution = policy.action_distribution(observation, mask)
    assert distribution[-1] == 0.0
    assert distribution.sum() == pytest.approx(1.0)
    first = policy.select_action(observation, mask)
    second = policy.select_action(observation, mask)
    assert first == second
    assert first["action_idx"] == len(RL_ACTION_SPACE) - 2


def test_policy_sampling_requires_explicit_rng(tmp_path):
    policy = IQLPolicy.load(_checkpoint(tmp_path / "model.pth"), require_validated=False)
    with pytest.raises(ValueError, match="explicit RNG"):
        policy.select_action(np.zeros(STATE_DIM), np.ones(8), deterministic=False)
    first = policy.select_action(
        np.zeros(STATE_DIM), np.ones(8), deterministic=False,
        rng=np.random.default_rng(42),
    )
    second = policy.select_action(
        np.zeros(STATE_DIM), np.ones(8), deterministic=False,
        rng=np.random.default_rng(42),
    )
    assert first == second


@pytest.mark.parametrize("observation,mask", [
    (np.zeros(STATE_DIM - 1), np.ones(8)),
    (np.full(STATE_DIM, np.nan), np.ones(8)),
    (np.zeros(STATE_DIM), np.ones(7)),
    (np.zeros(STATE_DIM), np.zeros(8)),
    (np.zeros(STATE_DIM), np.asarray([1, 1, 1, 1, 1, 1, 1, 2])),
])
def test_policy_rejects_invalid_runtime_inputs(tmp_path, observation, mask):
    policy = IQLPolicy.load(_checkpoint(tmp_path / "model.pth"), require_validated=False)
    with pytest.raises(ValueError):
        policy.select_action(observation, mask)


def test_policy_fails_closed_on_schema_and_incomplete_provenance(tmp_path):
    bad = _checkpoint(tmp_path / "bad.pth", feature_names=["wrong"] * STATE_DIM)
    with pytest.raises(ValueError, match="state features"):
        IQLPolicy.load(bad, require_validated=False)
    good = _checkpoint(tmp_path / "good.pth")
    with pytest.raises(ValueError, match="complete v7 provenance"):
        IQLPolicy.load(good)
    with pytest.raises(ValueError, match="all five"):
        IQLPolicy.load(good, require_validated=False, protocol_file="protocol.json")
