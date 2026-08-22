import json

import pytest

from modules.module_06_belief.belief_config import BeliefModelConfig


def test_config_round_trip_and_hash_are_deterministic(tmp_path):
    config = BeliefModelConfig(
        class_centers=(0.25, 0.45, 0.75),
        fit_metadata={"source": "train"},
    )
    path = tmp_path / "belief.json"
    config.save(path)
    loaded = BeliefModelConfig.load(path)
    assert loaded == config
    assert loaded.config_hash == config.config_hash
    assert json.loads(config.canonical_json())["schema_version"] == "belief-v2"


def test_config_rejects_unordered_centers():
    with pytest.raises(ValueError):
        BeliefModelConfig(class_centers=(0.5, 0.4, 0.8))
