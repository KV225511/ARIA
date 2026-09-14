import json

import pytest

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.calibration_protocol import canonical_json_hash
from modules.module_07_rl.calibration_protocol_v7 import (
    CALIBRATION_PROTOCOL_VERSION, LOW_CLASS_LOGIT_BIAS_VALUES,
    MAX_CALIBRATION_CANDIDATES, freeze_calibration_protocol_v7,
    update_protocol_state_v7, validate_calibration_protocol_v7,
    validate_protocol_state_v7,
)


def _lineage():
    config = BeliefModelConfig(minimum_effective_evidence=1.5)
    return {
        "protocol": {
            "protocol_hash": "v6-protocol", "raw_file_sha256": "raw-bytes",
            "raw_dataset_hash": "raw-canonical", "episode_count": 33,
            "transition_count": 99, "identity_component_count": 33,
            "target_component_counts": [21, 6, 6],
            "split_assignment_hash": "assignment", "locked_test_assignment_hash": "locked",
        },
        "state": {"state_hash": "v6-state"},
        "report_hash": "v6-report", "inventory_hash": "v6-inventory",
        "manifest_hash": "manifest", "config": config,
    }


def test_v7_freeze_hash_is_deterministic_and_state_is_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modules.module_07_rl.calibration_protocol_v7.validate_parent_v6_lineage",
        lambda *args, **kwargs: _lineage(),
    )
    lock = tmp_path / "requirements.lock"
    lock.write_text("numpy==fixture\n", encoding="utf-8")
    output = tmp_path / "v7"
    protocol = freeze_calibration_protocol_v7(
        {}, {}, {}, {}, {}, {}, output, lock, repository=tmp_path,
    )
    state_path = output / "protocol" / "calibration_protocol_state_v2.json"
    state = validate_protocol_state_v7(state_path, protocol=protocol)
    assert protocol["protocol_schema_version"] == CALIBRATION_PROTOCOL_VERSION
    assert protocol["candidate_values"]["low_class_logit_bias"] == LOW_CLASS_LOGIT_BIAS_VALUES
    assert protocol["maximum_calibration_candidates"] == MAX_CALIBRATION_CANDIDATES == 6
    assert protocol["maximum_validation_executions"] == 0
    assert protocol["protocol_hash"] == canonical_json_hash({
        key: value for key, value in protocol.items() if key != "protocol_hash"
    })
    frozen_hash = protocol["protocol_hash"]
    updated = update_protocol_state_v7(
        state_path, "PROVISIONAL_TRAINING_CV", protocol=protocol,
        preparation_executions=1, belief_config_hash="v7-config",
        last_attempt_hash="attempt",
    )
    assert updated["protocol_hash"] == frozen_hash
    assert updated["current_status"] == "PROVISIONAL_TRAINING_CV"
    assert (output / "release").is_dir()


def test_v7_protocol_rejects_tampering_and_unknown_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modules.module_07_rl.calibration_protocol_v7.validate_parent_v6_lineage",
        lambda *args, **kwargs: _lineage(),
    )
    lock = tmp_path / "requirements.lock"
    lock.write_text("fixture\n", encoding="utf-8")
    protocol = freeze_calibration_protocol_v7(
        {}, {}, {}, {}, {}, {}, tmp_path / "v7", lock, repository=tmp_path,
    )
    tampered = dict(protocol)
    tampered["maximum_calibration_candidates"] = 7
    with pytest.raises(ValueError, match="hash is invalid"):
        validate_calibration_protocol_v7(tampered)
    rehashed = dict(tampered)
    rehashed.pop("protocol_hash")
    rehashed["protocol_hash"] = canonical_json_hash(rehashed)
    with pytest.raises(ValueError, match="maximum_calibration_candidates changed"):
        validate_calibration_protocol_v7(rehashed)
    unknown = dict(protocol)
    unknown["surprise"] = True
    unknown.pop("protocol_hash")
    unknown["protocol_hash"] = canonical_json_hash(unknown)
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_calibration_protocol_v7(unknown)


def test_v7_state_refuses_release_without_completed_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "modules.module_07_rl.calibration_protocol_v7.validate_parent_v6_lineage",
        lambda *args, **kwargs: _lineage(),
    )
    lock = tmp_path / "requirements.lock"
    lock.write_text("fixture\n", encoding="utf-8")
    output = tmp_path / "v7"
    protocol = freeze_calibration_protocol_v7(
        {}, {}, {}, {}, {}, {}, output, lock, repository=tmp_path,
    )
    state_path = output / "protocol" / "calibration_protocol_state_v2.json"
    with pytest.raises(ValueError, match="invalid protocol state transition"):
        update_protocol_state_v7(
            state_path, "VALIDATED_SYNTHETIC", protocol=protocol,
        )
