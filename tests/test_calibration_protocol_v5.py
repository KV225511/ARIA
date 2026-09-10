import json

import pytest

from modules.module_07_rl.calibration_protocol import canonical_json_hash
from modules.module_07_rl.calibration_protocol_v5 import (
    CALIBRATION_CANDIDATE_VALUES,
    CALIBRATION_PROTOCOL_VERSION,
    MAX_CALIBRATION_CANDIDATES,
    freeze_calibration_protocol_v5,
    validate_calibration_protocol_v5,
)


def _raw():
    return [
        {
            "episode_id": f"episode-{index}",
            "resume_content_hash": f"resume-{index}",
            "jd_content_hash": f"jd-{index}",
            "true_label": index % 3,
            "done": True,
        }
        for index in range(3)
    ]


def _manifest(rows):
    value = {
        "schema_version": "aria-split-manifest-v4",
        "parent_manifest_hash": "parent",
        "raw_dataset_hash": canonical_json_hash(rows),
        "migration_algorithm_version": "train-to-validation-component-rebalance-v1",
        "locked_test_assignment_hash": "locked",
        "assignments": {row["episode_id"]: "train" for row in rows},
    }
    value["manifest_hash"] = canonical_json_hash(value)
    return value


def _freeze(tmp_path, output_name="derived"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = _raw()
    raw = tmp_path / "raw.json"
    lock = tmp_path / "requirements.lock"
    raw.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lock.write_text("numpy==fixture\n", encoding="utf-8")
    protocol = freeze_calibration_protocol_v5(
        raw,
        _manifest(rows),
        tmp_path / output_name,
        lock,
        expected_counts=(3, 3, 3),
        target_component_counts=(1, 1, 1),
    )
    return raw, lock, protocol


def test_v5_protocol_is_deterministic_and_search_is_narrow(tmp_path):
    raw, lock, first = _freeze(tmp_path, "one")
    second = freeze_calibration_protocol_v5(
        raw,
        _manifest(_raw()),
        tmp_path / "two",
        lock,
        expected_counts=(3, 3, 3),
        target_component_counts=(1, 1, 1),
    )
    assert first["protocol_hash"] == second["protocol_hash"]
    assert first["protocol_schema_version"] == CALIBRATION_PROTOCOL_VERSION
    assert first["candidate_values"] == CALIBRATION_CANDIDATE_VALUES
    assert first["maximum_calibration_candidates"] == MAX_CALIBRATION_CANDIDATES == 4
    assert first["v5_change_scope"] == "lower-repeat-discount-only"
    validate_calibration_protocol_v5(first, raw_file=raw)


def test_v5_protocol_tampering_and_overwrite_fail_closed(tmp_path):
    raw, lock, protocol = _freeze(tmp_path)
    tampered = dict(protocol)
    tampered["candidate_values"] = {"extra": [999]}
    unsigned = dict(tampered)
    unsigned.pop("protocol_hash")
    tampered["protocol_hash"] = canonical_json_hash(unsigned)
    with pytest.raises(ValueError, match="candidate_values changed"):
        validate_calibration_protocol_v5(tampered)
    with pytest.raises(FileExistsError, match="already frozen"):
        freeze_calibration_protocol_v5(
            raw,
            _manifest(_raw()),
            tmp_path / "derived",
            lock,
            expected_counts=(3, 3, 3),
            target_component_counts=(1, 1, 1),
        )
