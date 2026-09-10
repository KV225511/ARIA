import json

import pytest

from modules.module_07_rl.calibration_protocol import (
    CALIBRATION_PROTOCOL_VERSION,
    canonical_json_hash,
    freeze_calibration_protocol,
    validate_calibration_protocol,
    validate_environment_fingerprint,
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


def _manifest(transitions):
    manifest = {
        "schema_version": "aria-split-manifest-v4",
        "parent_manifest_hash": "parent",
        "raw_dataset_hash": canonical_json_hash(transitions),
        "migration_algorithm_version": "train-to-validation-component-rebalance-v1",
        "locked_test_assignment_hash": "locked",
        "assignments": {item["episode_id"]: "train" for item in transitions},
    }
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    return manifest


def _freeze(tmp_path, raw_text=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    transitions = _raw()
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(raw_text or json.dumps(transitions), encoding="utf-8")
    lock = tmp_path / "requirements.lock"
    lock.write_text("numpy==1\n", encoding="utf-8")
    protocol = freeze_calibration_protocol(
        raw_file, _manifest(transitions), tmp_path / "derived", lock,
        expected_counts=(3, 3, 3), target_component_counts=(1, 1, 1),
    )
    return raw_file, protocol


def test_protocol_hash_is_deterministic_and_environment_is_valid(tmp_path):
    raw_file, first = _freeze(tmp_path / "one")
    repeated = freeze_calibration_protocol(
        raw_file, _manifest(_raw()), tmp_path / "one" / "derived-again",
        tmp_path / "one" / "requirements.lock",
        expected_counts=(3, 3, 3), target_component_counts=(1, 1, 1),
    )
    _, second = _freeze(tmp_path / "two")
    # Paths are provenance, so normalize them before comparing the hash inputs.
    assert first["raw_dataset_hash"] == second["raw_dataset_hash"]
    assert first["candidate_values"] == second["candidate_values"]
    assert first["protocol_hash"] == repeated["protocol_hash"]
    validate_calibration_protocol(first, raw_file=raw_file)
    fingerprint = validate_environment_fingerprint(
        tmp_path / "one" / "derived" / "protocol" / "environment_fingerprint_v1.json"
    )
    assert fingerprint["python_version"]
    assert fingerprint["architecture"]


def test_protocol_detects_tampering_and_unsupported_version(tmp_path):
    _, protocol = _freeze(tmp_path)
    tampered = dict(protocol)
    tampered["bootstrap_samples"] = 999
    with pytest.raises(ValueError, match="protocol hash"):
        validate_calibration_protocol(tampered)
    unsupported = dict(protocol)
    unsupported["protocol_schema_version"] = "future"
    unsigned = dict(unsupported)
    unsigned.pop("protocol_hash")
    unsupported["protocol_hash"] = canonical_json_hash(unsigned)
    with pytest.raises(ValueError, match="unsupported calibration protocol"):
        validate_calibration_protocol(unsupported)


def test_frozen_protocol_refuses_overwrite(tmp_path):
    raw_file, _ = _freeze(tmp_path)
    with pytest.raises(FileExistsError, match="already frozen"):
        freeze_calibration_protocol(
            raw_file, _manifest(_raw()), tmp_path / "derived",
            tmp_path / "requirements.lock",
            expected_counts=(3, 3, 3), target_component_counts=(1, 1, 1),
        )


def test_raw_byte_hash_and_canonical_dataset_hash_are_distinct_contracts(tmp_path):
    compact = json.dumps(_raw(), separators=(",", ":"))
    pretty = json.dumps(_raw(), indent=4)
    _, first = _freeze(tmp_path / "compact", compact)
    _, second = _freeze(tmp_path / "pretty", pretty)
    assert first["raw_file_sha256"] != second["raw_file_sha256"]
    assert first["raw_dataset_hash"] == second["raw_dataset_hash"]
    assert first["protocol_schema_version"] == CALIBRATION_PROTOCOL_VERSION
