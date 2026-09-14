import json

import pytest

from modules.module_07_rl.calibration_protocol import canonical_json_hash
from modules.module_07_rl.calibration_protocol_v5 import (
    freeze_calibration_protocol_v5,
    update_protocol_status_v5,
)
from modules.module_07_rl.calibration_protocol_v6 import (
    CALIBRATION_PROTOCOL_VERSION,
    MAX_CALIBRATION_CANDIDATES,
    freeze_calibration_protocol_v6,
    update_protocol_state,
    validate_calibration_protocol_v6,
    validate_parent_v5_lineage,
    validate_protocol_state,
)


def _rows():
    return [{
        "episode_id": f"episode-{index}",
        "resume_content_hash": f"resume-{index}",
        "jd_content_hash": f"jd-{index}",
        "true_label": index % 3,
        "done": True,
    } for index in range(3)]


def _parent_artifacts(tmp_path):
    rows = _rows()
    raw = tmp_path / "raw.json"
    lock = tmp_path / "requirements.lock"
    raw.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lock.write_text("numpy==fixture\n", encoding="utf-8")
    manifest = {
        "schema_version": "aria-split-manifest-v4",
        "parent_manifest_hash": "parent-v3",
        "raw_dataset_hash": canonical_json_hash(rows),
        "seed": 42,
        "migration_algorithm_version": "train-to-validation-component-rebalance-v1",
        "assignments": {row["episode_id"]: "train" for row in rows},
        "summary": {},
        "locked_test_assignment_hash": "locked",
    }
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    parent_dir = tmp_path / "v5"
    protocol = freeze_calibration_protocol_v5(
        raw, manifest, parent_dir, lock,
        expected_counts=(3, 3, 3), target_component_counts=(1, 1, 1),
    )
    manifest.update({
        "producer_version": "aria-belief-calibration-v5",
        "supported_consumer_versions": ["aria-split-manifest-v4"],
        "protocol_hash": protocol["protocol_hash"],
    })
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    report = {
        "schema_version": "aria-calibration-cv-report-v2",
        "protocol_hash": protocol["protocol_hash"],
        "selection_status": "FAILED",
        "candidate_count": 4,
        "candidate_limit": 4,
        "validation_used_for_selection": False,
        "attempted_candidates": [],
        "best_failing_candidate": {},
    }
    report["report_hash"] = canonical_json_hash(report)
    inventory = {
        "schema_version": "aria-raw-split-inventory-v2",
        "protocol_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"],
        "raw_dataset_hash": protocol["raw_dataset_hash"],
        "split_manifest_hash": manifest["manifest_hash"],
        "artifacts": {},
    }
    inventory["inventory_hash"] = canonical_json_hash(inventory)
    protocol_path = parent_dir / "protocol" / "calibration_protocol_v5.json"
    update_protocol_status_v5(protocol_path, "FAILED")
    return raw, lock, protocol_path, report, manifest, inventory


def test_v6_protocol_hash_is_stable_across_state_changes(tmp_path):
    raw, lock, parent_protocol, report, manifest, inventory = _parent_artifacts(tmp_path)
    lineage = validate_parent_v5_lineage(parent_protocol, report, manifest, inventory)
    protocol = freeze_calibration_protocol_v6(
        raw, manifest, tmp_path / "v6", lock,
        lineage=lineage, expected_counts=(3, 3, 3),
    )
    state_path = tmp_path / "v6" / "protocol" / "calibration_protocol_state_v1.json"
    before = protocol["protocol_hash"]
    updated = update_protocol_state(
        state_path, "FAILED", protocol=protocol, failure_reason="fixture",
    )
    assert protocol["protocol_schema_version"] == CALIBRATION_PROTOCOL_VERSION
    assert protocol["maximum_calibration_candidates"] == MAX_CALIBRATION_CANDIDATES == 3
    assert validate_calibration_protocol_v6(protocol)["protocol_hash"] == before
    assert updated["protocol_hash"] == before
    assert updated["current_status"] == "FAILED"
    with pytest.raises(ValueError, match="invalid protocol state transition"):
        update_protocol_state(state_path, "PROVISIONAL_SYNTHETIC", protocol=protocol)


def test_v6_rejects_parent_lineage_and_state_tampering(tmp_path):
    raw, lock, parent_protocol, report, manifest, inventory = _parent_artifacts(tmp_path)
    tampered_report = dict(report)
    tampered_report["candidate_count"] = 3
    with pytest.raises(ValueError, match="report hash"):
        validate_parent_v5_lineage(parent_protocol, tampered_report, manifest, inventory)
    lineage = validate_parent_v5_lineage(parent_protocol, report, manifest, inventory)
    protocol = freeze_calibration_protocol_v6(
        raw, manifest, tmp_path / "v6", lock,
        lineage=lineage, expected_counts=(3, 3, 3),
    )
    state = json.loads(
        (tmp_path / "v6" / "protocol" / "calibration_protocol_state_v1.json").read_text()
    )
    state["revision"] = 99
    with pytest.raises(ValueError, match="state hash"):
        validate_protocol_state(state, protocol=protocol)


def test_v6_rejects_missing_or_rehashed_frozen_search_fields(tmp_path):
    raw, lock, parent_protocol, report, manifest, inventory = _parent_artifacts(tmp_path)
    lineage = validate_parent_v5_lineage(parent_protocol, report, manifest, inventory)
    protocol = freeze_calibration_protocol_v6(
        raw, manifest, tmp_path / "v6", lock,
        lineage=lineage, expected_counts=(3, 3, 3),
    )
    missing = dict(protocol)
    missing.pop("parent_v5_report_hash")
    missing.pop("protocol_hash")
    missing["protocol_hash"] = canonical_json_hash(missing)
    with pytest.raises(ValueError, match="missing"):
        validate_calibration_protocol_v6(missing)

    changed = dict(protocol)
    changed["all_random_seeds"] = {"cross_validation": 7, "bootstrap": 42}
    changed.pop("protocol_hash")
    changed["protocol_hash"] = canonical_json_hash(changed)
    with pytest.raises(ValueError, match="all_random_seeds changed"):
        validate_calibration_protocol_v6(changed)
