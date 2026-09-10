"""Additive frozen protocol contract for ARIA calibration v5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from modules.module_07_rl.calibration_protocol import (
    ENVIRONMENT_FINGERPRINT_VERSION,
    KNOWN_CANONICAL_DATASET_HASH,
    NUMERICAL_TOLERANCES,
    REQUIRED_PROTOCOL_FIELDS,
    SUPPORTED_PROTOCOL_STATUSES,
    atomic_json_write,
    build_environment_fingerprint,
    canonical_json_hash,
    file_sha256,
    validate_environment_fingerprint,
)
from modules.module_07_rl.dataset_audit import (
    CALIBRATION_GATE_THRESHOLDS,
    VALIDATION_GATE_VERSION,
)
from modules.module_07_rl.dataset_split import (
    connected_identity_components,
    group_transitions_into_episodes,
)
from modules.module_07_rl.metrics import METRICS_SCHEMA_VERSION


CALIBRATION_PROTOCOL_VERSION = "aria-calibration-protocol-v5"
CALIBRATION_ALGORITHM_VERSION = "aria-belief-calibration-v5"
DEVELOPMENT_BUNDLE_VERSION = "aria-development-bundle-v2"
CALIBRATION_CV_REPORT_VERSION = "aria-calibration-cv-report-v2"

CALIBRATION_STAGE_SEQUENCE = [
    "v4_best_failing_anchor",
    "lower_repeat_discount_power",
]
CALIBRATION_CANDIDATE_VALUES = {
    "anchor": {
        "scale_shrinkage": [1.00],
        "aggregation_temperature": [2.00],
        "minimum_assessment_confidence": [0.45],
        "repeat_discount_power": [0.25],
        "max_skill_effective_sample_size": [3],
    },
    "lower_repeat_discount_power": [0.00, 0.10, 0.20],
}
MAX_CALIBRATION_CANDIDATES = 4


def _load_json(value: dict | str | Path) -> dict:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def freeze_calibration_protocol_v5(
    raw_file: str | Path,
    split_manifest: dict | str | Path,
    output_dir: str | Path,
    dependency_lock_path: str | Path,
    *,
    target_component_counts: tuple[int, int, int] = (21, 6, 6),
    bootstrap_samples: int = 1000,
    repository: str | Path | None = None,
    expected_counts: tuple[int, int, int] | None = None,
    parent_v4_report_hash: str | None = None,
) -> dict:
    """Freeze v5 without mutating or replacing any v4 protocol artifact."""
    base = Path(output_dir) / "protocol"
    protocol_path = base / "calibration_protocol_v5.json"
    if protocol_path.exists():
        raise FileExistsError(f"calibration protocol v5 is already frozen: {protocol_path}")
    raw_path = Path(raw_file).resolve()
    raw_bytes = raw_path.read_bytes()
    transitions = json.loads(raw_bytes.decode("utf-8"))
    manifest = _load_json(split_manifest)
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if not manifest_hash or manifest_hash != canonical_json_hash(unsigned_manifest):
        raise ValueError("split manifest hash is invalid")
    episodes = group_transitions_into_episodes(transitions)
    components = connected_identity_components(transitions)
    actual_counts = (len(episodes), len(transitions), len(components))
    required_counts = expected_counts or (600, 9018, 33)
    if actual_counts != tuple(required_counts):
        raise ValueError(
            f"raw corpus counts {actual_counts} do not match expected {tuple(required_counts)}"
        )
    raw_dataset_hash = canonical_json_hash(transitions)
    if expected_counts is None and raw_dataset_hash != KNOWN_CANONICAL_DATASET_HASH:
        raise ValueError("raw corpus canonical JSON hash does not match the production contract")
    if manifest.get("raw_dataset_hash") != raw_dataset_hash:
        raise ValueError("split manifest raw dataset hash does not match raw corpus")
    fingerprint = build_environment_fingerprint(
        dependency_lock_path, repository=repository,
    )
    protocol = {
        "protocol_schema_version": CALIBRATION_PROTOCOL_VERSION,
        "protocol_status": "FROZEN_FOR_DEVELOPMENT",
        "raw_file_path": str(raw_path),
        "raw_file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_dataset_hash": raw_dataset_hash,
        "episode_count": len(episodes),
        "transition_count": len(transitions),
        "identity_component_count": len(components),
        "parent_split_manifest_hash": manifest.get("parent_manifest_hash", manifest_hash),
        "locked_test_assignment_hash": manifest.get("locked_test_assignment_hash"),
        "target_component_counts": list(target_component_counts),
        "metric_schema_version": METRICS_SCHEMA_VERSION,
        "gate_policy_version": VALIDATION_GATE_VERSION,
        "gate_thresholds": dict(CALIBRATION_GATE_THRESHOLDS),
        "split_migration_algorithm": manifest.get(
            "migration_algorithm_version", "train-to-validation-component-rebalance-v1",
        ),
        "cross_validation_algorithm": "grouped-identity-component-3fold-greedy-v1",
        "calibration_algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "calibration_stage_sequence": list(CALIBRATION_STAGE_SEQUENCE),
        "candidate_values": CALIBRATION_CANDIDATE_VALUES,
        "candidate_selection_rule": [
            "highest_macro_f1", "lowest_ordinal_mae", "lowest_ece",
            "lowest_abstention_rate", "lexicographically_smallest_parameters",
        ],
        "bootstrap_method": "paired-identity-component-percentile-v1",
        "bootstrap_samples": int(bootstrap_samples),
        "all_random_seeds": {"split": 42, "cross_validation": 42, "bootstrap": 42},
        "numerical_tolerances": dict(NUMERICAL_TOLERANCES),
        "code_commit": fingerprint["code_commit"],
        "git_dirty": fingerprint["git_dirty"],
        "dependency_lock_path": fingerprint["dependency_lock_path"],
        "dependency_lock_hash": fingerprint["dependency_lock_hash"],
        "environment_fingerprint_hash": fingerprint["environment_fingerprint_hash"],
        "maximum_validation_executions": 1,
        "validation_executions": 0,
        "locked_test_policy": "application-level-one-attempt-guard-v1",
        "maximum_calibration_candidates": MAX_CALIBRATION_CANDIDATES,
        "parent_v4_report_hash": parent_v4_report_hash,
        "v5_change_scope": "lower-repeat-discount-only",
    }
    protocol["protocol_hash"] = canonical_json_hash(protocol)
    atomic_json_write(base / "environment_fingerprint_v1.json", fingerprint)
    atomic_json_write(protocol_path, protocol)
    return protocol


def validate_calibration_protocol_v5(
    value: dict | str | Path,
    *,
    raw_file: str | Path | None = None,
    split_manifest: dict | str | Path | None = None,
    environment_fingerprint: dict | str | Path | None = None,
) -> dict:
    protocol = _load_json(value)
    missing = REQUIRED_PROTOCOL_FIELDS - protocol.keys()
    if missing:
        raise ValueError(f"calibration protocol is missing {sorted(missing)}")
    if protocol.get("protocol_schema_version") != CALIBRATION_PROTOCOL_VERSION:
        raise ValueError("unsupported calibration protocol version")
    if protocol.get("protocol_status") not in SUPPORTED_PROTOCOL_STATUSES:
        raise ValueError("unsupported calibration protocol status")
    if (
        protocol["protocol_status"] in {"PROVISIONAL_SYNTHETIC", "VALIDATED_SYNTHETIC"}
        and not protocol.get("belief_config_hash")
    ):
        raise ValueError("release-ready protocol is missing belief config hash")
    unsigned = dict(protocol)
    stored_hash = unsigned.pop("protocol_hash", None)
    if not stored_hash or stored_hash != canonical_json_hash(unsigned):
        raise ValueError("calibration protocol hash is invalid")
    fixed = {
        "calibration_algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "metric_schema_version": METRICS_SCHEMA_VERSION,
        "gate_policy_version": VALIDATION_GATE_VERSION,
        "gate_thresholds": CALIBRATION_GATE_THRESHOLDS,
        "calibration_stage_sequence": CALIBRATION_STAGE_SEQUENCE,
        "candidate_values": CALIBRATION_CANDIDATE_VALUES,
        "maximum_validation_executions": 1,
        "maximum_calibration_candidates": MAX_CALIBRATION_CANDIDATES,
        "v5_change_scope": "lower-repeat-discount-only",
    }
    for field, expected in fixed.items():
        if protocol.get(field) != expected:
            raise ValueError(f"calibration protocol {field} changed")
    if raw_file is not None:
        raw_bytes = Path(raw_file).read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != protocol["raw_file_sha256"]:
            raise ValueError("raw file byte hash changed")
        if canonical_json_hash(json.loads(raw_bytes.decode("utf-8"))) != protocol["raw_dataset_hash"]:
            raise ValueError("raw canonical dataset hash changed")
    if split_manifest is not None:
        manifest = _load_json(split_manifest)
        unsigned_manifest = dict(manifest)
        manifest_hash = unsigned_manifest.pop("manifest_hash", None)
        if not manifest_hash or manifest_hash != canonical_json_hash(unsigned_manifest):
            raise ValueError("split manifest hash is invalid")
        if manifest.get("locked_test_assignment_hash") != protocol["locked_test_assignment_hash"]:
            raise ValueError("locked-test assignment hash changed")
        if manifest.get("raw_dataset_hash") != protocol["raw_dataset_hash"]:
            raise ValueError("split manifest raw dataset hash changed")
    if environment_fingerprint is not None:
        fingerprint = validate_environment_fingerprint(environment_fingerprint)
        if fingerprint["environment_fingerprint_hash"] != protocol["environment_fingerprint_hash"]:
            raise ValueError("environment fingerprint hash changed")
    return protocol


def update_protocol_status_v5(
    protocol_file: str | Path,
    status: str,
    **additional_fields: Any,
) -> dict:
    if status not in SUPPORTED_PROTOCOL_STATUSES:
        raise ValueError("unsupported calibration protocol status")
    protocol = validate_calibration_protocol_v5(protocol_file)
    protocol.update(additional_fields)
    protocol["protocol_status"] = status
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = canonical_json_hash(protocol)
    atomic_json_write(protocol_file, protocol)
    return protocol


def validate_development_bundle_v5(
    value: dict | str | Path,
    *,
    protocol: dict | str | Path,
    belief_config_hash: str,
    split_manifest: dict | str | Path,
    train_file: str | Path | None = None,
    validation_file: str | Path | None = None,
) -> dict:
    bundle = _load_json(value)
    if bundle.get("schema_version") != DEVELOPMENT_BUNDLE_VERSION:
        raise ValueError("unsupported development bundle schema version")
    if DEVELOPMENT_BUNDLE_VERSION not in bundle.get("supported_consumer_versions", []):
        raise ValueError("development bundle is not compatible with this consumer")
    unsigned = dict(bundle)
    stored_hash = unsigned.pop("bundle_hash", None)
    if not stored_hash or stored_hash != canonical_json_hash(unsigned):
        raise ValueError("development bundle hash is invalid")
    manifest = _load_json(split_manifest)
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if not manifest_hash or manifest_hash != canonical_json_hash(unsigned_manifest):
        raise ValueError("split manifest hash is invalid")
    checked_protocol = validate_calibration_protocol_v5(protocol, split_manifest=manifest)
    expected = {
        "protocol_hash": checked_protocol["protocol_hash"],
        "raw_file_sha256": checked_protocol["raw_file_sha256"],
        "raw_dataset_hash": checked_protocol["raw_dataset_hash"],
        "split_manifest_hash": manifest_hash,
        "belief_config_hash": belief_config_hash,
        "environment_fingerprint_hash": checked_protocol["environment_fingerprint_hash"],
    }
    for field, expected_value in expected.items():
        if bundle.get(field) != expected_value:
            raise ValueError(f"development bundle {field} mismatch")
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict) or "replayed_test" in artifacts:
        raise ValueError("invalid development bundle artifact inventory")
    for name, path in (("replayed_train", train_file), ("replayed_validation", validation_file)):
        if path is None:
            continue
        record = artifacts.get(name, {})
        if record.get("sha256") != file_sha256(path):
            raise ValueError(f"development bundle {name} hash mismatch")
        expected_split = "train" if name.endswith("train") else "validation"
        if record.get("split") != expected_split:
            raise ValueError(f"development bundle {name} split mismatch")
    return bundle
