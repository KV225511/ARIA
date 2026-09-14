"""Stable protocol, lineage, state, and bundle contracts for calibration v6."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.calibration_protocol import (
    KNOWN_CANONICAL_DATASET_HASH,
    NUMERICAL_TOLERANCES,
    atomic_json_write,
    build_environment_fingerprint,
    canonical_json_hash,
    file_sha256,
)
from modules.module_07_rl.calibration_protocol_v5 import (
    CALIBRATION_PROTOCOL_VERSION as PARENT_PROTOCOL_VERSION,
    validate_calibration_protocol_v5,
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


CALIBRATION_PROTOCOL_VERSION = "aria-calibration-protocol-v6"
CALIBRATION_ALGORITHM_VERSION = "aria-belief-calibration-v6"
CALIBRATION_CV_REPORT_VERSION = "aria-calibration-cv-report-v3"
DEVELOPMENT_BUNDLE_VERSION = "aria-development-bundle-v3"
DEVELOPMENT_REPORT_VERSION = "aria-development-calibration-v6"
RAW_SPLIT_INVENTORY_VERSION = "aria-raw-split-inventory-v3"
PROTOCOL_STATE_VERSION = "aria-calibration-protocol-state-v1"
VALIDATION_ATTEMPT_VERSION = "aria-validation-attempt-v1"

CALIBRATION_STAGE_SEQUENCE = [
    "v5_best_failing_anchor",
    "lower_minimum_effective_evidence",
]
CALIBRATION_CANDIDATE_VALUES = {
    "anchor": {
        "scale_shrinkage": [1.0],
        "aggregation_temperature": [2.0],
        "minimum_assessment_confidence": [0.45],
        "repeat_discount_power": [0.25],
        "max_skill_effective_sample_size": [3],
        "minimum_skill_coverage": [3],
        "minimum_effective_evidence": [2.0],
    },
    "lower_minimum_effective_evidence": [1.75, 1.5],
}
MAX_CALIBRATION_CANDIDATES = 3
V6_CHANGE_SCOPE = "lower-minimum-effective-evidence-only"
FIXED_BEHAVIOR_CONFIGURATION = {
    "duplicate_question_multiplier": 0.25,
    "posterior_floor": 1e-4,
    "minimum_skill_coverage": 3,
    "minimum_effective_evidence": 2.0,
}
ALLOWED_STATE_TRANSITIONS = {
    "FROZEN_FOR_DEVELOPMENT": {"FAILED", "PROVISIONAL_SYNTHETIC"},
    "PROVISIONAL_SYNTHETIC": {"FAILED", "VALIDATED_SYNTHETIC"},
    "VALIDATED_SYNTHETIC": set(),
    "FAILED": set(),
}

REQUIRED_PROTOCOL_FIELDS = {
    "protocol_schema_version", "initial_protocol_status", "artifact_root",
    "raw_file_path", "raw_file_sha256", "raw_dataset_hash", "episode_count",
    "transition_count", "identity_component_count", "target_component_counts",
    "split_assignment_hash", "locked_test_assignment_hash",
    "metric_schema_version", "gate_policy_version", "gate_thresholds",
    "cross_validation_algorithm", "calibration_algorithm_version",
    "calibration_stage_sequence", "candidate_values", "candidate_selection_rule",
    "fixed_behavior_configuration", "emission_fit_algorithm", "bootstrap_method",
    "bootstrap_samples", "all_random_seeds", "numerical_tolerances",
    "maximum_calibration_candidates", "maximum_validation_executions",
    "locked_test_policy", "v6_change_scope", "parent_v5_protocol_final_hash",
    "parent_v5_protocol_frozen_hash", "parent_v5_report_hash",
    "parent_v5_split_manifest_hash", "parent_v5_inventory_hash", "code_commit",
    "git_dirty", "dependency_lock_path", "dependency_lock_hash",
    "environment_fingerprint_hash", "protocol_hash",
}
REQUIRED_STATE_FIELDS = {
    "schema_version", "protocol_hash", "current_status", "validation_executions",
    "revision", "last_attempt_hash", "belief_config_hash", "state_hash",
}
OPTIONAL_STATE_FIELDS = {"failure_reason", "locked_test_report_hash"}


def _load(value: dict | str | Path) -> dict:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _validate_canonical_hash(value: dict, field: str, description: str) -> str:
    unsigned = dict(value)
    stored = unsigned.pop(field, None)
    if not stored or stored != canonical_json_hash(unsigned):
        raise ValueError(f"{description} hash is invalid")
    return stored


def split_assignment_hash(manifest: dict) -> str:
    return canonical_json_hash({
        "raw_dataset_hash": manifest.get("raw_dataset_hash"),
        "seed": manifest.get("seed"),
        "assignments": manifest.get("assignments"),
        "summary": manifest.get("summary"),
        "locked_test_assignment_hash": manifest.get("locked_test_assignment_hash"),
    })


def validate_parent_v5_lineage(
    protocol_value,
    report_value,
    manifest_value,
    inventory_value,
) -> dict:
    protocol = validate_calibration_protocol_v5(protocol_value)
    if protocol["protocol_schema_version"] != PARENT_PROTOCOL_VERSION:
        raise ValueError("parent protocol is not calibration v5")
    if protocol["protocol_status"] != "FAILED":
        raise ValueError("parent v5 protocol must be FAILED")
    if int(protocol.get("validation_executions", 0)) != 0:
        raise ValueError("parent v5 protocol must not have consumed validation")

    frozen = dict(protocol)
    frozen["protocol_status"] = "FROZEN_FOR_DEVELOPMENT"
    frozen.pop("protocol_hash", None)
    frozen_hash = canonical_json_hash(frozen)

    report = _load(report_value)
    report_hash = _validate_canonical_hash(report, "report_hash", "parent v5 report")
    if report.get("schema_version") != "aria-calibration-cv-report-v2":
        raise ValueError("unsupported parent v5 report schema")
    if report.get("selection_status") != "FAILED":
        raise ValueError("parent v5 report must record failed selection")
    if report.get("candidate_count") != 4 or report.get("candidate_limit") != 4:
        raise ValueError("parent v5 report does not contain the frozen four-candidate search")
    if report.get("validation_used_for_selection") is not False:
        raise ValueError("parent v5 report used validation for selection")
    if report.get("protocol_hash") != frozen_hash:
        raise ValueError("parent v5 report does not reference the frozen protocol hash")

    manifest = _load(manifest_value)
    manifest_hash = _validate_canonical_hash(manifest, "manifest_hash", "parent v5 manifest")
    if manifest.get("schema_version") != "aria-split-manifest-v4":
        raise ValueError("unsupported parent v5 split manifest")
    if manifest.get("protocol_hash") != frozen_hash:
        raise ValueError("parent v5 manifest does not reference the frozen protocol hash")
    if manifest.get("raw_dataset_hash") != protocol["raw_dataset_hash"]:
        raise ValueError("parent v5 raw dataset hash mismatch")
    if manifest.get("locked_test_assignment_hash") != protocol["locked_test_assignment_hash"]:
        raise ValueError("parent v5 locked-test assignment mismatch")

    inventory = _load(inventory_value)
    inventory_hash = _validate_canonical_hash(
        inventory, "inventory_hash", "parent v5 raw split inventory",
    )
    if inventory.get("schema_version") != "aria-raw-split-inventory-v2":
        raise ValueError("unsupported parent v5 raw split inventory")
    if inventory.get("split_manifest_hash") != manifest_hash:
        raise ValueError("parent v5 inventory manifest hash mismatch")
    if inventory.get("raw_file_sha256") != protocol["raw_file_sha256"]:
        raise ValueError("parent v5 inventory raw file hash mismatch")
    return {
        "protocol": protocol,
        "protocol_final_hash": protocol["protocol_hash"],
        "protocol_frozen_hash": frozen_hash,
        "report": report,
        "report_hash": report_hash,
        "manifest": manifest,
        "manifest_hash": manifest_hash,
        "inventory": inventory,
        "inventory_hash": inventory_hash,
    }


def freeze_calibration_protocol_v6(
    raw_file,
    split_manifest,
    output_dir,
    dependency_lock_path,
    *,
    lineage: dict,
    bootstrap_samples=1000,
    repository=None,
    expected_counts=None,
    artifact_root=None,
) -> dict:
    if int(bootstrap_samples) != 1000:
        raise ValueError("calibration protocol v6 freezes exactly 1000 bootstrap samples")
    output = Path(output_dir).resolve()
    frozen_artifact_root = Path(artifact_root or output).resolve()
    protocol_path = output / "protocol" / "calibration_protocol_v6.json"
    state_path = output / "protocol" / "calibration_protocol_state_v1.json"
    if protocol_path.exists() or state_path.exists():
        raise FileExistsError(f"calibration protocol v6 is already frozen: {protocol_path}")
    raw_path = Path(raw_file).resolve()
    raw_bytes = raw_path.read_bytes()
    transitions = json.loads(raw_bytes.decode("utf-8"))
    manifest = _load(split_manifest)
    episodes = group_transitions_into_episodes(transitions)
    components = connected_identity_components(transitions)
    required = expected_counts or (600, 9018, 33)
    actual = (len(episodes), len(transitions), len(components))
    if actual != tuple(required):
        raise ValueError(f"raw corpus counts {actual} do not match expected {tuple(required)}")
    raw_dataset_hash = canonical_json_hash(transitions)
    if expected_counts is None and raw_dataset_hash != KNOWN_CANONICAL_DATASET_HASH:
        raise ValueError("raw corpus canonical JSON hash does not match production contract")
    if manifest.get("raw_dataset_hash") != raw_dataset_hash:
        raise ValueError("v6 manifest raw dataset hash mismatch")
    fingerprint = build_environment_fingerprint(dependency_lock_path, repository=repository)
    defaults = BeliefModelConfig().to_dict()
    fixed_behavior = {key: defaults[key] for key in FIXED_BEHAVIOR_CONFIGURATION}
    if fixed_behavior != FIXED_BEHAVIOR_CONFIGURATION:
        raise RuntimeError("belief-v2 defaults differ from the frozen v6 behavior contract")
    protocol = {
        "protocol_schema_version": CALIBRATION_PROTOCOL_VERSION,
        "initial_protocol_status": "FROZEN_FOR_DEVELOPMENT",
        "artifact_root": str(frozen_artifact_root),
        "raw_file_path": str(raw_path),
        "raw_file_sha256": file_sha256(raw_path),
        "raw_dataset_hash": raw_dataset_hash,
        "episode_count": len(episodes),
        "transition_count": len(transitions),
        "identity_component_count": len(components),
        "target_component_counts": [21, 6, 6],
        "split_assignment_hash": split_assignment_hash(manifest),
        "locked_test_assignment_hash": manifest["locked_test_assignment_hash"],
        "metric_schema_version": METRICS_SCHEMA_VERSION,
        "gate_policy_version": VALIDATION_GATE_VERSION,
        "gate_thresholds": dict(CALIBRATION_GATE_THRESHOLDS),
        "cross_validation_algorithm": "grouped-identity-component-3fold-greedy-v1",
        "calibration_algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "calibration_stage_sequence": list(CALIBRATION_STAGE_SEQUENCE),
        "candidate_values": CALIBRATION_CANDIDATE_VALUES,
        "candidate_selection_rule": [
            "highest_macro_f1", "lowest_ordinal_mae", "lowest_ece",
            "lowest_abstention_rate", "lexicographically_smallest_parameters",
        ],
        "fixed_behavior_configuration": fixed_behavior,
        "emission_fit_algorithm": "episode_balanced_weighted_median_mad",
        "bootstrap_method": "paired-identity-component-percentile-v1",
        "bootstrap_samples": int(bootstrap_samples),
        "all_random_seeds": {"cross_validation": 42, "bootstrap": 42},
        "numerical_tolerances": dict(NUMERICAL_TOLERANCES),
        "maximum_calibration_candidates": MAX_CALIBRATION_CANDIDATES,
        "maximum_validation_executions": 1,
        "locked_test_policy": "application-level-one-attempt-guard-v1",
        "v6_change_scope": V6_CHANGE_SCOPE,
        "parent_v5_protocol_final_hash": lineage["protocol_final_hash"],
        "parent_v5_protocol_frozen_hash": lineage["protocol_frozen_hash"],
        "parent_v5_report_hash": lineage["report_hash"],
        "parent_v5_split_manifest_hash": lineage["manifest_hash"],
        "parent_v5_inventory_hash": lineage["inventory_hash"],
        "code_commit": fingerprint["code_commit"],
        "git_dirty": fingerprint["git_dirty"],
        "dependency_lock_path": fingerprint["dependency_lock_path"],
        "dependency_lock_hash": fingerprint["dependency_lock_hash"],
        "environment_fingerprint_hash": fingerprint["environment_fingerprint_hash"],
    }
    protocol["protocol_hash"] = canonical_json_hash(protocol)
    state = {
        "schema_version": PROTOCOL_STATE_VERSION,
        "protocol_hash": protocol["protocol_hash"],
        "current_status": "FROZEN_FOR_DEVELOPMENT",
        "validation_executions": 0,
        "revision": 1,
        "last_attempt_hash": None,
        "belief_config_hash": None,
    }
    state["state_hash"] = canonical_json_hash(state)
    atomic_json_write(output / "protocol" / "environment_fingerprint_v1.json", fingerprint)
    atomic_json_write(protocol_path, protocol)
    atomic_json_write(state_path, state)
    return protocol


def validate_calibration_protocol_v6(value, *, raw_file=None, split_manifest=None) -> dict:
    protocol = _load(value)
    missing = REQUIRED_PROTOCOL_FIELDS - protocol.keys()
    if missing:
        raise ValueError(f"calibration protocol is missing {sorted(missing)}")
    if protocol.get("protocol_schema_version") != CALIBRATION_PROTOCOL_VERSION:
        raise ValueError("unsupported calibration protocol version")
    _validate_canonical_hash(protocol, "protocol_hash", "calibration protocol")
    fixed = {
        "initial_protocol_status": "FROZEN_FOR_DEVELOPMENT",
        "metric_schema_version": METRICS_SCHEMA_VERSION,
        "gate_policy_version": VALIDATION_GATE_VERSION,
        "gate_thresholds": CALIBRATION_GATE_THRESHOLDS,
        "calibration_algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "calibration_stage_sequence": CALIBRATION_STAGE_SEQUENCE,
        "candidate_values": CALIBRATION_CANDIDATE_VALUES,
        "maximum_calibration_candidates": 3,
        "maximum_validation_executions": 1,
        "v6_change_scope": V6_CHANGE_SCOPE,
        "fixed_behavior_configuration": FIXED_BEHAVIOR_CONFIGURATION,
        "target_component_counts": [21, 6, 6],
        "cross_validation_algorithm": "grouped-identity-component-3fold-greedy-v1",
        "candidate_selection_rule": [
            "highest_macro_f1", "lowest_ordinal_mae", "lowest_ece",
            "lowest_abstention_rate", "lexicographically_smallest_parameters",
        ],
        "emission_fit_algorithm": "episode_balanced_weighted_median_mad",
        "bootstrap_method": "paired-identity-component-percentile-v1",
        "bootstrap_samples": 1000,
        "all_random_seeds": {"cross_validation": 42, "bootstrap": 42},
        "numerical_tolerances": NUMERICAL_TOLERANCES,
        "locked_test_policy": "application-level-one-attempt-guard-v1",
    }
    for field, expected in fixed.items():
        if protocol.get(field) != expected:
            raise ValueError(f"calibration protocol {field} changed")
    if raw_file is not None:
        if file_sha256(raw_file) != protocol["raw_file_sha256"]:
            raise ValueError("raw file byte hash changed")
        rows = json.loads(Path(raw_file).read_text(encoding="utf-8"))
        if canonical_json_hash(rows) != protocol["raw_dataset_hash"]:
            raise ValueError("raw canonical dataset hash changed")
    if split_manifest is not None:
        manifest = _load(split_manifest)
        _validate_canonical_hash(manifest, "manifest_hash", "split manifest")
        if split_assignment_hash(manifest) != protocol["split_assignment_hash"]:
            raise ValueError("split assignment hash changed")
        if manifest.get("locked_test_assignment_hash") != protocol["locked_test_assignment_hash"]:
            raise ValueError("locked-test assignment hash changed")
    return protocol


def validate_protocol_state(value, *, protocol) -> dict:
    state = _load(value)
    missing = REQUIRED_STATE_FIELDS - state.keys()
    if missing:
        raise ValueError(f"protocol state is missing {sorted(missing)}")
    unknown = set(state) - REQUIRED_STATE_FIELDS - OPTIONAL_STATE_FIELDS
    if unknown:
        raise ValueError(f"protocol state has unsupported fields {sorted(unknown)}")
    if state.get("schema_version") != PROTOCOL_STATE_VERSION:
        raise ValueError("unsupported protocol state version")
    _validate_canonical_hash(state, "state_hash", "protocol state")
    checked = validate_calibration_protocol_v6(protocol)
    if state.get("protocol_hash") != checked["protocol_hash"]:
        raise ValueError("protocol state references a different protocol")
    if state.get("current_status") not in ALLOWED_STATE_TRANSITIONS:
        raise ValueError("unsupported protocol state")
    if not isinstance(state.get("validation_executions"), int) or not 0 <= state["validation_executions"] <= 1:
        raise ValueError("invalid validation execution count")
    if not isinstance(state.get("revision"), int) or state["revision"] < 1:
        raise ValueError("invalid protocol state revision")
    if state["current_status"] in {"PROVISIONAL_SYNTHETIC", "VALIDATED_SYNTHETIC"} and state["validation_executions"] != 1:
        raise ValueError("successful protocol state must record one validation execution")
    return state


def update_protocol_state(state_file, status, *, protocol, **fields) -> dict:
    current = validate_protocol_state(state_file, protocol=protocol)
    if status not in ALLOWED_STATE_TRANSITIONS[current["current_status"]]:
        raise ValueError(
            f"invalid protocol state transition {current['current_status']} -> {status}"
        )
    updated = dict(current)
    updated.update(fields)
    updated["current_status"] = status
    updated["revision"] = int(current["revision"]) + 1
    unknown = set(updated) - REQUIRED_STATE_FIELDS - OPTIONAL_STATE_FIELDS
    if unknown:
        raise ValueError(f"unsupported protocol state fields {sorted(unknown)}")
    updated.pop("state_hash", None)
    updated["state_hash"] = canonical_json_hash(updated)
    atomic_json_write(state_file, updated)
    return updated


def exclusive_json_create(path: str | Path, value: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return output


def validate_development_bundle_v6(
    value,
    *,
    protocol,
    state,
    belief_config_hash,
    split_manifest,
    train_file=None,
    validation_file=None,
) -> dict:
    bundle = _load(value)
    if bundle.get("schema_version") != DEVELOPMENT_BUNDLE_VERSION:
        raise ValueError("unsupported development bundle schema version")
    if DEVELOPMENT_BUNDLE_VERSION not in bundle.get("supported_consumer_versions", []):
        raise ValueError("development bundle is consumer-incompatible")
    _validate_canonical_hash(bundle, "bundle_hash", "development bundle")
    checked_protocol = validate_calibration_protocol_v6(protocol, split_manifest=split_manifest)
    checked_state = validate_protocol_state(state, protocol=checked_protocol)
    if checked_state["current_status"] not in {"PROVISIONAL_SYNTHETIC", "VALIDATED_SYNTHETIC"}:
        raise ValueError("protocol state does not permit training")
    manifest = _load(split_manifest)
    manifest_hash = _validate_canonical_hash(manifest, "manifest_hash", "split manifest")
    expected = {
        "protocol_hash": checked_protocol["protocol_hash"],
        "raw_file_sha256": checked_protocol["raw_file_sha256"],
        "raw_dataset_hash": checked_protocol["raw_dataset_hash"],
        "split_manifest_hash": manifest_hash,
        "split_assignment_hash": checked_protocol["split_assignment_hash"],
        "belief_config_hash": belief_config_hash,
        "environment_fingerprint_hash": checked_protocol["environment_fingerprint_hash"],
    }
    for field, expected_value in expected.items():
        if bundle.get(field) != expected_value:
            raise ValueError(f"development bundle {field} mismatch")
    artifacts = bundle.get("artifacts", {})
    if "replayed_test" in artifacts:
        raise ValueError("development bundle contains replayed test data")
    for name, path, split in (
        ("replayed_train", train_file, "train"),
        ("replayed_validation", validation_file, "validation"),
    ):
        if path is not None:
            record = artifacts.get(name, {})
            if record.get("sha256") != file_sha256(path) or record.get("split") != split:
                raise ValueError(f"development bundle {name} mismatch")
    return bundle
