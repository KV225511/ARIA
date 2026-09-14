"""Frozen v7 training-only Low-class correction and release contracts.

V7 has no independent development-validation budget. Its parent v6 validation
has already been observed, so selection is restricted to original training
components and the locked test is the sole clean release evaluation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.calibration_protocol import (
    NUMERICAL_TOLERANCES, atomic_json_write, build_environment_fingerprint,
    canonical_json_hash, file_sha256,
)
from modules.module_07_rl.calibration_protocol_v6 import (
    split_assignment_hash, validate_calibration_protocol_v6,
    validate_protocol_state as validate_protocol_state_v6,
)

CALIBRATION_PROTOCOL_VERSION = "aria-calibration-protocol-v7"
CALIBRATION_ALGORITHM_VERSION = "aria-belief-calibration-v7"
CALIBRATION_CV_REPORT_VERSION = "aria-calibration-cv-report-v4"
DEVELOPMENT_BUNDLE_VERSION = "aria-development-bundle-v4"
DEVELOPMENT_REPORT_VERSION = "aria-development-calibration-v7"
PROTOCOL_STATE_VERSION = "aria-calibration-protocol-state-v2"
PREPARATION_ATTEMPT_VERSION = "aria-v7-preparation-attempt-v1"
LOW_CLASS_LOGIT_BIAS_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MAX_CALIBRATION_CANDIDATES = 6
V7_CHANGE_SCOPE = "low-class-logit-bias-only"
V7_GATE_THRESHOLDS = {
    "minimum_overall_accuracy": 0.60, "minimum_macro_f1": 0.60,
    "minimum_beginner_recall": 0.65, "minimum_beginner_precision": 0.70,
    "minimum_mid_recall": 0.90, "minimum_expert_recall": 0.90,
    "maximum_classified_prediction_share": 0.60,
    "maximum_abstention_rate": 0.15,
    "maximum_expected_calibration_error": 0.15,
}
ALLOWED_STATE_TRANSITIONS = {
    "FROZEN_FOR_DEVELOPMENT": {"FAILED", "PROVISIONAL_TRAINING_CV"},
    "PROVISIONAL_TRAINING_CV": {"FAILED", "VALIDATED_SYNTHETIC"},
    "VALIDATED_SYNTHETIC": set(), "FAILED": set(),
}
REQUIRED_PROTOCOL_FIELDS = {
    "protocol_schema_version", "initial_protocol_status", "artifact_root",
    "raw_file_sha256", "raw_dataset_hash", "episode_count", "transition_count",
    "identity_component_count", "target_component_counts", "split_assignment_hash",
    "locked_test_assignment_hash", "split_manifest_hash",
    "calibration_algorithm_version", "cross_validation_algorithm",
    "candidate_values", "candidate_selection_rule", "gate_thresholds",
    "maximum_calibration_candidates", "maximum_validation_executions",
    "candidate_selection_source", "independent_development_validation_available",
    "clean_final_evaluation_source", "locked_test_policy", "v7_change_scope",
    "parent_v6_protocol_hash", "parent_v6_state_hash", "parent_v6_report_hash",
    "parent_v6_inventory_hash", "parent_v6_belief_config_hash",
    "parent_v6_belief_config", "code_commit", "git_dirty",
    "dependency_lock_path", "dependency_lock_hash",
    "environment_fingerprint_hash", "numerical_tolerances", "protocol_hash",
}
REQUIRED_STATE_FIELDS = {
    "schema_version", "protocol_hash", "current_status", "revision",
    "preparation_executions", "belief_config_hash", "last_attempt_hash", "state_hash",
}
OPTIONAL_STATE_FIELDS = {"failure_reason", "locked_test_report_hash"}


def _load(value) -> dict:
    return value if isinstance(value, dict) else json.loads(Path(value).read_text(encoding="utf-8"))


def _validated_hash(value: dict, field: str, description: str) -> str:
    unsigned = dict(value)
    stored = unsigned.pop(field, None)
    if not stored or stored != canonical_json_hash(unsigned):
        raise ValueError(f"{description} hash is invalid")
    return stored


def validate_parent_v6_lineage(
    parent_protocol, parent_state, parent_report, parent_config,
    split_manifest, raw_split_inventory,
) -> dict:
    protocol = validate_calibration_protocol_v6(parent_protocol, split_manifest=split_manifest)
    state = validate_protocol_state_v6(parent_state, protocol=protocol)
    if state["current_status"] != "PROVISIONAL_SYNTHETIC" or state["validation_executions"] != 1:
        raise ValueError("v7 requires a once-validated PROVISIONAL_SYNTHETIC v6 parent")
    config = parent_config if isinstance(parent_config, BeliefModelConfig) else (
        BeliefModelConfig.from_dict(parent_config) if isinstance(parent_config, dict)
        else BeliefModelConfig.load(parent_config)
    )
    if config.schema_version != "belief-v2" or config.config_hash != state["belief_config_hash"]:
        raise ValueError("parent v6 belief configuration hash mismatch")
    report = _load(parent_report)
    report_hash = _validated_hash(report, "report_hash", "parent v6 report")
    if report.get("schema_version") != "aria-calibration-cv-report-v3" or report.get("selection_status") != "ELIGIBLE":
        raise ValueError("parent v6 report is not an eligible v3 CV report")
    if report.get("protocol_hash") != protocol["protocol_hash"]:
        raise ValueError("parent v6 report lineage mismatch")
    if report.get("validation_used_for_selection") is not False:
        raise ValueError("parent v6 report used validation for selection")
    selected_parameters = (report.get("selected_candidate") or {}).get("parameters", {})
    if float(selected_parameters.get("minimum_effective_evidence", -1)) != float(config.minimum_effective_evidence):
        raise ValueError("parent v6 report and final configuration disagree")
    manifest = _load(split_manifest)
    manifest_hash = _validated_hash(manifest, "manifest_hash", "split manifest")
    if manifest.get("schema_version") != "aria-split-manifest-v4":
        raise ValueError("unsupported split manifest schema")
    if split_assignment_hash(manifest) != protocol["split_assignment_hash"]:
        raise ValueError("parent v6 split assignment hash mismatch")
    if manifest.get("locked_test_assignment_hash") != protocol["locked_test_assignment_hash"]:
        raise ValueError("parent v6 locked-test assignment hash mismatch")
    inventory = _load(raw_split_inventory)
    inventory_hash = _validated_hash(inventory, "inventory_hash", "parent v6 inventory")
    if inventory.get("schema_version") != "aria-raw-split-inventory-v3":
        raise ValueError("unsupported parent v6 inventory schema")
    for field, expected in {
        "protocol_hash": protocol["protocol_hash"], "raw_file_sha256": protocol["raw_file_sha256"],
        "raw_dataset_hash": protocol["raw_dataset_hash"], "split_manifest_hash": manifest_hash,
        "split_assignment_hash": protocol["split_assignment_hash"],
    }.items():
        if inventory.get(field) != expected:
            raise ValueError(f"parent v6 inventory {field} mismatch")
    return {
        "protocol": protocol, "state": state, "report": report, "config": config,
        "manifest": manifest, "inventory": inventory, "report_hash": report_hash,
        "manifest_hash": manifest_hash, "inventory_hash": inventory_hash,
    }


def freeze_calibration_protocol_v7(
    parent_protocol, parent_state, parent_report, parent_config,
    split_manifest, raw_split_inventory, output_dir, dependency_lock_path,
    *, repository=None,
) -> dict:
    lineage = validate_parent_v6_lineage(
        parent_protocol, parent_state, parent_report, parent_config,
        split_manifest, raw_split_inventory,
    )
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"calibration v7 output already exists: {output}")
    fingerprint = build_environment_fingerprint(dependency_lock_path, repository=repository)
    parent = lineage["protocol"]
    protocol = {
        "protocol_schema_version": CALIBRATION_PROTOCOL_VERSION,
        "initial_protocol_status": "FROZEN_FOR_DEVELOPMENT", "artifact_root": str(output),
        "raw_file_sha256": parent["raw_file_sha256"], "raw_dataset_hash": parent["raw_dataset_hash"],
        "episode_count": parent["episode_count"], "transition_count": parent["transition_count"],
        "identity_component_count": parent["identity_component_count"],
        "target_component_counts": list(parent["target_component_counts"]),
        "split_assignment_hash": parent["split_assignment_hash"],
        "locked_test_assignment_hash": parent["locked_test_assignment_hash"],
        "split_manifest_hash": lineage["manifest_hash"],
        "calibration_algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "cross_validation_algorithm": "reuse-authenticated-v6-training-oof-v1",
        "candidate_values": {"low_class_logit_bias": list(LOW_CLASS_LOGIT_BIAS_VALUES)},
        "candidate_selection_rule": ["highest_macro_f1", "highest_minimum_class_recall", "highest_beginner_recall", "lowest_ordinal_mae", "lowest_ece", "lexicographically_smallest_parameters"],
        "gate_thresholds": dict(V7_GATE_THRESHOLDS),
        "maximum_calibration_candidates": MAX_CALIBRATION_CANDIDATES,
        "maximum_validation_executions": 0,
        "candidate_selection_source": "original-training-components-only",
        "independent_development_validation_available": False,
        "clean_final_evaluation_source": "locked-test-only",
        "locked_test_policy": "application-level-one-attempt-guard-v1",
        "v7_change_scope": V7_CHANGE_SCOPE,
        "parent_v6_protocol_hash": parent["protocol_hash"], "parent_v6_state_hash": lineage["state"]["state_hash"],
        "parent_v6_report_hash": lineage["report_hash"], "parent_v6_inventory_hash": lineage["inventory_hash"],
        "parent_v6_belief_config_hash": lineage["config"].config_hash,
        "parent_v6_belief_config": lineage["config"].to_dict(),
        "code_commit": fingerprint["code_commit"], "git_dirty": fingerprint["git_dirty"],
        "dependency_lock_path": fingerprint["dependency_lock_path"], "dependency_lock_hash": fingerprint["dependency_lock_hash"],
        "environment_fingerprint_hash": fingerprint["environment_fingerprint_hash"],
        "numerical_tolerances": dict(NUMERICAL_TOLERANCES),
    }
    protocol["protocol_hash"] = canonical_json_hash(protocol)
    state = {
        "schema_version": PROTOCOL_STATE_VERSION, "protocol_hash": protocol["protocol_hash"],
        "current_status": "FROZEN_FOR_DEVELOPMENT", "revision": 1,
        "preparation_executions": 0, "belief_config_hash": None, "last_attempt_hash": None,
    }
    state["state_hash"] = canonical_json_hash(state)
    atomic_json_write(output / "protocol" / "environment_fingerprint_v1.json", fingerprint)
    atomic_json_write(output / "protocol" / "calibration_protocol_v7.json", protocol)
    atomic_json_write(output / "protocol" / "calibration_protocol_state_v2.json", state)
    for directory in ("calibration", "manifests", "replayed", "audits", "release"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    return protocol


def validate_calibration_protocol_v7(value, *, split_manifest=None) -> dict:
    protocol = _load(value)
    missing = REQUIRED_PROTOCOL_FIELDS - protocol.keys()
    if missing:
        raise ValueError(f"calibration protocol is missing {sorted(missing)}")
    unknown = set(protocol) - REQUIRED_PROTOCOL_FIELDS
    if unknown:
        raise ValueError(f"calibration protocol has unsupported fields {sorted(unknown)}")
    if protocol.get("protocol_schema_version") != CALIBRATION_PROTOCOL_VERSION:
        raise ValueError("unsupported calibration protocol version")
    _validated_hash(protocol, "protocol_hash", "calibration protocol")
    fixed = {
        "initial_protocol_status": "FROZEN_FOR_DEVELOPMENT", "calibration_algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "cross_validation_algorithm": "reuse-authenticated-v6-training-oof-v1",
        "candidate_values": {"low_class_logit_bias": LOW_CLASS_LOGIT_BIAS_VALUES},
        "candidate_selection_rule": ["highest_macro_f1", "highest_minimum_class_recall", "highest_beginner_recall", "lowest_ordinal_mae", "lowest_ece", "lexicographically_smallest_parameters"],
        "gate_thresholds": V7_GATE_THRESHOLDS, "maximum_calibration_candidates": 6,
        "maximum_validation_executions": 0, "candidate_selection_source": "original-training-components-only",
        "independent_development_validation_available": False, "clean_final_evaluation_source": "locked-test-only",
        "locked_test_policy": "application-level-one-attempt-guard-v1", "v7_change_scope": V7_CHANGE_SCOPE,
        "numerical_tolerances": NUMERICAL_TOLERANCES,
    }
    for field, expected in fixed.items():
        if protocol.get(field) != expected:
            raise ValueError(f"calibration protocol {field} changed")
    parent_config = BeliefModelConfig.from_dict(protocol["parent_v6_belief_config"])
    if parent_config.schema_version != "belief-v2" or parent_config.config_hash != protocol["parent_v6_belief_config_hash"]:
        raise ValueError("v7 protocol parent configuration hash mismatch")
    if split_manifest is not None:
        manifest = _load(split_manifest)
        manifest_hash = _validated_hash(manifest, "manifest_hash", "split manifest")
        if manifest_hash != protocol["split_manifest_hash"] or split_assignment_hash(manifest) != protocol["split_assignment_hash"]:
            raise ValueError("split manifest or assignment hash changed")
        if manifest.get("locked_test_assignment_hash") != protocol["locked_test_assignment_hash"]:
            raise ValueError("locked-test assignment hash changed")
    return protocol


def validate_protocol_state_v7(value, *, protocol) -> dict:
    state = _load(value)
    missing = REQUIRED_STATE_FIELDS - state.keys()
    if missing:
        raise ValueError(f"protocol state is missing {sorted(missing)}")
    unknown = set(state) - REQUIRED_STATE_FIELDS - OPTIONAL_STATE_FIELDS
    if unknown:
        raise ValueError(f"protocol state has unsupported fields {sorted(unknown)}")
    if state.get("schema_version") != PROTOCOL_STATE_VERSION:
        raise ValueError("unsupported protocol state version")
    _validated_hash(state, "state_hash", "protocol state")
    checked = validate_calibration_protocol_v7(protocol)
    if state.get("protocol_hash") != checked["protocol_hash"] or state.get("current_status") not in ALLOWED_STATE_TRANSITIONS:
        raise ValueError("protocol state is incompatible")
    if not isinstance(state.get("revision"), int) or state["revision"] < 1 or state.get("preparation_executions") not in (0, 1):
        raise ValueError("invalid protocol state counters")
    if state["current_status"] in {"PROVISIONAL_TRAINING_CV", "VALIDATED_SYNTHETIC"} and (state["preparation_executions"] != 1 or not state.get("belief_config_hash")):
        raise ValueError("successful v7 state lacks completed preparation provenance")
    return state


def update_protocol_state_v7(state_file, status, *, protocol, **fields) -> dict:
    current = validate_protocol_state_v7(state_file, protocol=protocol)
    if status not in ALLOWED_STATE_TRANSITIONS[current["current_status"]]:
        raise ValueError(f"invalid protocol state transition {current['current_status']} -> {status}")
    updated = dict(current)
    updated.update(fields)
    updated["current_status"] = status
    updated["revision"] = current["revision"] + 1
    updated.pop("state_hash", None)
    if set(updated) - REQUIRED_STATE_FIELDS - OPTIONAL_STATE_FIELDS:
        raise ValueError("unsupported protocol state fields")
    updated["state_hash"] = canonical_json_hash(updated)
    atomic_json_write(state_file, updated)
    return updated


def exclusive_json_create(path: str | Path, value: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return output


def validate_development_bundle_v7(value, *, protocol, state, belief_config_hash, split_manifest, train_file=None, validation_file=None) -> dict:
    bundle = _load(value)
    if bundle.get("schema_version") != DEVELOPMENT_BUNDLE_VERSION or DEVELOPMENT_BUNDLE_VERSION not in bundle.get("supported_consumer_versions", []):
        raise ValueError("unsupported or incompatible development bundle schema")
    _validated_hash(bundle, "bundle_hash", "development bundle")
    checked_protocol = validate_calibration_protocol_v7(protocol, split_manifest=split_manifest)
    checked_state = validate_protocol_state_v7(state, protocol=checked_protocol)
    if checked_state["current_status"] not in {"PROVISIONAL_TRAINING_CV", "VALIDATED_SYNTHETIC"}:
        raise ValueError("protocol state does not permit training")
    manifest = _load(split_manifest)
    manifest_hash = _validated_hash(manifest, "manifest_hash", "split manifest")
    for field, expected in {
        "protocol_hash": checked_protocol["protocol_hash"], "raw_file_sha256": checked_protocol["raw_file_sha256"],
        "raw_dataset_hash": checked_protocol["raw_dataset_hash"], "split_manifest_hash": manifest_hash,
        "split_assignment_hash": checked_protocol["split_assignment_hash"], "belief_config_hash": belief_config_hash,
        "environment_fingerprint_hash": checked_protocol["environment_fingerprint_hash"],
    }.items():
        if bundle.get(field) != expected:
            raise ValueError(f"development bundle {field} mismatch")
    artifacts = bundle.get("artifacts", {})
    if "replayed_test" in artifacts or "locked_test_evaluation" in artifacts:
        raise ValueError("development bundle contains locked-test output")
    for name, path, split in (("replayed_train", train_file, "train"), ("replayed_validation", validation_file, "validation")):
        if path is not None:
            record = artifacts.get(name, {})
            if record.get("sha256") != file_sha256(path) or record.get("split") != split:
                raise ValueError(f"development bundle {name} mismatch")
    return bundle
