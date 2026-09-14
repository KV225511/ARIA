"""One-attempt locked-test release evaluator for calibration protocols v4-v7.

The attempt file is an application-level guard. A user who can delete or edit
local artifacts can bypass it; it is not cryptographic enforcement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.belief_calibration import (
    _gate_candidate,
    _prediction_rows,
    _metrics_from_vectors,
    paired_component_bootstrap,
)
from modules.module_07_rl.calibration_protocol import (
    CALIBRATION_PROTOCOL_VERSION,
    atomic_json_write,
    canonical_json_hash,
    file_sha256,
    update_protocol_status,
    validate_calibration_protocol,
)
from modules.module_07_rl.calibration_protocol_v5 import (
    CALIBRATION_ALGORITHM_VERSION as CALIBRATION_ALGORITHM_VERSION_V5,
    CALIBRATION_PROTOCOL_VERSION as CALIBRATION_PROTOCOL_VERSION_V5,
    update_protocol_status_v5,
    validate_calibration_protocol_v5,
)
from modules.module_07_rl.calibration_protocol_v6 import (
    CALIBRATION_ALGORITHM_VERSION as CALIBRATION_ALGORITHM_VERSION_V6,
    CALIBRATION_PROTOCOL_VERSION as CALIBRATION_PROTOCOL_VERSION_V6,
    update_protocol_state,
    validate_calibration_protocol_v6,
    validate_protocol_state,
)
from modules.module_07_rl.calibration_protocol_v7 import (
    CALIBRATION_ALGORITHM_VERSION as CALIBRATION_ALGORITHM_VERSION_V7,
    CALIBRATION_PROTOCOL_VERSION as CALIBRATION_PROTOCOL_VERSION_V7,
    V7_GATE_THRESHOLDS,
    update_protocol_state_v7,
    validate_calibration_protocol_v7,
    validate_protocol_state_v7,
)
from modules.module_07_rl.dataset_split import (
    connected_identity_components,
    group_transitions_into_episodes,
)


LOCKED_TEST_EVALUATION_VERSION = "aria-locked-test-evaluation-v1"
RELEASE_ATTEMPT_VERSION = "aria-release-attempt-v1"


def _gate_v7(metrics: dict) -> tuple[bool, list[str]]:
    base_passed, reasons = _gate_candidate(metrics, minimum_components=6)
    per_class = metrics.get("per_class", {})
    checks = {
        "beginner_recall_below_v7_minimum": per_class.get(0, {}).get("recall", 0.0) >= V7_GATE_THRESHOLDS["minimum_beginner_recall"],
        "beginner_precision_below_v7_minimum": per_class.get(0, {}).get("precision", 0.0) >= V7_GATE_THRESHOLDS["minimum_beginner_precision"],
        "mid_recall_below_v7_minimum": per_class.get(1, {}).get("recall", 0.0) >= V7_GATE_THRESHOLDS["minimum_mid_recall"],
        "expert_recall_below_v7_minimum": per_class.get(2, {}).get("recall", 0.0) >= V7_GATE_THRESHOLDS["minimum_expert_recall"],
    }
    reasons.extend(name for name, passed in checks.items() if not passed)
    return base_passed and all(checks.values()), reasons


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_manifest(manifest: dict) -> str:
    if manifest.get("schema_version") != "aria-split-manifest-v4":
        raise ValueError("unsupported split manifest schema version")
    stored = manifest.get("manifest_hash")
    unsigned = dict(manifest)
    unsigned.pop("manifest_hash", None)
    if not stored or stored != canonical_json_hash(unsigned):
        raise ValueError("split manifest hash is invalid")
    return stored


def _exclusive_json_create(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def evaluate_locked_test_once(
    locked_test: str | Path,
    belief_config: str | Path,
    protocol: str | Path,
    split_manifest: str | Path,
    output_dir: str | Path,
    protocol_state: str | Path | None = None,
    raw_split_inventory: str | Path | None = None,
) -> dict:
    """Validate frozen inputs, consume the one attempt, then read the test split."""
    protocol_path = Path(protocol)
    protocol_preview = _load_json(protocol_path)
    protocol_version = protocol_preview.get("protocol_schema_version")
    if protocol_version == CALIBRATION_PROTOCOL_VERSION:
        checked_protocol = validate_calibration_protocol(protocol_preview)
        protocol_updater = update_protocol_status
        producer_version = "aria-belief-calibration-v4"
        current_status = checked_protocol["protocol_status"]
        expected_config_hash = checked_protocol.get("belief_config_hash")
    elif protocol_version == CALIBRATION_PROTOCOL_VERSION_V5:
        checked_protocol = validate_calibration_protocol_v5(protocol_preview)
        protocol_updater = update_protocol_status_v5
        producer_version = CALIBRATION_ALGORITHM_VERSION_V5
        current_status = checked_protocol["protocol_status"]
        expected_config_hash = checked_protocol.get("belief_config_hash")
    elif protocol_version == CALIBRATION_PROTOCOL_VERSION_V6:
        checked_protocol = validate_calibration_protocol_v6(protocol_preview)
        state_path = Path(protocol_state) if protocol_state else protocol_path.with_name(
            "calibration_protocol_state_v1.json"
        )
        checked_state = validate_protocol_state(state_path, protocol=checked_protocol)
        protocol_updater = lambda _path, status: update_protocol_state(
            state_path, status, protocol=checked_protocol,
        )
        producer_version = CALIBRATION_ALGORITHM_VERSION_V6
        current_status = checked_state["current_status"]
        expected_config_hash = checked_state.get("belief_config_hash")
        if Path(output_dir).resolve() != Path(checked_protocol["artifact_root"]).resolve():
            raise ValueError("output directory differs from the frozen artifact root")
        if raw_split_inventory is None:
            raise ValueError("locked test v6 requires the raw split inventory")
        inventory = _load_json(raw_split_inventory)
        unsigned_inventory = dict(inventory)
        stored_inventory_hash = unsigned_inventory.pop("inventory_hash", None)
        if (
            inventory.get("schema_version") != "aria-raw-split-inventory-v3"
            or not stored_inventory_hash
            or stored_inventory_hash != canonical_json_hash(unsigned_inventory)
            or inventory.get("protocol_hash") != checked_protocol["protocol_hash"]
        ):
            raise ValueError("invalid v6 raw split inventory")
    elif protocol_version == CALIBRATION_PROTOCOL_VERSION_V7:
        checked_protocol = validate_calibration_protocol_v7(protocol_preview)
        state_path = Path(protocol_state) if protocol_state else protocol_path.with_name(
            "calibration_protocol_state_v2.json"
        )
        checked_state = validate_protocol_state_v7(state_path, protocol=checked_protocol)
        producer_version = CALIBRATION_ALGORITHM_VERSION_V7
        current_status = checked_state["current_status"]
        expected_config_hash = checked_state.get("belief_config_hash")
        if Path(output_dir).resolve() != Path(checked_protocol["artifact_root"]).resolve():
            raise ValueError("output directory differs from the frozen artifact root")
        if raw_split_inventory is None:
            raise ValueError("locked test v7 requires the parent v6 raw split inventory")
        inventory = _load_json(raw_split_inventory)
        unsigned_inventory = dict(inventory)
        stored_inventory_hash = unsigned_inventory.pop("inventory_hash", None)
        if (
            inventory.get("schema_version") != "aria-raw-split-inventory-v3"
            or stored_inventory_hash != canonical_json_hash(unsigned_inventory)
            or stored_inventory_hash != checked_protocol["parent_v6_inventory_hash"]
        ):
            raise ValueError("invalid v7 parent raw split inventory")
    else:
        raise ValueError("unsupported calibration protocol version")
    release_dir = Path(output_dir).resolve() / "release"
    attempt_path = release_dir / "release_attempt_v1.json"
    if attempt_path.exists():
        raise FileExistsError(
            f"release attempt already exists and cannot be repeated: {attempt_path}"
        )
    required_status = (
        "PROVISIONAL_TRAINING_CV"
        if protocol_version == CALIBRATION_PROTOCOL_VERSION_V7
        else "PROVISIONAL_SYNTHETIC"
    )
    if current_status != required_status:
        raise ValueError(f"locked test requires {required_status} protocol status")
    config = BeliefModelConfig.load(belief_config)
    expected_schema = "belief-v3" if protocol_version == CALIBRATION_PROTOCOL_VERSION_V7 else "belief-v2"
    if config.schema_version != expected_schema:
        raise ValueError("unsupported belief configuration schema")
    if config.config_hash != expected_config_hash:
        raise ValueError("belief configuration hash mismatch")
    manifest = _load_json(split_manifest)
    manifest_hash = _validate_manifest(manifest)
    if protocol_version == CALIBRATION_PROTOCOL_VERSION_V6:
        checked_protocol = validate_calibration_protocol_v6(
            checked_protocol,
            split_manifest=manifest,
        )
        if inventory.get("split_manifest_hash") != manifest_hash:
            raise ValueError("v6 raw split inventory manifest hash mismatch")
    elif protocol_version == CALIBRATION_PROTOCOL_VERSION_V7:
        checked_protocol = validate_calibration_protocol_v7(
            checked_protocol, split_manifest=manifest,
        )
        if inventory.get("split_manifest_hash") != manifest_hash:
            raise ValueError("v7 parent raw split inventory manifest hash mismatch")
    if config.split_manifest_hash != manifest_hash:
        raise ValueError("belief configuration split manifest hash mismatch")
    if manifest.get("raw_dataset_hash") != checked_protocol["raw_dataset_hash"]:
        raise ValueError("split manifest raw dataset hash mismatch")
    if manifest.get("locked_test_assignment_hash") != checked_protocol["locked_test_assignment_hash"]:
        raise ValueError("locked-test assignment hash mismatch")

    attempt = {
        "schema_version": RELEASE_ATTEMPT_VERSION,
        "producer_version": producer_version,
        "supported_consumer_versions": [RELEASE_ATTEMPT_VERSION],
        "state": "STARTED",
        "protocol_hash": checked_protocol["protocol_hash"],
        "raw_file_sha256": checked_protocol["raw_file_sha256"],
        "raw_dataset_hash": checked_protocol["raw_dataset_hash"],
        "belief_config_hash": config.config_hash,
        "split_manifest_hash": manifest_hash,
        "locked_test_assignment_hash": checked_protocol["locked_test_assignment_hash"],
        "parent_artifact_hashes": {
            "protocol": checked_protocol["protocol_hash"],
            "split_manifest": manifest_hash,
        },
        "git_commit": checked_protocol["code_commit"],
        "environment_fingerprint_hash": checked_protocol["environment_fingerprint_hash"],
        "guard_scope": "application-level-one-attempt; not cryptographic enforcement",
    }
    _exclusive_json_create(attempt_path, attempt)

    try:
        # No test bytes or labels are read before the durable attempt exists.
        if protocol_version in {CALIBRATION_PROTOCOL_VERSION_V6, CALIBRATION_PROTOCOL_VERSION_V7}:
            locked_record = inventory.get("artifacts", {}).get("locked_test", {})
            expected_locked_hash = (
                locked_record.get("sha256") if isinstance(locked_record, dict)
                else locked_record
            )
            if expected_locked_hash != file_sha256(locked_test):
                raise ValueError("locked test file hash mismatch")
        transitions = _load_json(locked_test)
        if not transitions:
            raise ValueError("locked test split is empty")
        episodes = group_transitions_into_episodes(transitions)
        expected_ids = sorted(
            episode_id for episode_id, split in manifest.get("assignments", {}).items()
            if split == "test"
        )
        actual_ids = sorted(str(episode[0].get("episode_id")) for episode in episodes if episode)
        if actual_ids != expected_ids:
            raise ValueError("locked test episodes do not match manifest assignments")
        locked_payload = {
            "episode_ids": actual_ids,
            "resume_content_hashes": sorted({
                str(item["resume_content_hash"]) for item in transitions
                if item.get("resume_content_hash")
            }),
            "jd_content_hashes": sorted({
                str(item["jd_content_hash"]) for item in transitions
                if item.get("jd_content_hash")
            }),
        }
        if canonical_json_hash(locked_payload) != checked_protocol["locked_test_assignment_hash"]:
            raise ValueError("locked test identity assignment content is invalid")
        if any(item.get("dataset_split") not in (None, "test") for item in transitions):
            raise ValueError("locked test file contains a non-test transition")
        rows = _prediction_rows(transitions, config)
        if len(rows) != len(episodes):
            raise ValueError("every locked-test episode must produce an assessment")
        metrics = _metrics_from_vectors(
            [row["true_label"] for row in rows],
            [row["prediction"] for row in rows],
            [row["probabilities"] for row in rows],
        )
        metrics["identity_component_count"] = len(connected_identity_components(transitions))
        passed, reasons = (
            _gate_v7(metrics) if protocol_version == CALIBRATION_PROTOCOL_VERSION_V7
            else _gate_candidate(metrics, minimum_components=6)
        )
        comparison = None
        if protocol_version == CALIBRATION_PROTOCOL_VERSION_V7:
            baseline_config = BeliefModelConfig.from_dict(
                checked_protocol["parent_v6_belief_config"]
            )
            baseline_rows = _prediction_rows(transitions, baseline_config)
            if [row["episode_id"] for row in baseline_rows] != [row["episode_id"] for row in rows]:
                raise ValueError("v6 and v7 locked-test episode sets differ")
            baseline_metrics = _metrics_from_vectors(
                [row["true_label"] for row in baseline_rows],
                [row["prediction"] for row in baseline_rows],
                [row["probabilities"] for row in baseline_rows],
            )
            non_regression = {
                "overall_accuracy": metrics["overall_accuracy"] >= baseline_metrics["overall_accuracy"] - 0.02,
                "macro_f1": metrics["macro_f1"] >= baseline_metrics["macro_f1"] - 0.02,
            }
            if not all(non_regression.values()):
                reasons.extend(
                    f"locked_test_{name}_non_regression_failed"
                    for name, ok in non_regression.items() if not ok
                )
                passed = False
            comparison = {
                "baseline_protocol_hash": checked_protocol["parent_v6_protocol_hash"],
                "baseline_belief_config_hash": baseline_config.config_hash,
                "baseline_metrics": baseline_metrics,
                "non_regression": non_regression,
                "paired_component_bootstrap": paired_component_bootstrap(
                    transitions,
                    [row["prediction"] for row in baseline_rows],
                    [row["probabilities"] for row in baseline_rows],
                    [row["prediction"] for row in rows],
                    [row["probabilities"] for row in rows],
                    samples=1000, seed=42,
                ),
            }
        report = {
            "schema_version": LOCKED_TEST_EVALUATION_VERSION,
            "producer_version": producer_version,
            "supported_consumer_versions": [LOCKED_TEST_EVALUATION_VERSION],
            "protocol_hash": checked_protocol["protocol_hash"],
            "raw_file_sha256": checked_protocol["raw_file_sha256"],
            "raw_dataset_hash": checked_protocol["raw_dataset_hash"],
            "split_manifest_hash": manifest_hash,
            "belief_config_hash": config.config_hash,
            "parent_artifact_hashes": {
                "protocol": checked_protocol["protocol_hash"],
                "split_manifest": manifest_hash,
            },
            "git_commit": checked_protocol["code_commit"],
            "environment_fingerprint_hash": checked_protocol["environment_fingerprint_hash"],
            "metrics": metrics,
            "rejection_reasons": reasons,
            "passes_quality_gates": passed,
            "final_status": "VALIDATED_SYNTHETIC" if passed else "FAILED",
            "evaluates_learned_policy": False,
        }
        if comparison is not None:
            report["v6_v7_paired_comparison"] = comparison
        report["report_hash"] = canonical_json_hash(report)
        atomic_json_write(release_dir / "locked_test_evaluation_v1.json", report)
        attempt["state"] = "COMPLETED"
        attempt["evaluation_report_hash"] = report["report_hash"]
        attempt["final_status"] = report["final_status"]
        atomic_json_write(attempt_path, attempt)
        if protocol_version == CALIBRATION_PROTOCOL_VERSION_V6:
            update_protocol_state(
                state_path,
                report["final_status"],
                protocol=checked_protocol,
                belief_config_hash=config.config_hash,
                last_attempt_hash=canonical_json_hash(attempt),
                locked_test_report_hash=report["report_hash"],
            )
        elif protocol_version == CALIBRATION_PROTOCOL_VERSION_V7:
            update_protocol_state_v7(
                state_path, report["final_status"], protocol=checked_protocol,
                belief_config_hash=config.config_hash,
                last_attempt_hash=canonical_json_hash(attempt),
                locked_test_report_hash=report["report_hash"],
            )
        else:
            protocol_updater(protocol_path, report["final_status"])
        return report
    except BaseException as exc:
        attempt["state"] = "FAILED"
        attempt["failure_type"] = type(exc).__name__
        atomic_json_write(attempt_path, attempt)
        if protocol_version == CALIBRATION_PROTOCOL_VERSION_V6:
            latest_state = validate_protocol_state(state_path, protocol=checked_protocol)
            if latest_state["current_status"] == "PROVISIONAL_SYNTHETIC":
                update_protocol_state(
                    state_path,
                    "FAILED",
                    protocol=checked_protocol,
                    belief_config_hash=config.config_hash,
                    last_attempt_hash=canonical_json_hash(attempt),
                )
        elif protocol_version == CALIBRATION_PROTOCOL_VERSION_V7:
            latest_state = validate_protocol_state_v7(state_path, protocol=checked_protocol)
            if latest_state["current_status"] == "PROVISIONAL_TRAINING_CV":
                update_protocol_state_v7(
                    state_path, "FAILED", protocol=checked_protocol,
                    belief_config_hash=config.config_hash,
                    last_attempt_hash=canonical_json_hash(attempt),
                )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-attempt ARIA locked-test evaluator (application-level guard).",
    )
    parser.add_argument("--locked-test", required=True)
    parser.add_argument("--belief-config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-state")
    parser.add_argument("--raw-split-inventory")
    args = parser.parse_args()
    try:
        result = evaluate_locked_test_once(
            args.locked_test, args.belief_config, args.protocol,
            args.split_manifest, args.output_dir,
            protocol_state=args.protocol_state,
            raw_split_inventory=args.raw_split_inventory,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"[ERROR] {error}\n")
    print(json.dumps(result, indent=2))
