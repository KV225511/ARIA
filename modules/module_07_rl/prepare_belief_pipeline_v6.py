"""Calibration v6 preparation with authenticated inputs and one validation attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.belief_calibration import (
    select_training_only_calibration_v6,
    validate_selected_calibration,
)
from modules.module_07_rl.calibration_protocol import atomic_json_write, canonical_json_hash, file_sha256
from modules.module_07_rl.calibration_protocol_v6 import (
    CALIBRATION_ALGORITHM_VERSION,
    CALIBRATION_CV_REPORT_VERSION,
    DEVELOPMENT_BUNDLE_VERSION,
    DEVELOPMENT_REPORT_VERSION,
    RAW_SPLIT_INVENTORY_VERSION,
    VALIDATION_ATTEMPT_VERSION,
    exclusive_json_create,
    freeze_calibration_protocol_v6,
    split_assignment_hash,
    update_protocol_state,
    validate_calibration_protocol_v6,
    validate_parent_v5_lineage,
    validate_protocol_state,
)
from modules.module_07_rl.dataset_split import (
    connected_identity_components,
    group_transitions_into_episodes,
)
from modules.module_07_rl.replay_dataset import REPLAY_SCHEMA_VERSION, replay_one_episode


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_manifest(manifest):
    unsigned = dict(manifest)
    stored = unsigned.pop("manifest_hash", None)
    if manifest.get("schema_version") != "aria-split-manifest-v4":
        raise ValueError("unsupported split manifest schema")
    if not stored or stored != canonical_json_hash(unsigned):
        raise ValueError("split manifest hash is invalid")
    return stored


def _validate_inventory(inventory):
    unsigned = dict(inventory)
    stored = unsigned.pop("inventory_hash", None)
    if inventory.get("schema_version") != RAW_SPLIT_INVENTORY_VERSION:
        raise ValueError("unsupported v6 raw split inventory")
    if not stored or stored != canonical_json_hash(unsigned):
        raise ValueError("raw split inventory hash is invalid")
    return stored


def _split_rows(transitions, assignments, split_name):
    rows = []
    for episode in group_transitions_into_episodes(transitions):
        episode_id = str(episode[0].get("episode_id"))
        if assignments.get(episode_id) == split_name:
            rows.extend({**row, "dataset_split": split_name} for row in episode)
    return rows


def _authenticate_split(rows, split_name, manifest, expected_sha256=None):
    if expected_sha256 is not None:
        raise TypeError("use _authenticate_split_file for byte-hash validation")
    if not rows:
        raise ValueError(f"{split_name} split is empty")
    if any(row.get("dataset_split") != split_name for row in rows):
        raise ValueError(f"{split_name} input contains a relabelled transition")
    episodes = group_transitions_into_episodes(rows)
    episode_ids = [str(episode[0].get("episode_id")) for episode in episodes if episode]
    expected_ids = sorted(
        episode_id for episode_id, assigned in manifest["assignments"].items()
        if assigned == split_name
    )
    if sorted(episode_ids) != expected_ids or len(episode_ids) != len(set(episode_ids)):
        raise ValueError(f"{split_name} episode membership is not exact")
    summary = manifest["summary"][split_name]
    if len(rows) != int(summary["transitions"]):
        raise ValueError(f"{split_name} transition count changed")
    if len(episodes) != int(summary["episodes"]):
        raise ValueError(f"{split_name} episode count changed")
    if len(connected_identity_components(rows)) != int(summary["identity_components"]):
        raise ValueError(f"{split_name} component count changed")
    return rows


def _authenticate_split_file(path, split_name, manifest, inventory):
    record = inventory.get("artifacts", {}).get(
        "locked_test" if split_name == "test" else split_name, {},
    )
    expected = record.get("sha256") if isinstance(record, dict) else record
    if expected != file_sha256(path):
        raise ValueError(f"{split_name} split byte hash mismatch")
    return _authenticate_split(_load(path), split_name, manifest)


def freeze_development_splits_v6(
    raw_file,
    parent_v3_manifest_file,
    parent_v5_protocol_file,
    parent_v5_report_file,
    parent_v5_manifest_file,
    parent_v5_inventory_file,
    output_dir,
    *,
    dependency_lock_path=None,
    seed=42,
    expected_counts=None,
):
    if seed != 42:
        raise ValueError("calibration v6 requires split seed 42")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"calibration v6 output already exists: {output}")
    raw_path = Path(raw_file).resolve()
    if output == raw_path or raw_path in output.parents:
        raise ValueError("v6 output overlaps the raw corpus path")
    lineage = validate_parent_v5_lineage(
        parent_v5_protocol_file, parent_v5_report_file,
        parent_v5_manifest_file, parent_v5_inventory_file,
    )
    before = raw_path.read_bytes()
    transitions = json.loads(before.decode("utf-8"))
    parent_v3 = _load(parent_v3_manifest_file)
    if parent_v3.get("schema_version") != "aria-split-manifest-v3":
        raise ValueError("v6 freeze requires an aria-split-manifest-v3 parent")
    unsigned_v3 = dict(parent_v3)
    parent_v3_hash = unsigned_v3.pop("manifest_hash", None)
    if not parent_v3_hash or parent_v3_hash != canonical_json_hash(unsigned_v3):
        raise ValueError("parent v3 manifest hash is invalid")
    if parent_v3.get("raw_dataset_hash") != canonical_json_hash(transitions):
        raise ValueError("parent v3 manifest raw hash mismatch")
    assignments = dict(lineage["manifest"]["assignments"])
    if set(assignments) != {
        str(episode[0].get("episode_id"))
        for episode in group_transitions_into_episodes(transitions) if episode
    }:
        raise ValueError("v5 assignments do not exactly cover the raw corpus")
    manifest = {
        "schema_version": "aria-split-manifest-v4",
        "parent_manifest_hash": parent_v3_hash,
        "raw_dataset_hash": canonical_json_hash(transitions),
        "seed": 42,
        "migration_algorithm_version": "reuse-validated-v5-assignment-v1",
        "requested_component_counts": [21, 6, 6],
        "actual_component_counts": [
            len(connected_identity_components(_split_rows(transitions, assignments, name)))
            for name in ("train", "validation", "test")
        ],
        "assignments": assignments,
        "summary": lineage["manifest"]["summary"],
        "locked_test_assignment_hash": lineage["manifest"]["locked_test_assignment_hash"],
        "locked_test_assignment_preserved": True,
    }
    if manifest["actual_component_counts"] != [21, 6, 6]:
        raise ValueError("v6 assignments do not contain 21/6/6 components")
    if assignments != lineage["manifest"]["assignments"]:
        raise ValueError("v6 assignments differ from v5")

    staging = output.with_name(f"{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    split_paths = {}
    for split_name in ("train", "validation", "test"):
        relative = Path("raw-splits") / ("locked" if split_name == "test" else "") / f"{split_name}.json"
        split_paths[split_name] = staging / relative
        atomic_json_write(split_paths[split_name], _split_rows(transitions, assignments, split_name))
    parent_artifacts = lineage["inventory"].get("artifacts", {})
    for split_name in ("train", "validation", "test"):
        key = "locked_test" if split_name == "test" else split_name
        parent_value = parent_artifacts.get(key)
        parent_hash = parent_value.get("sha256") if isinstance(parent_value, dict) else parent_value
        if parent_hash and file_sha256(split_paths[split_name]) != parent_hash:
            raise ValueError(f"v6 {split_name} split bytes differ from v5")

    lock_path = dependency_lock_path or Path(__file__).resolve().parents[2] / "requirements.txt"
    protocol = freeze_calibration_protocol_v6(
        raw_path, manifest, staging, lock_path,
        lineage=lineage, expected_counts=expected_counts,
        artifact_root=output,
    )
    manifest.update({
        "producer_version": CALIBRATION_ALGORITHM_VERSION,
        "supported_consumer_versions": ["aria-split-manifest-v4"],
        "protocol_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"],
        "split_assignment_hash": protocol["split_assignment_hash"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
        "git_commit": protocol["code_commit"],
        "parent_artifact_hashes": {
            "split_manifest_v3": parent_v3_hash,
            "split_manifest_v5": lineage["manifest_hash"],
        },
    })
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    manifest_path = staging / "manifests" / "split_manifest_v4.json"
    atomic_json_write(manifest_path, manifest)
    inventory = {
        "schema_version": RAW_SPLIT_INVENTORY_VERSION,
        "producer_version": CALIBRATION_ALGORITHM_VERSION,
        "supported_consumer_versions": [RAW_SPLIT_INVENTORY_VERSION],
        "protocol_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"],
        "raw_dataset_hash": protocol["raw_dataset_hash"],
        "split_manifest_hash": manifest["manifest_hash"],
        "split_assignment_hash": protocol["split_assignment_hash"],
        "parent_artifact_hashes": {"raw_split_inventory_v5": lineage["inventory_hash"]},
        "git_commit": protocol["code_commit"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
        "artifacts": {
            "train": {"path": str(output / "raw-splits" / "train.json"), "sha256": file_sha256(split_paths["train"])},
            "validation": {"path": str(output / "raw-splits" / "validation.json"), "sha256": file_sha256(split_paths["validation"])},
            "locked_test": {"path": str(output / "raw-splits" / "locked" / "test.json"), "sha256": file_sha256(split_paths["test"])},
            "split_manifest": {"path": str(output / "manifests" / "split_manifest_v4.json"), "sha256": file_sha256(manifest_path)},
        },
    }
    inventory["inventory_hash"] = canonical_json_hash(inventory)
    atomic_json_write(staging / "manifests" / "raw_split_inventory_v3.json", inventory)
    for directory in ("calibration", "replayed", "audits", "release"):
        (staging / directory).mkdir(parents=True, exist_ok=True)
    if raw_path.read_bytes() != before:
        raise RuntimeError("raw dataset changed while freezing v6")
    validate_calibration_protocol_v6(
        staging / "protocol" / "calibration_protocol_v6.json",
        raw_file=raw_path, split_manifest=manifest,
    )
    validate_protocol_state(
        staging / "protocol" / "calibration_protocol_state_v1.json",
        protocol=protocol,
    )
    _validate_inventory(inventory)
    staging.replace(output)
    return manifest


def prepare_development_calibration_v6(
    train_file,
    validation_file,
    manifest_file,
    inventory_file,
    protocol_file,
    state_file,
    parent_v5_report_file,
    output_dir,
    *,
    bootstrap_samples=1000,
):
    output = Path(output_dir).resolve()
    protocol = validate_calibration_protocol_v6(protocol_file, split_manifest=manifest_file)
    if output != Path(protocol["artifact_root"]).resolve():
        raise ValueError("output directory differs from the frozen artifact root")
    state = validate_protocol_state(state_file, protocol=protocol)
    if state["current_status"] != "FROZEN_FOR_DEVELOPMENT":
        raise ValueError("protocol is not frozen for development")
    attempt_path = output / "protocol" / "validation_attempt_v1.json"
    if attempt_path.exists():
        existing_attempt = _load(attempt_path)
        unsigned_attempt = dict(existing_attempt)
        stored_attempt_hash = unsigned_attempt.pop("attempt_hash", None)
        attempt_reference = (
            stored_attempt_hash
            if stored_attempt_hash == canonical_json_hash(unsigned_attempt)
            else file_sha256(attempt_path)
        )
        update_protocol_state(
            state_file, "FAILED", protocol=protocol,
            validation_executions=1,
            last_attempt_hash=attempt_reference,
            belief_config_hash=existing_attempt.get("belief_config_hash"),
            failure_reason="existing-validation-attempt",
        )
        raise FileExistsError("validation was already attempted for calibration protocol v6")
    manifest = _load(manifest_file)
    manifest_hash = _validate_manifest(manifest)
    inventory = _load(inventory_file)
    _validate_inventory(inventory)
    inventory_expected = {
        "protocol_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"],
        "raw_dataset_hash": protocol["raw_dataset_hash"],
        "split_manifest_hash": manifest_hash,
        "split_assignment_hash": protocol["split_assignment_hash"],
    }
    for field, expected in inventory_expected.items():
        if inventory.get(field) != expected:
            raise ValueError(f"inventory {field} mismatch")
    if manifest.get("protocol_hash") != protocol["protocol_hash"]:
        raise ValueError("split manifest protocol hash mismatch")
    if manifest.get("raw_file_sha256") != protocol["raw_file_sha256"]:
        raise ValueError("split manifest raw file hash mismatch")
    training = _authenticate_split_file(train_file, "train", manifest, inventory)
    parent_report = _load(parent_v5_report_file)
    unsigned_parent_report = dict(parent_report)
    stored_parent_report_hash = unsigned_parent_report.pop("report_hash", None)
    if (
        stored_parent_report_hash != canonical_json_hash(unsigned_parent_report)
        or stored_parent_report_hash != protocol["parent_v5_report_hash"]
        or parent_report.get("schema_version") != "aria-calibration-cv-report-v2"
        or parent_report.get("selection_status") != "FAILED"
    ):
        raise ValueError("parent v5 report no longer matches the frozen v6 lineage")
    selection = select_training_only_calibration_v6(
        training, protocol["raw_dataset_hash"], manifest_hash,
        protocol["protocol_hash"], parent_v5_report=parent_report,
    )
    serializable = dict(selection)
    if isinstance(serializable.get("config"), BeliefModelConfig):
        serializable["config"] = serializable["config"].to_dict()
    serializable.update({
        "raw_file_sha256": protocol["raw_file_sha256"],
        "split_assignment_hash": protocol["split_assignment_hash"],
        "parent_artifact_hashes": {
            "parent_v5_report": protocol["parent_v5_report_hash"],
            "split_manifest": manifest_hash,
        },
        "git_commit": protocol["code_commit"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
    })
    serializable.pop("report_hash", None)
    serializable["report_hash"] = canonical_json_hash(serializable)
    selection_hash = serializable["report_hash"]
    cv_path = output / "calibration" / "training_cv_report_v3.json"
    atomic_json_write(cv_path, serializable)
    if selection["selection_status"] != "ELIGIBLE":
        updated = update_protocol_state(
            state_file, "FAILED", protocol=protocol,
            failure_reason=selection["selection_status"],
        )
        return {
            "schema_version": DEVELOPMENT_REPORT_VERSION,
            "protocol_hash": protocol["protocol_hash"],
            "state_hash": updated["state_hash"],
            "selection_status": selection["selection_status"],
            "validation_evaluated": False,
            "test_metrics_locked": True,
        }

    config = selection["config"]
    metadata = dict(config.fit_metadata)
    metadata.update({
        "protocol_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"],
        "split_assignment_hash": protocol["split_assignment_hash"],
        "parent_v5_report_hash": protocol["parent_v5_report_hash"],
        "training_cv_report_hash": selection_hash,
        "producer_version": CALIBRATION_ALGORITHM_VERSION,
        "supported_consumer_versions": ["belief-v2"],
    })
    config = config.with_updates(fit_metadata=metadata)
    config_path = output / "calibration" / "belief_model_v2.json"
    config.save(config_path)
    if BeliefModelConfig.load(config_path).config_hash != config.config_hash:
        raise RuntimeError("saved belief configuration hash changed")

    attempt = {
        "schema_version": VALIDATION_ATTEMPT_VERSION,
        "producer_version": CALIBRATION_ALGORITHM_VERSION,
        "protocol_hash": protocol["protocol_hash"],
        "state": "STARTED",
        "attempt_number": 1,
        "belief_config_hash": config.config_hash,
        "expected_validation_file_sha256": inventory["artifacts"]["validation"]["sha256"],
    }
    attempt["attempt_hash"] = canonical_json_hash(attempt)
    exclusive_json_create(attempt_path, attempt)
    try:
        validation = _authenticate_split_file(
            validation_file, "validation", manifest, inventory,
        )
        validation_result = validate_selected_calibration(
            validation, config, bootstrap_samples=bootstrap_samples,
        )
        replayed_paths = {}
        for split_name, rows in (("train", training), ("validation", validation)):
            replayed = []
            for episode in group_transitions_into_episodes(rows):
                replayed.extend(replay_one_episode(
                    episode, config, protocol["raw_dataset_hash"], manifest_hash,
                ))
            for row in replayed:
                row.update({
                    "dataset_split": split_name,
                    "schema_version": REPLAY_SCHEMA_VERSION,
                    "producer_version": CALIBRATION_ALGORITHM_VERSION,
                    "supported_consumer_versions": [REPLAY_SCHEMA_VERSION],
                    "protocol_hash": protocol["protocol_hash"],
                    "raw_file_sha256": protocol["raw_file_sha256"],
                    "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
                    "split_assignment_hash": protocol["split_assignment_hash"],
                    "git_commit": protocol["code_commit"],
                    "parent_artifact_hashes": {
                        "split_manifest": manifest_hash,
                        "belief_config": config.config_hash,
                    },
                    "belief_config_hash": config.config_hash,
                })
            replayed_paths[split_name] = output / "replayed" / f"{split_name}.json"
            atomic_json_write(replayed_paths[split_name], replayed)

        report = {
            "schema_version": DEVELOPMENT_REPORT_VERSION,
            "producer_version": CALIBRATION_ALGORITHM_VERSION,
            "supported_consumer_versions": [DEVELOPMENT_REPORT_VERSION],
            "protocol_hash": protocol["protocol_hash"],
            "raw_file_sha256": protocol["raw_file_sha256"],
            "raw_dataset_hash": protocol["raw_dataset_hash"],
            "split_manifest_hash": manifest_hash,
            "split_assignment_hash": protocol["split_assignment_hash"],
            "belief_config_hash": config.config_hash,
            "parent_artifact_hashes": {
                "training_cv_report": selection_hash,
                "parent_v5_report": protocol["parent_v5_report_hash"],
            },
            "git_commit": protocol["code_commit"],
            "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
            "validation_used_for_selection": False,
            "validation_execution_count": 1,
            "test_metrics_locked": True,
            "calibration": {
                "candidate_count": selection["candidate_count"],
                "selection_report_hash": selection_hash,
                "validation": validation_result,
            },
        }
        report["report_hash"] = canonical_json_hash(report)
        report_path = output / "calibration" / "development_calibration_report_v6.json"
        atomic_json_write(report_path, report)
        artifacts = {
            "split_manifest": {"path": str(Path(manifest_file).resolve()), "sha256": file_sha256(manifest_file)},
            "replayed_train": {"path": str(replayed_paths["train"]), "sha256": file_sha256(replayed_paths["train"]), "split": "train"},
            "replayed_validation": {"path": str(replayed_paths["validation"]), "sha256": file_sha256(replayed_paths["validation"]), "split": "validation"},
            "belief_config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "validation_report": {"path": str(report_path), "sha256": file_sha256(report_path)},
        }
        bundle = {
            "schema_version": DEVELOPMENT_BUNDLE_VERSION,
            "producer_version": CALIBRATION_ALGORITHM_VERSION,
            "supported_consumer_versions": [DEVELOPMENT_BUNDLE_VERSION],
            "protocol_hash": protocol["protocol_hash"],
            "raw_file_sha256": protocol["raw_file_sha256"],
            "raw_dataset_hash": protocol["raw_dataset_hash"],
            "split_manifest_hash": manifest_hash,
            "split_assignment_hash": protocol["split_assignment_hash"],
            "belief_config_hash": config.config_hash,
            "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
            "parent_artifact_hashes": {
                "protocol": protocol["protocol_hash"],
                "training_cv_report": selection_hash,
            },
            "git_commit": protocol["code_commit"],
            "artifacts": artifacts,
        }
        bundle["bundle_hash"] = canonical_json_hash(bundle)
        atomic_json_write(output / "manifests" / "development_bundle_v3.json", bundle)
        attempt["state"] = "COMPLETED"
        attempt["passes_quality_gates"] = validation_result["passes_quality_gates"]
        attempt.pop("attempt_hash", None)
        attempt["attempt_hash"] = canonical_json_hash(attempt)
        atomic_json_write(attempt_path, attempt)
        final_status = "PROVISIONAL_SYNTHETIC" if validation_result["passes_quality_gates"] else "FAILED"
        updated = update_protocol_state(
            state_file, final_status, protocol=protocol,
            validation_executions=1,
            last_attempt_hash=attempt["attempt_hash"],
            belief_config_hash=config.config_hash,
        )
        report["state_hash"] = updated["state_hash"]
        report["development_bundle_hash"] = bundle["bundle_hash"]
        return report
    except BaseException as exc:
        attempt["state"] = "INTERRUPTED"
        attempt["failure_type"] = type(exc).__name__
        attempt.pop("attempt_hash", None)
        attempt["attempt_hash"] = canonical_json_hash(attempt)
        atomic_json_write(attempt_path, attempt)
        current = validate_protocol_state(state_file, protocol=protocol)
        if current["current_status"] == "FROZEN_FOR_DEVELOPMENT":
            update_protocol_state(
                state_file, "FAILED", protocol=protocol,
                validation_executions=1,
                last_attempt_hash=attempt["attempt_hash"],
                belief_config_hash=config.config_hash,
            )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="ARIA calibration protocol v6")
    parser.add_argument("output_dir")
    parser.add_argument("--raw-file")
    parser.add_argument("--base-split-manifest")
    parser.add_argument("--parent-v5-protocol")
    parser.add_argument("--parent-v5-report")
    parser.add_argument("--parent-v5-split-manifest")
    parser.add_argument("--parent-v5-inventory")
    parser.add_argument("--development-train")
    parser.add_argument("--development-validation")
    parser.add_argument("--split-manifest")
    parser.add_argument("--inventory")
    parser.add_argument("--protocol")
    parser.add_argument("--protocol-state")
    parser.add_argument("--dependency-lock")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args(argv)
    calibrating = any((
        args.development_train, args.development_validation, args.split_manifest,
        args.inventory, args.protocol, args.protocol_state,
    ))
    if calibrating:
        required = (
            args.development_train, args.development_validation, args.split_manifest,
            args.inventory, args.protocol, args.protocol_state, args.parent_v5_report,
        )
        if not all(required):
            parser.error("v6 calibration requires all development and protocol inputs")
        result = prepare_development_calibration_v6(
            args.development_train, args.development_validation,
            args.split_manifest, args.inventory, args.protocol,
            args.protocol_state, args.parent_v5_report, args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
        )
    else:
        required = (
            args.raw_file, args.base_split_manifest, args.parent_v5_protocol,
            args.parent_v5_report, args.parent_v5_split_manifest,
            args.parent_v5_inventory,
        )
        if not all(required):
            parser.error("v6 freeze requires raw, v3 manifest, and all parent-v5 artifacts")
        result = freeze_development_splits_v6(
            args.raw_file, args.base_split_manifest, args.parent_v5_protocol,
            args.parent_v5_report, args.parent_v5_split_manifest,
            args.parent_v5_inventory, args.output_dir,
            dependency_lock_path=args.dependency_lock,
        )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
