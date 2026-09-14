"""Prepare calibration v7 without consuming a new validation decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.belief_calibration import select_training_only_calibration_v7
from modules.module_07_rl.calibration_protocol import atomic_json_write, canonical_json_hash, file_sha256
from modules.module_07_rl.calibration_protocol_v7 import (
    CALIBRATION_ALGORITHM_VERSION, DEVELOPMENT_BUNDLE_VERSION,
    DEVELOPMENT_REPORT_VERSION, PREPARATION_ATTEMPT_VERSION,
    exclusive_json_create, freeze_calibration_protocol_v7,
    update_protocol_state_v7, validate_calibration_protocol_v7,
    validate_protocol_state_v7,
)
from modules.module_07_rl.dataset_split import connected_identity_components, group_transitions_into_episodes
from modules.module_07_rl.replay_dataset import REPLAY_SCHEMA_VERSION, replay_one_episode


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_artifact_hash(value, field, description):
    unsigned = dict(value)
    stored = unsigned.pop(field, None)
    if not stored or stored != canonical_json_hash(unsigned):
        raise ValueError(f"{description} hash is invalid")
    return stored


def _authenticate_split_file(path, split_name, manifest, inventory):
    key = "locked_test" if split_name == "test" else split_name
    record = inventory.get("artifacts", {}).get(key, {})
    expected = record.get("sha256") if isinstance(record, dict) else record
    if expected != file_sha256(path):
        raise ValueError(f"{split_name} split byte hash mismatch")
    rows = _load(path)
    if not rows or any(row.get("dataset_split") != split_name for row in rows):
        raise ValueError(f"{split_name} split is empty or relabelled")
    episodes = group_transitions_into_episodes(rows)
    actual_ids = sorted(str(episode[0].get("episode_id")) for episode in episodes if episode)
    expected_ids = sorted(key for key, value in manifest["assignments"].items() if value == split_name)
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise ValueError(f"{split_name} episode membership is not exact")
    summary = manifest["summary"][split_name]
    if len(rows) != int(summary["transitions"]) or len(episodes) != int(summary["episodes"]):
        raise ValueError(f"{split_name} counts changed")
    if len(connected_identity_components(rows)) != int(summary["identity_components"]):
        raise ValueError(f"{split_name} component count changed")
    return rows


def _write_replay(rows, split_name, config, protocol, manifest_hash, output):
    replayed = []
    for episode in group_transitions_into_episodes(rows):
        replayed.extend(replay_one_episode(
            episode, config, protocol["raw_dataset_hash"], manifest_hash,
        ))
    for row in replayed:
        row.update({
            "dataset_split": split_name, "schema_version": REPLAY_SCHEMA_VERSION,
            "producer_version": CALIBRATION_ALGORITHM_VERSION,
            "supported_consumer_versions": [REPLAY_SCHEMA_VERSION],
            "protocol_hash": protocol["protocol_hash"],
            "raw_file_sha256": protocol["raw_file_sha256"],
            "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
            "split_assignment_hash": protocol["split_assignment_hash"],
            "git_commit": protocol["code_commit"],
            "parent_artifact_hashes": {
                "split_manifest": manifest_hash, "belief_config": config.config_hash,
                "parent_v6_belief_config": protocol["parent_v6_belief_config_hash"],
            },
            "belief_config_hash": config.config_hash,
        })
    path = output / "replayed" / f"{split_name}.json"
    atomic_json_write(path, replayed)
    return path


def prepare_development_calibration_v7(
    train_file, validation_file, split_manifest_file, raw_split_inventory_file,
    parent_v6_report_file, parent_v6_config_file, protocol_file, state_file,
    output_dir,
):
    output = Path(output_dir).resolve()
    protocol = validate_calibration_protocol_v7(protocol_file, split_manifest=split_manifest_file)
    if output != Path(protocol["artifact_root"]).resolve():
        raise ValueError("output directory differs from frozen artifact root")
    state = validate_protocol_state_v7(state_file, protocol=protocol)
    if state["current_status"] != "FROZEN_FOR_DEVELOPMENT":
        raise ValueError("v7 protocol is not frozen for development")
    attempt_path = output / "protocol" / "preparation_attempt_v1.json"
    if attempt_path.exists():
        raise FileExistsError("v7 preparation was already attempted")

    manifest = _load(split_manifest_file)
    manifest_hash = _canonical_artifact_hash(manifest, "manifest_hash", "split manifest")
    if manifest_hash != protocol["split_manifest_hash"]:
        raise ValueError("split manifest hash mismatch")
    inventory = _load(raw_split_inventory_file)
    inventory_hash = _canonical_artifact_hash(inventory, "inventory_hash", "raw split inventory")
    if inventory_hash != protocol["parent_v6_inventory_hash"]:
        raise ValueError("parent raw split inventory hash mismatch")
    training = _authenticate_split_file(train_file, "train", manifest, inventory)

    parent_report = _load(parent_v6_report_file)
    if _canonical_artifact_hash(parent_report, "report_hash", "parent v6 report") != protocol["parent_v6_report_hash"]:
        raise ValueError("parent v6 report hash mismatch")
    parent_config = BeliefModelConfig.load(parent_v6_config_file)
    if parent_config.config_hash != protocol["parent_v6_belief_config_hash"]:
        raise ValueError("parent v6 belief configuration hash mismatch")
    selection = select_training_only_calibration_v7(
        training, parent_report, parent_config, protocol["raw_dataset_hash"],
        manifest_hash, protocol["protocol_hash"],
    )
    config = selection.get("config")
    if isinstance(config, BeliefModelConfig):
        metadata = dict(config.fit_metadata)
        metadata.update({
            "protocol_hash": protocol["protocol_hash"],
            "raw_file_sha256": protocol["raw_file_sha256"],
            "split_assignment_hash": protocol["split_assignment_hash"],
            "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
            "producer_version": CALIBRATION_ALGORITHM_VERSION,
            "supported_consumer_versions": ["belief-v3"],
        })
        config = config.with_updates(fit_metadata=metadata)
        selection["config"] = config
        selection["belief_config_hash"] = config.config_hash
    serializable = dict(selection)
    if isinstance(serializable.get("config"), BeliefModelConfig):
        serializable["config"] = serializable["config"].to_dict()
    serializable.update({
        "raw_file_sha256": protocol["raw_file_sha256"],
        "split_assignment_hash": protocol["split_assignment_hash"],
        "git_commit": protocol["code_commit"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
        "parent_artifact_hashes": {
            "parent_v6_protocol": protocol["parent_v6_protocol_hash"],
            "parent_v6_report": protocol["parent_v6_report_hash"],
            "parent_v6_belief_config": protocol["parent_v6_belief_config_hash"],
            "split_manifest": manifest_hash,
        },
    })
    serializable.pop("report_hash", None)
    serializable["report_hash"] = canonical_json_hash(serializable)
    cv_path = output / "calibration" / "training_cv_report_v4.json"
    atomic_json_write(cv_path, serializable)
    if selection["selection_status"] != "ELIGIBLE":
        updated = update_protocol_state_v7(
            state_file, "FAILED", protocol=protocol,
            failure_reason="training-only-selection-failed",
        )
        return {
            "schema_version": DEVELOPMENT_REPORT_VERSION,
            "selection_status": "FAILED", "validation_evaluated": False,
            "locked_test_metrics_produced": False, "state_hash": updated["state_hash"],
        }

    config_path = output / "calibration" / "belief_model_v3.json"
    config.save(config_path)
    if BeliefModelConfig.load(config_path).config_hash != config.config_hash:
        raise RuntimeError("saved v7 belief configuration hash changed")

    attempt = {
        "schema_version": PREPARATION_ATTEMPT_VERSION,
        "producer_version": CALIBRATION_ALGORITHM_VERSION,
        "protocol_hash": protocol["protocol_hash"], "state": "STARTED",
        "attempt_number": 1, "belief_config_hash": config.config_hash,
        "validation_role": "previously-observed-training-early-stopping-only",
        "expected_validation_file_sha256": inventory["artifacts"]["validation"]["sha256"],
    }
    attempt["attempt_hash"] = canonical_json_hash(attempt)
    exclusive_json_create(attempt_path, attempt)
    try:
        # Validation was observed in v6. It is read only after training-only
        # selection and is never scored as a v7 calibration decision.
        validation = _authenticate_split_file(validation_file, "validation", manifest, inventory)
        replayed_train = _write_replay(training, "train", config, protocol, manifest_hash, output)
        replayed_validation = _write_replay(validation, "validation", config, protocol, manifest_hash, output)
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
            "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
            "git_commit": protocol["code_commit"],
            "parent_artifact_hashes": {"training_cv_report": serializable["report_hash"]},
            "selection_status": "ELIGIBLE", "validation_used_for_selection": False,
            "validation_evaluated": False, "validation_execution_count": 0,
            "validation_role": "previously-observed-training-early-stopping-only",
            "clean_final_evaluation_source": "locked-test-only",
            "test_metrics_locked": True,
        }
        report["report_hash"] = canonical_json_hash(report)
        report_path = output / "calibration" / "development_calibration_report_v7.json"
        atomic_json_write(report_path, report)
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
            "git_commit": protocol["code_commit"],
            "parent_artifact_hashes": {
                "protocol": protocol["protocol_hash"], "training_cv_report": serializable["report_hash"],
                "parent_v6_inventory": inventory_hash,
            },
            "validation_role": "previously-observed-training-early-stopping-only",
            "artifacts": {
                "split_manifest": {"path": str(Path(split_manifest_file).resolve()), "sha256": file_sha256(split_manifest_file)},
                "replayed_train": {"path": str(replayed_train), "sha256": file_sha256(replayed_train), "split": "train"},
                "replayed_validation": {"path": str(replayed_validation), "sha256": file_sha256(replayed_validation), "split": "validation"},
                "belief_config": {"path": str(config_path), "sha256": file_sha256(config_path)},
                "development_report": {"path": str(report_path), "sha256": file_sha256(report_path)},
            },
        }
        bundle["bundle_hash"] = canonical_json_hash(bundle)
        atomic_json_write(output / "manifests" / "development_bundle_v4.json", bundle)
        attempt.update({"state": "COMPLETED", "development_bundle_hash": bundle["bundle_hash"]})
        attempt.pop("attempt_hash", None)
        attempt["attempt_hash"] = canonical_json_hash(attempt)
        atomic_json_write(attempt_path, attempt)
        updated = update_protocol_state_v7(
            state_file, "PROVISIONAL_TRAINING_CV", protocol=protocol,
            preparation_executions=1, belief_config_hash=config.config_hash,
            last_attempt_hash=attempt["attempt_hash"],
        )
        return {**report, "state_hash": updated["state_hash"], "development_bundle_hash": bundle["bundle_hash"]}
    except BaseException as exc:
        attempt.update({"state": "INTERRUPTED", "failure_type": type(exc).__name__})
        attempt.pop("attempt_hash", None)
        attempt["attempt_hash"] = canonical_json_hash(attempt)
        atomic_json_write(attempt_path, attempt)
        current = validate_protocol_state_v7(state_file, protocol=protocol)
        if current["current_status"] == "FROZEN_FOR_DEVELOPMENT":
            update_protocol_state_v7(
                state_file, "FAILED", protocol=protocol, preparation_executions=1,
                belief_config_hash=config.config_hash, last_attempt_hash=attempt["attempt_hash"],
            )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="ARIA calibration protocol v7")
    parser.add_argument("output_dir")
    parser.add_argument("--parent-v6-protocol")
    parser.add_argument("--parent-v6-state")
    parser.add_argument("--parent-v6-report")
    parser.add_argument("--parent-v6-config")
    parser.add_argument("--split-manifest")
    parser.add_argument("--raw-split-inventory")
    parser.add_argument("--dependency-lock")
    parser.add_argument("--development-train")
    parser.add_argument("--development-validation")
    parser.add_argument("--protocol")
    parser.add_argument("--protocol-state")
    args = parser.parse_args(argv)
    if args.development_train or args.development_validation or args.protocol or args.protocol_state:
        required = (args.development_train, args.development_validation, args.split_manifest, args.raw_split_inventory, args.parent_v6_report, args.parent_v6_config, args.protocol, args.protocol_state)
        if not all(required):
            parser.error("v7 preparation requires all development, parent, and protocol inputs")
        result = prepare_development_calibration_v7(
            args.development_train, args.development_validation, args.split_manifest,
            args.raw_split_inventory, args.parent_v6_report, args.parent_v6_config,
            args.protocol, args.protocol_state, args.output_dir,
        )
    else:
        required = (args.parent_v6_protocol, args.parent_v6_state, args.parent_v6_report, args.parent_v6_config, args.split_manifest, args.raw_split_inventory)
        if not all(required):
            parser.error("v7 freeze requires all parent-v6 lineage artifacts")
        lock = args.dependency_lock or Path(__file__).resolve().parents[2] / "requirements.txt"
        result = freeze_calibration_protocol_v7(
            args.parent_v6_protocol, args.parent_v6_state, args.parent_v6_report,
            args.parent_v6_config, args.split_manifest, args.raw_split_inventory,
            args.output_dir, lock,
        )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
