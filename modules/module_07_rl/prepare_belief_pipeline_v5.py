"""Additive v5 development pipeline with training-only candidate selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.belief_calibration import (
    select_training_only_calibration_v5,
    validate_selected_calibration,
)
from modules.module_07_rl.calibration_protocol import file_sha256
from modules.module_07_rl.calibration_protocol_v5 import (
    CALIBRATION_ALGORITHM_VERSION,
    DEVELOPMENT_BUNDLE_VERSION,
    freeze_calibration_protocol_v5,
    update_protocol_status_v5,
    validate_calibration_protocol_v5,
)
from modules.module_07_rl.dataset_split import group_transitions_into_episodes
from modules.module_07_rl.replay_dataset import (
    REPLAY_SCHEMA_VERSION,
    _atomic_json_write,
    canonical_json_hash,
    migrate_split_manifest,
    replay_one_episode,
)


DEVELOPMENT_REPORT_VERSION = "aria-development-calibration-v5"
RAW_SPLIT_INVENTORY_VERSION = "aria-raw-split-inventory-v2"


def _validated_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "aria-split-manifest-v4":
        raise ValueError("v5 development calibration requires split_manifest_v4")
    unsigned = dict(manifest)
    stored = unsigned.pop("manifest_hash", None)
    if not stored or stored != canonical_json_hash(unsigned):
        raise ValueError("development split manifest hash is invalid")
    return manifest


def freeze_development_splits_v5(
    raw_file,
    parent_manifest_file,
    output_dir,
    *,
    seed=42,
    dependency_lock_path=None,
    expected_counts=None,
    parent_v4_report=None,
):
    """Freeze a fresh v5 artifact tree without changing v4 or raw bytes."""
    output = Path(output_dir)
    protocol_path = output / "protocol" / "calibration_protocol_v5.json"
    if protocol_path.exists():
        raise FileExistsError(f"calibration protocol v5 is already frozen: {protocol_path}")
    raw_path = Path(raw_file)
    before = raw_path.read_bytes()
    transitions = json.loads(before.decode("utf-8"))
    parent = json.loads(Path(parent_manifest_file).read_text(encoding="utf-8"))
    manifest = migrate_split_manifest(transitions, parent, seed=seed)
    assignments = manifest["assignments"]
    split_paths = {}
    for split_name in ("train", "validation", "test"):
        rows = []
        for episode in group_transitions_into_episodes(transitions):
            if episode and assignments[str(episode[0].get("episode_id"))] == split_name:
                rows.extend({**row, "dataset_split": split_name} for row in episode)
        relative = Path("raw-splits") / ("locked" if split_name == "test" else "") / f"{split_name}.json"
        split_paths[split_name] = output / relative
        _atomic_json_write(split_paths[split_name], rows)

    parent_v4_report_hash = file_sha256(parent_v4_report) if parent_v4_report else None
    lock_path = dependency_lock_path or Path(__file__).resolve().parents[2] / "requirements.txt"
    protocol = freeze_calibration_protocol_v5(
        raw_path,
        manifest,
        output,
        lock_path,
        expected_counts=expected_counts,
        parent_v4_report_hash=parent_v4_report_hash,
    )
    manifest.update({
        "producer_version": CALIBRATION_ALGORITHM_VERSION,
        "supported_consumer_versions": ["aria-split-manifest-v4"],
        "protocol_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
        "git_commit": protocol["code_commit"],
        "parent_artifact_hashes": {"split_manifest_v3": manifest["parent_manifest_hash"]},
    })
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    manifest_path = output / "manifests" / "split_manifest_v4.json"
    _atomic_json_write(manifest_path, manifest)

    inventory = {
        "schema_version": RAW_SPLIT_INVENTORY_VERSION,
        "producer_version": CALIBRATION_ALGORITHM_VERSION,
        "supported_consumer_versions": [RAW_SPLIT_INVENTORY_VERSION],
        "protocol_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"],
        "raw_dataset_hash": protocol["raw_dataset_hash"],
        "split_manifest_hash": manifest["manifest_hash"],
        "parent_artifact_hashes": {"raw_corpus": protocol["raw_file_sha256"]},
        "git_commit": protocol["code_commit"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
        "artifacts": {
            "train": file_sha256(split_paths["train"]),
            "validation": file_sha256(split_paths["validation"]),
            "locked_test": file_sha256(split_paths["test"]),
        },
    }
    inventory["inventory_hash"] = canonical_json_hash(inventory)
    _atomic_json_write(output / "manifests" / "raw_split_inventory_v2.json", inventory)
    for directory in ("calibration", "replayed", "audits", "release"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    if raw_path.read_bytes() != before:
        raise RuntimeError("raw dataset changed while freezing v5 development splits")
    return manifest


def prepare_development_calibration_v5(
    train_file,
    validation_file,
    manifest_file,
    protocol_file,
    output_dir,
    *,
    bootstrap_samples=1000,
):
    """Select on train CV, then consume the protocol's sole validation run."""
    manifest = _validated_manifest(manifest_file)
    protocol = validate_calibration_protocol_v5(protocol_file, split_manifest=manifest)
    if protocol["protocol_status"] != "FROZEN_FOR_DEVELOPMENT":
        raise ValueError("protocol is not frozen for a first development validation")
    if int(protocol.get("validation_executions", 0)) != 0:
        raise ValueError("validation has already executed for this protocol")

    training = json.loads(Path(train_file).read_text(encoding="utf-8"))
    if any(row.get("dataset_split") != "train" for row in training):
        raise ValueError("development calibration training input contains a non-train split")
    selection = select_training_only_calibration_v5(
        training,
        raw_dataset_hash=protocol["raw_dataset_hash"],
        split_manifest_hash=manifest["manifest_hash"],
        protocol_hash=protocol["protocol_hash"],
    )
    output = Path(output_dir)
    serializable = dict(selection)
    if isinstance(serializable.get("config"), BeliefModelConfig):
        serializable["config"] = serializable["config"].to_dict()
    serializable.update({
        "raw_file_sha256": protocol["raw_file_sha256"],
        "parent_artifact_hashes": {"split_manifest": manifest["manifest_hash"]},
        "git_commit": protocol["code_commit"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
    })
    serializable.pop("report_hash", None)
    serializable["report_hash"] = canonical_json_hash(serializable)
    selection_hash = serializable["report_hash"]
    _atomic_json_write(output / "calibration" / "training_cv_report_v2.json", serializable)
    if selection["selection_status"] != "ELIGIBLE":
        updated = update_protocol_status_v5(protocol_file, "FAILED")
        return {
            "schema_version": DEVELOPMENT_REPORT_VERSION,
            "protocol_hash": updated["protocol_hash"],
            "selection_status": "FAILED",
            "validation_evaluated": False,
            "test_metrics_locked": True,
        }

    config = selection["config"]
    metadata = dict(config.fit_metadata)
    metadata.update({
        "protocol_freeze_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
        "git_commit": protocol["code_commit"],
        "producer_version": CALIBRATION_ALGORITHM_VERSION,
        "supported_consumer_versions": ["belief-v2"],
        "parent_artifact_hashes": {
            "split_manifest": manifest["manifest_hash"],
            "training_cv_report": selection_hash,
        },
    })
    config = config.with_updates(fit_metadata=metadata)
    config_path = output / "calibration" / "belief_model_v2.json"
    config.save(config_path)

    update_protocol_status_v5(
        protocol_file,
        "FROZEN_FOR_DEVELOPMENT",
        validation_executions=1,
        validation_state="STARTED",
    )
    try:
        validation = json.loads(Path(validation_file).read_text(encoding="utf-8"))
        if any(row.get("dataset_split") != "validation" for row in validation):
            raise ValueError("development validation input contains a non-validation split")
        validation_result = validate_selected_calibration(
            validation,
            config,
            bootstrap_samples=bootstrap_samples,
        )
    except BaseException:
        update_protocol_status_v5(
            protocol_file,
            "FAILED",
            validation_executions=1,
            validation_state="INTERRUPTED",
            belief_config_hash=config.config_hash,
        )
        raise

    passed = validation_result["passes_quality_gates"]
    updated = update_protocol_status_v5(
        protocol_file,
        "PROVISIONAL_SYNTHETIC" if passed else "FAILED",
        validation_executions=1,
        validation_state="COMPLETED",
        belief_config_hash=config.config_hash,
    )
    replayed_paths = {}
    for split_name, transitions in (("train", training), ("validation", validation)):
        rows = []
        for episode in group_transitions_into_episodes(transitions):
            rows.extend(replay_one_episode(
                episode,
                config,
                protocol["raw_dataset_hash"],
                manifest["manifest_hash"],
            ))
        for row in rows:
            row.update({
                "dataset_split": split_name,
                "schema_version": REPLAY_SCHEMA_VERSION,
                "producer_version": CALIBRATION_ALGORITHM_VERSION,
                "supported_consumer_versions": [REPLAY_SCHEMA_VERSION],
                "protocol_hash": updated["protocol_hash"],
                "raw_file_sha256": updated["raw_file_sha256"],
                "environment_fingerprint_hash": updated["environment_fingerprint_hash"],
                "git_commit": updated["code_commit"],
                "parent_artifact_hashes": {
                    "split_manifest": manifest["manifest_hash"],
                    "belief_config": config.config_hash,
                },
                "belief_config_hash": config.config_hash,
            })
        replayed_paths[split_name] = output / "replayed" / f"{split_name}.json"
        _atomic_json_write(replayed_paths[split_name], rows)

    report = {
        "schema_version": DEVELOPMENT_REPORT_VERSION,
        "producer_version": CALIBRATION_ALGORITHM_VERSION,
        "supported_consumer_versions": [DEVELOPMENT_REPORT_VERSION],
        "protocol_hash": updated["protocol_hash"],
        "raw_file_sha256": updated["raw_file_sha256"],
        "raw_dataset_hash": updated["raw_dataset_hash"],
        "split_manifest_hash": manifest["manifest_hash"],
        "belief_config_hash": config.config_hash,
        "parent_artifact_hashes": {
            "training_cv_report": selection_hash,
            "split_manifest": manifest["manifest_hash"],
        },
        "git_commit": updated["code_commit"],
        "environment_fingerprint_hash": updated["environment_fingerprint_hash"],
        "validation_used_for_selection": False,
        "validation_execution_count": 1,
        "test_metrics_locked": True,
        "calibration": {
            "config_hash": config.config_hash,
            "selection_report_hash": selection_hash,
            "candidate_count": selection["candidate_count"],
            "validation": validation_result,
        },
    }
    report_path = output / "calibration" / "development_calibration_report_v5.json"
    _atomic_json_write(report_path, report)
    artifacts = {
        "split_manifest": {"path": str(Path(manifest_file).resolve()), "sha256": file_sha256(manifest_file)},
        "replayed_train": {"path": str(replayed_paths["train"].resolve()), "sha256": file_sha256(replayed_paths["train"]), "split": "train"},
        "replayed_validation": {"path": str(replayed_paths["validation"].resolve()), "sha256": file_sha256(replayed_paths["validation"]), "split": "validation"},
        "belief_config": {"path": str(config_path.resolve()), "sha256": file_sha256(config_path)},
        "validation_report": {"path": str(report_path.resolve()), "sha256": file_sha256(report_path)},
    }
    bundle = {
        "schema_version": DEVELOPMENT_BUNDLE_VERSION,
        "producer_version": CALIBRATION_ALGORITHM_VERSION,
        "supported_consumer_versions": [DEVELOPMENT_BUNDLE_VERSION],
        "protocol_hash": updated["protocol_hash"],
        "raw_file_sha256": updated["raw_file_sha256"],
        "raw_dataset_hash": updated["raw_dataset_hash"],
        "split_manifest_hash": manifest["manifest_hash"],
        "belief_config_hash": config.config_hash,
        "parent_artifact_hashes": {
            "protocol": updated["protocol_hash"],
            "split_manifest": manifest["manifest_hash"],
            "training_cv_report": selection_hash,
        },
        "git_commit": updated["code_commit"],
        "environment_fingerprint_hash": updated["environment_fingerprint_hash"],
        "artifacts": artifacts,
    }
    bundle["bundle_hash"] = canonical_json_hash(bundle)
    _atomic_json_write(output / "manifests" / "development_bundle_v2.json", bundle)
    report["development_bundle_hash"] = bundle["bundle_hash"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="ARIA additive calibration protocol v5")
    parser.add_argument("output_dir")
    parser.add_argument("--raw-file")
    parser.add_argument("--base-split-manifest")
    parser.add_argument("--development-train")
    parser.add_argument("--development-validation")
    parser.add_argument("--split-manifest")
    parser.add_argument("--protocol")
    parser.add_argument("--dependency-lock")
    parser.add_argument("--parent-v4-report")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args(argv)
    calibrating = any((args.development_train, args.development_validation, args.split_manifest, args.protocol))
    if calibrating:
        if not all((args.development_train, args.development_validation, args.split_manifest, args.protocol)):
            parser.error("development calibration requires train, validation, split manifest, and protocol")
        result = prepare_development_calibration_v5(
            args.development_train,
            args.development_validation,
            args.split_manifest,
            args.protocol,
            args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
        )
    else:
        if not args.raw_file or not args.base_split_manifest:
            parser.error("split freeze requires --raw-file and --base-split-manifest")
        result = freeze_development_splits_v5(
            args.raw_file,
            args.base_split_manifest,
            args.output_dir,
            seed=args.split_seed,
            dependency_lock_path=args.dependency_lock,
            parent_v4_report=args.parent_v4_report,
        )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
