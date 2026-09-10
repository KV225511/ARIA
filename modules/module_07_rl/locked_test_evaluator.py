"""One-attempt locked-test release evaluator for calibration protocol v4.

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
)
from modules.module_07_rl.calibration_protocol import (
    atomic_json_write,
    canonical_json_hash,
    file_sha256,
    update_protocol_status,
    validate_calibration_protocol,
)
from modules.module_07_rl.dataset_split import (
    connected_identity_components,
    group_transitions_into_episodes,
)


LOCKED_TEST_EVALUATION_VERSION = "aria-locked-test-evaluation-v1"
RELEASE_ATTEMPT_VERSION = "aria-release-attempt-v1"


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
) -> dict:
    """Validate frozen inputs, consume the one attempt, then read the test split."""
    release_dir = Path(output_dir) / "release"
    attempt_path = release_dir / "release_attempt_v1.json"
    if attempt_path.exists():
        raise FileExistsError(
            f"release attempt already exists and cannot be repeated: {attempt_path}"
        )
    protocol_path = Path(protocol)
    checked_protocol = validate_calibration_protocol(protocol_path)
    if checked_protocol["protocol_status"] != "PROVISIONAL_SYNTHETIC":
        raise ValueError("locked test requires PROVISIONAL_SYNTHETIC protocol status")
    config = BeliefModelConfig.load(belief_config)
    if config.schema_version != "belief-v2":
        raise ValueError("unsupported belief configuration schema")
    if config.config_hash != checked_protocol.get("belief_config_hash"):
        raise ValueError("belief configuration hash mismatch")
    manifest = _load_json(split_manifest)
    manifest_hash = _validate_manifest(manifest)
    if config.split_manifest_hash != manifest_hash:
        raise ValueError("belief configuration split manifest hash mismatch")
    if manifest.get("raw_dataset_hash") != checked_protocol["raw_dataset_hash"]:
        raise ValueError("split manifest raw dataset hash mismatch")
    if manifest.get("locked_test_assignment_hash") != checked_protocol["locked_test_assignment_hash"]:
        raise ValueError("locked-test assignment hash mismatch")

    attempt = {
        "schema_version": RELEASE_ATTEMPT_VERSION,
        "producer_version": "aria-belief-calibration-v4",
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
        passed, reasons = _gate_candidate(metrics, minimum_components=6)
        report = {
            "schema_version": LOCKED_TEST_EVALUATION_VERSION,
            "producer_version": "aria-belief-calibration-v4",
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
        report["report_hash"] = canonical_json_hash(report)
        atomic_json_write(release_dir / "locked_test_evaluation_v1.json", report)
        attempt["state"] = "COMPLETED"
        attempt["evaluation_report_hash"] = report["report_hash"]
        attempt["final_status"] = report["final_status"]
        atomic_json_write(attempt_path, attempt)
        update_protocol_status(protocol_path, report["final_status"])
        return report
    except BaseException as exc:
        attempt["state"] = "FAILED"
        attempt["failure_type"] = type(exc).__name__
        atomic_json_write(attempt_path, attempt)
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
    args = parser.parse_args()
    try:
        result = evaluate_locked_test_once(
            args.locked_test, args.belief_config, args.protocol,
            args.split_manifest, args.output_dir,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"[ERROR] {error}\n")
    print(json.dumps(result, indent=2))
