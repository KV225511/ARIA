"""Frozen protocol and provenance contracts for ARIA calibration v4."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

from modules.module_07_rl.dataset_audit import (
    CALIBRATION_GATE_THRESHOLDS,
    VALIDATION_GATE_VERSION,
)
from modules.module_07_rl.dataset_split import (
    connected_identity_components,
    group_transitions_into_episodes,
)
from modules.module_07_rl.metrics import METRICS_SCHEMA_VERSION


CALIBRATION_PROTOCOL_VERSION = "aria-calibration-protocol-v4"
ENVIRONMENT_FINGERPRINT_VERSION = "aria-environment-fingerprint-v1"
CALIBRATION_ALGORITHM_VERSION = "aria-belief-calibration-v4"
DEVELOPMENT_BUNDLE_VERSION = "aria-development-bundle-v1"
KNOWN_CANONICAL_DATASET_HASH = (
    "50f9f5da589b0859667310eeed97717414f80e44fd1ee518ec9f953a978daf2c"
)

SUPPORTED_PROTOCOL_STATUSES = {
    "FROZEN_FOR_DEVELOPMENT", "PROVISIONAL_SYNTHETIC",
    "VALIDATED_SYNTHETIC", "FAILED",
}

CALIBRATION_STAGE_SEQUENCE = [
    "training_fit_baseline", "scale_shrinkage",
    "aggregation_and_abstention", "evidence_dependence",
]
CALIBRATION_CANDIDATE_VALUES = {
    "baseline": {
        "repeat_discount_power": [0.50],
        "max_skill_effective_sample_size": [5],
        "aggregation_temperature": [1.00],
        "minimum_assessment_confidence": [0.50],
    },
    "scale_shrinkage": [0.00, 0.25, 0.50, 0.75, 1.00],
    "aggregation_temperature": [0.80, 1.00, 1.20, 1.50, 2.00],
    "minimum_assessment_confidence": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "repeat_discount_power": [0.25, 0.50, 0.75],
    "max_skill_effective_sample_size": [3, 5, 8],
}
NUMERICAL_TOLERANCES = {
    "probability_sum_absolute": 1e-6,
    "floating_comparison_absolute": 1e-12,
    "scale_minimum": 0.03,
    "scale_maximum": 0.35,
}

REQUIRED_PROTOCOL_FIELDS = {
    "protocol_schema_version", "protocol_status", "raw_file_path",
    "raw_file_sha256", "raw_dataset_hash", "episode_count",
    "transition_count", "identity_component_count",
    "parent_split_manifest_hash", "locked_test_assignment_hash",
    "target_component_counts", "metric_schema_version",
    "gate_policy_version", "gate_thresholds", "split_migration_algorithm",
    "cross_validation_algorithm", "calibration_stage_sequence",
    "candidate_values", "candidate_selection_rule", "bootstrap_method",
    "bootstrap_samples", "all_random_seeds", "numerical_tolerances",
    "code_commit", "git_dirty", "dependency_lock_path",
    "dependency_lock_hash", "environment_fingerprint_hash",
    "maximum_validation_executions", "locked_test_policy", "protocol_hash",
}


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def _git_state(repository: str | Path | None = None) -> tuple[str, bool]:
    root = Path(repository or Path(__file__).resolve().parents[2]).resolve()
    prefix = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    try:
        commit = subprocess.run(
            prefix + ["rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=10,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            prefix + ["status", "--porcelain"], check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "unavailable", True


def build_environment_fingerprint(
    dependency_lock_path: str | Path,
    *,
    repository: str | Path | None = None,
    code_commit: str | None = None,
    git_dirty: bool | None = None,
) -> dict:
    """Build a deterministic description of the execution environment."""
    lock_path = Path(dependency_lock_path).resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(f"Dependency lockfile does not exist: {lock_path}")
    detected_commit, detected_dirty = _git_state(repository)
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        deterministic = bool(torch.are_deterministic_algorithms_enabled())
        backend = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch_version, cuda_version, deterministic, backend = (
            "not-installed", None, False, "cpu",
        )
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = "not-installed"
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "not-installed"
    fingerprint = {
        "schema_version": ENVIRONMENT_FINGERPRINT_VERSION,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "pytorch_version": torch_version,
        "cuda_version": cuda_version,
        "numpy_version": numpy_version,
        "scikit_learn_version": sklearn_version,
        "device_backend": backend,
        "deterministic_algorithms_enabled": deterministic,
        "dependency_lock_path": str(lock_path),
        "dependency_lock_hash": file_sha256(lock_path),
        "code_commit": code_commit if code_commit is not None else detected_commit,
        "git_dirty": bool(detected_dirty if git_dirty is None else git_dirty),
    }
    fingerprint["environment_fingerprint_hash"] = canonical_json_hash(fingerprint)
    return fingerprint


def _load_json(value: dict | str | Path) -> dict:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def freeze_calibration_protocol(
    raw_file: str | Path,
    split_manifest: dict | str | Path,
    output_dir: str | Path,
    dependency_lock_path: str | Path,
    *,
    target_component_counts: tuple[int, int, int] = (21, 6, 6),
    bootstrap_samples: int = 1000,
    repository: str | Path | None = None,
    expected_counts: tuple[int, int, int] | None = None,
) -> dict:
    """Freeze and atomically persist protocol and environment documents.

    ``expected_counts`` is ``(episodes, transitions, components)``. Production
    callers should use the default corpus contract; tests may pass fixture counts.
    """
    base = Path(output_dir) / "protocol"
    protocol_path = base / "calibration_protocol_v4.json"
    if protocol_path.exists():
        raise FileExistsError(
            f"calibration protocol v4 is already frozen: {protocol_path}"
        )
    raw_path = Path(raw_file).resolve()
    raw_bytes = raw_path.read_bytes()
    transitions = json.loads(raw_bytes.decode("utf-8"))
    manifest = _load_json(split_manifest)
    manifest_hash = manifest.get("manifest_hash")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_hash", None)
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
    canonical_hash = canonical_json_hash(transitions)
    if expected_counts is None and canonical_hash != KNOWN_CANONICAL_DATASET_HASH:
        raise ValueError("raw corpus canonical JSON hash does not match the production contract")
    if manifest.get("raw_dataset_hash") != canonical_hash:
        raise ValueError("split manifest raw dataset hash does not match raw corpus")
    fingerprint = build_environment_fingerprint(
        dependency_lock_path, repository=repository,
    )
    protocol = {
        "protocol_schema_version": CALIBRATION_PROTOCOL_VERSION,
        "protocol_status": "FROZEN_FOR_DEVELOPMENT",
        "raw_file_path": str(raw_path),
        "raw_file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_dataset_hash": canonical_hash,
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
    }
    protocol["protocol_hash"] = canonical_json_hash(protocol)
    atomic_json_write(base / "environment_fingerprint_v1.json", fingerprint)
    atomic_json_write(protocol_path, protocol)
    return protocol


def validate_environment_fingerprint(value: dict | str | Path) -> dict:
    fingerprint = _load_json(value)
    if fingerprint.get("schema_version") != ENVIRONMENT_FINGERPRINT_VERSION:
        raise ValueError("unsupported environment fingerprint version")
    stored = fingerprint.get("environment_fingerprint_hash")
    unsigned = dict(fingerprint)
    unsigned.pop("environment_fingerprint_hash", None)
    if not stored or stored != canonical_json_hash(unsigned):
        raise ValueError("environment fingerprint hash is invalid")
    return fingerprint


def validate_calibration_protocol(
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
    if protocol["protocol_schema_version"] != CALIBRATION_PROTOCOL_VERSION:
        raise ValueError("unsupported calibration protocol version")
    if protocol["protocol_status"] not in SUPPORTED_PROTOCOL_STATUSES:
        raise ValueError("unsupported calibration protocol status")
    if (
        protocol["protocol_status"] in {"PROVISIONAL_SYNTHETIC", "VALIDATED_SYNTHETIC"}
        and not protocol.get("belief_config_hash")
    ):
        raise ValueError("release-ready protocol is missing belief config hash")
    stored_hash = protocol["protocol_hash"]
    unsigned = dict(protocol)
    unsigned.pop("protocol_hash", None)
    if stored_hash != canonical_json_hash(unsigned):
        raise ValueError("calibration protocol hash is invalid")
    if protocol.get("calibration_algorithm_version") != CALIBRATION_ALGORITHM_VERSION:
        raise ValueError("unsupported calibration algorithm version")
    if protocol["metric_schema_version"] != METRICS_SCHEMA_VERSION:
        raise ValueError("unsupported metric schema version")
    if protocol["gate_policy_version"] != VALIDATION_GATE_VERSION:
        raise ValueError("unsupported gate policy version")
    if protocol["gate_thresholds"] != CALIBRATION_GATE_THRESHOLDS:
        raise ValueError("calibration gate thresholds changed")
    if protocol["calibration_stage_sequence"] != CALIBRATION_STAGE_SEQUENCE:
        raise ValueError("calibration stage sequence changed")
    if protocol["candidate_values"] != CALIBRATION_CANDIDATE_VALUES:
        raise ValueError("calibration search space changed")
    if protocol["maximum_validation_executions"] != 1:
        raise ValueError("maximum validation executions changed")
    if raw_file is not None:
        raw_path = Path(raw_file)
        raw_bytes = raw_path.read_bytes()
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
    if environment_fingerprint is not None:
        fingerprint = validate_environment_fingerprint(environment_fingerprint)
        if fingerprint["environment_fingerprint_hash"] != protocol["environment_fingerprint_hash"]:
            raise ValueError("environment fingerprint hash changed")
    return protocol


def update_protocol_status(
    protocol_file: str | Path,
    status: str,
    **additional_fields: Any,
) -> dict:
    if status not in SUPPORTED_PROTOCOL_STATUSES:
        raise ValueError("unsupported calibration protocol status")
    protocol = validate_calibration_protocol(protocol_file)
    protocol.update(additional_fields)
    protocol["protocol_status"] = status
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = canonical_json_hash(protocol)
    atomic_json_write(protocol_file, protocol)
    return protocol


# Descriptive alias for callers that prefer the task wording.
build_and_write_calibration_protocol = freeze_calibration_protocol


def validate_development_bundle(
    value: dict | str | Path,
    *,
    protocol: dict | str | Path,
    belief_config_hash: str,
    split_manifest: dict | str | Path,
    train_file: str | Path | None = None,
    validation_file: str | Path | None = None,
) -> dict:
    """Fail closed on incompatible or tampered development artifacts."""
    bundle = _load_json(value)
    if bundle.get("schema_version") != DEVELOPMENT_BUNDLE_VERSION:
        raise ValueError("unsupported development bundle schema version")
    if DEVELOPMENT_BUNDLE_VERSION not in bundle.get("supported_consumer_versions", []):
        raise ValueError("development bundle is not compatible with this consumer")
    stored = bundle.get("bundle_hash")
    unsigned = dict(bundle)
    unsigned.pop("bundle_hash", None)
    if not stored or stored != canonical_json_hash(unsigned):
        raise ValueError("development bundle hash is invalid")
    manifest = _load_json(split_manifest)
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if not manifest_hash or manifest_hash != canonical_json_hash(unsigned_manifest):
        raise ValueError("split manifest hash is invalid")
    checked_protocol = validate_calibration_protocol(protocol, split_manifest=manifest)
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
    if not isinstance(artifacts, dict):
        raise ValueError("development bundle has no artifact inventory")
    for name, path in (("replayed_train", train_file), ("replayed_validation", validation_file)):
        if path is None:
            continue
        record = artifacts.get(name, {})
        if record.get("sha256") != file_sha256(path):
            raise ValueError(f"development bundle {name} hash mismatch")
        expected_split = "train" if name.endswith("train") else "validation"
        if record.get("split") != expected_split:
            raise ValueError(f"development bundle {name} split mismatch")
    if "replayed_test" in artifacts:
        raise ValueError("development bundle must not contain replayed test data")
    return bundle
