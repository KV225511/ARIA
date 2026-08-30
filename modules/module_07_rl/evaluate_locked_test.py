"""Explicit, post-freeze evaluation of stored beliefs on the locked test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.dataset_audit import audit_locked_test
from modules.module_07_rl.metrics import build_belief_report
from modules.module_07_rl.replay_dataset import REPLAY_SCHEMA_VERSION
from modules.module_07_rl.reward_model import REWARD_SCHEMA_VERSION
from modules.module_07_rl.state_builder import STATE_FEATURE_NAMES, STATE_SCHEMA_VERSION


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_locked_test(
    test_file: str | Path,
    config_file: str | Path,
    output_file: str | Path,
    *,
    confirm_config_frozen: bool = False,
):
    """Unlock test labels only after an explicit configuration-freeze assertion."""
    if not confirm_config_frozen:
        raise ValueError("Refusing to unlock test metrics without config-freeze confirmation")

    test_path = Path(test_file)
    config_path = Path(config_file)
    output_path = Path(output_file)
    if output_path.exists():
        raise FileExistsError(
            f"Locked-test report already exists and will not be overwritten: {output_path}"
        )

    config = BeliefModelConfig.load(config_path)
    transitions = json.loads(test_path.read_text(encoding="utf-8"))
    if not transitions:
        raise ValueError("Locked test split is empty")
    expected = {
        "dataset_split": "test",
        "belief_config_hash": config.config_hash,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
    }
    for index, transition in enumerate(transitions):
        for field, value in expected.items():
            if transition.get(field) != value:
                raise ValueError(
                    f"Locked test transition {index} has incompatible {field}"
                )
        if tuple(transition.get("state_feature_names", ())) != STATE_FEATURE_NAMES:
            raise ValueError(
                f"Locked test transition {index} has incompatible state feature semantics"
            )
    raw_hashes = {item.get("raw_dataset_hash") for item in transitions}
    split_hashes = {item.get("split_manifest_hash") for item in transitions}
    if len(raw_hashes) != 1 or None in raw_hashes:
        raise ValueError("Locked test split has inconsistent raw dataset hashes")
    if len(split_hashes) != 1 or None in split_hashes:
        raise ValueError("Locked test split has inconsistent split manifest hashes")
    if config.raw_dataset_hash and config.raw_dataset_hash not in raw_hashes:
        raise ValueError("Locked test raw dataset hash does not match frozen config")
    if config.split_manifest_hash and config.split_manifest_hash not in split_hashes:
        raise ValueError("Locked test split hash does not match frozen config")

    report = {
        "schema_version": "aria-locked-test-report-v3",
        "evaluation_type": "stored_belief_verdict",
        "evaluates_learned_policy": False,
        "test_metrics_unlocked": True,
        "belief_config_hash": config.config_hash,
        "belief_config_file_hash": _file_sha256(config_path),
        "test_file_hash": _file_sha256(test_path),
        "raw_dataset_hash": next(iter(raw_hashes)),
        "split_manifest_hash": next(iter(split_hashes)),
        "state_schema_version": STATE_SCHEMA_VERSION,
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "locked_test_gate": audit_locked_test(transitions),
        "stored_belief_report": build_belief_report(transitions),
        "policy_evaluation_limitation": (
            "This report evaluates the frozen belief verdict only. Logged v3 "
            "propensities support a separate OPE report, and final policy release "
            "requires fresh learned-policy rollouts."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    # Class metrics may mix integer labels with an abstention key; sorting
    # heterogeneous JSON keys raises in Python 3.
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file")
    parser.add_argument("config_file")
    parser.add_argument("output_file")
    parser.add_argument(
        "--confirm-config-frozen",
        action="store_true",
        help="Required acknowledgement that all calibration and model selection are complete.",
    )
    args = parser.parse_args()
    try:
        result = evaluate_locked_test(
            args.test_file,
            args.config_file,
            args.output_file,
            confirm_config_frozen=args.confirm_config_frozen,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"[ERROR] {error}\n")
    print(json.dumps(result, indent=2))
