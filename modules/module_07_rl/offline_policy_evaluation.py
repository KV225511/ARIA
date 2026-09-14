"""Diagnostic-only weighted importance sampling for an IQL checkpoint."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch

from modules.module_07_rl.train import IQLNetworks
from modules.module_07_rl.state_builder import STATE_DIM
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE
from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.calibration_protocol import atomic_json_write, canonical_json_hash, file_sha256
from modules.module_07_rl.calibration_protocol_v7 import (
    validate_calibration_protocol_v7, validate_development_bundle_v7,
    validate_protocol_state_v7,
)


def evaluate_logged_policy(
    checkpoint_file, replayed_file, *, ratio_clip=20.0,
    protocol_file=None, protocol_state_file=None, development_bundle_file=None,
):
    """Evaluate logged non-test trajectories; never claim causal validation."""
    if ratio_clip <= 0:
        raise ValueError("ratio_clip must be positive")
    rows = json.loads(Path(replayed_file).read_text(encoding="utf-8"))
    if not rows or any(row.get("dataset_split") == "test" for row in rows):
        raise ValueError("OPE requires non-test replayed transitions")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if checkpoint.get("state_schema_version") is None:
        raise ValueError("unsupported checkpoint")
    nets = IQLNetworks(STATE_DIM, len(RL_ACTION_SPACE)).eval()
    nets.load_state_dict(checkpoint["model_state_dict"])
    ratios, rewards, target_entropies = [], [], []
    with torch.no_grad():
        for row in rows:
            state = torch.as_tensor([row["obs"]], dtype=torch.float32)
            logits = nets.policy_net(state).numpy()[0]
            mask = np.asarray(row["action_mask_before"], dtype=bool)
            logits[~mask] = -np.inf
            shifted = logits - np.max(logits[mask])
            probabilities = np.exp(shifted, where=mask, out=np.zeros_like(shifted))
            probabilities /= probabilities.sum()
            action = int(row["action_idx"])
            behavior = float(row["behavior_action_probability"])
            if behavior <= 0 or not mask[action]:
                raise ValueError("invalid logged behavior support")
            ratio = float(probabilities[action] / behavior)
            ratios.append(ratio)
            rewards.append(float(row["reward"]))
            target_entropies.append(float(-np.sum(probabilities[mask] * np.log(probabilities[mask] + 1e-12))))
    clipped = np.minimum(np.asarray(ratios), ratio_clip)
    weights = clipped / clipped.sum()
    ess = float(1.0 / np.sum(weights ** 2))
    provenance = {}
    supplied = (protocol_file, protocol_state_file, development_bundle_file)
    if any(supplied) and not all(supplied):
        raise ValueError("OPE provenance requires protocol, state, and development bundle")
    if all(supplied):
        protocol = validate_calibration_protocol_v7(protocol_file)
        state = validate_protocol_state_v7(protocol_state_file, protocol=protocol)
        if state["current_status"] not in {"PROVISIONAL_TRAINING_CV", "VALIDATED_SYNTHETIC"}:
            raise ValueError("v7 protocol state does not permit OPE")
        bundle_preview = json.loads(Path(development_bundle_file).read_text(encoding="utf-8"))
        artifacts = bundle_preview.get("artifacts", {})
        split_manifest = artifacts.get("split_manifest", {}).get("path")
        config_file = artifacts.get("belief_config", {}).get("path")
        if not split_manifest or not config_file:
            raise ValueError("development bundle lacks OPE provenance artifacts")
        config = BeliefModelConfig.load(config_file)
        split_names = {row.get("dataset_split") for row in rows}
        if len(split_names) != 1 or next(iter(split_names)) not in {"train", "validation"}:
            raise ValueError("OPE input must contain exactly one non-test split")
        split_name = next(iter(split_names))
        bundle = validate_development_bundle_v7(
            bundle_preview, protocol=protocol, state=state,
            belief_config_hash=config.config_hash, split_manifest=split_manifest,
            train_file=replayed_file if split_name == "train" else None,
            validation_file=replayed_file if split_name == "validation" else None,
        )
        expected_checkpoint = {
            "protocol_hash": protocol["protocol_hash"],
            "belief_config_hash": config.config_hash,
            "development_bundle_hash": bundle["bundle_hash"],
            "raw_file_sha256": protocol["raw_file_sha256"],
            "raw_dataset_hash": protocol["raw_dataset_hash"],
        }
        for field, expected in expected_checkpoint.items():
            if checkpoint.get(field) != expected:
                raise ValueError(f"checkpoint {field} mismatch")
        provenance = {
            **expected_checkpoint,
            "checkpoint_sha256": file_sha256(checkpoint_file),
            "replayed_file_sha256": file_sha256(replayed_file),
            "split_manifest_hash": bundle["split_manifest_hash"],
            "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
            "git_commit": protocol["code_commit"],
            "dataset_split": split_name,
        }
    report = {
        "schema_version": "aria-offline-policy-evaluation-v1",
        "producer_version": "aria-offline-policy-evaluation-v1",
        "supported_consumer_versions": ["aria-offline-policy-evaluation-v1"],
        "evaluates_learned_policy": True,
        "evaluation_method": "clipped-weighted-importance-sampling-diagnostic",
        "release_gate": False,
        "requires_fresh_rollouts_for_causal_claims": True,
        "ratio_clip": float(ratio_clip),
        "num_transitions": len(rows),
        "weighted_reward_estimate": float(np.sum(weights * np.asarray(rewards))),
        "effective_sample_size": ess,
        "maximum_unclipped_ratio": float(np.max(ratios)),
        "ratio_clipping_rate": float(np.mean(np.asarray(ratios) > ratio_clip)),
        "target_policy_action_entropy": float(np.mean(target_entropies)),
        **provenance,
    }
    report["report_hash"] = canonical_json_hash(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic ARIA offline policy evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replayed-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratio-clip", type=float, default=20.0)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-state", required=True)
    parser.add_argument("--development-bundle", required=True)
    args = parser.parse_args()
    report = evaluate_logged_policy(
        args.checkpoint, args.replayed_data, ratio_clip=args.ratio_clip,
        protocol_file=args.protocol, protocol_state_file=args.protocol_state,
        development_bundle_file=args.development_bundle,
    )
    atomic_json_write(Path(args.output), report)
    print(json.dumps(report, indent=2))
