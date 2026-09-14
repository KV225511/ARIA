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


def evaluate_logged_policy(checkpoint_file, replayed_file, *, ratio_clip=20.0):
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
    return {
        "schema_version": "aria-offline-policy-evaluation-v1",
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
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic ARIA offline policy evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replayed-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratio-clip", type=float, default=20.0)
    args = parser.parse_args()
    report = evaluate_logged_policy(args.checkpoint, args.replayed_data, ratio_clip=args.ratio_clip)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
