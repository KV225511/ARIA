"""Deterministic IQL training over replayed, versioned ARIA transitions."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.dataset_audit import (
    audit_calibration_validation,
    audit_offline_rl_support,
    audit_raw_evidence,
)
from modules.module_07_rl.dataset_split import apply_terminal_outcome_rewards
from modules.module_07_rl.metrics import build_belief_report
from modules.module_07_rl.rl_spec import REWARD_COEFFICIENTS, RL_ACTION_SPACE
from modules.module_07_rl.state_builder import (
    STATE_DIM,
    STATE_FEATURE_NAMES,
    STATE_SCHEMA_VERSION,
)
from modules.module_07_rl.rl_spec import ACTION_SCHEMA_VERSION
from modules.module_07_rl.reward_model import REWARD_SCHEMA_VERSION
from modules.module_07_rl.transition_schema import (
    REQUIRED_POLICY_FIELDS,
    TRANSITION_SCHEMA_VERSION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DERIVED_DIR = PROJECT_ROOT / "data" / "synthetic" / "derived"
DEFAULT_TRAIN_FILE = DEFAULT_DERIVED_DIR / "splits" / "train.json"
DEFAULT_VALIDATION_FILE = DEFAULT_DERIVED_DIR / "splits" / "validation.json"
DEFAULT_CONFIG_FILE = DEFAULT_DERIVED_DIR / "belief_model_v2.json"
DEFAULT_CHECKPOINT = Path(__file__).with_name("aria_iql_belief_v3.pth")


class IQLNetworks(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self.q2_net = copy.deepcopy(self.q1_net)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, action_dim),
        )


def get_action_one_hot(action, action_dim):
    one_hot = np.zeros(action_dim, dtype=np.float32)
    one_hot[action] = 1.0
    return one_hot


def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return weight * diff**2


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # Older Torch versions do not support warn_only.
        torch.use_deterministic_algorithms(True)


def validate_replayed_dataset(dataset, config, expected_split):
    if not dataset:
        raise ValueError(f"{expected_split} replay dataset is empty")
    required = {
        "obs", "next_obs", "action_idx", "reward", "done",
        "state_schema_version", "state_feature_names", "belief_config_hash",
        "dataset_split", "transition_schema_version", "action_schema_version",
        "reward_schema_version", *REQUIRED_POLICY_FIELDS,
    }
    for index, transition in enumerate(dataset):
        missing = required - transition.keys()
        if missing:
            raise ValueError(f"Transition {index} is missing {sorted(missing)}")
        if transition["state_schema_version"] != STATE_SCHEMA_VERSION:
            raise ValueError(f"Transition {index} uses an incompatible state schema")
        if transition["transition_schema_version"] != TRANSITION_SCHEMA_VERSION:
            raise ValueError(f"Transition {index} uses an incompatible transition schema")
        if transition["action_schema_version"] != ACTION_SCHEMA_VERSION:
            raise ValueError(f"Transition {index} uses an incompatible action schema")
        if transition["reward_schema_version"] != REWARD_SCHEMA_VERSION:
            raise ValueError(f"Transition {index} uses an incompatible reward schema")
        if tuple(transition["state_feature_names"]) != STATE_FEATURE_NAMES:
            raise ValueError(f"Transition {index} uses incompatible state feature semantics")
        if transition["belief_config_hash"] != config.config_hash:
            raise ValueError(f"Transition {index} belief config hash mismatch")
        if transition["dataset_split"] != expected_split:
            raise ValueError(f"Transition {index} is assigned to the wrong split")
        for field in ("obs", "next_obs"):
            values = np.asarray(transition[field], dtype=float)
            if values.shape != (STATE_DIM,) or not np.all(np.isfinite(values)):
                raise ValueError(f"Transition {index} has invalid {field}")
        action_idx = transition["action_idx"]
        if not isinstance(action_idx, int) or not 0 <= action_idx < len(RL_ACTION_SPACE):
            raise ValueError(f"Transition {index} has invalid action_idx")
        action_mask = np.asarray(transition["action_mask_before"], dtype=float)
        behavior_probs = np.asarray(transition["behavior_action_probs"], dtype=float)
        expected_shape = (len(RL_ACTION_SPACE),)
        if action_mask.shape != expected_shape or not np.all(
            np.isin(action_mask, (0.0, 1.0))
        ):
            raise ValueError(f"Transition {index} has an invalid pre-action mask")
        if action_mask[action_idx] != 1.0:
            raise ValueError(f"Transition {index} selected an action masked as illegal")
        if behavior_probs.shape != expected_shape or not np.all(
            np.isfinite(behavior_probs)
        ) or np.any(behavior_probs < 0.0) or not np.isclose(behavior_probs.sum(), 1.0):
            raise ValueError(f"Transition {index} has invalid behavior propensities")
        selected_probability = float(transition["behavior_action_probability"])
        if selected_probability <= 0.0 or not np.isclose(
            selected_probability, behavior_probs[action_idx]
        ):
            raise ValueError(f"Transition {index} has inconsistent selected propensity")
        expected_kind = "stop" if action_idx == RL_ACTION_SPACE.index(
            "conclude_interview"
        ) else "question"
        if transition["transition_kind"] != expected_kind:
            raise ValueError(f"Transition {index} has inconsistent transition_kind")
        if expected_kind == "stop":
            if not transition["done"]:
                raise ValueError(f"Transition {index} contains a non-terminal legal stop")
            if transition.get("semantic_score") is not None:
                raise ValueError(f"Transition {index} stop action contains synthetic evidence")
            if not np.allclose(transition["obs"], transition["next_obs"]):
                raise ValueError(f"Transition {index} stop action mutates policy state")
        reward = float(transition["reward"])
        if not math.isfinite(reward):
            raise ValueError(f"Transition {index} has non-finite reward")


def _arrays(dataset):
    action_indices = np.asarray([item["action_idx"] for item in dataset], dtype=np.int64)
    return {
        "obs": np.asarray([item["obs"] for item in dataset], dtype=np.float32),
        "actions": np.asarray(
            [get_action_one_hot(index, len(RL_ACTION_SPACE)) for index in action_indices],
            dtype=np.float32,
        ),
        "action_indices": action_indices,
        "rewards": np.asarray([item["reward"] for item in dataset], dtype=np.float32).reshape(-1, 1),
        "next_obs": np.asarray([item["next_obs"] for item in dataset], dtype=np.float32),
        "done": np.asarray([item["done"] for item in dataset], dtype=np.float32).reshape(-1, 1),
    }


def _deterministic_forward(module, values):
    was_training = module.training
    module.eval()
    result = module(values)
    module.train(was_training)
    return result


def _validation_objective(nets, arrays, device, gamma, expectile):
    modes = {name: module.training for name, module in nets.named_children()}
    nets.eval()
    with torch.no_grad():
        s = torch.as_tensor(arrays["obs"], device=device)
        action = torch.as_tensor(arrays["actions"], device=device)
        action_idx = torch.as_tensor(arrays["action_indices"], device=device)
        reward = torch.as_tensor(arrays["rewards"], device=device)
        next_state = torch.as_tensor(arrays["next_obs"], device=device)
        done = torch.as_tensor(arrays["done"], device=device)
        q1 = nets.q1_net(torch.cat([s, action], dim=1))
        q2 = nets.q2_net(torch.cat([s, action], dim=1))
        v = nets.v_net(s)
        v_next = nets.v_net(next_state)
        q_min = torch.minimum(q1, q2)
        v_loss = expectile_loss(q_min - v, expectile).mean()
        target = reward + (1 - done) * gamma * v_next
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        behavior_nll = F.cross_entropy(nets.policy_net(s), action_idx)
        objective = float((v_loss + q_loss + behavior_nll).item())
    for name, module in nets.named_children():
        module.train(modes[name])
    return objective


def _atomic_torch_save(value, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, output)


def train_iql_policy(
    train_file=DEFAULT_TRAIN_FILE,
    validation_file=DEFAULT_VALIDATION_FILE,
    belief_config_file=DEFAULT_CONFIG_FILE,
    output_file=DEFAULT_CHECKPOINT,
    total_epochs=100,
    batch_size=256,
    lr=1e-4,
    gamma=0.99,
    tau=0.005,
    expectile=0.8,
    beta=3.0,
    seed=42,
    enforce_belief_quality_gate=True,
    early_stopping_patience=10,
    early_stopping_min_delta=1e-4,
):
    """Train without loading the locked test split."""
    train_path = Path(train_file)
    validation_path = Path(validation_file)
    config = BeliefModelConfig.load(belief_config_file)
    training_source = json.loads(train_path.read_text(encoding="utf-8"))
    validation_source = json.loads(validation_path.read_text(encoding="utf-8"))
    validate_replayed_dataset(training_source, config, "train")
    validate_replayed_dataset(validation_source, config, "validation")

    combined_for_integrity = training_source + validation_source
    raw_gate = audit_raw_evidence(
        combined_for_integrity,
        min_episodes=1,
        min_independent_components=2,
    )
    offline_gate = audit_offline_rl_support(training_source)
    belief_gate = audit_calibration_validation(validation_source)
    print(json.dumps({
        "raw_evidence_gate": raw_gate,
        "offline_rl_gate": offline_gate,
        "validation_belief_gate": belief_gate,
    }, indent=2))
    if not raw_gate["passes_quality_gates"]:
        raise RuntimeError("Raw evidence gate failed; training is not permitted")
    if not offline_gate["passes_quality_gates"]:
        raise RuntimeError("Offline-RL support gate failed; training is not permitted")
    if enforce_belief_quality_gate and not belief_gate["passes_quality_gates"]:
        raise RuntimeError("Validation belief gate failed; freeze calibration before training")

    training = [dict(item) for item in training_source]
    validation = [dict(item) for item in validation_source]
    omega = REWARD_COEFFICIENTS.get("omega", 5.0)
    train_mismatches = apply_terminal_outcome_rewards(training, omega)
    validation_mismatches = apply_terminal_outcome_rewards(validation, omega)
    train_arrays = _arrays(training)
    validation_arrays = _arrays(validation)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = IQLNetworks(STATE_DIM, len(RL_ACTION_SPACE)).to(device)
    q1_target = copy.deepcopy(nets.q1_net).to(device).eval()
    q2_target = copy.deepcopy(nets.q2_net).to(device).eval()
    for parameter in list(q1_target.parameters()) + list(q2_target.parameters()):
        parameter.requires_grad_(False)

    optimizer_v = optim.Adam(nets.v_net.parameters(), lr=lr)
    optimizer_q = optim.Adam(
        list(nets.q1_net.parameters()) + list(nets.q2_net.parameters()), lr=lr
    )
    optimizer_policy = optim.Adam(nets.policy_net.parameters(), lr=lr)
    scheduler_v = CosineAnnealingLR(optimizer_v, T_max=total_epochs, eta_min=1e-5)
    scheduler_q = CosineAnnealingLR(optimizer_q, T_max=total_epochs, eta_min=1e-5)
    scheduler_policy = CosineAnnealingLR(optimizer_policy, T_max=total_epochs, eta_min=1e-5)

    num_samples = len(training)
    batch_size = min(batch_size, num_samples)
    num_batches = math.ceil(num_samples / batch_size)
    best_validation_objective = math.inf
    best_epoch = None
    epochs_without_improvement = 0
    epochs_completed = 0
    stopped_early = False
    output_path = Path(output_file)

    for epoch in range(total_epochs):
        epochs_completed = epoch + 1
        nets.train()
        indices = np.random.default_rng(seed + epoch).permutation(num_samples)
        totals = {"v": 0.0, "q": 0.0, "policy": 0.0}
        for start in range(0, num_samples, batch_size):
            batch = indices[start:start + batch_size]
            s = torch.as_tensor(train_arrays["obs"][batch], device=device)
            action = torch.as_tensor(train_arrays["actions"][batch], device=device)
            action_idx = torch.as_tensor(train_arrays["action_indices"][batch], device=device)
            reward = torch.as_tensor(train_arrays["rewards"][batch], device=device)
            next_state = torch.as_tensor(train_arrays["next_obs"][batch], device=device)
            done = torch.as_tensor(train_arrays["done"][batch], device=device)

            with torch.no_grad():
                q_target_behavior = torch.minimum(
                    q1_target(torch.cat([s, action], dim=1)),
                    q2_target(torch.cat([s, action], dim=1)),
                )
            value = nets.v_net(s)
            value_loss = expectile_loss(q_target_behavior - value, expectile).mean()
            optimizer_v.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(nets.v_net.parameters(), 1.0)
            optimizer_v.step()

            with torch.no_grad():
                next_value = _deterministic_forward(nets.v_net, next_state)
                q_backup = reward + (1 - done) * gamma * next_value
            q1 = nets.q1_net(torch.cat([s, action], dim=1))
            q2 = nets.q2_net(torch.cat([s, action], dim=1))
            q_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)
            optimizer_q.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(nets.q1_net.parameters()) + list(nets.q2_net.parameters()), 1.0
            )
            optimizer_q.step()

            with torch.no_grad():
                current_value = _deterministic_forward(nets.v_net, s)
                advantage = q_target_behavior - current_value
                weight = torch.exp(beta * advantage).clamp(max=100.0)
            log_probs = F.log_softmax(nets.policy_net(s), dim=-1)
            policy_loss = -(weight * log_probs.gather(1, action_idx.unsqueeze(1))).mean()
            optimizer_policy.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(nets.policy_net.parameters(), 1.0)
            optimizer_policy.step()

            with torch.no_grad():
                for source, target in zip(nets.q1_net.parameters(), q1_target.parameters()):
                    target.copy_(tau * source + (1 - tau) * target)
                for source, target in zip(nets.q2_net.parameters(), q2_target.parameters()):
                    target.copy_(tau * source + (1 - tau) * target)
            totals["v"] += value_loss.item()
            totals["q"] += q_loss.item()
            totals["policy"] += policy_loss.item()

        validation_objective = _validation_objective(
            nets, validation_arrays, device, gamma, expectile
        )
        print(
            f"Epoch {epoch + 1}/{total_epochs} | "
            f"V {totals['v']/num_batches:.4f} | Q {totals['q']/num_batches:.4f} | "
            f"Pi {totals['policy']/num_batches:.4f} | "
            f"validation_objective {validation_objective:.4f}"
        )
        if validation_objective < best_validation_objective - early_stopping_min_delta:
            best_validation_objective = validation_objective
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            checkpoint = {
                "checkpoint_schema_version": "aria-iql-checkpoint-v3",
                "model_state_dict": nets.state_dict(),
                "state_schema_version": STATE_SCHEMA_VERSION,
                "state_feature_names": list(STATE_FEATURE_NAMES),
                "belief_config": config.to_dict(),
                "belief_config_hash": config.config_hash,
                "split_manifest_hash": config.split_manifest_hash,
                "training": {
                    "best_epoch": best_epoch,
                    "validation_selection_metric": "offline_iql_validation_objective",
                    "validation_objective": best_validation_objective,
                    "seed": seed,
                    "hyperparameters": {
                        "gamma": gamma, "tau": tau, "expectile": expectile,
                        "beta": beta, "learning_rate": lr,
                        "early_stopping_patience": early_stopping_patience,
                        "early_stopping_min_delta": early_stopping_min_delta,
                    },
                    "train_terminal_mismatches": train_mismatches,
                    "validation_terminal_mismatches": validation_mismatches,
                },
            }
            _atomic_torch_save(checkpoint, output_path)
        else:
            epochs_without_improvement += 1
        scheduler_v.step()
        scheduler_q.step()
        scheduler_policy.step()
        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            print(
                f"Early stopping after epoch {epoch + 1}: validation objective "
                f"did not improve by {early_stopping_min_delta} for "
                f"{early_stopping_patience} epochs."
            )
            break

    result = {
        "checkpoint": str(output_path),
        "best_epoch": best_epoch,
        "best_validation_objective": best_validation_objective,
        "epochs_completed": epochs_completed,
        "stopped_early": stopped_early,
        "validation_belief_report": build_belief_report(validation_source),
        "evaluates_learned_policy": False,
        "policy_evaluation_limitation": (
            "The v3 corpus logs propensities for offline policy evaluation, but "
            "this training command does not run OPE; final release still requires "
            "fresh learned-policy rollouts."
        ),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--validation-file", default=str(DEFAULT_VALIDATION_FILE))
    parser.add_argument("--belief-config", default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--output", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--allow-belief-gate-failure",
        action="store_true",
        help="Experimental only; raw-evidence and offline-RL gates remain enforced.",
    )
    args = parser.parse_args()
    try:
        train_iql_policy(
            train_file=args.train_file,
            validation_file=args.validation_file,
            belief_config_file=args.belief_config,
            output_file=args.output,
            total_epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            enforce_belief_quality_gate=not args.allow_belief_gate_failure,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"[ERROR] {error}\n")
