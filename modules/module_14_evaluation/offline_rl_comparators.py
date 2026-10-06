"""Same-data offline-RL comparators for the frozen ARIA transition corpus.

The implementations are intentionally compact, deterministic reference
implementations. They train new comparator checkpoints and never mutate ARIA's
dataset, belief configuration, or IQL checkpoint.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.calibration_protocol import atomic_json_write, canonical_json_hash
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE
from modules.module_07_rl.state_builder import STATE_DIM, STATE_SCHEMA_VERSION
from modules.module_07_rl.train import validate_replayed_dataset
from modules.module_14_evaluation.sota_training import file_sha256


COMPARATOR_CHECKPOINT_VERSION = "aria-offline-rl-comparator-v1"
ACTION_DIM = len(RL_ACTION_SPACE)


@dataclass(frozen=True)
class ComparatorConfig:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 3e-4
    hidden_dim: int = 128
    gamma: float = 0.99
    cql_alpha: float = 1.0
    context_length: int = 12
    transformer_layers: int = 2
    transformer_heads: int = 4


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_frozen_development_data(project_root: str | Path) -> tuple[list[dict], list[dict], dict]:
    root = Path(project_root).resolve()
    v7 = root / "data/synthetic/v3/production-grounding-v10/derived-calibration-v7"
    train_file = v7 / "replayed/train.json"
    validation_file = v7 / "replayed/validation.json"
    belief_file = v7 / "calibration/belief_model_v3.json"
    for path in (train_file, validation_file, belief_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = BeliefModelConfig.load(belief_file)
    train_rows = _load_json(train_file)
    validation_rows = _load_json(validation_file)
    validate_replayed_dataset(train_rows, config, "train")
    validate_replayed_dataset(validation_rows, config, "validation")
    provenance = {
        "train_file": str(train_file.relative_to(root)),
        "train_sha256": file_sha256(train_file),
        "validation_file": str(validation_file.relative_to(root)),
        "validation_sha256": file_sha256(validation_file),
        "belief_config_file": str(belief_file.relative_to(root)),
        "belief_config_hash": config.config_hash,
        "state_schema_version": STATE_SCHEMA_VERSION,
    }
    return train_rows, validation_rows, provenance


def _transition_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    return {
        "obs": np.asarray([row["obs"] for row in rows], dtype=np.float32),
        "next_obs": np.asarray([row["next_obs"] for row in rows], dtype=np.float32),
        "action": np.asarray([row["action_idx"] for row in rows], dtype=np.int64),
        "reward": np.asarray([row["reward"] for row in rows], dtype=np.float32),
        "done": np.asarray([row["done"] for row in rows], dtype=np.float32),
        "mask": np.asarray([row["action_mask_before"] for row in rows], dtype=np.float32),
        "next_mask": np.asarray([row["action_mask"] for row in rows], dtype=np.float32),
    }


class MLP(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state):
        return self.network(state)


def _masked_accuracy(values: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor) -> float:
    masked = values.masked_fill(~masks.bool(), -torch.inf)
    return float((masked.argmax(dim=1) == actions).float().mean().item())


def train_behavior_cloning(train_rows: list[dict], validation_rows: list[dict],
                           config: ComparatorConfig) -> tuple[nn.Module, dict]:
    _seed_everything(config.seed)
    train = _transition_arrays(train_rows)
    validation = _transition_arrays(validation_rows)
    model = MLP(ACTION_DIM, config.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(TensorDataset(
        torch.from_numpy(train["obs"]), torch.from_numpy(train["action"]),
    ), batch_size=config.batch_size, shuffle=True,
       generator=torch.Generator().manual_seed(config.seed))
    best_state, best_accuracy, history = None, -1.0, []
    val_obs = torch.from_numpy(validation["obs"])
    val_actions = torch.from_numpy(validation["action"])
    val_masks = torch.from_numpy(validation["mask"])
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for states, actions in loader:
            loss = F.cross_entropy(model(states), actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            accuracy = _masked_accuracy(model(val_obs), val_actions, val_masks)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_action_accuracy": accuracy})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), {"selection_metric": "validation_action_accuracy",
                          "best_validation_action_accuracy": best_accuracy, "history": history}


def train_discrete_cql(train_rows: list[dict], validation_rows: list[dict],
                       config: ComparatorConfig) -> tuple[nn.Module, dict]:
    _seed_everything(config.seed)
    train = _transition_arrays(train_rows)
    validation = _transition_arrays(validation_rows)
    q_network = MLP(ACTION_DIM, config.hidden_dim)
    target_network = copy.deepcopy(q_network).eval()
    optimizer = torch.optim.AdamW(q_network.parameters(), lr=config.learning_rate)
    tensors = [torch.from_numpy(train[name]) for name in
               ("obs", "action", "reward", "next_obs", "done", "mask", "next_mask")]
    loader = DataLoader(TensorDataset(*tensors), batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed))
    best_state, best_accuracy, history = None, -1.0, []
    val_obs = torch.from_numpy(validation["obs"])
    val_actions = torch.from_numpy(validation["action"])
    val_masks = torch.from_numpy(validation["mask"])
    for epoch in range(config.epochs):
        q_network.train()
        losses = []
        for state, action, reward, next_state, done, mask, next_mask in loader:
            q_values = q_network(state)
            selected_q = q_values.gather(1, action[:, None]).squeeze(1)
            with torch.no_grad():
                next_q = target_network(next_state).masked_fill(~next_mask.bool(), -torch.inf)
                next_value = next_q.max(dim=1).values
                next_value = torch.where(torch.isfinite(next_value), next_value,
                                         torch.zeros_like(next_value))
                target = reward + (1.0 - done) * config.gamma * next_value
            bellman = F.smooth_l1_loss(selected_q, target)
            legal_q = q_values.masked_fill(~mask.bool(), -torch.inf)
            conservative = (torch.logsumexp(legal_q, dim=1) - selected_q).mean()
            loss = bellman + config.cql_alpha * conservative
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        target_network.load_state_dict(q_network.state_dict())
        q_network.eval()
        with torch.no_grad():
            accuracy = _masked_accuracy(q_network(val_obs), val_actions, val_masks)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_action_accuracy": accuracy})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(q_network.state_dict())
    q_network.load_state_dict(best_state)
    return q_network.eval(), {"selection_metric": "validation_action_accuracy",
                              "best_validation_action_accuracy": best_accuracy, "history": history}


class CausalReturnTransformer(nn.Module):
    def __init__(self, config: ComparatorConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.context_length = config.context_length
        self.state_embedding = nn.Linear(STATE_DIM, hidden)
        self.return_embedding = nn.Linear(1, hidden)
        self.action_embedding = nn.Embedding(ACTION_DIM + 1, hidden)
        self.position_embedding = nn.Embedding(config.context_length, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=config.transformer_heads,
            dim_feedforward=hidden * 4, dropout=0.1, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, ACTION_DIM)

    def forward(self, states, previous_actions, returns_to_go, padding_mask):
        batch, length, _ = states.shape
        positions = torch.arange(length, device=states.device).expand(batch, length)
        values = (self.state_embedding(states) + self.action_embedding(previous_actions)
                  + self.return_embedding(returns_to_go) + self.position_embedding(positions))
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                            device=states.device), diagonal=1)
        encoded = self.transformer(values, mask=causal_mask,
                                   src_key_padding_mask=padding_mask)
        return self.head(self.norm(encoded))


def _episode_rows(rows: list[dict]) -> list[list[dict]]:
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["episode_id"]].append((index, row))
    # Replay files preserve transition order; the original index is the stable tie-breaker.
    return [[row for _, row in sorted(items)] for _, items in sorted(grouped.items())]


def _transformer_examples(rows: list[dict], context_length: int,
                          return_scale: float) -> tuple[np.ndarray, ...]:
    examples = []
    start_action = ACTION_DIM
    for episode in _episode_rows(rows):
        rewards = np.asarray([row["reward"] for row in episode], dtype=np.float32)
        returns = np.flip(np.cumsum(np.flip(rewards))).copy() / return_scale
        for end in range(len(episode)):
            start = max(0, end - context_length + 1)
            selected = episode[start:end + 1]
            length = len(selected)
            states = np.zeros((context_length, STATE_DIM), dtype=np.float32)
            previous = np.full(context_length, start_action, dtype=np.int64)
            rtg = np.zeros((context_length, 1), dtype=np.float32)
            padding = np.ones(context_length, dtype=bool)
            offset = context_length - length
            states[offset:] = np.asarray([row["obs"] for row in selected], dtype=np.float32)
            for local, global_index in enumerate(range(start, end + 1)):
                previous[offset + local] = (start_action if global_index == 0
                                            else episode[global_index - 1]["action_idx"])
                rtg[offset + local, 0] = returns[global_index]
            padding[offset:] = False
            examples.append((states, previous, rtg, padding,
                             int(episode[end]["action_idx"]),
                             np.asarray(episode[end]["action_mask_before"], dtype=np.float32)))
    return (
        np.asarray([example[0] for example in examples], dtype=np.float32),
        np.asarray([example[1] for example in examples], dtype=np.int64),
        np.asarray([example[2] for example in examples], dtype=np.float32),
        np.asarray([example[3] for example in examples], dtype=bool),
        np.asarray([example[4] for example in examples], dtype=np.int64),
        np.asarray([example[5] for example in examples], dtype=np.float32),
    )


def train_decision_transformer(train_rows: list[dict], validation_rows: list[dict],
                               config: ComparatorConfig) -> tuple[nn.Module, dict]:
    _seed_everything(config.seed)
    episode_returns = [sum(float(row["reward"]) for row in episode)
                       for episode in _episode_rows(train_rows)]
    return_scale = max(float(np.max(np.abs(episode_returns))), 1.0)
    target_return = float(np.quantile(episode_returns, 0.9))
    train = _transformer_examples(train_rows, config.context_length, return_scale)
    validation = _transformer_examples(validation_rows, config.context_length, return_scale)
    train_tensors = [torch.from_numpy(value) for value in train[:5]]
    loader = DataLoader(TensorDataset(*train_tensors), batch_size=config.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(config.seed))
    model = CausalReturnTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    val_tensors = [torch.from_numpy(value) for value in validation]
    best_state, best_accuracy, history = None, -1.0, []
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for states, previous, rtg, padding, action in loader:
            logits = model(states, previous, rtg, padding)[:, -1]
            loss = F.cross_entropy(logits, action)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            logits = model(*val_tensors[:4])[:, -1]
            accuracy = _masked_accuracy(logits, val_tensors[4], val_tensors[5])
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_action_accuracy": accuracy})
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), {
        "selection_metric": "validation_action_accuracy",
        "best_validation_action_accuracy": best_accuracy,
        "return_scale": return_scale,
        "target_return": target_return,
        "history": history,
    }


def _checkpoint_payload(name: str, model: nn.Module, config: ComparatorConfig,
                        training: dict, provenance: dict) -> dict:
    metadata = {
        "schema_version": COMPARATOR_CHECKPOINT_VERSION,
        "algorithm": name,
        "config": asdict(config),
        "training": training,
        "provenance": provenance,
        "dataset_mutated": False,
        "aria_iql_mutated": False,
    }
    metadata["metadata_hash"] = canonical_json_hash(metadata)
    return {**metadata, "model_state_dict": model.state_dict()}


def train_all_comparators(project_root: str | Path, output_dir: str | Path,
                          config: ComparatorConfig = ComparatorConfig()) -> dict:
    train_rows, validation_rows, provenance = load_frozen_development_data(project_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trainers = {
        "behavior_cloning": train_behavior_cloning,
        "discrete_cql": train_discrete_cql,
        "decision_transformer": train_decision_transformer,
    }
    results = {}
    for name, trainer in trainers.items():
        model, training = trainer(train_rows, validation_rows, config)
        checkpoint = output / f"{name}.pth"
        torch.save(_checkpoint_payload(name, model, config, training, provenance), checkpoint)
        results[name] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "best_validation_action_accuracy": training["best_validation_action_accuracy"],
        }
    report = {
        "schema_version": COMPARATOR_CHECKPOINT_VERSION,
        "config": asdict(config),
        "provenance": provenance,
        "comparators": results,
        "test_split_used": False,
    }
    report["report_hash"] = canonical_json_hash(report)
    atomic_json_write(output / "training_report.json", report)
    return report


class GreedyNetworkPolicy:
    def __init__(self, model: nn.Module, name: str):
        self.model = model.eval()
        self.name = name

    def select_action(self, observation, mask, **kwargs):
        legal = np.asarray(mask, dtype=bool)
        with torch.no_grad():
            values = self.model(torch.as_tensor(np.asarray(observation)[None], dtype=torch.float32))[0]
        masked = values.numpy().astype(float)
        masked[~legal] = -np.inf
        action = int(np.argmax(masked))
        probabilities = np.zeros(ACTION_DIM, dtype=float)
        probabilities[action] = 1.0
        return {"action_idx": action, "action_probabilities": probabilities.tolist(),
                "selected_action_probability": 1.0}


class DecisionTransformerPolicy:
    def __init__(self, model: CausalReturnTransformer, *, target_return: float,
                 return_scale: float):
        self.model = model.eval()
        self.remaining_return = float(target_return)
        self.return_scale = float(return_scale)
        self.observations: list[np.ndarray] = []
        self.previous_actions: list[int] = []
        self.last_action = ACTION_DIM

    def select_action(self, observation, mask, **kwargs):
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.previous_actions.append(self.last_action)
        length = min(len(self.observations), self.model.context_length)
        states = np.zeros((1, self.model.context_length, STATE_DIM), dtype=np.float32)
        previous = np.full((1, self.model.context_length), ACTION_DIM, dtype=np.int64)
        rtg = np.zeros((1, self.model.context_length, 1), dtype=np.float32)
        padding = np.ones((1, self.model.context_length), dtype=bool)
        offset = self.model.context_length - length
        states[0, offset:] = self.observations[-length:]
        previous[0, offset:] = self.previous_actions[-length:]
        rtg[0, offset:, 0] = self.remaining_return / self.return_scale
        padding[0, offset:] = False
        with torch.no_grad():
            logits = self.model(torch.from_numpy(states), torch.from_numpy(previous),
                                torch.from_numpy(rtg), torch.from_numpy(padding))[0, -1]
        legal = np.asarray(mask, dtype=bool)
        values = logits.numpy().astype(float)
        values[~legal] = -np.inf
        action = int(np.argmax(values))
        self.last_action = action
        probabilities = np.zeros(ACTION_DIM, dtype=float)
        probabilities[action] = 1.0
        return {"action_idx": action, "action_probabilities": probabilities.tolist(),
                "selected_action_probability": 1.0}

    def observe_transition(self, reward: float) -> None:
        self.remaining_return -= float(reward)


def load_comparator_factory(checkpoint_file: str | Path):
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != COMPARATOR_CHECKPOINT_VERSION:
        raise ValueError("unsupported comparator checkpoint")
    config = ComparatorConfig(**checkpoint["config"])
    algorithm = checkpoint["algorithm"]
    if algorithm == "decision_transformer":
        model = CausalReturnTransformer(config)
    elif algorithm in {"behavior_cloning", "discrete_cql"}:
        model = MLP(ACTION_DIM, config.hidden_dim)
    else:
        raise ValueError(f"unknown comparator algorithm: {algorithm}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    if algorithm == "decision_transformer":
        training = checkpoint["training"]
        return lambda _seed: DecisionTransformerPolicy(
            model, target_return=training["target_return"], return_scale=training["return_scale"]
        )
    return lambda _seed: GreedyNetworkPolicy(model, algorithm)
