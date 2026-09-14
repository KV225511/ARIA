"""Fresh-environment rollout and evaluation for a loaded ARIA IQL policy.

This module deliberately has no dataset-file or locked-test input.  It evaluates
only actions selected online by the supplied policy in the supplied environment.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Iterable
import math

import numpy as np

from modules.module_07_rl.calibration_protocol import atomic_json_write, canonical_json_hash
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE


ROLLOUT_SCHEMA_VERSION = "aria-iql-fresh-rollout-v1"
ROLLOUT_REPORT_VERSION = "aria-iql-rollout-report-v1"


def _evidence_values(value):
    if isinstance(value, dict):
        return (
            value.get("semantic_score"),
            value.get("behavior_score"),
            value.get("cognitive_load"),
        )
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(value)
    raise ValueError("evidence provider must return a mapping or three values")


def rollout_episode(
    env,
    policy,
    evidence_provider: Callable,
    *,
    episode_id: str,
    seed: int = 42,
    deterministic: bool = True,
    temperature: float = 1.0,
    maximum_steps: int = 30,
) -> dict:
    """Execute one genuinely on-policy episode in a fresh environment."""
    if not episode_id or maximum_steps <= 0:
        raise ValueError("episode_id and a positive maximum_steps are required")
    observation, reset_info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    transitions = []
    terminated = truncated = False
    for turn in range(maximum_steps):
        mask = np.asarray(env.get_action_mask())
        decision = policy.select_action(
            observation,
            mask,
            deterministic=deterministic,
            rng=None if deterministic else rng,
            temperature=temperature,
        )
        action_idx = int(decision["action_idx"])
        illegal = not (0 <= action_idx < len(RL_ACTION_SPACE)) or not bool(mask[action_idx])
        if illegal:
            raise ValueError("learned policy selected an illegal action")
        action_name = RL_ACTION_SPACE[action_idx]
        if action_name == "conclude_interview":
            next_observation, reward, terminated, truncated, info = env.step_with_scores(
                action_idx, None, None, None
            )
            evidence = None
        else:
            target_skill = env.select_target_skill(action_idx)
            evidence = evidence_provider(
                env=env,
                episode_id=episode_id,
                turn_id=turn,
                action_idx=action_idx,
                action_name=action_name,
                target_skill=target_skill,
            )
            semantic, behavior, cognitive = _evidence_values(evidence)
            next_observation, reward, terminated, truncated, info = env.step_with_scores(
                action_idx, semantic, behavior, cognitive
            )
        transitions.append({
            "schema_version": ROLLOUT_SCHEMA_VERSION,
            "episode_id": episode_id,
            "turn_id": turn,
            "source_split": "fresh_rollout",
            "checkpoint_sha256": policy.checkpoint_sha256,
            "obs": np.asarray(observation, dtype=float).tolist(),
            "next_obs": np.asarray(next_observation, dtype=float).tolist(),
            "action_idx": action_idx,
            "action_name": action_name,
            "action_mask_before": np.asarray(mask, dtype=int).tolist(),
            "target_policy_action_probs": decision["action_probabilities"],
            "target_policy_action_probability": decision["selected_action_probability"],
            "action_selection_source": "learned_iql_policy",
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": dict(info or {}),
        })
        observation = next_observation
        if terminated or truncated:
            break
    if not transitions or not (terminated or truncated):
        raise RuntimeError("fresh rollout did not reach a terminal or truncated state")
    return {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "episode_id": episode_id,
        "seed": int(seed),
        "checkpoint_sha256": policy.checkpoint_sha256,
        "reset_info": dict(reset_info or {}),
        "transitions": transitions,
    }


def evaluate_fresh_rollouts(
    episodes: Iterable[dict],
    *,
    checkpoint_sha256: str,
    protocol_hash: str,
    belief_config_hash: str,
    output_file: str | Path | None = None,
) -> dict:
    episodes = list(episodes)
    if not episodes:
        raise ValueError("fresh policy evaluation requires at least one episode")
    all_rows = []
    episode_rewards, episode_lengths = [], []
    terminal_reasons = Counter()
    episode_ids = set()
    for episode in episodes:
        if episode.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
            raise ValueError("unsupported fresh-rollout schema")
        if episode.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("rollout checkpoint hash mismatch")
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id or episode_id in episode_ids:
            raise ValueError("fresh rollout episode IDs must be non-empty and unique")
        episode_ids.add(episode_id)
        rows = episode.get("transitions")
        if not isinstance(rows, list) or not rows:
            raise ValueError("fresh rollout contains no transitions")
        for index, row in enumerate(rows):
            if row.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
                raise ValueError("unsupported fresh-rollout transition schema")
            if row.get("episode_id") != episode_id or row.get("turn_id") != index:
                raise ValueError("fresh rollout transition identity or order mismatch")
            if row.get("source_split") != "fresh_rollout":
                raise ValueError("learned-policy evaluation rejects dataset split input")
            if row.get("action_selection_source") != "learned_iql_policy":
                raise ValueError("rollout contains an action not selected by the IQL policy")
            if row.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError("transition checkpoint hash mismatch")
            if index < len(rows) - 1 and (row.get("terminated") or row.get("truncated")):
                raise ValueError("fresh rollout contains transitions after termination")
        last = rows[-1]
        if not (last.get("terminated") or last.get("truncated")):
            raise ValueError("fresh rollout is incomplete")
        all_rows.extend(rows)
        episode_rewards.append(sum(float(row["reward"]) for row in rows))
        episode_lengths.append(len(rows))
        reason = last.get("info", {}).get("termination_reason")
        terminal_reasons[str(reason or ("terminated" if last.get("terminated") else "truncated"))] += 1

    illegal_actions = invalid_probability_rows = 0
    action_counts = Counter()
    for row in all_rows:
        action = row.get("action_idx")
        mask = np.asarray(row.get("action_mask_before"))
        probabilities = np.asarray(row.get("target_policy_action_probs"), dtype=float)
        if (
            not isinstance(action, int)
            or mask.shape != (len(RL_ACTION_SPACE),)
            or not np.all(np.isin(mask, (0, 1, False, True)))
            or action < 0
            or action >= len(RL_ACTION_SPACE)
            or not bool(mask[action])
            or row.get("action_name") != RL_ACTION_SPACE[action]
        ):
            illegal_actions += 1
        selected_probability = row.get("target_policy_action_probability")
        if (
            probabilities.shape != (len(RL_ACTION_SPACE),)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0)
            or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-9)
            or (mask.shape == probabilities.shape and np.any(probabilities[~mask.astype(bool)] != 0))
            or not isinstance(selected_probability, (int, float))
            or (isinstance(action, int) and 0 <= action < len(RL_ACTION_SPACE)
                and not math.isclose(float(selected_probability), float(probabilities[action]), abs_tol=1e-12))
        ):
            invalid_probability_rows += 1
        if isinstance(action, int) and 0 <= action < len(RL_ACTION_SPACE):
            action_counts[RL_ACTION_SPACE[action]] += 1

    rewards = np.asarray(episode_rewards, dtype=float)
    lengths = np.asarray(episode_lengths, dtype=float)
    report = {
        "schema_version": ROLLOUT_REPORT_VERSION,
        "producer_version": ROLLOUT_REPORT_VERSION,
        "supported_consumer_versions": [ROLLOUT_REPORT_VERSION],
        "evaluation_type": "learned_policy_rollout",
        "evaluation_method": "fresh-environment-on-policy",
        "fresh_rollouts": True,
        "evaluates_learned_policy": True,
        "uses_locked_test": False,
        "release_gate": False,
        "checkpoint_hash": checkpoint_sha256,
        "protocol_hash": protocol_hash,
        "belief_config_hash": belief_config_hash,
        "num_episodes": len(episodes),
        "num_transitions": len(all_rows),
        "completed_episodes": len(episodes),
        "completion_rate": 1.0,
        "illegal_action_count": illegal_actions,
        "invalid_probability_row_count": invalid_probability_rows,
        "episode_reward": {
            "mean": float(rewards.mean()), "std": float(rewards.std()),
            "minimum": float(rewards.min()), "maximum": float(rewards.max()),
        },
        "episode_length": {
            "mean": float(lengths.mean()), "minimum": int(lengths.min()),
            "maximum": int(lengths.max()),
        },
        "termination_reasons": dict(sorted(terminal_reasons.items())),
        "action_counts": {name: action_counts.get(name, 0) for name in RL_ACTION_SPACE},
        "limitations": [
            "Fresh synthetic rollouts do not establish real-candidate performance.",
            "This report is diagnostic and is not a replacement for the v7 locked-test release gate.",
        ],
    }
    report["report_hash"] = canonical_json_hash(report)
    if output_file is not None:
        atomic_json_write(Path(output_file), report)
    return report
