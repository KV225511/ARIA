"""Permutation-invariant state features for ARIA's offline-RL policy."""

from __future__ import annotations

import math
import numpy as np

from modules.module_07_rl.rl_spec import RL_ACTION_SPACE


STATE_SCHEMA_VERSION = "aria-state-v2"
COGNITIVE_LABELS = ("low", "anxiety", "ignorance", "confident_ignorance")
STATE_FEATURE_NAMES = (
    "global_p_beginner", "global_p_mid", "global_p_expert",
    "assessment_entropy_normalized", "mean_visited_entropy_normalized",
    "skill_coverage_fraction", "turn_fraction",
    "focus_p_beginner", "focus_p_mid", "focus_p_expert",
    "focus_ess_fraction", "consecutive_focus_fraction",
    "previous_semantic_score", "previous_evidence_reliability",
    "previous_behavior_score",
    "previous_cognitive_low", "previous_cognitive_anxiety",
    "previous_cognitive_ignorance", "previous_cognitive_confident_ignorance",
    "previous_incongruence_score",
    *(f"previous_action_{name}" for name in RL_ACTION_SPACE),
    "semantic_available", "behavior_available", "cognitive_available",
    "incongruence_available",
)
STATE_DIM = len(STATE_FEATURE_NAMES)


def _finite_clip(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(np.clip(number, 0.0, 1.0))


def build_policy_state(
    updater,
    total_skills: int,
    turn_id: int,
    current_skill: str | None = None,
    consecutive_focus_turns: int = 0,
    previous: dict | None = None,
) -> np.ndarray:
    """Build a fixed state using only information available before the action."""
    previous = previous or {}
    assessment = updater.get_aggregate_assessment()
    visited = len(assessment["visited_skills"])
    denominator = max(int(total_skills), 1)
    focus_belief = updater.get_belief(current_skill)
    focus_ess = updater.get_effective_sample_size(current_skill)

    cognitive = [0.0] * len(COGNITIVE_LABELS)
    cognitive_label = previous.get("cognitive_load")
    if cognitive_label in COGNITIVE_LABELS:
        cognitive[COGNITIVE_LABELS.index(cognitive_label)] = 1.0

    previous_action = [0.0] * len(RL_ACTION_SPACE)
    action_idx = previous.get("action_idx")
    if isinstance(action_idx, int) and 0 <= action_idx < len(previous_action):
        previous_action[action_idx] = 1.0

    availability = [
        float(previous.get("semantic_score") is not None),
        float(previous.get("behavior_score") is not None),
        float(cognitive_label in COGNITIVE_LABELS),
        float(previous.get("incongruence_score") is not None),
    ]
    features = np.asarray([
        *assessment["belief"],
        updater.get_assessment_entropy() / np.log(3.0),
        updater.get_mean_visited_entropy() / np.log(3.0),
        min(visited / denominator, 1.0),
        min(max(turn_id, 0) / 30.0, 1.0),
        *focus_belief,
        min(focus_ess / updater.config.max_skill_effective_sample_size, 1.0),
        min(max(consecutive_focus_turns, 0) / 10.0, 1.0),
        _finite_clip(previous.get("semantic_score")),
        _finite_clip(previous.get("evidence_reliability")),
        _finite_clip(previous.get("behavior_score")),
        *cognitive,
        _finite_clip(previous.get("incongruence_score")),
        *previous_action,
        *availability,
    ], dtype=np.float32)
    if features.shape != (STATE_DIM,):
        raise RuntimeError(f"State schema produced {len(features)} != {STATE_DIM} values")
    return features


def build_action_mask(env) -> np.ndarray:
    mask = np.ones(len(RL_ACTION_SPACE), dtype=np.float32)
    conclude_index = RL_ACTION_SPACE.index("conclude_interview")
    if not env.can_conclude():
        mask[conclude_index] = 0.0
    return mask
