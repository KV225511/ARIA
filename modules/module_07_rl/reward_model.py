"""Single-source reward functions for online steps and immutable replay."""

from __future__ import annotations

from modules.module_07_rl.rl_spec import REWARD_COEFFICIENTS


REWARD_SCHEMA_VERSION = "aria-reward-v2"


def compute_step_reward(
    information_gain: float,
    first_skill_visit: bool,
    previous_skill_count: int,
    cognitive_load: str,
    invalid_action: bool = False,
    coefficients=None,
) -> float:
    coefficients = coefficients or REWARD_COEFFICIENTS
    coverage_bonus = coefficients.get("zeta", 0.0) if first_skill_visit else 0.0
    redundancy_penalty = 0.05 * min(max(previous_skill_count, 0), 3)
    distress_penalty = coefficients.get("delta", 0.0) if cognitive_load == "anxiety" else 0.0
    invalid_penalty = 0.5 if invalid_action else 0.0
    return float(
        coefficients["alpha"] * max(float(information_gain), 0.0)
        + coverage_bonus
        - coefficients["beta"]
        - redundancy_penalty
        - distress_penalty
        - invalid_penalty
    )


def terminal_outcome_delta(true_label, predicted_label, omega: float) -> float:
    if predicted_label is None:
        return -0.25 * float(omega)
    distance = abs(int(true_label) - int(predicted_label))
    if distance == 0:
        return float(omega)
    if distance == 1:
        return -0.5 * float(omega)
    return -float(omega)


def apply_terminal_outcome_once(transition: dict, omega: float) -> bool:
    """Shape one terminal transition and refuse accidental double application."""
    if not transition.get("done") or "true_label" not in transition:
        return False
    if transition.get("terminal_outcome_reward_applied"):
        raise ValueError("Terminal outcome reward was already applied")
    predicted = transition.get("aria_label")
    transition["base_reward"] = float(transition.get("reward", 0.0))
    transition["terminal_outcome_delta"] = terminal_outcome_delta(
        transition["true_label"], predicted, omega
    )
    transition["reward"] = (
        transition["base_reward"] + transition["terminal_outcome_delta"]
    )
    transition["terminal_outcome_reward_applied"] = True
    return predicted != transition["true_label"]
