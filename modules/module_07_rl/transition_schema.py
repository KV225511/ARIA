"""Versioned contract for generated and replayed offline-RL transitions."""

TRANSITION_SCHEMA_VERSION = "aria-transition-v3"

REQUIRED_POLICY_FIELDS = (
    "action_mask_before",
    "behavior_action_probs",
    "behavior_action_probability",
    "transition_kind",
)
