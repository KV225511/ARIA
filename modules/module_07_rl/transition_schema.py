"""Versioned contract for generated and replayed offline-RL transitions."""

TRANSITION_SCHEMA_VERSION = "aria-transition-v4"

REQUIRED_POLICY_FIELDS = (
    "action_mask_before",
    "behavior_action_probs",
    "behavior_action_probability",
    "transition_kind",
    "role_profile_hash",
    "role_profile_schema_version",
    "grounding_contract_hash",
    "ontology_hash",
    "target_skill_id",
    "question_grounding_schema_version",
    "question_grounding_valid",
    "pairing_record",
    "generation_run_id",
    "plan_id",
)
