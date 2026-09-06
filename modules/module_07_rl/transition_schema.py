"""Versioned contract for generated and replayed offline-RL transitions."""

TRANSITION_SCHEMA_VERSION = "aria-transition-v4"
GENERATOR_SCHEMA_VERSION = "aria-simulator-v6"
FALLBACK_QUESTION_TEMPLATE_VERSION = "aria-grounded-fallback-v1"
QUESTION_GENERATION_MODES = frozenset({
    "llm",
    "deterministic_grounded_fallback",
})


def is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def has_valid_question_generation_provenance(transition: dict) -> bool:
    """Check mutually exclusive provenance for an accepted question."""
    mode = transition.get("question_generation_mode")
    if mode not in QUESTION_GENERATION_MODES:
        return False

    attempts = transition.get("question_generation_attempts")
    llm_attempts = transition.get("llm_question_generation_attempts")
    deterministic_attempts = transition.get(
        "deterministic_question_generation_attempts"
    )
    if (
        not is_plain_int(attempts)
        or not is_plain_int(llm_attempts)
        or attempts != llm_attempts
        or not 1 <= llm_attempts <= 3
    ):
        return False

    if mode == "llm":
        prompt_hash = transition.get("question_prompt_hash")
        return (
            transition.get("fallback_question_template_version") in (None, "")
            and isinstance(prompt_hash, str)
            and bool(prompt_hash.strip())
            and is_plain_int(transition.get("question_generation_seed"))
            and deterministic_attempts in (None, 0)
        )

    return (
        transition.get("fallback_question_template_version")
        == FALLBACK_QUESTION_TEMPLATE_VERSION
        and transition.get("question_prompt_hash") in (None, "")
        and transition.get("question_generation_seed") is None
        and llm_attempts == 3
        and deterministic_attempts == 1
    )


REQUIRED_POLICY_FIELDS = (
    "action_mask_before",
    "behavior_action_probs",
    "behavior_action_probability",
    "transition_kind",
    "generator_schema_version",
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
