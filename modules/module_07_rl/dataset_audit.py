"""Stage-specific quality gates for ARIA evidence, beliefs, and offline RL."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from modules.module_07_rl.dataset_split import (
    connected_identity_components,
    group_transitions_into_episodes,
    split_by_resume_jd_group,
)
from modules.module_07_rl.transition_schema import (
    GENERATOR_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    has_valid_question_generation_provenance,
    is_plain_int,
)
from modules.module_05_ontology.grounding import (
    GROUNDING_POLICY_VERSION,
    GROUNDING_SCHEMA_VERSION,
    ROLE_PROFILE_SCHEMA_VERSION,
    grounding_contract_hash,
)
from modules.module_07_rl.generation_policy import (
    BEHAVIOR_POLICY_VERSION,
    HARD_MAX_SWITCH_SHARE,
    HARD_MAX_TURN_RATE,
    HARD_NO_OVERLAP_BAND,
    MIN_DISTRIBUTION_GATE_EPISODES,
    PAIR_PLAN_SCHEMA_VERSION,
    TARGET_MAX_TURN_RATE,
    TARGET_NO_OVERLAP_BAND,
    TARGET_SWITCH_SHARE,
)


MIN_QUALITY_GATE_EPISODES = 200
MAX_DETERMINISTIC_FALLBACK_RATE = 0.10
MAX_CROSS_COMPONENT_FALLBACK_DUPLICATE_RATE = 0.25


def audit_generation_distribution(transitions: list[dict]) -> dict:
    """Measure policy and pair-plan targets without counting stop decisions as questions."""
    episodes = group_transitions_into_episodes(transitions)
    questions = [
        item for item in transitions
        if item.get("transition_kind") != "stop" and item.get("action_idx") != 7
    ]
    action_counts = Counter(
        item.get("action_name") or f"action_{item.get('action_idx')}"
        for item in questions
    )
    switch_count = sum(
        item.get("action_name") == "switch_topic" or item.get("action_idx") == 3
        for item in questions
    )
    switch_share = switch_count / len(questions) if questions else 0.0
    terminal = _terminal_records(transitions)
    max_turn_count = sum(item.get("termination_reason") == "max_turns" for item in terminal)
    max_turn_rate = max_turn_count / len(terminal) if terminal else 0.0
    pairing = Counter()
    blockers = Counter()
    stratified = {
        "persona": defaultdict(Counter),
        "dataset_split": defaultdict(Counter),
        "ontology_size_band": defaultdict(Counter),
    }
    derived_splits = {}
    if episodes and not all(
        episode and episode[0].get("dataset_split") for episode in episodes
    ):
        for split_name, split_transitions in split_by_resume_jd_group(transitions).items():
            for item in split_transitions:
                derived_splits[item.get("episode_id")] = split_name
    for episode in episodes:
        first = episode[0] if episode else {}
        pairing_record = next((item.get("pairing_record") for item in episode if item.get("pairing_record")), {})
        if pairing_record.get("pairing_class"):
            pairing[pairing_record["pairing_class"]] += 1
        final = episode[-1] if episode else {}
        ontology_size = int(first.get("ontology_size", 0) or 0)
        size_band = "1-5" if ontology_size <= 5 else "6-10" if ontology_size <= 10 else "11+"
        persona = first.get("persona_tier") or {
            0: "BEGINNER", 1: "MID", 2: "EXPERT",
        }.get(first.get("true_label"), "unknown")
        dimensions = {
            "persona": str(persona),
            "dataset_split": str(
                first.get("dataset_split")
                or derived_splits.get(first.get("episode_id"))
                or "unassigned"
            ),
            "ontology_size_band": size_band,
        }
        episode_questions = [
            item for item in episode
            if item.get("transition_kind") != "stop" and item.get("action_idx") != 7
        ]
        episode_switches = sum(
            item.get("action_name") == "switch_topic" or item.get("action_idx") == 3
            for item in episode_questions
        )
        for dimension, key in dimensions.items():
            row = stratified[dimension][key]
            row["episodes"] += 1
            row["questions"] += len(episode_questions)
            row["switch_topic"] += episode_switches
            row["max_turns"] += int(final.get("termination_reason") == "max_turns")
            row["no_evidence_overlap"] += int(
                pairing_record.get("pairing_class") == "no_evidence_overlap"
            )
        if final.get("termination_reason") == "max_turns":
            status_before = final.get("conclusion_status_before") or final.get("conclusion_status") or {}
            if isinstance(status_before, dict):
                for name, status in status_before.items():
                    if (
                        name != "can_conclude"
                        and isinstance(status, dict)
                        and not status.get("ready", False)
                    ):
                        blockers[name] += 1
    paired_episodes = sum(pairing.values())
    no_overlap_rate = (
        pairing["no_evidence_overlap"] / paired_episodes if paired_episodes else 0.0
    )
    current_contract = bool(transitions) and all(
        item.get("generator_schema_version") == GENERATOR_SCHEMA_VERSION
        for item in transitions
    )
    has_sample = len(terminal) >= MIN_DISTRIBUTION_GATE_EPISODES
    complete_current_contract = (
        current_contract
        and len(terminal) == len(episodes)
        and paired_episodes == len(episodes)
    )
    stratified_report = {}
    for dimension, rows in stratified.items():
        stratified_report[dimension] = {
            key: {
                **dict(values),
                "switch_topic_question_share": (
                    values["switch_topic"] / values["questions"]
                    if values["questions"] else None
                ),
                "max_turn_episode_rate": values["max_turns"] / values["episodes"],
                "no_evidence_overlap_rate": (
                    values["no_evidence_overlap"] / values["episodes"]
                ),
            }
            for key, values in sorted(rows.items())
        }
    target = (
        TARGET_SWITCH_SHARE[0] <= switch_share <= TARGET_SWITCH_SHARE[1]
        and max_turn_rate <= TARGET_MAX_TURN_RATE
        and TARGET_NO_OVERLAP_BAND[0] <= no_overlap_rate <= TARGET_NO_OVERLAP_BAND[1]
    )
    hard = (
        switch_share <= HARD_MAX_SWITCH_SHARE
        and max_turn_rate <= HARD_MAX_TURN_RATE
        and HARD_NO_OVERLAP_BAND[0] <= no_overlap_rate <= HARD_NO_OVERLAP_BAND[1]
    )
    return {
        "gate": "generation_distribution",
        "num_episodes": len(episodes),
        "question_action_counts": dict(action_counts),
        "switch_topic_question_share": switch_share,
        "max_turn_episode_count": max_turn_count,
        "max_turn_episode_rate": max_turn_rate,
        "pairing_class_counts": dict(pairing),
        "no_evidence_overlap_rate": no_overlap_rate,
        "max_turn_conclusion_blockers": dict(blockers),
        "stratified_metrics": stratified_report,
        "target_metrics": {
            "switch_topic_question_share": list(TARGET_SWITCH_SHARE),
            "maximum_turn_rate": TARGET_MAX_TURN_RATE,
            "no_evidence_overlap_rate": list(TARGET_NO_OVERLAP_BAND),
        },
        "distribution_contract_complete": complete_current_contract,
        "meets_target_metrics": (
            target and complete_current_contract if has_sample and current_contract else None
        ),
        "passes_distribution_gates": (
            hard and complete_current_contract if has_sample and current_contract else None
        ),
    }


def _valid_grounding_provenance(item: dict) -> bool:
    grounding = item.get("question_grounding")
    attempts = item.get("question_generation_attempts")
    return (
        isinstance(item.get("pairing_record"), dict)
        and bool(item.get("target_skill_id"))
        and isinstance(grounding, dict)
        and grounding.get("schema_version") == GROUNDING_SCHEMA_VERSION
        and grounding.get("grounding_policy_version") == GROUNDING_POLICY_VERSION
        and grounding.get("target_skill_id") == item.get("target_skill_id")
        and grounding.get("role_profile_hash") == item.get("role_profile_hash")
        and grounding.get("decision") == "accept"
        and grounding.get("valid") is True
        and item.get("question_grounding_valid") is True
        and is_plain_int(attempts)
        and 1 <= attempts <= 3
    )


def _attempt_bucket(value):
    return value if is_plain_int(value) else "invalid"


def _summary(values):
    finite = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(finite, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _terminal_records(transitions):
    return [episode[-1] for episode in group_transitions_into_episodes(transitions) if episode]


def _identity(transition, kind):
    return transition.get(f"{kind}_content_hash") or transition.get(f"{kind}_file")


def _identity_diagnostics(transitions):
    split_owners = {"resume": defaultdict(set), "jd": defaultdict(set)}
    names_by_hash = {"resume": defaultdict(set), "jd": defaultdict(set)}
    for transition in transitions:
        split_name = transition.get("dataset_split")
        for kind in ("resume", "jd"):
            identity = _identity(transition, kind)
            if split_name and identity:
                split_owners[kind][identity].add(split_name)
            content_hash = transition.get(f"{kind}_content_hash")
            filename = transition.get(f"{kind}_file")
            if content_hash and filename:
                names_by_hash[kind][content_hash].add(filename)
    leaking = {
        kind: sorted(identity for identity, owners in values.items() if len(owners) > 1)
        for kind, values in split_owners.items()
    }
    renamed_duplicates = {
        kind: {
            content_hash: sorted(names)
            for content_hash, names in hashes.items()
            if len(names) > 1
        }
        for kind, hashes in names_by_hash.items()
    }
    return leaking, renamed_duplicates


def _score_separation(transitions):
    scores = defaultdict(list)
    for transition in transitions:
        try:
            label = int(transition["true_label"])
            value = float(transition["semantic_score"])
        except (KeyError, TypeError, ValueError):
            continue
        if label in (0, 1, 2) and math.isfinite(value):
            scores[label].append(value)
    summaries = {str(label): _summary(scores[label]) for label in (0, 1, 2)}
    effects = {}
    for left, right in ((0, 1), (1, 2)):
        left_values, right_values = scores[left], scores[right]
        if not left_values or not right_values:
            effects[f"{left}_vs_{right}"] = None
            continue
        pooled = math.sqrt((np.var(left_values) + np.var(right_values)) / 2.0)
        difference = float(np.mean(right_values) - np.mean(left_values))
        effects[f"{left}_vs_{right}"] = (
            math.inf if pooled == 0.0 and difference > 0.0
            else 0.0 if pooled == 0.0
            else difference / pooled
        )
    return summaries, effects


def audit_raw_evidence(
    transitions: list[dict],
    min_episodes: int = MIN_QUALITY_GATE_EPISODES,
    min_independent_components: int = 3,
    allow_legacy: bool = False,
):
    episodes = group_transitions_into_episodes(transitions)
    terminal = _terminal_records(transitions)
    true_counts = Counter(
        int(item["true_label"]) for item in terminal if item.get("true_label") in (0, 1, 2)
    )
    question_transitions = [
        item for item in transitions
        if item.get("transition_kind") != "stop" and item.get("action_idx") != 7
    ]
    invalid = sum(item.get("evaluation_valid") is False for item in question_transitions)
    missing_validity = sum("evaluation_valid" not in item for item in question_transitions)
    contains_current_contract = any(
        item.get("transition_schema_version") == TRANSITION_SCHEMA_VERSION
        for item in transitions
    )
    grounding_required = not allow_legacy or contains_current_contract
    invalid_grounding = (
        sum(item.get("question_grounding_valid") is not True for item in question_transitions)
        if grounding_required else 0
    )
    missing_role_profiles = (
        sum(
            not item.get("role_profile_hash")
            or not item.get("ontology_hash")
            or not item.get("grounding_contract_hash")
            for item in question_transitions
        )
        if grounding_required else 0
    )
    current_contract = {
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "behavior_policy_version": BEHAVIOR_POLICY_VERSION,
        "pair_plan_schema_version": PAIR_PLAN_SCHEMA_VERSION,
        "role_profile_schema_version": ROLE_PROFILE_SCHEMA_VERSION,
        "question_grounding_schema_version": GROUNDING_SCHEMA_VERSION,
        "grounding_contract_hash": grounding_contract_hash(),
    }
    invalid_contract_provenance = (
        sum(
            any(item.get(field) != required for field, required in current_contract.items())
            for item in transitions
        )
        if grounding_required else 0
    )
    invalid_grounding_provenance = (
        sum(
            not _valid_grounding_provenance(item)
            for item in question_transitions
        )
        if grounding_required else 0
    )
    nonfinite_scores = 0
    missing_scores = 0
    for transition in question_transitions:
        if "semantic_score" not in transition:
            missing_scores += 1
            continue
        try:
            value = float(transition["semantic_score"])
        except (TypeError, ValueError):
            nonfinite_scores += 1
            continue
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            nonfinite_scores += 1

    candidate_models = {
        item.get("candidate_model") for item in transitions if item.get("candidate_model")
    }
    evaluator_models = {
        item.get("evaluator_model") for item in transitions if item.get("evaluator_model")
    }
    leaking, renamed_duplicates = _identity_diagnostics(transitions)
    components = connected_identity_components(transitions)
    score_summaries, adjacent_effects = _score_separation(transitions)
    generation_attempt_counts = Counter(
        _attempt_bucket(item.get("question_generation_attempts"))
        for item in question_transitions
    )
    retried_questions = sum(
        is_plain_int(item.get("question_generation_attempts"))
        and item["question_generation_attempts"] > 1
        for item in question_transitions
    )
    fallback_questions = sum(
        item.get("question_generation_mode") == "deterministic_grounded_fallback"
        for item in question_transitions
    )
    fallback_rate = (
        fallback_questions / len(question_transitions) if question_transitions else 0.0
    )
    invalid_generation_mode_provenance = (
        sum(
            not has_valid_question_generation_provenance(item)
            for item in question_transitions
        )
        if grounding_required else 0
    )

    episode_components = {}
    for component_index, component_episodes in enumerate(components):
        for episode in component_episodes:
            if episode and episode[0].get("episode_id") is not None:
                episode_components[str(episode[0]["episode_id"])] = component_index
    fallback_occurrences = defaultdict(list)
    for item in question_transitions:
        if item.get("question_generation_mode") != "deterministic_grounded_fallback":
            continue
        normalized = " ".join(str(item.get("question") or "").casefold().split())
        if not normalized:
            continue
        question_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        fallback_occurrences[question_hash].append({
            "episode_id": str(item.get("episode_id")),
            "component": episode_components.get(str(item.get("episode_id"))),
            "dataset_split": item.get("dataset_split"),
        })
    cross_component_fallback_duplicates = {}
    cross_component_fallback_duplicate_count = 0
    for question_hash, occurrences in fallback_occurrences.items():
        component_ids = {
            item["component"] for item in occurrences if item["component"] is not None
        }
        if len(component_ids) <= 1:
            continue
        cross_component_fallback_duplicate_count += len(occurrences) - 1
        cross_component_fallback_duplicates[question_hash] = {
            "occurrences": len(occurrences),
            "identity_components": sorted(component_ids),
            "dataset_splits": sorted({
                item["dataset_split"] for item in occurrences
                if item["dataset_split"] is not None
            }),
        }
    cross_component_fallback_duplicate_rate = (
        cross_component_fallback_duplicate_count / fallback_questions
        if fallback_questions else 0.0
    )
    pairing_classes = Counter()
    duplicate_questions = 0
    inconsistent_episode_grounding = 0
    distinct_targets_per_episode = []
    for episode in episodes:
        first_pairing = next(
            (item.get("pairing_record") for item in episode if item.get("pairing_record")),
            {},
        )
        if first_pairing.get("pairing_class"):
            pairing_classes[first_pairing["pairing_class"]] += 1
        questions = [
            " ".join(str(item.get("question", "")).casefold().split())
            for item in episode if item.get("question")
        ]
        duplicate_questions += len(questions) - len(set(questions))
        if grounding_required:
            profiles = {item.get("role_profile_hash") for item in episode}
            ontologies = {item.get("ontology_hash") for item in episode}
            if len(profiles - {None}) != 1 or len(ontologies - {None}) != 1:
                inconsistent_episode_grounding += 1
        distinct_targets_per_episode.append(len({
            item.get("target_skill_id") for item in episode
            if item.get("target_skill_id")
        }))
    distribution = audit_generation_distribution(transitions)
    warnings = []
    if len(episodes) < min_episodes:
        warnings.append(
            f"Dataset contains {len(episodes)} episodes; quality gate requires at least {min_episodes}."
        )
    if len(components) < min_independent_components:
        warnings.append(
            f"Dataset has {len(components)} independent identity components; at least {min_independent_components} are required."
        )
    if len(true_counts) != 3:
        warnings.append("Terminal ground truth does not contain all three classes.")
    elif max(true_counts.values()) - min(true_counts.values()) > max(1, 0.05 * len(terminal)):
        warnings.append("Terminal true-label distribution differs by more than 5%.")
    if invalid or missing_validity:
        warnings.append("Dataset contains invalid or unverified evaluator outputs.")
    if invalid_grounding or missing_role_profiles or invalid_grounding_provenance:
        warnings.append("Dataset contains invalid or unverified question grounding.")
    if invalid_contract_provenance:
        warnings.append("Dataset contains stale, missing, or mixed grounding contracts.")
    if fallback_rate > MAX_DETERMINISTIC_FALLBACK_RATE:
        warnings.append(
            "More than 10% of questions required deterministic grounding fallback."
        )
    if invalid_generation_mode_provenance:
        warnings.append(
            "Dataset contains invalid question-generation mode provenance."
        )
    if (
        cross_component_fallback_duplicate_rate
        > MAX_CROSS_COMPONENT_FALLBACK_DUPLICATE_RATE
    ):
        warnings.append(
            "More than 25% of deterministic fallback questions are exact "
            "duplicates across independent identity components."
        )
    if duplicate_questions:
        warnings.append("Dataset contains repeated questions within an episode.")
    if inconsistent_episode_grounding:
        warnings.append("Role-profile or ontology provenance changes within an episode.")
    if nonfinite_scores or missing_scores:
        warnings.append("Dataset contains missing, non-finite, or out-of-range semantic scores.")
    if candidate_models & evaluator_models:
        warnings.append("Candidate and evaluator model sets overlap.")
    if leaking["resume"] or leaking["jd"]:
        warnings.append("Resume or JD identities occur in more than one dataset split.")
    if renamed_duplicates["resume"] or renamed_duplicates["jd"]:
        warnings.append("Renamed duplicate resume or JD documents were detected by content hash.")
    class_means = [score_summaries[str(label)]["mean"] for label in (0, 1, 2)]
    if all(value is not None for value in class_means) and not (
        class_means[0] <= class_means[1] <= class_means[2]
    ):
        warnings.append("Semantic-score class means are not monotonically ordered.")
    weak_pairs = [
        name for name, effect in adjacent_effects.items()
        if effect is not None and effect < 0.20
    ]
    if weak_pairs:
        warnings.append("Adjacent semantic-score classes have negligible standardized separation.")
    if distribution["passes_distribution_gates"] is False:
        warnings.append("Generation distribution exceeds hard policy or pairing limits.")
    elif distribution["meets_target_metrics"] is False:
        warnings.append("Generation distribution misses one or more production target metrics.")
    return {
        "gate": "raw_evidence",
        "num_transitions": len(transitions),
        "num_episodes": len(episodes),
        "independent_identity_components": len(components),
        "terminal_true_label_counts": dict(true_counts),
        "semantic_scores_by_class": score_summaries,
        "adjacent_standardized_effects": adjacent_effects,
        "invalid_evaluations": invalid,
        "missing_evaluation_validity": missing_validity,
        "invalid_question_grounding": invalid_grounding,
        "missing_role_profile_provenance": missing_role_profiles,
        "invalid_grounding_provenance": invalid_grounding_provenance,
        "invalid_contract_provenance": invalid_contract_provenance,
        "question_generation_attempt_counts": dict(generation_attempt_counts),
        "question_grounding_retry_rate": (
            retried_questions / len(question_transitions) if question_transitions else None
        ),
        "deterministic_grounding_fallback_count": fallback_questions,
        "deterministic_grounding_fallback_rate": fallback_rate,
        "maximum_deterministic_grounding_fallback_rate": (
            MAX_DETERMINISTIC_FALLBACK_RATE
        ),
        "invalid_generation_mode_provenance": invalid_generation_mode_provenance,
        "cross_component_duplicate_fallback_question_count": (
            cross_component_fallback_duplicate_count
        ),
        "cross_component_duplicate_fallback_question_rate": (
            cross_component_fallback_duplicate_rate
        ),
        "maximum_cross_component_fallback_duplicate_rate": (
            MAX_CROSS_COMPONENT_FALLBACK_DUPLICATE_RATE
        ),
        "cross_component_duplicate_fallback_questions": (
            cross_component_fallback_duplicates
        ),
        "pairing_class_counts": dict(pairing_classes),
        "generation_distribution": distribution,
        "duplicate_questions_within_episode": duplicate_questions,
        "inconsistent_episode_grounding": inconsistent_episode_grounding,
        "distinct_targets_per_episode": _summary(distinct_targets_per_episode),
        "invalid_semantic_scores": nonfinite_scores,
        "missing_semantic_scores": missing_scores,
        "split_leaking_resumes": leaking["resume"],
        "split_leaking_jds": leaking["jd"],
        "renamed_duplicate_resumes": renamed_duplicates["resume"],
        "renamed_duplicate_jds": renamed_duplicates["jd"],
        "candidate_models": sorted(candidate_models),
        "evaluator_models": sorted(evaluator_models),
        "warnings": warnings,
        "passes_quality_gates": not warnings,
    }


def audit_belief_predictions(
    transitions: list[dict],
    collapse_threshold=0.60,
    gate_name="belief_predictions",
):
    terminal = _terminal_records(transitions)
    pairs = [
        (int(item["true_label"]), item.get("aria_label"), item)
        for item in terminal
        if item.get("true_label") in (0, 1, 2)
    ]
    missing_predictions = sum(prediction not in (0, 1, 2) for _, prediction, _ in pairs)
    classified = [(truth, int(prediction)) for truth, prediction, _ in pairs if prediction in (0, 1, 2)]
    true_counts = Counter(truth for truth, _, _ in pairs)
    predicted_counts = Counter(prediction for _, prediction in classified)
    accuracy = (
        float(np.mean([
            prediction in (0, 1, 2) and int(prediction) == truth
            for truth, prediction, _ in pairs
        ]))
        if pairs else None
    )
    per_class = {}
    f1_values = []
    recalls = []
    for label in (0, 1, 2):
        tp = sum(truth == label and pred == label for truth, pred in classified)
        fp = sum(truth != label and pred == label for truth, pred in classified)
        fn = sum(truth == label and pred != label for truth, pred in classified) + sum(
            truth == label and prediction not in (0, 1, 2)
            for truth, prediction, _ in pairs
        )
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
        recalls.append(recall)
    max_share = max(predicted_counts.values(), default=0) / max(len(pairs), 1)
    warnings = []
    if max_share > collapse_threshold:
        warnings.append("More than 60% of terminal predictions collapse to one class.")
    if len(predicted_counts) < 3:
        warnings.append("Terminal predictions do not contain all three classes.")
    from modules.module_07_rl.metrics import (
        component_bootstrap_intervals,
        compute_response_metrics,
    )

    beliefs = [item.get("aggregate_belief") for _, _, item in pairs]
    detailed = compute_response_metrics(
        [prediction for _, prediction, _ in pairs],
        [truth for truth, _, _ in pairs],
        beliefs=beliefs,
    ) if pairs else {}
    return {
        "gate": gate_name,
        "terminal_micro_f1": accuracy,
        "terminal_macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "terminal_balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "terminal_true_label_counts": dict(true_counts),
        "terminal_prediction_counts": dict(predicted_counts),
        "missing_or_abstained_predictions": missing_predictions,
        "per_class": per_class,
        "ordinal_mae": detailed.get("ordinal_mae"),
        "confusion_matrix": detailed.get("confusion_matrix"),
        "expected_calibration_error": detailed.get("expected_calibration_error"),
        "brier_score": detailed.get("brier_score"),
        "component_bootstrap_intervals": component_bootstrap_intervals(transitions),
        "max_prediction_share": max_share,
        "warnings": warnings,
        "passes_quality_gates": not warnings,
    }


def audit_calibration_validation(transitions: list[dict], collapse_threshold=0.60):
    """Validation-only gate used for calibration selection and model training."""
    return audit_belief_predictions(
        transitions,
        collapse_threshold=collapse_threshold,
        gate_name="calibration_validation",
    )


def audit_locked_test(transitions: list[dict], collapse_threshold=0.60):
    """Post-freeze test gate; callers must opt in to unlocking test labels."""
    return audit_belief_predictions(
        transitions,
        collapse_threshold=collapse_threshold,
        gate_name="locked_test",
    )


def audit_learned_policy_evaluation(report: dict):
    """Reject reports that mistake fixed logged actions for policy rollouts."""
    if not isinstance(report, dict):
        return {
            "gate": "learned_policy",
            "warnings": ["Learned-policy input must be a rollout report object."],
            "passes_quality_gates": False,
        }
    warnings = []
    if report.get("evaluation_type") != "learned_policy_rollout":
        warnings.append("Evaluation is not labeled as a learned-policy rollout.")
    if report.get("fresh_rollouts") is not True:
        warnings.append("Learned-policy evaluation requires fresh rollouts.")
    if not report.get("checkpoint_hash"):
        warnings.append("Learned-policy evaluation is missing a checkpoint hash.")
    if int(report.get("num_episodes", 0) or 0) <= 0:
        warnings.append("Learned-policy evaluation contains no rollout episodes.")
    return {
        "gate": "learned_policy",
        "warnings": warnings,
        "passes_quality_gates": not warnings,
    }


def audit_offline_rl_support(transitions: list[dict]):
    action_counts = Counter(
        int(item["action_idx"]) for item in transitions if isinstance(item.get("action_idx"), int)
    )
    missing_actions = [index for index in range(8) if action_counts[index] == 0]
    low_support_actions = [
        index for index in range(8)
        if 0 < action_counts[index] < max(20, 0.01 * len(transitions))
    ]
    nonterminal_rewards = [
        float(item["reward"]) for item in transitions
        if not item.get("done") and "reward" in item and math.isfinite(float(item["reward"]))
    ]
    duplicate_terminal_shaping = sum(
        bool(item.get("terminal_outcome_reward_applied")) and "base_reward" not in item
        for item in transitions
    )
    invalid_masks = 0
    illegal_selected_actions = 0
    invalid_propensities = 0
    inconsistent_selected_propensities = 0
    invalid_stop_transitions = 0
    for item in transitions:
        action_idx = item.get("action_idx")
        mask = np.asarray(item.get("action_mask_before", []), dtype=float)
        probabilities = np.asarray(item.get("behavior_action_probs", []), dtype=float)
        if mask.shape != (8,) or not np.all(np.isin(mask, (0.0, 1.0))):
            invalid_masks += 1
        elif isinstance(action_idx, int) and 0 <= action_idx < 8 and mask[action_idx] != 1.0:
            illegal_selected_actions += 1
        if (
            probabilities.shape != (8,)
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or (
                mask.shape == (8,)
                and np.any(probabilities[mask == 0.0] != 0.0)
            )
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            invalid_propensities += 1
        elif isinstance(action_idx, int) and 0 <= action_idx < 8:
            try:
                logged = float(item["behavior_action_probability"])
            except (KeyError, TypeError, ValueError):
                inconsistent_selected_propensities += 1
            else:
                if logged <= 0.0 or not np.isclose(logged, probabilities[action_idx]):
                    inconsistent_selected_propensities += 1
        if action_idx == 7 and (
            item.get("transition_kind") != "stop"
            or not item.get("done")
            or item.get("semantic_score") is not None
            or item.get("obs") != item.get("next_obs")
        ):
            invalid_stop_transitions += 1
    warnings = []
    if missing_actions:
        warnings.append("Offline dataset has no support for one or more actions.")
    if low_support_actions:
        warnings.append("Offline dataset has weak support for one or more actions.")
    if len(nonterminal_rewards) > 1 and float(np.std(nonterminal_rewards)) < 0.01:
        warnings.append("Non-terminal reward variance is nearly flat.")
    if duplicate_terminal_shaping:
        warnings.append("Terminal reward metadata is inconsistent with single application.")
    if invalid_masks or illegal_selected_actions:
        warnings.append("Pre-action masks are missing, malformed, or contradict selected actions.")
    if invalid_propensities or inconsistent_selected_propensities:
        warnings.append("Behavior-policy propensities are missing, malformed, or inconsistent.")
    if invalid_stop_transitions:
        warnings.append("Stop transitions are non-terminal or contain fabricated evidence/state changes.")
    return {
        "gate": "offline_rl_support",
        "action_counts": dict(action_counts),
        "missing_actions": missing_actions,
        "low_support_actions": low_support_actions,
        "nonterminal_reward": _summary(nonterminal_rewards),
        "invalid_action_masks": invalid_masks,
        "illegal_selected_actions": illegal_selected_actions,
        "invalid_behavior_propensities": invalid_propensities,
        "inconsistent_selected_propensities": inconsistent_selected_propensities,
        "invalid_stop_transitions": invalid_stop_transitions,
        "warnings": warnings,
        "passes_quality_gates": not warnings,
    }


def audit_dataset(transitions: list[dict], min_episodes=MIN_QUALITY_GATE_EPISODES):
    """Backward-compatible composite report; new code should select a stage."""
    raw = audit_raw_evidence(
        transitions,
        min_episodes=min_episodes,
        allow_legacy=True,
    )
    belief = audit_belief_predictions(transitions)
    offline = audit_offline_rl_support(transitions)
    episodes = group_transitions_into_episodes(transitions)
    terminal = _terminal_records(transitions)
    rewards_by_class = defaultdict(list)
    for item in transitions:
        if item.get("true_label") in (0, 1, 2) and "reward" in item:
            rewards_by_class[int(item["true_label"])].append(item["reward"])
    warnings = list(dict.fromkeys(raw["warnings"] + belief["warnings"]))
    return {
        "num_transitions": len(transitions),
        "num_episodes": len(episodes),
        "terminal_micro_f1": belief["terminal_micro_f1"],
        "terminal_macro_f1": belief["terminal_macro_f1"],
        "terminal_true_label_counts": belief["terminal_true_label_counts"],
        "terminal_prediction_counts": belief["terminal_prediction_counts"],
        "episode_length": _summary([len(episode) for episode in episodes]),
        "terminal_skill_coverage": _summary([item.get("skills_covered", 0) for item in terminal]),
        "semantic_scores_by_class": raw["semantic_scores_by_class"],
        "rewards_by_class": {
            str(label): _summary(values) for label, values in sorted(rewards_by_class.items())
        },
        "action_counts": offline["action_counts"],
        "behavior_policy_counts": dict(Counter(
            item.get("behavior_policy") for item in transitions if item.get("behavior_policy")
        )),
        "invalid_evaluations": raw["invalid_evaluations"],
        "split_leaking_resumes": raw["split_leaking_resumes"],
        "split_leaking_jds": raw["split_leaking_jds"],
        "candidate_models": raw["candidate_models"],
        "evaluator_models": raw["evaluator_models"],
        "stage_reports": {"raw": raw, "belief": belief, "offline_rl": offline},
        "warnings": warnings,
        "passes_quality_gates": not warnings,
    }


def audit_file(dataset_file, output_file=None, min_episodes=MIN_QUALITY_GATE_EPISODES, stage="composite"):
    transitions = json.loads(Path(dataset_file).read_text(encoding="utf-8"))
    functions = {
        "raw": lambda: audit_raw_evidence(transitions, min_episodes=min_episodes),
        "belief": lambda: audit_belief_predictions(transitions),
        "calibration_validation": lambda: audit_calibration_validation(transitions),
        "locked_test": lambda: audit_locked_test(transitions),
        "offline_rl": lambda: audit_offline_rl_support(transitions),
        "generation_distribution": lambda: audit_generation_distribution(transitions),
        "learned_policy": lambda: audit_learned_policy_evaluation(transitions),
        "composite": lambda: audit_dataset(transitions, min_episodes=min_episodes),
    }
    report = functions[stage]()
    if output_file:
        Path(output_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("--output")
    parser.add_argument("--min-episodes", type=int, default=MIN_QUALITY_GATE_EPISODES)
    parser.add_argument(
        "--stage",
        choices=(
            "raw", "belief", "calibration_validation", "locked_test",
            "offline_rl", "generation_distribution", "learned_policy", "composite",
        ),
        default="composite",
    )
    args = parser.parse_args()
    print(json.dumps(audit_file(
        args.dataset_file, args.output, args.min_episodes, args.stage
    ), indent=2))
