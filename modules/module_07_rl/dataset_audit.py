"""Stage-specific quality gates for ARIA evidence, beliefs, and offline RL."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from modules.module_07_rl.dataset_split import (
    connected_identity_components,
    group_transitions_into_episodes,
)


MIN_QUALITY_GATE_EPISODES = 200


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
):
    episodes = group_transitions_into_episodes(transitions)
    terminal = _terminal_records(transitions)
    true_counts = Counter(
        int(item["true_label"]) for item in terminal if item.get("true_label") in (0, 1, 2)
    )
    invalid = sum(item.get("evaluation_valid") is False for item in transitions)
    missing_validity = sum("evaluation_valid" not in item for item in transitions)
    nonfinite_scores = 0
    missing_scores = 0
    for transition in transitions:
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
    warnings = []
    if missing_actions:
        warnings.append("Offline dataset has no support for one or more actions.")
    if low_support_actions:
        warnings.append("Offline dataset has weak support for one or more actions.")
    if len(nonterminal_rewards) > 1 and float(np.std(nonterminal_rewards)) < 0.01:
        warnings.append("Non-terminal reward variance is nearly flat.")
    if duplicate_terminal_shaping:
        warnings.append("Terminal reward metadata is inconsistent with single application.")
    return {
        "gate": "offline_rl_support",
        "action_counts": dict(action_counts),
        "missing_actions": missing_actions,
        "low_support_actions": low_support_actions,
        "nonterminal_reward": _summary(nonterminal_rewards),
        "warnings": warnings,
        "passes_quality_gates": not warnings,
    }


def audit_dataset(transitions: list[dict], min_episodes=MIN_QUALITY_GATE_EPISODES):
    """Backward-compatible composite report; new code should select a stage."""
    raw = audit_raw_evidence(transitions, min_episodes=min_episodes)
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
            "offline_rl", "learned_policy", "composite",
        ),
        default="composite",
    )
    args = parser.parse_args()
    print(json.dumps(audit_file(
        args.dataset_file, args.output, args.min_episodes, args.stage
    ), indent=2))
