"""Training-only grouped calibration and one-shot validation for ARIA."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from itertools import product
from typing import Iterable

import numpy as np

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_06_belief.belief_state import BeliefStateUpdater
from modules.module_07_rl.dataset_split import (
    connected_identity_components,
    group_transitions_into_episodes,
)
from modules.module_07_rl.dataset_audit import CALIBRATION_GATE_THRESHOLDS


CALIBRATION_CV_REPORT_VERSION = "aria-calibration-cv-report-v1"
CALIBRATION_REPORT_VERSION = "aria-calibration-report-v4"
CV_FOLDS = 3
CV_SEED = 42
MAX_CALIBRATION_CANDIDATES = 45
PARAMETER_ORDER = (
    "repeat_discount_power", "max_skill_effective_sample_size",
    "aggregation_temperature", "minimum_assessment_confidence",
    "scale_shrinkage",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weighted_quantile(values, weights, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    cutoff = quantile * cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, cutoff), len(values) - 1)])


def _episode_balanced_scores(transitions: list[dict]):
    by_class = defaultdict(list)
    episode_counts = Counter()
    for episode in group_transitions_into_episodes(transitions):
        if not episode or "true_label" not in episode[-1]:
            continue
        label = int(episode[-1]["true_label"])
        valid_scores = []
        for transition in episode:
            if transition.get("evaluation_valid") is False:
                continue
            try:
                score = float(transition["semantic_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(score) and 0.0 <= score <= 1.0:
                valid_scores.append(score)
        if not valid_scores:
            continue
        episode_counts[label] += 1
        turn_weight = 1.0 / len(valid_scores)
        by_class[label].extend((score, turn_weight) for score in valid_scores)
    return by_class, episode_counts


def fit_emission_config(
    training_transitions: list[dict],
    base_config: BeliefModelConfig | None = None,
    raw_dataset_hash: str = "",
    split_manifest_hash: str = "",
) -> BeliefModelConfig:
    """Fit robust class emissions using training labels only."""
    base = base_config or BeliefModelConfig()
    by_class, episode_counts = _episode_balanced_scores(training_transitions)
    if any(not by_class[label] for label in (0, 1, 2)):
        raise ValueError("Training split must contain replayable evidence for all classes")

    centers = []
    raw_scales = []
    all_residuals = []
    all_weights = []
    for label in (0, 1, 2):
        values = [item[0] for item in by_class[label]]
        weights = [item[1] for item in by_class[label]]
        center = _weighted_quantile(values, weights, 0.5)
        deviations = [abs(value - center) for value in values]
        scale = max(1.4826 * _weighted_quantile(deviations, weights, 0.5), 0.03)
        centers.append(center)
        raw_scales.append(scale)
        all_residuals.extend(deviations)
        all_weights.extend(weights)

    # Isotonic projection without manufacturing a large class gap.
    centers = np.maximum.accumulate(np.asarray(centers, dtype=float))
    pooled_scale = max(
        1.4826 * _weighted_quantile(all_residuals, all_weights, 0.5),
        0.03,
    )
    shrinkage_strength = 5.0
    scales = []
    for label, scale in enumerate(raw_scales):
        n_effective = float(episode_counts[label])
        variance = (
            n_effective * scale**2 + shrinkage_strength * pooled_scale**2
        ) / (n_effective + shrinkage_strength)
        scales.append(float(np.clip(math.sqrt(variance), 0.03, 0.35)))

    total_episodes = sum(episode_counts.values())
    prior = tuple(
        (episode_counts[label] + 1.0) / (total_episodes + 3.0)
        for label in (0, 1, 2)
    )
    metadata = dict(base.fit_metadata)
    metadata.update({
        "fit_source": "training_split_only",
        "estimator": "episode_balanced_weighted_median_mad",
        "episode_counts": {str(k): int(v) for k, v in episode_counts.items()},
        "pooled_scale": pooled_scale,
    })
    return base.with_updates(
        class_centers=tuple(float(value) for value in centers),
        class_scales=tuple(scales),
        class_prior=prior,
        raw_dataset_hash=raw_dataset_hash,
        split_manifest_hash=split_manifest_hash,
        fit_metadata=metadata,
    )


def replay_episode(episode: list[dict], config: BeliefModelConfig | float):
    if isinstance(config, (int, float)):
        config = BeliefModelConfig.legacy(float(config))
    skills = sorted({
        transition["target_skill"]
        for transition in episode
        if transition.get("target_skill")
    })
    if not skills:
        return None
    updater = BeliefStateUpdater(skills, config=config)
    for transition in episode:
        skill = transition.get("target_skill")
        if not skill or transition.get("evaluation_valid") is False:
            continue
        required = ("semantic_score", "cognitive_load")
        if not all(field in transition for field in required):
            continue
        try:
            updater.update_belief(
                skill,
                transition["semantic_score"],
                transition.get("cognitive_load", "low"),
                behavior_score=transition.get("behavior_score"),
                evidence_confidence=transition.get("evaluator_confidence", 1.0),
                stt_confidence=transition.get("stt_confidence", 1.0),
                modality_confidence=transition.get("modality_confidence", 1.0),
                question_fingerprint=transition.get("question_fingerprint"),
            )
        except (TypeError, ValueError):
            continue
    if not updater.get_visited_skills():
        return None
    return updater.get_aggregate_assessment()


def _expected_calibration_error(confidences, correct, bins=10):
    if not confidences:
        return None
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidences >= lower) & (
            confidences <= upper if index == bins - 1 else confidences < upper
        )
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(confidences[mask])) - float(np.mean(correct[mask]))
            )
    return float(ece)


def evaluate_config(transitions: list[dict], config: BeliefModelConfig):
    """Evaluate a config through the shared abstention-aware metric contract."""
    from modules.module_07_rl.metrics import compute_classification_metrics
    true_labels = []
    predicted_labels = []
    confidences = []
    beliefs = []
    skipped = 0
    abstained = 0
    for episode in group_transitions_into_episodes(transitions):
        if not episode or "true_label" not in episode[-1]:
            skipped += 1
            continue
        assessment = replay_episode(episode, config)
        if assessment is None:
            skipped += 1
            continue
        true_labels.append(int(episode[-1]["true_label"]))
        predicted_labels.append(assessment["label"])
        confidences.append(assessment["confidence"])
        beliefs.append(assessment["belief"])
        abstained += assessment["label"] is None

    if not true_labels:
        return {
            "num_episodes": 0,
            "micro_f1": None,
            "macro_f1": None,
            "skipped_episodes": skipped,
        }
    metrics = compute_classification_metrics(true_labels, predicted_labels, beliefs)
    metrics.update({
        "num_episodes": len(true_labels),
        "mean_confidence": float(np.mean(confidences)),
        "skipped_episodes": skipped,
        "prediction_counts": metrics["decision_prediction_counts"],
        "per_class_recall": {label: item["recall"] for label, item in metrics["per_class"].items()},
        "max_prediction_share": metrics["maximum_classified_prediction_share"],
        "passes_collapse_gate": (
            metrics["maximum_classified_prediction_share"] <= 0.60
            and len(metrics["decision_prediction_counts"]) == 3
        ),
    })
    return metrics


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _episode_identity(episode: list[dict]) -> tuple[str, str, str]:
    first = episode[0]
    return (
        str(first.get("episode_id")),
        str(first.get("resume_content_hash") or first.get("resume_file", "unknown_resume")),
        str(first.get("jd_content_hash") or first.get("jd_file", "unknown_jd")),
    )


def build_grouped_cv_folds(
    training_transitions: list[dict],
    folds: int = CV_FOLDS,
    seed: int = CV_SEED,
):
    """Assign intact resume/JD components to exactly three deterministic folds."""
    if folds != CV_FOLDS:
        raise ValueError("calibration protocol v4 requires exactly three folds")
    if seed != CV_SEED:
        raise ValueError("calibration protocol v4 requires seed 42")
    if not training_transitions:
        raise ValueError("training transitions are empty")
    if any(item.get("dataset_split") in {"validation", "test"} for item in training_transitions):
        raise ValueError("grouped CV accepts training transitions only")

    component_records = []
    for component in connected_identity_components(training_transitions):
        episodes = sorted(component, key=lambda episode: _episode_identity(episode)[0])
        episode_ids = [_episode_identity(episode)[0] for episode in episodes]
        resumes = sorted({_episode_identity(episode)[1] for episode in episodes})
        jds = sorted({_episode_identity(episode)[2] for episode in episodes})
        class_counts = Counter(
            int(episode[-1]["true_label"])
            for episode in episodes
            if episode and episode[-1].get("true_label") in (0, 1, 2)
        )
        identity = {
            "episode_ids": episode_ids,
            "resume_identities": resumes,
            "jd_identities": jds,
            "seed": seed,
        }
        component_records.append({
            **identity,
            "component_hash": _canonical_hash(identity),
            "episode_count": len(episodes),
            "class_episode_counts": {str(label): int(class_counts[label]) for label in (0, 1, 2)},
        })
    if len(component_records) < folds:
        raise ValueError("grouped CV requires at least three identity components")

    # Largest and most class-skewed components are placed first. Candidate fold
    # scores minimize global episode imbalance, then global per-class imbalance.
    component_records.sort(key=lambda item: (
        -item["episode_count"],
        -max(item["class_episode_counts"].values()),
        item["component_hash"],
    ))
    fold_episode_counts = [0] * folds
    fold_class_counts = [{label: 0 for label in (0, 1, 2)} for _ in range(folds)]
    for component in component_records:
        choices = []
        for fold_index in range(folds):
            episode_counts = list(fold_episode_counts)
            episode_counts[fold_index] += component["episode_count"]
            class_counts = [dict(item) for item in fold_class_counts]
            for label in (0, 1, 2):
                class_counts[fold_index][label] += component["class_episode_counts"][str(label)]
            episode_imbalance = max(episode_counts) - min(episode_counts)
            class_imbalance = sum(
                max(item[label] for item in class_counts) - min(item[label] for item in class_counts)
                for label in (0, 1, 2)
            )
            tie = _canonical_hash({
                "component_hash": component["component_hash"],
                "fold": fold_index,
                "seed": seed,
            })
            choices.append((episode_imbalance, class_imbalance, tie, fold_index))
        chosen = min(choices)[-1]
        component["fold"] = chosen
        fold_episode_counts[chosen] += component["episode_count"]
        for label in (0, 1, 2):
            fold_class_counts[chosen][label] += component["class_episode_counts"][str(label)]

    assignments = {
        episode_id: component["fold"]
        for component in component_records
        for episode_id in component["episode_ids"]
    }
    fold_reports = []
    for fold_index in range(folds):
        identities = sorted(
            item["component_hash"] for item in component_records if item["fold"] == fold_index
        )
        fold_payload = {
            "fold": fold_index,
            "component_hashes": identities,
            "episode_ids": sorted(
                episode_id for episode_id, assigned in assignments.items() if assigned == fold_index
            ),
            "episode_count": fold_episode_counts[fold_index],
            "class_episode_counts": {
                str(label): fold_class_counts[fold_index][label] for label in (0, 1, 2)
            },
        }
        fold_payload["fold_hash"] = _canonical_hash(fold_payload)
        fold_reports.append(fold_payload)
    report = {
        "schema_version": "aria-grouped-cv-folds-v1",
        "algorithm": "grouped-identity-component-3fold-greedy-v1",
        "fold_count": folds,
        "seed": seed,
        "assignments": assignments,
        "components": sorted(component_records, key=lambda item: item["component_hash"]),
        "folds": fold_reports,
    }
    report["report_hash"] = _canonical_hash(report)
    return report


def apply_scale_shrinkage(config: BeliefModelConfig, alpha: float) -> BeliefModelConfig:
    """Shrink fitted class scales toward the fitted pooled scale with clipping."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("scale_shrinkage must be in [0, 1]")
    pooled = float(config.fit_metadata.get("pooled_scale"))
    scales = tuple(float(np.clip(
        (1.0 - alpha) * float(scale) + alpha * pooled, 0.03, 0.35,
    )) for scale in config.class_scales)
    metadata = dict(config.fit_metadata)
    metadata["scale_shrinkage"] = alpha
    return config.with_updates(class_scales=scales, fit_metadata=metadata)


def _candidate_parameters(parameters: dict) -> dict:
    defaults = {
        "scale_shrinkage": 0.0,
        "aggregation_temperature": 1.0,
        "minimum_assessment_confidence": 0.50,
        "repeat_discount_power": 0.50,
        "max_skill_effective_sample_size": 5.0,
    }
    unknown = set(parameters) - set(defaults)
    if unknown:
        raise ValueError(f"unknown calibration parameters: {sorted(unknown)}")
    defaults.update({key: float(value) for key, value in parameters.items()})
    return defaults


def _parameter_tuple(parameters: dict) -> tuple:
    return tuple(float(parameters[name]) for name in PARAMETER_ORDER)


def _prediction_rows(transitions: list[dict], config: BeliefModelConfig) -> list[dict]:
    rows = []
    for episode in sorted(
        group_transitions_into_episodes(transitions),
        key=lambda item: _episode_identity(item)[0] if item else "",
    ):
        if not episode or episode[-1].get("true_label") not in (0, 1, 2):
            continue
        assessment = replay_episode(episode, config)
        if assessment is None:
            continue
        episode_id, resume, jd = _episode_identity(episode)
        rows.append({
            "episode_id": episode_id,
            "resume_identity": resume,
            "jd_identity": jd,
            "true_label": int(episode[-1]["true_label"]),
            "prediction": assessment["label"],
            "probabilities": [float(value) for value in assessment["belief"]],
        })
    return rows


def _gate_candidate(metrics: dict, *, minimum_components: int | None = None) -> tuple[bool, list[str]]:
    thresholds = CALIBRATION_GATE_THRESHOLDS
    checks = {
        "overall_accuracy_below_minimum": metrics.get("overall_accuracy") is not None and metrics["overall_accuracy"] >= thresholds["minimum_overall_accuracy"],
        "macro_f1_below_minimum": metrics.get("macro_f1") is not None and metrics["macro_f1"] >= thresholds["minimum_macro_f1"],
        "minimum_class_recall_below_minimum": metrics.get("minimum_class_recall") is not None and metrics["minimum_class_recall"] >= thresholds["minimum_class_recall"],
        "classified_prediction_share_above_maximum": metrics.get("maximum_classified_prediction_share") is not None and metrics["maximum_classified_prediction_share"] <= thresholds["maximum_classified_prediction_share"],
        "abstention_rate_above_maximum": metrics.get("abstention_rate") is not None and metrics["abstention_rate"] <= thresholds["maximum_abstention_rate"],
        "ece_above_maximum": metrics.get("expected_calibration_error") is not None and metrics["expected_calibration_error"] <= thresholds["maximum_expected_calibration_error"],
        "missing_true_class": set(map(int, metrics.get("true_label_counts", {}))) == {0, 1, 2},
        "missing_decision_class": set(map(int, metrics.get("decision_prediction_counts", {}))) == {0, 1, 2},
    }
    def all_finite(value):
        if value is None:
            return False
        if isinstance(value, dict):
            return all(all_finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(all_finite(item) for item in value)
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            return math.isfinite(float(value))
        return True

    checks["non_finite_metrics"] = bool(metrics) and all_finite(metrics)
    if minimum_components is not None:
        checks["insufficient_identity_components"] = (
            int(metrics.get("identity_component_count", 0)) >= minimum_components
        )
    reasons = [name for name, passed in checks.items() if not passed]
    return not reasons, reasons


def evaluate_candidate_cross_validated(
    training_transitions: list[dict],
    candidate_parameters: dict,
    folds,
):
    """Evaluate one parameter tuple exclusively with training-fold OOF predictions."""
    if any(item.get("dataset_split") in {"validation", "test"} for item in training_transitions):
        raise ValueError("candidate evaluation accepts training transitions only")
    fold_report = build_grouped_cv_folds(training_transitions) if folds == 3 else folds
    if not isinstance(fold_report, dict) or fold_report.get("fold_count") != 3:
        raise ValueError("folds must be a three-fold grouped CV report")
    parameters = _candidate_parameters(candidate_parameters)
    episodes = group_transitions_into_episodes(training_transitions)
    rows, fit_hashes = [], []
    error = None
    try:
        for held_out in range(3):
            fit_rows, held_out_rows = [], []
            for episode in episodes:
                episode_id = _episode_identity(episode)[0]
                destination = fold_report["assignments"].get(episode_id)
                if destination is None:
                    raise ValueError(f"episode {episode_id} is absent from CV folds")
                (held_out_rows if destination == held_out else fit_rows).extend(episode)
            base = BeliefModelConfig(
                repeat_discount_power=parameters["repeat_discount_power"],
                max_skill_effective_sample_size=parameters["max_skill_effective_sample_size"],
                aggregation_temperature=parameters["aggregation_temperature"],
                minimum_assessment_confidence=parameters["minimum_assessment_confidence"],
            )
            fitted = fit_emission_config(fit_rows, base_config=base)
            fitted = apply_scale_shrinkage(fitted, parameters["scale_shrinkage"])
            fit_hashes.append(fitted.config_hash)
            fold_rows = _prediction_rows(held_out_rows, fitted)
            for row in fold_rows:
                row["held_out_fold"] = held_out
            rows.extend(fold_rows)
        from modules.module_07_rl.metrics import compute_classification_metrics
        metrics = compute_classification_metrics(
            [row["true_label"] for row in rows],
            [row["prediction"] for row in rows],
            [row["probabilities"] for row in rows],
        )
    except (TypeError, ValueError) as exc:
        error = str(exc)
        metrics = {}
    eligible, reasons = _gate_candidate(metrics)
    if error:
        reasons.insert(0, f"evaluation_error:{error}")
        eligible = False
    return {
        "parameters": parameters,
        "parameter_tuple": list(_parameter_tuple(parameters)),
        "eligible": eligible,
        "rejection_reasons": reasons,
        "metrics": metrics,
        "out_of_fold_predictions": rows,
        "fold_config_hashes": fit_hashes,
        "fold_report_hash": fold_report["report_hash"],
    }


def _rank_key(candidate: dict) -> tuple:
    metrics = candidate["metrics"]
    def finite_or(name, fallback):
        value = metrics.get(name)
        return float(value) if value is not None and math.isfinite(float(value)) else fallback
    return (
        -finite_or("macro_f1", -math.inf),
        finite_or("ordinal_mae", math.inf),
        finite_or("expected_calibration_error", math.inf),
        finite_or("abstention_rate", math.inf),
        tuple(candidate["parameter_tuple"]),
    )


def _constraint_deficit(candidate: dict) -> float:
    metrics = candidate.get("metrics", {})
    thresholds = CALIBRATION_GATE_THRESHOLDS
    specifications = (
        ("overall_accuracy", thresholds["minimum_overall_accuracy"], "lower"),
        ("macro_f1", thresholds["minimum_macro_f1"], "lower"),
        ("minimum_class_recall", thresholds["minimum_class_recall"], "lower"),
        ("maximum_classified_prediction_share", thresholds["maximum_classified_prediction_share"], "upper"),
        ("abstention_rate", thresholds["maximum_abstention_rate"], "upper"),
        ("expected_calibration_error", thresholds["maximum_expected_calibration_error"], "upper"),
    )
    total = 0.0
    for name, threshold, direction in specifications:
        value = metrics.get(name)
        if value is None or not math.isfinite(float(value)):
            total += 1e6
        elif direction == "lower":
            total += max(0.0, threshold - float(value)) / threshold
        else:
            total += max(0.0, float(value) - threshold) / threshold
    return total


def _best_failing(candidates: list[dict]) -> dict:
    return min(candidates, key=lambda item: (
        _constraint_deficit(item),
        _rank_key(item) if item.get("metrics") and item["metrics"].get("macro_f1") is not None
        else (math.inf, math.inf, math.inf, math.inf, tuple(item["parameter_tuple"])),
    ))


def select_training_only_calibration(
    training_transitions: list[dict],
    raw_dataset_hash: str,
    split_manifest_hash: str,
    protocol_hash: str,
):
    """Run the frozen sequential search without accepting evaluation data."""
    fold_report = build_grouped_cv_folds(training_transitions, folds=3, seed=42)
    attempted: list[dict] = []
    stages = []

    def run_stage(name: str, parameters: list[dict]):
        results = []
        for values in parameters:
            if len(attempted) >= MAX_CALIBRATION_CANDIDATES:
                raise AssertionError("calibration candidate limit exceeded")
            result = evaluate_candidate_cross_validated(training_transitions, values, fold_report)
            result["candidate_index"] = len(attempted) + 1
            result["stage"] = name
            attempted.append(result)
            results.append(result)
        eligible = [item for item in results if item["eligible"]]
        stage = {
            "stage": name, "candidate_count": len(results),
            "eligible_count": len(eligible),
        }
        stages.append(stage)
        return min(eligible, key=_rank_key) if eligible else None, _best_failing(results)

    baseline = {
        "scale_shrinkage": 0.0, "repeat_discount_power": 0.50,
        "max_skill_effective_sample_size": 5,
        "aggregation_temperature": 1.00,
        "minimum_assessment_confidence": 0.50,
    }
    selected, anchor = run_stage("A", [baseline])
    if selected is None:
        stage_b = [{**baseline, "scale_shrinkage": alpha} for alpha in (0.0, 0.25, 0.50, 0.75, 1.0)]
        selected, anchor = run_stage("B", stage_b)
    if selected is None:
        fixed = dict(anchor["parameters"])
        stage_c = [
            {**fixed, "aggregation_temperature": temperature,
             "minimum_assessment_confidence": confidence}
            for temperature, confidence in product(
                (0.80, 1.00, 1.20, 1.50, 2.00),
                (0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
            )
        ]
        selected, anchor = run_stage("C", stage_c)
    if selected is None:
        fixed = dict(anchor["parameters"])
        stage_d = [
            {**fixed, "repeat_discount_power": power,
             "max_skill_effective_sample_size": cap}
            for power, cap in product((0.25, 0.50, 0.75), (3, 5, 8))
        ]
        selected, anchor = run_stage("D", stage_d)

    assert len(attempted) <= MAX_CALIBRATION_CANDIDATES
    report = {
        "schema_version": CALIBRATION_CV_REPORT_VERSION,
        "producer_version": "aria-belief-calibration-v4",
        "supported_consumer_versions": [CALIBRATION_CV_REPORT_VERSION],
        "protocol_hash": protocol_hash,
        "raw_dataset_hash": raw_dataset_hash,
        "split_manifest_hash": split_manifest_hash,
        "cross_validation": fold_report,
        "stages": stages,
        "attempted_candidates": attempted,
        "candidate_count": len(attempted),
        "candidate_limit": MAX_CALIBRATION_CANDIDATES,
        "selection_status": "ELIGIBLE" if selected else "FAILED",
        "selected_candidate": selected,
        "best_failing_candidate": None if selected else anchor,
        "validation_used_for_selection": False,
    }
    report["selection_decision_hash"] = _canonical_hash(report)
    if selected:
        base = BeliefModelConfig(
            repeat_discount_power=selected["parameters"]["repeat_discount_power"],
            max_skill_effective_sample_size=selected["parameters"]["max_skill_effective_sample_size"],
            aggregation_temperature=selected["parameters"]["aggregation_temperature"],
            minimum_assessment_confidence=selected["parameters"]["minimum_assessment_confidence"],
        )
        final_config = fit_emission_config(
            training_transitions, base_config=base,
            raw_dataset_hash=raw_dataset_hash,
            split_manifest_hash=split_manifest_hash,
        )
        final_config = apply_scale_shrinkage(
            final_config, selected["parameters"]["scale_shrinkage"],
        )
        metadata = dict(final_config.fit_metadata)
        metadata.update({
            "calibration_algorithm_version": "aria-belief-calibration-v4",
            "protocol_hash": protocol_hash,
            "selection_decision_hash": report["selection_decision_hash"],
            "selected_parameters": selected["parameters"],
        })
        final_config = final_config.with_updates(fit_metadata=metadata)
        report["config"] = final_config
        report["belief_config_hash"] = final_config.config_hash
    hashable_report = dict(report)
    if isinstance(hashable_report.get("config"), BeliefModelConfig):
        hashable_report["config"] = hashable_report["config"].to_dict()
    report["report_hash"] = _canonical_hash(hashable_report)
    return report


def select_training_only_calibration_v5(
    training_transitions: list[dict],
    raw_dataset_hash: str,
    split_manifest_hash: str,
    protocol_hash: str,
):
    """Run the narrow v5 search without repeating or changing protocol v4."""
    from modules.module_07_rl.calibration_protocol_v5 import (
        CALIBRATION_ALGORITHM_VERSION as V5_ALGORITHM_VERSION,
        CALIBRATION_CV_REPORT_VERSION as V5_REPORT_VERSION,
        MAX_CALIBRATION_CANDIDATES as V5_CANDIDATE_LIMIT,
    )

    fold_report = build_grouped_cv_folds(training_transitions, folds=3, seed=42)
    attempted: list[dict] = []
    stages = []

    def run_stage(name: str, parameter_sets: list[dict]):
        results = []
        for parameters in parameter_sets:
            if len(attempted) >= V5_CANDIDATE_LIMIT:
                raise AssertionError("calibration v5 candidate limit exceeded")
            result = evaluate_candidate_cross_validated(
                training_transitions, parameters, fold_report,
            )
            result["candidate_index"] = len(attempted) + 1
            result["stage"] = name
            attempted.append(result)
            results.append(result)
        eligible = [item for item in results if item["eligible"]]
        stages.append({
            "stage": name,
            "candidate_count": len(results),
            "eligible_count": len(eligible),
        })
        return min(eligible, key=_rank_key) if eligible else None

    anchor = {
        "scale_shrinkage": 1.00,
        "aggregation_temperature": 2.00,
        "minimum_assessment_confidence": 0.45,
        "repeat_discount_power": 0.25,
        "max_skill_effective_sample_size": 3,
    }
    selected = run_stage("A_V4_ANCHOR", [anchor])
    if selected is None:
        selected = run_stage("B_LOWER_REPEAT_DISCOUNT", [
            {**anchor, "repeat_discount_power": power}
            for power in (0.00, 0.10, 0.20)
        ])
    assert len(attempted) <= V5_CANDIDATE_LIMIT
    best_failing = None if selected else _best_failing(attempted)
    report = {
        "schema_version": V5_REPORT_VERSION,
        "producer_version": V5_ALGORITHM_VERSION,
        "supported_consumer_versions": [V5_REPORT_VERSION],
        "protocol_hash": protocol_hash,
        "raw_dataset_hash": raw_dataset_hash,
        "split_manifest_hash": split_manifest_hash,
        "cross_validation": fold_report,
        "stages": stages,
        "attempted_candidates": attempted,
        "candidate_count": len(attempted),
        "candidate_limit": V5_CANDIDATE_LIMIT,
        "selection_status": "ELIGIBLE" if selected else "FAILED",
        "selected_candidate": selected,
        "best_failing_candidate": best_failing,
        "validation_used_for_selection": False,
        "v5_change_scope": "lower-repeat-discount-only",
    }
    report["selection_decision_hash"] = _canonical_hash(report)
    if selected:
        parameters = selected["parameters"]
        base = BeliefModelConfig(
            repeat_discount_power=parameters["repeat_discount_power"],
            max_skill_effective_sample_size=parameters["max_skill_effective_sample_size"],
            aggregation_temperature=parameters["aggregation_temperature"],
            minimum_assessment_confidence=parameters["minimum_assessment_confidence"],
        )
        config = fit_emission_config(
            training_transitions,
            base_config=base,
            raw_dataset_hash=raw_dataset_hash,
            split_manifest_hash=split_manifest_hash,
        )
        config = apply_scale_shrinkage(config, parameters["scale_shrinkage"])
        metadata = dict(config.fit_metadata)
        metadata.update({
            "calibration_algorithm_version": V5_ALGORITHM_VERSION,
            "protocol_hash_at_selection": protocol_hash,
            "selection_decision_hash": report["selection_decision_hash"],
            "selected_parameters": parameters,
            "v5_change_scope": "lower-repeat-discount-only",
        })
        config = config.with_updates(fit_metadata=metadata)
        report["config"] = config
        report["belief_config_hash"] = config.config_hash
    hashable = dict(report)
    if isinstance(hashable.get("config"), BeliefModelConfig):
        hashable["config"] = hashable["config"].to_dict()
    report["report_hash"] = _canonical_hash(hashable)
    return report


def _metrics_from_vectors(truth, predictions, probabilities):
    from modules.module_07_rl.metrics import compute_classification_metrics
    return compute_classification_metrics(truth, predictions, probabilities)


def paired_component_bootstrap(
    validation_transitions,
    baseline_predictions,
    baseline_probabilities,
    calibrated_predictions,
    calibrated_probabilities,
    samples=1000,
    seed=42,
):
    """Paired percentile bootstrap over intact connected identity components."""
    episodes = sorted(group_transitions_into_episodes(validation_transitions), key=lambda x: _episode_identity(x)[0])
    truths = [int(episode[-1]["true_label"]) for episode in episodes]
    vectors = (baseline_predictions, baseline_probabilities, calibrated_predictions, calibrated_probabilities)
    if any(len(value) != len(episodes) for value in vectors):
        raise ValueError("bootstrap prediction vectors must match validation episodes")
    episode_index = {_episode_identity(episode)[0]: index for index, episode in enumerate(episodes)}
    component_indices = []
    component_hashes = []
    for component in connected_identity_components(validation_transitions):
        ids = sorted(_episode_identity(episode)[0] for episode in component)
        component_indices.append([episode_index[item] for item in ids])
        component_hashes.append(_canonical_hash(ids))
    if not component_indices:
        raise ValueError("bootstrap requires identity components")
    metric_names = {
        "overall_accuracy": "overall_accuracy",
        "macro_f1": "macro_f1",
        "minimum_class_recall": "minimum_class_recall",
        "expected_calibration_error": "expected_calibration_error",
        "abstention_rate": "abstention_rate",
    }

    def compare(indices):
        baseline = _metrics_from_vectors(
            [truths[i] for i in indices], [baseline_predictions[i] for i in indices],
            [baseline_probabilities[i] for i in indices],
        )
        calibrated = _metrics_from_vectors(
            [truths[i] for i in indices], [calibrated_predictions[i] for i in indices],
            [calibrated_probabilities[i] for i in indices],
        )
        return {
            name: float(calibrated[key]) - float(baseline[key])
            for name, key in metric_names.items()
        }

    point = compare(list(range(len(episodes))))
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in metric_names}
    sampled_component_indices = []
    for _ in range(int(samples)):
        chosen = rng.integers(0, len(component_indices), size=len(component_indices)).tolist()
        sampled_component_indices.append(chosen)
        indices = [index for component in chosen for index in component_indices[component]]
        differences = compare(indices)
        for name, value in differences.items():
            draws[name].append(value)
    intervals = {
        name: {
            "point_difference": point[name],
            "percentile_95_interval": [
                float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)),
            ],
        }
        for name, values in draws.items()
    }
    leave_one_out = []
    for omitted in range(len(component_indices)):
        indices = [index for i, component in enumerate(component_indices) if i != omitted for index in component]
        leave_one_out.append({
            "omitted_component_hash": component_hashes[omitted],
            "differences": compare(indices) if indices else None,
        })
    return {
        "schema_version": "aria-paired-component-bootstrap-v1",
        "method": "paired-identity-component-percentile",
        "samples": int(samples),
        "seed": int(seed),
        "component_hashes": component_hashes,
        "sampled_component_indices": sampled_component_indices,
        "intervals": intervals,
        "leave_one_component_out": leave_one_out,
        "diagnostic_only": True,
    }


def tune_validation_config(
    fitted_config: BeliefModelConfig,
    validation_transitions: list[dict],
    repeat_discount_powers: Iterable[float] = (0.25, 0.5, 0.75),
    ess_caps: Iterable[float] = (3.0, 5.0, 8.0),
    temperatures: Iterable[float] = (0.8, 1.0, 1.2),
    confidence_thresholds: Iterable[float] = (0.50, 0.60, 0.70),
):
    raise ValueError(
        "validation-selected tuning is disabled; use select_training_only_calibration"
    )


def bootstrap_emission_stability(training_transitions, samples=100, seed=42):
    episodes = group_transitions_into_episodes(training_transitions)
    if not episodes:
        raise ValueError("No episodes available for bootstrap")
    rng = np.random.default_rng(seed)
    centers = []
    ordered = 0
    for sample_index in range(samples):
        sampled = []
        for draw_number, draw_index in enumerate(
            rng.integers(0, len(episodes), size=len(episodes))
        ):
            sampled_episode_id = f"bootstrap-{sample_index}-{draw_number}"
            for transition in episodes[int(draw_index)]:
                copied = dict(transition)
                copied["episode_id"] = sampled_episode_id
                sampled.append(copied)
        try:
            config = fit_emission_config(sampled)
        except ValueError:
            continue
        centers.append(config.class_centers)
        ordered += all(
            left <= right
            for left, right in zip(config.class_centers, config.class_centers[1:])
        )
    if not centers:
        return {"successful_samples": 0, "ordered_fraction": 0.0}
    array = np.asarray(centers)
    return {
        "successful_samples": len(centers),
        "center_mean": array.mean(axis=0).tolist(),
        "center_std": array.std(axis=0).tolist(),
        "ordered_fraction": ordered / len(centers),
    }


def validate_selected_calibration(
    validation_transitions: list[dict],
    config: BeliefModelConfig,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    minimum_components: int = 6,
):
    """Evaluate a frozen config once and compare paired original predictions."""
    calibrated_rows = _prediction_rows(validation_transitions, config)
    episodes = sorted(
        group_transitions_into_episodes(validation_transitions),
        key=lambda item: _episode_identity(item)[0] if item else "",
    )
    if len(calibrated_rows) != len(episodes):
        raise ValueError("every validation episode must produce a calibrated assessment")
    truths, baseline_predictions, baseline_probabilities = [], [], []
    for episode in episodes:
        terminal = episode[-1]
        truth = int(terminal["true_label"])
        prediction = terminal.get("aria_label")
        probability = terminal.get("aggregate_belief")
        if probability is None:
            # A deterministic compatibility representation for legacy stored
            # predictions. This is used only for baseline diagnostics/ECE.
            probability = [1.0 / 3.0] * 3 if prediction not in (0, 1, 2) else [
                1.0 if index == int(prediction) else 0.0 for index in (0, 1, 2)
            ]
        truths.append(truth)
        baseline_predictions.append(prediction)
        baseline_probabilities.append(probability)
    calibrated_predictions = [row["prediction"] for row in calibrated_rows]
    calibrated_probabilities = [row["probabilities"] for row in calibrated_rows]
    calibrated_metrics = _metrics_from_vectors(
        truths, calibrated_predictions, calibrated_probabilities,
    )
    calibrated_metrics["identity_component_count"] = len(
        connected_identity_components(validation_transitions)
    )
    baseline_metrics = _metrics_from_vectors(
        truths, baseline_predictions, baseline_probabilities,
    )
    eligible, rejection_reasons = _gate_candidate(
        calibrated_metrics, minimum_components=minimum_components,
    )
    non_regression = {
        "overall_accuracy": calibrated_metrics["overall_accuracy"] >= baseline_metrics["overall_accuracy"] - 0.02,
        "macro_f1": calibrated_metrics["macro_f1"] >= baseline_metrics["macro_f1"] - 0.02,
    }
    if not all(non_regression.values()):
        rejection_reasons.extend(
            f"non_regression_{name}" for name, passed in non_regression.items() if not passed
        )
    passed = eligible and all(non_regression.values())
    bootstrap = paired_component_bootstrap(
        validation_transitions,
        baseline_predictions, baseline_probabilities,
        calibrated_predictions, calibrated_probabilities,
        samples=bootstrap_samples, seed=seed,
    )
    return {
        "schema_version": "aria-validation-evaluation-v1",
        "validation_execution_number": 1,
        "candidate_frozen_before_validation": True,
        "belief_config_hash": config.config_hash,
        "calibrated_metrics": calibrated_metrics,
        "baseline_metrics": baseline_metrics,
        "non_regression": non_regression,
        "rejection_reasons": rejection_reasons,
        "passes_quality_gates": passed,
        "paired_component_bootstrap": bootstrap,
    }


def calibrate_belief_model(
    training_transitions,
    validation_transitions,
    raw_dataset_hash="",
    split_manifest_hash="",
    bootstrap_samples=1000,
    select_on_validation=False,
    protocol_hash="",
):
    """Compatibility orchestrator: training-only selection, then one validation."""
    if select_on_validation:
        raise ValueError(
            "validation-selected calibration is forbidden by calibration protocol v4"
        )
    selection = select_training_only_calibration(
        training_transitions,
        raw_dataset_hash=raw_dataset_hash,
        split_manifest_hash=split_manifest_hash,
        protocol_hash=protocol_hash,
    )
    if selection["selection_status"] != "ELIGIBLE":
        return {
            "config": None,
            "config_hash": None,
            "selection": selection,
            "validation_evaluated": False,
            "validation_used_for_selection": False,
        }
    config = selection["config"]
    validation = validate_selected_calibration(
        validation_transitions, config,
        bootstrap_samples=bootstrap_samples,
    )
    return {
        "config": config,
        "config_hash": config.config_hash,
        "selection": selection,
        "validation": validation,
        "validation_metrics": validation["calibrated_metrics"],
        "num_candidates": selection["candidate_count"],
        "validation_evaluated": True,
        "validation_used_for_selection": False,
    }


# Explicit legacy wrappers for older scripts and tests.
def evaluate_likelihood_sigma(transitions: list[dict], likelihood_sigma: float):
    result = evaluate_config(transitions, BeliefModelConfig.legacy(likelihood_sigma))
    result["likelihood_sigma"] = float(likelihood_sigma)
    return result


def calibrate_likelihood_sigma(
    validation_transitions: list[dict],
    candidates=(0.12, 0.16, 0.20, 0.22, 0.26, 0.30),
):
    results = [evaluate_likelihood_sigma(validation_transitions, sigma) for sigma in candidates]
    usable = [result for result in results if result["micro_f1"] is not None]
    if not usable:
        raise ValueError("No validation episodes contained replayable evaluator evidence")
    best = max(
        usable,
        key=lambda result: (
            result["micro_f1"], result["macro_f1"],
            -abs(result["likelihood_sigma"] - 0.22),
        ),
    )
    return {"best": best, "candidates": results}


def calibrate_files(train_file, validation_file, output_file, bootstrap_samples=100):
    train_path, validation_path = Path(train_file), Path(validation_file)
    training = json.loads(train_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    result = calibrate_belief_model(
        training,
        validation,
        raw_dataset_hash=file_sha256(train_path),
        split_manifest_hash=hashlib.sha256(
            (file_sha256(train_path) + file_sha256(validation_path)).encode("ascii")
        ).hexdigest(),
        bootstrap_samples=bootstrap_samples,
    )
    result["config"].save(output_file)
    return {
        key: value.to_dict() if isinstance(value, BeliefModelConfig) else value
        for key, value in result.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("validation_file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(calibrate_files(
        args.train_file,
        args.validation_file,
        args.output,
        bootstrap_samples=args.bootstrap_samples,
    ), indent=2))
