"""Train-fit, validation-tuned calibration for ARIA belief emissions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_06_belief.belief_state import BeliefStateUpdater
from modules.module_07_rl.dataset_split import group_transitions_into_episodes


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
    correct = [pred == true for pred, true in zip(predicted_labels, true_labels)]
    micro = float(np.mean(correct))
    per_class_f1 = []
    per_class_recall = {}
    for label in (0, 1, 2):
        tp = sum(pred == label and true == label for pred, true in zip(predicted_labels, true_labels))
        fp = sum(pred == label and true != label for pred, true in zip(predicted_labels, true_labels))
        fn = sum(pred != label and true == label for pred, true in zip(predicted_labels, true_labels))
        denominator = 2 * tp + fp + fn
        per_class_f1.append(0.0 if denominator == 0 else 2 * tp / denominator)
        per_class_recall[label] = 0.0 if tp + fn == 0 else tp / (tp + fn)
    ordinal_errors = [
        2 if pred is None else abs(int(pred) - true)
        for pred, true in zip(predicted_labels, true_labels)
    ]
    one_hot = np.eye(3)[np.asarray(true_labels, dtype=int)]
    brier = float(np.mean(np.sum((np.asarray(beliefs) - one_hot) ** 2, axis=1)))
    classified_predictions = [pred for pred in predicted_labels if pred is not None]
    prediction_counts = Counter(classified_predictions)
    max_share = max(prediction_counts.values(), default=0) / len(true_labels)
    return {
        "num_episodes": len(true_labels),
        "micro_f1": micro,
        "macro_f1": float(np.mean(per_class_f1)),
        "ordinal_mae": float(np.mean(ordinal_errors)),
        "expected_calibration_error": _expected_calibration_error(confidences, correct),
        "brier_score": brier,
        "mean_confidence": float(np.mean(confidences)),
        "abstention_rate": abstained / len(true_labels),
        "skipped_episodes": skipped,
        "true_label_counts": dict(Counter(true_labels)),
        "prediction_counts": dict(prediction_counts),
        "per_class_recall": per_class_recall,
        "max_prediction_share": max_share,
        "passes_collapse_gate": max_share <= 0.60 and len(prediction_counts) == 3,
    }


def tune_validation_config(
    fitted_config: BeliefModelConfig,
    validation_transitions: list[dict],
    repeat_discount_powers: Iterable[float] = (0.25, 0.5, 0.75),
    ess_caps: Iterable[float] = (3.0, 5.0, 8.0),
    temperatures: Iterable[float] = (0.8, 1.0, 1.2),
    confidence_thresholds: Iterable[float] = (0.50, 0.60, 0.70),
):
    candidates = []
    for power in repeat_discount_powers:
        for cap in ess_caps:
            for temperature in temperatures:
                for threshold in confidence_thresholds:
                    config = fitted_config.with_updates(
                        repeat_discount_power=float(power),
                        max_skill_effective_sample_size=float(cap),
                        aggregation_temperature=float(temperature),
                        minimum_assessment_confidence=float(threshold),
                    )
                    metrics = evaluate_config(validation_transitions, config)
                    candidates.append({
                        "config": config,
                        "config_hash": config.config_hash,
                        "metrics": metrics,
                    })
    usable = [
        item for item in candidates
        if item["metrics"].get("micro_f1") is not None
        and item["metrics"].get("passes_collapse_gate")
    ]
    if not usable:
        raise ValueError("No validation configuration passed the prediction-collapse gate")
    best = max(
        usable,
        key=lambda item: (
            item["metrics"]["macro_f1"],
            -item["metrics"]["ordinal_mae"],
            -item["metrics"]["expected_calibration_error"],
            -abs(item["config"].repeat_discount_power - 0.5),
            -abs(item["config"].aggregation_temperature - 1.0),
        ),
    )
    metadata = dict(best["config"].fit_metadata)
    metadata["validation_selection"] = {
        "objective": ["collapse_gate", "macro_f1", "ordinal_mae", "ece"],
        "num_candidates": len(candidates),
        "metrics": best["metrics"],
    }
    selected = best["config"].with_updates(fit_metadata=metadata)
    return selected, candidates


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


def calibrate_belief_model(
    training_transitions,
    validation_transitions,
    raw_dataset_hash="",
    split_manifest_hash="",
    bootstrap_samples=100,
):
    fitted = fit_emission_config(
        training_transitions,
        raw_dataset_hash=raw_dataset_hash,
        split_manifest_hash=split_manifest_hash,
    )
    selected, candidates = tune_validation_config(fitted, validation_transitions)
    stability = bootstrap_emission_stability(
        training_transitions, samples=bootstrap_samples
    )
    metadata = dict(selected.fit_metadata)
    metadata["bootstrap_stability"] = stability
    selected = selected.with_updates(fit_metadata=metadata)
    return {
        "config": selected,
        "config_hash": selected.config_hash,
        "training_fit": fitted.to_dict(),
        "validation_metrics": evaluate_config(validation_transitions, selected),
        "bootstrap_stability": stability,
        "num_candidates": len(candidates),
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
