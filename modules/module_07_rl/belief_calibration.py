"""Replay evaluator evidence to calibrate Bayesian likelihood separation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from modules.module_06_belief.belief_state import BeliefStateUpdater
from modules.module_07_rl.dataset_split import group_transitions_into_episodes


def replay_episode(episode: list[dict], likelihood_sigma: float):
    skills = sorted({
        transition["target_skill"]
        for transition in episode
        if transition.get("target_skill")
    })
    if not skills:
        return None

    updater = BeliefStateUpdater(skills, likelihood_sigma=likelihood_sigma)
    for transition in episode:
        skill = transition.get("target_skill")
        if not skill or transition.get("evaluation_valid") is False:
            continue
        required = ("semantic_score", "behavior_score", "cognitive_load")
        if not all(field in transition for field in required):
            continue
        updater.update_belief(
            skill,
            transition["semantic_score"],
            transition["cognitive_load"],
            transition["behavior_score"],
            evidence_confidence=transition.get("evaluator_confidence", 1.0),
        )

    if not updater.get_visited_skills():
        return None
    return updater.get_aggregate_assessment()


def evaluate_likelihood_sigma(transitions: list[dict], likelihood_sigma: float):
    true_labels = []
    predicted_labels = []
    confidences = []
    skipped_episodes = 0

    for episode in group_transitions_into_episodes(transitions):
        if not episode or "true_label" not in episode[-1]:
            skipped_episodes += 1
            continue
        assessment = replay_episode(episode, likelihood_sigma)
        if assessment is None:
            skipped_episodes += 1
            continue
        true_labels.append(int(episode[-1]["true_label"]))
        predicted_labels.append(assessment["label"])
        confidences.append(assessment["confidence"])

    if not true_labels:
        return {
            "likelihood_sigma": float(likelihood_sigma),
            "num_episodes": 0,
            "micro_f1": None,
            "macro_f1": None,
            "mean_confidence": None,
            "skipped_episodes": skipped_episodes,
        }

    true_array = np.asarray(true_labels)
    predicted_array = np.asarray(predicted_labels)
    micro_f1 = float(np.mean(true_array == predicted_array))
    per_class_f1 = []
    for label in (0, 1, 2):
        true_positive = int(np.sum((true_array == label) & (predicted_array == label)))
        false_positive = int(np.sum((true_array != label) & (predicted_array == label)))
        false_negative = int(np.sum((true_array == label) & (predicted_array != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)

    return {
        "likelihood_sigma": float(likelihood_sigma),
        "num_episodes": len(true_labels),
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(per_class_f1)),
        "mean_confidence": float(np.mean(confidences)),
        "skipped_episodes": skipped_episodes,
        "true_label_counts": {
            label: int(np.sum(true_array == label)) for label in (0, 1, 2)
        },
        "prediction_counts": {
            label: int(np.sum(predicted_array == label)) for label in (0, 1, 2)
        },
    }


def calibrate_likelihood_sigma(
    validation_transitions: list[dict],
    candidates=(0.12, 0.16, 0.20, 0.22, 0.26, 0.30),
):
    results = [
        evaluate_likelihood_sigma(validation_transitions, sigma)
        for sigma in candidates
    ]
    usable = [result for result in results if result["micro_f1"] is not None]
    if not usable:
        raise ValueError("No validation episodes contained replayable evaluator evidence")

    # Macro-F1 breaks micro-F1 ties so a collapsed majority-class solution
    # cannot win merely because of mild class imbalance.
    best = max(
        usable,
        key=lambda result: (
            result["micro_f1"],
            result["macro_f1"],
            -abs(result["likelihood_sigma"] - 0.22),
        ),
    )
    return {"best": best, "candidates": results}


def calibrate_file(validation_file: str | Path, output_file: str | Path | None = None):
    with Path(validation_file).open("r", encoding="utf-8") as handle:
        validation_transitions = json.load(handle)
    report = calibrate_likelihood_sigma(validation_transitions)
    if output_file:
        with Path(output_file).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_file")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = calibrate_file(args.validation_file, args.output)
    print(json.dumps(result, indent=2))
    print(
        "Set ARIA_BELIEF_SIGMA to",
        result["best"]["likelihood_sigma"],
        "for training and held-out evaluation.",
    )
