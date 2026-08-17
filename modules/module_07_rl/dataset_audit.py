"""Quality gates for generated ARIA offline-RL datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from modules.module_07_rl.dataset_split import group_transitions_into_episodes


def _summary(values):
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def audit_dataset(transitions: list[dict]) -> dict:
    episodes = group_transitions_into_episodes(transitions)
    terminal = [episode[-1] for episode in episodes if episode]
    true_labels = [item["true_label"] for item in terminal if "true_label" in item]
    predicted_labels = [item["aria_label"] for item in terminal if "aria_label" in item]
    paired_count = min(len(true_labels), len(predicted_labels))
    accuracy = None
    if paired_count:
        accuracy = float(np.mean(
            np.asarray(true_labels[:paired_count])
            == np.asarray(predicted_labels[:paired_count])
        ))

    rewards_by_class = defaultdict(list)
    scores_by_class = defaultdict(list)
    for transition in transitions:
        label = transition.get("true_label")
        if label is None:
            continue
        if "reward" in transition:
            rewards_by_class[int(label)].append(transition["reward"])
        if "semantic_score" in transition:
            scores_by_class[int(label)].append(transition["semantic_score"])

    episode_lengths = [len(episode) for episode in episodes]
    terminal_coverages = [
        item.get("skills_covered", 0) for item in terminal
    ]
    invalid_evaluations = sum(
        transition.get("evaluation_valid") is False
        for transition in transitions
    )

    resume_split_owners = defaultdict(set)
    jd_split_owners = defaultdict(set)
    for transition in transitions:
        split_name = transition.get("dataset_split")
        if split_name:
            resume_split_owners[transition.get("resume_file")].add(split_name)
            jd_split_owners[transition.get("jd_file")].add(split_name)
    leaking_resumes = [
        resume for resume, owners in resume_split_owners.items() if len(owners) > 1
    ]
    leaking_jds = [
        jd for jd, owners in jd_split_owners.items() if len(owners) > 1
    ]

    candidate_models = {
        transition.get("candidate_model") for transition in transitions
        if transition.get("candidate_model")
    }
    evaluator_models = {
        transition.get("evaluator_model") for transition in transitions
        if transition.get("evaluator_model")
    }

    warnings = []
    true_counts = Counter(true_labels)
    predicted_counts = Counter(predicted_labels)
    if true_counts and max(true_counts.values()) - min(true_counts.values()) > max(1, 0.05 * len(terminal)):
        warnings.append("Terminal true-label distribution differs by more than 5%.")
    if predicted_counts and max(predicted_counts.values()) / max(len(terminal), 1) > 0.60:
        warnings.append("More than 60% of terminal predictions collapse to one class.")
    if invalid_evaluations:
        warnings.append("Dataset contains invalid evaluator outputs.")
    if leaking_resumes or leaking_jds:
        warnings.append("Resume or JD identities occur in more than one dataset split.")
    if candidate_models & evaluator_models:
        warnings.append("Candidate and evaluator model sets overlap.")
    for label, rewards in rewards_by_class.items():
        if len(rewards) > 1 and float(np.std(rewards)) < 0.01:
            warnings.append(f"Class {label} reward variance is nearly flat.")

    return {
        "num_transitions": len(transitions),
        "num_episodes": len(episodes),
        "terminal_micro_f1": accuracy,
        "terminal_true_label_counts": dict(true_counts),
        "terminal_prediction_counts": dict(predicted_counts),
        "episode_length": _summary(episode_lengths),
        "terminal_skill_coverage": _summary(terminal_coverages),
        "semantic_scores_by_class": {
            str(label): _summary(values) for label, values in sorted(scores_by_class.items())
        },
        "rewards_by_class": {
            str(label): _summary(values) for label, values in sorted(rewards_by_class.items())
        },
        "action_counts": dict(Counter(
            transition.get("action_idx") for transition in transitions
            if "action_idx" in transition
        )),
        "behavior_policy_counts": dict(Counter(
            transition.get("behavior_policy") for transition in transitions
            if transition.get("behavior_policy")
        )),
        "invalid_evaluations": invalid_evaluations,
        "split_leaking_resumes": leaking_resumes,
        "split_leaking_jds": leaking_jds,
        "candidate_models": sorted(candidate_models),
        "evaluator_models": sorted(evaluator_models),
        "warnings": warnings,
        "passes_quality_gates": not warnings,
    }


def audit_file(dataset_file: str | Path, output_file: str | Path | None = None):
    dataset_path = Path(dataset_file)
    with dataset_path.open("r", encoding="utf-8") as handle:
        transitions = json.load(handle)
    report = audit_dataset(transitions)

    if output_file is not None:
        output_path = Path(output_file)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(audit_file(args.dataset_file, args.output), indent=2))
