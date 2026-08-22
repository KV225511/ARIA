"""Episode-safe dataset splitting for ARIA offline RL data."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


SPLIT_NAMES = ("train", "validation", "test")


def apply_terminal_outcome_rewards(transitions: list[dict], omega: float) -> int:
    """Apply supervised outcome reward only to terminal training transitions."""
    from modules.module_07_rl.reward_model import apply_terminal_outcome_once

    mismatches = 0
    for transition in transitions:
        if transition.get("done") and "true_label" in transition:
            mismatches += int(apply_terminal_outcome_once(transition, omega))
    return mismatches


def group_transitions_into_episodes(transitions: list[dict]) -> list[list[dict]]:
    """Group transitions without allowing an episode to cross a split boundary."""
    if not transitions:
        return []

    if all("episode_id" in transition for transition in transitions):
        grouped = defaultdict(list)
        order = []
        for transition in transitions:
            episode_id = str(transition["episode_id"])
            if episode_id not in grouped:
                order.append(episode_id)
            grouped[episode_id].append(transition)
        return [grouped[episode_id] for episode_id in order]

    episodes = []
    current = []
    for transition in transitions:
        current.append(transition)
        if transition.get("done", False):
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return episodes


def _episode_group_key(episode: list[dict]) -> tuple[str, str]:
    first = episode[0]
    return (
        str(first.get("resume_content_hash") or first.get("resume_file", "unknown_resume")),
        str(first.get("jd_content_hash") or first.get("jd_file", "unknown_jd")),
    )


def _connected_resume_jd_groups(episodes: list[list[dict]]):
    """Group the bipartite resume/JD graph into leakage-safe components."""
    parent = {}

    def find(item):
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    episode_keys = []
    for episode in episodes:
        resume, jd = _episode_group_key(episode)
        resume_key = ("resume", resume)
        jd_key = ("jd", jd)
        union(resume_key, jd_key)
        episode_keys.append((episode, resume_key))

    components = defaultdict(list)
    for episode, resume_key in episode_keys:
        components[find(resume_key)].append(episode)
    return list(components.items())


def connected_identity_components(transitions: list[dict]):
    """Public identity-component view used by audits and bootstrap reporting."""
    return [
        episodes
        for _, episodes in _connected_resume_jd_groups(
            group_transitions_into_episodes(transitions)
        )
    ]


def split_by_resume_jd_group(
    transitions: list[dict],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split complete resume/JD groups to prevent evaluation leakage."""
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must contain three non-negative values")
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        raise ValueError("at least one split ratio must be positive")
    ratios = tuple(ratio / total_ratio for ratio in ratios)

    episodes = group_transitions_into_episodes(transitions)
    groups = _connected_resume_jd_groups(episodes)
    # Shuffle first for deterministic random tie-breaking, then place the
    # largest components first. Without this ordering, a small component can
    # consume the train target before a train-sized component is considered,
    # leaving validation or test empty even when an exact allocation exists.
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    split_targets = {
        "train": ratios[0] * len(episodes),
        "validation": ratios[1] * len(episodes),
        "test": ratios[2] * len(episodes),
    }
    assigned_episode_counts = {name: 0 for name in SPLIT_NAMES}
    split_episodes = {name: [] for name in SPLIT_NAMES}

    for _, group_episodes in groups:
        destination = max(
            SPLIT_NAMES,
            key=lambda name: split_targets[name] - assigned_episode_counts[name],
        )
        split_episodes[destination].extend(group_episodes)
        assigned_episode_counts[destination] += len(group_episodes)

    result = {name: [] for name in SPLIT_NAMES}
    for split_name, episodes_for_split in split_episodes.items():
        for episode in episodes_for_split:
            for transition in episode:
                copied = dict(transition)
                copied["dataset_split"] = split_name
                result[split_name].append(copied)
    return result


def write_splits(dataset_file: str | Path, output_dir: str | Path, seed: int = 42):
    dataset_path = Path(dataset_file)
    with dataset_path.open("r", encoding="utf-8") as handle:
        transitions = json.load(handle)

    splits = split_by_resume_jd_group(transitions, seed=seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for split_name, split_transitions in splits.items():
        with (output_path / f"{split_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(split_transitions, handle, indent=2)
    return {name: len(items) for name, items in splits.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("output_dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(write_splits(args.dataset_file, args.output_dir, seed=args.seed))
