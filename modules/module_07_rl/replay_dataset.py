"""Immutable replay of raw ARIA evidence into versioned belief-v2 transitions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_06_belief.belief_state import BeliefStateUpdater
from modules.module_07_rl.dataset_split import (
    SPLIT_NAMES,
    connected_identity_components,
    group_transitions_into_episodes,
    split_by_resume_jd_group,
)
from modules.module_07_rl.reward_model import (
    REWARD_SCHEMA_VERSION,
    compute_step_reward,
    compute_stop_reward,
)
from modules.module_07_rl.rl_spec import ACTION_SCHEMA_VERSION, RL_ACTION_SPACE
from modules.module_07_rl.transition_schema import TRANSITION_SCHEMA_VERSION
from modules.module_07_rl.state_builder import (
    STATE_SCHEMA_VERSION,
    STATE_FEATURE_NAMES,
    build_policy_state,
)


MIN_INTERVIEW_TURNS = 10
MIN_SKILLS_COVERED = 5
MAX_TURNS = 30
REPLAY_SCHEMA_VERSION = "aria-replay-v3"


def canonical_json_hash(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _infer_ontology_size(first_transition: dict) -> tuple[int, str]:
    if isinstance(first_transition.get("ontology_nodes"), list):
        return max(len(first_transition["ontology_nodes"]), 1), "ontology_nodes"
    if first_transition.get("ontology_size"):
        return max(int(first_transition["ontology_size"]), 1), "ontology_size"
    obs = first_transition.get("obs")
    if isinstance(obs, list) and len(obs) >= 150:
        triples = np.asarray(obs[:150], dtype=float).reshape(50, 3)
        count = int(np.sum(np.any(np.abs(triples) > 1e-12, axis=1)))
        if count:
            return count, "legacy_observation"
    # Fixed fallback avoids looking at future target skills.
    return 50, "fixed_fallback"


def _question_fingerprint(transition: dict) -> str | None:
    if transition.get("question_fingerprint"):
        return str(transition["question_fingerprint"])
    question = transition.get("question")
    if not question:
        return None
    normalized = " ".join(str(question).lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _confidence(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _action_mask(turn_id, assessment, total_skills, config, valid_evidence_count):
    mask = [1.0] * len(RL_ACTION_SPACE)
    required = min(
        max(MIN_SKILLS_COVERED, config.minimum_skill_coverage), total_skills
    )
    can_conclude = (
        turn_id >= MIN_INTERVIEW_TURNS
        and len(assessment["visited_skills"]) >= required
        and valid_evidence_count >= 5
    )
    mask[RL_ACTION_SPACE.index("conclude_interview")] = float(can_conclude)
    return mask, can_conclude


def replay_one_episode(
    episode: list[dict],
    config: BeliefModelConfig,
    raw_dataset_hash: str,
    split_manifest_hash: str,
) -> list[dict]:
    if not episode:
        return []
    total_skills, ontology_size_source = _infer_ontology_size(episode[0])
    updater = BeliefStateUpdater([], config=config)
    current_skill = None
    consecutive_focus = 0
    previous = {}
    interview_turn_id = 0
    valid_evidence_count = 0
    replayed = []

    for turn_index, source in enumerate(episode):
        if source.get("transition_schema_version") != TRANSITION_SCHEMA_VERSION:
            raise ValueError(
                "Raw transition is not aria-transition-v3; regenerate it because "
                "legacy action propensities cannot be reconstructed safely"
            )
        if source.get("action_schema_version") != ACTION_SCHEMA_VERSION:
            raise ValueError("Raw transition uses an incompatible action schema")
        action_idx = int(source.get("action_idx", 0))
        if not 0 <= action_idx < len(RL_ACTION_SPACE):
            raise ValueError(f"Invalid action_idx in episode: {action_idx}")
        source_mask = np.asarray(source.get("action_mask_before", []), dtype=float)
        source_probabilities = np.asarray(
            source.get("behavior_action_probs", []), dtype=float
        )
        if source_mask.shape != (len(RL_ACTION_SPACE),) or source_mask[action_idx] != 1.0:
            raise ValueError("Raw transition selected a masked or unverified action")
        if (
            source_probabilities.shape != (len(RL_ACTION_SPACE),)
            or not np.all(np.isfinite(source_probabilities))
            or np.any(source_probabilities < 0.0)
            or not np.isclose(source_probabilities.sum(), 1.0)
            or not np.isclose(
                float(source.get("behavior_action_probability", -1.0)),
                source_probabilities[action_idx],
            )
        ):
            raise ValueError("Raw transition has invalid behavior-policy propensities")
        action_name = RL_ACTION_SPACE[action_idx]
        assessment_before = updater.get_aggregate_assessment()
        action_mask, can_conclude_before = _action_mask(
            interview_turn_id,
            assessment_before,
            total_skills,
            config,
            valid_evidence_count,
        )
        obs = build_policy_state(
            updater,
            total_skills=total_skills,
            turn_id=interview_turn_id,
            current_skill=current_skill,
            consecutive_focus_turns=consecutive_focus,
            valid_evidence_count=valid_evidence_count,
            previous=previous,
        )

        target_skill = source.get("target_skill")
        valid = (
            action_name != "conclude_interview"
            and source.get("evaluation_valid") is True
            and source.get("semantic_score") is not None
        )
        information_gain = 0.0
        old_count = 0
        if action_name != "conclude_interview":
            if not target_skill:
                target_skill = current_skill or "__unknown_skill__"
            target_skill = updater.ensure_skill(target_skill)
            old_entropy = updater._calculate_entropy(updater.get_belief(target_skill))
            old_count = updater.get_evidence_count(target_skill)
            old_ess = updater.get_effective_sample_size(target_skill)
        if valid and source.get("semantic_score") is not None:
            try:
                updater.update_belief(
                    target_skill,
                    source["semantic_score"],
                    source.get("cognitive_load", "low"),
                    behavior_score=source.get("behavior_score"),
                    evidence_confidence=source.get("evaluator_confidence", 1.0),
                    stt_confidence=source.get("stt_confidence", 1.0),
                    modality_confidence=source.get("modality_confidence", 1.0),
                    question_fingerprint=_question_fingerprint(source),
                )
            except (TypeError, ValueError):
                valid = False
        if action_name != "conclude_interview":
            interview_turn_id += 1
            if valid:
                valid_evidence_count += 1
            new_entropy = updater._calculate_entropy(updater.get_belief(target_skill))
            new_ess = updater.get_effective_sample_size(target_skill)
            information_gain = max(0.0, old_entropy - new_entropy) * max(new_ess - old_ess, 0.0)

        assessment = updater.get_aggregate_assessment()
        _, can_conclude = _action_mask(
            interview_turn_id,
            assessment,
            total_skills,
            config,
            valid_evidence_count,
        )
        if action_name == "conclude_interview":
            reward = compute_stop_reward(not can_conclude_before)
        else:
            reward = compute_step_reward(
                information_gain,
                first_skill_visit=old_count == 0,
                previous_skill_count=old_count,
                cognitive_load=source.get("cognitive_load", "low"),
            )

        derived_done = action_name == "conclude_interview" and can_conclude_before
        source_done = bool(source.get("done"))
        done = derived_done or (source_done and action_name != "conclude_interview") or interview_turn_id >= MAX_TURNS
        if derived_done:
            termination_reason = "explicit_conclusion"
        elif interview_turn_id >= MAX_TURNS:
            termination_reason = "max_turns"
        elif source_done:
            termination_reason = "source_episode_end"
        else:
            termination_reason = None

        reliability = float(
            _confidence(source.get("evaluator_confidence", 1.0))
            * _confidence(source.get("stt_confidence", 1.0))
            * _confidence(source.get("modality_confidence", 1.0))
        ) if valid else 0.0
        next_previous = previous if action_name == "conclude_interview" else {
            "semantic_score": source.get("semantic_score") if valid else None,
            "evidence_reliability": reliability,
            "behavior_score": source.get("behavior_score"),
            "cognitive_load": source.get("cognitive_load"),
            "incongruence_score": source.get("incongruence_score"),
            "action_idx": action_idx,
        }
        next_consecutive = (
            consecutive_focus if action_name == "conclude_interview"
            else consecutive_focus + 1 if current_skill == target_skill else 0
        )
        next_obs = build_policy_state(
            updater,
            total_skills=total_skills,
            turn_id=interview_turn_id,
            current_skill=current_skill if action_name == "conclude_interview" else target_skill,
            consecutive_focus_turns=next_consecutive,
            valid_evidence_count=valid_evidence_count,
            previous=next_previous,
        )

        derived = dict(source)
        for name in (
            "obs", "next_obs", "reward", "done", "aria_label", "aria_raw_label",
            "assessment_status", "aggregate_belief", "aggregate_confidence",
            "effective_evidence", "skills_covered", "evaluation_valid",
            "evidence_reliability", "information_gain", "action_mask",
            "action_mask_before",
            "termination_reason",
        ):
            if name in source:
                derived[f"raw_{name}"] = source[name]
        derived.update({
            "obs": obs.tolist(),
            "next_obs": next_obs.tolist(),
            "reward": reward,
            "done": done,
            "aria_label": assessment["label"],
            "aria_raw_label": assessment["raw_label"],
            "assessment_status": assessment["status"],
            "aggregate_belief": assessment["belief"].tolist(),
            "aggregate_confidence": assessment["confidence"],
            "effective_evidence": assessment["effective_evidence"],
            "skills_covered": len(assessment["visited_skills"]),
            "evaluation_valid": valid,
            "evidence_reliability": reliability,
            "information_gain": information_gain,
            "action_mask_before": action_mask,
            "action_mask": _action_mask(
                interview_turn_id, assessment, total_skills, config, valid_evidence_count
            )[0],
            "valid_evidence_count": valid_evidence_count,
            "transition_kind": (
                "stop" if action_name == "conclude_interview" else "question"
            ),
            "action_name": action_name,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "transition_schema_version": TRANSITION_SCHEMA_VERSION,
            "termination_reason": termination_reason,
            "ontology_size": total_skills,
            "ontology_size_source": ontology_size_source,
            "belief_config_hash": config.config_hash,
            "raw_dataset_hash": raw_dataset_hash,
            "split_manifest_hash": split_manifest_hash,
            "belief_schema_version": config.schema_version,
            "state_schema_version": STATE_SCHEMA_VERSION,
            "state_feature_names": list(STATE_FEATURE_NAMES),
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "replay_schema_version": REPLAY_SCHEMA_VERSION,
        })
        replayed.append(derived)
        if action_name != "conclude_interview":
            current_skill = target_skill
        consecutive_focus = next_consecutive
        previous = next_previous
        if done:
            break

    if replayed and not replayed[-1]["done"]:
        replayed[-1]["done"] = True
        replayed[-1]["termination_reason"] = "source_episode_exhausted"
    return replayed


def create_split_manifest(transitions: list[dict], seed=42):
    splits = split_by_resume_jd_group(transitions, seed=seed)
    assignments = {}
    split_summary = {}
    for split_name, items in splits.items():
        episodes = group_transitions_into_episodes(items)
        for episode in episodes:
            if episode:
                assignments[str(episode[0].get("episode_id"))] = split_name
        split_summary[split_name] = {
            "transitions": len(items),
            "episodes": len(episodes),
            "resumes": sorted({str(item.get("resume_file")) for item in items}),
            "jds": sorted({str(item.get("jd_file")) for item in items}),
            "resume_content_hashes": sorted({
                str(item.get("resume_content_hash")) for item in items
                if item.get("resume_content_hash")
            }),
            "jd_content_hashes": sorted({
                str(item.get("jd_content_hash")) for item in items
                if item.get("jd_content_hash")
            }),
            "identity_components": len(connected_identity_components(items)),
        }
    manifest = {
        "schema_version": "aria-split-manifest-v3",
        "raw_dataset_hash": canonical_json_hash(transitions),
        "state_schema_version": STATE_SCHEMA_VERSION,
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "seed": int(seed),
        "assignments": assignments,
        "summary": split_summary,
    }
    manifest["locked_test_assignment_hash"] = canonical_json_hash({
        "episode_ids": sorted(
            episode_id for episode_id, split_name in assignments.items()
            if split_name == "test"
        ),
        "resume_content_hashes": split_summary["test"]["resume_content_hashes"],
        "jd_content_hashes": split_summary["test"]["jd_content_hashes"],
    })
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    return manifest


def _classification_summary(pairs):
    valid = [(true, pred) for true, pred in pairs if true is not None]
    if not valid:
        return {"num_episodes": 0}
    correct = [true == pred for true, pred in valid]
    return {
        "num_episodes": len(valid),
        "accuracy_micro_f1": float(np.mean(correct)),
        "true_counts": dict(Counter(true for true, _ in valid)),
        "prediction_counts": dict(Counter(str(pred) for _, pred in valid)),
        "abstentions": sum(pred is None for _, pred in valid),
    }


def replay_dataset(transitions, config, split_manifest=None, unlock_test_report=False):
    raw_hash = canonical_json_hash(transitions)
    if config.raw_dataset_hash and config.raw_dataset_hash != raw_hash:
        raise ValueError("Belief config raw_dataset_hash does not match replay input")
    manifest = split_manifest or create_split_manifest(transitions)
    stored_manifest_hash = manifest.get("manifest_hash")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_hash", None)
    computed_manifest_hash = canonical_json_hash(unsigned_manifest)
    if stored_manifest_hash and stored_manifest_hash != computed_manifest_hash:
        raise ValueError("Split manifest content does not match its manifest_hash")
    manifest_hash = stored_manifest_hash or computed_manifest_hash
    if manifest.get("raw_dataset_hash") != raw_hash:
        raise ValueError("Split manifest raw_dataset_hash does not match replay input")
    if config.split_manifest_hash and config.split_manifest_hash != manifest_hash:
        raise ValueError("Belief config split_manifest_hash does not match replay split")
    assignments = manifest["assignments"]

    replayed = []
    raw_terminal_pairs = {name: [] for name in SPLIT_NAMES}
    derived_terminal_pairs = {name: [] for name in SPLIT_NAMES}
    for episode in group_transitions_into_episodes(transitions):
        if not episode:
            continue
        episode_id = str(episode[0].get("episode_id"))
        split_name = assignments.get(episode_id)
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"Episode {episode_id} is absent from split manifest")
        derived_episode = replay_one_episode(
            episode, config, raw_hash, manifest_hash
        )
        for item in derived_episode:
            item["dataset_split"] = split_name
        replayed.extend(derived_episode)
        raw_terminal_pairs[split_name].append((
            episode[-1].get("true_label"), episode[-1].get("aria_label")
        ))
        if derived_episode:
            derived_terminal_pairs[split_name].append((
                derived_episode[-1].get("true_label"),
                derived_episode[-1].get("aria_label"),
            ))
    report = {
        "schema_version": "aria-replay-comparison-v3",
        "raw_dataset_hash": raw_hash,
        "split_manifest_hash": manifest_hash,
        "belief_config_hash": config.config_hash,
        "belief_schema_version": config.schema_version,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "splits": {
            split_name: (
                {
                    "raw": _classification_summary(raw_terminal_pairs[split_name]),
                    "replayed": _classification_summary(derived_terminal_pairs[split_name]),
                }
                if split_name != "test" or unlock_test_report
                else {"status": "locked", "metrics_computed": False}
            )
            for split_name in SPLIT_NAMES
        },
        "derived_transitions": len(replayed),
    }
    return replayed, manifest, report


def replay_file(
    raw_file,
    config_file,
    output_dir,
    split_seed=42,
    unlock_test_report=False,
):
    raw_path = Path(raw_file)
    before_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    transitions = json.loads(raw_path.read_text(encoding="utf-8"))
    config = BeliefModelConfig.load(config_file)
    manifest = create_split_manifest(transitions, seed=split_seed)
    replayed, manifest, report = replay_dataset(
        transitions, config, manifest, unlock_test_report=unlock_test_report
    )
    after_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if before_hash != after_hash:
        raise RuntimeError("Raw dataset changed during replay")

    output = Path(output_dir)
    _atomic_json_write(output / "split_manifest_v3.json", manifest)
    _atomic_json_write(output / "qwen_rl_dataset_belief_v3.json", replayed)
    _atomic_json_write(output / "replay_comparison_v3.json", report)
    for split_name in SPLIT_NAMES:
        _atomic_json_write(
            output / "splits" / f"{split_name}.json",
            [item for item in replayed if item["dataset_split"] == split_name],
        )
    return report


def prepare_calibrate_replay(
    raw_file,
    output_dir,
    split_seed=42,
    bootstrap_samples=100,
):
    """Split raw evidence, fit/tune calibration, and replay without test metrics."""
    from modules.module_07_rl.belief_calibration import calibrate_belief_model

    raw_path = Path(raw_file)
    before_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    transitions = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_hash = canonical_json_hash(transitions)
    manifest = create_split_manifest(transitions, seed=split_seed)
    assignments = manifest["assignments"]
    raw_splits = {name: [] for name in SPLIT_NAMES}
    for episode in group_transitions_into_episodes(transitions):
        if episode:
            raw_splits[assignments[str(episode[0].get("episode_id"))]].extend(episode)
    calibration = calibrate_belief_model(
        raw_splits["train"],
        raw_splits["validation"],
        raw_dataset_hash=raw_hash,
        split_manifest_hash=manifest["manifest_hash"],
        bootstrap_samples=bootstrap_samples,
    )
    config = calibration["config"]
    fit_metadata = dict(config.fit_metadata)
    fit_metadata.update({
        "state_schema_version": STATE_SCHEMA_VERSION,
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "test_metrics_locked_at_freeze": True,
    })
    config = config.with_updates(fit_metadata=fit_metadata)
    calibration["config"] = config
    calibration["config_hash"] = config.config_hash
    replayed, manifest, comparison = replay_dataset(
        transitions, config, manifest, unlock_test_report=False
    )
    after_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if before_hash != after_hash:
        raise RuntimeError("Raw dataset changed during calibration/replay")

    output = Path(output_dir)
    config.save(output / "belief_model_v2.json")
    _atomic_json_write(output / "split_manifest_v3.json", manifest)
    _atomic_json_write(output / "qwen_rl_dataset_belief_v3.json", replayed)
    _atomic_json_write(output / "replay_comparison_v3.json", comparison)
    calibration_payload = {
        key: value.to_dict() if isinstance(value, BeliefModelConfig) else value
        for key, value in calibration.items()
    }
    calibration_report = {
        "schema_version": "aria-calibration-report-v3",
        "raw_dataset_hash": raw_hash,
        "split_manifest_hash": manifest["manifest_hash"],
        "belief_config_hash": config.config_hash,
        "belief_schema_version": config.schema_version,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "test_metrics_locked": True,
        "calibration": calibration_payload,
    }
    _atomic_json_write(output / "calibration_report_v3.json", calibration_report)
    for split_name in SPLIT_NAMES:
        _atomic_json_write(
            output / "splits" / f"{split_name}.json",
            [item for item in replayed if item["dataset_split"] == split_name],
        )
    return {
        "calibration": calibration_report,
        "comparison": comparison,
        "test_metrics_locked": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file")
    parser.add_argument("config_file")
    parser.add_argument("output_dir")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--unlock-test-report", action="store_true")
    args = parser.parse_args()
    print(json.dumps(replay_file(
        args.raw_file,
        args.config_file,
        args.output_dir,
        args.split_seed,
        unlock_test_report=args.unlock_test_report,
    ), indent=2))
