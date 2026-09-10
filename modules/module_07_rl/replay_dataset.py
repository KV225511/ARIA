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
from modules.module_07_rl.transition_schema import (
    GENERATOR_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    has_valid_question_generation_provenance,
)
from modules.module_07_rl.state_builder import (
    STATE_SCHEMA_VERSION,
    STATE_FEATURE_NAMES,
    build_policy_state,
)


MIN_INTERVIEW_TURNS = 10
MIN_SKILLS_COVERED = 5
MAX_TURNS = 30
REPLAY_SCHEMA_VERSION = "aria-replay-v4"


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


def _action_mask(
    turn_id, assessment, total_skills, config, valid_evidence_count,
    current_skill=None,
):
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
    if current_skill is None:
        mask[RL_ACTION_SPACE.index("ask_follow_up_same_topic")] = 0.0
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
                f"Raw transition is not {TRANSITION_SCHEMA_VERSION}; regenerate it because "
                "legacy action propensities cannot be reconstructed safely"
            )
        if source.get("generator_schema_version") != GENERATOR_SCHEMA_VERSION:
            raise ValueError("Raw transition uses an incompatible generator schema")
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
            or np.any(source_probabilities[source_mask == 0.0] != 0.0)
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
            current_skill,
        )
        if not np.array_equal(source_mask, np.asarray(action_mask, dtype=float)):
            raise ValueError(
                "Raw transition pre-action mask does not match replayed legality"
            )
        if action_name == "conclude_interview":
            if source.get("target_skill_id") is not None:
                raise ValueError("Raw stop transition contains a target skill")
            if source.get("question_grounding_valid") is not None:
                raise ValueError("Raw stop transition contains question grounding")
        else:
            if not has_valid_question_generation_provenance(source):
                raise ValueError(
                    "Raw question transition has invalid generation provenance"
                )
            if source.get("question_grounding_valid") is not True:
                raise ValueError("Raw question transition is not grounding-validated")
            if not source.get("target_skill_id"):
                raise ValueError("Raw question transition lacks a stable target skill")
            if not source.get("role_profile_hash") or not source.get("ontology_hash"):
                raise ValueError("Raw question transition lacks grounding provenance")
            if not source.get("grounding_contract_hash"):
                raise ValueError("Raw question transition lacks a grounding contract hash")
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
            current_skill if action_name == "conclude_interview" else target_skill,
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
                interview_turn_id, assessment, total_skills, config,
                valid_evidence_count,
                current_skill if action_name == "conclude_interview" else target_skill,
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


def migrate_split_manifest(transitions: list[dict], parent_manifest: dict, target_component_counts=(21, 6, 6), seed=42):
    """Move whole train components to validation without touching locked test.

    Selection uses only structural identity information and episode counts.
    """
    if parent_manifest.get("schema_version") != "aria-split-manifest-v3":
        raise ValueError("split migration requires an aria-split-manifest-v3 parent")
    unsigned_parent = dict(parent_manifest)
    stored_parent_hash = unsigned_parent.pop("manifest_hash", None)
    if not stored_parent_hash or stored_parent_hash != canonical_json_hash(unsigned_parent):
        raise ValueError("parent split manifest hash is invalid")
    raw_hash = canonical_json_hash(transitions)
    if parent_manifest.get("raw_dataset_hash") != raw_hash:
        raise ValueError("parent split manifest does not match raw transitions")
    if tuple(target_component_counts) != (21, 6, 6):
        raise ValueError("v4 migration supports only component targets (21, 6, 6)")
    assignments = dict(parent_manifest.get("assignments", {}))
    episodes = group_transitions_into_episodes(transitions)
    known_ids = {str(episode[0].get("episode_id")) for episode in episodes if episode}
    if set(assignments) != known_ids:
        raise ValueError("parent split assignments do not exactly match raw episodes")
    components = connected_identity_components(transitions)
    by_split = {name: [] for name in SPLIT_NAMES}
    for component in components:
        episode_ids = sorted(str(episode[0].get("episode_id")) for episode in component if episode)
        splits = {assignments[episode_id] for episode_id in episode_ids}
        if len(splits) != 1:
            raise ValueError("identity component crosses a parent split")
        split_name = next(iter(splits))
        if split_name not in by_split:
            raise ValueError("parent manifest contains an unknown split")
        by_split[split_name].append(component)
    counts = tuple(len(by_split[name]) for name in SPLIT_NAMES)
    if counts[2] != 6 or counts[1] > 6 or counts[0] < 21:
        raise ValueError(f"unsupported parent component allocation: {counts}")
    needed = 6 - counts[1]
    if counts[0] - needed != 21:
        raise ValueError(f"parent allocation cannot reach (21, 6, 6): {counts}")

    validation_episode_count = sum(len(component) for component in by_split["validation"])
    candidates = []
    for component in by_split["train"]:
        first_records = [episode[0] for episode in component if episode]
        if not first_records or any(not item.get("resume_content_hash") or not item.get("jd_content_hash") for item in first_records):
            continue
        identity = {
            "episode_ids": sorted(str(item.get("episode_id")) for item in first_records),
            "resume_content_hashes": sorted(str(item["resume_content_hash"]) for item in first_records),
            "jd_content_hashes": sorted(str(item["jd_content_hash"]) for item in first_records),
            "seed": int(seed),
        }
        candidates.append((component, abs(validation_episode_count + len(component) - 90), canonical_json_hash(identity)))
    candidates.sort(key=lambda item: (item[1], item[2]))
    if len(candidates) < needed:
        raise ValueError("not enough structurally eligible train components for migration")
    moved = [item[0] for item in candidates[:needed]]
    moved_ids = {str(episode[0].get("episode_id")) for component in moved for episode in component if episode}
    for episode_id in moved_ids:
        assignments[episode_id] = "validation"

    split_items = {name: [] for name in SPLIT_NAMES}
    for episode in episodes:
        if episode:
            split_items[assignments[str(episode[0].get("episode_id"))]].extend(episode)
    summary = {}
    for name, items in split_items.items():
        summary[name] = {
            "transitions": len(items),
            "episodes": len(group_transitions_into_episodes(items)),
            "identity_components": len(connected_identity_components(items)),
            "resume_content_hashes": sorted({str(item["resume_content_hash"]) for item in items if item.get("resume_content_hash")}),
            "jd_content_hashes": sorted({str(item["jd_content_hash"]) for item in items if item.get("jd_content_hash")}),
        }
    actual = tuple(summary[name]["identity_components"] for name in SPLIT_NAMES)
    if actual != tuple(target_component_counts):
        raise RuntimeError(f"migration produced unexpected component allocation: {actual}")
    locked_payload = {
        "episode_ids": sorted(key for key, value in assignments.items() if value == "test"),
        "resume_content_hashes": summary["test"]["resume_content_hashes"],
        "jd_content_hashes": summary["test"]["jd_content_hashes"],
    }
    locked_hash = canonical_json_hash(locked_payload)
    if locked_hash != parent_manifest.get("locked_test_assignment_hash"):
        raise RuntimeError("locked test assignment changed during migration")
    manifest = {
        "schema_version": "aria-split-manifest-v4",
        "parent_manifest_hash": stored_parent_hash,
        "raw_dataset_hash": raw_hash,
        "seed": int(seed),
        "migration_algorithm_version": "train-to-validation-component-rebalance-v1",
        "requested_component_counts": list(target_component_counts),
        "actual_component_counts": list(actual),
        "assignments": assignments,
        "summary": summary,
        "locked_test_assignment_hash": locked_hash,
        "locked_test_assignment_preserved": True,
    }
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    return manifest


def freeze_development_splits(
    raw_file, parent_manifest_file, output_dir, seed=42,
    dependency_lock_path=None, expected_counts=None,
):
    """Create immutable train/validation inputs without replaying locked test."""
    existing_protocol = Path(output_dir) / "protocol" / "calibration_protocol_v4.json"
    if existing_protocol.exists():
        raise FileExistsError(
            f"calibration protocol v4 is already frozen: {existing_protocol}"
        )
    raw_path = Path(raw_file)
    before_bytes = raw_path.read_bytes()
    transitions = json.loads(before_bytes.decode("utf-8"))
    parent = json.loads(Path(parent_manifest_file).read_text(encoding="utf-8"))
    manifest = migrate_split_manifest(transitions, parent, seed=seed)
    output = Path(output_dir)
    assignments = manifest["assignments"]
    for split_name in SPLIT_NAMES:
        items = []
        for episode in group_transitions_into_episodes(transitions):
            if episode and assignments[str(episode[0].get("episode_id"))] == split_name:
                items.extend({**transition, "dataset_split": split_name} for transition in episode)
        destination = output / "raw-splits" / ("locked" if split_name == "test" else "") / f"{split_name}.json"
        _atomic_json_write(destination, items)
    from modules.module_07_rl.calibration_protocol import file_sha256, freeze_calibration_protocol
    lock_path = dependency_lock_path or Path(__file__).resolve().parents[2] / "requirements.txt"
    protocol = freeze_calibration_protocol(
        raw_path, manifest, output, lock_path,
        expected_counts=expected_counts,
    )
    manifest.update({
        "producer_version": "aria-belief-calibration-v4",
        "supported_consumer_versions": ["aria-split-manifest-v4"],
        "protocol_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
        "git_commit": protocol["code_commit"],
        "parent_artifact_hashes": {"split_manifest_v3": manifest["parent_manifest_hash"]},
    })
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    _atomic_json_write(output / "manifests" / "split_manifest_v4.json", manifest)
    raw_split_inventory = {
        "schema_version": "aria-raw-split-inventory-v1",
        "producer_version": "aria-belief-calibration-v4",
        "supported_consumer_versions": ["aria-raw-split-inventory-v1"],
        "protocol_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"],
        "raw_dataset_hash": protocol["raw_dataset_hash"],
        "split_manifest_hash": manifest["manifest_hash"],
        "parent_artifact_hashes": {"raw_corpus": protocol["raw_file_sha256"]},
        "git_commit": protocol["code_commit"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
        "artifacts": {
            "train": file_sha256(output / "raw-splits" / "train.json"),
            "validation": file_sha256(output / "raw-splits" / "validation.json"),
            "locked_test": file_sha256(output / "raw-splits" / "locked" / "test.json"),
        },
    }
    raw_split_inventory["inventory_hash"] = canonical_json_hash(raw_split_inventory)
    _atomic_json_write(output / "manifests" / "raw_split_inventory_v1.json", raw_split_inventory)
    for directory in ("calibration", "replayed", "audits", "release"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    after_bytes = raw_path.read_bytes()
    if after_bytes != before_bytes:
        raise RuntimeError("raw dataset changed while freezing development splits")
    return manifest


def prepare_development_calibration(
    train_file, validation_file, manifest_file, protocol_file, output_dir,
    bootstrap_samples=1000,
):
    """Fit/replay development data only; this function has no test-data input."""
    from modules.module_07_rl.belief_calibration import (
        select_training_only_calibration,
        validate_selected_calibration,
    )
    from modules.module_07_rl.calibration_protocol import (
        DEVELOPMENT_BUNDLE_VERSION,
        atomic_json_write,
        file_sha256,
        update_protocol_status,
        validate_calibration_protocol,
    )
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "aria-split-manifest-v4":
        raise ValueError("development calibration requires split_manifest_v4")
    unsigned_manifest = dict(manifest)
    stored_manifest_hash = unsigned_manifest.pop("manifest_hash", None)
    if not stored_manifest_hash or stored_manifest_hash != canonical_json_hash(unsigned_manifest):
        raise ValueError("development split manifest hash is invalid")
    protocol = validate_calibration_protocol(protocol_file, split_manifest=manifest)
    if protocol["protocol_status"] != "FROZEN_FOR_DEVELOPMENT":
        raise ValueError("protocol is not frozen for a first development validation")
    if int(protocol.get("validation_executions", 0)) != 0:
        raise ValueError("validation has already executed for this protocol")
    training = json.loads(Path(train_file).read_text(encoding="utf-8"))
    if any(item.get("dataset_split") != "train" for item in training):
        raise ValueError("development calibration training input contains a non-train split")
    raw_hash = manifest["raw_dataset_hash"]
    if raw_hash != protocol["raw_dataset_hash"]:
        raise ValueError("manifest and protocol raw dataset hashes differ")
    selection = select_training_only_calibration(
        training, raw_dataset_hash=raw_hash,
        split_manifest_hash=manifest["manifest_hash"],
        protocol_hash=protocol["protocol_hash"],
    )
    output = Path(output_dir)
    serializable_selection = dict(selection)
    if isinstance(serializable_selection.get("config"), BeliefModelConfig):
        serializable_selection["config"] = serializable_selection["config"].to_dict()
    serializable_selection.update({
        "raw_file_sha256": protocol["raw_file_sha256"],
        "parent_artifact_hashes": {"split_manifest": manifest["manifest_hash"]},
        "git_commit": protocol["code_commit"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
    })
    serializable_selection.pop("report_hash", None)
    serializable_selection["report_hash"] = canonical_json_hash(serializable_selection)
    selection_artifact_hash = serializable_selection["report_hash"]
    _atomic_json_write(output / "calibration" / "training_cv_report_v1.json", serializable_selection)
    if selection["selection_status"] != "ELIGIBLE":
        updated = update_protocol_status(protocol_file, "FAILED")
        return {
            "schema_version": "aria-development-calibration-v4",
            "protocol_hash": updated["protocol_hash"],
            "selection_status": "FAILED",
            "validation_evaluated": False,
            "test_metrics_locked": True,
        }
    config = selection["config"]
    metadata = dict(config.fit_metadata)
    metadata.update({
        "protocol_freeze_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"],
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
        "git_commit": protocol["code_commit"],
        "producer_version": "aria-belief-calibration-v4",
        "supported_consumer_versions": ["belief-v2"],
        "parent_artifact_hashes": {
            "split_manifest": manifest["manifest_hash"],
            "training_cv_report": selection_artifact_hash,
        },
    })
    # Metadata and behavior are frozen together. The config hash below is the
    # exact hash evaluated on validation and later admitted by the protocol.
    config = config.with_updates(fit_metadata=metadata)
    config.save(output / "calibration" / "belief_model_v2.json")
    # Claim the sole validation execution before reading any validation labels.
    # An interruption leaves validation_executions=1 and therefore fails closed.
    claimed_protocol = update_protocol_status(
        protocol_file, "FROZEN_FOR_DEVELOPMENT",
        validation_executions=1, validation_state="STARTED",
    )
    # Validation is intentionally not opened until selection is final and the
    # single execution has been durably claimed.
    try:
        validation = json.loads(Path(validation_file).read_text(encoding="utf-8"))
        if any(item.get("dataset_split") != "validation" for item in validation):
            raise ValueError("development validation input contains a non-validation split")
        validation_result = validate_selected_calibration(
            validation, config, bootstrap_samples=bootstrap_samples,
        )
    except BaseException:
        update_protocol_status(
            protocol_file, "FAILED", validation_executions=1,
            validation_state="INTERRUPTED", belief_config_hash=config.config_hash,
        )
        raise
    validation_passed = validation_result["passes_quality_gates"]
    validation_result["belief_config_hash"] = config.config_hash
    updated_protocol = update_protocol_status(
        protocol_file,
        "PROVISIONAL_SYNTHETIC" if validation_passed else "FAILED",
        validation_executions=1,
        validation_state="COMPLETED",
        belief_config_hash=config.config_hash,
    )
    replayed = {}
    for split_name, transitions in (("train", training), ("validation", validation)):
        rows = []
        for episode in group_transitions_into_episodes(transitions):
            rows.extend(replay_one_episode(episode, config, raw_hash, manifest["manifest_hash"]))
        for row in rows:
            row["dataset_split"] = split_name
            row.update({
                "schema_version": REPLAY_SCHEMA_VERSION,
                "producer_version": "aria-belief-calibration-v4",
                "supported_consumer_versions": [REPLAY_SCHEMA_VERSION],
                "protocol_hash": updated_protocol["protocol_hash"],
                "raw_file_sha256": updated_protocol["raw_file_sha256"],
                "environment_fingerprint_hash": updated_protocol["environment_fingerprint_hash"],
                "git_commit": updated_protocol["code_commit"],
                "parent_artifact_hashes": {
                    "split_manifest": manifest["manifest_hash"],
                    "belief_config": config.config_hash,
                },
                "belief_config_hash": config.config_hash,
            })
        replayed[split_name] = rows
    for split_name, rows in replayed.items():
        _atomic_json_write(output / "replayed" / f"{split_name}.json", rows)
    report = {
        "schema_version": "aria-development-calibration-v4",
        "producer_version": "aria-belief-calibration-v4",
        "supported_consumer_versions": ["aria-development-calibration-v4"],
        "protocol_hash": updated_protocol["protocol_hash"],
        "raw_file_sha256": updated_protocol["raw_file_sha256"],
        "raw_dataset_hash": raw_hash,
        "split_manifest_hash": manifest["manifest_hash"],
        "belief_config_hash": config.config_hash,
        "parent_artifact_hashes": {
            "training_cv_report": selection_artifact_hash,
            "split_manifest": manifest["manifest_hash"],
        },
        "git_commit": updated_protocol["code_commit"],
        "environment_fingerprint_hash": updated_protocol["environment_fingerprint_hash"],
        "validation_used_for_selection": False,
        "validation_execution_count": 1,
        "test_metrics_locked": True,
        "calibration": {
            "config_hash": config.config_hash,
            "selection_report_hash": selection_artifact_hash,
            "candidate_count": selection["candidate_count"],
            "validation": validation_result,
        },
    }
    _atomic_json_write(output / "calibration" / "development_calibration_report_v4.json", report)
    artifacts = {
        "split_manifest": {"path": str(Path(manifest_file).resolve()), "sha256": file_sha256(manifest_file)},
        "replayed_train": {"path": str(output / "replayed" / "train.json"), "sha256": file_sha256(output / "replayed" / "train.json"), "split": "train"},
        "replayed_validation": {"path": str(output / "replayed" / "validation.json"), "sha256": file_sha256(output / "replayed" / "validation.json"), "split": "validation"},
        "belief_config": {"path": str(output / "calibration" / "belief_model_v2.json"), "sha256": file_sha256(output / "calibration" / "belief_model_v2.json")},
        "validation_report": {"path": str(output / "calibration" / "development_calibration_report_v4.json"), "sha256": file_sha256(output / "calibration" / "development_calibration_report_v4.json")},
    }
    bundle = {
        "schema_version": DEVELOPMENT_BUNDLE_VERSION,
        "producer_version": "aria-belief-calibration-v4",
        "supported_consumer_versions": [DEVELOPMENT_BUNDLE_VERSION],
        "protocol_hash": updated_protocol["protocol_hash"],
        "raw_file_sha256": updated_protocol["raw_file_sha256"],
        "raw_dataset_hash": raw_hash,
        "split_manifest_hash": manifest["manifest_hash"],
        "belief_config_hash": config.config_hash,
        "parent_artifact_hashes": {
            "protocol": updated_protocol["protocol_hash"],
            "split_manifest": manifest["manifest_hash"],
            "training_cv_report": selection_artifact_hash,
        },
        "git_commit": updated_protocol["code_commit"],
        "environment_fingerprint_hash": updated_protocol["environment_fingerprint_hash"],
        "artifacts": artifacts,
    }
    bundle["bundle_hash"] = canonical_json_hash(bundle)
    atomic_json_write(output / "manifests" / "development_bundle_v1.json", bundle)
    report["development_bundle_hash"] = bundle["bundle_hash"]
    return report


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
    """Disabled legacy workflow that replayed all splits, including test."""
    raise RuntimeError(
        "legacy calibrate/replay is disabled; use freeze_development_splits and "
        "prepare_development_calibration under protocol v4"
    )


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
