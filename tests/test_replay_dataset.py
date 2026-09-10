from copy import deepcopy
import json
import pytest

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.replay_dataset import replay_dataset
from modules.module_07_rl.rl_spec import ACTION_SCHEMA_VERSION
from modules.module_07_rl.transition_schema import (
    GENERATOR_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
)
from modules.module_07_rl.state_builder import (
    STATE_DIM,
    STATE_FEATURE_NAMES,
    STATE_SCHEMA_VERSION,
)


def _raw_episode(index, label, score):
    return [
        {
            "episode_id": f"episode-{index}",
            "resume_file": f"resume-{index}.pdf",
            "jd_file": f"jd-{index}.pdf",
            "true_label": label,
            "aria_label": 1,
            "target_skill": "Python",
            "semantic_score": score,
            "behavior_score": 1.0 - score,
            "cognitive_load": "low",
            "evaluator_confidence": 1.0,
            "evaluation_valid": True,
            "action_idx": 3,
            "action": [0, 0, 0, 1, 0, 0, 0, 0],
            "reward": -0.1,
            "obs": [1 / 3] * 6 + [0.0] * 144 + [1.0, 0.0],
            "next_obs": [1 / 3] * 6 + [0.0] * 144 + [1.0, 0.1],
            "done": turn == 1,
            "question": f"Question {turn}",
            "transition_schema_version": TRANSITION_SCHEMA_VERSION,
            "generator_schema_version": GENERATOR_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "action_mask_before": (
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
                if turn == 0 else [1.0] * 7 + [0.0]
            ),
            "behavior_action_probs": (
                [1.0 / 6.0, 1.0 / 6.0, 0.0, 1.0 / 6.0,
                 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.0]
                if turn == 0 else [1.0 / 7.0] * 7 + [0.0]
            ),
            "behavior_action_probability": 1.0 / (6.0 if turn == 0 else 7.0),
            "target_skill_id": "python",
            "question_grounding_valid": True,
            "role_profile_hash": f"profile-{index}",
            "ontology_hash": f"ontology-{index}",
            "grounding_contract_hash": "grounding-contract-hash",
            "question_generation_attempts": 1,
            "llm_question_generation_attempts": 1,
            "deterministic_question_generation_attempts": 0,
            "question_generation_mode": "llm",
            "fallback_question_template_version": None,
            "question_prompt_hash": f"prompt-hash-{index}-{turn}",
            "question_generation_seed": index * 100 + turn,
        }
        for turn in range(2)
    ]


def _raw_dataset():
    transitions = []
    for index in range(9):
        label = index % 3
        transitions.extend(_raw_episode(index, label, (0.1, 0.5, 0.9)[label]))
    return transitions


def test_replay_is_deterministic_and_does_not_mutate_raw_data():
    raw = _raw_dataset()
    snapshot = json.dumps(raw, sort_keys=True)
    first, manifest, _ = replay_dataset(raw, BeliefModelConfig())
    second, _, _ = replay_dataset(raw, BeliefModelConfig(), manifest)
    assert first == second
    assert json.dumps(raw, sort_keys=True) == snapshot


def test_replay_rejects_a_tampered_locked_split_manifest():
    raw = _raw_dataset()
    _, manifest, _ = replay_dataset(raw, BeliefModelConfig())
    tampered = deepcopy(manifest)
    episode_id = next(iter(tampered["assignments"]))
    tampered["assignments"][episode_id] = "test"
    with pytest.raises(ValueError, match="manifest_hash"):
        replay_dataset(raw, BeliefModelConfig(), tampered)


def test_replay_builds_versioned_fixed_states_and_preserves_raw_fields():
    replayed, _, report = replay_dataset(_raw_dataset(), BeliefModelConfig())
    assert replayed
    assert all(len(item["obs"]) == STATE_DIM for item in replayed)
    assert all(item["state_schema_version"] == STATE_SCHEMA_VERSION for item in replayed)
    assert all(item["state_feature_names"] == list(STATE_FEATURE_NAMES) for item in replayed)
    assert all("raw_obs" in item and "raw_reward" in item for item in replayed)
    assert report["derived_transitions"] == len(replayed)


def test_first_state_does_not_include_future_evidence():
    raw = _raw_dataset()
    changed = deepcopy(raw)
    changed[1]["semantic_score"] = 0.0
    first, _, _ = replay_dataset(raw, BeliefModelConfig())
    second, _, _ = replay_dataset(changed, BeliefModelConfig())
    assert first[0]["obs"] == second[0]["obs"]


def test_stable_state_feature_names_match_vector_positions():
    replayed, _, _ = replay_dataset(_raw_dataset(), BeliefModelConfig())
    first = dict(zip(STATE_FEATURE_NAMES, replayed[0]["obs"]))
    second = dict(zip(STATE_FEATURE_NAMES, replayed[0]["next_obs"]))
    assert first["semantic_available"] == 0.0
    assert second["semantic_available"] == 1.0
    assert abs(second["previous_semantic_score"] - 0.1) < 1e-6
    assert "action_mask" not in STATE_FEATURE_NAMES
def test_split_migration_moves_one_component_without_changing_locked_test():
    from modules.module_07_rl.replay_dataset import canonical_json_hash, migrate_split_manifest
    transitions = []
    assignments = {}
    for index in range(33):
        episode_id = f"episode-{index}"
        split = "train" if index < 22 else "validation" if index < 27 else "test"
        assignments[episode_id] = split
        transitions.append({
            "episode_id": episode_id,
            "resume_content_hash": f"resume-{index}",
            "jd_content_hash": f"jd-{index}",
            "done": True,
        })
    test_ids = sorted(key for key, value in assignments.items() if value == "test")
    parent = {
        "schema_version": "aria-split-manifest-v3",
        "raw_dataset_hash": canonical_json_hash(transitions),
        "assignments": assignments,
        "locked_test_assignment_hash": canonical_json_hash({
            "episode_ids": test_ids,
            "resume_content_hashes": [f"resume-{index}" for index in range(27, 33)],
            "jd_content_hashes": [f"jd-{index}" for index in range(27, 33)],
        }),
    }
    parent["manifest_hash"] = canonical_json_hash(parent)
    migrated = migrate_split_manifest(transitions, parent)
    assert migrated["actual_component_counts"] == [21, 6, 6]
    assert migrated["locked_test_assignment_hash"] == parent["locked_test_assignment_hash"]
    assert all(migrated["assignments"][episode_id] == "test" for episode_id in test_ids)


def test_fixture_end_to_end_development_never_replays_locked_test(tmp_path):
    from modules.module_07_rl.replay_dataset import (
        canonical_json_hash,
        freeze_development_splits,
        prepare_development_calibration,
    )
    transitions = []
    assignments = {}
    for index in range(33):
        rows = _raw_episode(index, index % 3, (0.1, 0.5, 0.9)[index % 3])
        third = dict(rows[-1])
        rows[-1]["done"] = False
        third["done"] = True
        third["question"] = "Question 2"
        third["question_prompt_hash"] = f"prompt-hash-{index}-2"
        third["question_generation_seed"] = index * 100 + 2
        rows.append(third)
        for turn, row in enumerate(rows):
            row["target_skill"] = f"Skill-{turn}"
            row["target_skill_id"] = f"skill-{turn}"
            row["resume_content_hash"] = f"resume-content-{index}"
            row["jd_content_hash"] = f"jd-content-{index}"
        transitions.extend(rows)
        assignments[f"episode-{index}"] = (
            "train" if index < 22 else "validation" if index < 27 else "test"
        )
    parent = {
        "schema_version": "aria-split-manifest-v3",
        "raw_dataset_hash": canonical_json_hash(transitions),
        "assignments": assignments,
        "locked_test_assignment_hash": canonical_json_hash({
            "episode_ids": [f"episode-{index}" for index in range(27, 33)],
            "resume_content_hashes": [f"resume-content-{index}" for index in range(27, 33)],
            "jd_content_hashes": [f"jd-content-{index}" for index in range(27, 33)],
        }),
    }
    parent["manifest_hash"] = canonical_json_hash(parent)
    raw_file = tmp_path / "qwen_rl_dataset.json"
    parent_file = tmp_path / "split_manifest_v3.json"
    lock_file = tmp_path / "requirements.lock"
    raw_file.write_text(json.dumps(transitions, indent=2), encoding="utf-8")
    parent_file.write_text(json.dumps(parent), encoding="utf-8")
    lock_file.write_text("numpy==fixture\n", encoding="utf-8")
    before = raw_file.read_bytes()
    output = tmp_path / "derived-calibration-v4"
    frozen = freeze_development_splits(
        raw_file, parent_file, output,
        dependency_lock_path=lock_file,
        expected_counts=(33, 99, 33),
    )
    report = prepare_development_calibration(
        output / "raw-splits" / "train.json",
        output / "raw-splits" / "validation.json",
        output / "manifests" / "split_manifest_v4.json",
        output / "protocol" / "calibration_protocol_v4.json",
        output,
        bootstrap_samples=10,
    )
    protocol = json.loads(
        (output / "protocol" / "calibration_protocol_v4.json").read_text()
    )
    assert report["validation_used_for_selection"] is False
    assert protocol["validation_executions"] == 1
    assert protocol["protocol_status"] == "PROVISIONAL_SYNTHETIC"
    assert frozen["locked_test_assignment_hash"] == parent["locked_test_assignment_hash"]
    assert raw_file.read_bytes() == before
    assert not (output / "replayed" / "test.json").exists()
    assert not (output / "release" / "locked_test_evaluation_v1.json").exists()
    with pytest.raises(ValueError, match="not frozen for a first"):
        prepare_development_calibration(
            output / "raw-splits" / "train.json",
            output / "raw-splits" / "validation.json",
            output / "manifests" / "split_manifest_v4.json",
            output / "protocol" / "calibration_protocol_v4.json",
            output,
            bootstrap_samples=10,
        )


def test_v5_fixture_end_to_end_is_additive_and_test_locked(tmp_path):
    from modules.module_07_rl.prepare_belief_pipeline_v5 import (
        freeze_development_splits_v5,
        prepare_development_calibration_v5,
    )
    from modules.module_07_rl.replay_dataset import canonical_json_hash

    transitions = []
    assignments = {}
    for index in range(33):
        rows = _raw_episode(index, index % 3, (0.1, 0.5, 0.9)[index % 3])
        terminal = dict(rows[-1])
        rows[-1]["done"] = False
        terminal.update({
            "done": True,
            "question": "Question 2",
            "question_prompt_hash": f"prompt-hash-{index}-2",
            "question_generation_seed": index * 100 + 2,
        })
        rows.append(terminal)
        for turn, row in enumerate(rows):
            row.update({
                "target_skill": f"Skill-{turn}",
                "target_skill_id": f"skill-{turn}",
                "resume_content_hash": f"resume-content-{index}",
                "jd_content_hash": f"jd-content-{index}",
            })
        transitions.extend(rows)
        assignments[f"episode-{index}"] = (
            "train" if index < 22 else "validation" if index < 27 else "test"
        )
    parent = {
        "schema_version": "aria-split-manifest-v3",
        "raw_dataset_hash": canonical_json_hash(transitions),
        "assignments": assignments,
        "locked_test_assignment_hash": canonical_json_hash({
            "episode_ids": [f"episode-{index}" for index in range(27, 33)],
            "resume_content_hashes": [f"resume-content-{index}" for index in range(27, 33)],
            "jd_content_hashes": [f"jd-content-{index}" for index in range(27, 33)],
        }),
    }
    parent["manifest_hash"] = canonical_json_hash(parent)
    raw_file = tmp_path / "qwen_rl_dataset.json"
    parent_file = tmp_path / "split_manifest_v3.json"
    lock_file = tmp_path / "requirements.lock"
    raw_file.write_text(json.dumps(transitions, indent=2), encoding="utf-8")
    parent_file.write_text(json.dumps(parent), encoding="utf-8")
    lock_file.write_text("numpy==fixture\n", encoding="utf-8")
    before = raw_file.read_bytes()
    output = tmp_path / "derived-calibration-v5"
    frozen = freeze_development_splits_v5(
        raw_file,
        parent_file,
        output,
        dependency_lock_path=lock_file,
        expected_counts=(33, 99, 33),
    )
    report = prepare_development_calibration_v5(
        output / "raw-splits" / "train.json",
        output / "raw-splits" / "validation.json",
        output / "manifests" / "split_manifest_v4.json",
        output / "protocol" / "calibration_protocol_v5.json",
        output,
        bootstrap_samples=10,
    )
    protocol = json.loads(
        (output / "protocol" / "calibration_protocol_v5.json").read_text()
    )
    assert report["validation_used_for_selection"] is False
    assert report["calibration"]["candidate_count"] <= 4
    assert protocol["validation_executions"] == 1
    assert protocol["protocol_status"] == "PROVISIONAL_SYNTHETIC"
    assert frozen["locked_test_assignment_hash"] == parent["locked_test_assignment_hash"]
    assert raw_file.read_bytes() == before
    assert not (output / "replayed" / "test.json").exists()
    assert not (output / "release" / "locked_test_evaluation_v1.json").exists()
    with pytest.raises(ValueError, match="not frozen for a first"):
        prepare_development_calibration_v5(
            output / "raw-splits" / "train.json",
            output / "raw-splits" / "validation.json",
            output / "manifests" / "split_manifest_v4.json",
            output / "protocol" / "calibration_protocol_v5.json",
            output,
            bootstrap_samples=10,
        )
