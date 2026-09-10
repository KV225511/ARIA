import json
from unittest.mock import patch

import pytest
import torch

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.state_builder import (
    STATE_DIM,
    STATE_FEATURE_NAMES,
    STATE_SCHEMA_VERSION,
)
from modules.module_07_rl.train import (
    CHECKPOINT_SCHEMA_VERSION,
    train_iql_policy,
    validate_replayed_dataset,
)
from modules.module_07_rl.calibration_protocol import (
    CALIBRATION_ALGORITHM_VERSION,
    CALIBRATION_CANDIDATE_VALUES,
    CALIBRATION_PROTOCOL_VERSION,
    CALIBRATION_STAGE_SEQUENCE,
    DEVELOPMENT_BUNDLE_VERSION,
    NUMERICAL_TOLERANCES,
    canonical_json_hash,
    file_sha256,
)
from modules.module_07_rl.dataset_audit import CALIBRATION_GATE_THRESHOLDS, VALIDATION_GATE_VERSION
from modules.module_07_rl.metrics import METRICS_SCHEMA_VERSION
from modules.module_07_rl.rl_spec import ACTION_SCHEMA_VERSION
from modules.module_07_rl.reward_model import REWARD_SCHEMA_VERSION
from modules.module_07_rl.transition_schema import (
    GENERATOR_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
)
from modules.module_05_ontology.grounding import (
    GROUNDING_POLICY_VERSION,
    GROUNDING_SCHEMA_VERSION,
    ROLE_PROFILE_SCHEMA_VERSION,
    grounding_contract_hash,
)


def _transition(index, split):
    label = index % 3
    action = index % 8
    is_stop = action == 7
    state = [0.0] * STATE_DIM
    state[label] = 1.0
    return {
        "episode_id": f"{split}-{index}",
        "resume_file": f"{split}-resume-{index}.pdf",
        "jd_file": f"{split}-jd-{index}.pdf",
        "dataset_split": split,
        "true_label": label,
        "aria_label": label,
        "semantic_score": None if is_stop else (0.1, 0.5, 0.9)[label],
        "behavior_score": None if is_stop else 0.5,
        "cognitive_load": None if is_stop else "low",
        "evaluator_confidence": None if is_stop else 1.0,
        "evaluation_valid": None if is_stop else True,
        "action_idx": action,
        "obs": state,
        "next_obs": state,
        "reward": 0.1 * action,
        "done": True,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "state_feature_names": list(STATE_FEATURE_NAMES),
        "belief_config_hash": BeliefModelConfig().config_hash,
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "transition_kind": "stop" if is_stop else "question",
        "role_profile_hash": "profile-hash",
        "role_profile_schema_version": ROLE_PROFILE_SCHEMA_VERSION,
        "grounding_contract_hash": grounding_contract_hash(),
        "ontology_hash": "ontology-hash",
        "target_skill_id": None if is_stop else "python",
        "question_grounding_schema_version": GROUNDING_SCHEMA_VERSION,
        "question_grounding_valid": None if is_stop else True,
        "question_grounding": None if is_stop else {
            "schema_version": GROUNDING_SCHEMA_VERSION,
            "grounding_policy_version": GROUNDING_POLICY_VERSION,
            "target_skill_id": "python",
            "role_profile_hash": "profile-hash",
            "decision": "accept",
            "valid": True,
            "reasons": [],
        },
        "question_generation_attempts": None if is_stop else 1,
        "llm_question_generation_attempts": None if is_stop else 1,
        "deterministic_question_generation_attempts": None if is_stop else 0,
        "question_generation_mode": None if is_stop else "llm",
        "fallback_question_template_version": None,
        "question_prompt_hash": None if is_stop else f"prompt-{split}-{index}",
        "question_generation_seed": None if is_stop else index,
        "pairing_record": {"pairing_class": "evidence_overlap"},
        "generation_run_id": "generation-run",
        "plan_id": "generation-plan",
        "action_mask_before": [1.0] * 8,
        "behavior_action_probs": [1.0 / 8.0] * 8,
        "behavior_action_probability": 1.0 / 8.0,
    }


def _write_training_contract(tmp_path, train, validation):
    manifest = {
        "schema_version": "aria-split-manifest-v4",
        "raw_dataset_hash": "raw-canonical-hash",
        "locked_test_assignment_hash": "locked-assignment-hash",
        "assignments": {},
    }
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    manifest_file = tmp_path / "split_manifest_v4.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    config = BeliefModelConfig(
        split_manifest_hash=manifest["manifest_hash"],
        raw_dataset_hash=manifest["raw_dataset_hash"],
    )
    config_file = tmp_path / "belief.json"
    config.save(config_file)
    protocol = {
        "protocol_schema_version": CALIBRATION_PROTOCOL_VERSION,
        "protocol_status": "PROVISIONAL_SYNTHETIC",
        "raw_file_path": "fixture.json", "raw_file_sha256": "raw-byte-hash",
        "raw_dataset_hash": manifest["raw_dataset_hash"], "episode_count": 184,
        "transition_count": 184, "identity_component_count": 184,
        "parent_split_manifest_hash": "parent", "locked_test_assignment_hash": "locked-assignment-hash",
        "target_component_counts": [21, 6, 6], "metric_schema_version": METRICS_SCHEMA_VERSION,
        "gate_policy_version": VALIDATION_GATE_VERSION, "gate_thresholds": CALIBRATION_GATE_THRESHOLDS,
        "split_migration_algorithm": "train-to-validation-component-rebalance-v1",
        "cross_validation_algorithm": "grouped-identity-component-3fold-greedy-v1",
        "calibration_algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "calibration_stage_sequence": CALIBRATION_STAGE_SEQUENCE,
        "candidate_values": CALIBRATION_CANDIDATE_VALUES,
        "candidate_selection_rule": ["highest_macro_f1", "lowest_ordinal_mae", "lowest_ece", "lowest_abstention_rate", "lexicographically_smallest_parameters"],
        "bootstrap_method": "paired-identity-component-percentile-v1", "bootstrap_samples": 1000,
        "all_random_seeds": {"split": 42, "cross_validation": 42, "bootstrap": 42},
        "numerical_tolerances": NUMERICAL_TOLERANCES, "code_commit": "fixture", "git_dirty": True,
        "dependency_lock_path": "requirements.txt", "dependency_lock_hash": "lock-hash",
        "environment_fingerprint_hash": "environment-hash", "maximum_validation_executions": 1,
        "validation_executions": 1, "locked_test_policy": "application-level-one-attempt-guard-v1",
        "belief_config_hash": config.config_hash,
    }
    protocol["protocol_hash"] = canonical_json_hash(protocol)
    protocol_file = tmp_path / "calibration_protocol_v4.json"
    protocol_file.write_text(json.dumps(protocol), encoding="utf-8")
    for rows, split in ((train, "train"), (validation, "validation")):
        for row in rows:
            row.update({
                "dataset_split": split, "belief_config_hash": config.config_hash,
                "schema_version": "aria-replay-v4",
                "supported_consumer_versions": ["aria-replay-v4"],
                "protocol_hash": protocol["protocol_hash"], "raw_file_sha256": protocol["raw_file_sha256"],
                "raw_dataset_hash": protocol["raw_dataset_hash"], "split_manifest_hash": manifest["manifest_hash"],
                "environment_fingerprint_hash": protocol["environment_fingerprint_hash"],
            })
    train_file = tmp_path / "train.json"
    validation_file = tmp_path / "validation.json"
    train_file.write_text(json.dumps(train), encoding="utf-8")
    validation_file.write_text(json.dumps(validation), encoding="utf-8")
    bundle = {
        "schema_version": DEVELOPMENT_BUNDLE_VERSION, "producer_version": CALIBRATION_ALGORITHM_VERSION,
        "supported_consumer_versions": [DEVELOPMENT_BUNDLE_VERSION], "protocol_hash": protocol["protocol_hash"],
        "raw_file_sha256": protocol["raw_file_sha256"], "raw_dataset_hash": protocol["raw_dataset_hash"],
        "split_manifest_hash": manifest["manifest_hash"], "belief_config_hash": config.config_hash,
        "environment_fingerprint_hash": protocol["environment_fingerprint_hash"], "parent_artifact_hashes": {},
        "git_commit": "fixture",
        "artifacts": {
            "split_manifest": {"path": str(manifest_file), "sha256": file_sha256(manifest_file)},
            "replayed_train": {"path": str(train_file), "sha256": file_sha256(train_file), "split": "train"},
            "replayed_validation": {"path": str(validation_file), "sha256": file_sha256(validation_file), "split": "validation"},
            "belief_config": {"path": str(config_file), "sha256": file_sha256(config_file)},
        },
    }
    bundle["bundle_hash"] = canonical_json_hash(bundle)
    bundle_file = tmp_path / "development_bundle_v1.json"
    bundle_file.write_text(json.dumps(bundle), encoding="utf-8")
    return config, config_file, train_file, validation_file, bundle_file, protocol_file


def test_dataset_validation_rejects_wrong_schema():
    transition = _transition(0, "train")
    transition["state_schema_version"] = "legacy"
    try:
        validate_replayed_dataset([transition], BeliefModelConfig(), "train")
    except ValueError as error:
        assert "incompatible state schema" in str(error)
    else:
        raise AssertionError("Expected incompatible schema rejection")


def test_dataset_validation_rejects_test_transition_for_training():
    transition = _transition(0, "test")
    with pytest.raises(ValueError, match="wrong split"):
        validate_replayed_dataset([transition], BeliefModelConfig(), "train")


def test_training_requires_protocol_and_development_bundle(tmp_path):
    with pytest.raises(ValueError, match="development bundle manifest and calibration protocol"):
        train_iql_policy(
            train_file=tmp_path / "train.json",
            validation_file=tmp_path / "validation.json",
            belief_config_file=tmp_path / "belief.json",
            output_file=tmp_path / "checkpoint.pth",
            total_epochs=1,
        )


def test_training_saves_versioned_best_checkpoint_without_test_input(tmp_path):
    train = [_transition(index, "train") for index in range(160)]
    validation = [_transition(index, "validation") for index in range(24)]
    config, config_file, train_file, validation_file, bundle_file, protocol_file = _write_training_contract(tmp_path, train, validation)
    checkpoint_file = tmp_path / "checkpoint.pth"

    with patch(
        "modules.module_07_rl.train.audit_raw_evidence",
        return_value={"passes_quality_gates": True},
    ), patch(
        "modules.module_07_rl.train.audit_calibration_validation",
        return_value={"passes_quality_gates": True},
    ):
        result = train_iql_policy(
            train_file=train_file,
            validation_file=validation_file,
            belief_config_file=config_file,
            development_bundle_file=bundle_file,
            calibration_protocol_file=protocol_file,
            output_file=checkpoint_file,
            total_epochs=1,
            batch_size=64,
            seed=7,
        )
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["state_schema_version"] == STATE_SCHEMA_VERSION
    assert checkpoint["belief_config_hash"] == config.config_hash
    assert result["evaluates_learned_policy"] is False
    assert result["epochs_completed"] == 1
    assert result["stopped_early"] is False


def test_training_stops_after_validation_patience(tmp_path):
    config, config_file, train_file, validation_file, bundle_file, protocol_file = _write_training_contract(
        tmp_path,
        [_transition(index, "train") for index in range(160)],
        [_transition(index, "validation") for index in range(24)],
    )

    with patch(
        "modules.module_07_rl.train._validation_objective",
        return_value=3.0,
    ), patch(
        "modules.module_07_rl.train.audit_raw_evidence",
        return_value={"passes_quality_gates": True},
    ), patch(
        "modules.module_07_rl.train.audit_calibration_validation",
        return_value={"passes_quality_gates": True},
    ):
        result = train_iql_policy(
            train_file=train_file,
            validation_file=validation_file,
            belief_config_file=config_file,
            development_bundle_file=bundle_file,
            calibration_protocol_file=protocol_file,
            output_file=tmp_path / "checkpoint.pth",
            total_epochs=20,
            batch_size=160,
            seed=7,
            early_stopping_patience=2,
            early_stopping_min_delta=1e-4,
        )

    assert result["best_epoch"] == 1
    assert result["epochs_completed"] == 3
    assert result["stopped_early"] is True
