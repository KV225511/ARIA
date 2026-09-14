import json

import pytest

from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_07_rl.calibration_protocol import (
    CALIBRATION_ALGORITHM_VERSION,
    CALIBRATION_CANDIDATE_VALUES,
    CALIBRATION_PROTOCOL_VERSION,
    CALIBRATION_STAGE_SEQUENCE,
    NUMERICAL_TOLERANCES,
    canonical_json_hash,
    file_sha256,
)
from modules.module_07_rl.dataset_audit import CALIBRATION_GATE_THRESHOLDS, VALIDATION_GATE_VERSION
from modules.module_07_rl.locked_test_evaluator import evaluate_locked_test_once
from modules.module_07_rl.metrics import METRICS_SCHEMA_VERSION
from modules.module_07_rl.calibration_protocol_v5 import (
    CALIBRATION_ALGORITHM_VERSION as CALIBRATION_ALGORITHM_VERSION_V5,
    CALIBRATION_CANDIDATE_VALUES as CALIBRATION_CANDIDATE_VALUES_V5,
    CALIBRATION_PROTOCOL_VERSION as CALIBRATION_PROTOCOL_VERSION_V5,
    CALIBRATION_STAGE_SEQUENCE as CALIBRATION_STAGE_SEQUENCE_V5,
    MAX_CALIBRATION_CANDIDATES as MAX_CALIBRATION_CANDIDATES_V5,
)
from modules.module_07_rl.calibration_protocol_v6 import (
    CALIBRATION_ALGORITHM_VERSION as CALIBRATION_ALGORITHM_VERSION_V6,
    CALIBRATION_CANDIDATE_VALUES as CALIBRATION_CANDIDATE_VALUES_V6,
    CALIBRATION_PROTOCOL_VERSION as CALIBRATION_PROTOCOL_VERSION_V6,
    CALIBRATION_STAGE_SEQUENCE as CALIBRATION_STAGE_SEQUENCE_V6,
    MAX_CALIBRATION_CANDIDATES as MAX_CALIBRATION_CANDIDATES_V6,
    PROTOCOL_STATE_VERSION,
    RAW_SPLIT_INVENTORY_VERSION,
    split_assignment_hash,
)


def _locked_fixture(tmp_path):
    transitions = []
    assignments = {}
    for index in range(6):
        episode_id = f"test-{index}"
        assignments[episode_id] = "test"
        label = index % 3
        for turn in range(3):
            transitions.append({
                "episode_id": episode_id, "resume_file": f"resume-{index}",
                "jd_file": f"jd-{index}", "dataset_split": "test",
                "resume_content_hash": f"resume-content-{index}",
                "jd_content_hash": f"jd-content-{index}",
                "true_label": label, "target_skill": f"skill-{turn}",
                "semantic_score": (0.1, 0.5, 0.9)[label],
                "behavior_score": 0.5, "cognitive_load": "low",
                "evaluator_confidence": 1.0, "evaluation_valid": True,
                "done": turn == 2,
            })
    manifest = {
        "schema_version": "aria-split-manifest-v4",
        "raw_dataset_hash": canonical_json_hash(transitions),
        "locked_test_assignment_hash": canonical_json_hash({
            "episode_ids": sorted(assignments),
            "resume_content_hashes": [f"resume-content-{index}" for index in range(6)],
            "jd_content_hashes": [f"jd-content-{index}" for index in range(6)],
        }),
        "assignments": assignments,
    }
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    config = BeliefModelConfig(
        raw_dataset_hash=manifest["raw_dataset_hash"],
        split_manifest_hash=manifest["manifest_hash"],
    )
    config_file = tmp_path / "belief.json"
    config.save(config_file)
    protocol = {
        "protocol_schema_version": CALIBRATION_PROTOCOL_VERSION, "protocol_status": "PROVISIONAL_SYNTHETIC",
        "raw_file_path": "fixture", "raw_file_sha256": "bytes", "raw_dataset_hash": manifest["raw_dataset_hash"],
        "episode_count": 6, "transition_count": 18, "identity_component_count": 6,
        "parent_split_manifest_hash": "parent", "locked_test_assignment_hash": manifest["locked_test_assignment_hash"],
        "target_component_counts": [21, 6, 6], "metric_schema_version": METRICS_SCHEMA_VERSION,
        "gate_policy_version": VALIDATION_GATE_VERSION, "gate_thresholds": CALIBRATION_GATE_THRESHOLDS,
        "split_migration_algorithm": "train-to-validation-component-rebalance-v1",
        "cross_validation_algorithm": "grouped-identity-component-3fold-greedy-v1",
        "calibration_algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "calibration_stage_sequence": CALIBRATION_STAGE_SEQUENCE, "candidate_values": CALIBRATION_CANDIDATE_VALUES,
        "candidate_selection_rule": ["highest_macro_f1", "lowest_ordinal_mae", "lowest_ece", "lowest_abstention_rate", "lexicographically_smallest_parameters"],
        "bootstrap_method": "paired-identity-component-percentile-v1", "bootstrap_samples": 1000,
        "all_random_seeds": {"split": 42, "cross_validation": 42, "bootstrap": 42},
        "numerical_tolerances": NUMERICAL_TOLERANCES, "code_commit": "fixture", "git_dirty": True,
        "dependency_lock_path": "lock", "dependency_lock_hash": "lock-hash",
        "environment_fingerprint_hash": "environment", "maximum_validation_executions": 1,
        "validation_executions": 1, "locked_test_policy": "application-level-one-attempt-guard-v1",
        "belief_config_hash": config.config_hash,
    }
    protocol["protocol_hash"] = canonical_json_hash(protocol)
    protocol_file = tmp_path / "protocol.json"
    protocol_file.write_text(json.dumps(protocol), encoding="utf-8")
    test_file = tmp_path / "locked.json"
    test_file.write_text(json.dumps(transitions), encoding="utf-8")
    return test_file, config_file, protocol_file, manifest_file


def test_locked_evaluator_consumes_one_attempt_and_refuses_repeat(tmp_path):
    inputs = _locked_fixture(tmp_path)
    report = evaluate_locked_test_once(*inputs, tmp_path / "derived")
    assert report["schema_version"] == "aria-locked-test-evaluation-v1"
    attempt = json.loads((tmp_path / "derived" / "release" / "release_attempt_v1.json").read_text())
    assert attempt["state"] == "COMPLETED"
    with pytest.raises(FileExistsError):
        evaluate_locked_test_once(*inputs, tmp_path / "derived")


def test_locked_evaluator_rejects_hash_mismatch_before_consuming_attempt(tmp_path):
    test_file, config_file, protocol_file, manifest_file = _locked_fixture(tmp_path)
    config = BeliefModelConfig.load(config_file).with_updates(split_manifest_hash="wrong")
    config.save(config_file)
    output = tmp_path / "derived"
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_locked_test_once(test_file, config_file, protocol_file, manifest_file, output)
    assert not (output / "release" / "release_attempt_v1.json").exists()


def test_locked_evaluator_accepts_v5_and_still_refuses_repeat(tmp_path):
    test_file, config_file, protocol_file, manifest_file = _locked_fixture(tmp_path)
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    protocol.update({
        "protocol_schema_version": CALIBRATION_PROTOCOL_VERSION_V5,
        "calibration_algorithm_version": CALIBRATION_ALGORITHM_VERSION_V5,
        "calibration_stage_sequence": CALIBRATION_STAGE_SEQUENCE_V5,
        "candidate_values": CALIBRATION_CANDIDATE_VALUES_V5,
        "maximum_calibration_candidates": MAX_CALIBRATION_CANDIDATES_V5,
        "parent_v4_report_hash": "fixture-v4-report",
        "v5_change_scope": "lower-repeat-discount-only",
    })
    protocol.pop("protocol_hash")
    protocol["protocol_hash"] = canonical_json_hash(protocol)
    protocol_file.write_text(json.dumps(protocol), encoding="utf-8")
    output = tmp_path / "derived-v5"
    report = evaluate_locked_test_once(
        test_file, config_file, protocol_file, manifest_file, output,
    )
    assert report["producer_version"] == CALIBRATION_ALGORITHM_VERSION_V5
    with pytest.raises(FileExistsError):
        evaluate_locked_test_once(
            test_file, config_file, protocol_file, manifest_file, output,
        )


def test_v6_locked_evaluator_claims_attempt_before_test_hash_and_fails_state(tmp_path):
    test_file, config_file, protocol_file, manifest_file = _locked_fixture(tmp_path)
    output = (tmp_path / "derived-v6").resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    protocol.update({
        "protocol_schema_version": CALIBRATION_PROTOCOL_VERSION_V6,
        "initial_protocol_status": "FROZEN_FOR_DEVELOPMENT",
        "artifact_root": str(output),
        "split_assignment_hash": split_assignment_hash(manifest),
        "calibration_algorithm_version": CALIBRATION_ALGORITHM_VERSION_V6,
        "calibration_stage_sequence": CALIBRATION_STAGE_SEQUENCE_V6,
        "candidate_values": CALIBRATION_CANDIDATE_VALUES_V6,
        "maximum_calibration_candidates": MAX_CALIBRATION_CANDIDATES_V6,
        "all_random_seeds": {"cross_validation": 42, "bootstrap": 42},
        "fixed_behavior_configuration": {
            "duplicate_question_multiplier": 0.25,
            "posterior_floor": 1e-4,
            "minimum_skill_coverage": 3,
            "minimum_effective_evidence": 2.0,
        },
        "emission_fit_algorithm": "episode_balanced_weighted_median_mad",
        "v6_change_scope": "lower-minimum-effective-evidence-only",
        "parent_v5_protocol_final_hash": "parent-final",
        "parent_v5_protocol_frozen_hash": "parent-frozen",
        "parent_v5_report_hash": "parent-report",
        "parent_v5_split_manifest_hash": "parent-manifest",
        "parent_v5_inventory_hash": "parent-inventory",
    })
    for obsolete in (
        "protocol_status", "validation_executions", "belief_config_hash",
        "parent_split_manifest_hash", "split_migration_algorithm",
    ):
        protocol.pop(obsolete, None)
    protocol.pop("protocol_hash", None)
    protocol["protocol_hash"] = canonical_json_hash(protocol)
    protocol_file.write_text(json.dumps(protocol), encoding="utf-8")
    config = BeliefModelConfig.load(config_file)
    state = {
        "schema_version": PROTOCOL_STATE_VERSION,
        "protocol_hash": protocol["protocol_hash"],
        "current_status": "PROVISIONAL_SYNTHETIC",
        "validation_executions": 1,
        "revision": 2,
        "last_attempt_hash": "validation-attempt",
        "belief_config_hash": config.config_hash,
    }
    state["state_hash"] = canonical_json_hash(state)
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    inventory = {
        "schema_version": RAW_SPLIT_INVENTORY_VERSION,
        "protocol_hash": protocol["protocol_hash"],
        "split_manifest_hash": manifest["manifest_hash"],
        "artifacts": {"locked_test": {"sha256": file_sha256(test_file)}},
    }
    inventory["inventory_hash"] = canonical_json_hash(inventory)
    inventory_file = tmp_path / "inventory.json"
    inventory_file.write_text(json.dumps(inventory), encoding="utf-8")
    test_file.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="locked test file hash mismatch"):
        evaluate_locked_test_once(
            test_file, config_file, protocol_file, manifest_file, output,
            protocol_state=state_file, raw_split_inventory=inventory_file,
        )
    attempt = json.loads((output / "release" / "release_attempt_v1.json").read_text())
    failed_state = json.loads(state_file.read_text())
    assert attempt["state"] == "FAILED"
    assert failed_state["current_status"] == "FAILED"
    with pytest.raises(FileExistsError):
        evaluate_locked_test_once(
            test_file, config_file, protocol_file, manifest_file, output,
            protocol_state=state_file, raw_split_inventory=inventory_file,
        )
