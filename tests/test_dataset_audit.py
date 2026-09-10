from modules.module_07_rl.dataset_audit import (
    audit_calibration_validation,
    audit_learned_policy_evaluation,
    audit_belief_predictions,
    audit_dataset,
    audit_generation_distribution,
    audit_raw_evidence,
)
from modules.module_07_rl.transition_schema import (
    FALLBACK_QUESTION_TEMPLATE_VERSION,
    GENERATOR_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
)
from modules.module_07_rl.generation_policy import (
    BEHAVIOR_POLICY_VERSION,
    PAIR_PLAN_SCHEMA_VERSION,
)
from modules.module_05_ontology.grounding import (
    GROUNDING_POLICY_VERSION,
    GROUNDING_SCHEMA_VERSION,
    ROLE_PROFILE_SCHEMA_VERSION,
    grounding_contract_hash,
)


def _transition(episode, label, prediction, done, reward, model_pair=("candidate", "judge")):
    return {
        "episode_id": episode,
        "true_label": label,
        "aria_label": prediction,
        "done": done,
        "reward": reward,
        "semantic_score": 0.2 + 0.3 * label,
        "skills_covered": 5,
        "evaluation_valid": True,
        "action_idx": label,
        "behavior_policy": "coverage_heuristic",
        "candidate_model": model_pair[0],
        "evaluator_model": model_pair[1],
        "resume_file": f"resume-{episode}",
        "jd_file": f"jd-{episode}",
    }


def test_audit_reports_balanced_terminal_metrics():
    transitions = []
    for label in range(3):
        transitions.extend([
            _transition(f"ep-{label}", label, label, False, -0.1 - label * 0.1),
            _transition(f"ep-{label}", label, label, True, 0.2 + label * 0.1),
        ])

    report = audit_dataset(transitions, min_episodes=3)
    assert report["num_episodes"] == 3
    assert report["terminal_micro_f1"] == 1.0
    assert report["terminal_true_label_counts"] == {0: 1, 1: 1, 2: 1}
    assert report["invalid_evaluations"] == 0
    # This small fixture validates metric aggregation, not the production
    # calibration gate, which intentionally requires six components and ECE.
    assert report["stage_reports"]["raw"]["passes_quality_gates"] is True


def test_validation_and_policy_gates_remain_distinct():
    transitions = []
    for label in range(3):
        transitions.append(_transition(f"ep-{label}", label, label, True, 0.2))
    validation = audit_calibration_validation(transitions)
    assert validation["gate"] == "calibration_validation"
    assert "ordinal_mae" in validation
    assert "confusion_matrix" in validation
    fixed_offline_report = {
        "evaluation_type": "stored_belief_verdict",
        "evaluates_learned_policy": False,
        "num_episodes": 3,
    }
    assert not audit_learned_policy_evaluation(fixed_offline_report)[
        "passes_quality_gates"
    ]
    rollout_report = {
        "evaluation_type": "learned_policy_rollout",
        "fresh_rollouts": True,
        "checkpoint_hash": "abc123",
        "num_episodes": 3,
    }
    assert audit_learned_policy_evaluation(rollout_report)["passes_quality_gates"]


def test_generation_distribution_enforces_ideal_metrics_at_60_episodes():
    transitions = []
    for index in range(60):
        item = {
            "episode_id": f"episode-{index}",
            "done": True,
            "transition_kind": "question",
            "termination_reason": "explicit_conclusion",
            "action_idx": 3 if index < 18 else 0,
            "action_name": "switch_topic" if index < 18 else "increase_difficulty",
            "generator_schema_version": GENERATOR_SCHEMA_VERSION,
            "pairing_record": {
                "pairing_class": (
                    "no_evidence_overlap" if index < 18 else "evidence_overlap"
                )
            },
        }
        transitions.append(item)
    report = audit_generation_distribution(transitions)
    assert report["switch_topic_question_share"] == 0.30
    assert report["no_evidence_overlap_rate"] == 0.30
    assert report["meets_target_metrics"] is True
    assert report["passes_distribution_gates"] is True

def _current_grounded_transition(episode: str, label: int) -> dict:
    item = _transition(episode, label, label, True, 0.1)
    item.update({
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "behavior_policy_version": BEHAVIOR_POLICY_VERSION,
        "pair_plan_schema_version": PAIR_PLAN_SCHEMA_VERSION,
        "role_profile_schema_version": ROLE_PROFILE_SCHEMA_VERSION,
        "question_grounding_schema_version": GROUNDING_SCHEMA_VERSION,
        "grounding_contract_hash": grounding_contract_hash(),
        "transition_kind": "question",
        "resume_content_hash": f"resume-hash-{episode}",
        "jd_content_hash": f"jd-hash-{episode}",
        "question": f"How would you apply Python in scenario {episode}?",
        "question_grounding_valid": True,
        "question_generation_attempts": 1,
        "llm_question_generation_attempts": 1,
        "deterministic_question_generation_attempts": 0,
        "question_generation_mode": "llm",
        "fallback_question_template_version": None,
        "question_prompt_hash": f"prompt-hash-{episode}",
        "question_generation_seed": 42 + label,
        "target_skill_id": "python",
        "role_profile_hash": f"profile-{episode}",
        "ontology_hash": f"ontology-{episode}",
        "pairing_record": {"pairing_class": "evidence_overlap"},
        "question_grounding": {
            "schema_version": GROUNDING_SCHEMA_VERSION,
            "grounding_policy_version": GROUNDING_POLICY_VERSION,
            "target_skill_id": "python",
            "role_profile_hash": f"profile-{episode}",
            "decision": "accept",
            "valid": True,
            "reasons": [],
        },
    })
    return item


def test_audit_flags_prediction_collapse_and_model_overlap():
    transitions = [
        _transition(f"ep-{index}", index % 3, 0, True, -0.1, ("same", "same"))
        for index in range(9)
    ]
    report = audit_dataset(transitions, min_episodes=3)
    assert report["passes_quality_gates"] is False
    assert any("collapse" in warning for warning in report["warnings"])
    assert any("overlap" in warning for warning in report["warnings"])


def test_audit_requires_enough_episodes_for_reliable_evaluation():
    transitions = [
        _transition(f"ep-{index}", index % 3, index % 3, True, 0.1)
        for index in range(9)
    ]
    report = audit_dataset(transitions, min_episodes=10)
    assert report["passes_quality_gates"] is False
    assert any("at least 10" in warning for warning in report["warnings"])


def test_raw_gate_does_not_fail_only_because_stored_beliefs_collapse():
    transitions = [
        _transition(f"ep-{index}", index % 3, 1, True, 0.1)
        for index in range(9)
    ]
    raw = audit_raw_evidence(
        transitions,
        min_episodes=3,
        min_independent_components=3,
        allow_legacy=True,
    )
    belief = audit_belief_predictions(transitions)
    assert raw["passes_quality_gates"] is True
    assert belief["passes_quality_gates"] is False


def test_legacy_transitions_do_not_require_v4_grounding_consistency():
    transitions = [
        _transition(f"legacy-{label}", label, label, True, 0.1)
        for label in range(3)
    ]
    report = audit_raw_evidence(
        transitions,
        min_episodes=3,
        min_independent_components=3,
        allow_legacy=True,
    )
    assert report["inconsistent_episode_grounding"] == 0
    assert report["passes_quality_gates"] is True


def test_raw_gate_rejects_unknown_transition_contract_by_default():
    transitions = [
        _transition(f"unknown-{label}", label, label, True, 0.1)
        for label in range(3)
    ]
    for item in transitions:
        item["transition_schema_version"] = "aria-transition-v999"

    report = audit_raw_evidence(
        transitions, min_episodes=3, min_independent_components=3
    )

    assert report["invalid_contract_provenance"] == 3
    assert report["passes_quality_gates"] is False


def test_raw_gate_reports_and_caps_deterministic_grounding_fallbacks():
    transitions = []
    for label in range(3):
        item = _transition(f"fallback-{label}", label, label, True, 0.1)
        item["question_generation_mode"] = (
            "deterministic_grounded_fallback" if label == 0 else "llm"
        )
        transitions.append(item)
    report = audit_raw_evidence(
        transitions, min_episodes=3, min_independent_components=3
    )
    assert report["deterministic_grounding_fallback_count"] == 1
    assert report["deterministic_grounding_fallback_rate"] == 1 / 3
    assert any("10%" in warning for warning in report["warnings"])


def test_v6_generation_mode_requires_fallback_template_provenance():
    transitions = [
        _current_grounded_transition(f"mode-{label}", label)
        for label in range(3)
    ]
    transitions[0]["question_generation_mode"] = "deterministic_grounded_fallback"

    invalid = audit_raw_evidence(
        transitions, min_episodes=3, min_independent_components=3
    )
    assert invalid["invalid_generation_mode_provenance"] == 1
    assert invalid["passes_quality_gates"] is False

    transitions[0].update({
        "fallback_question_template_version": FALLBACK_QUESTION_TEMPLATE_VERSION,
        "question_prompt_hash": None,
        "question_generation_seed": None,
        "question_generation_attempts": 3,
        "llm_question_generation_attempts": 3,
        "deterministic_question_generation_attempts": 1,
    })
    valid = audit_raw_evidence(
        transitions, min_episodes=3, min_independent_components=3
    )
    assert valid["invalid_generation_mode_provenance"] == 0


def test_current_raw_gate_rejects_stale_contract_provenance():
    transitions = [
        _current_grounded_transition(f"contract-{label}", label)
        for label in range(3)
    ]
    transitions[0]["generator_schema_version"] = "aria-simulator-v5"

    report = audit_raw_evidence(
        transitions, min_episodes=3, min_independent_components=3
    )

    assert report["invalid_contract_provenance"] == 1
    assert report["passes_quality_gates"] is False
    assert any("contracts" in warning for warning in report["warnings"])


def test_raw_gate_reports_excessive_cross_component_fallback_duplicates():
    transitions = [
        _current_grounded_transition(f"duplicate-{label}", label % 3)
        for label in range(3)
    ]
    for item in transitions:
        item.update({
            "question": "How would you apply Python and verify the result?",
            "question_generation_mode": "deterministic_grounded_fallback",
            "fallback_question_template_version": FALLBACK_QUESTION_TEMPLATE_VERSION,
            "question_prompt_hash": None,
            "question_generation_seed": None,
            "question_generation_attempts": 3,
            "llm_question_generation_attempts": 3,
            "deterministic_question_generation_attempts": 1,
        })

    report = audit_raw_evidence(
        transitions, min_episodes=3, min_independent_components=3
    )

    assert report["cross_component_duplicate_fallback_question_count"] == 2
    assert report["cross_component_duplicate_fallback_question_rate"] == 2 / 3
    assert len(report["cross_component_duplicate_fallback_questions"]) == 1
    assert any("25%" in warning for warning in report["warnings"])


def test_content_hash_detects_renamed_duplicate_and_cross_split_leakage():
    transitions = []
    for index, split in enumerate(("train", "validation", "test")):
        item = _transition(f"ep-{index}", index, index, True, 0.1)
        item["dataset_split"] = split
        item["resume_content_hash"] = "same-content" if index < 2 else "other"
        item["resume_file"] = f"renamed-{index}.pdf"
        item["jd_content_hash"] = f"jd-{index}"
        transitions.append(item)
    report = audit_raw_evidence(
        transitions, min_episodes=3, min_independent_components=1
    )
    assert report["split_leaking_resumes"] == ["same-content"]
    assert report["renamed_duplicate_resumes"]


def test_v4_raw_audit_requires_and_reports_grounding_provenance():
    transitions = [
        _current_grounded_transition(f"grounded-{label}", label)
        for label in range(3)
    ]
    for label, item in enumerate(transitions):
        item["question_generation_attempts"] = label + 1
        item["llm_question_generation_attempts"] = label + 1

    report = audit_raw_evidence(
        transitions, min_episodes=3, min_independent_components=3
    )

    assert report["passes_quality_gates"] is True
    assert report["invalid_grounding_provenance"] == 0
    assert report["question_generation_attempt_counts"] == {1: 1, 2: 1, 3: 1}
    assert report["question_grounding_retry_rate"] == 2 / 3
    assert report["pairing_class_counts"] == {"evidence_overlap": 3}
