import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from modules.module_07_rl.llm_simulator import (
    build_candidate_system_prompt,
    build_append_sweep_pairs,
    build_evaluator_prompt,
    build_split_safe_sweep_pairs,
    evaluate_answer,
    _can_use_deterministic_grounding_fallback,
    _format_duration,
    generate_llm_response,
    _next_episode_index,
    _three_way_counts,
    report_ollama_capacity,
    run_simulation,
    simulate_episode,
    validate_append_provenance,
)
from modules.module_07_rl.ollama_client import BoundedOllamaClient
from modules.module_07_rl.data_loader import LoadedDocumentPair, ResumeDocument
from modules.module_07_rl.dataset_split import (
    connected_identity_components,
    split_by_resume_jd_group,
)
from modules.module_05_ontology.grounding import (
    build_role_profile,
    grounding_contract_hash,
    validate_grounded_question,
)
from modules.module_07_rl.transition_schema import (
    FALLBACK_QUESTION_TEMPLATE_VERSION,
    GENERATOR_SCHEMA_VERSION,
)
from modules.module_08_llm.generator import (
    LLMQuestionGenerator,
    build_grounded_fallback_question,
    build_question_retry_correction,
    normalize_ollama_keep_alive,
)


def _terminal(ep, pair):
    return {
        "episode_id": f"episode_{ep}",
        "resume_file": pair[0],
        "jd_file": pair[1],
        "candidate_model": "qwen2.5:7b",
        "evaluator_model": "gemma3:4b",
        "transition_schema_version": "aria-transition-v4",
        "generator_schema_version": "aria-simulator-v6",
        "role_profile_schema_version": "aria-role-profile-v1",
        "question_grounding_schema_version": "aria-question-grounding-v1",
        "grounding_contract_hash": grounding_contract_hash(),
        "done": True,
    }


def test_evaluator_returns_structured_valid_result():
    payload = (
        '{"semantic_score": 0.82, "behavior_score": 0.76, '
        '"cog_load": "low", "confidence": 0.9, '
        '"rubric_evidence": ["Explained the trade-off"]}'
    )
    with patch(
        "modules.module_07_rl.llm_simulator.generate_llm_response",
        new=AsyncMock(return_value=payload),
    ):
        result = asyncio.run(evaluate_answer("Question", "Answer"))

    semantic, behavior, load, confidence, evidence, valid = result
    assert semantic == 0.82
    assert behavior == 0.76
    assert load == "low"
    assert confidence == 0.9
    assert evidence == ["Explained the trade-off"]
    assert valid is True


def test_evaluator_rejects_invalid_output_after_retry():
    mock_call = AsyncMock(side_effect=["not json", "still not json"])
    with patch(
        "modules.module_07_rl.llm_simulator.generate_llm_response",
        new=mock_call,
    ):
        result = asyncio.run(evaluate_answer("Question", "Answer"))

    assert result[-1] is False
    assert mock_call.await_count == 2


def test_evaluator_uses_deterministic_anchored_prompt():
    payload = (
        '{"semantic_score": 0.5, "behavior_score": 0.6, '
        '"cog_load": "low", "confidence": 0.8, "rubric_evidence": []}'
    )
    mock_call = AsyncMock(return_value=payload)
    with patch(
        "modules.module_07_rl.llm_simulator.generate_llm_response",
        new=mock_call,
    ):
        asyncio.run(evaluate_answer("What is an index?", "It speeds up reads."))

    prompt = mock_call.await_args.args[0]
    assert mock_call.await_args.kwargs["temperature"] == 0.0
    assert "Do not cluster answers near 0.50" in prompt
    assert "semantic_score: 0.10" in prompt
    assert "semantic_score: 0.90" in prompt


def test_candidate_personas_have_distinct_depth_and_length_constraints():
    beginner = build_candidate_system_prompt("BEGINNER", "resume")
    mid = build_candidate_system_prompt("MID", "resume")
    expert = build_candidate_system_prompt("EXPERT", "resume")

    assert "one- or two-sentence" in beginner
    assert "novice misconceptions" in beginner
    assert "implementation-level" in mid
    assert "two- or three-sentence" in mid
    assert "four- to six-sentence" in expert
    assert "failure mode or edge case" in expert


def test_evaluator_prompt_does_not_contain_persona_label():
    prompt = build_evaluator_prompt("Question", "Answer")
    assert "assigned skill persona" not in prompt.lower()


def test_sweep_pairs_produce_three_leakage_safe_splits():
    pairs = build_split_safe_sweep_pairs(
        [Path(f"resume-{index}.pdf") for index in range(7)],
        [Path(f"jd-{index}.pdf") for index in range(12)],
        max_episodes=300,
        seed=42,
        component_targets=(1, 1, 1),
    )
    transitions = [
        {
            "episode_id": f"episode-{index}",
            "resume_file": resume,
            "jd_file": jd,
            "done": True,
        }
        for index, (resume, jd) in enumerate(pairs)
    ]
    splits = split_by_resume_jd_group(transitions, seed=42)

    assert {name: len(items) for name, items in splits.items()} == {
        "train": 210,
        "validation": 45,
        "test": 45,
    }
    for field in ("resume_file", "jd_file"):
        identities = {
            name: {item[field] for item in items}
            for name, items in splits.items()
        }
        assert identities["train"].isdisjoint(identities["validation"])
        assert identities["train"].isdisjoint(identities["test"])
        assert identities["validation"].isdisjoint(identities["test"])


def test_production_pair_plan_builds_32_independent_identity_components():
    pairs = build_split_safe_sweep_pairs(
        [Path(f"resume-{index}.pdf") for index in range(40)],
        [Path(f"jd-{index}.pdf") for index in range(40)],
        max_episodes=600,
        seed=42,
    )
    transitions = [
        {
            "episode_id": f"episode-{index}",
            "resume_file": resume,
            "jd_file": jd,
            "done": True,
        }
        for index, (resume, jd) in enumerate(pairs)
    ]
    assert len(pairs) == 600
    assert len(connected_identity_components(transitions)) == 32


@pytest.mark.parametrize("total", range(3, 101))
def test_three_way_counts_are_nonempty_and_exact(total):
    counts = _three_way_counts(total)
    assert sum(counts) == total
    assert all(count >= 1 for count in counts)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("max_episodes", (3, 4, 7, 31, 100))
def test_split_pair_planning_remains_identity_disjoint(seed, max_episodes):
    pairs = build_split_safe_sweep_pairs(
        [Path(f"resume-{index}.pdf") for index in range(7)],
        [Path(f"jd-{index}.pdf") for index in range(10)],
        max_episodes=max_episodes,
        seed=seed,
        component_targets=(1, 1, 1),
    )
    transitions = [
        {
            "episode_id": f"episode_{index}",
            "resume_file": resume,
            "jd_file": jd,
            "done": True,
        }
        for index, (resume, jd) in enumerate(pairs)
    ]
    splits = split_by_resume_jd_group(transitions, seed=42)
    assert sum(map(len, splits.values())) == max_episodes
    for field in ("resume_file", "jd_file"):
        identities = [{item[field] for item in splits[name]} for name in splits]
        assert identities[0].isdisjoint(identities[1])
        assert identities[0].isdisjoint(identities[2])
        assert identities[1].isdisjoint(identities[2])


@pytest.mark.parametrize(
    ("seconds", "formatted"),
    ((0, "0s"), (59, "59s"), (60, "1m 00s"), (3661, "1h 01m")),
)
def test_duration_formatting(seconds, formatted):
    assert _format_duration(seconds) == formatted


def test_question_prompt_requires_target_skill():
    generator = LLMQuestionGenerator()
    history = [
        {"q": f"Prior question {index}?", "a": f"Prior answer {index}."}
        for index in range(8)
    ]
    prompt = generator._build_prompt(
        "probe_foundation",
        {"Docker": [0.4, 0.4, 0.2]},
        "Resume",
        history,
        target_skill="Docker",
    )
    assert "Required Target Skill: Docker" in prompt
    assert "interview turn about the Required Target Skill" in prompt
    assert "Prior question 0?" in prompt
    assert "Prior answer 0." not in prompt


def test_question_retry_correction_names_rejected_output_and_changes_angle():
    rejected = "What is Python?"
    second = build_question_retry_correction(
        ["question duplicates accepted history"], rejected, 2
    )
    third = build_question_retry_correction(
        ["question duplicates accepted history"], rejected, 3
    )
    assert rejected in second
    assert "Retry 2 of 3" in second
    assert "Retry 3 of 3" in third
    assert second != third


def test_grounded_fallback_is_action_specific_and_never_exactly_repeats_history():
    first = build_grounded_fallback_question("switch_topic", "Python", [])
    second = build_grounded_fallback_question(
        "switch_topic", "Python", [{"q": first, "a": "answer"}]
    )
    assert first != second
    assert "Python" in first
    assert "Python" in second
    assert first.endswith("?")
    assert second.endswith("?")
    assert FALLBACK_QUESTION_TEMPLATE_VERSION == "aria-grounded-fallback-v1"


def test_deterministic_grounding_fallback_never_masks_an_api_outage():
    duplicate = "How would you apply Python?"
    assert _can_use_deterministic_grounding_fallback([duplicate] * 3)
    assert not _can_use_deterministic_grounding_fallback(["", "", ""])
    assert not _can_use_deterministic_grounding_fallback([duplicate, "", duplicate])


@pytest.mark.parametrize(
    ("action", "semantic_markers"),
    (
        (
            "increase_difficulty",
            ("difficult", "advanced", "severe", "break down", "conflicting", "experienced"),
        ),
        (
            "decrease_difficulty",
            ("core principles", "junior", "simple", "basic", "beginner"),
        ),
        (
            "ask_follow_up_same_topic",
            ("earlier", "previous"),
        ),
        (
            "switch_topic",
            ("apply", "implementation", "moving to", "challenge", "new task", "compare"),
        ),
        (
            "probe_foundation",
            ("underlying", "assumptions", "internally", "first principles", "core rules", "foundational"),
        ),
        (
            "ask_behavioral",
            ("tell me", "describe a", "give an example"),
        ),
        (
            "ask_situational",
            ("suppose", "imagine"),
        ),
    ),
)
def test_grounded_fallback_variation_preserves_logged_action_semantics(
    action, semantic_markers
):
    questions = {
        build_grounded_fallback_question(
            action,
            "Python",
            [],
            grounding_context={"role_title": "Job Summary"},
            variation_key=f"resume|jd|{turn}|{action}",
        )
        for turn in range(100)
    }

    assert len(questions) == 6
    assert all("job summary" not in question.casefold() for question in questions)
    assert all(
        any(marker in question.casefold() for marker in semantic_markers)
        for question in questions
    )


@pytest.mark.parametrize(
    ("action", "target_skill_id", "target_skill", "aliases"),
    (
        ("decrease_difficulty", "test-validation", "Test and Validation", ("test plan",)),
        ("switch_topic", "pcb-design", "PCB Design", ("pcb", "board layout")),
        ("switch_topic", "design-patterns", "Design Patterns", ("factory pattern",)),
    ),
)
def test_grounded_fallback_recovers_the_observed_v8_targets(
    action, target_skill_id, target_skill, aliases
):
    packet = {
        "target_skill_id": target_skill_id,
        "target_skill": target_skill,
        "target_aliases": list(aliases),
        "target_definition": target_skill,
        "jd_evidence": [target_skill],
        "allowed_skill_ids": [target_skill_id],
        "acronym_resolutions": [],
    }
    questions = {
        build_grounded_fallback_question(
            action,
            target_skill,
            [],
            grounding_context=packet,
            variation_key=f"resume|jd|{turn}|{action}",
        )
        for turn in range(100)
    }

    assert len(questions) == 6
    assert all(
        validate_grounded_question(question, packet, [])["valid"]
        for question in questions
    )


def test_simulate_episode_recovers_after_three_duplicate_llm_questions():
    jd_text = (
        "Role: Software Engineer\nRequired Python programming, REST API "
        "development, SQL relational database work, object-oriented "
        "programming, design patterns, and Linux operations."
    )
    resume_text = (
        "Software engineer experienced with Python, REST API development, "
        "SQL, object-oriented programming, design patterns, and Linux."
    )
    profile = build_role_profile(jd_text, resume_text)
    python_skill = next(skill for skill in profile.skills if skill.skill_id == "python")

    class FakeOntology:
        role_profile = None
        inferred_experience = "Mid-Level"
        successors = {}

        def adapt_to_candidate(self, supplied_jd, supplied_resume):
            assert supplied_jd == jd_text
            assert supplied_resume == resume_text
            self.role_profile = profile
            return True

        def get_all_skills(self):
            return [skill.canonical_name for skill in profile.skills]

        def get_skill_metadata(self, target):
            if target in {python_skill.skill_id, python_skill.canonical_name}:
                return python_skill
            return None

    class FakeBeliefUpdater:
        beliefs = {"Python": np.asarray([0.2, 0.5, 0.3])}

        def get_aggregate_assessment(self):
            return {
                "label": 1,
                "raw_label": 1,
                "belief": np.asarray([0.2, 0.6, 0.2]),
                "confidence": 0.6,
                "visited_skills": ["Python"],
            }

    class FakeEnv:
        def __init__(self, _role):
            self.ontology = FakeOntology()
            self.belief_updater = FakeBeliefUpdater()
            self.action_space = SimpleNamespace(n=8)
            self.turn_id = 0
            self.valid_evidence_count = 0

        def sync_ontology_nodes(self):
            return None

        def reset(self):
            return np.zeros(33, dtype=np.float32), {}

        def get_action_mask(self):
            return np.ones(8, dtype=np.float32)

        def select_target_skill(self, _action_idx):
            return "Python"

        def step_with_scores(self, _action_idx, *_scores, **_kwargs):
            self.turn_id += 1
            self.valid_evidence_count += 1
            return (
                np.zeros(33, dtype=np.float32),
                0.25,
                self.turn_id >= 2,
                False,
                {"termination_reason": "test_complete"},
            )

    accepted_question = "How would you apply Python to solve a small task?"

    class RepeatingQuestionGenerator:
        instance = None

        def __init__(self, **_kwargs):
            self.calls = 0
            RepeatingQuestionGenerator.instance = self

        def _build_prompt(self, *_args, **_kwargs):
            return f"question-prompt-{self.calls}"

        async def generate_question(self, **_kwargs):
            self.calls += 1
            return accepted_question

    loaded_pair = LoadedDocumentPair(
        resume_text=resume_text,
        jd_text=jd_text,
        resume_id="opensporks:test",
        jd_id="test-jd.pdf",
        resume_content_hash="resume-hash",
        jd_content_hash="jd-hash",
        resume_source_type="opensporks_csv",
        resume_category="INFORMATION-TECHNOLOGY",
        resume_source_file_hash="source-hash",
        resume_prompt_hash="resume-prompt-hash",
    )
    probabilities = [0.0] * 8
    probabilities[3] = 1.0
    diagnostics = {}

    with (
        patch("modules.module_07_rl.llm_simulator.ARIAInterviewEnv", FakeEnv),
        patch(
            "modules.module_07_rl.llm_simulator.LLMQuestionGenerator",
            RepeatingQuestionGenerator,
        ),
        patch(
            "modules.module_07_rl.llm_simulator.load_specific_pair",
            return_value=loaded_pair,
        ),
        patch(
            "modules.module_07_rl.llm_simulator.select_behavior_action",
            return_value=(3, "test_policy", probabilities),
        ),
        patch(
            "modules.module_07_rl.llm_simulator.generate_llm_response",
            new=AsyncMock(return_value="A grounded candidate answer."),
        ),
        patch(
            "modules.module_07_rl.llm_simulator.evaluate_answer",
            new=AsyncMock(return_value=(0.7, 0.6, "low", 0.9, ["evidence"], True)),
        ),
    ):
        transitions = asyncio.run(simulate_episode(
            0,
            ("opensporks:test", "test-jd.pdf"),
            1,
            asyncio.Semaphore(1),
            persona_tier="MID",
            failure_diagnostics=diagnostics,
        ))

    assert RepeatingQuestionGenerator.instance.calls == 4
    assert len(transitions) == 2
    assert transitions[0]["question_generation_mode"] == "llm"
    recovered = transitions[1]
    assert recovered["question"] != accepted_question
    assert recovered["question_grounding_valid"] is True
    assert recovered["question_generation_mode"] == "deterministic_grounded_fallback"
    assert recovered["llm_question_generation_attempts"] == 3
    assert recovered["deterministic_question_generation_attempts"] == 1
    assert recovered["fallback_question_template_version"] == (
        FALLBACK_QUESTION_TEMPLATE_VERSION
    )
    assert recovered["question_prompt_hash"] is None
    assert recovered["question_generation_seed"] is None
    assert diagnostics[0]["status"] == "complete"
    assert len(diagnostics[0]["rejected_question_attempts"]) == 3
    assert len(diagnostics[0]["fallback_questions"]) == 1


def test_append_uses_unused_documents_as_new_identity_components():
    original_resumes = [Path(f"resume-{index}.pdf") for index in range(6)]
    original_jds = [Path(f"jd-{index}.pdf") for index in range(9)]
    original_pairs = build_split_safe_sweep_pairs(
        original_resumes, original_jds, max_episodes=30, seed=42,
        component_targets=(1, 1, 1),
    )
    existing = [
        {
            "episode_id": f"episode_{index}",
            "resume_file": resume,
            "jd_file": jd,
                "candidate_model": "qwen2.5:7b",
                "evaluator_model": "gemma3:4b",
                "transition_schema_version": "aria-transition-v4",
                "done": True,
        }
        for index, (resume, jd) in enumerate(original_pairs)
    ]
    all_resumes = original_resumes + [Path(f"new-resume-{index}.pdf") for index in range(3)]
    all_jds = original_jds + [Path(f"new-jd-{index}.pdf") for index in range(3)]
    appended, mode = build_append_sweep_pairs(
        existing, all_resumes, all_jds, max_episodes=30, seed=43,
        component_targets=(1, 1, 1),
    )
    assert mode == "new_identity_components"
    assert len(appended) == 30
    assert all(resume.startswith("new-resume-") for resume, _ in appended)
    assert all(jd.startswith("new-jd-") for _, jd in appended)


def test_append_fallback_does_not_cross_existing_identity_partitions():
    resumes = [Path(f"resume-{index}.pdf") for index in range(6)]
    jds = [Path(f"jd-{index}.pdf") for index in range(9)]
    original_pairs = build_split_safe_sweep_pairs(
        resumes, jds, max_episodes=30, seed=42,
        component_targets=(1, 1, 1),
    )
    existing = [
        {
            "episode_id": f"episode_{index}",
            "resume_file": resume,
            "jd_file": jd,
            "done": True,
        }
        for index, (resume, jd) in enumerate(original_pairs)
    ]
    appended, mode = build_append_sweep_pairs(
        existing, resumes, jds, max_episodes=30, seed=43,
        component_targets=(1, 1, 1),
    )
    assert mode == "existing_identity_partitions"
    combined = existing + [
        {
            "episode_id": f"episode_{30 + index}",
            "resume_file": resume,
            "jd_file": jd,
            "done": True,
        }
        for index, (resume, jd) in enumerate(appended)
    ]
    splits = split_by_resume_jd_group(combined, seed=42)
    for field in ("resume_file", "jd_file"):
        identities = {
            name: {item[field] for item in items}
            for name, items in splits.items()
        }
        assert identities["train"].isdisjoint(identities["validation"])
        assert identities["train"].isdisjoint(identities["test"])
        assert identities["validation"].isdisjoint(identities["test"])


def test_append_provenance_and_episode_ids_are_protected():
    existing = [
        {
            "episode_id": "episode_199",
            "candidate_model": "qwen2.5:7b",
            "evaluator_model": "gemma3:4b",
            "transition_schema_version": "aria-transition-v4",
            "generator_schema_version": "aria-simulator-v6",
            "role_profile_schema_version": "aria-role-profile-v1",
            "question_grounding_schema_version": "aria-question-grounding-v1",
            "grounding_contract_hash": grounding_contract_hash(),
        }
    ]
    assert _next_episode_index(existing) == 200
    validate_append_provenance(existing, "qwen2.5:7b", "gemma3:4b")
    with pytest.raises(ValueError, match="provenance differs"):
        validate_append_provenance(existing, "qwen2.5:1.5b", "gemma3:4b")
    with pytest.raises(ValueError, match="must remain distinct"):
        validate_append_provenance(existing, "gemma3:4b", "gemma3:4b")


def test_append_rejects_legacy_transition_schema():
    existing = [{
        "candidate_model": "qwen2.5:7b",
        "evaluator_model": "gemma3:4b",
        "transition_schema_version": "aria-transition-v3",
    }]
    with pytest.raises(ValueError, match="transition schema is incompatible"):
        validate_append_provenance(existing, "qwen2.5:7b", "gemma3:4b")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("generator_schema_version", "aria-simulator-v5"),
        ("role_profile_schema_version", None),
        ("question_grounding_schema_version", "aria-question-grounding-old"),
        ("grounding_contract_hash", "wrong-contract-hash"),
    ),
)
def test_append_rejects_incompatible_generation_contract(field, value):
    existing = [_terminal(0, ("resume.pdf", "jd.pdf"))]
    existing[0][field] = value

    with pytest.raises(ValueError, match=field):
        validate_append_provenance(existing, "qwen2.5:7b", "gemma3:4b")


def test_append_rejects_invalid_question_generation_provenance():
    existing = [_terminal(0, ("resume.pdf", "jd.pdf"))]
    existing[0].update({
        "transition_kind": "question",
        "action_idx": 3,
        "question_generation_mode": "deterministic_grounded_fallback",
        "fallback_question_template_version": "wrong-template-version",
    })

    with pytest.raises(ValueError, match="question-generation provenance"):
        validate_append_provenance(existing, "qwen2.5:7b", "gemma3:4b")


def test_question_generator_uses_memory_bounded_ollama_settings(monkeypatch):
    monkeypatch.setenv("ARIA_OLLAMA_KEEP_ALIVE", "-1")
    monkeypatch.setenv("ARIA_OLLAMA_NUM_CTX", "4096")
    generator = LLMQuestionGenerator(model="qwen2.5:7b")
    assert generator.keep_alive == -1
    assert generator.num_ctx == 4096


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("-1", -1), ("0", 0), ("3600", 3600), ("5m", "5m"), ("-1m", "-1m")),
)
def test_keep_alive_environment_values_are_json_type_safe(raw, expected):
    assert normalize_ollama_keep_alive(raw) == expected


def test_direct_ollama_request_sends_context_and_keep_alive_settings():
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "answer"}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json, timeout):
            captured.update({"url": url, "payload": json, "timeout": timeout})
            return Response()

    with patch("modules.module_07_rl.llm_simulator.httpx.AsyncClient", return_value=Client()):
        result = asyncio.run(generate_llm_response("prompt", "qwen2.5:7b"))

    assert result == "answer"
    assert captured["payload"]["keep_alive"] == -1
    assert captured["payload"]["options"]["num_ctx"] == 4096
    assert captured["payload"]["stream"] is False


def test_shared_ollama_client_enforces_per_model_request_limit():
    active = 0
    maximum_active = 0

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "ok"}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, *args, **kwargs):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.005)
            active -= 1
            return Response()

        async def aclose(self):
            return None

    async def exercise():
        with patch("modules.module_07_rl.ollama_client.httpx.AsyncClient", Client):
            client = BoundedOllamaClient(
                "http://localhost:11434", {"candidate": 2}
            )
            await asyncio.gather(*(
                client.generate({"model": "candidate", "prompt": str(index)})
                for index in range(8)
            ))
            await client.aclose()

    asyncio.run(exercise())
    assert maximum_active == 2


def test_shared_ollama_client_never_overlaps_distinct_model_phases():
    active_models = {}
    overlap_detected = False

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "ok"}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, *args, **kwargs):
            nonlocal overlap_detected
            model = kwargs["json"]["model"]
            active_models[model] = active_models.get(model, 0) + 1
            overlap_detected = overlap_detected or len(active_models) > 1
            await asyncio.sleep(0.005)
            active_models[model] -= 1
            if active_models[model] == 0:
                del active_models[model]
            return Response()

        async def aclose(self):
            return None

    async def exercise():
        with patch("modules.module_07_rl.ollama_client.httpx.AsyncClient", Client):
            client = BoundedOllamaClient(
                "http://localhost:11434", {"candidate": 3, "evaluator": 2}
            )
            await asyncio.gather(*(
                client.generate({"model": model, "prompt": str(index)})
                for index, model in enumerate(
                    ["candidate", "evaluator", "candidate", "evaluator"] * 2
                )
            ))
            await client.aclose()

    asyncio.run(exercise())
    assert overlap_detected is False


def test_question_generator_request_sends_context_and_keep_alive_settings():
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "What is an index?"}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json, timeout):
            captured.update({"url": url, "payload": json, "timeout": timeout})
            return Response()

    generator = LLMQuestionGenerator(
        model="qwen2.5:7b", keep_alive="-1", num_ctx=4096
    )
    with patch("modules.module_08_llm.generator.httpx.AsyncClient", return_value=Client()):
        result = asyncio.run(generator.generate_question(
            "probe_foundation", {"SQL": [0.3, 0.4, 0.3]}, "resume", [],
            temperature=0.6, generation_seed=12345,
        ))

    assert result == "What is an index?"
    assert captured["payload"]["keep_alive"] == -1
    assert captured["payload"]["options"]["num_ctx"] == 4096
    assert captured["payload"]["options"]["temperature"] == 0.6
    assert captured["payload"]["options"]["seed"] == 12345


def test_synthetic_question_generation_disables_fallbacks():
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            raise RuntimeError("Ollama unavailable")

    generator = LLMQuestionGenerator(
        model="qwen2.5:7b",
        allow_fallback=False,
    )
    with patch("modules.module_08_llm.generator.httpx.AsyncClient", return_value=Client()):
        result = asyncio.run(generator.generate_question(
            "probe_foundation", {"SQL": [0.3, 0.4, 0.3]}, "resume", []
        ))

    assert result == ""


def test_capacity_report_rejects_dual_residency_for_eight_gb_weight_total():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            if url.endswith("/api/tags"):
                return Response({"models": [
                    {"name": "qwen2.5:7b", "size": 4_700_000_000},
                    {"name": "gemma3:4b", "size": 3_300_000_000},
                ]})
            return Response({"models": []})

    with patch("modules.module_07_rl.llm_simulator.httpx.AsyncClient", Client):
        report = asyncio.run(report_ollama_capacity(8.0))

    assert report["combined_weight_files_gb"] == 8.0
    assert report["conservative_model_budget_gb"] == 6.75
    assert report["full_dual_gpu_residency_feasible"] is False


def test_append_run_preserves_existing_data_and_checkpoints_new_episodes(tmp_path):
    dataset_file = tmp_path / "qwen_rl_dataset.json"
    existing = [
        {
            "episode_id": f"episode_{index}",
            "resume_file": f"resume-{index}.pdf",
            "jd_file": f"jd-{index}.pdf",
            "candidate_model": "qwen2.5:7b",
            "evaluator_model": "gemma3:4b",
            "transition_schema_version": "aria-transition-v4",
            "generator_schema_version": "aria-simulator-v6",
            "role_profile_schema_version": "aria-role-profile-v1",
            "question_grounding_schema_version": "aria-question-grounding-v1",
            "grounding_contract_hash": grounding_contract_hash(),
            "done": True,
        }
        for index in range(3)
    ]
    dataset_file.write_text(json.dumps(existing), encoding="utf-8")
    resumes = [Path(f"resume-{index}.pdf") for index in range(6)]
    jds = [Path(f"jd-{index}.pdf") for index in range(6)]

    async def fake_episode(ep, pair, total_eps, semaphore, **kwargs):
        transition = {
            "episode_id": f"episode_{ep}",
            "resume_file": pair[0],
            "jd_file": pair[1],
            "candidate_model": "qwen2.5:7b",
            "evaluator_model": "gemma3:4b",
            "done": True,
        }
        if kwargs["display_number"] == 1:
            transition["question_generation_mode"] = (
                "deterministic_grounded_fallback"
            )
            kwargs["failure_diagnostics"][ep] = {
                "episode_id": f"episode_{ep}",
                "resume_file": pair[0],
                "jd_file": pair[1],
                "status": "complete",
                "rejected_question_attempts": [{"retry_number": 3}],
                "fallback_questions": [{
                    "template_version": FALLBACK_QUESTION_TEMPLATE_VERSION,
                    "question": "fallback question",
                }],
            }
        return [transition]

    def fake_get_all_pdfs(directory):
        return resumes if "resume" in str(directory).lower() else jds

    with (
        patch(
            "modules.module_07_rl.llm_simulator.get_all_pdfs",
            side_effect=fake_get_all_pdfs,
        ),
        patch("modules.module_07_rl.llm_simulator.is_valid_resume", return_value=True),
        patch("modules.module_07_rl.llm_simulator.is_valid_jd", return_value=True),
        patch("modules.module_07_rl.llm_simulator.simulate_episode", side_effect=fake_episode),
    ):
        combined = asyncio.run(run_simulation(
            sweep=True,
            max_episodes=3,
            max_concurrent=2,
            identity_component_targets=(1, 1, 1),
            seed=43,
            dataset_file=dataset_file,
            append=True,
            check_ollama_capacity=False,
            resume_source="pdf",
        ))

    assert len(combined) == 6
    assert combined[:3] == existing
    assert {item["episode_id"] for item in combined[3:]} == {
        "episode_3", "episode_4", "episode_5"
    }
    backups = list((tmp_path / "backups").glob("*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == existing
    manifest_path = next((tmp_path / "manifests").glob("*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["new_deterministic_grounding_fallback_count"] == 1
    assert manifest["combined_deterministic_grounding_fallback_count"] == 1
    assert len(manifest["successful_fallback_episode_diagnostics"]) == 1
    assert manifest["canonical_dataset_hash"] == hashlib.sha256(
        dataset_file.read_bytes()
    ).hexdigest()
    assert manifest["partial_dataset_hash"] is None


def test_plain_sweep_refuses_to_overwrite_existing_dataset(tmp_path):
    dataset_file = tmp_path / "dataset.json"
    original = [_terminal(0, ("resume-0.pdf", "jd-0.pdf"))]
    original_bytes = json.dumps(original).encode("utf-8")
    dataset_file.write_bytes(original_bytes)

    with pytest.raises(ValueError, match="Refusing to erase"):
        asyncio.run(run_simulation(
            sweep=True,
            max_episodes=3,
            identity_component_targets=(1, 1, 1),
            dataset_file=dataset_file,
            check_ollama_capacity=False,
            resume_source="pdf",
        ))

    assert dataset_file.read_bytes() == original_bytes


def test_episode_exception_is_isolated_and_other_results_are_checkpointed(tmp_path):
    dataset_file = tmp_path / "dataset.json"
    existing = [_terminal(index, (f"resume-{index}.pdf", f"jd-{index}.pdf"))
                for index in range(3)]
    dataset_file.write_text(json.dumps(existing), encoding="utf-8")
    resumes = [Path(f"resume-{index}.pdf") for index in range(6)]
    jds = [Path(f"jd-{index}.pdf") for index in range(6)]

    async def fake_episode(ep, pair, total_eps, semaphore, **kwargs):
        if kwargs["display_number"] == 2:
            kwargs["failure_diagnostics"][ep] = {
                "episode_id": f"episode_{ep}",
                "resume_file": pair[0],
                "jd_file": pair[1],
                "status": "failed",
                "failure_stage": "question_grounding",
                "failure_reason": "question grounding failed after 3 attempts",
                "role_profile_hash": "profile-hash",
                "role_profile": {"schema_version": "aria-role-profile-v1"},
                "pairing_record": {
                    "schema_version": "aria-pairing-record-v1",
                    "pairing_class": "evidence_overlap",
                },
                "rejected_question_attempts": [{
                    "turn": 2,
                    "action": "switch_topic",
                    "target_skill_id": "python",
                    "retry_number": 3,
                    "raw_generated_question": "Question: invalid output",
                    "normalized_question": "invalid output",
                    "validation_reasons": [
                        "question must be a non-empty interrogative"
                    ],
                }],
            }
            raise ConnectionError("temporary Ollama disconnect")
        await asyncio.sleep(0.001 if kwargs["display_number"] == 1 else 0)
        return [_terminal(ep, pair)]

    def fake_get_all_pdfs(directory):
        return resumes if "resume" in str(directory).lower() else jds

    with (
        patch("modules.module_07_rl.llm_simulator.get_all_pdfs", side_effect=fake_get_all_pdfs),
        patch("modules.module_07_rl.llm_simulator.is_valid_resume", return_value=True),
        patch("modules.module_07_rl.llm_simulator.is_valid_jd", return_value=True),
        patch("modules.module_07_rl.llm_simulator.simulate_episode", side_effect=fake_episode),
    ):
        with pytest.raises(RuntimeError, match="partial output is not eligible"):
            asyncio.run(run_simulation(
                sweep=True,
                max_episodes=3,
                max_concurrent=2,
                identity_component_targets=(1, 1, 1),
                dataset_file=dataset_file,
                append=True,
                check_ollama_capacity=False,
                resume_source="pdf",
            ))

    canonical = json.loads(dataset_file.read_text(encoding="utf-8"))
    assert canonical == existing
    partial_path = next((tmp_path / "failed_runs").glob("*.partial.json"))
    checkpoint = json.loads(partial_path.read_text(encoding="utf-8"))
    assert len(checkpoint) == 5
    assert {item["episode_id"] for item in checkpoint[3:]} == {"episode_3", "episode_5"}
    manifest_path = next((tmp_path / "manifests").glob("*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failed_diagnostic = manifest["failed_episode_diagnostics"]["1"]
    assert failed_diagnostic["role_profile_hash"] == "profile-hash"
    assert failed_diagnostic["rejected_question_attempts"][0]["retry_number"] == 3
    assert manifest["pairing_records"][0]["episode_status"] == "failed"
    assert manifest["combined_transition_count"] == len(existing)
    assert manifest["partial_combined_transition_count"] == len(checkpoint)
    assert manifest["canonical_dataset_hash"] == hashlib.sha256(
        dataset_file.read_bytes()
    ).hexdigest()
    assert manifest["partial_dataset_hash"] == hashlib.sha256(
        partial_path.read_bytes()
    ).hexdigest()


def test_all_episode_failures_preserve_original_bytes_and_raise(tmp_path):
    dataset_file = tmp_path / "dataset.json"
    existing = [_terminal(index, (f"resume-{index}.pdf", f"jd-{index}.pdf"))
                for index in range(3)]
    original_bytes = json.dumps(existing).encode("utf-8")
    dataset_file.write_bytes(original_bytes)
    resumes = [Path(f"resume-{index}.pdf") for index in range(6)]
    jds = [Path(f"jd-{index}.pdf") for index in range(6)]

    async def fail_episode(*args, **kwargs):
        raise TimeoutError("Ollama timed out")

    def fake_get_all_pdfs(directory):
        return resumes if "resume" in str(directory).lower() else jds

    with (
        patch("modules.module_07_rl.llm_simulator.get_all_pdfs", side_effect=fake_get_all_pdfs),
        patch("modules.module_07_rl.llm_simulator.is_valid_resume", return_value=True),
        patch("modules.module_07_rl.llm_simulator.is_valid_jd", return_value=True),
        patch("modules.module_07_rl.llm_simulator.simulate_episode", side_effect=fail_episode),
    ):
        with pytest.raises(RuntimeError, match="partial output is not eligible"):
            asyncio.run(run_simulation(
                sweep=True,
                max_episodes=3,
                max_concurrent=2,
                identity_component_targets=(1, 1, 1),
                dataset_file=dataset_file,
                append=True,
                check_ollama_capacity=False,
                resume_source="pdf",
            ))

    assert dataset_file.read_bytes() == original_bytes


def test_run_never_exceeds_requested_episode_concurrency(tmp_path):
    dataset_file = tmp_path / "dataset.json"
    resumes = [Path(f"resume-{index}.pdf") for index in range(6)]
    jds = [Path(f"jd-{index}.pdf") for index in range(6)]
    active = 0
    maximum_active = 0

    async def fake_episode(ep, pair, total_eps, semaphore, **kwargs):
        nonlocal active, maximum_active
        async with semaphore:
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.005)
            active -= 1
        return [_terminal(ep, pair)]

    def fake_get_all_pdfs(directory):
        return resumes if "resume" in str(directory).lower() else jds

    with (
        patch("modules.module_07_rl.llm_simulator.get_all_pdfs", side_effect=fake_get_all_pdfs),
        patch("modules.module_07_rl.llm_simulator.is_valid_resume", return_value=True),
        patch("modules.module_07_rl.llm_simulator.is_valid_jd", return_value=True),
        patch("modules.module_07_rl.llm_simulator.simulate_episode", side_effect=fake_episode),
    ):
        asyncio.run(run_simulation(
            sweep=True,
            max_episodes=6,
            max_concurrent=2,
            identity_component_targets=(1, 1, 1),
            dataset_file=dataset_file,
            check_ollama_capacity=False,
            resume_source="pdf",
        ))

    assert maximum_active == 2


def test_csv_sweep_records_resume_source_manifest(tmp_path):
    dataset_file = tmp_path / "dataset.json"
    resumes = [
        ResumeDocument(
            source_id=f"opensporks:{index}",
            prompt_text="technical resume\n--- END OF DOCUMENT ---",
            content_hash=f"resume-hash-{index}",
            category="ENGINEERING",
            source_type="opensporks_csv",
            source_file_hash="csv-hash",
        )
        for index in range(3)
    ]
    jds = [Path(f"jd-{index}.pdf") for index in range(3)]

    async def fake_episode(ep, pair, total_eps, semaphore, **kwargs):
        assert kwargs["resume_source"] == "csv"
        assert kwargs["resume_categories"] == ("ENGINEERING",)
        return [_terminal(ep, pair)]

    with (
        patch(
            "modules.module_07_rl.llm_simulator.get_groundable_jd_documents",
            return_value=(jds, {
                "pdf_files": 3,
                "filename_excluded_files": [],
                "unreadable_files": [],
                "duplicate_content_groups": [],
                "unique_readable_content_hashes": 3,
            }),
        ),
        patch("modules.module_07_rl.llm_simulator.get_resume_documents", return_value=resumes),
        patch(
            "modules.module_07_rl.llm_simulator.get_resume_source_manifest",
            return_value={
                "source_type": "opensporks_csv",
                "source_file_hash": "csv-hash",
                "selected_categories": ["ENGINEERING"],
                "selected_resume_count": 3,
                "unique_content_hashes": 3,
            },
        ),
        patch("modules.module_07_rl.llm_simulator.simulate_episode", side_effect=fake_episode),
    ):
        combined = asyncio.run(run_simulation(
            sweep=True,
            max_episodes=3,
            max_concurrent=2,
            identity_component_targets=(1, 1, 1),
            dataset_file=dataset_file,
            replace_existing=True,
            check_ollama_capacity=False,
            resume_source="csv",
            resume_csv_path=tmp_path / "clean.csv",
            resume_categories=("ENGINEERING",),
        ))

    assert len(combined) == 3
    manifest_path = next((tmp_path / "manifests").glob("*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["resume_source"]["source_type"] == "opensporks_csv"
    assert manifest["resume_source"]["source_file_hash"] == "csv-hash"
    assert all(
        item["resume_file"].startswith("opensporks:")
        for item in manifest["planned_document_pairs"]
    )


def test_csv_append_requires_matching_source_hash():
    existing = [{
        "candidate_model": "qwen2.5:7b",
        "evaluator_model": "gemma3:4b",
        "transition_schema_version": "aria-transition-v4",
        "generator_schema_version": "aria-simulator-v6",
        "role_profile_schema_version": "aria-role-profile-v1",
        "question_grounding_schema_version": "aria-question-grounding-v1",
        "grounding_contract_hash": grounding_contract_hash(),
        "resume_source_type": "opensporks_csv",
        "resume_source_file_hash": "old-hash",
    }]

    with pytest.raises(ValueError, match="CSV hash differs"):
        validate_append_provenance(
            existing,
            "qwen2.5:7b",
            "gemma3:4b",
            resume_source_type="opensporks_csv",
            resume_source_file_hash="new-hash",
        )


def test_csv_append_rejects_missing_source_provenance():
    existing = [{
        "candidate_model": "qwen2.5:7b",
        "evaluator_model": "gemma3:4b",
        "transition_schema_version": "aria-transition-v4",
        "generator_schema_version": "aria-simulator-v6",
        "role_profile_schema_version": "aria-role-profile-v1",
        "question_grounding_schema_version": "aria-question-grounding-v1",
        "grounding_contract_hash": grounding_contract_hash(),
    }]

    with pytest.raises(ValueError, match="lack cleaned-CSV source provenance"):
        validate_append_provenance(
            existing,
            "qwen2.5:7b",
            "gemma3:4b",
            resume_source_type="opensporks_csv",
            resume_source_file_hash="new-hash",
        )
