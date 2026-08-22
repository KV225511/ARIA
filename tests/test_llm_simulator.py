import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from modules.module_07_rl.llm_simulator import (
    build_candidate_system_prompt,
    build_evaluator_prompt,
    build_split_safe_sweep_pairs,
    evaluate_answer,
)
from modules.module_07_rl.dataset_split import split_by_resume_jd_group
from modules.module_08_llm.generator import LLMQuestionGenerator


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


def test_question_prompt_requires_target_skill():
    generator = LLMQuestionGenerator()
    prompt = generator._build_prompt(
        "probe_foundation",
        {"Docker": [0.4, 0.4, 0.2]},
        "Resume",
        [],
        target_skill="Docker",
    )
    assert "Required Target Skill: Docker" in prompt
    assert "question about the Required Target Skill" in prompt
