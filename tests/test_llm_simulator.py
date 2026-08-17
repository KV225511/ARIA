import asyncio
from unittest.mock import AsyncMock, patch

from modules.module_07_rl.llm_simulator import evaluate_answer
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
