import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from modules.module_07_rl.llm_simulator import (
    build_candidate_system_prompt,
    build_append_sweep_pairs,
    build_evaluator_prompt,
    build_split_safe_sweep_pairs,
    evaluate_answer,
    _format_duration,
    generate_llm_response,
    _next_episode_index,
    _three_way_counts,
    report_ollama_capacity,
    run_simulation,
    validate_append_provenance,
)
from modules.module_07_rl.ollama_client import BoundedOllamaClient
from modules.module_07_rl.dataset_split import (
    connected_identity_components,
    split_by_resume_jd_group,
)
from modules.module_08_llm.generator import (
    LLMQuestionGenerator,
    normalize_ollama_keep_alive,
)


def _terminal(ep, pair):
    return {
        "episode_id": f"episode_{ep}",
        "resume_file": pair[0],
        "jd_file": pair[1],
        "candidate_model": "qwen2.5:7b",
        "evaluator_model": "gemma3:4b",
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
    prompt = generator._build_prompt(
        "probe_foundation",
        {"Docker": [0.4, 0.4, 0.2]},
        "Resume",
        [],
        target_skill="Docker",
    )
    assert "Required Target Skill: Docker" in prompt
    assert "question about the Required Target Skill" in prompt


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
        }
    ]
    assert _next_episode_index(existing) == 200
    validate_append_provenance(existing, "qwen2.5:7b", "gemma3:4b")
    with pytest.raises(ValueError, match="provenance differs"):
        validate_append_provenance(existing, "qwen2.5:1.5b", "gemma3:4b")
    with pytest.raises(ValueError, match="must remain distinct"):
        validate_append_provenance(existing, "gemma3:4b", "gemma3:4b")


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
            "probe_foundation", {"SQL": [0.3, 0.4, 0.3]}, "resume", []
        ))

    assert result == "What is an index?"
    assert captured["payload"]["keep_alive"] == -1
    assert captured["payload"]["options"]["num_ctx"] == 4096


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
            "done": True,
        }
        for index in range(3)
    ]
    dataset_file.write_text(json.dumps(existing), encoding="utf-8")
    resumes = [Path(f"resume-{index}.pdf") for index in range(6)]
    jds = [Path(f"jd-{index}.pdf") for index in range(6)]

    async def fake_episode(ep, pair, total_eps, semaphore, **kwargs):
        return [{
            "episode_id": f"episode_{ep}",
            "resume_file": pair[0],
            "jd_file": pair[1],
            "candidate_model": "qwen2.5:7b",
            "evaluator_model": "gemma3:4b",
            "done": True,
        }]

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
        ))

    assert len(combined) == 6
    assert combined[:3] == existing
    assert {item["episode_id"] for item in combined[3:]} == {
        "episode_3", "episode_4", "episode_5"
    }
    backups = list((tmp_path / "backups").glob("*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == existing


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
        combined = asyncio.run(run_simulation(
            sweep=True,
            max_episodes=3,
            max_concurrent=2,
            identity_component_targets=(1, 1, 1),
            dataset_file=dataset_file,
            append=True,
            check_ollama_capacity=False,
        ))

    assert len(combined) == 5
    assert json.loads(dataset_file.read_text(encoding="utf-8")) == combined
    assert {item["episode_id"] for item in combined[3:]} == {"episode_3", "episode_5"}


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
        with pytest.raises(RuntimeError, match="No new episodes"):
            asyncio.run(run_simulation(
                sweep=True,
                max_episodes=3,
                max_concurrent=2,
                identity_component_targets=(1, 1, 1),
                dataset_file=dataset_file,
                append=True,
                check_ollama_capacity=False,
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
        ))

    assert maximum_active == 2
