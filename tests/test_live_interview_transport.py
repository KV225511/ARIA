import asyncio
import sys
import types
import uuid

import pytest


LEGACY_TRANSPORT_SKIP = pytest.mark.skip(
    reason="This test injects a legacy in-memory session; WebSocket v2 uses authenticated PostgreSQL state."
)

from fastapi.testclient import TestClient

import app as live_app
from modules.module_05_ontology.graph import SkillOntologyGraph
from modules.module_06_belief.belief_state import BeliefStateUpdater
from modules.module_08_llm.generator import build_grounded_fallback_question
from modules.module_01_stt import transcriber


JD = (
    "Backend Developer role: Python, REST API, SQL, database design, Docker, "
    "Linux, authentication, Kubernetes, CI/CD, and system design. "
    "Build and test reliable services and diagnose production failures."
)


class StreamingGenerator:
    async def generate_question_stream(self, **kwargs):
        question = build_grounded_fallback_question(
            kwargs["action"], kwargs["target_skill"], kwargs["history"]
        )
        middle = len(question) // 2
        yield question[:middle]
        yield question[middle:]


def create_session(monkeypatch):
    ontology = SkillOntologyGraph()
    assert ontology.adapt_to_candidate(JD, "Three years building Python services.")
    session_id = str(uuid.uuid4())
    live_app.sessions[session_id] = {
        "ontology": ontology,
        "belief": BeliefStateUpdater(ontology.get_all_skills()),
        "history": [],
        "resume": "Three years building Python services.",
        "jd": JD,
        "role_profile": ontology.role_profile,
        "current_target": None,
        "role": ontology.inferred_role,
        "experience": ontology.inferred_experience,
        "turn": 0,
    }
    monkeypatch.setattr(live_app, "llm_gen", StreamingGenerator())
    return session_id


def receive_question(websocket):
    start = websocket.receive_json()
    first = websocket.receive_json()
    second = websocket.receive_json()
    final = websocket.receive_json()
    assert start["type"] == "aria_stream_start"
    assert first["type"] == second["type"] == "aria_chunk"
    assert final["type"] == "aria_question"
    assert first["text"] + second["text"] == final["text"]
    return final


@LEGACY_TRANSPORT_SKIP
def test_audio_error_keeps_websocket_open_and_text_can_continue(monkeypatch):
    session_id = create_session(monkeypatch)

    async def fail_audio(_audio):
        raise RuntimeError("decoder unavailable")

    monkeypatch.setattr(live_app, "transcribe_candidate_audio", fail_audio)
    with TestClient(live_app.app) as client:
        with client.websocket_connect(f"/ws/interview/{session_id}") as websocket:
            receive_question(websocket)
            websocket.send_json({"type": "candidate_audio", "audio_base64": "bad"})
            assert websocket.receive_json()["type"] == "audio_error"
            websocket.send_json({"type": "candidate_answer", "text": "A technical answer"})
            receive_question(websocket)


@LEGACY_TRANSPORT_SKIP
def test_empty_browser_recording_reports_cause_and_keeps_session(monkeypatch):
    session_id = create_session(monkeypatch)
    with TestClient(live_app.app) as client:
        with client.websocket_connect(f"/ws/interview/{session_id}") as websocket:
            receive_question(websocket)
            websocket.send_json({"type": "candidate_audio", "audio_base64": ""})
            error = websocket.receive_json()
            assert error["type"] == "audio_error"
            assert "empty" in error["message"].lower()
            websocket.send_json({"type": "candidate_answer", "text": "A technical answer"})
            receive_question(websocket)


@LEGACY_TRANSPORT_SKIP
def test_voice_answer_gets_transcript_and_streamed_followup(monkeypatch):
    session_id = create_session(monkeypatch)

    async def successful_audio(_audio):
        return "A spoken technical answer"

    monkeypatch.setattr(live_app, "transcribe_candidate_audio", successful_audio)
    with TestClient(live_app.app) as client:
        with client.websocket_connect(f"/ws/interview/{session_id}") as websocket:
            receive_question(websocket)
            websocket.send_json({"type": "candidate_audio", "audio_base64": "recording"})
            transcript = websocket.receive_json()
            assert transcript == {
                "type": "transcription_result",
                "text": "A spoken technical answer",
            }
            receive_question(websocket)


def test_empty_recording_is_rejected_without_transcription():
    try:
        asyncio.run(live_app.transcribe_candidate_audio(""))
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("An empty recording must be rejected")


def test_speech_model_uses_cpu_when_cuda_is_unavailable(monkeypatch):
    calls = []

    def unavailable_cuda(_device):
        raise RuntimeError("CUDA driver unavailable")

    class FakeWhisperModel:
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        types.SimpleNamespace(get_supported_compute_types=unavailable_cuda),
    )
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    monkeypatch.setattr(transcriber, "DEVICE", "cuda")
    transcriber.Transcriber()._get_model()
    assert calls[0][1] == {"device": "cpu", "compute_type": "int8"}
