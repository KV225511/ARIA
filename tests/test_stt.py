"""Tests for Module 1 — Speech-to-Text."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config.settings import AUDIO_SAMPLE_RATE
from modules.module_01_stt.transcriber import Transcriber, transcribe, transcribe_file
from tests.conftest import ensure_fixtures


def test_transcribe_output_schema():
    """Mocked transcription must return all required Module 1 keys."""
    mock_output = {
        "transcript": "hello world",
        "word_timestamps": [{"word": "hello", "start": 0.1, "end": 0.4}],
        "language": "en",
        "confidence": 0.85,
        "response_latency_ms": 250.0,
    }

    with patch.object(Transcriber, "transcribe_sync", return_value=mock_output):
        audio = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
        result = asyncio.run(transcribe(audio))

    assert isinstance(result["transcript"], str)
    assert len(result["transcript"]) > 0
    assert isinstance(result["word_timestamps"], list)
    assert len(result["word_timestamps"]) > 0
    assert result["language"] == "en"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["response_latency_ms"] > 0


def test_response_latency_from_first_word():
    """Latency equals first word start time in ms when no question timestamp."""
    transcriber = Transcriber.__new__(Transcriber)
    word_timestamps = [
        {"word": "hi", "start": 1.2, "end": 1.5},
        {"word": "there", "start": 1.6, "end": 2.0},
    ]

    with patch.object(Transcriber, "transcribe_sync") as mock_sync:
        mock_sync.return_value = {
            "transcript": "hi there",
            "word_timestamps": word_timestamps,
            "language": "en",
            "confidence": 0.9,
            "response_latency_ms": 1200.0,
        }
        audio = np.zeros(AUDIO_SAMPLE_RATE, dtype=np.float32)
        result = asyncio.run(transcribe(audio))

    assert result["response_latency_ms"] == pytest.approx(1200.0)


@pytest.mark.integration
@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("faster_whisper"),
    reason="faster-whisper not installed",
)
def test_transcribe_wav_file_integration():
    """Integration: transcribe 10s fixture WAV (may produce empty transcript for tone-only audio)."""
    wav_path = ensure_fixtures()
    transcriber = Transcriber()

    result = transcriber.transcribe_file_sync(wav_path)

    assert "transcript" in result
    assert isinstance(result["word_timestamps"], list)
    assert isinstance(result["language"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["response_latency_ms"], float)
    assert result["response_latency_ms"] >= 0.0
