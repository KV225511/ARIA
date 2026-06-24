"""Tests for Module 2 — Vision."""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import VALID_EMOTION_LABELS
from modules.module_02_vision.emotion import remap_emotion_label
from modules.module_02_vision.face_mesh import AU_KEYS, _estimate_au_activations
from modules.module_02_vision.gaze import _eye_contact_from_gaze
from modules.module_02_vision.vision_processor import (
    FRAME_OUTPUT_KEYS,
    VisionProcessor,
    _empty_turn_summary,
)


def test_emotion_label_remapping():
    assert remap_emotion_label("happy") == "engaged"
    assert remap_emotion_label("fear") == "nervous"
    assert remap_emotion_label("surprise") == "confused"
    assert remap_emotion_label("neutral") == "blank"


def test_au_activations_schema():
    """Synthetic landmarks produce all nine AU keys in 0-1 range."""
    landmarks = np.random.randn(468, 3).astype(np.float32)
    landmarks[:, 0] = np.abs(landmarks[:, 0]) * 100 + 100
    landmarks[:, 1] = np.abs(landmarks[:, 1]) * 100 + 100

    au = _estimate_au_activations(landmarks)
    assert set(au.keys()) == set(AU_KEYS)
    for value in au.values():
        assert 0.0 <= value <= 1.0


def test_eye_contact_score_range():
    assert 0.0 <= _eye_contact_from_gaze(0, 0) <= 1.0
    assert _eye_contact_from_gaze(0, 0) > _eye_contact_from_gaze(45, 30)


def test_empty_turn_summary_schema():
    summary = _empty_turn_summary()
    assert summary["emotion_label"] in VALID_EMOTION_LABELS
    assert set(summary["au_activations"].keys()) == set(AU_KEYS)
    assert "yaw" in summary["gaze_vector"]
    assert "pitch" in summary["gaze_vector"]
    assert summary["blink_rate"] == 0.0


def test_summarize_turn_from_mock_frames():
    processor = VisionProcessor()
    processor.start_turn(0.0)

    mock_frame = {
        "landmarks": np.zeros((468, 3), dtype=np.float32),
        "au_activations": {k: 0.5 for k in AU_KEYS},
        "emotion_label": "engaged",
        "emotion_confidence": 0.8,
        "gaze_vector": {"yaw": 2.0, "pitch": -1.0},
        "eye_contact_score": 0.9,
        "head_pose": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "blink_detected": True,
        "blink_duration_ms": 120.0,
    }
    processor._turn_frames = [mock_frame, mock_frame]

    summary = processor.summarize_turn(turn_duration_ms=1000.0)
    assert summary["emotion_label"] == "engaged"
    assert summary["au_activations"]["AU12"] == pytest.approx(0.5)
    assert summary["eye_contact_score"] == pytest.approx(0.9)
    assert summary["blink_rate"] == pytest.approx(120.0)  # 1 blink in 1/60 min


def test_frame_output_keys_complete():
    assert len(FRAME_OUTPUT_KEYS) == 9


@pytest.mark.integration
def test_process_single_frame_integration():
    """Integration: process a blank frame (may return None if no face)."""
    processor = VisionProcessor()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = processor.process_frame(frame, timestamp_ms=0.0)

    if result is not None:
        assert FRAME_OUTPUT_KEYS.issubset(result.keys())
        assert result["emotion_label"] in VALID_EMOTION_LABELS
        assert "yaw" in result["gaze_vector"]
        assert "pitch" in result["gaze_vector"]
