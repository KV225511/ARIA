"""Tests for Module 3 — Prosody (extractor, baseline, pipeline)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.settings import AUDIO_SAMPLE_RATE, MFCC_COEFFICIENTS
from modules.module_03_prosody.baseline import ProsodyBaselineManager
from modules.module_03_prosody.extractor import ProsodyExtractor
from modules.module_03_prosody.pipeline import process_prosody_turn
from tests.conftest import ensure_prosody_fixture

RAW_PROSODY_KEYS = frozenset(
    {
        "pitch_mean",
        "pitch_variance",
        "pitch_range",
        "speech_rate",
        "pause_count",
        "pause_total_duration_ms",
        "disfluency_count",
        "disfluency_timestamps",
        "response_latency_ms",
        "energy_mean",
        "jitter",
        "shimmer",
        "mfcc_vector",
        "speech_to_silence_ratio",
    }
)

DEVIATION_KEYS = frozenset(
    {"pitch_deviation", "rate_deviation", "energy_deviation"}
)

SAMPLE_WORD_TIMESTAMPS = [
    {"word": "I", "start": 0.5, "end": 0.6},
    {"word": "think", "start": 0.65, "end": 0.9},
    {"word": "um", "start": 1.0, "end": 1.2},
    {"word": "machine", "start": 1.3, "end": 1.7},
    {"word": "learning", "start": 1.75, "end": 2.2},
]


def _make_extractor_with_mocks() -> ProsodyExtractor:
    """ProsodyExtractor with mocked openSMILE to avoid heavy model init in unit tests."""
    extractor = ProsodyExtractor()

    combined_lld = pd.DataFrame(
        {
            "F0semitoneFrom27.5Hz_sma3nz": [40.0, 41.0, 42.0],
            "Loudness_sma3": [0.5, 0.6, 0.55],
        }
    )
    func_features = {
        "jitterLocal_sma3nz_amean": 0.01,
        "shimmerLocaldB_sma3nz_amean": 0.02,
    }
    for i in range(1, MFCC_COEFFICIENTS + 1):
        func_features[f"mfcc{i}_sma3_amean"] = float(i) * 0.1

    func_df = pd.DataFrame([func_features])
    extractor.smile_lld.process_signal = MagicMock(return_value=combined_lld)
    extractor.smile_functionals.process_signal = MagicMock(return_value=func_df)
    return extractor


@pytest.fixture
def sample_audio() -> np.ndarray:
    duration_s = 2.0
    t = np.linspace(0, duration_s, int(AUDIO_SAMPLE_RATE * duration_s), dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def speech_intervals() -> np.ndarray:
    return np.array(
        [
            [0, 8000],
            [15000, 20000],   # gap 7000 samples = 437.5 ms (> 250 ms)
            [28000, 32000],   # gap 8000 samples = 500 ms (> 250 ms)
        ],
        dtype=int,
    )


class TestProsodyBaselineManager:
    def test_baseline_turns_set_deviations_to_none(self):
        manager = ProsodyBaselineManager(baseline_turns=2)
        features = {
            "pitch_mean": 120.0,
            "speech_rate": 3.5,
            "energy_mean": 0.4,
        }

        for turn_id in (1, 2):
            result = manager.update_with_baseline("c1", turn_id, dict(features))
            assert result["pitch_deviation"] is None
            assert result["rate_deviation"] is None
            assert result["energy_deviation"] is None

    def test_turn_three_computes_deviations(self):
        manager = ProsodyBaselineManager(baseline_turns=2)
        baseline_features = {
            "pitch_mean": 100.0,
            "speech_rate": 2.0,
            "energy_mean": 0.5,
        }

        manager.update_with_baseline("c1", 1, dict(baseline_features))
        manager.update_with_baseline("c1", 2, dict(baseline_features))

        current = {
            "pitch_mean": 120.0,
            "speech_rate": 2.5,
            "energy_mean": 0.6,
        }
        result = manager.update_with_baseline("c1", 3, dict(current))

        assert result["pitch_deviation"] == pytest.approx(0.2)
        assert result["rate_deviation"] == pytest.approx(0.25)
        assert result["energy_deviation"] == pytest.approx(0.2)

    def test_safe_deviation_zero_baseline_returns_zero(self):
        manager = ProsodyBaselineManager()
        assert manager._safe_deviation(10.0, 0.0) == 0.0


class TestProsodyExtractorHelpers:
    def test_validate_audio_rejects_empty(self):
        extractor = ProsodyExtractor()
        with pytest.raises(ValueError, match="empty"):
            extractor._validate_audio(np.array([], dtype=np.float32))

    def test_validate_audio_rejects_multidimensional(self):
        extractor = ProsodyExtractor()
        with pytest.raises(ValueError, match="1D"):
            extractor._validate_audio(np.zeros((2, 100), dtype=np.float32))

    def test_compute_pause_features_counts_gaps_over_250ms(self, speech_intervals):
        extractor = ProsodyExtractor()
        result = extractor._compute_pause_features(speech_intervals, AUDIO_SAMPLE_RATE)

        assert result["pause_count"] == 2
        assert result["pause_total_duration_ms"] > 0.0

    def test_compute_speech_to_silence_ratio(self, speech_intervals):
        extractor = ProsodyExtractor()
        total_duration = 2.0
        ratio = extractor._compute_speech_to_silence_ratio(
            speech_intervals, total_duration, AUDIO_SAMPLE_RATE
        )
        assert ratio > 0.0

    def test_compute_speech_rate_positive(self):
        extractor = ProsodyExtractor()
        rate = extractor._compute_speech_rate(SAMPLE_WORD_TIMESTAMPS)
        assert rate > 0.0

    def test_compute_disfluencies_detects_fillers(self):
        extractor = ProsodyExtractor()
        result = extractor._compute_disfluencies(SAMPLE_WORD_TIMESTAMPS)

        assert result["disfluency_count"] == 1
        assert result["disfluency_timestamps"] == [pytest.approx(1.0)]

    def test_estimate_syllables(self):
        extractor = ProsodyExtractor()
        assert extractor._estimate_syllables("hello") >= 1
        assert extractor._estimate_syllables("machine") >= 2

    def test_normalize_response_latency(self):
        extractor = ProsodyExtractor()
        assert extractor._normalize_response_latency(850.0) == 850.0
        assert extractor._normalize_response_latency(None) == 0.0
        assert extractor._normalize_response_latency(-5.0) == 0.0
        assert extractor._normalize_response_latency("bad") == 0.0


    def test_pitch_features_converted_to_hz(self):
        extractor = ProsodyExtractor()
        lld_df = pd.DataFrame({"F0semitoneFrom27.5Hz_sma3nz": [12.0, 24.0]})

        result = extractor._compute_pitch_features(lld_df)

        # 12 semitones → 55.0 Hz, 24 semitones → 110.0 Hz
        assert result["pitch_mean"] == pytest.approx(82.5)
        assert result["pitch_range"] == pytest.approx(55.0)

    def test_compute_mfcc_returns_nonzero_librosa_coefficients(self, sample_audio):
        extractor = ProsodyExtractor()
        mfcc = extractor._compute_mfcc(sample_audio, AUDIO_SAMPLE_RATE)

        assert len(mfcc) == MFCC_COEFFICIENTS
        assert not all(v == 0.0 for v in mfcc)


class TestProsodyExtractorExtract:
    def test_extract_returns_all_raw_keys(self, sample_audio):
        extractor = _make_extractor_with_mocks()

        with patch.object(
            extractor,
            "_detect_speech_intervals",
            return_value=np.array([[0, len(sample_audio) // 2]], dtype=int),
        ):
            result = extractor.extract(
                audio_clip=sample_audio,
                word_timestamps=SAMPLE_WORD_TIMESTAMPS,
                response_latency_ms=500.0,
            )

        assert RAW_PROSODY_KEYS.issubset(result.keys())
        assert len(result["mfcc_vector"]) == MFCC_COEFFICIENTS
        assert isinstance(result["pause_count"], int)
        assert isinstance(result["disfluency_timestamps"], list)
        assert result["response_latency_ms"] == 500.0


class TestProsodyPipeline:
    def test_process_prosody_turn_includes_deviations(self, sample_audio):
        manager = ProsodyBaselineManager(baseline_turns=2)
        extractor = _make_extractor_with_mocks()

        with patch(
            "modules.module_03_prosody.pipeline.prosody_extractor", extractor
        ), patch(
            "modules.module_03_prosody.pipeline.prosody_baseline_manager", manager
        ), patch.object(
            extractor,
            "_detect_speech_intervals",
            return_value=np.array([[0, len(sample_audio) // 2]], dtype=int),
        ):
            turn1 = process_prosody_turn(
                audio_clip=sample_audio,
                turn_id=1,
                candidate_id="pipe_c1",
                word_timestamps=SAMPLE_WORD_TIMESTAMPS,
                response_latency_ms=300.0,
            )
            turn3 = process_prosody_turn(
                audio_clip=sample_audio,
                turn_id=3,
                candidate_id="pipe_c1",
                word_timestamps=SAMPLE_WORD_TIMESTAMPS,
                response_latency_ms=300.0,
            )

        assert turn1["pitch_deviation"] is None
        assert DEVIATION_KEYS.issubset(turn3.keys())
        assert isinstance(turn3["pitch_deviation"], float)


@pytest.mark.integration
@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("opensmile"),
    reason="opensmile not installed",
)
class TestProsodyIntegration:
    def test_extract_30_second_audio_clip(self):
        wav_path = ensure_prosody_fixture(duration_s=30.0)
        import soundfile as sf

        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        extractor = ProsodyExtractor()
        result = extractor.extract(
            audio_clip=audio,
            word_timestamps=SAMPLE_WORD_TIMESTAMPS,
            response_latency_ms=400.0,
        )

        assert RAW_PROSODY_KEYS.issubset(result.keys())
        assert len(result["mfcc_vector"]) == 13
        assert isinstance(result["speech_rate"], float)
        assert result["speech_rate"] >= 0.0

    def test_full_pipeline_baseline_flow(self):
        wav_path = ensure_prosody_fixture(duration_s=30.0)
        import soundfile as sf

        audio, _ = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Use fresh manager/extractor via direct calls to avoid singleton state bleed
        from modules.module_03_prosody.baseline import ProsodyBaselineManager
        from modules.module_03_prosody.extractor import ProsodyExtractor

        extractor = ProsodyExtractor()
        manager = ProsodyBaselineManager(baseline_turns=2)

        raw = extractor.extract(
            audio,
            word_timestamps=SAMPLE_WORD_TIMESTAMPS,
            response_latency_ms=200.0,
        )
        t1 = manager.update_with_baseline("int_c2", 1, raw)
        raw = extractor.extract(
            audio,
            word_timestamps=SAMPLE_WORD_TIMESTAMPS,
            response_latency_ms=200.0,
        )
        t2 = manager.update_with_baseline("int_c2", 2, raw)
        raw = extractor.extract(
            audio,
            word_timestamps=SAMPLE_WORD_TIMESTAMPS,
            response_latency_ms=200.0,
        )
        t3 = manager.update_with_baseline("int_c2", 3, raw)

        assert t1["pitch_deviation"] is None
        assert t2["pitch_deviation"] is None
        assert t3["pitch_deviation"] is not None
