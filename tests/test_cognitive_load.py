"""
Unit tests for Module 10 — Cognitive Load Classifier.

Tests all 4 classification quadrants, distress score computation,
baseline turn behavior, edge cases, and output contract compliance.
"""

import pytest

from modules.module_10_cognitive_load import CognitiveLoadClassifier
from modules.module_10_cognitive_load.classifier import (
    COGNITIVE_LOAD_LABELS,
    _clamp,
    _sigmoid_normalize,
)


# ── FIXTURES ───────────────────────────────────────────────────────────────

@pytest.fixture
def classifier():
    return CognitiveLoadClassifier()


@pytest.fixture
def high_stress_prosody():
    """Prosody features indicating high physiological stress (turn 3+)."""
    return {
        "pitch_mean": 250.0,
        "pitch_variance": 45.0,
        "pitch_range": 120.0,
        "speech_rate": 2.5,
        "pause_count": 8,
        "pause_total_duration_ms": 4500.0,
        "disfluency_count": 6,
        "response_latency_ms": 3500.0,
        "energy_mean": 0.55,
        "jitter": 0.04,
        "shimmer": 0.12,
        "mfcc_vector": [0.0] * 13,
        "speech_to_silence_ratio": 1.2,
        # Baseline deviations (turn 3+)
        "pitch_deviation": 0.45,
        "rate_deviation": -0.35,
        "energy_deviation": 0.30,
    }


@pytest.fixture
def low_stress_prosody():
    """Prosody features indicating low stress (turn 3+)."""
    return {
        "pitch_mean": 170.0,
        "pitch_variance": 12.0,
        "pitch_range": 40.0,
        "speech_rate": 4.2,
        "pause_count": 2,
        "pause_total_duration_ms": 800.0,
        "disfluency_count": 1,
        "response_latency_ms": 800.0,
        "energy_mean": 0.70,
        "jitter": 0.008,
        "shimmer": 0.06,
        "mfcc_vector": [0.0] * 13,
        "speech_to_silence_ratio": 3.5,
        # Baseline deviations (turn 3+)
        "pitch_deviation": 0.05,
        "rate_deviation": 0.02,
        "energy_deviation": -0.03,
    }


@pytest.fixture
def stressed_vision():
    """Vision summary indicating stress — low eye contact, nervous emotion."""
    return {
        "emotion_label": "nervous",
        "emotion_confidence": 0.75,
        "eye_contact_score": 0.25,
        "blink_rate": 28.0,
        "gaze_vector": {"yaw": 15.0, "pitch": -5.0},
        "head_pose": {"roll": 2.0, "pitch": -8.0, "yaw": 12.0},
    }


@pytest.fixture
def calm_vision():
    """Vision summary indicating calmness — high eye contact, confident emotion."""
    return {
        "emotion_label": "confident",
        "emotion_confidence": 0.88,
        "eye_contact_score": 0.85,
        "blink_rate": 16.0,
        "gaze_vector": {"yaw": 3.0, "pitch": -2.0},
        "head_pose": {"roll": 1.0, "pitch": -3.0, "yaw": 5.0},
    }


# ── OUTPUT CONTRACT TESTS ─────────────────────────────────────────────────

class TestOutputContract:
    """Verify all outputs match the Module 10 spec."""

    def test_output_has_all_required_keys(self, classifier, low_stress_prosody, calm_vision):
        result = classifier.classify(
            prosody=low_stress_prosody,
            vision=calm_vision,
            semantic_score=0.85,
            turn_id=3,
            candidate_id="test_001",
        )
        assert "cognitive_load_label" in result
        assert "distress_score" in result
        assert "confidence" in result
        assert "signals_used" in result

    def test_label_is_valid(self, classifier, low_stress_prosody, calm_vision):
        result = classifier.classify(
            prosody=low_stress_prosody,
            vision=calm_vision,
            semantic_score=0.85,
            turn_id=3,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] in COGNITIVE_LOAD_LABELS

    def test_distress_score_bounded(self, classifier, high_stress_prosody, stressed_vision):
        result = classifier.classify(
            prosody=high_stress_prosody,
            vision=stressed_vision,
            semantic_score=0.30,
            turn_id=5,
            candidate_id="test_001",
        )
        assert 0.0 <= result["distress_score"] <= 1.0

    def test_confidence_bounded(self, classifier, low_stress_prosody, calm_vision):
        result = classifier.classify(
            prosody=low_stress_prosody,
            vision=calm_vision,
            semantic_score=0.85,
            turn_id=3,
            candidate_id="test_001",
        )
        assert 0.0 <= result["confidence"] <= 1.0

    def test_signals_used_is_list(self, classifier, low_stress_prosody, calm_vision):
        result = classifier.classify(
            prosody=low_stress_prosody,
            vision=calm_vision,
            semantic_score=0.85,
            turn_id=3,
            candidate_id="test_001",
        )
        assert isinstance(result["signals_used"], list)
        assert len(result["signals_used"]) > 0


# ── 4-QUADRANT CLASSIFICATION TESTS ───────────────────────────────────────

class TestClassificationQuadrants:
    """Test the four cognitive load quadrants."""

    def test_low_load_high_semantic(self, classifier, low_stress_prosody, calm_vision):
        """Low stress + high semantic = 'low' (optimal mastery)."""
        result = classifier.classify(
            prosody=low_stress_prosody,
            vision=calm_vision,
            semantic_score=0.85,
            turn_id=3,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] == "low"

    def test_anxiety_high_stress_high_semantic(self, classifier, high_stress_prosody, stressed_vision):
        """High stress + high semantic = 'anxiety' (knows but nervous)."""
        result = classifier.classify(
            prosody=high_stress_prosody,
            vision=stressed_vision,
            semantic_score=0.80,
            turn_id=4,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] == "anxiety"

    def test_ignorance_high_stress_low_semantic(self, classifier, high_stress_prosody, stressed_vision):
        """High stress + low semantic = 'ignorance' (doesn't know)."""
        result = classifier.classify(
            prosody=high_stress_prosody,
            vision=stressed_vision,
            semantic_score=0.20,
            turn_id=4,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] == "ignorance"

    def test_confident_ignorance_low_stress_low_semantic(self, classifier, low_stress_prosody, calm_vision):
        """Low stress + low semantic = 'confident_ignorance' (bluffing)."""
        result = classifier.classify(
            prosody=low_stress_prosody,
            vision=calm_vision,
            semantic_score=0.15,
            turn_id=3,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] == "confident_ignorance"


# ── BASELINE TURN BEHAVIOR ────────────────────────────────────────────────

class TestBaselineTurns:
    """Test behavior during baseline calibration turns (1–2)."""

    def test_turn_1_defaults_to_low(self, classifier, high_stress_prosody, stressed_vision):
        """Baseline turns always return 'low' regardless of signals."""
        result = classifier.classify(
            prosody=high_stress_prosody,
            vision=stressed_vision,
            semantic_score=0.10,
            turn_id=1,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] == "low"
        assert result["confidence"] == 0.3
        assert "baseline_turn_default" in result["signals_used"]

    def test_turn_2_defaults_to_low(self, classifier, high_stress_prosody, stressed_vision):
        result = classifier.classify(
            prosody=high_stress_prosody,
            vision=stressed_vision,
            semantic_score=0.10,
            turn_id=2,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] == "low"

    def test_turn_3_uses_real_classification(self, classifier, high_stress_prosody, stressed_vision):
        """Turn 3+ should use actual classification logic."""
        result = classifier.classify(
            prosody=high_stress_prosody,
            vision=stressed_vision,
            semantic_score=0.10,
            turn_id=3,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] != "low" or result["distress_score"] > 0
        assert "baseline_turn_default" not in result["signals_used"]


# ── EDGE CASES ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test robustness with missing or extreme inputs."""

    def test_none_prosody(self, classifier, calm_vision):
        """Should handle None prosody gracefully."""
        result = classifier.classify(
            prosody=None,
            vision=calm_vision,
            semantic_score=0.5,
            turn_id=3,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] in COGNITIVE_LOAD_LABELS
        assert 0.0 <= result["distress_score"] <= 1.0

    def test_none_vision(self, classifier, low_stress_prosody):
        """Should handle None vision gracefully."""
        result = classifier.classify(
            prosody=low_stress_prosody,
            vision=None,
            semantic_score=0.5,
            turn_id=3,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] in COGNITIVE_LOAD_LABELS

    def test_empty_dicts(self, classifier):
        """Should handle empty prosody and vision dicts."""
        result = classifier.classify(
            prosody={},
            vision={},
            semantic_score=0.5,
            turn_id=3,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] in COGNITIVE_LOAD_LABELS

    def test_semantic_score_clamped(self, classifier, low_stress_prosody, calm_vision):
        """Semantic score should be clamped to [0, 1]."""
        result = classifier.classify(
            prosody=low_stress_prosody,
            vision=calm_vision,
            semantic_score=1.5,  # Out of range
            turn_id=3,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] in COGNITIVE_LOAD_LABELS

    def test_missing_deviation_fields(self, classifier, calm_vision):
        """Prosody without deviation fields (as during early turns)."""
        prosody = {
            "pitch_mean": 180.0,
            "speech_rate": 4.0,
            "energy_mean": 0.65,
            "disfluency_count": 1,
            "jitter": 0.01,
            "pause_count": 2,
            # No deviation fields
        }
        result = classifier.classify(
            prosody=prosody,
            vision=calm_vision,
            semantic_score=0.75,
            turn_id=5,
            candidate_id="test_001",
        )
        assert result["cognitive_load_label"] in COGNITIVE_LOAD_LABELS
        assert len(result["signals_used"]) > 0


# ── UTILITY FUNCTION TESTS ────────────────────────────────────────────────

class TestUtilities:
    def test_clamp_within_range(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_clamp_below_range(self):
        assert _clamp(-0.5, 0.0, 1.0) == 0.0

    def test_clamp_above_range(self):
        assert _clamp(1.5, 0.0, 1.0) == 1.0

    def test_sigmoid_normalize_at_midpoint(self):
        result = _sigmoid_normalize(0.5, midpoint=0.5, steepness=10.0)
        assert abs(result - 0.5) < 0.01

    def test_sigmoid_normalize_high_value(self):
        result = _sigmoid_normalize(10.0, midpoint=0.5, steepness=10.0)
        assert result > 0.99

    def test_sigmoid_normalize_low_value(self):
        result = _sigmoid_normalize(0.0, midpoint=5.0, steepness=2.0)
        assert result < 0.01

    def test_sigmoid_normalize_overflow_protection(self):
        """Should not raise on extreme values."""
        result = _sigmoid_normalize(1e10, midpoint=0.0, steepness=1000.0)
        assert 0.0 <= result <= 1.0

        result = _sigmoid_normalize(-1e10, midpoint=0.0, steepness=1000.0)
        assert 0.0 <= result <= 1.0
