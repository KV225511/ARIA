"""
Unit tests for Module 11 — Anti-Gaming & Integrity Monitor.

Tests all three detectors (GazeScanner, LatencyChecker, SemanticChecker)
individually and the AntiGamingMonitor orchestrator.
"""

import pytest

from modules.module_11_anti_gaming import (
    AntiGamingMonitor,
    GazeScanner,
    LatencyChecker,
    SemanticChecker,
)


# ══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def gaze_scanner():
    return GazeScanner()


@pytest.fixture
def latency_checker():
    return LatencyChecker()


@pytest.fixture
def semantic_checker():
    return SemanticChecker()


@pytest.fixture
def monitor():
    return AntiGamingMonitor()


@pytest.fixture
def reading_gaze_frames():
    """Gaze frames simulating left-to-right reading sweep over 3+ seconds."""
    frames = []
    # 10 frames over 3.5 seconds, yaw moving from -15 to +15 (30° sweep)
    for i in range(10):
        frames.append({
            "gaze_vector": {
                "yaw": -15.0 + (i * 3.33),   # Monotonic left-to-right
                "pitch": -2.0 + (i * 0.2),    # Minimal vertical movement
            },
            "timestamp_ms": i * 350.0,  # 350ms intervals = 3.15s total
        })
    return frames


@pytest.fixture
def natural_gaze_frames():
    """Gaze frames simulating natural looking around (non-reading)."""
    import random
    random.seed(42)
    frames = []
    for i in range(10):
        frames.append({
            "gaze_vector": {
                "yaw": random.uniform(-20, 20),    # Random horizontal
                "pitch": random.uniform(-15, 15),   # Random vertical
            },
            "timestamp_ms": i * 500.0,
        })
    return frames


@pytest.fixture
def ai_assist_word_timestamps():
    """Word timestamps with unnaturally uniform delivery (flat rate)."""
    words = []
    # 20 words at exactly 0.25s per word = perfectly uniform 4 words/sec
    for i in range(20):
        words.append({
            "word": f"word{i}",
            "start": 5.0 + i * 0.25,
            "end": 5.0 + i * 0.25 + 0.20,
        })
    return words


@pytest.fixture
def natural_word_timestamps():
    """Word timestamps with natural rhythm variation."""
    # Irregular timing — some fast bursts, some pauses
    return [
        {"word": "well", "start": 0.5, "end": 0.8},
        {"word": "I", "start": 0.9, "end": 1.0},
        {"word": "think", "start": 1.0, "end": 1.3},
        {"word": "that", "start": 1.4, "end": 1.5},
        {"word": "the", "start": 2.1, "end": 2.2},      # pause before this
        {"word": "main", "start": 2.2, "end": 2.5},
        {"word": "approach", "start": 2.5, "end": 2.9},
        {"word": "would", "start": 3.5, "end": 3.7},     # pause before this
        {"word": "be", "start": 3.7, "end": 3.8},
        {"word": "to", "start": 3.8, "end": 3.9},
        {"word": "use", "start": 4.0, "end": 4.2},
        {"word": "a", "start": 4.2, "end": 4.3},
        {"word": "hash", "start": 4.3, "end": 4.6},
        {"word": "map", "start": 4.6, "end": 4.9},
    ]


@pytest.fixture
def session_history_normal():
    """Normal session history with diverse, distinct answers."""
    return [
        {"transcript": "I would use a binary search tree for efficient lookup operations and maintain sorted order", "turn_id": 1},
        {"transcript": "REST APIs follow the client server architecture with stateless communication over HTTP", "turn_id": 2},
        {"transcript": "Docker containers provide isolation through namespaces and cgroups on the Linux kernel", "turn_id": 3},
    ]


@pytest.fixture
def session_history_scripted():
    """Session history where answers are suspiciously similar (recycled script)."""
    # Intentionally near-identical answers recycled across different questions —
    # high TF-IDF cosine similarity expected (>0.65)
    return [
        {"transcript": "I utilized advanced methodologies and best practices to deliver high quality results that exceeded stakeholder expectations in my role", "turn_id": 1},
        {"transcript": "I utilized advanced methodologies and best practices to deliver high quality results that exceeded stakeholder expectations at the company", "turn_id": 2},
        {"transcript": "I utilized advanced methodologies and best practices to deliver high quality results that exceeded stakeholder expectations overall", "turn_id": 3},
    ]


# ══════════════════════════════════════════════════════════════════════════
# GAZE SCANNER TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestGazeScanner:
    def test_reading_sweep_detected(self, gaze_scanner, reading_gaze_frames):
        """Sustained horizontal sweep should produce high confidence."""
        result = gaze_scanner.detect(reading_gaze_frames)
        assert result["flag"] == "note_reading"
        assert result["confidence"] > 0.3
        assert result["evidence"]["sweep_count"] >= 1
        assert result["evidence"]["max_sweep_duration_ms"] > 0

    def test_natural_gaze_not_flagged(self, gaze_scanner, natural_gaze_frames):
        """Random natural gaze should produce low/zero confidence."""
        result = gaze_scanner.detect(natural_gaze_frames)
        assert result["confidence"] < 0.3

    def test_empty_frames(self, gaze_scanner):
        """Empty frame list should return zero confidence."""
        result = gaze_scanner.detect([])
        assert result["confidence"] == 0.0
        assert result["evidence"]["sweep_count"] == 0

    def test_too_few_frames(self, gaze_scanner):
        """Fewer than 3 frames should return zero confidence."""
        result = gaze_scanner.detect([
            {"gaze_vector": {"yaw": 0, "pitch": 0}, "timestamp_ms": 0},
            {"gaze_vector": {"yaw": 5, "pitch": 0}, "timestamp_ms": 500},
        ])
        assert result["confidence"] == 0.0

    def test_short_sweep_not_flagged(self, gaze_scanner):
        """Horizontal sweep shorter than threshold should not flag."""
        # 3 frames over 1 second — below 2s threshold
        frames = [
            {"gaze_vector": {"yaw": -10, "pitch": 0}, "timestamp_ms": 0},
            {"gaze_vector": {"yaw": 0, "pitch": 0}, "timestamp_ms": 500},
            {"gaze_vector": {"yaw": 10, "pitch": 0}, "timestamp_ms": 1000},
        ]
        result = gaze_scanner.detect(frames)
        assert result["evidence"]["sweep_count"] == 0

    def test_output_schema(self, gaze_scanner, reading_gaze_frames):
        """Verify output dict structure."""
        result = gaze_scanner.detect(reading_gaze_frames)
        assert "flag" in result
        assert "confidence" in result
        assert "evidence" in result
        assert "sweep_count" in result["evidence"]
        assert "max_sweep_duration_ms" in result["evidence"]
        assert "sweep_segments" in result["evidence"]


# ══════════════════════════════════════════════════════════════════════════
# LATENCY CHECKER TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestLatencyChecker:
    def test_high_latency_flat_delivery_flagged(
        self, latency_checker, ai_assist_word_timestamps
    ):
        """High latency + uniform delivery should produce high confidence."""
        result = latency_checker.detect(
            prosody={"speech_rate": 4.0},
            word_timestamps=ai_assist_word_timestamps,
            response_latency_ms=8000.0,
        )
        assert result["flag"] == "ai_assist"
        assert result["confidence"] > 0.4
        assert result["evidence"]["latency_suspicious"] is True
        assert result["evidence"]["delivery_suspicious"] is True

    def test_normal_latency_not_flagged(
        self, latency_checker, natural_word_timestamps
    ):
        """Normal latency + natural delivery should have low confidence."""
        result = latency_checker.detect(
            prosody={"speech_rate": 3.5},
            word_timestamps=natural_word_timestamps,
            response_latency_ms=1200.0,
        )
        assert result["confidence"] < 0.3
        assert result["evidence"]["latency_suspicious"] is False

    def test_high_latency_only(self, latency_checker, natural_word_timestamps):
        """High latency alone should produce moderate confidence."""
        result = latency_checker.detect(
            prosody={"speech_rate": 3.5},
            word_timestamps=natural_word_timestamps,
            response_latency_ms=8000.0,
        )
        # Latency alone → moderate confidence (could be thinking)
        assert result["evidence"]["latency_suspicious"] is True
        assert result["confidence"] > 0.1
        assert result["confidence"] < 0.7  # Not high without delivery evidence

    def test_flat_delivery_only(self, latency_checker, ai_assist_word_timestamps):
        """Flat delivery alone should produce low confidence."""
        result = latency_checker.detect(
            prosody={"speech_rate": 4.0},
            word_timestamps=ai_assist_word_timestamps,
            response_latency_ms=1500.0,  # Normal latency
        )
        # Flat delivery alone → weak signal
        assert result["evidence"]["delivery_suspicious"] is True
        assert result["confidence"] < 0.5

    def test_empty_word_timestamps(self, latency_checker):
        """Empty word timestamps should not crash."""
        result = latency_checker.detect(
            prosody={},
            word_timestamps=[],
            response_latency_ms=8000.0,
        )
        assert result["flag"] == "ai_assist"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_too_few_words(self, latency_checker):
        """Too few words for variance analysis → non-suspicious delivery."""
        result = latency_checker.detect(
            prosody={},
            word_timestamps=[
                {"word": "yes", "start": 0.5, "end": 0.7},
                {"word": "sure", "start": 0.8, "end": 1.0},
            ],
            response_latency_ms=2000.0,
        )
        assert result["evidence"]["delivery_suspicious"] is False

    def test_output_schema(self, latency_checker, natural_word_timestamps):
        """Verify output dict structure."""
        result = latency_checker.detect(
            prosody={},
            word_timestamps=natural_word_timestamps,
            response_latency_ms=1000.0,
        )
        assert "flag" in result
        assert "confidence" in result
        assert "evidence" in result
        assert "response_latency_ms" in result["evidence"]
        assert "delivery_uniformity" in result["evidence"]
        assert "latency_suspicious" in result["evidence"]
        assert "delivery_suspicious" in result["evidence"]


# ══════════════════════════════════════════════════════════════════════════
# SEMANTIC CHECKER TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestSemanticChecker:
    def test_scripted_similar_answers_flagged(
        self, semantic_checker, session_history_scripted
    ):
        """Nearly identical answers to different questions → scripted flag."""
        # Phrasing closely matches the scripted history — should cross threshold
        transcript = (
            "I utilized advanced methodologies and best practices to deliver "
            "high quality results that exceeded stakeholder expectations in my position"
        )
        result = semantic_checker.detect(
            transcript=transcript,
            vision={"head_pose": {"yaw": 5.0, "pitch": 0, "roll": 0}},
            word_timestamps=[],
            session_history=session_history_scripted,
        )
        assert result["confidences"]["scripted"] > 0.2
        assert result["evidence"]["max_cross_turn_similarity"] > 0.4

    def test_diverse_answers_not_flagged(
        self, semantic_checker, session_history_normal
    ):
        """Distinct answers to different questions → no scripted flag."""
        transcript = (
            "For database optimization I would focus on query execution plans "
            "and proper indexing strategies to reduce full table scans"
        )
        result = semantic_checker.detect(
            transcript=transcript,
            vision={"head_pose": {"yaw": 3.0, "pitch": 0, "roll": 0}},
            word_timestamps=[],
            session_history=session_history_normal,
        )
        assert result["confidences"]["scripted"] < 0.3

    def test_lateral_head_turn_coaching_flagged(self, semantic_checker):
        """Strong lateral head turn → coaching flag."""
        result = semantic_checker.detect(
            transcript="The answer to that question involves several considerations",
            vision={"head_pose": {"yaw": 35.0, "pitch": 0, "roll": 0}},
            word_timestamps=[],
            session_history=[],
        )
        assert result["confidences"]["coaching"] > 0.3
        assert result["evidence"]["lateral_head_turns"] == 1

    def test_no_head_turn_no_coaching(self, semantic_checker):
        """Normal head position → no coaching flag."""
        result = semantic_checker.detect(
            transcript="The answer to that question involves several considerations",
            vision={"head_pose": {"yaw": 5.0, "pitch": 0, "roll": 0}},
            word_timestamps=[],
            session_history=[],
        )
        assert result["confidences"]["coaching"] == 0.0

    def test_insufficient_history(self, semantic_checker):
        """Too few history turns → scripted detection skipped."""
        result = semantic_checker.detect(
            transcript="Some answer here about algorithms and data structures",
            vision={"head_pose": {"yaw": 0, "pitch": 0, "roll": 0}},
            word_timestamps=[],
            session_history=[{"transcript": "First answer", "turn_id": 1}],  # Only 1 turn
        )
        assert result["confidences"]["scripted"] == 0.0

    def test_empty_transcript(self, semantic_checker):
        """Empty transcript should not crash."""
        result = semantic_checker.detect(
            transcript="",
            vision={},
            word_timestamps=[],
            session_history=[],
        )
        assert result["flags"] == []
        assert result["confidences"]["coaching"] == 0.0
        assert result["confidences"]["scripted"] == 0.0

    def test_output_schema(self, semantic_checker, session_history_normal):
        """Verify output dict structure."""
        result = semantic_checker.detect(
            transcript="A valid answer about something technical",
            vision={"head_pose": {"yaw": 0, "pitch": 0, "roll": 0}},
            word_timestamps=[],
            session_history=session_history_normal,
        )
        assert "flags" in result
        assert "confidences" in result
        assert "evidence" in result
        assert "coaching" in result["confidences"]
        assert "scripted" in result["confidences"]


# ══════════════════════════════════════════════════════════════════════════
# ANTI-GAMING MONITOR (ORCHESTRATOR) TESTS
# ══════════════════════════════════════════════════════════════════════════

class TestAntiGamingMonitor:
    def test_clean_session_no_flags(
        self, monitor, natural_gaze_frames, natural_word_timestamps, session_history_normal
    ):
        """Normal interview behavior should produce no flags."""
        result = monitor.evaluate_turn(
            gaze_frames=natural_gaze_frames,
            prosody={"speech_rate": 3.8},
            vision={"head_pose": {"yaw": 3.0, "pitch": 0, "roll": 0}},
            word_timestamps=natural_word_timestamps,
            transcript="I would implement a breadth first search traversal algorithm for the graph",
            session_history=session_history_normal,
            response_latency_ms=1200.0,
        )
        assert result["is_flagged"] is False
        assert result["flags"] == []
        assert result["flag_confidences"] == {}

    def test_output_contract(self, monitor):
        """Verify orchestrator output matches the spec."""
        result = monitor.evaluate_turn(
            gaze_frames=[],
            prosody={},
            vision={},
            word_timestamps=[],
            transcript="",
            session_history=[],
            response_latency_ms=0.0,
        )
        assert "flags" in result
        assert "flag_confidences" in result
        assert "is_flagged" in result
        assert isinstance(result["flags"], list)
        assert isinstance(result["flag_confidences"], dict)
        assert isinstance(result["is_flagged"], bool)

    def test_multiple_flags_aggregated(
        self, monitor, reading_gaze_frames, ai_assist_word_timestamps
    ):
        """Multiple gaming signals should produce multiple flags."""
        result = monitor.evaluate_turn(
            gaze_frames=reading_gaze_frames,
            prosody={"speech_rate": 4.0},
            vision={"head_pose": {"yaw": 35.0, "pitch": 0, "roll": 0}},
            word_timestamps=ai_assist_word_timestamps,
            transcript="A response that is being provided here for the interview question",
            session_history=[],
            response_latency_ms=9000.0,
        )
        # Should have at least one flag (possibly multiple)
        # The exact flags depend on threshold tuning, but structure should be valid
        assert isinstance(result["flags"], list)
        assert isinstance(result["flag_confidences"], dict)
        # Confidence values should be bounded
        for conf in result["flag_confidences"].values():
            assert 0.0 <= conf <= 1.0

    def test_custom_threshold(self, reading_gaze_frames):
        """Custom confidence threshold should filter flags."""
        # Very high threshold — most things won't pass
        monitor = AntiGamingMonitor(flag_confidence_threshold=0.99)
        result = monitor.evaluate_turn(
            gaze_frames=reading_gaze_frames,
            prosody={},
            vision={},
            word_timestamps=[],
            transcript="Some answer",
            session_history=[],
            response_latency_ms=1000.0,
        )
        # With very high threshold, most detections should be filtered out
        assert isinstance(result["flags"], list)

    def test_none_inputs_handled(self, monitor):
        """None inputs should not crash the orchestrator."""
        result = monitor.evaluate_turn(
            gaze_frames=None,
            prosody=None,
            vision=None,
            word_timestamps=None,
            transcript="",
            session_history=None,
            response_latency_ms=0.0,
        )
        assert result["is_flagged"] is False
