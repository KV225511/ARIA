"""
Module 11 — Anti-Gaming & Integrity Monitor

Three parallel detectors for real-time interview integrity monitoring:
    1. GazeScanner     — note reading detection via gaze patterns
    2. LatencyChecker  — AI assistance detection via latency uniformity
    3. SemanticChecker — coaching/scripting detection via semantic analysis

Plus AntiGamingMonitor orchestrator wrapping all three.

Owner: Krissh
"""

from .gaze_scanner import GazeScanner
from .latency_checker import LatencyChecker
from .semantic_checker import SemanticChecker

__all__ = [
    "GazeScanner",
    "LatencyChecker",
    "SemanticChecker",
    "AntiGamingMonitor",
]


class AntiGamingMonitor:
    """
    Orchestrator that runs all three anti-gaming detectors in sequence
    and aggregates their results into the turn_signal output contract.

    Output contract (matches ARIA_Coding_Assistant_Guide Section 5, Module 11):
        {
            "flags": list,              # empty list if clean
            "flag_confidences": dict,   # confidence per flag
            "is_flagged": bool
        }

    Thread-safe — each detector is stateless per-call (session history
    is passed in, not stored internally).

    Usage:
        monitor = AntiGamingMonitor()
        result = monitor.evaluate_turn(
            gaze_frames=[...],
            prosody=prosody_features,
            vision=vision_summary,
            word_timestamps=[...],
            transcript="...",
            session_history=[...],
        )
    """

    def __init__(
        self,
        gaze_scanner: GazeScanner | None = None,
        latency_checker: LatencyChecker | None = None,
        semantic_checker: SemanticChecker | None = None,
        flag_confidence_threshold: float = 0.5,
    ) -> None:
        self.gaze_scanner = gaze_scanner or GazeScanner()
        self.latency_checker = latency_checker or LatencyChecker()
        self.semantic_checker = semantic_checker or SemanticChecker()
        self.flag_confidence_threshold = flag_confidence_threshold

    def evaluate_turn(
        self,
        gaze_frames: list[dict] | None = None,
        prosody: dict | None = None,
        vision: dict | None = None,
        word_timestamps: list[dict] | None = None,
        transcript: str = "",
        session_history: list[dict] | None = None,
        response_latency_ms: float = 0.0,
    ) -> dict:
        """
        Run all three detectors and aggregate results.

        Args:
            gaze_frames: List of per-frame gaze dicts from Module 2
                         (each with 'gaze_vector', 'timestamp_ms').
            prosody: Module 3 output dict.
            vision: Module 2 per-turn summary dict.
            word_timestamps: Word-level timestamps from Module 1.
            transcript: Candidate's answer transcript.
            session_history: List of previous turn dicts, each containing
                             at minimum {"transcript": str, "turn_id": int}.
            response_latency_ms: Response latency from Module 1.

        Returns:
            Dict matching Module 11 output contract.
        """
        flags: list[str] = []
        flag_confidences: dict[str, float] = {}

        # ── Detector 1: Note Reading ──────────────────────────────────────
        if gaze_frames:
            gaze_result = self.gaze_scanner.detect(gaze_frames)
            if (gaze_result["confidence"]
                    >= self.flag_confidence_threshold):
                flags.append("note_reading")
                flag_confidences["note_reading"] = gaze_result["confidence"]

        # ── Detector 2: AI Assistance ─────────────────────────────────────
        if prosody or word_timestamps:
            latency_result = self.latency_checker.detect(
                prosody=prosody or {},
                word_timestamps=word_timestamps or [],
                response_latency_ms=response_latency_ms,
            )
            if (latency_result["confidence"]
                    >= self.flag_confidence_threshold):
                flags.append("ai_assist")
                flag_confidences["ai_assist"] = latency_result["confidence"]

        # ── Detector 3: Coaching / Scripting ──────────────────────────────
        if transcript:
            semantic_result = self.semantic_checker.detect(
                transcript=transcript,
                vision=vision or {},
                word_timestamps=word_timestamps or [],
                session_history=session_history or [],
            )
            for flag_name in ("coaching", "scripted"):
                conf = semantic_result.get("confidences", {}).get(flag_name, 0.0)
                if conf >= self.flag_confidence_threshold:
                    flags.append(flag_name)
                    flag_confidences[flag_name] = conf

        return {
            "flags": flags,
            "flag_confidences": flag_confidences,
            "is_flagged": len(flags) > 0,
        }
