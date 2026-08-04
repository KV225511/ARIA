"""
Module 11 — Anti-Gaming: AI Assistance Detection via Latency & Delivery Analysis

Detects candidates who may be receiving AI-generated answers by identifying
a characteristic pattern:
    1. Prolonged pre-answer latency (typing a prompt / waiting for AI response)
    2. Unnaturally uniform speech delivery (reading AI output verbatim)

Humans have natural rhythm variation — speech rate fluctuates between phrases.
AI-read answers are delivered with suspiciously flat, uniform cadence.

Input:
    prosody: dict          — Module 3 output with speech_rate, jitter, etc.
    word_timestamps: list  — Module 1 word-level timestamps
    response_latency_ms: float — Module 1 response latency

Output:
    {
        "flag": "ai_assist",
        "confidence": float,        # 0.0–1.0
        "evidence": {
            "response_latency_ms": float,
            "delivery_uniformity": float,
            "speech_rate_variance": float,
            "latency_suspicious": bool,
            "delivery_suspicious": bool,
        }
    }

Owner: Krissh
"""

from __future__ import annotations

import math
from typing import Any


# ── DETECTION THRESHOLDS ───────────────────────────────────────────────────

# Latency above this (ms) is considered suspicious — candidate may be
# waiting for an AI tool to generate an answer before speaking.
# Typical human response latency: 500–3000ms; AI-assisted: 5000–15000ms
SUSPICIOUS_LATENCY_MS = 5000.0

# Very high latency — almost certainly waiting for something
HIGH_LATENCY_MS = 10000.0

# Speech rate coefficient of variation below this threshold indicates
# unnaturally uniform delivery. Normal speech: CV ~0.15–0.40
# AI-read delivery: CV ~0.02–0.08
UNIFORMITY_CV_THRESHOLD = 0.08

# Minimum word count to reliably compute delivery uniformity.
# Short answers don't provide enough data for variance analysis.
MIN_WORDS_FOR_ANALYSIS = 8

# Segment size (in words) for computing local speech rate variance.
SEGMENT_SIZE_WORDS = 4


class LatencyChecker:
    """
    Detects AI assistance patterns by analyzing response latency
    combined with speech delivery uniformity.

    Both signals must be present for high confidence — high latency
    alone could be thinking time, and uniform delivery alone could
    be a well-rehearsed answer.

    Stateless per-call — no session memory required.

    Usage:
        checker = LatencyChecker()
        result = checker.detect(
            prosody=prosody_features,
            word_timestamps=word_timestamps,
            response_latency_ms=8500.0,
        )
    """

    def __init__(
        self,
        suspicious_latency_ms: float = SUSPICIOUS_LATENCY_MS,
        high_latency_ms: float = HIGH_LATENCY_MS,
        uniformity_threshold: float = UNIFORMITY_CV_THRESHOLD,
        min_words: int = MIN_WORDS_FOR_ANALYSIS,
        segment_size: int = SEGMENT_SIZE_WORDS,
    ) -> None:
        self.suspicious_latency_ms = suspicious_latency_ms
        self.high_latency_ms = high_latency_ms
        self.uniformity_threshold = uniformity_threshold
        self.min_words = min_words
        self.segment_size = segment_size

    def detect(
        self,
        prosody: dict[str, Any],
        word_timestamps: list[dict[str, Any]],
        response_latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        """
        Analyze response latency and delivery uniformity for AI assistance.

        Args:
            prosody: Module 3 output dict.
            word_timestamps: Word-level timestamps from Module 1, each:
                {"word": str, "start": float, "end": float}
            response_latency_ms: Response latency from Module 1.

        Returns:
            Detection result dict with flag, confidence, and evidence.
        """
        response_latency_ms = max(0.0, float(response_latency_ms))

        # ── Latency analysis ──────────────────────────────────────────────
        latency_suspicious = response_latency_ms >= self.suspicious_latency_ms
        latency_score = self._compute_latency_score(response_latency_ms)

        # ── Delivery uniformity analysis ──────────────────────────────────
        delivery_uniformity, speech_rate_variance = (
            self._compute_delivery_uniformity(word_timestamps)
        )
        delivery_suspicious = delivery_uniformity < self.uniformity_threshold

        # ── Combined confidence ───────────────────────────────────────────
        # Both signals reinforce each other — multiplicative interaction
        if latency_suspicious and delivery_suspicious:
            # Strong evidence: high latency + flat delivery
            confidence = 0.5 + 0.3 * latency_score + 0.2 * (1.0 - delivery_uniformity / max(self.uniformity_threshold, 0.01))
        elif latency_suspicious:
            # Only latency is suspicious — moderate signal
            confidence = 0.15 + 0.25 * latency_score
        elif delivery_suspicious:
            # Only delivery is suspicious — weak signal (could be rehearsed)
            confidence = 0.10 + 0.15 * (1.0 - delivery_uniformity / max(self.uniformity_threshold, 0.01))
        else:
            confidence = 0.0

        return {
            "flag": "ai_assist",
            "confidence": _clamp(confidence, 0.0, 1.0),
            "evidence": {
                "response_latency_ms": response_latency_ms,
                "delivery_uniformity": delivery_uniformity,
                "speech_rate_variance": speech_rate_variance,
                "latency_suspicious": latency_suspicious,
                "delivery_suspicious": delivery_suspicious,
            },
        }

    def _compute_latency_score(self, latency_ms: float) -> float:
        """
        Compute a 0–1 score for how suspicious the latency is.

        Linear ramp from suspicious threshold to high threshold.
        """
        if latency_ms < self.suspicious_latency_ms:
            return 0.0
        if latency_ms >= self.high_latency_ms:
            return 1.0

        # Linear interpolation between thresholds
        range_ms = self.high_latency_ms - self.suspicious_latency_ms
        if range_ms < 1e-6:
            return 1.0
        return (latency_ms - self.suspicious_latency_ms) / range_ms

    def _compute_delivery_uniformity(
        self,
        word_timestamps: list[dict[str, Any]],
    ) -> tuple[float, float]:
        """
        Compute the coefficient of variation (CV) of local speech rate
        across segments of the answer.

        Returns:
            (cv, variance) — lower CV = more uniform = more suspicious
        """
        if not word_timestamps or len(word_timestamps) < self.min_words:
            # Not enough data — return neutral (non-suspicious) value
            return 1.0, 0.0

        # Compute local speech rates for sliding segments
        segment_rates: list[float] = []

        for i in range(0, len(word_timestamps) - self.segment_size + 1):
            segment = word_timestamps[i:i + self.segment_size]
            start = segment[0].get("start", 0.0)
            end = segment[-1].get("end", 0.0)
            duration = float(end) - float(start)

            if duration > 0.01:  # Avoid division by zero
                rate = self.segment_size / duration  # words per second
                segment_rates.append(rate)

        if len(segment_rates) < 2:
            return 1.0, 0.0

        # Coefficient of variation = std / mean
        mean_rate = sum(segment_rates) / len(segment_rates)
        if mean_rate < 1e-6:
            return 1.0, 0.0

        variance = sum((r - mean_rate) ** 2 for r in segment_rates) / len(segment_rates)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_rate

        return cv, variance


# ── UTILITY FUNCTIONS ──────────────────────────────────────────────────────

def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to [low, high]."""
    return max(low, min(high, value))
