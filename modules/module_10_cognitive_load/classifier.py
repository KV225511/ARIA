"""
Module 10 — Cognitive Load Classifier

Rule-based classifier that separates four distinct cognitive states during
an interview turn by combining physiological stress signals with semantic
answer quality.

Classification quadrants:
    ┌──────────────────────┬──────────────────────────────┐
    │  High Stress         │  High Stress                 │
    │  + High Semantic     │  + Low Semantic              │
    │  = ANXIETY           │  = IGNORANCE                 │
    │  (knows, but nervous)│  (doesn't know)              │
    ├──────────────────────┼──────────────────────────────┤
    │  Low Stress          │  Low Stress                  │
    │  + High Semantic     │  + Low Semantic              │
    │  = LOW               │  = CONFIDENT_IGNORANCE       │
    │  (optimal mastery)   │  (bluffing / overconfident)  │
    └──────────────────────┴──────────────────────────────┘

Inputs consumed (from Modules 1–4, already implemented):
    - prosody features (Module 3): pitch deviation, rate deviation, energy
      deviation, disfluency count, jitter, speech rate, pause count
    - vision summary (Module 2): eye contact score, gaze breaks, emotion label
    - semantic score (Module 1 SemanticGrader or external): float 0.0–1.0

Output contract (matches ARIA_Coding_Assistant_Guide Section 5, Module 10):
    {
        "cognitive_load_label": str,    # "low" / "anxiety" / "ignorance" / "confident_ignorance"
        "distress_score": float,        # continuous 0.0–1.0
        "confidence": float,            # classifier confidence 0.0–1.0
        "signals_used": list            # which signals drove this prediction
    }

Owner: Krissh
"""

from __future__ import annotations

import math
from typing import Any

from config.settings import BASELINE_TURNS


# ── VALID OUTPUT LABELS ────────────────────────────────────────────────────

COGNITIVE_LOAD_LABELS = frozenset({
    "low",
    "anxiety",
    "ignorance",
    "confident_ignorance",
})


# ── THRESHOLDS ─────────────────────────────────────────────────────────────
# Composite distress score threshold: above = "high stress"
DISTRESS_HIGH_THRESHOLD = 0.45

# Semantic score threshold: above = "high semantic" (candidate knows the answer)
SEMANTIC_HIGH_THRESHOLD = 0.55

# Individual signal weights for composite distress scoring.
# Weights sum to 1.0 — each captures a distinct physiological stress channel.
DISTRESS_WEIGHTS = {
    "pitch_deviation":   0.20,  # Vocal pitch shift from baseline
    "rate_deviation":    0.15,  # Speech rate change from baseline
    "disfluency_rate":   0.20,  # Filler words per second (um, uh, erm)
    "gaze_instability":  0.15,  # Eye contact loss / gaze breaks
    "jitter":            0.10,  # Vocal jitter (cycle-to-cycle F0 variation)
    "pause_density":     0.10,  # Within-answer pause frequency
    "energy_deviation":  0.10,  # Vocal energy shift from baseline
}

_NUM_DISTRESS_SIGNALS = len(DISTRESS_WEIGHTS)

class CognitiveLoadClassifier:
    """
    Rule-based 4-quadrant cognitive load classifier.

    Stateless — each call is independent. No model weights or GPU usage.
    Thread-safe — no shared mutable state.

    Usage:
        classifier = CognitiveLoadClassifier()
        result = classifier.classify(
            prosody=prosody_features,
            vision=vision_summary,
            semantic_score=0.85,
            turn_id=3,
            candidate_id="cand_001",
        )
    """

    def __init__(
        self,
        semantic_threshold: float = SEMANTIC_HIGH_THRESHOLD,
        distress_threshold: float = DISTRESS_HIGH_THRESHOLD,
        baseline_turns: int = BASELINE_TURNS,
    ) -> None:
        self.semantic_threshold = semantic_threshold
        self.distress_threshold = distress_threshold
        self.baseline_turns = baseline_turns

    # ── PUBLIC API ─────────────────────────────────────────────────────────

    def classify(
        self,
        prosody: dict[str, Any],
        vision: dict[str, Any],
        semantic_score: float,
        turn_id: int,
        candidate_id: str,
        word_count: int = 1,
    ) -> dict[str, Any]:
        """
        Classify the cognitive load state for a single interview turn.

        Args:
            prosody: Module 3 output dict (with baseline deviations for turn 3+).
            vision: Module 2 per-turn summary dict.
            semantic_score: Semantic similarity score 0.0–1.0 from Module 1
                            SemanticGrader or external evaluation.
            turn_id: 1-indexed turn number in the session.
            candidate_id: Candidate identifier (for logging context).
            word_count: Number of words spoken in the turn.

        Returns:
            Dict matching Module 10 output contract.
        """
        # Ensure inputs are safe dicts
        prosody = prosody or {}
        vision = vision or {}
        semantic_score = _clamp(float(semantic_score), 0.0, 1.0)
        word_count = max(1, word_count)

        # For baseline turns (1–2), deviation fields are None — we have
        # insufficient signal to separate anxiety from ignorance.
        if turn_id <= self.baseline_turns:
            return {
                "cognitive_load_label": None,
                "distress_score": 0.0,
                "confidence": 0.3,  # Low confidence — no baseline yet
                "signals_used": ["baseline_turn_default"],
            }

        # ── Compute composite distress score ───────────────────────────────
        distress_score, signals_used = self._compute_distress_score(
            prosody, vision, word_count
        )

        # ── 4-quadrant classification ──────────────────────────────────────
        is_high_stress = distress_score >= self.distress_threshold
        is_high_semantic = semantic_score >= self.semantic_threshold

        if is_high_stress and is_high_semantic:
            label = "anxiety"
        elif is_high_stress and not is_high_semantic:
            label = "ignorance"
        elif not is_high_stress and is_high_semantic:
            label = "low"
        else:
            label = "confident_ignorance"

        # ── Classifier confidence ──────────────────────────────────────────
        # Higher confidence when signals clearly fall on one side of thresholds.
        confidence = self._compute_confidence(
            distress_score, semantic_score, signals_used
        )

        return {
            "cognitive_load_label": label,
            "distress_score": _clamp(distress_score, 0.0, 1.0),
            "confidence": _clamp(confidence, 0.0, 1.0),
            "signals_used": signals_used,
        }

    # ── DISTRESS SCORE COMPUTATION ─────────────────────────────────────────

    def _compute_distress_score(
        self,
        prosody: dict[str, Any],
        vision: dict[str, Any],
        word_count: int,
    ) -> tuple[float, list[str]]:
        """
        Compute a composite distress score from available physiological signals.

        Each signal is normalized to [0, 1] and weighted. Missing signals
        are skipped and their weight redistributed to available signals.

        Returns:
            (distress_score, list_of_signals_that_contributed)
        """
        raw_signals: dict[str, float] = {}
        signals_used: list[str] = []

        # ── Pitch deviation ────────────────────────────────────────────────
        pitch_dev = prosody.get("pitch_deviation")
        if pitch_dev is not None:
            # Absolute deviation — both higher and lower pitch indicate stress
            raw_signals["pitch_deviation"] = _sigmoid_normalize(
                abs(float(pitch_dev)), midpoint=0.3, steepness=6.0
            )
            signals_used.append("pitch_deviation")

        # ── Speech rate deviation ──────────────────────────────────────────
        rate_dev = prosody.get("rate_deviation")
        if rate_dev is not None:
            # Negative deviation (slower speech) and positive (rushed) both
            # indicate stress, but slow is more associated with ignorance
            raw_signals["rate_deviation"] = _sigmoid_normalize(
                abs(float(rate_dev)), midpoint=0.25, steepness=6.0
            )
            signals_used.append("rate_deviation")

        # ── Disfluency rate ────────────────────────────────────────────────
        disfluency_count = float(prosody.get("disfluency_count", 0))
        speech_rate = float(prosody.get("speech_rate", 0.0))
        if speech_rate > 0:
            # Disfluencies per second of speaking time
            # Estimate speaking duration from word count / speech rate
            speaking_duration = word_count / max(speech_rate, 0.1)
            disfluency_rate = disfluency_count / max(speaking_duration, 1.0)
            raw_signals["disfluency_rate"] = _sigmoid_normalize(
                disfluency_rate, midpoint=0.15, steepness=10.0
            )
            signals_used.append("disfluency_rate")
        elif disfluency_count > 0:
            # Have disfluencies but no speech rate — use raw count
            raw_signals["disfluency_rate"] = _sigmoid_normalize(
                disfluency_count, midpoint=3.0, steepness=1.0
            )
            signals_used.append("disfluency_rate")

        # ── Gaze instability ──────────────────────────────────────────────
        eye_contact = vision.get("eye_contact_score")
        if eye_contact is not None:
            # Low eye contact → high gaze instability → higher stress signal
            gaze_instability = 1.0 - _clamp(float(eye_contact), 0.0, 1.0)
            raw_signals["gaze_instability"] = gaze_instability
            signals_used.append("gaze_instability")

        # ── Jitter ─────────────────────────────────────────────────────────
        jitter = prosody.get("jitter")
        if jitter is not None:
            raw_signals["jitter"] = _sigmoid_normalize(
                float(jitter), midpoint=0.02, steepness=100.0
            )
            signals_used.append("jitter")

        # ── Pause density ──────────────────────────────────────────────────
        pause_count = prosody.get("pause_count")
        if pause_count is not None:
            raw_signals["pause_density"] = _sigmoid_normalize(
                float(pause_count), midpoint=4.0, steepness=0.8
            )
            signals_used.append("pause_density")

        # ── Energy deviation ───────────────────────────────────────────────
        energy_dev = prosody.get("energy_deviation")
        if energy_dev is not None:
            raw_signals["energy_deviation"] = _sigmoid_normalize(
                abs(float(energy_dev)), midpoint=0.3, steepness=6.0
            )
            signals_used.append("energy_deviation")

        # ── Weighted combination with redistribution ───────────────────────
        if not raw_signals:
            return 0.0, ["no_signals_available"]

        # Compute total weight of available signals for redistribution
        available_weight = sum(
            DISTRESS_WEIGHTS[name]
            for name in raw_signals
            if name in DISTRESS_WEIGHTS
        )

        if available_weight < 1e-6:
            return 0.0, signals_used

        # Weighted sum with weight redistribution (missing signals' weight
        # is proportionally redistributed to available signals)
        distress_score = sum(
            (DISTRESS_WEIGHTS.get(name, 0.0) / available_weight) * value
            for name, value in raw_signals.items()
        )

        return _clamp(distress_score, 0.0, 1.0), signals_used

    # ── CONFIDENCE ESTIMATION ──────────────────────────────────────────────

    def _compute_confidence(
        self,
        distress_score: float,
        semantic_score: float,
        signals_used: list[str],
    ) -> float:
        """
        Estimate classifier confidence based on signal clarity and coverage.

        Confidence is higher when:
        - Distress and semantic scores are far from their thresholds (clear signal)
        - More physiological signals are available (better coverage)
        """
        # Distance from threshold boundaries → clearer signal = higher confidence
        distress_clarity = abs(distress_score - self.distress_threshold)
        semantic_clarity = abs(semantic_score - self.semantic_threshold)

        # Combine clarity scores (both contribute equally)
        clarity_score = (distress_clarity + semantic_clarity) / 2.0

        # Signal coverage factor: more signals = more reliable
        max_signals = _NUM_DISTRESS_SIGNALS
        # Exclude meta-signals like "baseline_turn_default", "no_signals_available"
        actual_signals = len([
            s for s in signals_used
            if s in DISTRESS_WEIGHTS
        ])
        coverage = actual_signals / max(max_signals, 1)

        # Final confidence: blend of clarity and coverage
        # Base confidence of 0.4, scaled up by clarity and coverage
        confidence = 0.4 + 0.35 * clarity_score + 0.25 * coverage

        return _clamp(confidence, 0.0, 1.0)


# ── UTILITY FUNCTIONS ──────────────────────────────────────────────────────

def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to [low, high]."""
    return max(low, min(high, value))


def _sigmoid_normalize(
    value: float,
    midpoint: float = 0.5,
    steepness: float = 10.0,
) -> float:
    """
    Soft sigmoid normalization to [0, 1].

    Maps raw signal values to a smooth 0–1 range using a logistic curve.
    Values at `midpoint` map to 0.5. `steepness` controls transition sharpness.

    This avoids hard threshold artifacts and gracefully handles outliers.
    """
    try:
        exponent = -steepness * (value - midpoint)
        # Guard against overflow in exp()
        if exponent > 500:
            return 0.0
        if exponent < -500:
            return 1.0
        return 1.0 / (1.0 + math.exp(exponent))
    except (OverflowError, ValueError):
        return 0.5
