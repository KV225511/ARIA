"""
DeepFace emotion classifier with interview-context label remapping.

Generic DeepFace labels (happy, neutral, fear, ...) are mapped to:
engaged / confused / nervous / confident / blank

Output includes both:
- A single `emotion_label` + `emotion_confidence` (backward compatible)
- A full `emotion_distribution` dict preserving information across all
  interview-context labels via probability-weighted remapping.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config.settings import (
    EMOTION_LABEL_MAP,
    EMOTION_WEIGHT_MAP,
    VALID_EMOTION_LABELS,
)


def _empty_emotion_result() -> dict[str, Any]:
    """Default output when no face or analysis fails."""
    return {
        "emotion_label": "blank",
        "emotion_confidence": 0.0,
        "emotion_distribution": {label: 0.0 for label in VALID_EMOTION_LABELS},
    }


class EmotionAnalyzer:
    """DeepFace-based emotion detection — model loaded once at init."""

    def __init__(self) -> None:
        # DeepFace imported lazily to avoid heavy import at module load
        self._backend = "opencv"

    def process_frame(
        self,
        frame: np.ndarray,
        face_detected: bool = True,
    ) -> dict[str, Any]:
        """
        Classify emotion in a BGR frame.

        Args:
            frame: BGR image from OpenCV.
            face_detected: Whether face_mesh found a face in this frame.
                When False, skips DeepFace entirely to avoid classifying
                background noise as emotion.

        Returns:
            Dict with emotion_label, emotion_confidence, and
            emotion_distribution (probability across all interview labels).
        """
        # P2 — Guard against no-face: DeepFace with enforce_detection=False
        # will analyze random regions if no face is present, producing
        # garbage emotion labels.  Skip when face_mesh says no face.
        if not face_detected:
            return _empty_emotion_result()

        import os
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        from deepface import DeepFace

        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
                detector_backend=self._backend,
            )
            if isinstance(result, list):
                result = result[0]

            raw_emotions: dict[str, float] = result.get("emotion", {})
            if not raw_emotions:
                return _empty_emotion_result()

            return _build_emotion_output(raw_emotions)
        except Exception:
            return _empty_emotion_result()


def _build_emotion_output(raw_emotions: dict[str, float]) -> dict[str, Any]:
    """Convert DeepFace raw emotion probabilities to interview-context output.

    Produces both a single label (backward compat) and a full distribution
    that preserves information lost by the discrete label map.
    """
    # Normalize raw DeepFace percentages to 0-1 probabilities
    total = sum(raw_emotions.values())
    if total < 1e-6:
        return _empty_emotion_result()

    probs = {k.lower(): v / total for k, v in raw_emotions.items()}

    # Build interview-context distribution via weighted remapping.
    # Each DeepFace emotion distributes its probability across all
    # interview labels using EMOTION_WEIGHT_MAP, avoiding the lossy
    # 4→1 collapse of EMOTION_LABEL_MAP.
    distribution: dict[str, float] = {label: 0.0 for label in VALID_EMOTION_LABELS}

    for deepface_label, prob in probs.items():
        weights = EMOTION_WEIGHT_MAP.get(deepface_label)
        if weights:
            for interview_label, weight in weights.items():
                distribution[interview_label] += prob * weight
        else:
            # Unknown label — contribute to "blank"
            distribution["blank"] += prob

    # Dominant interview-context label = highest probability
    dominant = max(distribution, key=distribution.get)  # type: ignore[arg-type]
    confidence = distribution[dominant]

    # Ensure label is valid
    if dominant not in VALID_EMOTION_LABELS:
        dominant = "blank"

    return {
        "emotion_label": dominant,
        "emotion_confidence": float(confidence),
        "emotion_distribution": {k: round(float(v), 4) for k, v in distribution.items()},
    }


def remap_emotion_label(generic_label: str) -> str:
    """Map a generic DeepFace label to interview-context label."""
    label = EMOTION_LABEL_MAP.get(generic_label.lower(), "blank")
    return label if label in VALID_EMOTION_LABELS else "blank"
