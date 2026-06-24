"""
DeepFace emotion classifier with interview-context label remapping.

Generic DeepFace labels (happy, neutral, fear, ...) are mapped to:
engaged / confused / nervous / confident / blank
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config.settings import EMOTION_LABEL_MAP, VALID_EMOTION_LABELS


class EmotionAnalyzer:
    """DeepFace-based emotion detection — model loaded once at init."""

    def __init__(self) -> None:
        # DeepFace imported lazily to avoid heavy import at module load
        self._backend = "opencv"

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Classify emotion in a BGR frame.

        Returns interview-context label and confidence.
        Falls back to blank/0.0 if face analysis fails.
        """
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

            emotions: dict[str, float] = result.get("emotion", {})
            if not emotions:
                return {"emotion_label": "blank", "emotion_confidence": 0.0}

            dominant = max(emotions, key=emotions.get)
            confidence = float(emotions[dominant]) / 100.0
            mapped = EMOTION_LABEL_MAP.get(dominant.lower(), "blank")

            # High-confidence neutral with slight positive valence -> engaged
            if mapped == "blank" and confidence < 0.5:
                mapped = "blank"
            elif mapped == "blank" and confidence >= 0.7:
                mapped = "confident"

            if mapped not in VALID_EMOTION_LABELS:
                mapped = "blank"

            return {
                "emotion_label": mapped,
                "emotion_confidence": confidence,
            }
        except Exception:
            return {"emotion_label": "blank", "emotion_confidence": 0.0}


def remap_emotion_label(generic_label: str) -> str:
    """Map a generic DeepFace label to interview-context label."""
    label = EMOTION_LABEL_MAP.get(generic_label.lower(), "blank")
    return label if label in VALID_EMOTION_LABELS else "blank"
