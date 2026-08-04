"""
Module 2 Extension — Temporal Action Unit (AU) Sequence Tracking

Analyzes frame-by-frame Action Unit time-series across an interview turn to capture
dynamic micro-expressions, facial onset velocities, and temporal emotion transitions.
"""

from typing import Any, Dict, List
import numpy as np


class TemporalAUTracker:
    """
    Tracks and classifies dynamic facial micro-expressions over time.

    Instead of averaging static per-frame probabilities, this tracker evaluates
    the temporal derivatives (velocities) and variances of 15 key Action Units.
    """

    AU_NAMES = [
        "brow_inner_up", "brow_outer_up", "brow_lower",
        "eye_wide", "cheek_raise", "lid_tighten",
        "nose_wrinkle", "lip_corner_pull", "lip_corner_depress",
        "lower_lip_depress", "lip_press", "lip_pucker",
        "lip_stretch", "jaw_drop", "mouth_stretch"
    ]

    def __init__(self, fps_estimate: float = 30.0) -> None:
        self.fps_estimate = max(float(fps_estimate), 1.0)

    def extract_temporal_features(self, turn_frames: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract temporal statistical features from a sequence of frame dictionaries.
        """
        if not turn_frames or len(turn_frames) < 2:
            return {
                "au_velocity_mean": 0.0,
                "au_variance_mean": 0.0,
                "micro_expression_intensity": 0.0,
                "temporal_emotion_prediction": "blank",
                "temporal_confidence": 0.0,
            }

        # Build T x 15 matrix of AU activations
        au_matrix = []
        for frame in turn_frames:
            aus = frame.get("au_activations", {})
            row = [float(aus.get(name, 0.0)) for name in self.AU_NAMES]
            au_matrix.append(row)

        au_arr = np.array(au_matrix, dtype=np.float32)  # Shape: (T, 15)

        # Calculate temporal velocity (first derivative across frames)
        dt = 1.0 / self.fps_estimate
        velocities = np.diff(au_arr, axis=0) / dt  # Shape: (T-1, 15)

        # Mean absolute velocity across all AUs
        mean_abs_velocity = float(np.mean(np.abs(velocities)))

        # Variance across time for each AU
        au_variances = np.var(au_arr, axis=0)
        mean_variance = float(np.mean(au_variances))

        # Identify specific dynamic micro-expressions
        brow_lower_var = float(au_variances[2])
        lip_pull_var = float(au_variances[7])
        lip_press_var = float(au_variances[10])

        micro_intensity = float(np.max(np.abs(velocities)))

        # FIX H2 — Confidence values are now computed from measured evidence
        # rather than hardcoded magic numbers. Each branch scales confidence
        # by the strength of the discriminating signal.
        prediction = "blank"

        if mean_variance < 0.001 and micro_intensity < 0.5:
            prediction = "blank"
            # High confidence when evidence is unambiguously flat (near-zero variance)
            # Scale: 0.5 (minimum baseline) + boost for lower variance
            variance_flatness = 1.0 - min(mean_variance / 0.001, 1.0)
            confidence = 0.50 + 0.40 * variance_flatness

        elif brow_lower_var > 0.05 or lip_press_var > 0.05:
            if lip_pull_var > 0.03:
                prediction = "confused"  # mixed smile + brow furrow
                # Confidence scales with how strongly both signals co-occur
                signal = min((brow_lower_var + lip_pull_var) / 0.12, 1.0)
                confidence = 0.50 + 0.35 * signal
            else:
                prediction = "nervous"   # tension in brows/lips without smile
                signal = min((brow_lower_var + lip_press_var) / 0.15, 1.0)
                confidence = 0.50 + 0.35 * signal

        elif lip_pull_var > 0.04 and brow_lower_var < 0.01:
            prediction = "confident"     # steady smile, relaxed brow
            signal = min(lip_pull_var / 0.10, 1.0)
            confidence = 0.55 + 0.30 * signal

        elif mean_abs_velocity > 2.0:
            prediction = "engaged"       # animated facial expressions
            signal = min((mean_abs_velocity - 2.0) / 8.0, 1.0)
            confidence = 0.50 + 0.30 * signal

        else:
            # Default: low-activity session with no clear signal
            prediction = "blank"
            confidence = 0.40

        # Clamp confidence to [0, 1]
        confidence = float(max(0.0, min(1.0, confidence)))

        return {
            "au_velocity_mean": mean_abs_velocity,
            "au_variance_mean": mean_variance,
            "micro_expression_intensity": micro_intensity,
            "temporal_emotion_prediction": prediction,
            "temporal_confidence": confidence,
        }
