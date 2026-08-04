"""
Module 11 — Anti-Gaming: Note Reading Detection via Gaze Patterns

Detects horizontal left-to-right eye saccades across consecutive frames
that indicate the candidate is reading from notes or a second screen.

Natural thinking gaze patterns are random or upward; reading patterns
show sustained, monotonic horizontal sweeps for >2.0 seconds.

Input:
    List of per-frame gaze dicts from Module 2, each containing:
    {
        "gaze_vector": {"yaw": float, "pitch": float},   # degrees
        "timestamp_ms": float,
    }

Output:
    {
        "flag": "note_reading",
        "confidence": float,        # 0.0–1.0
        "evidence": {
            "sweep_count": int,
            "max_sweep_duration_ms": float,
            "sweep_segments": list,
        }
    }

Owner: Krissh
"""

from __future__ import annotations

from typing import Any


# ── DETECTION THRESHOLDS ───────────────────────────────────────────────────

# Minimum duration (ms) for a horizontal sweep to be flagged as reading
MIN_SWEEP_DURATION_MS = 2000.0

# Gaze yaw must change by at least this many degrees across the sweep
# to qualify as a meaningful horizontal movement (not just jitter)
MIN_YAW_RANGE_DEGREES = 8.0

# Maximum allowed pitch variation (degrees) during a horizontal sweep.
# Reading produces flat horizontal motion; thinking produces vertical drift.
MAX_PITCH_VARIATION_DEGREES = 10.0

# Maximum allowed yaw reversal per step (degrees). True reading sweeps
# are monotonic; frequent direction changes indicate natural looking around.
MAX_YAW_REVERSAL_DEGREES = 3.0

# Fraction of steps in a sweep that may violate monotonicity before the
# sweep is rejected. Allows for 1-2 noisy frames in an otherwise clean sweep.
MAX_REVERSAL_RATIO = 0.25


class GazeScanner:
    """
    Detects note reading behavior by analyzing gaze vector sequences
    for sustained horizontal sweeps.

    Stateless per-call — all state is derived from the input frame list.

    Usage:
        scanner = GazeScanner()
        result = scanner.detect(gaze_frames)
    """

    def __init__(
        self,
        min_sweep_duration_ms: float = MIN_SWEEP_DURATION_MS,
        min_yaw_range: float = MIN_YAW_RANGE_DEGREES,
        max_pitch_variation: float = MAX_PITCH_VARIATION_DEGREES,
        max_reversal_ratio: float = MAX_REVERSAL_RATIO,
    ) -> None:
        self.min_sweep_duration_ms = min_sweep_duration_ms
        self.min_yaw_range = min_yaw_range
        self.max_pitch_variation = max_pitch_variation
        self.max_reversal_ratio = max_reversal_ratio

    def detect(self, gaze_frames: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyze a sequence of gaze frames for reading-like sweep patterns.

        Args:
            gaze_frames: List of per-frame gaze dicts. Each must contain:
                - "gaze_vector": {"yaw": float, "pitch": float}
                - "timestamp_ms": float

        Returns:
            Detection result dict with flag, confidence, and evidence.
        """
        if not gaze_frames or len(gaze_frames) < 3:
            return _no_detection()

        # Extract yaw/pitch/timestamp sequences
        yaw_seq: list[float] = []
        pitch_seq: list[float] = []
        time_seq: list[float] = []

        for frame in gaze_frames:
            gaze = frame.get("gaze_vector", {})
            yaw = gaze.get("yaw")
            pitch = gaze.get("pitch")
            ts = frame.get("timestamp_ms")

            if yaw is None or pitch is None or ts is None:
                continue

            yaw_seq.append(float(yaw))
            pitch_seq.append(float(pitch))
            time_seq.append(float(ts))

        if len(yaw_seq) < 3:
            return _no_detection()

        # ── Detect horizontal sweep segments ───────────────────────────────
        sweeps = self._find_sweep_segments(yaw_seq, pitch_seq, time_seq)

        if not sweeps:
            return _no_detection()

        # ── Compute confidence from sweep evidence ─────────────────────────
        max_duration = max(s["duration_ms"] for s in sweeps)
        total_sweep_time = sum(s["duration_ms"] for s in sweeps)
        total_session_time = time_seq[-1] - time_seq[0] if len(time_seq) > 1 else 1.0

        # Confidence factors:
        # 1. Longest sweep duration relative to threshold (capped at 2x threshold)
        duration_factor = min(max_duration / self.min_sweep_duration_ms, 2.0) / 2.0

        # 2. Total sweep time as fraction of total turn time
        coverage_factor = min(total_sweep_time / max(total_session_time, 1.0), 1.0)

        # 3. Number of distinct sweeps (multiple reading passes → higher confidence)
        count_factor = min(len(sweeps) / 3.0, 1.0)

        confidence = 0.4 * duration_factor + 0.35 * coverage_factor + 0.25 * count_factor

        return {
            "flag": "note_reading",
            "confidence": _clamp(confidence, 0.0, 1.0),
            "evidence": {
                "sweep_count": len(sweeps),
                "max_sweep_duration_ms": max_duration,
                "sweep_segments": sweeps,
            },
        }

    def _find_sweep_segments(
        self,
        yaw_seq: list[float],
        pitch_seq: list[float],
        time_seq: list[float],
    ) -> list[dict[str, Any]]:
        """
        Identify contiguous segments of monotonic horizontal gaze movement
        with constrained vertical variation.

        A valid sweep:
        - Lasts >= min_sweep_duration_ms
        - Has yaw range >= min_yaw_range degrees
        - Has pitch variation <= max_pitch_variation degrees
        - Has <= max_reversal_ratio fraction of non-monotonic steps
        """
        sweeps: list[dict[str, Any]] = []
        n = len(yaw_seq)
        i = 0

        while i < n - 1:
            # Start a candidate sweep at frame i
            start_idx = i
            direction = None  # "left_to_right" or "right_to_left"
            reversals = 0
            steps = 0

            j = i + 1
            while j < n:
                yaw_delta = yaw_seq[j] - yaw_seq[j - 1]
                steps += 1

                # Determine initial direction from first significant movement
                if direction is None:
                    if abs(yaw_delta) > 0.5:  # Ignore sub-degree noise
                        direction = "left_to_right" if yaw_delta > 0 else "right_to_left"
                    j += 1
                    continue

                # Check if this step is consistent with sweep direction
                is_consistent = (
                    (direction == "left_to_right" and yaw_delta >= -MAX_YAW_REVERSAL_DEGREES)
                    or (direction == "right_to_left" and yaw_delta <= MAX_YAW_REVERSAL_DEGREES)
                )

                if not is_consistent:
                    reversals += 1

                # Check if reversal ratio exceeded — break sweep
                if steps > 0 and (reversals / steps) > self.max_reversal_ratio:
                    break

                j += 1

            end_idx = j - 1

            # Validate the candidate sweep
            if end_idx > start_idx and direction is not None:
                duration = time_seq[end_idx] - time_seq[start_idx]
                yaw_range = abs(yaw_seq[end_idx] - yaw_seq[start_idx])
                pitch_values = pitch_seq[start_idx:end_idx + 1]
                pitch_var = max(pitch_values) - min(pitch_values)

                if (
                    duration >= self.min_sweep_duration_ms
                    and yaw_range >= self.min_yaw_range
                    and pitch_var <= self.max_pitch_variation
                ):
                    sweeps.append({
                        "start_ms": time_seq[start_idx],
                        "end_ms": time_seq[end_idx],
                        "duration_ms": duration,
                        "yaw_range": yaw_range,
                        "direction": direction,
                    })

            # Move past this segment
            i = max(i + 1, end_idx)

        return sweeps


# ── UTILITY FUNCTIONS ──────────────────────────────────────────────────────

def _no_detection() -> dict[str, Any]:
    """Return a clean (no flag) detection result."""
    return {
        "flag": "note_reading",
        "confidence": 0.0,
        "evidence": {
            "sweep_count": 0,
            "max_sweep_duration_ms": 0.0,
            "sweep_segments": [],
        },
    }


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to [low, high]."""
    return max(low, min(high, value))
