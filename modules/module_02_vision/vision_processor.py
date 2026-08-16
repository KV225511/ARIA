"""
Vision module orchestrator — parallel per-frame processing and per-turn summarization.

Runs MediaPipe (face_mesh), DeepFace (emotion), and L2CS-Net (gaze) in parallel
threads for each frame. Aggregates frame outputs into a per-turn summary at turn end.
"""

from __future__ import annotations

import concurrent.futures
from collections import Counter
from typing import Any

import numpy as np

from config.settings import VALID_EMOTION_LABELS
from modules.module_02_vision.emotion import EmotionAnalyzer
from modules.module_02_vision.face_mesh import AU_KEYS, FaceMeshAnalyzer
from modules.module_02_vision.gaze import GazeEstimator
from modules.module_02_vision.baseline import VisionBaselineManager

# Module-level singletons — loaded once, not per frame
_face_mesh: FaceMeshAnalyzer | None = None
_emotion: EmotionAnalyzer | None = None
_gaze: GazeEstimator | None = None


def _get_analyzers() -> tuple[FaceMeshAnalyzer, EmotionAnalyzer, GazeEstimator]:
    global _face_mesh, _emotion, _gaze
    if _face_mesh is None:
        _face_mesh = FaceMeshAnalyzer()
    if _emotion is None:
        _emotion = EmotionAnalyzer()
    if _gaze is None:
        _gaze = GazeEstimator()
    return _face_mesh, _emotion, _gaze


def _empty_emotion_distribution() -> dict[str, float]:
    return {label: 0.0 for label in VALID_EMOTION_LABELS}


class VisionProcessor:
    """Process webcam frames and produce per-turn vision summaries."""

    def __init__(self) -> None:
        self._face_mesh, self._emotion, self._gaze = _get_analyzers()
        self._turn_frames: list[dict[str, Any]] = []
        self._turn_start_ms: float = 0.0
        # P2 — Reuse a single thread pool across frames instead of
        # creating/destroying one per 500ms frame.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.baseline_manager = VisionBaselineManager()

    def start_turn(self, timestamp_ms: float = 0.0) -> None:
        """Reset frame buffer for a new conversational turn."""
        self._turn_frames = []
        self._turn_start_ms = timestamp_ms
        # P1 — Clear blink state so a blink straddling two turns
        # doesn't produce a phantom cross-turn blink.
        self._face_mesh.reset_state()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float = 0.0,
    ) -> dict[str, Any] | None:
        """
        Process one BGR frame with parallel analyzers.

        Returns full per-frame schema dict, or None if no face detected.
        """
        # Run face mesh and emotion in parallel.
        # Gaze runs after mesh completes (needs head_pose for fallback).
        mesh_future = self._executor.submit(
            self._face_mesh.process_frame, frame, timestamp_ms
        )

        # We don't know if a face exists yet.  Submit emotion speculatively;
        # if no face, we'll discard the result.
        emotion_future = self._executor.submit(
            self._emotion.process_frame, frame, True  # optimistic face_detected
        )

        mesh_result = mesh_future.result()
        face_detected = mesh_result is not None

        if not face_detected:
            # Discard speculative emotion result (DeepFace would
            # have analysed background noise) and return None.
            try:
                emotion_future.result()  # consume to avoid dangling future
            except Exception:
                pass
            return None

        # FIX C1 — Wrap emotion result in try/except. Previously the second
        # `if not face_detected` block (dead code) was the intended fallback
        # but was unreachable. Now if DeepFace raises (OOM, crash, timeout),
        # we safely fall back to a blank emotion result instead of propagating.
        try:
            emotion_result = emotion_future.result()
        except Exception:
            emotion_result = {
                "emotion_label": "blank",
                "emotion_confidence": 0.0,
                "emotion_distribution": _empty_emotion_distribution(),
            }

        head_pose = mesh_result["head_pose"]
        gaze_future = self._executor.submit(
            self._gaze.process_frame, frame, head_pose
        )
        gaze_result = gaze_future.result()

        frame_output = {
            # FIX H3 — Strip raw landmarks from stored frames to prevent unbounded
            # memory accumulation (478 × 3 floats per frame × N frames per session).
            # Landmarks are only needed by FaceMeshAnalyzer internally per frame.
            "au_activations": mesh_result["au_activations"],
            "emotion_label": emotion_result["emotion_label"],
            "emotion_confidence": emotion_result["emotion_confidence"],
            "emotion_distribution": emotion_result.get(
                "emotion_distribution", _empty_emotion_distribution()
            ),
            "gaze_vector": gaze_result["gaze_vector"],
            "eye_contact_score": gaze_result["eye_contact_score"],
            "head_pose": mesh_result["head_pose"],
            "blink_detected": mesh_result["blink_detected"],
            "blink_duration_ms": mesh_result["blink_duration_ms"],
        }
        self._turn_frames.append(frame_output)
        return frame_output

    def summarize_turn(
        self,
        turn_duration_ms: float | None = None,
        candidate_id: str | None = None,
        turn_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate all frames collected during the current turn.

        Args:
            turn_duration_ms: Duration of the turn in ms (for blink_rate).
                If None, estimated from frame count * 500ms.
            candidate_id: Unique candidate ID for baseline deviation tracking.
            turn_id: Current conversational turn number (1-indexed).
        """
        if not self._turn_frames:
            return _empty_turn_summary()

        labels = [f["emotion_label"] for f in self._turn_frames]
        emotion_label = Counter(labels).most_common(1)[0][0]

        au_means = {}
        for key in AU_KEYS:
            # FIX M2 — Guard against missing AU key in any frame dict.
            # A partially populated au_activations dict would raise KeyError here.
            au_means[key] = float(
                np.mean([f["au_activations"].get(key, 0.0) for f in self._turn_frames])
            )

        # Average emotion distribution across all frames in the turn
        dist_means: dict[str, float] = {label: 0.0 for label in VALID_EMOTION_LABELS}
        for label in VALID_EMOTION_LABELS:
            values = [
                f.get("emotion_distribution", {}).get(label, 0.0)
                for f in self._turn_frames
            ]
            dist_means[label] = round(float(np.mean(values)), 4)

        gaze_yaw = float(
            np.mean([f["gaze_vector"]["yaw"] for f in self._turn_frames])
        )
        gaze_pitch = float(
            np.mean([f["gaze_vector"]["pitch"] for f in self._turn_frames])
        )
        eye_contact = float(
            np.mean([f["eye_contact_score"] for f in self._turn_frames])
        )

        head_roll = float(np.mean([f["head_pose"]["roll"] for f in self._turn_frames]))
        head_pitch = float(np.mean([f["head_pose"]["pitch"] for f in self._turn_frames]))
        head_yaw = float(np.mean([f["head_pose"]["yaw"] for f in self._turn_frames]))

        blink_count = sum(1 for f in self._turn_frames if f["blink_detected"])
        if turn_duration_ms is None:
            turn_duration_ms = len(self._turn_frames) * 500.0
        duration_min = max(turn_duration_ms / 60000.0, 1e-6)
        blink_rate = blink_count / duration_min

        summary = {
            "emotion_label": emotion_label,
            "emotion_distribution": dist_means,
            "au_activations": au_means,
            "gaze_vector": {"yaw": gaze_yaw, "pitch": gaze_pitch},
            "eye_contact_score": eye_contact,
            "head_pose": {
                "roll": head_roll,
                "pitch": head_pitch,
                "yaw": head_yaw,
            },
            "blink_rate": float(blink_rate),
            "au_deviations": None,
            "eye_contact_deviation": None,
            "blink_rate_deviation": None,
        }

        if candidate_id is not None and turn_id is not None:
            summary = self.baseline_manager.update_with_baseline(
                candidate_id=candidate_id, turn_id=turn_id, vision_summary=summary
            )

        return summary

    def get_turn_frames(self) -> list[dict]:
        """FIX C4 — Public accessor for the current turn's processed frame list.

        Use this instead of accessing the private `_turn_frames` attribute directly,
        so external tools remain stable if the internal attribute is ever refactored.
        Returns a copy to prevent external mutation of internal state.
        """
        return list(self._turn_frames)

    def close(self) -> None:
        """Shut down the thread pool executor."""
        self._executor.shutdown(wait=False)


def _empty_turn_summary() -> dict[str, Any]:
    return {
        "emotion_label": "blank",
        "emotion_distribution": {label: 0.0 for label in VALID_EMOTION_LABELS},
        "au_activations": {key: 0.0 for key in AU_KEYS},
        "gaze_vector": {"yaw": 0.0, "pitch": 0.0},
        "eye_contact_score": 0.0,
        "head_pose": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "blink_rate": 0.0,
        "au_deviations": None,
        "eye_contact_deviation": None,
        "blink_rate_deviation": None,
    }


# Expected keys in per-frame output (for tests and downstream validation)
FRAME_OUTPUT_KEYS = frozenset(
    {
        "au_activations",
        "emotion_label",
        "emotion_confidence",
        "emotion_distribution",
        "gaze_vector",
        "eye_contact_score",
        "head_pose",
        "blink_detected",
        "blink_duration_ms",
    }
)
