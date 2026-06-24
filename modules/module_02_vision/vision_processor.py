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

from modules.module_02_vision.emotion import EmotionAnalyzer
from modules.module_02_vision.face_mesh import AU_KEYS, FaceMeshAnalyzer
from modules.module_02_vision.gaze import GazeEstimator

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


class VisionProcessor:
    """Process webcam frames and produce per-turn vision summaries."""

    def __init__(self) -> None:
        self._face_mesh, self._emotion, self._gaze = _get_analyzers()
        self._turn_frames: list[dict[str, Any]] = []
        self._turn_start_ms: float = 0.0

    def start_turn(self, timestamp_ms: float = 0.0) -> None:
        """Reset frame buffer for a new conversational turn."""
        self._turn_frames = []
        self._turn_start_ms = timestamp_ms

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float = 0.0,
    ) -> dict[str, Any] | None:
        """
        Process one BGR frame with parallel analyzers.

        Returns full per-frame schema dict, or None if no face detected.
        """
        mesh_result: dict[str, Any] | None = None
        emotion_result: dict[str, Any] = {"emotion_label": "blank", "emotion_confidence": 0.0}
        gaze_result: dict[str, Any] = {
            "gaze_vector": {"yaw": 0.0, "pitch": 0.0},
            "eye_contact_score": 0.5,
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            mesh_future = executor.submit(
                self._face_mesh.process_frame, frame, timestamp_ms
            )
            emotion_future = executor.submit(self._emotion.process_frame, frame)

            mesh_result = mesh_future.result()
            emotion_result = emotion_future.result()

            head_pose = mesh_result["head_pose"] if mesh_result else None
            gaze_future = executor.submit(
                self._gaze.process_frame, frame, head_pose
            )
            gaze_result = gaze_future.result()

        if mesh_result is None:
            return None

        frame_output = {
            "landmarks": mesh_result["landmarks"],
            "au_activations": mesh_result["au_activations"],
            "emotion_label": emotion_result["emotion_label"],
            "emotion_confidence": emotion_result["emotion_confidence"],
            "gaze_vector": gaze_result["gaze_vector"],
            "eye_contact_score": gaze_result["eye_contact_score"],
            "head_pose": mesh_result["head_pose"],
            "blink_detected": mesh_result["blink_detected"],
            "blink_duration_ms": mesh_result["blink_duration_ms"],
        }
        self._turn_frames.append(frame_output)
        return frame_output

    def summarize_turn(self, turn_duration_ms: float | None = None) -> dict[str, Any]:
        """
        Aggregate all frames collected during the current turn.

        Args:
            turn_duration_ms: Duration of the turn in ms (for blink_rate).
                If None, estimated from frame count * 500ms.
        """
        if not self._turn_frames:
            return _empty_turn_summary()

        labels = [f["emotion_label"] for f in self._turn_frames]
        emotion_label = Counter(labels).most_common(1)[0][0]

        au_means = {}
        for key in AU_KEYS:
            au_means[key] = float(
                np.mean([f["au_activations"][key] for f in self._turn_frames])
            )

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

        return {
            "emotion_label": emotion_label,
            "au_activations": au_means,
            "gaze_vector": {"yaw": gaze_yaw, "pitch": gaze_pitch},
            "eye_contact_score": eye_contact,
            "head_pose": {
                "roll": head_roll,
                "pitch": head_pitch,
                "yaw": head_yaw,
            },
            "blink_rate": float(blink_rate),
        }


def _empty_turn_summary() -> dict[str, Any]:
    return {
        "emotion_label": "blank",
        "au_activations": {key: 0.0 for key in AU_KEYS},
        "gaze_vector": {"yaw": 0.0, "pitch": 0.0},
        "eye_contact_score": 0.0,
        "head_pose": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "blink_rate": 0.0,
    }


# Expected keys in per-frame output (for tests and downstream validation)
FRAME_OUTPUT_KEYS = frozenset(
    {
        "landmarks",
        "au_activations",
        "emotion_label",
        "emotion_confidence",
        "gaze_vector",
        "eye_contact_score",
        "head_pose",
        "blink_detected",
        "blink_duration_ms",
    }
)
