"""
L2CS-Net gaze estimation — yaw/pitch in degrees and eye contact score.

Weights: download L2CSNet_gaze360.pkl from the official L2CS-Net repo and place in
models/ or set L2CS_WEIGHTS_PATH in .env.

Install: pip install git+https://github.com/Ahmednull/L2CS-Net.git
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from config.settings import DEVICE, L2CS_ARCH, L2CS_WEIGHTS_PATH


class GazeEstimator:
    """L2CS-Net gaze pipeline — lazy-loaded on first use."""

    def __init__(self) -> None:
        self._pipeline = None
        self._load_error: str | None = None

    def _ensure_loaded(self) -> bool:
        if self._pipeline is not None:
            return True
        if self._load_error is not None:
            return False

        weights = Path(L2CS_WEIGHTS_PATH)
        if not weights.exists():
            self._load_error = (
                f"L2CS weights not found at {weights}. "
                "Download L2CSNet_gaze360.pkl from https://github.com/Ahmednull/L2CS-Net"
            )
            return False

        try:
            import torch
            from l2cs import Pipeline, select_device

            device_str = "0" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
            device = select_device(device_str, batch_size=1)
            self._pipeline = Pipeline(
                weights=str(weights),
                arch=L2CS_ARCH,
                device=device,
            )
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def process_frame(
        self,
        frame: np.ndarray,
        head_pose: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Estimate gaze direction and eye contact score for one BGR frame.

        eye_contact_score: 1.0 when gaze is near camera center, 0.0 when looking away.
        Falls back to head-pose-based estimate if L2CS weights unavailable.
        """
        if self._ensure_loaded():
            return self._process_l2cs(frame)

        return self._fallback_from_head_pose(head_pose)

    def _process_l2cs(self, frame: np.ndarray) -> dict[str, Any]:
        results = self._pipeline.step(frame)

        if not results or not results.pitch or not results.yaw:
            return self._default_gaze()

        # L2CS returns radians; convert to degrees for schema
        pitch_rad = float(np.mean(results.pitch))
        yaw_rad = float(np.mean(results.yaw))
        pitch_deg = float(np.degrees(pitch_rad))
        yaw_deg = float(np.degrees(yaw_rad))

        eye_contact = _eye_contact_from_gaze(yaw_deg, pitch_deg)
        return {
            "gaze_vector": {"yaw": yaw_deg, "pitch": pitch_deg},
            "eye_contact_score": eye_contact,
        }

    def _fallback_from_head_pose(
        self, head_pose: dict[str, float] | None
    ) -> dict[str, Any]:
        """Approximate gaze from head pose when L2CS is unavailable."""
        if head_pose is None:
            return self._default_gaze()

        yaw = head_pose.get("yaw", 0.0)
        pitch = head_pose.get("pitch", 0.0)
        return {
            "gaze_vector": {"yaw": float(yaw), "pitch": float(pitch)},
            "eye_contact_score": _eye_contact_from_gaze(yaw, pitch),
        }

    @staticmethod
    def _default_gaze() -> dict[str, Any]:
        return {
            "gaze_vector": {"yaw": 0.0, "pitch": 0.0},
            "eye_contact_score": 0.5,
        }


def _eye_contact_from_gaze(yaw: float, pitch: float) -> float:
    """
    Map gaze angles to 0-1 eye contact score.

    Assumes camera is at frame center; score decays as |yaw| and |pitch| increase.
    """
    yaw_penalty = min(abs(yaw) / 30.0, 1.0)
    pitch_penalty = min(abs(pitch) / 25.0, 1.0)
    score = 1.0 - 0.6 * yaw_penalty - 0.4 * pitch_penalty
    return float(np.clip(score, 0.0, 1.0))
