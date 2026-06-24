"""
MediaPipe Face Mesh — 468 landmarks, Action Unit activations, head pose, blink detection.

Action Units are estimated geometrically from landmark distances and angles.
MediaPipe does not output AUs natively; we derive them from facial geometry.
"""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from config.settings import (
    EAR_BLINK_THRESHOLD,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
)

# MediaPipe Face Mesh landmark indices (subset used for AU / pose / blink)
_LANDMARK = mp.solutions.face_mesh.FaceMesh

# Eye landmarks for Eye Aspect Ratio (EAR) blink detection
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)

# Landmark groups for AU estimation
_INNER_BROW = (107, 336)
_OUTER_BROW = (70, 300)
_BROW_CENTER = (9, 10)
_UPPER_LID = (159, 386)
_CHEEK = (50, 280)
_LIP_CORNER = (61, 291)
_UPPER_LIP = (13, 14)
_LOWER_LIP = (17, 18)
_CHIN = (152,)
_NOSE_TIP = (1,)

AU_KEYS = (
    "AU1", "AU2", "AU4", "AU6", "AU12", "AU15", "AU17", "AU23", "AU25"
)


class FaceMeshAnalyzer:
    """MediaPipe Face Mesh processor — load once, reuse per frame."""

    def __init__(self) -> None:
        self._mesh = _LANDMARK(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        )
        self._prev_ear: float | None = None
        self._blink_start: float | None = None
        self._frame_time_ms: float = 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float = 0.0,
    ) -> dict[str, Any] | None:
        """
        Process one BGR frame.

        Returns None if no face is detected (caller should skip frame in summary).
        """
        self._frame_time_ms = timestamp_ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        landmarks = np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in face.landmark],
            dtype=np.float32,
        )

        au_activations = _estimate_au_activations(landmarks)
        head_pose = _estimate_head_pose(landmarks, w, h)
        ear = _eye_aspect_ratio(landmarks)
        blink_detected, blink_duration_ms = self._detect_blink(ear, timestamp_ms)

        return {
            "landmarks": landmarks,
            "au_activations": au_activations,
            "head_pose": head_pose,
            "blink_detected": blink_detected,
            "blink_duration_ms": blink_duration_ms,
            "ear": ear,
        }

    def _detect_blink(
        self, ear: float, timestamp_ms: float
    ) -> tuple[bool, float]:
        """Detect blink via Eye Aspect Ratio crossing threshold."""
        blink_detected = False
        blink_duration_ms = 0.0

        if self._prev_ear is not None:
            if self._prev_ear >= EAR_BLINK_THRESHOLD and ear < EAR_BLINK_THRESHOLD:
                self._blink_start = timestamp_ms
            elif (
                self._blink_start is not None
                and self._prev_ear < EAR_BLINK_THRESHOLD
                and ear >= EAR_BLINK_THRESHOLD
            ):
                blink_detected = True
                blink_duration_ms = timestamp_ms - self._blink_start
                self._blink_start = None

        self._prev_ear = ear
        return blink_detected, blink_duration_ms

    def close(self) -> None:
        self._mesh.close()


def _pt(landmarks: np.ndarray, idx: int) -> np.ndarray:
    return landmarks[idx, :2]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _normalize(value: float, low: float, high: float) -> float:
    """Clamp and scale to 0-1 activation range."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _estimate_au_activations(landmarks: np.ndarray) -> dict[str, float]:
    """
    Geometric AU estimation from 468 landmarks.

    Values are normalized 0.0-1.0 activations, not FACS-certified intensities.
    Calibrated for relative change detection across a turn, not absolute FACS scoring.
    """
    inner_brow_l, inner_brow_r = _pt(landmarks, 107), _pt(landmarks, 336)
    outer_brow_l, outer_brow_r = _pt(landmarks, 70), _pt(landmarks, 300)
    brow_mid = (_pt(landmarks, 9) + _pt(landmarks, 10)) / 2
    upper_lid_l, upper_lid_r = _pt(landmarks, 159), _pt(landmarks, 386)
    cheek_l, cheek_r = _pt(landmarks, 50), _pt(landmarks, 280)
    lip_l, lip_r = _pt(landmarks, 61), _pt(landmarks, 291)
    upper_lip = (_pt(landmarks, 13) + _pt(landmarks, 14)) / 2
    lower_lip = (_pt(landmarks, 17) + _pt(landmarks, 18)) / 2
    chin = _pt(landmarks, 152)
    nose = _pt(landmarks, 1)

    interocular = _dist(_pt(landmarks, 33), _pt(landmarks, 263))
    if interocular < 1e-6:
        interocular = 1.0

    # AU1 — inner brow raise: inner brow moves up relative to upper lid
    au1_l = _dist(inner_brow_l, upper_lid_l) / interocular
    au1_r = _dist(inner_brow_r, upper_lid_r) / interocular
    au1 = _normalize((au1_l + au1_r) / 2, 0.08, 0.18)

    # AU2 — outer brow raise
    au2_l = _dist(outer_brow_l, upper_lid_l) / interocular
    au2_r = _dist(outer_brow_r, upper_lid_r) / interocular
    au2 = _normalize((au2_l + au2_r) / 2, 0.10, 0.22)

    # AU4 — brow lowerer: brow close to eye (inverse of raise)
    au4 = 1.0 - _normalize(_dist(brow_mid, (upper_lid_l + upper_lid_r) / 2) / interocular, 0.06, 0.14)

    # AU6 — cheek raiser: cheek lifts toward eye
    au6_l = _dist(cheek_l, upper_lid_l) / interocular
    au6_r = _dist(cheek_r, upper_lid_r) / interocular
    au6 = _normalize(0.20 - (au6_l + au6_r) / 2, 0.0, 0.08)

    # AU12 — lip corner puller (smile): corners move up/out
    mouth_width = _dist(lip_l, lip_r) / interocular
    mouth_center_y = (lip_l[1] + lip_r[1]) / 2
    nose_y = nose[1]
    corner_lift = (nose_y - mouth_center_y) / interocular
    au12 = _normalize(mouth_width * 0.5 + corner_lift, 0.15, 0.45)

    # AU15 — lip corner depressor
    au15 = _normalize(0.35 - corner_lift, 0.0, 0.15)

    # AU17 — chin raiser
    au17 = _normalize(_dist(chin, lower_lip) / interocular, 0.04, 0.12)

    # AU23 — lip tightener: reduced lip separation
    lip_sep = _dist(upper_lip, lower_lip) / interocular
    au23 = _normalize(0.08 - lip_sep, 0.0, 0.06)

    # AU25 — lips part
    au25 = _normalize(lip_sep, 0.02, 0.10)

    return {
        "AU1": au1,
        "AU2": au2,
        "AU4": au4,
        "AU6": au6,
        "AU12": au12,
        "AU15": au15,
        "AU17": au17,
        "AU23": au23,
        "AU25": au25,
    }


def _eye_aspect_ratio(landmarks: np.ndarray) -> float:
    """Average EAR across both eyes."""
    ears = []
    for indices in (_LEFT_EYE, _RIGHT_EYE):
        p = [_pt(landmarks, i) for i in indices]
        vertical = _dist(p[1], p[5]) + _dist(p[2], p[4])
        horizontal = _dist(p[0], p[3])
        if horizontal > 1e-6:
            ears.append(vertical / (2.0 * horizontal))
    return float(np.mean(ears)) if ears else 0.3


def _estimate_head_pose(
    landmarks: np.ndarray, frame_w: int, frame_h: int
) -> dict[str, float]:
    """
    Estimate head pose (roll, pitch, yaw) in degrees using solvePnP.

    Uses six canonical 3D face model points matched to MediaPipe landmarks.
    """
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype=np.float64,
    )
    image_points = np.array(
        [
            landmarks[1, :2],
            landmarks[152, :2],
            landmarks[263, :2],
            landmarks[33, :2],
            landmarks[287, :2],
            landmarks[57, :2],
        ],
        dtype=np.float64,
    )

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    if sy < 1e-6:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = 0.0
    else:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(-rotation_matrix[2, 0], sy)
        roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return {
        "roll": float(np.degrees(roll)),
        "pitch": float(np.degrees(pitch)),
        "yaw": float(np.degrees(yaw)),
    }
