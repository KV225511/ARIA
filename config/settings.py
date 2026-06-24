"""Global constants, model names, and paths for ARIA."""

import os
from pathlib import Path

# Project root (ARIA/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Device and model settings
MODEL_WHISPER = "large-v3"
WHISPER_COMPUTE_TYPE = "int8"
DEVICE = "cuda"

# Audio / video capture
VIDEO_FRAME_INTERVAL_MS = 500
AUDIO_SAMPLE_RATE = 16000
MFCC_COEFFICIENTS = 13

# Interview session
BASELINE_TURNS = 2
TERMINATION_ENTROPY_THRESHOLD = 0.3

# DeepFace emotion label remapping (generic -> interview-context)
EMOTION_LABEL_MAP = {
    "happy": "engaged",
    "neutral": "blank",
    "fear": "nervous",
    "surprise": "confused",
    "sad": "nervous",
    "disgust": "nervous",
    "angry": "nervous",
}

VALID_EMOTION_LABELS = frozenset(
    {"engaged", "confused", "nervous", "confident", "blank"}
)

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# Model weight paths (override via .env)
MODELS_DIR = PROJECT_ROOT / "models"
L2CS_WEIGHTS_PATH = os.getenv(
    "L2CS_WEIGHTS_PATH",
    str(MODELS_DIR / "L2CSNet_gaze360.pkl"),
)
L2CS_ARCH = "ResNet50"

# MediaPipe
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# Blink detection (Eye Aspect Ratio threshold)
EAR_BLINK_THRESHOLD = 0.21
