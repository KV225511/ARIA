"""Global constants, model names, and paths for ARIA."""

import os
from pathlib import Path

# DeepFace / TensorFlow compatibility: use legacy Keras 2 so LocallyConnected2D exists
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

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

# ── Security limits ────────────────────────────────────────────────────────
MAX_AUDIO_DURATION_S = 600          # 10 minutes — reject anything longer
MAX_FRAME_RESOLUTION = (3840, 2160) # 4K — reject frames larger than this
ALLOWED_AUDIO_DIR = PROJECT_ROOT  # transcribe_file_sync restricted to project directory

# Interview session
BASELINE_TURNS = 2
TERMINATION_ENTROPY_THRESHOLD = 0.3

# DeepFace emotion label remapping (generic -> interview-context)
# Kept for backward compatibility and simple label lookups.
EMOTION_LABEL_MAP = {
    "happy": "engaged",
    "neutral": "blank",
    "fear": "nervous",
    "surprise": "confused",
    "sad": "nervous",
    "disgust": "nervous",
    "angry": "nervous",
}

# Probability-preserving emotion weight map.
# Each DeepFace label distributes its probability across interview-context
# labels, avoiding the lossy 4→1 collapse of EMOTION_LABEL_MAP.
# Row values must sum to 1.0.
#                          engaged  confused  nervous  confident  blank
EMOTION_WEIGHT_MAP = {
    "happy":     {"engaged": 0.85, "confused": 0.0,  "nervous": 0.0,  "confident": 0.10, "blank": 0.05},
    "neutral":   {"engaged": 0.05, "confused": 0.0,  "nervous": 0.05, "confident": 0.30, "blank": 0.60},
    "fear":      {"engaged": 0.0,  "confused": 0.10, "nervous": 0.80, "confident": 0.0,  "blank": 0.10},
    "surprise":  {"engaged": 0.10, "confused": 0.65, "nervous": 0.15, "confident": 0.0,  "blank": 0.10},
    "sad":       {"engaged": 0.0,  "confused": 0.05, "nervous": 0.60, "confident": 0.0,  "blank": 0.35},
    "disgust":   {"engaged": 0.0,  "confused": 0.10, "nervous": 0.50, "confident": 0.0,  "blank": 0.40},
    "angry":     {"engaged": 0.0,  "confused": 0.05, "nervous": 0.40, "confident": 0.10, "blank": 0.45},
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
FACE_LANDMARKER_MODEL_PATH = str(MODELS_DIR / "face_landmarker.task")
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# Blink detection (Eye Aspect Ratio threshold)
EAR_BLINK_THRESHOLD = 0.21
