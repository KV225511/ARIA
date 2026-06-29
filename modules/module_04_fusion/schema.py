"""
Module 4 — Fusion Schema

This file defines the fixed feature order for the ARIA turn-level fused vector.
Every fused vector must follow this exact order.

Rule:
    len(fused_vector) == len(FULL_FEATURE_SCHEMA)
"""


# ── SEMANTIC / TEXT FEATURES ───────────────────────────────────────────────

SEMANTIC_FEATURES = [
    "semantic_similarity",
    "question_relevance",
    "answer_completeness",
    "semantic_confidence",
    "predicted_competency_score",
]


# ── VISION FEATURES ────────────────────────────────────────────────────────

VISION_FEATURES = [
    "emotion_confidence",
    "eye_contact_ratio",
    "blink_rate",
    "head_movement_score",
    "engagement_score",
    "dominant_emotion_score",
]


# ── PROSODY FEATURES ───────────────────────────────────────────────────────

PROSODY_FEATURES = [
    "pitch_mean",
    "pitch_variance",
    "pitch_range",
    "speech_rate",
    "pause_count",
    "pause_total_duration_ms",
    "energy_mean",
    "jitter",
    "shimmer",
    "pitch_deviation",
    "rate_deviation",
    "energy_deviation",
    "speech_to_silence_ratio",
]


# ── AUDIO EMOTION / COGNITIVE LOAD FEATURES ────────────────────────────────

AUDIO_EMOTION_FEATURES = [
    "audio_emotion_confidence",
    "audio_emotion_score",
    "cognitive_load_score",
    "audio_stress_score",
]


# ── LABEL TO NUMERIC SCORE MAPPINGS ────────────────────────────────────────

COMPETENCY_SCORE_MAP = {
    "beginner": 0.0,
    "mid": 0.5,
    "intermediate": 0.5,
    "expert": 1.0,
}


VISION_EMOTION_SCORE_MAP = {
    "blank": 0.0,
    "neutral": 0.1,
    "nervous": 0.25,
    "confused": 0.40,
    "engaged": 0.75,
    "confident": 1.0,
}


COGNITIVE_LOAD_SCORE_MAP = {
    "low_load": 0.0,
    "medium_load": 0.5,
    "mid_load": 0.5,
    "high_load": 1.0,
}


AUDIO_EMOTION_SCORE_MAP = {
    "calm": 0.1,
    "neutral": 0.2,
    "happy": 0.3,
    "surprised": 0.6,
    "sad": 0.7,
    "disgust": 0.8,
    "fearful": 0.85,
    "fear": 0.85,
    "angry": 0.9,
}


# ── FINAL FULL FEATURE ORDER ───────────────────────────────────────────────

FULL_FEATURE_SCHEMA = (
    SEMANTIC_FEATURES
    + VISION_FEATURES
    + PROSODY_FEATURES
    + AUDIO_EMOTION_FEATURES
)


# ── MODALITY FEATURE GROUPS ────────────────────────────────────────────────

MODALITY_FEATURES = {
    "semantic": SEMANTIC_FEATURES,
    "vision": VISION_FEATURES,
    "prosody": PROSODY_FEATURES,
    "audio_emotion": AUDIO_EMOTION_FEATURES,
}


# ── BASIC VALIDATION ───────────────────────────────────────────────────────

EXPECTED_FUSED_VECTOR_SIZE = len(FULL_FEATURE_SCHEMA)