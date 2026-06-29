"""
Module 4 — Dynamic Multimodal Fusion Schema

This file defines the fixed feature order for the ARIA turn-level fused vector.

Rule:
    len(fused_vector) == len(FULL_FEATURE_SCHEMA)

This schema is non-lossy:
    - competency is represented as probabilities
    - emotion is represented as probabilities
    - AU activations and deviations are preserved
    - prosody baseline deviations are preserved
"""


# ── MODALITY NAMES ─────────────────────────────────────────────────────────

TEXT_MODALITY = "text"
VISION_MODALITY = "vision"
PROSODY_MODALITY = "prosody"

MODALITIES = [
    TEXT_MODALITY,
    VISION_MODALITY,
    PROSODY_MODALITY,
]


# ── LABEL SPACES ───────────────────────────────────────────────────────────

COMPETENCY_LABELS = [
    "beginner",
    "mid",
    "expert",
]


VISION_EMOTION_LABELS = [
    "blank",
    "nervous",
    "confused",
    "engaged",
    "confident",
]


# ── TEXT / STT / SEMANTIC FEATURES ─────────────────────────────────────────

STT_FEATURES = [
    "stt_confidence",
    "stt_response_latency_ms",
    "transcript_word_count",
    "transcript_duration_ms",
]


SEMANTIC_SCALAR_FEATURES = [
    "semantic_similarity",
    "question_relevance",
    "answer_completeness",
    "semantic_confidence",
]


SEMANTIC_DISTRIBUTION_FEATURES = [
    f"competency_{label}_prob"
    for label in COMPETENCY_LABELS
]


TEXT_FEATURES = (
    STT_FEATURES
    + SEMANTIC_SCALAR_FEATURES
    + SEMANTIC_DISTRIBUTION_FEATURES
)


# ── VISION FEATURES ────────────────────────────────────────────────────────

VISION_SCALAR_FEATURES = [
    "vision_confidence",
    "emotion_confidence",
    "eye_contact_score",
    "blink_rate",
    "blink_rate_deviation",
]


VISION_EMOTION_DISTRIBUTION_FEATURES = [
    f"vision_emotion_{label}_prob"
    for label in VISION_EMOTION_LABELS
]


VISION_GAZE_FEATURES = [
    "gaze_yaw",
    "gaze_pitch",
]


VISION_HEAD_POSE_FEATURES = [
    "head_roll",
    "head_pitch",
    "head_yaw",
]


AU_FEATURES = [
    "AU1",
    "AU2",
    "AU4",
    "AU6",
    "AU12",
    "AU15",
    "AU17",
    "AU23",
    "AU25",
]


AU_ACTIVATION_FEATURES = [
    f"au_{au}_activation"
    for au in AU_FEATURES
]


AU_DEVIATION_FEATURES = [
    f"au_{au}_deviation"
    for au in AU_FEATURES
]


VISION_FEATURES = (
    VISION_SCALAR_FEATURES
    + VISION_EMOTION_DISTRIBUTION_FEATURES
    + VISION_GAZE_FEATURES
    + VISION_HEAD_POSE_FEATURES
    + AU_ACTIVATION_FEATURES
    + AU_DEVIATION_FEATURES
)


# ── PROSODY FEATURES ───────────────────────────────────────────────────────

PROSODY_SCALAR_FEATURES = [
    "pitch_mean",
    "pitch_variance",
    "pitch_range",
    "speech_rate",
    "pause_count",
    "pause_total_duration_ms",
    "disfluency_count",
    "prosody_response_latency_ms",
    "energy_mean",
    "jitter",
    "shimmer",
    "speech_to_silence_ratio",
]


PROSODY_DEVIATION_FEATURES = [
    "pitch_deviation",
    "rate_deviation",
    "energy_deviation",
]


MFCC_FEATURES = [
    f"mfcc_{i + 1}"
    for i in range(13)
]


PROSODY_FEATURES = (
    PROSODY_SCALAR_FEATURES
    + PROSODY_DEVIATION_FEATURES
    + MFCC_FEATURES
)


# ── FINAL FULL FEATURE ORDER ───────────────────────────────────────────────

FULL_FEATURE_SCHEMA = (
    TEXT_FEATURES
    + VISION_FEATURES
    + PROSODY_FEATURES
)


# ── MODALITY FEATURE GROUPS ────────────────────────────────────────────────

MODALITY_FEATURES = {
    TEXT_MODALITY: TEXT_FEATURES,
    VISION_MODALITY: VISION_FEATURES,
    PROSODY_MODALITY: PROSODY_FEATURES,
}


# ── FEATURE INDEX MAP ──────────────────────────────────────────────────────

FEATURE_INDEX = {
    feature_name: index
    for index, feature_name in enumerate(FULL_FEATURE_SCHEMA)
}


MODALITY_INDEX_RANGES = {}

_start = 0

for modality_name, feature_list in MODALITY_FEATURES.items():
    _end = _start + len(feature_list)

    MODALITY_INDEX_RANGES[modality_name] = {
        "start": _start,
        "end": _end,
        "size": len(feature_list),
    }

    _start = _end


FUSED_VECTOR_DIM = len(FULL_FEATURE_SCHEMA)
EXPECTED_FUSED_VECTOR_SIZE = FUSED_VECTOR_DIM


# ── VALIDATION ─────────────────────────────────────────────────────────────

def validate_schema() -> None:
    if len(FULL_FEATURE_SCHEMA) != len(set(FULL_FEATURE_SCHEMA)):
        seen = set()
        duplicates = []

        for feature in FULL_FEATURE_SCHEMA:
            if feature in seen:
                duplicates.append(feature)
            seen.add(feature)

        raise ValueError(f"Duplicate features found in schema: {duplicates}")

    expected_size = sum(
        len(features)
        for features in MODALITY_FEATURES.values()
    )

    if expected_size != FUSED_VECTOR_DIM:
        raise ValueError(
            f"Schema size mismatch: {expected_size} != {FUSED_VECTOR_DIM}"
        )


validate_schema()