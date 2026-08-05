"""
Module 4 — Dynamic Feature Normalizer

This file converts raw STT, semantic, vision, and prosody outputs into
clean numeric vectors aligned with schema.py.

It also:
    - tracks candidate history
    - imputes missing modalities dynamically
    - creates modality masks
    - creates modality confidence values

It does NOT make decisions about skill, lying, anxiety, or hiring.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import (
    AU_FEATURES,
    COMPETENCY_LABELS,
    FULL_FEATURE_SCHEMA,
    MFCC_FEATURES,
    MODALITY_FEATURES,
    PROSODY_FEATURES,
    PROSODY_MODALITY,
    TEXT_FEATURES,
    TEXT_MODALITY,
    VISION_EMOTION_LABELS,
    VISION_FEATURES,
    VISION_MODALITY,
)


@dataclass
class NormalizedModality:
    values: list[float]
    mask: float
    confidence: float
    imputed_features: list[str]


class DynamicFeatureNormalizer:
    def __init__(self, history_size: int = 5):
        self.history_size = history_size

        self._history = defaultdict(
            lambda: defaultdict(
                lambda: deque(maxlen=self.history_size)
            )
        )

    def reset(self, candidate_id: str) -> None:
        """Clear stored normalization history for one candidate."""
        if candidate_id in self._history:
            del self._history[candidate_id]

    # ── BASIC HELPERS ──────────────────────────────────────────────────────

    def safe_float(self, val: Any) -> float | None:
        if val is None:
            return None

        if isinstance(val, bool):
            return 1.0 if val else 0.0

        try:
            new_val = float(val)
        except (TypeError, ValueError):
            return None

        if math.isnan(new_val) or math.isinf(new_val):
            return None

        return new_val

    def clip_probability(self, val: Any) -> float | None:
        new_val = self.safe_float(val)

        if new_val is None:
            return None

        return max(0.0, min(1.0, new_val))

    def _compress_positive(self, value: Any, scale: float) -> float | None:
        value = self.safe_float(value)

        if value is None:
            return None

        value = max(0.0, value)
        scale = max(float(scale), 1e-6)

        return value / (value + scale)

    def _compress_signed(self, value: Any, scale: float) -> float | None:
        value = self.safe_float(value)

        if value is None:
            return None

        scale = max(float(scale), 1e-6)

        return math.tanh(value / scale)

    # ── FEATURE SCALING ────────────────────────────────────────────────────

    def scale_feature(self, feature_name: str, value: Any) -> float | None:
        name = feature_name.lower()

        if (
            name.endswith("_prob")
            or "confidence" in name
            or name.endswith("_score")
            or "activation" in name
        ):
            return self.clip_probability(value)

        if name.endswith("_ms") or "latency" in name or "duration" in name:
            return self._compress_positive(value, scale=5000.0)

        if "word_count" in name:
            return self._compress_positive(value, scale=100.0)

        if "pause_count" in name or "disfluency_count" in name:
            return self._compress_positive(value, scale=10.0)

        if "blink_rate" in name:
            return self._compress_positive(value, scale=60.0)

        if "pitch_mean" in name:
            return self._compress_positive(value, scale=300.0)

        if "pitch_variance" in name or "pitch_range" in name:
            return self._compress_positive(value, scale=150.0)

        if "speech_rate" in name:
            return self._compress_positive(value, scale=8.0)

        if "energy_mean" in name:
            return self._compress_positive(value, scale=1.0)

        if "jitter" in name or "shimmer" in name:
            return self._compress_positive(value, scale=1.0)

        if "speech_to_silence_ratio" in name:
            return self._compress_positive(value, scale=10.0)

        if "deviation" in name:
            return self._compress_signed(value, scale=2.0)

        if "gaze_yaw" in name or "gaze_pitch" in name:
            return self._compress_signed(value, scale=45.0)

        if "head_roll" in name or "head_pitch" in name or "head_yaw" in name:
            return self._compress_signed(value, scale=45.0)

        if name.startswith("mfcc_"):
            return self._compress_signed(value, scale=50.0)

        return self.safe_float(value)

    # ── DISTRIBUTION HELPERS ───────────────────────────────────────────────

    def normalize_distribution(
        self,
        distribution: dict[str, Any] | None,
        labels: list[str],
    ) -> dict[str, float] | None:
        if not distribution:
            return None

        lowered = {
            str(key).lower().strip(): value
            for key, value in distribution.items()
        }

        cleaned = {}

        for label in labels:
            prob = self.clip_probability(lowered.get(label))
            cleaned[label] = 0.0 if prob is None else prob

        total = sum(cleaned.values())

        if total <= 1e-8:
            return None

        return {
            label: value / total
            for label, value in cleaned.items()
        }

    def distribution_from_label(
        self,
        label: str | None,
        confidence: Any,
        labels: list[str],
    ) -> dict[str, float] | None:
        if label is None:
            return None

        label = str(label).lower().strip()

        if label not in labels:
            return None

        conf = self.clip_probability(confidence)

        if conf is None:
            conf = 0.70

        remaining = 1.0 - conf
        other_labels = [item for item in labels if item != label]

        distribution = {}

        for item in labels:
            if item == label:
                distribution[item] = conf
            else:
                distribution[item] = remaining / max(len(other_labels), 1)

        return distribution

    # ── HISTORY / IMPUTATION HELPERS ───────────────────────────────────────

    def _get_history_mean(
        self,
        candidate_id: str,
        modality: str,
        expected_size: int,
    ) -> list[float] | None:
        history = self._history[candidate_id][modality]

        if len(history) == 0:
            return None

        arr = np.array(list(history), dtype=np.float32)

        if arr.ndim != 2 or arr.shape[1] != expected_size:
            return None

        mean_vector = np.mean(arr, axis=0)

        return mean_vector.astype(float).tolist()

    def _impute_full_modality(
        self,
        candidate_id: str,
        modality: str,
        feature_names: list[str],
    ) -> list[float]:
        history_mean = self._get_history_mean(
            candidate_id=candidate_id,
            modality=modality,
            expected_size=len(feature_names),
        )

        if history_mean is not None:
            return history_mean

        return [0.0 for _ in feature_names]

    def _update_history(
        self,
        candidate_id: str,
        modality: str,
        values: list[float],
        mask: float,
    ) -> None:
        if math.isclose(mask, 0.0, abs_tol=1e-9):
            return

        self._history[candidate_id][modality].append(values)

    def _build_modality_result(
        self,
        candidate_id: str,
        modality: str,
        feature_names: list[str],
        raw_values: dict[str, Any] | None,
        confidence: Any,
        present: bool,
    ) -> NormalizedModality:
        if not present or raw_values is None:
            values = self._impute_full_modality(
                candidate_id=candidate_id,
                modality=modality,
                feature_names=feature_names,
            )

            return NormalizedModality(
                values=values,
                mask=0.0,
                confidence=0.0,
                imputed_features=list(feature_names),
            )

        history_mean = self._get_history_mean(
            candidate_id=candidate_id,
            modality=modality,
            expected_size=len(feature_names),
        )

        values = []
        imputed_features = []

        for index, feature_name in enumerate(feature_names):
            scaled = self.scale_feature(
                feature_name=feature_name,
                value=raw_values.get(feature_name),
            )

            if scaled is None:
                imputed_features.append(feature_name)

                if history_mean is not None:
                    scaled = history_mean[index]
                else:
                    scaled = 0.0

            values.append(float(scaled))

        modality_confidence = self.clip_probability(confidence)

        if modality_confidence is None:
            observed_count = len(feature_names) - len(imputed_features)
            modality_confidence = observed_count / max(len(feature_names), 1)

        result = NormalizedModality(
            values=values,
            mask=1.0,
            confidence=float(modality_confidence),
            imputed_features=imputed_features,
        )

        self._update_history(
            candidate_id=candidate_id,
            modality=modality,
            values=values,
            mask=result.mask,
        )

        return result

    # ── TEXT NORMALIZATION ─────────────────────────────────────────────────

    def _transcript_word_count(self, transcript: str | None) -> int:
        if not transcript:
            return 0

        words = re.findall(r"\b\w+\b", transcript)

        return len(words)

    def _transcript_duration_ms(self, word_timestamps: list[dict] | None) -> float | None:
        if not word_timestamps:
            return None

        try:
            first_start = self.safe_float(word_timestamps[0].get("start"))
            last_end = self.safe_float(word_timestamps[-1].get("end"))
        except (AttributeError, IndexError):
            return None

        if first_start is None or last_end is None:
            return None

        duration_seconds = max(0.0, last_end - first_start)

        return duration_seconds * 1000.0

    def _extract_competency_distribution(
        self,
        semantic_features: dict[str, Any] | None,
    ) -> dict[str, float] | None:
        if not semantic_features:
            return None

        for key in [
            "competency_distribution",
            "competency_probs",
            "competency_probabilities",
        ]:
            distribution = self.normalize_distribution(
                semantic_features.get(key),
                COMPETENCY_LABELS,
            )

            if distribution is not None:
                return distribution

        return self.distribution_from_label(
            label=semantic_features.get("predicted_competency"),
            confidence=semantic_features.get(
                "semantic_confidence",
                semantic_features.get("confidence"),
            ),
            labels=COMPETENCY_LABELS,
        )

    def normalize_text(
        self,
        candidate_id: str,
        stt_result: dict[str, Any] | None,
        semantic_features: dict[str, Any] | None,
    ) -> NormalizedModality:
        stt_result = stt_result or {}
        semantic_features = semantic_features or {}

        transcript = stt_result.get("transcript", "")
        word_timestamps = stt_result.get("word_timestamps", [])

        present = bool(transcript) or bool(semantic_features)

        competency_dist = self._extract_competency_distribution(semantic_features)

        raw = {
            "stt_confidence": stt_result.get("confidence"),
            "stt_response_latency_ms": stt_result.get("response_latency_ms"),
            "response_latency_ms": stt_result.get("response_latency_ms"),
            "transcript_word_count": self._transcript_word_count(transcript),
            "transcript_duration_ms": self._transcript_duration_ms(word_timestamps),
            "semantic_similarity": semantic_features.get("semantic_similarity"),
            "question_relevance": semantic_features.get("question_relevance"),
            "answer_completeness": semantic_features.get("answer_completeness"),
            "semantic_confidence": semantic_features.get(
                "semantic_confidence",
                semantic_features.get("confidence"),
            ),
        }

        if competency_dist:
            for label in COMPETENCY_LABELS:
                raw[f"competency_{label}_prob"] = competency_dist.get(label)

        confidence = raw.get("semantic_confidence", raw.get("stt_confidence"))

        return self._build_modality_result(
            candidate_id=candidate_id,
            modality=TEXT_MODALITY,
            feature_names=TEXT_FEATURES,
            raw_values=raw,
            confidence=confidence,
            present=present,
        )

    # ── VISION NORMALIZATION ───────────────────────────────────────────────

    def _extract_vision_emotion_distribution(
        self,
        vision_summary: dict[str, Any] | None,
    ) -> dict[str, float] | None:
        if not vision_summary:
            return None

        distribution = self.normalize_distribution(
            vision_summary.get("emotion_distribution"),
            VISION_EMOTION_LABELS,
        )

        if distribution is not None:
            return distribution

        label = vision_summary.get(
            "emotion_label",
            vision_summary.get("dominant_emotion"),
        )

        return self.distribution_from_label(
            label=label,
            confidence=vision_summary.get("emotion_confidence"),
            labels=VISION_EMOTION_LABELS,
        )

    def normalize_vision(
        self,
        candidate_id: str,
        vision_summary: dict[str, Any] | None,
    ) -> NormalizedModality:
        vision_summary = vision_summary or {}

        present = bool(vision_summary)

        gaze = vision_summary.get("gaze_vector", {}) or {}
        head_pose = vision_summary.get("head_pose", {}) or {}
        au_activations = vision_summary.get("au_activations", {}) or {}
        au_deviations = vision_summary.get("au_deviations", {}) or {}

        emotion_dist = self._extract_vision_emotion_distribution(vision_summary)

        raw = {
            "vision_confidence": vision_summary.get(
                "vision_confidence",
                vision_summary.get("confidence", vision_summary.get("emotion_confidence")),
            ),
            "emotion_confidence": vision_summary.get("emotion_confidence"),
            "eye_contact_score": vision_summary.get(
                "eye_contact_score",
                vision_summary.get("eye_contact_ratio"),
            ),
            "blink_rate": vision_summary.get("blink_rate"),
            "blink_rate_deviation": vision_summary.get("blink_rate_deviation"),
            "gaze_yaw": gaze.get("yaw"),
            "gaze_pitch": gaze.get("pitch"),
            "head_roll": head_pose.get("roll"),
            "head_pitch": head_pose.get("pitch"),
            "head_yaw": head_pose.get("yaw"),
        }

        if emotion_dist:
            for label in VISION_EMOTION_LABELS:
                raw[f"vision_emotion_{label}_prob"] = emotion_dist.get(label)

        for au in AU_FEATURES:
            raw[f"au_{au}_activation"] = au_activations.get(au)
            raw[f"au_{au}_deviation"] = au_deviations.get(au)

        confidence = raw.get("vision_confidence", raw.get("emotion_confidence"))

        return self._build_modality_result(
            candidate_id=candidate_id,
            modality=VISION_MODALITY,
            feature_names=VISION_FEATURES,
            raw_values=raw,
            confidence=confidence,
            present=present,
        )

    # ── PROSODY NORMALIZATION ──────────────────────────────────────────────

    def _estimate_prosody_confidence(self, prosody_features: dict[str, Any]) -> float:
        important_keys = [
            "pitch_mean",
            "speech_rate",
            "energy_mean",
            "jitter",
            "shimmer",
            "speech_to_silence_ratio",
        ]

        observed = 0

        for key in important_keys:
            if self.safe_float(prosody_features.get(key)) is not None:
                observed += 1

        return observed / len(important_keys)

    def normalize_prosody(
        self,
        candidate_id: str,
        prosody_features: dict[str, Any] | None,
    ) -> NormalizedModality:
        prosody_features = prosody_features or {}

        present = bool(prosody_features)

        raw = {}

        for feature_name in PROSODY_FEATURES:
            raw[feature_name] = prosody_features.get(feature_name)

        raw["prosody_response_latency_ms"] = prosody_features.get("response_latency_ms")
        raw["response_latency_ms"] = prosody_features.get("response_latency_ms")

        mfcc_vector = prosody_features.get("mfcc_vector", [])

        if isinstance(mfcc_vector, (list, tuple)):
            for index, feature_name in enumerate(MFCC_FEATURES):
                raw[feature_name] = (
                    mfcc_vector[index]
                    if index < len(mfcc_vector)
                    else None
                )

        confidence = prosody_features.get("prosody_confidence")

        if confidence is None:
            confidence = self._estimate_prosody_confidence(prosody_features)

        return self._build_modality_result(
            candidate_id=candidate_id,
            modality=PROSODY_MODALITY,
            feature_names=PROSODY_FEATURES,
            raw_values=raw,
            confidence=confidence,
            present=present,
        )

    # ── COMPLETE TURN NORMALIZATION ────────────────────────────────────────

    def normalize_turn(
        self,
        candidate_id: str,
        stt_result: dict[str, Any] | None = None,
        semantic_features: dict[str, Any] | None = None,
        vision_summary: dict[str, Any] | None = None,
        prosody_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = self.normalize_text(
            candidate_id=candidate_id,
            stt_result=stt_result,
            semantic_features=semantic_features,
        )

        vision = self.normalize_vision(
            candidate_id=candidate_id,
            vision_summary=vision_summary,
        )

        prosody = self.normalize_prosody(
            candidate_id=candidate_id,
            prosody_features=prosody_features,
        )

        modality_results = {
            TEXT_MODALITY: text,
            VISION_MODALITY: vision,
            PROSODY_MODALITY: prosody,
        }

        normalized_vector = [0.0] * len(FULL_FEATURE_SCHEMA)
        idx = 0
        modality_mask = {}
        modality_confidences = {}
        imputed_features = {}

        for modality_name, result in modality_results.items():
            length = len(result.values)
            normalized_vector[idx:idx+length] = result.values
            idx += length
            modality_mask[modality_name] = result.mask
            modality_confidences[modality_name] = result.confidence
            imputed_features[modality_name] = result.imputed_features

        if len(normalized_vector) != len(FULL_FEATURE_SCHEMA):
            raise ValueError(
                f"Normalized vector length mismatch: "
                f"{len(normalized_vector)} != {len(FULL_FEATURE_SCHEMA)}"
            )

        return {
            "normalized_vector": normalized_vector,
            "feature_names": FULL_FEATURE_SCHEMA,
            "vector_dim": len(normalized_vector),
            "modality_mask": modality_mask,
            "modality_confidences": modality_confidences,
            "imputed_features": imputed_features,
            "normalized_modalities": modality_results,
        }