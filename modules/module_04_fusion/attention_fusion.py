"""
Module 4 — Dynamic Cross-Modal Attention Fusion

This file takes normalized modality vectors from normalizer.py and produces
a dynamically weighted fused vector.

Input:
    {
        "normalized_vector": list[float],
        "modality_mask": {"text": float, "vision": float, "prosody": float},
        "modality_confidences": {"text": float, "vision": float, "prosody": float},
        ...
    }

Output:
    {
        "fused_vector": list[float],
        "fusion_method": "cross_modal_attention_gated",
        "modality_weights": dict,
        "modality_confidences": dict,
        "modality_mask": dict,
        "cross_modal_dissonance": float,
        "vector_dim": int,
        ...
    }
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np

from .schema import (
    FULL_FEATURE_SCHEMA,
    MODALITY_INDEX_RANGES,
    PROSODY_MODALITY,
    TEXT_MODALITY,
    VISION_MODALITY,
)


class DynamicAttentionFusion:
    """
    Dynamic attention-based fusion for text, vision, and prosody.

    It does not blindly concatenate modalities.
    Instead, it calculates modality reliability using:

        1. modality mask
        2. modality confidence
        3. cross-modal dissonance
        4. signal strength

    Then each modality section of the vector is weighted dynamically.
    """

    def __init__(
        self,
        temperature: float = 0.75,
        epsilon: float = 1e-8,
        confidence_power: float = 1.25,
        dissonance_penalty: float = 0.35,
    ):
        self.temperature = max(float(temperature), epsilon)
        self.epsilon = epsilon
        self.confidence_power = confidence_power
        self.dissonance_penalty = dissonance_penalty

        self.modality_order = [
            TEXT_MODALITY,
            VISION_MODALITY,
            PROSODY_MODALITY,
        ]

    # ── BASIC HELPERS ──────────────────────────────────────────────────────

    def _clip01(self, value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not math.isfinite(value):
            return 0.0

        return max(0.0, min(1.0, value))

    def _as_vector(self, vector: Any) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)

        if arr.ndim != 1:
            raise ValueError("normalized_vector must be a 1D vector")

        if arr.shape[0] != len(FULL_FEATURE_SCHEMA):
            raise ValueError(
                f"Vector size mismatch: {arr.shape[0]} != {len(FULL_FEATURE_SCHEMA)}"
            )

        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return arr

    # ── MODALITY SLICING ───────────────────────────────────────────────────

    def _slice_modalities(self, vector: np.ndarray) -> dict[str, np.ndarray]:
        modality_vectors = {}

        for modality in self.modality_order:
            range_info = MODALITY_INDEX_RANGES[modality]
            start = range_info["start"]
            end = range_info["end"]

            modality_vectors[modality] = vector[start:end]

        return modality_vectors

    # ── SOFTMAX ATTENTION ──────────────────────────────────────────────────

    def _masked_softmax(
        self,
        raw_scores: dict[str, float],
        modality_mask: dict[str, float],
    ) -> dict[str, float]:
        active_modalities = []

        for modality in self.modality_order:
            mask = self._clip01(modality_mask.get(modality, 0.0))
            score = float(raw_scores.get(modality, 0.0))

            if mask > 0.0 and score > self.epsilon:
                active_modalities.append(modality)

        if not active_modalities:
            present_modalities = [
                modality
                for modality in self.modality_order
                if self._clip01(modality_mask.get(modality, 0.0)) > 0.0
            ]

            if not present_modalities:
                return {
                    modality: 0.0
                    for modality in self.modality_order
                }

            uniform_weight = 1.0 / len(present_modalities)

            return { modality: uniform_weight if modality in present_modalities else 0.0
                for modality in self.modality_order
            }

        scores = np.array(
            [
                raw_scores[modality] / self.temperature
                for modality in active_modalities
            ],
            dtype=np.float32,
        )

        max_score = float(np.max(scores))

        exp_scores = np.exp(scores - max_score)
        denominator = float(np.sum(exp_scores))

        if denominator <= self.epsilon:
            uniform_weight = 1.0 / len(active_modalities)

            return {
                modality: uniform_weight if modality in active_modalities else 0.0
                for modality in self.modality_order
            }

        active_weights = {
            modality: float(exp_scores[index] / denominator)
            for index, modality in enumerate(active_modalities)
        }

        return {
            modality: active_weights.get(modality, 0.0)
            for modality in self.modality_order
        }

    # ── SIGNAL STRENGTH ────────────────────────────────────────────────────

    def _signal_strength(self, vector: np.ndarray) -> float:
        if vector.size == 0:
            return 0.0

        mean_abs_value = float(np.mean(np.abs(vector)))

        if not math.isfinite(mean_abs_value):
            return 0.0

        return self._clip01(mean_abs_value)

    # ── CROSS-MODAL DISSONANCE ─────────────────────────────────────────────

    def _cosine_distance(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        min_len = min(vec_a.shape[0], vec_b.shape[0])

        if min_len == 0:
            return 0.0

        a = vec_a[:min_len]
        b = vec_b[:min_len]

        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))

        if norm_a <= self.epsilon or norm_b <= self.epsilon:
            return 0.0

        cosine_similarity = float(np.dot(a, b) / (norm_a * norm_b))
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        distance = 1.0 - cosine_similarity

        return self._clip01(distance)

    def _compute_cross_modal_dissonance(
        self,
        modality_vectors: dict[str, np.ndarray],
        modality_mask: dict[str, float],
    ) -> float:
        active_modalities = [
            modality
            for modality in self.modality_order
            if self._clip01(modality_mask.get(modality, 0.0)) > 0.0
        ]

        if len(active_modalities) < 2:
            return 0.0

        distances = []

        for modality_a, modality_b in combinations(active_modalities, 2):
            distance = self._cosine_distance(
                modality_vectors[modality_a],
                modality_vectors[modality_b],
            )

            distances.append(distance)

        if not distances:
            return 0.0

        return self._clip01(float(np.mean(distances)))

    # ── MODALITY WEIGHT COMPUTATION ────────────────────────────────────────

    def _compute_modality_weights(
        self,
        modality_vectors: dict[str, np.ndarray],
        modality_mask: dict[str, float],
        modality_confidences: dict[str, float],
        cross_modal_dissonance: float,
    ) -> dict[str, float]:
        active_modalities = [
            modality
            for modality in self.modality_order
            if self._clip01(modality_mask.get(modality, 0.0)) > 0.0
        ]

        raw_scores = {}

        for modality in self.modality_order:
            mask = self._clip01(modality_mask.get(modality, 0.0))
            confidence = self._clip01(modality_confidences.get(modality, 0.0))
            signal_strength = self._signal_strength(modality_vectors[modality])

            if mask <= 0.0:
                raw_scores[modality] = 0.0
                continue

            # Compute specific cross-modal dissonance for this modality against other active modalities
            other_active = [m for m in active_modalities if m != modality]
            if other_active:
                dissonances = [
                    self._cosine_distance(modality_vectors[modality], modality_vectors[other])
                    for other in other_active
                ]
                mod_dissonance = float(np.mean(dissonances))
            else:
                mod_dissonance = cross_modal_dissonance

            dissonance_factor = 1.0 - (
                self.dissonance_penalty * self._clip01(mod_dissonance)
            )
            dissonance_factor = max(self.epsilon, dissonance_factor)

            confidence_score = confidence ** self.confidence_power

            raw_score = (
                mask
                * confidence_score
                * (0.5 + 0.5 * signal_strength)
                * dissonance_factor
            )

            raw_scores[modality] = float(raw_score)

        return self._masked_softmax(
            raw_scores=raw_scores,
            modality_mask=modality_mask,
        )

    # ── APPLY WEIGHTS TO VECTOR ────────────────────────────────────────────

    def _apply_modality_weights(
        self,
        vector: np.ndarray,
        modality_weights: dict[str, float],
        modality_mask: dict[str, float] | None = None,
    ) -> list[float]:
        fused_vector = vector.copy()
        modality_mask = modality_mask or {}

        active_count = sum(
            1 for m in self.modality_order if self._clip01(modality_mask.get(m, 0.0)) > 0.0
        )
        scale_multiplier = max(1.0, float(active_count))

        for modality in self.modality_order:
            range_info = MODALITY_INDEX_RANGES[modality]
            start = range_info["start"]
            end = range_info["end"]

            weight = float(modality_weights.get(modality, 0.0))
            gating_multiplier = weight * scale_multiplier

            fused_vector[start:end] = fused_vector[start:end] * gating_multiplier

        return fused_vector.astype(float).tolist()

    # ── PUBLIC FUSION METHOD ───────────────────────────────────────────────

    def fuse(self, normalized_output: dict[str, Any]) -> dict[str, Any]:
        vector = self._as_vector(
            normalized_output.get("normalized_vector")
        )

        modality_mask = normalized_output.get("modality_mask", {}) or {}
        modality_confidences = (
            normalized_output.get("modality_confidences", {}) or {}
        )

        clean_modality_mask = {
            modality: self._clip01(modality_mask.get(modality, 0.0))
            for modality in self.modality_order
        }

        clean_modality_confidences = {
            modality: self._clip01(modality_confidences.get(modality, 0.0))
            for modality in self.modality_order
        }

        modality_vectors = self._slice_modalities(vector)

        cross_modal_dissonance = self._compute_cross_modal_dissonance(
            modality_vectors=modality_vectors,
            modality_mask=clean_modality_mask,
        )

        modality_weights = self._compute_modality_weights(
            modality_vectors=modality_vectors,
            modality_mask=clean_modality_mask,
            modality_confidences=clean_modality_confidences,
            cross_modal_dissonance=cross_modal_dissonance,
        )

        fused_vector = self._apply_modality_weights(
            vector=vector,
            modality_weights=modality_weights,
            modality_mask=clean_modality_mask,
        )

        return {
            "fused_vector": fused_vector,
            "fusion_method": "cross_modal_attention_gated",
            "modality_weights": modality_weights,
            "modality_confidences": clean_modality_confidences,
            "modality_mask": clean_modality_mask,
            "cross_modal_dissonance": cross_modal_dissonance,
            "vector_dim": len(fused_vector),
        }