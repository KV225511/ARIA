"""
Module 4 — Unweighted Concatenation Fusion Baseline

This file implements the standard concatenation baseline for comparison against
DynamicAttentionFusion. It takes normalized modality vectors from normalizer.py
and concatenates them without attention gating or dissonance penalization.
"""

from __future__ import annotations

from typing import Any

from .normalizer import DynamicFeatureNormalizer
from .schema import (
    PROSODY_MODALITY,
    TEXT_MODALITY,
    VISION_MODALITY,
)


class ConcatFusion:
    """
    Simple unweighted concatenation fusion baseline.
    Keeps feature scale intact (weight = 1.0 per active modality).
    """

    def __init__(self):
        self.modality_order = [
            TEXT_MODALITY,
            VISION_MODALITY,
            PROSODY_MODALITY,
        ]

    def _clip01(self, value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))

    def fuse(self, normalized_output: dict[str, Any]) -> dict[str, Any]:
        normalized_vector = normalized_output.get("normalized_vector", [])
        modality_mask = normalized_output.get("modality_mask", {}) or {}
        modality_confidences = normalized_output.get("modality_confidences", {}) or {}

        clean_modality_mask = {
            m: self._clip01(modality_mask.get(m, 0.0)) for m in self.modality_order
        }
        clean_modality_confidences = {
            m: self._clip01(modality_confidences.get(m, 0.0)) for m in self.modality_order
        }

        active_count = sum(1 for m in self.modality_order if clean_modality_mask[m] > 0.0)
        uniform_weight = 1.0 / active_count if active_count > 0 else 0.0

        modality_weights = {
            m: uniform_weight if clean_modality_mask[m] > 0.0 else 0.0
            for m in self.modality_order
        }

        return {
            "fused_vector": list(normalized_vector),
            "fusion_method": "unweighted_concatenation",
            "modality_weights": modality_weights,
            "modality_confidences": clean_modality_confidences,
            "modality_mask": clean_modality_mask,
            "cross_modal_dissonance": 0.0,
            "vector_dim": len(normalized_vector),
        }


class ConcatenationFusionEngine:
    """
    Public engine wrapper for the concatenation baseline.
    Mirrors MultimodalFusionEngine interface.
    """

    def __init__(self, history_size: int = 5):
        self.normalizer = DynamicFeatureNormalizer(history_size=history_size)
        self.concat_fusion = ConcatFusion()

    def fuse_turn(
        self,
        candidate_id: str,
        turn_id: str | int,
        stt_result: dict[str, Any] | None = None,
        semantic_features: dict[str, Any] | None = None,
        vision_summary: dict[str, Any] | None = None,
        prosody_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_output = self.normalizer.normalize_turn(
            candidate_id=candidate_id,
            stt_result=stt_result,
            semantic_features=semantic_features,
            vision_summary=vision_summary,
            prosody_features=prosody_features,
        )

        fusion_output = self.concat_fusion.fuse(normalized_output=normalized_output)

        return {
            "candidate_id": candidate_id,
            "turn_id": turn_id,
            "fused_vector": fusion_output["fused_vector"],
            "fusion_method": fusion_output["fusion_method"],
            "modality_weights": fusion_output["modality_weights"],
            "modality_confidences": fusion_output["modality_confidences"],
            "modality_mask": fusion_output["modality_mask"],
            "cross_modal_dissonance": fusion_output["cross_modal_dissonance"],
            "vector_dim": fusion_output["vector_dim"],
            "feature_names": normalized_output["feature_names"],
            "imputed_features": normalized_output["imputed_features"],
            "normalization_debug": {
                "normalized_vector": normalized_output["normalized_vector"],
                "normalized_vector_dim": normalized_output["vector_dim"],
            },
        }

    def reset_candidate_history(self, candidate_id: str) -> None:
        if candidate_id in self.normalizer._history:
            del self.normalizer._history[candidate_id]
