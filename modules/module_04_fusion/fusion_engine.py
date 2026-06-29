"""
Module 4 — Multimodal Fusion Engine

This file is the public entry point for Module 4.

It connects:

    raw module outputs
        ↓
    DynamicFeatureNormalizer
        ↓
    DynamicAttentionFusion
        ↓
    final turn-level fused signal

This is the file other modules should call.
"""

from __future__ import annotations

from typing import Any

from .attention_fusion import DynamicAttentionFusion
from .normalizer import DynamicFeatureNormalizer


class MultimodalFusionEngine:
    """
    Public fusion engine for ARIA.

    It takes one interview turn's multimodal outputs:

        - STT result
        - semantic features
        - vision summary
        - prosody features

    and returns one final fused turn signal.
    """

    def __init__(
        self,
        history_size: int = 5,
        attention_temperature: float = 0.75,
        confidence_power: float = 1.25,
        dissonance_penalty: float = 0.35,
    ):
        self.normalizer = DynamicFeatureNormalizer(
            history_size=history_size,
        )

        self.attention_fusion = DynamicAttentionFusion(
            temperature=attention_temperature,
            confidence_power=confidence_power,
            dissonance_penalty=dissonance_penalty,
        )

    def fuse_turn(
        self,
        candidate_id: str,
        turn_id: str | int,
        stt_result: dict[str, Any] | None = None,
        semantic_features: dict[str, Any] | None = None,
        vision_summary: dict[str, Any] | None = None,
        prosody_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fuse one interview turn.

        Parameters:
            candidate_id:
                Unique candidate/session id.

            turn_id:
                Current interview turn id.

            stt_result:
                Output from Module 1 STT.

            semantic_features:
                Semantic/NLP output for answer quality and competency.

            vision_summary:
                Output from Module 2 vision pipeline.

            prosody_features:
                Output from Module 3 prosody pipeline.

        Returns:
            Final turn-level fused signal.
        """

        normalized_output = self.normalizer.normalize_turn(
            candidate_id=candidate_id,
            stt_result=stt_result,
            semantic_features=semantic_features,
            vision_summary=vision_summary,
            prosody_features=prosody_features,
        )

        fusion_output = self.attention_fusion.fuse(
            normalized_output=normalized_output,
        )

        final_output = {
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

        self._validate_output(final_output)

        return final_output

    def _validate_output(self, output: dict[str, Any]) -> None:
        fused_vector = output.get("fused_vector", [])
        vector_dim = output.get("vector_dim")

        if len(fused_vector) != vector_dim:
            raise ValueError(
                f"Fused vector length mismatch: {len(fused_vector)} != {vector_dim}"
            )

        required_keys = [
            "candidate_id",
            "turn_id",
            "fused_vector",
            "fusion_method",
            "modality_weights",
            "modality_confidences",
            "modality_mask",
            "cross_modal_dissonance",
            "vector_dim",
            "feature_names",
            "imputed_features",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in output
        ]

        if missing_keys:
            raise ValueError(
                f"Fusion output is missing required keys: {missing_keys}"
            )

    def reset_candidate_history(self, candidate_id: str) -> None:
        """
        Clear stored normalization history for one candidate.

        Use this when a new interview session starts and you do not want
        previous turns to affect imputation.
        """

        if candidate_id in self.normalizer._history:
            del self.normalizer._history[candidate_id]