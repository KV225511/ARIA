"""
Module 4 — Dynamic Multimodal Fusion

Public exports for the ARIA fusion module.

Usage:
    from modules.module_04_fusion import MultimodalFusionEngine
"""

from .attention_fusion import DynamicAttentionFusion
from .concat_fusion import ConcatFusion, ConcatenationFusionEngine
from .fusion_engine import MultimodalFusionEngine
from .normalizer import DynamicFeatureNormalizer, NormalizedModality
from .schema import (
    FUSED_VECTOR_DIM,
    FULL_FEATURE_SCHEMA,
    MODALITY_FEATURES,
    MODALITY_INDEX_RANGES,
    PROSODY_MODALITY,
    TEXT_MODALITY,
    VISION_MODALITY,
)

__all__ = [
    "DynamicAttentionFusion",
    "ConcatFusion",
    "ConcatenationFusionEngine",
    "MultimodalFusionEngine",
    "DynamicFeatureNormalizer",
    "NormalizedModality",
    "FUSED_VECTOR_DIM",
    "FULL_FEATURE_SCHEMA",
    "MODALITY_FEATURES",
    "MODALITY_INDEX_RANGES",
    "TEXT_MODALITY",
    "VISION_MODALITY",
    "PROSODY_MODALITY",
]