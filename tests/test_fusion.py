"""
Unit tests for Module 4 — Multimodal Fusion Engine & Concatenation Baseline.
"""

import pytest
from modules.module_04_fusion import (
    ConcatenationFusionEngine,
    FUSED_VECTOR_DIM,
    MultimodalFusionEngine,
)


@pytest.fixture
def sample_inputs():
    return {
        "candidate_id": "cand_001",
        "turn_id": 1,
        "stt_result": {
            "transcript": "I am very confident in my leadership skills.",
            "confidence": 0.95,
            "response_latency_ms": 350.0,
        },
        "semantic_features": {
            "semantic_similarity": 0.88,
            "question_relevance": 0.90,
            "answer_completeness": 0.85,
            "predicted_competency": "expert",
            "confidence": 0.92,
        },
        "vision_summary": {
            "emotion_label": "confident",
            "emotion_confidence": 0.89,
            "eye_contact_score": 0.82,
            "blink_rate": 18.0,
        },
        "prosody_features": {
            "pitch_mean": 180.0,
            "speech_rate": 4.5,
            "energy_mean": 0.75,
            "prosody_confidence": 0.85,
        },
    }


def test_attention_fusion_engine_basic(sample_inputs):
    engine = MultimodalFusionEngine()
    output = engine.fuse_turn(**sample_inputs)

    assert output["candidate_id"] == "cand_001"
    assert output["turn_id"] == 1
    assert output["vector_dim"] == FUSED_VECTOR_DIM
    assert len(output["fused_vector"]) == FUSED_VECTOR_DIM
    assert output["fusion_method"] == "cross_modal_attention_gated"
    assert 0.0 <= output["cross_modal_dissonance"] <= 1.0


def test_concatenation_fusion_engine_basic(sample_inputs):
    engine = ConcatenationFusionEngine()
    output = engine.fuse_turn(**sample_inputs)

    assert output["candidate_id"] == "cand_001"
    assert output["turn_id"] == 1
    assert output["vector_dim"] == FUSED_VECTOR_DIM
    assert len(output["fused_vector"]) == FUSED_VECTOR_DIM
    assert output["fusion_method"] == "unweighted_concatenation"
    assert output["cross_modal_dissonance"] == 0.0


def test_missing_modality_imputation(sample_inputs):
    engine = MultimodalFusionEngine()
    # Turn 1 with all modalities
    engine.fuse_turn(**sample_inputs)

    # Turn 2 missing vision
    inputs_t2 = sample_inputs.copy()
    inputs_t2["turn_id"] = 2
    inputs_t2["vision_summary"] = None

    output_t2 = engine.fuse_turn(**inputs_t2)
    assert output_t2["modality_mask"]["vision"] == 0.0
    assert len(output_t2["fused_vector"]) == FUSED_VECTOR_DIM
