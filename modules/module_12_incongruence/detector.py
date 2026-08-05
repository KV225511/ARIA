import logging

logger = logging.getLogger(__name__)

DEFAULT_INCONGRUENCE_THRESHOLD = 0.4

class CrossModalIncongruenceDetector:
    def __init__(self, incongruence_threshold=DEFAULT_INCONGRUENCE_THRESHOLD):
        """
        Detects bluffing by finding incongruence between semantic depth and prosodic confidence.
        """
        self.threshold = incongruence_threshold

    def detect(self, semantic_score: float, prosody_confidence: float) -> dict:
        """
        Args:
            semantic_score (float): 0.0 to 1.0 representing how accurately the candidate answered (e.g. from BGE-M3 embedding comparison).
            prosody_confidence (float): 0.0 to 1.0 representing behavioral confidence (high speech rate, low pause).
            
        Returns:
            dict: The incongruence flag and the delta magnitude.
        """
        # Both scores should be between 0 and 1
        semantic_score = max(0.0, min(1.0, semantic_score))
        prosody_confidence = max(0.0, min(1.0, prosody_confidence))
        
        # Incongruence is specifically when they sound VERY confident but the answer is garbage (bluffing).
        # We don't penalize low confidence + high semantic (that's just anxiety, handled by Module 10).
        delta = prosody_confidence - semantic_score
        
        is_bluffing = False
        if delta > self.threshold:
            is_bluffing = True
            logger.info(f"Bluffing detected: delta={delta:.3f} (confidence={prosody_confidence:.2f}, semantic={semantic_score:.2f})")
            
        return {
            "incongruence_flag": is_bluffing,
            "magnitude": round(abs(delta), 3),
            "is_negative": delta < 0,
            "prosody_confidence": prosody_confidence,
            "semantic_score": semantic_score
        }
