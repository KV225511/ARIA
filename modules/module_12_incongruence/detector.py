class CrossModalIncongruenceDetector:
    def __init__(self, incongruence_threshold=0.4):
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
            
        return {
            "incongruence_flag": is_bluffing,
            "magnitude": round(delta, 3) if delta > 0 else 0.0,
            "prosody_confidence": prosody_confidence,
            "semantic_score": semantic_score
        }
