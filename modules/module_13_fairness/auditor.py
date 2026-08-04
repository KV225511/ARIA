import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class InterviewFairnessAuditor:
    def __init__(self):
        """
        Runs as a background audit process across sessions, tracking RL action distribution 
        against demographic-adjacent speech patterns (e.g. pitch).
        """
        self.session_data = []
        
    def log_turn(self, prosody_features: dict, rl_action: str):
        """
        Logs a single turn's features and the subsequent RL action chosen.
        """
        if not prosody_features:
            return
            
        self.session_data.append({
            "pitch_hz": prosody_features.get("pitch_f0_hz", 0),
            "speech_rate": prosody_features.get("speech_rate_syllables_per_sec", 0),
            "action": rl_action
        })
        
    def generate_audit_report(self) -> dict:
        """
        Generates a fairness report showing action distribution across speech patterns.
        """
        if not self.session_data:
            return {"status": "insufficient_data"}
            
        action_counts = defaultdict(int)
        pitch_by_action = defaultdict(list)
        
        for turn in self.session_data:
            action = turn["action"]
            action_counts[action] += 1
            pitch_by_action[action].append(turn["pitch_hz"])
            
        # Calculate average pitch per action type to check for bias
        # For example, if "increase_difficulty" is exclusively given to low-pitch (traditionally male) speakers.
        pitch_bias_stats = {}
        for action, pitches in pitch_by_action.items():
            valid_pitches = [p for p in pitches if p > 0]
            avg_pitch = sum(valid_pitches) / len(valid_pitches) if valid_pitches else 0
            pitch_bias_stats[action] = round(avg_pitch, 2)
            
        report = {
            "total_turns_audited": len(self.session_data),
            "action_distribution": dict(action_counts),
            "average_pitch_by_action_hz": pitch_bias_stats,
            "bias_detected": self._detect_bias(pitch_bias_stats)
        }
        
        return report
        
    def _detect_bias(self, pitch_bias_stats: dict) -> bool:
        """
        Simple heuristic: if the average pitch for 'increase_difficulty' is radically different 
        from 'probe_foundation', it flags potential bias.
        """
        if "increase_difficulty" in pitch_bias_stats and "probe_foundation" in pitch_bias_stats:
            diff = abs(pitch_bias_stats["increase_difficulty"] - pitch_bias_stats["probe_foundation"])
            if diff > 50: # 50Hz difference is significant
                return True
        return False
