import json
import logging
from collections import defaultdict, deque
import statistics
import itertools

logger = logging.getLogger(__name__)

# Configurable Constants
PITCH_BIAS_THRESHOLD_HZ = 50.0
SPEECH_RATE_BIAS_THRESHOLD = 1.5

class InterviewFairnessAuditor:
    def __init__(self, max_turns_per_session=1000):
        """
        Runs as a background audit process, tracking RL action distribution 
        against demographic-adjacent speech patterns (e.g. pitch).
        """
        self.session_data = defaultdict(lambda: deque(maxlen=max_turns_per_session))
        
    def log_turn(self, session_id: str, prosody_features: dict, rl_action: str):
        """
        Logs a single turn's features and the subsequent RL action chosen.
        """
        if not prosody_features:
            return
            
        self.session_data[session_id].append({
            "pitch_hz": prosody_features.get("pitch_f0_hz", 0),
            "speech_rate": prosody_features.get("speech_rate_syllables_per_sec", 0),
            "action": rl_action
        })
        
    def generate_audit_report(self, session_id: str) -> dict:
        """
        Generates a fairness report showing action distribution across speech patterns.
        """
        session_turns = self.session_data.get(session_id, [])
        if not session_turns:
            return {"status": "insufficient_data"}
            
        action_counts = defaultdict(int)
        pitch_by_action = defaultdict(list)
        rate_by_action = defaultdict(list)
        
        for turn in session_turns:
            action = turn["action"]
            action_counts[action] += 1
            if turn["pitch_hz"] > 0:
                pitch_by_action[action].append(turn["pitch_hz"])
            if turn["speech_rate"] > 0:
                rate_by_action[action].append(turn["speech_rate"])
            
        # Calculate averages
        pitch_bias_stats = {}
        for action, pitches in pitch_by_action.items():
            pitch_bias_stats[action] = round(statistics.mean(pitches), 2) if pitches else 0.0
            
        rate_bias_stats = {}
        for action, rates in rate_by_action.items():
            rate_bias_stats[action] = round(statistics.mean(rates), 2) if rates else 0.0
            
        report = {
            "total_turns_audited": len(session_turns),
            "action_distribution": dict(action_counts),
            "average_pitch_by_action_hz": pitch_bias_stats,
            "average_speech_rate_by_action": rate_bias_stats,
            "bias_detected": self._detect_bias(pitch_bias_stats, rate_bias_stats)
        }
        
        return report
        
    def _detect_bias(self, pitch_bias_stats: dict, rate_bias_stats: dict) -> bool:
        """
        Checks for large discrepancies in pitch or speech rate across any pair of actions.
        """
        # Check all action pairs for pitch bias
        actions = list(pitch_bias_stats.keys())
        for a1, a2 in itertools.combinations(actions, 2):
            if abs(pitch_bias_stats[a1] - pitch_bias_stats[a2]) > PITCH_BIAS_THRESHOLD_HZ:
                return True
                
        # Check all action pairs for speech rate bias
        actions = list(rate_bias_stats.keys())
        for a1, a2 in itertools.combinations(actions, 2):
            if abs(rate_bias_stats[a1] - rate_bias_stats[a2]) > SPEECH_RATE_BIAS_THRESHOLD:
                return True
                
        return False

    def clear_session(self, session_id: str):
        """Cleans up memory for a completed session."""
        if session_id in self.session_data:
            del self.session_data[session_id]
