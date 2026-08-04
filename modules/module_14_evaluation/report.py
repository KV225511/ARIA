import json
import requests
import logging

logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv

load_dotenv()

class ReportGenerator:
    def __init__(self, ollama_host=None, model=None):
        ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.api_endpoint = f"{ollama_host}/api/generate"
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")

    def generate_report(self, belief_state: dict, cognitive_load_summary: dict, anti_gaming_report: dict, fairness_report: dict) -> dict:
        """
        Generates the final comprehensive interview report.
        """
        
        # 1. Calculate Technical Score
        technical_score = self._calculate_technical_score(belief_state)
        
        # 2. Rule-based Recommendation
        if anti_gaming_report.get("flags_detected", 0) > 1:
            recommendation = "Reject (Integrity Flags)"
        elif technical_score < 40.0:
            recommendation = "Reject (Technical Incompetency)"
        elif technical_score > 75.0:
            recommendation = "Strong Hire"
        else:
            recommendation = "Hire"
            
        # 3. LLM Narrative Generation
        narrative = self._generate_narrative(belief_state, technical_score)
        
        return {
            "candidate_score": technical_score,
            "recommendation": recommendation,
            "narrative_summary": narrative,
            "belief_state_snapshot": belief_state,
            "cognitive_load_profile": cognitive_load_summary,
            "integrity_assessment": anti_gaming_report,
            "fairness_audit": fairness_report
        }

    def _calculate_technical_score(self, belief_state: dict) -> float:
        """
        Converts probability distributions into a 0-100 score.
        Expert weight = 1.0, Mid = 0.5, Beginner = 0.0
        """
        if not belief_state:
            return 0.0
            
        total_score = 0
        nodes = 0
        for skill, probs in belief_state.items():
            # probs = [P(beginner), P(mid), P(expert)]
            node_score = (probs[1] * 0.5 + probs[2] * 1.0) * 100
            total_score += node_score
            nodes += 1
            
        return round(total_score / nodes, 2) if nodes > 0 else 0.0

    def _generate_narrative(self, belief_state: dict, score: float) -> str:
        prompt = f"""
You are an expert technical recruiter summarizing an AI-conducted interview.
The candidate achieved a technical score of {score}/100 based on Bayesian belief modeling.

Here are the final probabilities of their skill levels:
{json.dumps(belief_state)}

Write a short, professional 2-paragraph summary of the candidate's strengths and weaknesses.
Do not include any pleasantries or robotic phrasing like "Here is the summary".
"""
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            response = requests.post(self.api_endpoint, json=payload, timeout=20)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Failed to generate narrative via Ollama: {e}")
            return "Narrative generation failed. See raw technical score for assessment."
