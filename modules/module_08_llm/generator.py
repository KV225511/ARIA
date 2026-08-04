import requests
import json
import logging

logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv

load_dotenv()

class LLMQuestionGenerator:
    def __init__(self, ollama_host=None, model=None):
        """
        Initializes the LLM Question Generator using a local Ollama instance.
        """
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self.api_endpoint = f"{self.ollama_host}/api/generate"

    def generate_question(self, action: str, belief_state: dict, resume: str, history: list) -> str:
        """
        Generates a natural language question based on the RL agent's action and the candidate's state.
        
        Args:
            action: The discrete action selected by the RL agent (e.g., 'probe_foundation').
            belief_state: The current Bayesian belief probabilities for skill nodes.
            resume: The candidate's resume text.
            history: The list of previous Q&A turns.
            
        Returns:
            str: The generated question text.
        """
        prompt = self._build_prompt(action, belief_state, resume, history)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        }
        
        try:
            response = requests.post(self.api_endpoint, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama at {self.ollama_host}: {e}")
            # Fallback for testing/offline gracefully
            return f"Fallback Question: I see the action is {action}. Can you tell me more about your experience?"

    def _build_prompt(self, action: str, belief_state: dict, resume: str, history: list) -> str:
        history_text = "\n".join([f"Q: {t['q']}\nA: {t['a']}" for t in history[-3:]]) if history else "No previous questions."
        
        return f"""
You are ARIA, an expert technical interviewer.
Based on the current interview state, generate the next interview question to ask the candidate.

Candidate Resume Context:
{resume[:500]}...

Recent Conversation History:
{history_text}

Current Belief State of Candidate Skills:
{json.dumps(belief_state)}

The RL Agent has decided the next action is: {action}

Write a natural, conversational question that executes this action. Do not include any pleasantries or introductory text, just the question itself.
"""
