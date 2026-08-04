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
            response = requests.post(self.api_endpoint, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Failed to stream from Ollama at {self.ollama_host}: {e}")
            yield f"Fallback Question: I see the action is {action}. Can you tell me more about your experience?"

    def generate_question_stream(self, action: str, belief_state: dict, resume: str, history: list, role: str = "Developer", experience: str = "Mid-Level"):
        """
        Generates a natural language question and yields it word-by-word (streaming).
        """
        prompt = self._build_prompt(action, belief_state, resume, history, role, experience)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "num_ctx": 16384
            }
        }
        
        chunk_yielded = False
        try:
            response = requests.post(self.api_endpoint, json=payload, stream=True, timeout=(60, 300))
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            chunk_yielded = True
                            yield chunk
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to stream from Ollama at {self.ollama_host}: {e}")
            if not chunk_yielded:
                yield f"Fallback Question: I see the action is {action}. Can you tell me more about your experience?"
            return
            
        if not chunk_yielded:
            yield f"Fallback Question: I see the action is {action}. Can you tell me more about your experience?"

    def _build_prompt(self, action: str, belief_state: dict, resume: str, history: list, role: str, experience: str) -> str:
        history_text = "\n".join([f"Q: {t['q']}\nA: {t['a']}" for t in history[-2:]]) if history else "No previous questions."
        
        return f"""
You are ARIA, an expert technical interviewer interviewing a candidate for a {experience} {role} position.
CRITICAL INSTRUCTION: Adjust the tone, expectations, and complexity of your questions to perfectly match this {experience} level. For example, do not ask a fresher deep system design questions, and do not ask a senior developer basic syntax questions.

Candidate Resume Context:
{resume[:500]}...

Recent Conversation History:
{history_text}

Current Belief State of Candidate Skills:
{json.dumps(belief_state)}

The RL Agent has decided the next action is: {action}

Write a natural, conversational question that executes this action. Do not include any pleasantries or introductory text, just the question itself.
"""
