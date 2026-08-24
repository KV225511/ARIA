import httpx
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv

class LLMQuestionGenerator:
    def __init__(
        self,
        ollama_host=None,
        model=None,
        keep_alive=None,
        num_ctx=None,
    ):
        """
        Initializes the LLM Question Generator using a local Ollama instance.
        """
        load_dotenv()
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.keep_alive = (
            keep_alive
            if keep_alive is not None
            else os.getenv("ARIA_OLLAMA_KEEP_ALIVE", "-1")
        )
        self.num_ctx = int(
            num_ctx
            if num_ctx is not None
            else os.getenv("ARIA_OLLAMA_NUM_CTX", "4096")
        )
        self.api_endpoint = f"{self.ollama_host}/api/generate"

    async def generate_question(self, action: str, belief_state: dict, resume: str, history: list, role: str = "Developer", experience: str = "Mid-Level", target_skill: str | None = None) -> str:
        """
        Generates a natural language question based on the RL agent's action and the candidate's state.
        
        Args:
            action: The discrete action selected by the RL agent (e.g., 'probe_foundation').
            belief_state: The current Bayesian belief probabilities for skill nodes.
            resume: The candidate's resume text.
            history: The list of previous Q&A turns.
            role: The target role being interviewed for (default: 'Developer').
            experience: Candidate experience tier ('Entry-Level', 'Mid-Level', 'Senior').
            
        Returns:
            str: The generated question text.
        """
        prompt = self._build_prompt(
            action, belief_state, resume, history, role, experience, target_skill
        )
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.7,
                "num_ctx": self.num_ctx,
            }
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.api_endpoint, json=payload, timeout=300.0)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Failed to fetch from Ollama at {self.ollama_host}: {e}")
            return f"Fallback Question: I see the action is {action}. Can you tell me more about your experience?"

    async def generate_question_stream(self, action: str, belief_state: dict, resume: str, history: list, role: str = "Developer", experience: str = "Mid-Level", target_skill: str | None = None):
        """
        Generates a natural language question and yields it word-by-word (streaming).
        """
        prompt = self._build_prompt(
            action, belief_state, resume, history, role, experience, target_skill
        )
        
        yield {"type": "prompt_debug", "prompt": prompt}
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.7,
                "num_ctx": self.num_ctx,
            }
        }
        
        chunk_yielded = False
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", self.api_endpoint, json=payload, timeout=httpx.Timeout(60.0, read=300.0)) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
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

    def _build_prompt(self, action: str, belief_state: dict, resume: str, history: list, role: str = "Developer", experience: str = "Mid-Level", target_skill: str | None = None) -> str:
        history_text = "\n".join([f"Q: {t['q']}\nA: {t['a']}" for t in history[-5:]]) if history else "No previous questions."
        
        # Calculate entropy to find top 5 uncertain skills
        def calc_entropy(dist):
            return -sum(p * (0 if p <= 0 else __import__('math').log(p)) for p in dist)
            
        entropies = {skill: calc_entropy(dist) for skill, dist in belief_state.items()}
        top_skills = sorted(entropies.keys(), key=lambda k: entropies[k], reverse=True)[:5]
        filtered_belief_state = {k: belief_state[k] for k in top_skills}
        
        action_guide = {
            "increase_difficulty": "Ask a challenging, advanced architectural or edge-case question on the current topic. Assume solid basics.",
            "decrease_difficulty": "Ask a simpler, foundational question focusing on core concepts and basic syntax/principles.",
            "ask_follow_up_same_topic": "Ask a deep follow-up or require a concrete real-world example on the exact same topic just discussed.",
            "switch_topic": "Transition cleanly to a new topic from the most uncertain skills list.",
            "probe_foundation": "Ask about the fundamental internal mechanisms or mathematical/computational theory behind the concept.",
            "ask_behavioral": "Ask a STAR-method behavioral question: 'Tell me about a time you handled...'",
            "ask_situational": "Present a realistic workplace engineering challenge: 'Suppose our system experiences... how would you diagnose and fix it?'",
            "conclude_interview": "Ask a final concluding question inviting them to summarize their key strengths or discuss engineering trade-offs."
        }
        
        belief_summary = []
        for skill, dist in filtered_belief_state.items():
            dist_list = list(dist)
            max_idx = int(np.argmax(dist_list))
            level = ["Beginner", "Mid", "Expert"][max_idx]
            conf = round(float(dist_list[max_idx]), 2)
            belief_summary.append(f"- {skill}: Currently assessed as {level} (confidence: {conf})")
            
        belief_summary_str = "\n".join(belief_summary) if belief_summary else "No skill beliefs recorded yet."
        
        target_skill_text = target_skill or "the most uncertain relevant skill"

        return f"""You are ARIA, an expert, objective technical interviewer conducting an assessment for a {experience} {role} position.

CANDIDATE RESUME CONTEXT:
{resume[:1000]}

RECENT CONVERSATION HISTORY (Last turns):
{history_text}

CURRENT SKILL BELIEF STATE (Most uncertain skills):
{belief_summary_str}

RL AGENT DIRECTIVE:
- Selected Action: {action}
- Required Target Skill: {target_skill_text}
- Action Guidance: {action_guide.get(action, 'Ask a relevant technical question matching the target skill level.')}

CRITICAL RULES:
1. Generate exactly ONE clear, concise, direct question about the Required Target Skill and execute the RL action directive above.
2. STRICTLY do NOT repeat or rephrase any question from the conversation history.
3. Tone and complexity MUST align with a {experience} {role}.
4. Output ONLY the question text. Do not include introductory filler, greetings, or conversational remarks.
"""
