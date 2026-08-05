import asyncio
import json
import os
import sys
import random
import httpx
import numpy as np
from pathlib import Path

# Adjust imports to local module structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.module_07_rl.environment import ARIAInterviewEnv
from modules.module_08_llm.generator import LLMQuestionGenerator

# Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CANDIDATE_MODEL = "qwen2.5:7b"
EVALUATOR_MODEL = "qwen2.5:7b"
NUM_EPISODES = 2
DATASET_FILE = Path(__file__).parent.parent.parent / "data" / "synthetic" / "qwen_rl_dataset.json"

async def generate_llm_response(prompt: str, model: str, system: str = "") -> str:
    """Helper to call Ollama directly"""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.8}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120.0)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"Ollama API Error: {e}")
            return ""

def generate_persona():
    personas = [
        {"role": "Backend Developer", "strong": ["Python", "SQL", "APIs"], "weak": ["Docker", "Kubernetes", "CI/CD"]},
        {"role": "Backend Developer", "strong": ["System Design", "Microservices"], "weak": ["Algorithms", "Data Structures"]},
        {"role": "Backend Developer", "strong": ["Algorithms", "Python"], "weak": ["System Design", "Databases"]}
    ]
    return random.choice(personas)

async def evaluate_answer(question: str, answer: str) -> tuple:
    """Uses LLM to evaluate the answer and return (semantic_score, behavior_score, cog_load)"""
    prompt = f"""
    Evaluate the following answer to the interview question.
    Question: {question}
    Answer: {answer}
    
    Provide your evaluation in exactly this JSON format, nothing else:
    {{"semantic_score": 0.0, "behavior_score": 0.0, "cog_load": "low"}}
    
    - semantic_score: How technically correct and relevant the answer is (0.0 to 1.0, where 1.0 is perfect).
    - behavior_score: How confident and well-structured the communication is (0.0 to 1.0).
    - cog_load: 'low' if they answered easily, 'anxiety' if they seem stressed/rambling, 'ignorance' if they don't know it.
    """
    
    eval_text = await generate_llm_response(prompt, EVALUATOR_MODEL)
    
    # Fallbacks
    semantic_score = np.random.uniform(0.3, 0.7)
    behavior_score = np.random.uniform(0.3, 0.7)
    cog_load = "low"
    
    try:
        # Try to extract json block if there's markdown
        if "```json" in eval_text:
            eval_text = eval_text.split("```json")[1].split("```")[0]
        elif "```" in eval_text:
            eval_text = eval_text.split("```")[1].split("```")[0]
            
        data = json.loads(eval_text.strip())
        semantic_score = float(data.get("semantic_score", semantic_score))
        behavior_score = float(data.get("behavior_score", behavior_score))
        cog_load = data.get("cog_load", cog_load)
    except Exception as e:
        print(f"Evaluation parsing failed, using fallbacks. Error: {e}")
        
    return semantic_score, behavior_score, cog_load

def get_action_one_hot(action, action_dim):
    one_hot = np.zeros(action_dim, dtype=np.float32)
    one_hot[action] = 1.0
    return one_hot.tolist()

async def run_simulation():
    print(f"Starting Multi-Agent Simulation with {NUM_EPISODES} episodes...")
    
    env = ARIAInterviewEnv("backend_developer")
    interviewer = LLMQuestionGenerator(model=CANDIDATE_MODEL)
    
    os.makedirs(DATASET_FILE.parent, exist_ok=True)
    
    dataset = []
    if DATASET_FILE.exists():
        try:
            with open(DATASET_FILE, "r") as f:
                dataset = json.load(f)
            print(f"Loaded existing dataset with {len(dataset)} transitions.")
        except Exception:
            pass
            
    for ep in range(NUM_EPISODES):
        persona = generate_persona()
        print(f"\n--- Episode {ep+1}/{NUM_EPISODES} | Persona: Strong {persona['strong']}, Weak {persona['weak']} ---")
        
        system_prompt = f"""
        You are a candidate interviewing for a {persona['role']} position. 
        Your strong skills are: {', '.join(persona['strong'])}. Answer confidently and correctly about these.
        Your weak skills are: {', '.join(persona['weak'])}. If asked about these, hesitate, give partial/incorrect answers, or admit you don't know much.
        Keep your answers conversational and concise (2-4 sentences max).
        """
        
        obs, _ = env.reset()
        history = []
        done = False
        
        while not done:
            action_idx = env.action_space.sample()
            from modules.module_07_rl.rl_spec import RL_ACTION_SPACE
            action_name = RL_ACTION_SPACE[action_idx]
            
            belief_state = {k: v.tolist() for k, v in env.belief_updater.beliefs.items()}
            
            print(f"\nTurn {env.turn_id+1} | Action: {action_name}")
            
            # 1. Interviewer generates question
            question = await interviewer.generate_question(
                action=action_name,
                belief_state=belief_state,
                resume=f"Experienced {persona['role']}.",
                history=history
            )
            print(f"ARIA: {question}")
            
            # 2. Candidate answers
            answer = await generate_llm_response(question, CANDIDATE_MODEL, system=system_prompt)
            print(f"Candidate: {answer}")
            
            history.append({"q": question, "a": answer})
            
            # 3. Evaluate
            sem_score, beh_score, cog_load = await evaluate_answer(question, answer)
            print(f"Evaluation: Semantic={sem_score:.2f}, Behavior={beh_score:.2f}, Load={cog_load}")
            
            # 4. Environment Step
            next_obs, reward, terminated, truncated, info = env.step_with_scores(
                action_idx, sem_score, beh_score, cog_load
            )
            
            done = terminated or truncated
            
            # 5. Save Transition
            transition = {
                "obs": obs.tolist(),
                "action": get_action_one_hot(action_idx, env.action_space.n),
                "action_idx": int(action_idx),
                "reward": float(reward),
                "next_obs": next_obs.tolist(),
                "done": bool(done)
            }
            dataset.append(transition)
            obs = next_obs
            
    # Save dataset
    with open(DATASET_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nSimulation complete! Dataset saved with {len(dataset)} total transitions.")

if __name__ == "__main__":
    asyncio.run(run_simulation())
