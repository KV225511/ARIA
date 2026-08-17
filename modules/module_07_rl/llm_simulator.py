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
from modules.module_07_rl.data_loader import (
    get_random_pair, get_all_pdfs, RESUMES_DIR, JDS_DIR,
    get_specific_pair, is_valid_resume, is_valid_jd
)
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE

# Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CANDIDATE_MODEL = os.getenv("ARIA_CANDIDATE_MODEL", "qwen2.5:7b")
# Keep candidate generation and evaluation independent by default. Operators can
# override either model explicitly, but should not point both at the same model.
EVALUATOR_MODEL = os.getenv("ARIA_EVALUATOR_MODEL", "llama3.1")
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

async def evaluate_answer(question: str, answer: str) -> tuple:
    """Return scores plus a validity flag without consulting persona labels."""
    prompt = f"""Evaluate the following interview answer strictly and objectively against standard technical hiring rubrics.

Question: {question}
Answer: {answer}

Score calibration guide:
- semantic_score:
  * 0.0 - 0.3 (Beginner): Vague, incorrect, 'I do not know', or fundamentally flawed explanations.
  * 0.3 - 0.6 (Mid-Level): Partially accurate, understands basic concepts but misses edge-cases, system architecture, or depth.
  * 0.6 - 1.0 (Expert): Highly accurate, comprehensive, cites proper terminology, architecture patterns, trade-offs, and best practices.

- behavior_score:
  * 0.0 - 0.4: Disorganized, rambling, evasive, or low confidence.
  * 0.4 - 0.7: Clear communication with occasional hesitation or minor structure gaps.
  * 0.7 - 1.0: Articulate, structured, concise, and professional delivery.

- cog_load:
  * 'low': Answered smoothly with natural ease.
  * 'anxiety': High hesitation, repetitive stalling, nervous filler words, or stress.
  * 'ignorance': Candidate clearly does not know the subject and states or demonstrates lack of knowledge.

Output format requirement:
Return ONLY a valid JSON object without markdown fences, comments, or extra text:
{{"semantic_score": 0.0, "behavior_score": 0.0, "cog_load": "low"}}
"""
    
    eval_text = await generate_llm_response(prompt, EVALUATOR_MODEL)
    
    # A deterministic neutral fallback makes evaluator failures auditable and
    # reproducible. Invalid turns are tagged in the generated dataset.
    semantic_score = 0.5
    behavior_score = 0.5
    cog_load = "low"
    evaluation_valid = False
    
    try:
        if "```json" in eval_text:
            eval_text = eval_text.split("```json")[1].split("```")[0]
        elif "```" in eval_text:
            eval_text = eval_text.split("```")[1].split("```")[0]
            
        data = json.loads(eval_text.strip())
        semantic_score = float(data.get("semantic_score", semantic_score))
        behavior_score = float(data.get("behavior_score", behavior_score))
        cog_load = data.get("cog_load", cog_load)
        if cog_load not in {"low", "anxiety", "ignorance"}:
            raise ValueError(f"Invalid cognitive-load label: {cog_load}")
        if not (0.0 <= semantic_score <= 1.0 and 0.0 <= behavior_score <= 1.0):
            raise ValueError("Evaluator scores must be in [0, 1]")
        evaluation_valid = True
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
        
    return semantic_score, behavior_score, cog_load, evaluation_valid

def get_action_one_hot(action, action_dim):
    one_hot = np.zeros(action_dim, dtype=np.float32)
    one_hot[action] = 1.0
    return one_hot.tolist()

async def simulate_episode(ep: int, pair: tuple | None, total_eps: int, semaphore: asyncio.Semaphore) -> list:
    """Simulates a single episode, isolated to its own environment to avoid state conflicts."""
    async with semaphore:
        if pair is not None:
            resume_name, jd_name = pair
            resume_text, jd_text, _, _ = get_specific_pair(resume_name, jd_name)
        else:
            resume_text, jd_text, resume_name, jd_name = get_random_pair()
            
        print(f"\n--- Starting Episode {ep+1}/{total_eps} | Resume: {resume_name} | JD: {jd_name} ---")
        
        # Isolated env per task
        env = ARIAInterviewEnv("backend_developer")
        interviewer = LLMQuestionGenerator(model=CANDIDATE_MODEL)
        
        env.ontology.adapt_to_candidate(jd_text, resume_text)
        env.sync_ontology_nodes()
        obs, _ = env.reset()
        
        # Randomly assign skill tier per episode to ensure balanced class distribution
        persona_tier = random.choice(["BEGINNER", "MID", "EXPERT"])
        personas = {
            "BEGINNER": (
                "You are a fresh CS graduate with very limited experience. "
                "Give short, uncertain answers. Frequently admit you don't know something. "
                "Use vague language like 'I think', 'maybe', 'I'm not sure'. Give incorrect details sometimes."
            ),
            "MID": (
                f"You are a mid-level developer with 2-4 years of experience. "
                f"Base your answers on this resume context: {resume_text[:1000]}\n"
                f"Give competent but occasionally incomplete answers. Hesitate on advanced topics."
            ),
            "EXPERT": (
                "You are a senior engineer with 8+ years of experience. "
                "Give confident, technically deep answers. Cite specific tools, patterns, and trade-offs. "
                "Use precise terminology and give concrete real-world examples."
            ),
        }
        system_prompt = f"""You are a candidate interviewing for a position.
Your assigned skill persona for this interview: {persona_tier}
{personas[persona_tier]}
Keep your answers conversational and concise (2-4 sentences max).
"""
        print(f"  Persona: {persona_tier}")
        

        history = []
        done = False
        episode_transitions = []
        
        # Episode-level ground truth label derived from persona tier (not per-turn sem_score)
        persona_label_map = {"BEGINNER": 0, "MID": 1, "EXPERT": 2}
        episode_true_label = persona_label_map[persona_tier]
        
        # Map persona tier to role/experience for question generator
        experience_map = {"BEGINNER": "Entry-Level", "MID": "Mid-Level", "EXPERT": "Senior"}
        candidate_experience = experience_map[persona_tier]
        while not done:
            action_idx = env.action_space.sample()
            action_name = RL_ACTION_SPACE[action_idx]

            # Coverage-aware mask: the environment owns the conclusion contract.
            if action_name == "conclude_interview" and not env.can_conclude():
                non_conclude = [i for i in range(env.action_space.n) if RL_ACTION_SPACE[i] != "conclude_interview"]
                action_idx = random.choice(non_conclude)
                action_name = RL_ACTION_SPACE[action_idx]

            target_skill = env.select_target_skill(action_idx)
            
            belief_state = {k: v.tolist() for k, v in env.belief_updater.beliefs.items()}
            
            # 1. Interviewer generates question (pass persona experience for prompt calibration)
            question = await interviewer.generate_question(
                action=action_name,
                belief_state=belief_state,
                resume=resume_text[:1500],
                history=history,
                experience=candidate_experience,
                target_skill=target_skill,
            )
            
            # Guard: skip turn if question is empty (Ollama timeout / model hiccup)
            if not question or not question.strip():
                print(f"  [WARN] Episode {ep+1}: Empty question generated for action '{action_name}'. Skipping turn.")
                continue
            
            # 2. Candidate answers
            answer = await generate_llm_response(question, CANDIDATE_MODEL, system=system_prompt)
            if not answer or not answer.strip():
                answer = "I'm not sure about that."
            history.append({"q": question, "a": answer})
            
            # 3. Evaluate
            sem_score, beh_score, cog_load, evaluation_valid = await evaluate_answer(
                question, answer
            )
            if answer == "I'm not sure about that.":
                sem_score, beh_score, cog_load = 0.1, 0.2, "ignorance"
            
            # 4. Environment Step
            next_obs, reward, terminated, truncated, info = env.step_with_scores(
                action_idx, sem_score, beh_score, cog_load,
                target_skill=target_skill,
            )
            
            done = terminated or truncated
            
            # Use episode-level persona as ground truth (not noisy per-turn sem_score)
            true_label = episode_true_label
                
            # Global verdict uses only skills with evidence; untouched ontology
            # nodes cannot dominate the terminal class.
            assessment = env.belief_updater.get_aggregate_assessment()
            aria_label = assessment["label"]
            
            # 5. Save Transition
            transition = {
                "obs": obs.tolist(),
                "action": get_action_one_hot(action_idx, env.action_space.n),
                "action_idx": int(action_idx),
                "reward": float(reward),
                "next_obs": next_obs.tolist(),
                "done": bool(done),
                "resume_file": resume_name,
                "jd_file": jd_name,
                "true_label": true_label,
                "aria_label": aria_label,
                "aggregate_belief": assessment["belief"].tolist(),
                "aggregate_confidence": assessment["confidence"],
                "skills_covered": len(assessment["visited_skills"]),
                "target_skill": target_skill,
                "semantic_score": float(sem_score),
                "behavior_score": float(beh_score),
                "cognitive_load": cog_load,
                "evaluation_valid": evaluation_valid,
                "episode_id": f"episode_{ep}",
                "question": question,
                "jd_text": jd_text[:2000]
            }
            episode_transitions.append(transition)
            obs = next_obs
            
        print(f"--- Finished Episode {ep+1}/{total_eps} | Transitions: {len(episode_transitions)} ---")
        return episode_transitions

async def run_simulation(sweep: bool = False, max_episodes: int = 1000, max_concurrent: int = 5):
    os.makedirs(DATASET_FILE.parent, exist_ok=True)
    
    dataset = []
    if DATASET_FILE.exists() and not sweep:
        try:
            with open(DATASET_FILE, "r") as f:
                dataset = json.load(f)
            print(f"Loaded existing dataset with {len(dataset)} transitions.")
        except Exception:
            pass
            
    if sweep:
        all_jds = get_all_pdfs(JDS_DIR)
        all_resumes = get_all_pdfs(RESUMES_DIR)

        # Apply the same blocklists as get_random_pair()
        jds = [j for j in all_jds if is_valid_jd(j)]
        resumes = [r for r in all_resumes if is_valid_resume(r)]

        print(f"Sweep pool: {len(resumes)} valid resumes x {len(jds)} valid JDs "
              f"(filtered from {len(all_resumes)} resumes / {len(all_jds)} JDs)")

        # Generate all valid combinations and shuffle
        episodes_to_run = []
        for j in jds:
            for r in resumes:
                episodes_to_run.append((r.name, j.name))

        random.shuffle(episodes_to_run)

        if len(episodes_to_run) > max_episodes:
            print(f"Sweep generated {len(episodes_to_run)} pairs. Capping at {max_episodes} for performance.")
            episodes_to_run = episodes_to_run[:max_episodes]

        print(f"Starting Multi-Agent Sweep Simulation with {len(episodes_to_run)} total episodes...")
        dataset = [] # Start fresh for sweep
    else:
        num_eps = min(2, max_episodes) # fallback for small tests
        print(f"Starting Multi-Agent Simulation with {num_eps} random episodes...")
        episodes_to_run = [None] * num_eps

    semaphore = asyncio.Semaphore(max_concurrent)
    total_eps = len(episodes_to_run)
    
    # Run episodes concurrently
    tasks = [
        simulate_episode(ep, pair, total_eps, semaphore) 
        for ep, pair in enumerate(episodes_to_run)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Flatten the results
    for ep_transitions in results:
        dataset.extend(ep_transitions)
            
    # Save dataset
    with open(DATASET_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nSimulation complete! Dataset saved with {len(dataset)} total transitions.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run 1 episode for every JD-Resume permutation")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Maximum number of episodes to run (caps the sweep)")
    parser.add_argument("--max_concurrent", type=int, default=5, help="Number of concurrent LLM requests to make")
    args = parser.parse_args()
    
    asyncio.run(run_simulation(
        sweep=args.sweep, 
        max_episodes=args.max_episodes,
        max_concurrent=args.max_concurrent
    ))
