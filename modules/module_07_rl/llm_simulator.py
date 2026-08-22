import asyncio
import itertools
import json
import os
import sys
import random
import httpx
import numpy as np
from pathlib import Path

# Adjust imports to local module structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.module_07_rl.environment import ARIAInterviewEnv, MIN_SKILLS_COVERED
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
PERSONA_TIERS = ("BEGINNER", "MID", "EXPERT")
ACTION_TO_INDEX = {name: index for index, name in enumerate(RL_ACTION_SPACE)}
DEFAULT_SWEEP_EPISODES = 300
MIN_RECOMMENDED_EPISODES = 200
DATASET_SPLIT_RATIOS = (0.70, 0.15, 0.15)

async def generate_llm_response(
    prompt: str,
    model: str,
    system: str = "",
    temperature: float = 0.8,
) -> str:
    """Helper to call Ollama directly"""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": temperature}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120.0)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"Ollama API Error: {e}")
            return ""

def build_evaluator_prompt(question: str, answer: str) -> str:
    """Build a label-blind rubric prompt with concrete score anchors."""
    return f"""Evaluate the interview answer strictly against the technical question. Judge correctness, depth, trade-offs, and edge-case awareness. Do not reward length, confidence, terminology, or polished writing unless the technical substance supports it.

Question: {question}
Answer: {answer}

Score calibration guide:
- semantic_score:
  * 0.00 - 0.25: Incorrect, confused, irrelevant, or explicitly does not know. A plausible-sounding answer with a fundamental misconception belongs here.
  * 0.40 - 0.60: Correct implementation-level fundamentals, but limited architectural reasoning, trade-off analysis, failure handling, or depth.
  * 0.80 - 0.95: Correct and precise, with relevant architecture, complexity or operational trade-offs, failure modes, and edge cases.
  * Reserve 0.26 - 0.39 and 0.61 - 0.79 for genuinely intermediate cases. Do not cluster answers near 0.50 by default.

Calibration anchors (score the substance in the same way across topics):
1. Question: "What is a database index?"
   Answer: "It stores the whole table in memory, so every query becomes O(1). More indexes always make writes faster."
   semantic_score: 0.10 (fundamental misconceptions)
2. Question: "What is a database index?"
   Answer: "An index is a data structure that speeds up lookups on selected columns, but it uses storage and adds write overhead."
   semantic_score: 0.50 (correct fundamentals, limited design depth)
3. Question: "What is a database index?"
   Answer: "A B-tree index gives logarithmic point/range lookup while adding storage, cache pressure, and write amplification. I would choose column order from query predicates and cardinality, verify with query plans, and account for selectivity, covering indexes, locking, and workload-specific write costs."
   semantic_score: 0.90 (precise mechanisms, selection criteria, trade-offs, and operational checks)

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
{{"semantic_score": 0.0, "behavior_score": 0.0, "cog_load": "low", "confidence": 0.0, "rubric_evidence": ["brief reason"]}}
"""


async def evaluate_answer(question: str, answer: str) -> tuple:
    """Return scores plus a validity flag without consulting persona labels."""
    prompt = build_evaluator_prompt(question, answer)

    # Invalid evaluations are never converted into label-dependent fallback
    # scores. Retry once, then let the simulator reject the turn.
    semantic_score = 0.5
    behavior_score = 0.5
    cog_load = "low"
    confidence = 0.0
    rubric_evidence = []
    evaluation_valid = False

    for _ in range(2):
        eval_text = await generate_llm_response(
            prompt,
            EVALUATOR_MODEL,
            temperature=0.0,
        )
        try:
            if "```json" in eval_text:
                eval_text = eval_text.split("```json")[1].split("```")[0]
            elif "```" in eval_text:
                eval_text = eval_text.split("```")[1].split("```")[0]

            data = json.loads(eval_text.strip())
            semantic_score = float(data["semantic_score"])
            behavior_score = float(data["behavior_score"])
            cog_load = data["cog_load"]
            confidence = float(data.get("confidence", 0.5))
            rubric_evidence = data.get("rubric_evidence", [])
            if cog_load not in {"low", "anxiety", "ignorance"}:
                raise ValueError(f"Invalid cognitive-load label: {cog_load}")
            if not all(
                0.0 <= value <= 1.0
                for value in (semantic_score, behavior_score, confidence)
            ):
                raise ValueError("Evaluator scores and confidence must be in [0, 1]")
            if not isinstance(rubric_evidence, list):
                raise ValueError("rubric_evidence must be a list")
            evaluation_valid = True
            break
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    return (
        semantic_score,
        behavior_score,
        cog_load,
        confidence,
        rubric_evidence,
        evaluation_valid,
    )


def build_candidate_system_prompt(persona_tier: str, resume_text: str) -> str:
    """Create persona-specific answer constraints without exposing them to the judge."""
    if persona_tier not in PERSONA_TIERS:
        raise ValueError(f"Unknown persona tier: {persona_tier}")

    personas = {
        "BEGINNER": (
            "You are a fresh CS graduate with weak command of the subject. Give a short "
            "one- or two-sentence answer. When the topic is unfamiliar, say you do not "
            "know. Otherwise, demonstrate realistic novice misconceptions: confuse one "
            "core term, omit the mechanism, or suggest an incorrect implementation. Do "
            "not add advanced caveats, architecture patterns, or trade-off analysis."
        ),
        "MID": (
            "You are a mid-level developer with 2-4 years of experience. Base answers "
            f"on this resume context when relevant: {resume_text[:1000]}\n"
            "Give a correct, practical two- or three-sentence implementation-level "
            "answer. Mention normal APIs, syntax, or workflow, but usually omit system "
            "architecture, quantitative complexity analysis, rare failure modes, and "
            "cross-service trade-offs."
        ),
        "EXPERT": (
            "You are a senior engineer with 8+ years of experience. Give a technically "
            "precise four- to six-sentence answer. Explain the underlying mechanism and "
            "cite relevant architecture or distributed-systems patterns. Include time, "
            "space, scale, or operational trade-offs when applicable, plus at least one "
            "failure mode or edge case and a concrete production decision. Do not pad "
            "the answer with generic leadership language."
        ),
    }
    return f"""You are a candidate interviewing for a position.
Your assigned skill persona for this interview: {persona_tier}
{personas[persona_tier]}
Stay in persona consistently. Answer only the question asked and remain conversational.
"""


def _three_way_counts(total: int, ratios=DATASET_SPLIT_RATIOS) -> list[int]:
    """Allocate a total across three non-empty groups using largest remainders."""
    if total < 3:
        raise ValueError("At least three items are required for leakage-safe splitting")
    raw = [total * ratio / sum(ratios) for ratio in ratios]
    floors = [int(value) for value in raw]
    counts = floors.copy()
    for index in sorted(
        range(3),
        key=lambda item: raw[item] - floors[item],
        reverse=True,
    )[: total - sum(floors)]:
        counts[index] += 1
    for empty_index in [index for index, count in enumerate(counts) if count == 0]:
        donor_index = max(range(3), key=lambda index: counts[index])
        counts[donor_index] -= 1
        counts[empty_index] = 1
    return counts


def build_split_safe_sweep_pairs(
    resumes: list[Path],
    jds: list[Path],
    max_episodes: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Build disconnected resume/JD components sized for train/val/test.

    A full Cartesian sweep creates one giant bipartite component and therefore
    cannot be split without identity leakage. This builder partitions both
    document sets first, then samples (and, when needed, repeats) pairs only
    within the corresponding partition.
    """
    if max_episodes < 3:
        raise ValueError("At least three episodes are required for a three-way split")
    if len(resumes) < 3 or len(jds) < 3:
        raise ValueError(
            "Leakage-safe sweep generation requires at least three resumes and "
            "three job descriptions"
        )

    rng = random.Random(seed)
    resumes = list(resumes)
    jds = list(jds)
    rng.shuffle(resumes)
    rng.shuffle(jds)

    resume_counts = _three_way_counts(len(resumes))
    jd_counts = _three_way_counts(len(jds))
    episode_counts = _three_way_counts(max_episodes)

    def partitions(items, counts):
        result = []
        cursor = 0
        for count in counts:
            result.append(items[cursor:cursor + count])
            cursor += count
        return result

    resume_partitions = partitions(resumes, resume_counts)
    jd_partitions = partitions(jds, jd_counts)
    episodes_to_run = []
    for split_index, episode_count in enumerate(episode_counts):
        combinations = list(itertools.product(
            resume_partitions[split_index],
            jd_partitions[split_index],
        ))
        rng.shuffle(combinations)
        episodes_to_run.extend(
            (resume.name, jd.name)
            for resume, jd in itertools.islice(
                itertools.cycle(combinations),
                episode_count,
            )
        )

    return episodes_to_run

def get_action_one_hot(action, action_dim):
    one_hot = np.zeros(action_dim, dtype=np.float32)
    one_hot[action] = 1.0
    return one_hot.tolist()


def select_behavior_action(env: ARIAInterviewEnv, rng: random.Random) -> tuple[int, str]:
    """Select actions from a reproducible heuristic/exploration mixture."""
    exploration_roll = rng.random()

    # Twenty percent uniform exploration preserves broad offline-RL support.
    if exploration_roll < 0.20:
        allowed = list(range(env.action_space.n))
        if not env.can_conclude():
            allowed.remove(ACTION_TO_INDEX["conclude_interview"])
        return rng.choice(allowed), "uniform_exploration"

    assessment = env.belief_updater.get_aggregate_assessment()
    coverage = len(assessment["visited_skills"])
    required_coverage = min(MIN_SKILLS_COVERED, env.num_nodes)

    # Prioritize coverage before repeatedly probing already-observed skills.
    if coverage < required_coverage:
        return ACTION_TO_INDEX["switch_topic"], "coverage_heuristic"

    # Conclude only when the environment contract is met and the evidence is
    # reasonably decisive; otherwise seek class-appropriate information.
    if env.can_conclude() and assessment["confidence"] >= 0.72 and rng.random() < 0.35:
        return ACTION_TO_INDEX["conclude_interview"], "confidence_heuristic"

    label = assessment["label"]
    if label is None:
        label = assessment["raw_label"]
    candidates_by_label = {
        0: ("probe_foundation", "decrease_difficulty", "switch_topic"),
        1: ("ask_follow_up_same_topic", "ask_situational", "switch_topic"),
        2: ("increase_difficulty", "ask_situational", "switch_topic"),
    }
    action_name = rng.choice(candidates_by_label[label])
    return ACTION_TO_INDEX[action_name], "belief_heuristic"

async def simulate_episode(
    ep: int,
    pair: tuple | None,
    total_eps: int,
    semaphore: asyncio.Semaphore,
    persona_tier: str | None = None,
    seed: int = 42,
) -> list:
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
        
        # Cycle tiers to guarantee balance without leaking the label into scoring.
        persona_tier = persona_tier or PERSONA_TIERS[ep % len(PERSONA_TIERS)]
        if persona_tier not in PERSONA_TIERS:
            raise ValueError(f"Unknown persona tier: {persona_tier}")
        rng = random.Random(seed + ep)
        system_prompt = build_candidate_system_prompt(persona_tier, resume_text)
        print(f"  Persona: {persona_tier}")
        

        history = []
        done = False
        episode_transitions = []
        
        # Episode-level ground truth label derived from persona tier (not per-turn sem_score)
        persona_label_map = {"BEGINNER": 0, "MID": 1, "EXPERT": 2}
        episode_true_label = persona_label_map[persona_tier]
        
        # Question difficulty must not receive the ground-truth persona label.
        interviewer_experience = getattr(
            env.ontology, "inferred_experience", "Mid-Level"
        )
        consecutive_evaluation_failures = 0
        while not done:
            action_idx, behavior_policy = select_behavior_action(env, rng)
            action_name = RL_ACTION_SPACE[action_idx]

            target_skill = env.select_target_skill(action_idx)
            
            belief_state = {k: v.tolist() for k, v in env.belief_updater.beliefs.items()}
            
            # 1. Interviewer generates question (pass persona experience for prompt calibration)
            question = await interviewer.generate_question(
                action=action_name,
                belief_state=belief_state,
                resume=resume_text[:1500],
                history=history,
                experience=interviewer_experience,
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
            (
                sem_score,
                beh_score,
                cog_load,
                evaluator_confidence,
                rubric_evidence,
                evaluation_valid,
            ) = await evaluate_answer(question, answer)
            if answer == "I'm not sure about that.":
                sem_score, beh_score, cog_load = 0.1, 0.2, "ignorance"
                evaluator_confidence = max(evaluator_confidence, 0.8)
                rubric_evidence = rubric_evidence or ["Candidate explicitly did not know"]
                evaluation_valid = True
            if not evaluation_valid:
                consecutive_evaluation_failures += 1
                history.pop()
                print(
                    f"  [WARN] Episode {ep+1}: invalid evaluator output; "
                    "rejecting turn."
                )
                if consecutive_evaluation_failures >= 3:
                    print(f"  [ERROR] Episode {ep+1}: evaluator repeatedly failed.")
                    return []
                continue
            consecutive_evaluation_failures = 0
            
            # 4. Environment Step
            next_obs, reward, terminated, truncated, info = env.step_with_scores(
                action_idx, sem_score, beh_score, cog_load,
                target_skill=target_skill,
                evaluator_confidence=evaluator_confidence,
                question_fingerprint=question,
            )
            
            done = terminated or truncated
            
            # Use episode-level persona as ground truth (not noisy per-turn sem_score)
            true_label = episode_true_label
                
            # Global verdict uses only skills with evidence; untouched ontology
            # nodes cannot dominate the terminal class.
            assessment = env.belief_updater.get_aggregate_assessment()
            aria_label = assessment["label"]
            if aria_label is None:
                aria_label = assessment["raw_label"]
            
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
                "evaluator_confidence": float(evaluator_confidence),
                "rubric_evidence": rubric_evidence,
                "behavior_policy": behavior_policy,
                "candidate_model": CANDIDATE_MODEL,
                "evaluator_model": EVALUATOR_MODEL,
                "simulation_seed": seed + ep,
                "episode_id": f"episode_{ep}",
                "question": question,
                "jd_text": jd_text[:2000]
            }
            episode_transitions.append(transition)
            obs = next_obs
            
        print(f"--- Finished Episode {ep+1}/{total_eps} | Transitions: {len(episode_transitions)} ---")
        return episode_transitions

async def run_simulation(
    sweep: bool = False,
    max_episodes: int = DEFAULT_SWEEP_EPISODES,
    max_concurrent: int = 5,
    seed: int = 42,
):
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

        episodes_to_run = build_split_safe_sweep_pairs(
            resumes,
            jds,
            max_episodes=max_episodes,
            seed=seed,
        )

        print(
            "Starting leakage-safe sweep simulation with "
            f"{len(episodes_to_run)} total episodes..."
        )
        dataset = [] # Start fresh for sweep
    else:
        num_eps = min(2, max_episodes) # fallback for small tests
        print(f"Starting Multi-Agent Simulation with {num_eps} random episodes...")
        episodes_to_run = [None] * num_eps

    semaphore = asyncio.Semaphore(max_concurrent)
    total_eps = len(episodes_to_run)
    if total_eps < MIN_RECOMMENDED_EPISODES:
        print(
            f"[WARN] Only {total_eps} episodes will be generated; "
            f"at least {MIN_RECOMMENDED_EPISODES} are recommended for reliable "
            "calibration and held-out evaluation."
        )
    
    # Run episodes concurrently
    tasks = [
        simulate_episode(
            ep,
            pair,
            total_eps,
            semaphore,
            persona_tier=PERSONA_TIERS[ep % len(PERSONA_TIERS)],
            seed=seed,
        )
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
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=DEFAULT_SWEEP_EPISODES,
        help="Maximum number of episodes to run (default: 300; caps the sweep)",
    )
    parser.add_argument("--max_concurrent", type=int, default=5, help="Number of concurrent LLM requests to make")
    parser.add_argument("--seed", type=int, default=42, help="Simulation seed")
    args = parser.parse_args()
    
    asyncio.run(run_simulation(
        sweep=args.sweep, 
        max_episodes=args.max_episodes,
        max_concurrent=args.max_concurrent,
        seed=args.seed,
    ))
