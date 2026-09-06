import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
import re
import shutil
import sys
import random
import time
import uuid
import httpx
import numpy as np
from pathlib import Path

# Adjust imports to local module structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.module_07_rl.environment import ARIAInterviewEnv, MIN_SKILLS_COVERED, MIN_INTERVIEW_TURNS
from modules.module_08_llm.generator import (
    LLMQuestionGenerator,
    build_grounded_fallback_question,
    build_question_retry_correction,
    normalize_ollama_keep_alive,
)
from modules.module_07_rl.data_loader import (
    DEFAULT_CLEANED_RESUME_CSV,
    DEFAULT_RESUME_CATEGORIES,
    JDS_DIR,
    RESUMES_DIR,
    ResumeDocument,
    get_all_pdfs,
    get_resume_documents,
    get_resume_source_manifest,
    get_valid_jd_documents,
    is_valid_resume,
    is_valid_jd,
    load_random_pair,
    load_specific_pair,
)
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE
from modules.module_07_rl.rl_spec import ACTION_SCHEMA_VERSION
from modules.module_07_rl.ollama_client import BoundedOllamaClient
from modules.module_07_rl.state_builder import STATE_SCHEMA_VERSION
from modules.module_07_rl.reward_model import REWARD_SCHEMA_VERSION
from modules.module_07_rl.transition_schema import (
    FALLBACK_QUESTION_TEMPLATE_VERSION,
    GENERATOR_SCHEMA_VERSION,
    TRANSITION_SCHEMA_VERSION,
    has_valid_question_generation_provenance,
)
from modules.module_07_rl.generation_preflight import get_groundable_jd_documents
from modules.module_05_ontology.grounding import (
    GROUNDING_SCHEMA_VERSION,
    ROLE_PROFILE_SCHEMA_VERSION,
    build_pairing_record,
    grounding_contract_hash,
    grounding_packet,
    normalize_generated_question,
    validate_grounded_question,
)
from modules.module_07_rl.dataset_split import (
    SPLIT_NAMES,
    group_transitions_into_episodes,
    split_by_resume_jd_group,
)

# Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_KEEP_ALIVE = normalize_ollama_keep_alive(
    os.getenv("ARIA_OLLAMA_KEEP_ALIVE", "-1")
)
OLLAMA_NUM_CTX = int(os.getenv("ARIA_OLLAMA_NUM_CTX", "4096"))
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("ARIA_OLLAMA_TIMEOUT", "300"))
CANDIDATE_MODEL = os.getenv("ARIA_CANDIDATE_MODEL", "qwen2.5:7b")
# Keep candidate generation and evaluation independent by default. Operators can
# override either model explicitly, but should not point both at the same model.
EVALUATOR_MODEL = os.getenv("ARIA_EVALUATOR_MODEL", "gemma3:4b")
DATASET_FILE = Path(__file__).parent.parent.parent / "data" / "synthetic" / "qwen_rl_dataset.json"
PERSONA_TIERS = ("BEGINNER", "MID", "EXPERT")
ACTION_TO_INDEX = {name: index for index, name in enumerate(RL_ACTION_SPACE)}
DEFAULT_SWEEP_EPISODES = 300
MIN_RECOMMENDED_EPISODES = 200
DATASET_SPLIT_RATIOS = (0.70, 0.15, 0.15)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _can_use_deterministic_grounding_fallback(outputs: list[str]) -> bool:
    """Recover semantic rejections, but never convert an API outage into data."""
    return len(outputs) == 3 and all(
        isinstance(output, str) and bool(output.strip()) for output in outputs
    )


async def generate_llm_response(
    prompt: str,
    model: str,
    system: str = "",
    temperature: float = 0.8,
    client: BoundedOllamaClient | None = None,
) -> str:
    """Helper to call Ollama directly"""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }
    if client is not None:
        try:
            return (await client.generate(payload)).get("response", "").strip()
        except Exception as e:
            print(f"Ollama API Error: {e}")
            return ""
    async with httpx.AsyncClient() as transient_client:
        try:
            response = await transient_client.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=OLLAMA_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"Ollama API Error: {e}")
            return ""

def build_evaluator_prompt(
    question: str,
    answer: str,
    grounding_context: dict | None = None,
) -> str:
    """Build a label-blind rubric prompt with concrete score anchors."""
    grounding_context = grounding_context or {}
    return f"""Evaluate the interview answer strictly against the technical question and supplied competency context. Judge correctness, depth, trade-offs, and edge-case awareness. Do not reward length, confidence, terminology, or polished writing unless the technical substance supports it.

Target competency: {grounding_context.get('target_skill', 'unspecified')}
Competency definition: {grounding_context.get('target_definition', 'unspecified')}
Role domain: {grounding_context.get('role_domain', 'unspecified')}
JD evidence: {json.dumps(grounding_context.get('jd_evidence', []), ensure_ascii=False)}

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


async def evaluate_answer(
    question: str,
    answer: str,
    client: BoundedOllamaClient | None = None,
    grounding_context: dict | None = None,
) -> tuple:
    """Return scores plus a validity flag without consulting persona labels."""
    prompt = build_evaluator_prompt(question, answer, grounding_context)

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
            client=client,
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
    resumes: list[Path | ResumeDocument],
    jds: list[Path],
    max_episodes: int,
    seed: int,
    component_targets: tuple[int, int, int] = (20, 6, 6),
) -> list[tuple[str, str]]:
    """Build many disconnected resume/JD components for train/val/test.

    A full Cartesian sweep creates one giant bipartite component and therefore
    cannot be split without identity leakage. This builder partitions both
    document sets first, then samples (and, when needed, repeats) pairs only
    within the corresponding partition.
    """
    if len(component_targets) != 3 or any(count <= 0 for count in component_targets):
        raise ValueError("component_targets must contain three positive counts")
    required_components = sum(component_targets)
    if max_episodes < required_components:
        raise ValueError(
            f"At least {required_components} episodes are required to populate "
            "every requested identity component"
        )
    if max_episodes < 3:
        raise ValueError("At least three episodes are required for a three-way split")
    if len(resumes) < required_components or len(jds) < required_components:
        raise ValueError(
            "Leakage-safe generation requires one unique resume and JD per "
            f"identity component: requested {required_components}, available "
            f"resumes={len(resumes)}, JDs={len(jds)}"
        )

    rng = random.Random(seed)
    resumes = list(resumes)
    jds = list(jds)
    rng.shuffle(resumes)
    rng.shuffle(jds)

    episode_counts = _three_way_counts(max_episodes)
    if any(episodes < components for episodes, components in zip(
        episode_counts, component_targets
    )):
        raise ValueError("Episode split is too small for the requested component targets")

    def component_pools(items, count):
        pools = [[item] for item in items[:count]]
        for index, item in enumerate(items[count:]):
            pools[index % count].append(item)
        return pools

    resume_cursor = 0
    jd_cursor = 0
    episodes_to_run = []
    for split_index, (episode_count, component_count) in enumerate(zip(
        episode_counts, component_targets
    )):
        remaining_splits = 3 - split_index
        resume_take = component_count + max(
            0, (len(resumes) - resume_cursor - sum(component_targets[split_index:]))
            // remaining_splits
        )
        jd_take = component_count + max(
            0, (len(jds) - jd_cursor - sum(component_targets[split_index:]))
            // remaining_splits
        )
        resume_items = resumes[resume_cursor:resume_cursor + resume_take]
        jd_items = jds[jd_cursor:jd_cursor + jd_take]
        resume_cursor += resume_take
        jd_cursor += jd_take
        if split_index == 2:
            resume_items.extend(resumes[resume_cursor:])
            jd_items.extend(jds[jd_cursor:])

        resume_components = component_pools(resume_items, component_count)
        jd_components = component_pools(jd_items, component_count)
        base, remainder = divmod(episode_count, component_count)
        for component_index in range(component_count):
            component_episodes = base + int(component_index < remainder)
            combinations = list(itertools.product(
                resume_components[component_index],
                jd_components[component_index],
            ))
            rng.shuffle(combinations)
            episodes_to_run.extend(
                (resume.name, jd.name)
                for resume, jd in itertools.islice(
                    itertools.cycle(combinations), component_episodes
                )
            )

    return episodes_to_run


def _atomic_json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_bytes_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".restore.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _format_duration(seconds: float) -> str:
    """Format an ETA without implying more precision than the estimate has."""
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"Dataset must be a JSON list of transitions: {path}")
    return value


def _next_episode_index(transitions: list[dict]) -> int:
    indices = []
    for transition in transitions:
        match = re.fullmatch(r"episode[_-](\d+)", str(transition.get("episode_id", "")))
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


def _backup_before_append(path: Path) -> Path | None:
    if not path.exists():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    backup = path.parent / "backups" / f"{path.stem}.pre_append.{digest}{path.suffix}"
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    return backup


def validate_append_provenance(
    transitions: list[dict],
    candidate_model: str,
    evaluator_model: str,
    allow_model_mix: bool = False,
    resume_source_type: str | None = None,
    resume_source_file_hash: str | None = None,
) -> None:
    if candidate_model == evaluator_model:
        raise ValueError("Candidate and evaluator models must remain distinct")
    if transitions:
        existing_transition_schemas = {
            str(item.get("transition_schema_version") or "missing")
            for item in transitions
        }
        if existing_transition_schemas != {TRANSITION_SCHEMA_VERSION}:
            raise ValueError(
                "Append transition schema is incompatible with the current corpus: "
                f"existing={sorted(existing_transition_schemas)}, "
                f"required={TRANSITION_SCHEMA_VERSION}. Regenerate or use a new output path."
            )
        required_contracts = {
            "generator_schema_version": GENERATOR_SCHEMA_VERSION,
            "role_profile_schema_version": ROLE_PROFILE_SCHEMA_VERSION,
            "question_grounding_schema_version": GROUNDING_SCHEMA_VERSION,
            "grounding_contract_hash": grounding_contract_hash(),
        }
        for field, required in required_contracts.items():
            existing = {
                str(item.get(field) or "missing") for item in transitions
            }
            if existing != {required}:
                raise ValueError(
                    f"Append {field} is incompatible with the current corpus: "
                    f"existing={sorted(existing)}, required={required}. "
                    "Regenerate or use a new output path."
                )
        invalid_generation_records = [
            index
            for index, item in enumerate(transitions)
            if item.get("transition_kind") == "question"
            and not has_valid_question_generation_provenance(item)
        ]
        if invalid_generation_records:
            raise ValueError(
                "Append question-generation provenance is invalid at transition "
                f"indices {invalid_generation_records[:10]}. Regenerate or use a "
                "new output path."
            )
    existing_candidates = {
        str(item["candidate_model"])
        for item in transitions if item.get("candidate_model")
    }
    existing_evaluators = {
        str(item["evaluator_model"])
        for item in transitions if item.get("evaluator_model")
    }
    mismatch = (
        existing_candidates and candidate_model not in existing_candidates
    ) or (
        existing_evaluators and evaluator_model not in existing_evaluators
    )
    if mismatch and not allow_model_mix:
        raise ValueError(
            "Append model provenance differs from the existing corpus. "
            f"Existing candidate={sorted(existing_candidates)}, "
            f"evaluator={sorted(existing_evaluators)}; requested "
            f"candidate={candidate_model}, evaluator={evaluator_model}. "
            "Use --allow-model-mix only after accepting distribution-shift risk."
        )
    if transitions and resume_source_type == "opensporks_csv":
        missing_source_metadata = sum(
            not item.get("resume_source_type") or not item.get("resume_source_file_hash")
            for item in transitions
        )
        if missing_source_metadata:
            raise ValueError(
                "Existing transitions lack cleaned-CSV source provenance; "
                "append is not permitted"
            )
        existing_source_types = {
            str(item["resume_source_type"]) for item in transitions
        }
        existing_source_hashes = {
            str(item["resume_source_file_hash"]) for item in transitions
        }
        if existing_source_types != {resume_source_type}:
            raise ValueError(
                "Append resume source type differs from the existing corpus: "
                f"existing={sorted(existing_source_types)}, requested={resume_source_type}"
            )
        if (
            resume_source_file_hash
            and existing_source_hashes != {resume_source_file_hash}
        ):
            raise ValueError(
                "Append cleaned resume CSV hash differs from the existing corpus"
            )


def _partition_extra_paths(paths: list[Path], seed: int) -> dict[str, list[Path]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    result = {name: [] for name in SPLIT_NAMES}
    for index, path in enumerate(shuffled):
        result[SPLIT_NAMES[index % len(SPLIT_NAMES)]].append(path)
    return result


def build_append_sweep_pairs(
    existing_transitions: list[dict],
    resumes: list[Path | ResumeDocument],
    jds: list[Path],
    max_episodes: int,
    seed: int,
    component_targets: tuple[int, int, int] = (20, 6, 6),
) -> tuple[list[tuple[str, str]], str]:
    """Plan an append without connecting identities across existing splits.

    Completely unused resumes and JDs form new independent components when
    possible. Otherwise new documents are assigned to one existing partition
    and never crossed into another partition.
    """
    if max_episodes < 3:
        raise ValueError("An append needs at least three episodes for split balance")
    resume_by_name = {path.name: path for path in resumes}
    jd_by_name = {path.name: path for path in jds}
    used_resumes = {
        str(item["resume_file"])
        for item in existing_transitions if item.get("resume_file")
    }
    used_jds = {
        str(item["jd_file"])
        for item in existing_transitions if item.get("jd_file")
    }
    unused_resumes = [path for path in resumes if path.name not in used_resumes]
    unused_jds = [path for path in jds if path.name not in used_jds]

    if len(unused_resumes) >= 3 and len(unused_jds) >= 3:
        return (
            build_split_safe_sweep_pairs(
                unused_resumes,
                unused_jds,
                max_episodes=max_episodes,
                seed=seed,
                component_targets=component_targets,
            ),
            "new_identity_components",
        )

    existing_splits = split_by_resume_jd_group(existing_transitions, seed=42)
    resume_pools = {name: set() for name in SPLIT_NAMES}
    jd_pools = {name: set() for name in SPLIT_NAMES}
    for split_name, items in existing_splits.items():
        resume_pools[split_name].update(
            str(item["resume_file"]) for item in items if item.get("resume_file")
        )
        jd_pools[split_name].update(
            str(item["jd_file"]) for item in items if item.get("jd_file")
        )

    extra_resumes = _partition_extra_paths(unused_resumes, seed)
    extra_jds = _partition_extra_paths(unused_jds, seed + 1)
    for split_name in SPLIT_NAMES:
        resume_pools[split_name].update(path.name for path in extra_resumes[split_name])
        jd_pools[split_name].update(path.name for path in extra_jds[split_name])
        missing_resumes = resume_pools[split_name] - resume_by_name.keys()
        missing_jds = jd_pools[split_name] - jd_by_name.keys()
        if missing_resumes or missing_jds:
            raise ValueError(
                "Existing dataset references documents missing from disk: "
                f"resumes={sorted(missing_resumes)}, jds={sorted(missing_jds)}"
            )
        if not resume_pools[split_name] or not jd_pools[split_name]:
            raise ValueError(
                f"Cannot append safely because the {split_name} identity pool is empty"
            )

    episode_counts = _three_way_counts(max_episodes)
    used_pairs = {
        (str(item.get("resume_file")), str(item.get("jd_file")))
        for item in existing_transitions
    }
    rng = random.Random(seed)
    planned = []
    for split_name, episode_count in zip(SPLIT_NAMES, episode_counts):
        combinations = list(itertools.product(
            sorted(resume_pools[split_name]), sorted(jd_pools[split_name])
        ))
        rng.shuffle(combinations)
        fresh = [pair for pair in combinations if pair not in used_pairs]
        ordered = fresh + [pair for pair in combinations if pair in used_pairs]
        planned.extend(itertools.islice(itertools.cycle(ordered), episode_count))
    return planned, "existing_identity_partitions"


async def report_ollama_capacity(gpu_vram_gb: float) -> dict:
    """Report weight-size feasibility; actual residency is verified by ollama ps."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{OLLAMA_HOST}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        sizes = {item.get("name"): int(item.get("size", 0)) for item in models}
        running_response = await client.get(f"{OLLAMA_HOST}/api/ps")
        running_response.raise_for_status()
        running = running_response.json().get("models", [])
    selected = {
        CANDIDATE_MODEL: sizes.get(CANDIDATE_MODEL),
        EVALUATOR_MODEL: sizes.get(EVALUATOR_MODEL),
    }
    missing = [name for name, size in selected.items() if not size]
    if missing:
        raise ValueError(f"Ollama models are not installed: {missing}")
    weights_gb = sum(selected.values()) / 1_000_000_000
    # Reserve is intentionally conservative: Windows display use, CUDA runtime,
    # compute buffers, and both models' KV caches are not included in file sizes.
    safe_budget_gb = max(float(gpu_vram_gb) - 1.25, 0.0)
    result = {
        "candidate_model": CANDIDATE_MODEL,
        "evaluator_model": EVALUATOR_MODEL,
        "model_file_sizes_gb": {
            name: round(size / 1_000_000_000, 3) for name, size in selected.items()
        },
        "combined_weight_files_gb": round(weights_gb, 3),
        "declared_vram_gb": float(gpu_vram_gb),
        "conservative_model_budget_gb": round(safe_budget_gb, 3),
        "full_dual_gpu_residency_feasible": weights_gb <= safe_budget_gb,
        "ollama_num_ctx": OLLAMA_NUM_CTX,
        "ollama_keep_alive": OLLAMA_KEEP_ALIVE,
        "currently_running": running,
    }
    print(json.dumps({"ollama_capacity": result}, indent=2))
    return result

def get_action_one_hot(action, action_dim):
    one_hot = np.zeros(action_dim, dtype=np.float32)
    one_hot[action] = 1.0
    return one_hot.tolist()


def behavior_action_distribution(env: ARIAInterviewEnv) -> tuple[list[float], str]:
    """Return the exact probability of every action under the behavior policy."""
    epsilon = 0.25
    stop_index = ACTION_TO_INDEX["conclude_interview"]
    action_mask = env.get_action_mask().tolist()
    question_indices = [
        index for index in range(stop_index) if action_mask[index] > 0.0
    ]
    if not question_indices:
        raise RuntimeError("No legal question action is available")
    assessment = env.belief_updater.get_aggregate_assessment()
    coverage = len(assessment["visited_skills"])
    required_coverage = min(MIN_SKILLS_COVERED, env.num_nodes)
    if coverage < required_coverage:
        heuristic_indices = [ACTION_TO_INDEX["switch_topic"]]
        policy_name = "coverage_heuristic"
    else:
        label = assessment["label"]
        if label is None:
            label = assessment["raw_label"]
        candidates_by_label = {
            0: ("probe_foundation", "decrease_difficulty", "switch_topic"),
            1: ("ask_follow_up_same_topic", "ask_situational", "switch_topic"),
            2: ("increase_difficulty", "ask_situational", "switch_topic"),
        }
        heuristic_indices = [ACTION_TO_INDEX[name] for name in candidates_by_label[label]]
        policy_name = "belief_heuristic"

    heuristic_indices = [
        index for index in heuristic_indices if index in question_indices
    ] or question_indices

    question_probs = {
        index: epsilon / len(question_indices) for index in question_indices
    }
    for index in heuristic_indices:
        question_probs[index] += (1.0 - epsilon) / len(heuristic_indices)

    probabilities = [0.0] * len(RL_ACTION_SPACE)
    if env.can_conclude():
        stop_probability = min(
            0.15 + 0.05 * (env.turn_id - MIN_INTERVIEW_TURNS), 0.60
        )
        for index in question_indices:
            probabilities[index] = (1.0 - stop_probability) * question_probs[index]
        probabilities[stop_index] = stop_probability
        policy_name = f"{policy_name}_with_stop"
    else:
        for index in question_indices:
            probabilities[index] = question_probs[index]

    total = sum(probabilities)
    return [probability / total for probability in probabilities], policy_name


def select_behavior_action(
    env: ARIAInterviewEnv,
    rng: random.Random,
    include_probabilities: bool = False,
):
    probabilities, policy_name = behavior_action_distribution(env)
    roll = rng.random()
    cumulative = 0.0
    action_idx = len(probabilities) - 1
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if roll < cumulative:
            action_idx = index
            break
    if include_probabilities:
        return action_idx, policy_name, probabilities
    return action_idx, policy_name

async def simulate_episode(
    ep: int,
    pair: tuple | None,
    total_eps: int,
    semaphore: asyncio.Semaphore,
    persona_tier: str | None = None,
    seed: int = 42,
    display_number: int | None = None,
    generation_run_id: str = "",
    plan_id: str = "",
    generation_started_at: str = "",
    ollama_client: BoundedOllamaClient | None = None,
    resume_source: str = "csv",
    resume_csv_path: str | Path = DEFAULT_CLEANED_RESUME_CSV,
    resume_categories: tuple[str, ...] | list[str] | None = DEFAULT_RESUME_CATEGORIES,
    failure_diagnostics: dict[int, dict] | None = None,
) -> list:
    """Simulates a single episode, isolated to its own environment to avoid state conflicts."""
    async with semaphore:
        if pair is not None:
            resume_name, jd_name = pair
            loaded_pair = load_specific_pair(
                resume_name,
                jd_name,
                resume_source=resume_source,
                resume_csv_path=resume_csv_path,
                resume_categories=resume_categories,
            )
        else:
            loaded_pair = load_random_pair(
                resume_source=resume_source,
                resume_csv_path=resume_csv_path,
                resume_categories=resume_categories,
            )
        resume_text = loaded_pair.resume_text
        jd_text = loaded_pair.jd_text
        resume_name = loaded_pair.resume_id
        jd_name = loaded_pair.jd_id
            
        display_number = display_number if display_number is not None else ep + 1
        print(
            f"\n--- Starting New Episode {display_number}/{total_eps} "
            f"| ID: episode_{ep} | Resume: {resume_name} | JD: {jd_name} ---"
        )
        
        # Isolated env per task
        env = ARIAInterviewEnv("backend_developer")
        # Synthetic data must fail closed. A live-interview fallback question
        # is useful for UX, but converting an Ollama outage into training data
        # silently corrupts semantic labels.
        interviewer = LLMQuestionGenerator(
            model=CANDIDATE_MODEL,
            allow_fallback=False,
            client=ollama_client,
        )
        
        if not env.ontology.adapt_to_candidate(jd_text, resume_text):
            raise ValueError(f"Grounded role-profile extraction failed for {jd_name}")
        env.sync_ontology_nodes()
        role_profile = env.ontology.role_profile
        if role_profile is None:
            raise RuntimeError("Ontology adaptation returned no role profile")
        pairing_record = build_pairing_record(role_profile)
        grounding_contract = grounding_contract_hash()
        ontology_nodes = sorted(env.ontology.get_all_skills())
        ontology_edges = sorted(
            (source, target)
            for source, targets in env.ontology.successors.items()
            for target in targets
        )
        ontology_hash = _sha256_text(json.dumps(
            {"nodes": ontology_nodes, "edges": ontology_edges},
            sort_keys=True,
            separators=(",", ":"),
        ))
        obs, _ = env.reset()
        
        # Cycle tiers to guarantee balance without leaking the label into scoring.
        persona_tier = persona_tier or PERSONA_TIERS[ep % len(PERSONA_TIERS)]
        if persona_tier not in PERSONA_TIERS:
            raise ValueError(f"Unknown persona tier: {persona_tier}")
        rng = random.Random(seed + ep)
        system_prompt = build_candidate_system_prompt(persona_tier, resume_text)
        resume_content_hash = loaded_pair.resume_content_hash
        jd_content_hash = loaded_pair.jd_content_hash
        resume_prompt_hash = loaded_pair.resume_prompt_hash
        candidate_system_prompt_hash = _sha256_text(system_prompt)
        print(f"  Persona: {persona_tier}")
        episode_diagnostic = {
            "episode_id": f"episode_{ep}",
            "display_number": display_number,
            "resume_file": resume_name,
            "jd_file": jd_name,
            "persona_tier": persona_tier,
            "role_profile_hash": role_profile.profile_hash,
            "role_profile": role_profile.to_dict(),
            "pairing_record": pairing_record,
            "ontology_hash": ontology_hash,
            "ontology_nodes": ontology_nodes,
            "rejected_question_attempts": [],
            "fallback_questions": [],
            "status": "running",
        }
        if failure_diagnostics is not None:
            failure_diagnostics[ep] = episode_diagnostic
        

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
        while not done:
            action_idx, behavior_policy, behavior_probabilities = select_behavior_action(
                env, rng, include_probabilities=True
            )
            action_name = RL_ACTION_SPACE[action_idx]
            action_mask_before = env.get_action_mask().tolist()

            # A stop decision is a terminal transition, not a synthetic Q&A
            # turn. It therefore makes zero Ollama calls and adds no evidence.
            if action_name == "conclude_interview":
                next_obs, reward, terminated, truncated, info = env.step_with_scores(
                    action_idx, None, None, None
                )
                assessment = env.belief_updater.get_aggregate_assessment()
                aria_label = assessment["label"]
                if aria_label is None:
                    aria_label = assessment["raw_label"]
                transition = {
                    "obs": obs.tolist(),
                    "action": get_action_one_hot(action_idx, env.action_space.n),
                    "action_idx": action_idx,
                    "action_name": action_name,
                    "action_schema_version": ACTION_SCHEMA_VERSION,
                    "action_mask_before": action_mask_before,
                    "behavior_action_probs": behavior_probabilities,
                    "behavior_action_probability": behavior_probabilities[action_idx],
                    "reward": float(reward),
                    "next_obs": next_obs.tolist(),
                    "done": bool(terminated or truncated),
                    "transition_kind": "stop",
                    "termination_reason": info["termination_reason"],
                    "resume_file": resume_name,
                    "jd_file": jd_name,
                    "resume_content_hash": resume_content_hash,
                    "jd_content_hash": jd_content_hash,
                    "resume_prompt_hash": resume_prompt_hash,
                    "resume_source_type": loaded_pair.resume_source_type,
                    "resume_category": loaded_pair.resume_category,
                    "resume_source_file_hash": loaded_pair.resume_source_file_hash,
                    "true_label": episode_true_label,
                    "aria_label": aria_label,
                    "aggregate_belief": assessment["belief"].tolist(),
                    "aggregate_confidence": assessment["confidence"],
                    "skills_covered": len(assessment["visited_skills"]),
                    "valid_evidence_count": env.valid_evidence_count,
                    "target_skill": None,
                    "target_skill_id": None,
                    "target_skill_source": None,
                    "role_profile_hash": role_profile.profile_hash,
                    "role_profile_schema_version": ROLE_PROFILE_SCHEMA_VERSION,
                    "grounding_contract_hash": grounding_contract,
                    "pairing_record": pairing_record,
                    "ontology_hash": ontology_hash,
                    "ontology_nodes": ontology_nodes,
                    "ontology_size": len(ontology_nodes),
                    "question_grounding_schema_version": GROUNDING_SCHEMA_VERSION,
                    "question_grounding_valid": None,
                    "question_grounding": None,
                    "question_generation_attempts": 0,
                    "rejected_question_reasons": [],
                    "semantic_score": None,
                    "behavior_score": None,
                    "cognitive_load": None,
                    "evaluation_valid": None,
                    "evaluator_confidence": None,
                    "rubric_evidence": [],
                    "behavior_policy": behavior_policy,
                    "candidate_model": CANDIDATE_MODEL,
                    "evaluator_model": EVALUATOR_MODEL,
                    "simulation_seed": seed + ep,
                    "episode_id": f"episode_{ep}",
                    "generation_run_id": generation_run_id,
                    "plan_id": plan_id,
                    "generation_started_at": generation_started_at,
                    "generator_schema_version": GENERATOR_SCHEMA_VERSION,
                    "transition_schema_version": TRANSITION_SCHEMA_VERSION,
                    "state_schema_version": STATE_SCHEMA_VERSION,
                    "reward_schema_version": REWARD_SCHEMA_VERSION,
                    "ollama_num_ctx": OLLAMA_NUM_CTX,
                    "ollama_keep_alive": OLLAMA_KEEP_ALIVE,
                    "question": None,
                    "candidate_answer": None,
                    "question_prompt_hash": None,
                    "candidate_system_prompt_hash": candidate_system_prompt_hash,
                    "evaluator_prompt_hash": None,
                    "jd_text": jd_text[:2000],
                }
                episode_transitions.append(transition)
                obs = next_obs
                done = transition["done"]
                continue

            target_skill = env.select_target_skill(action_idx)
            
            belief_state = {k: v.tolist() for k, v in env.belief_updater.beliefs.items()}
            
            target_metadata = env.ontology.get_skill_metadata(target_skill)
            if target_metadata is None:
                raise RuntimeError(f"Target skill lacks grounding metadata: {target_skill}")
            question_grounding = grounding_packet(role_profile, target_metadata)

            # One sampled behavior action and target survive all retries. A
            # rejected generation therefore cannot bias logged propensities.
            rejected_question_reasons = []
            rejected_question_outputs = []
            question = ""
            question_prompt_hash = None
            question_generation_seed = None
            question_generation_mode = "llm"
            fallback_question_result = None
            fallback_attempted = False
            grounding_result = None
            for question_attempt in range(1, 4):
                correction = None
                if rejected_question_reasons:
                    correction = build_question_retry_correction(
                        rejected_question_reasons[-1],
                        rejected_question_outputs[-1],
                        question_attempt,
                    )
                question_generation_seed = (
                    int(seed)
                    + ep * 1_000_003
                    + env.turn_id * 1_009
                    + action_idx * 97
                    + question_attempt
                ) % 2_147_483_647
                question_prompt = interviewer._build_prompt(
                    action_name,
                    belief_state,
                    resume_text,
                    history,
                    role=role_profile.role_title,
                    experience=interviewer_experience,
                    target_skill=target_skill,
                    grounding_context=question_grounding,
                    correction=correction,
                )
                question_prompt_hash = _sha256_text(question_prompt)
                raw_question = await interviewer.generate_question(
                    action=action_name,
                    belief_state=belief_state,
                    resume=resume_text,
                    history=history,
                    role=role_profile.role_title,
                    experience=interviewer_experience,
                    target_skill=target_skill,
                    grounding_context=question_grounding,
                    correction=correction,
                    temperature=0.3 + 0.15 * (question_attempt - 1),
                    generation_seed=question_generation_seed,
                )
                question = normalize_generated_question(raw_question)
                grounding_result = validate_grounded_question(
                    question, question_grounding, history
                )
                if grounding_result["valid"]:
                    break
                episode_diagnostic["rejected_question_attempts"].append({
                    "turn": env.turn_id,
                    "action_idx": action_idx,
                    "action": action_name,
                    "target_skill": target_skill,
                    "target_skill_id": target_metadata.skill_id,
                    "retry_number": question_attempt,
                    "raw_generated_question": raw_question,
                    "normalized_question": question,
                    "validation_reasons": list(grounding_result["reasons"]),
                    "validation_result": grounding_result,
                    "question_prompt_hash": question_prompt_hash,
                    "question_generation_seed": question_generation_seed,
                })
                rejected_question_reasons.append(grounding_result["reasons"])
                rejected_question_outputs.append(question)
            fallback_eligible = _can_use_deterministic_grounding_fallback(
                rejected_question_outputs
            )
            if (
                (not grounding_result or not grounding_result["valid"])
                and fallback_eligible
            ):
                fallback_attempted = True
                fallback_question = build_grounded_fallback_question(
                    action_name,
                    target_skill,
                    history,
                    grounding_context=question_grounding,
                    variation_key=(
                        f"{resume_content_hash}|{jd_content_hash}|{env.turn_id}|"
                        f"{action_idx}"
                    ),
                )
                fallback_question_result = validate_grounded_question(
                    fallback_question,
                    question_grounding,
                    history,
                )
                episode_diagnostic["fallback_questions"].append({
                    "turn": env.turn_id,
                    "action": action_name,
                    "target_skill_id": target_metadata.skill_id,
                    "template_version": FALLBACK_QUESTION_TEMPLATE_VERSION,
                    "question": fallback_question,
                    "validation_result": fallback_question_result,
                })
                if fallback_question_result["valid"]:
                    question = fallback_question
                    grounding_result = fallback_question_result
                    question_generation_mode = "deterministic_grounded_fallback"
                    question_prompt_hash = None
                    question_generation_seed = None
            if not grounding_result or not grounding_result["valid"]:
                episode_diagnostic.update({
                    "status": "failed",
                    "failure_stage": "question_grounding",
                    "failure_turn": env.turn_id,
                    "failure_action": action_name,
                    "failure_target_skill": target_skill,
                    "failure_target_skill_id": target_metadata.skill_id,
                    "failure_reason": (
                        "question grounding failed after 3 LLM attempts and "
                        "deterministic fallback"
                        if fallback_attempted
                        else "question generation failed after 3 LLM attempts"
                    ),
                })
                print(
                    f"  [ERROR] Episode episode_{ep}: grounding failed for "
                    f"action '{action_name}' after 3 attempts."
                )
                for attempt in episode_diagnostic["rejected_question_attempts"][-3:]:
                    print(
                        "    "
                        f"attempt={attempt['retry_number']} "
                        f"target={attempt['target_skill_id']} "
                        f"reasons={attempt['validation_reasons']} "
                        f"raw={attempt['raw_generated_question']!r}"
                    )
                if episode_diagnostic["fallback_questions"]:
                    fallback = episode_diagnostic["fallback_questions"][-1]
                    print(
                        "    deterministic_fallback "
                        f"reasons={fallback['validation_result']['reasons']} "
                        f"question={fallback['question']!r}"
                    )
                return []
            
            # 2. Candidate answers
            answer = ""
            for answer_attempt in range(1, 4):
                answer = await generate_llm_response(
                    question,
                    CANDIDATE_MODEL,
                    system=system_prompt,
                    client=ollama_client,
                )
                if answer and answer.strip():
                    break
                print(
                    f"  [WARN] Episode episode_{ep}: candidate generation failed "
                    f"({answer_attempt}/3); retrying the same action, target, and question."
                )
            if not answer or not answer.strip():
                episode_diagnostic.update({
                    "status": "failed",
                    "failure_stage": "candidate_generation",
                    "failure_turn": env.turn_id,
                    "failure_action": action_name,
                    "failure_target_skill": target_skill,
                    "failure_target_skill_id": target_metadata.skill_id,
                    "failure_reason": "candidate generation failed after 3 attempts",
                })
                print(f"  [ERROR] Episode episode_{ep}: candidate repeatedly failed.")
                return []
            
            # 3. Evaluate
            evaluator_prompt_hash = _sha256_text(build_evaluator_prompt(
                question, answer, question_grounding
            ))
            (
                sem_score,
                beh_score,
                cog_load,
                evaluator_confidence,
                rubric_evidence,
                evaluation_valid,
            ) = await evaluate_answer(
                question,
                answer,
                client=ollama_client,
                grounding_context=question_grounding,
            )
            if not evaluation_valid:
                episode_diagnostic.update({
                    "status": "failed",
                    "failure_stage": "answer_evaluation",
                    "failure_turn": env.turn_id,
                    "failure_action": action_name,
                    "failure_target_skill": target_skill,
                    "failure_target_skill_id": target_metadata.skill_id,
                    "failure_reason": "evaluator remained invalid after bounded retries",
                })
                print(
                    f"  [ERROR] Episode episode_{ep}: evaluator remained invalid "
                    "after its bounded retries; aborting the transactional episode."
                )
                return []
            history.append({"q": question, "a": answer})
            
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
                "action_name": action_name,
                "action_schema_version": ACTION_SCHEMA_VERSION,
                "action_mask_before": action_mask_before,
                "behavior_action_probs": behavior_probabilities,
                "behavior_action_probability": behavior_probabilities[action_idx],
                "reward": float(reward),
                "next_obs": next_obs.tolist(),
                "done": bool(done),
                "resume_file": resume_name,
                "jd_file": jd_name,
                "resume_content_hash": resume_content_hash,
                "jd_content_hash": jd_content_hash,
                "resume_prompt_hash": resume_prompt_hash,
                "resume_source_type": loaded_pair.resume_source_type,
                "resume_category": loaded_pair.resume_category,
                "resume_source_file_hash": loaded_pair.resume_source_file_hash,
                "true_label": true_label,
                "aria_label": aria_label,
                "aggregate_belief": assessment["belief"].tolist(),
                "aggregate_confidence": assessment["confidence"],
                "skills_covered": len(assessment["visited_skills"]),
                "valid_evidence_count": env.valid_evidence_count,
                "target_skill": target_skill,
                "target_skill_id": target_metadata.skill_id,
                "target_skill_source": target_metadata.support_type,
                "target_skill_priority": target_metadata.priority,
                "role_profile_hash": role_profile.profile_hash,
                "role_profile_schema_version": ROLE_PROFILE_SCHEMA_VERSION,
                "grounding_contract_hash": grounding_contract,
                "pairing_record": pairing_record,
                "ontology_hash": ontology_hash,
                "ontology_nodes": ontology_nodes,
                "ontology_size": len(ontology_nodes),
                "question_grounding_schema_version": GROUNDING_SCHEMA_VERSION,
                "question_grounding_valid": grounding_result["valid"],
                "question_grounding": grounding_result,
                "question_generation_attempts": question_attempt,
                "llm_question_generation_attempts": question_attempt,
                "deterministic_question_generation_attempts": (
                    1
                    if question_generation_mode == "deterministic_grounded_fallback"
                    else 0
                ),
                "question_generation_mode": question_generation_mode,
                "fallback_question_template_version": (
                    FALLBACK_QUESTION_TEMPLATE_VERSION
                    if question_generation_mode == "deterministic_grounded_fallback"
                    else None
                ),
                "candidate_generation_attempts": answer_attempt,
                "rejected_question_reasons": rejected_question_reasons,
                "jd_evidence_hashes": [
                    _sha256_text(span.text) for span in target_metadata.jd_evidence
                ],
                "resume_evidence_hashes": [
                    _sha256_text(span.text) for span in target_metadata.resume_evidence
                ],
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
                "generation_run_id": generation_run_id,
                "plan_id": plan_id,
                "generation_started_at": generation_started_at,
                "generator_schema_version": GENERATOR_SCHEMA_VERSION,
                "transition_schema_version": TRANSITION_SCHEMA_VERSION,
                "state_schema_version": STATE_SCHEMA_VERSION,
                "reward_schema_version": REWARD_SCHEMA_VERSION,
                "transition_kind": "question",
                "termination_reason": info["termination_reason"],
                "ollama_num_ctx": OLLAMA_NUM_CTX,
                "ollama_keep_alive": OLLAMA_KEEP_ALIVE,
                "question": question,
                "candidate_answer": answer,
                "question_prompt_hash": question_prompt_hash,
                "question_generation_seed": question_generation_seed,
                "candidate_system_prompt_hash": candidate_system_prompt_hash,
                "evaluator_prompt_hash": evaluator_prompt_hash,
                "jd_text": jd_text[:2000]
            }
            episode_transitions.append(transition)
            obs = next_obs
            
        print(
            f"--- Finished New Episode {display_number}/{total_eps} "
            f"| ID: episode_{ep} | Transitions: {len(episode_transitions)} ---"
        )
        episode_diagnostic["status"] = "complete"
        return episode_transitions

async def run_simulation(
    sweep: bool = False,
    max_episodes: int = DEFAULT_SWEEP_EPISODES,
    max_concurrent: int = 4,
    candidate_request_concurrency: int = 3,
    evaluator_request_concurrency: int = 2,
    identity_component_targets: tuple[int, int, int] = (20, 6, 6),
    seed: int = 42,
    dataset_file: str | Path = DATASET_FILE,
    append: bool = False,
    replace_existing: bool = False,
    allow_model_mix: bool = False,
    gpu_vram_gb: float = 8.0,
    check_ollama_capacity: bool = True,
    resume_source: str = "csv",
    resume_csv_path: str | Path = DEFAULT_CLEANED_RESUME_CSV,
    resume_categories: tuple[str, ...] | list[str] | None = DEFAULT_RESUME_CATEGORIES,
):
    dataset_path = Path(dataset_file)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    original_dataset_bytes = (
        dataset_path.read_bytes() if dataset_path.exists() else None
    )
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be positive")
    if candidate_request_concurrency <= 0 or evaluator_request_concurrency <= 0:
        raise ValueError("Ollama request concurrency limits must be positive")
    if append and replace_existing:
        raise ValueError("Choose --append or --replace-existing, not both")
    if resume_source not in {"pdf", "csv"}:
        raise ValueError("resume_source must be 'pdf' or 'csv'")
    if CANDIDATE_MODEL == EVALUATOR_MODEL:
        raise ValueError("Candidate and evaluator models must remain distinct")
    if resume_source == "csv":
        resume_manifest = get_resume_source_manifest(
            resume_source,
            resume_csv_path,
            resume_categories,
        )
    else:
        resume_manifest = {
            "source_type": "pdf_directory",
            "source_directory": str(RESUMES_DIR.resolve()),
            "selected_categories": None,
            "source_file_hash": None,
        }
    if check_ollama_capacity:
        capacity = await report_ollama_capacity(gpu_vram_gb)
        if not capacity["full_dual_gpu_residency_feasible"]:
            print(
                "[INFO] Both quality models cannot safely remain fully resident "
                "on this GPU. Ollama will phase/queue them; use one loaded model "
                "at a time and batch two episodes per phase."
            )

    existing = _load_dataset(dataset_path)
    source_dataset = list(existing)
    if sweep and existing and not append and not replace_existing:
        raise ValueError(
            f"Refusing to erase existing dataset {dataset_path}. Use --append to "
            "preserve it or --replace-existing for an intentional replacement."
        )
    if replace_existing:
        existing = []
    if append:
        validate_append_provenance(
            existing,
            CANDIDATE_MODEL,
            EVALUATOR_MODEL,
            allow_model_mix=allow_model_mix,
            resume_source_type=resume_manifest["source_type"],
            resume_source_file_hash=resume_manifest.get("source_file_hash"),
        )
        backup = _backup_before_append(dataset_path)
        if backup:
            print(f"Immutable pre-append backup: {backup}")
    elif existing and not sweep:
        validate_append_provenance(
            existing,
            CANDIDATE_MODEL,
            EVALUATOR_MODEL,
            allow_model_mix=allow_model_mix,
            resume_source_type=resume_manifest["source_type"],
            resume_source_file_hash=resume_manifest.get("source_file_hash"),
        )
        backup = _backup_before_append(dataset_path)
        if backup:
            print(f"Immutable pre-append backup: {backup}")

    print(
        f"Starting with {len(group_transitions_into_episodes(existing))} existing "
        f"episodes and {len(existing)} existing transitions."
    )
            
    if sweep:
        if resume_source == "csv":
            jds, jd_manifest = get_groundable_jd_documents(JDS_DIR)
            resumes = get_resume_documents(
                resume_source,
                resume_csv_path,
                resume_categories,
            )
        else:
            all_jds = get_all_pdfs(JDS_DIR)
            jds = [j for j in all_jds if is_valid_jd(j)]
            jd_manifest = {
                "pdf_files": len(all_jds),
                "unique_readable_content_hashes": None,
            }
            all_resumes = get_all_pdfs(RESUMES_DIR)
            resumes = [path for path in all_resumes if is_valid_resume(path)]
            resume_manifest["selected_resume_count"] = len(resumes)
            resume_manifest["unique_content_hashes"] = None

        print(
            f"Sweep pool: {len(resumes)} valid {resume_source} resumes x "
            f"{len(jds)} valid JDs (filtered from {jd_manifest['pdf_files']} JDs)"
        )

        if append and existing:
            episodes_to_run, append_mode = build_append_sweep_pairs(
                existing,
                resumes,
                jds,
                max_episodes=max_episodes,
                seed=seed,
                component_targets=identity_component_targets,
            )
            print(f"Append identity mode: {append_mode}")
            if append_mode == "existing_identity_partitions":
                print(
                    "[WARN] Fewer than three unused resumes and JDs were available. "
                    "The append remains leakage-safe, but add new de-identified "
                    "documents to increase independent identity components."
                )
        else:
            episodes_to_run = build_split_safe_sweep_pairs(
                resumes,
                jds,
                max_episodes=max_episodes,
                seed=seed,
                component_targets=identity_component_targets,
            )

        print(
            "Starting leakage-safe sweep simulation with "
            f"{len(episodes_to_run)} total episodes..."
        )
    else:
        num_eps = min(2, max_episodes) # fallback for small tests
        print(f"Starting Multi-Agent Simulation with {num_eps} random episodes...")
        episodes_to_run = [None] * num_eps

    semaphore = asyncio.Semaphore(max_concurrent)
    ollama_client = BoundedOllamaClient(
        OLLAMA_HOST,
        {
            CANDIDATE_MODEL: candidate_request_concurrency,
            EVALUATOR_MODEL: evaluator_request_concurrency,
        },
        timeout=OLLAMA_REQUEST_TIMEOUT,
    )
    total_eps = len(episodes_to_run)
    final_episode_target = len(group_transitions_into_episodes(existing)) + total_eps
    if final_episode_target < MIN_RECOMMENDED_EPISODES:
        print(
            f"[WARN] Combined corpus will contain only {final_episode_target} episodes; "
            f"at least {MIN_RECOMMENDED_EPISODES} are recommended for reliable "
            "calibration and held-out evaluation."
        )

    start_index = _next_episode_index(existing)
    source_hash = (
        hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if dataset_path.exists() and existing else "new-corpus"
    )
    generation_started_at = datetime.now(timezone.utc).isoformat()
    planned_pairs_hash = _sha256_text(json.dumps(
        episodes_to_run,
        ensure_ascii=False,
        separators=(",", ":"),
    ))
    document_sources_hash = _sha256_text(json.dumps(
        {
            "resume_source": resume_manifest,
            "job_description_source": jd_manifest if sweep else None,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ))
    grounding_hash = grounding_contract_hash()
    run_material = (
        f"{source_hash}|{seed}|{start_index}|{total_eps}|"
        f"{CANDIDATE_MODEL}|{EVALUATOR_MODEL}|{GENERATOR_SCHEMA_VERSION}|"
        f"{FALLBACK_QUESTION_TEMPLATE_VERSION}|"
        f"{document_sources_hash}|{planned_pairs_hash}|{grounding_hash}|"
        f"{TRANSITION_SCHEMA_VERSION}|{STATE_SCHEMA_VERSION}|"
        f"{ACTION_SCHEMA_VERSION}|{REWARD_SCHEMA_VERSION}"
    )
    plan_id = hashlib.sha256(run_material.encode("utf-8")).hexdigest()[:16]
    generation_run_id = hashlib.sha256(
        f"{plan_id}|{generation_started_at}|{uuid.uuid4().hex}".encode("utf-8")
    ).hexdigest()[:16]
    manifest_path = dataset_path.parent / "manifests" / f"{generation_run_id}.json"
    run_manifest = {
        "generation_run_id": generation_run_id,
        "plan_id": plan_id,
        "status": "running",
        "generation_started_at": generation_started_at,
        "source_dataset_hash": source_hash,
        "simulation_seed": seed,
        "start_episode_index": start_index,
        "planned_episodes": total_eps,
        "planned_pairs_hash": planned_pairs_hash,
        "document_sources_hash": document_sources_hash,
        "identity_component_targets": list(identity_component_targets),
        "planned_document_pairs": [
            None if pair is None else {"resume_file": pair[0], "jd_file": pair[1]}
            for pair in episodes_to_run
        ],
        "persona_schedule": [
            PERSONA_TIERS[(start_index + order) % len(PERSONA_TIERS)]
            for order in range(total_eps)
        ],
        "candidate_model": CANDIDATE_MODEL,
        "evaluator_model": EVALUATOR_MODEL,
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "role_profile_schema_version": ROLE_PROFILE_SCHEMA_VERSION,
        "question_grounding_schema_version": GROUNDING_SCHEMA_VERSION,
        "grounding_contract_hash": grounding_hash,
        "fallback_question_template_version": FALLBACK_QUESTION_TEMPLATE_VERSION,
        "ollama_host": OLLAMA_HOST,
        "ollama_num_ctx": OLLAMA_NUM_CTX,
        "ollama_keep_alive": OLLAMA_KEEP_ALIVE,
        "episode_worker_concurrency": max_concurrent,
        "candidate_request_concurrency": candidate_request_concurrency,
        "evaluator_request_concurrency": evaluator_request_concurrency,
        "single_model_residency": True,
        "resume_source": resume_manifest,
        "job_description_source": jd_manifest if sweep else None,
    }
    _atomic_json_write(manifest_path, run_manifest)
    print(
        f"Generation run {generation_run_id}: adding {total_eps} episodes with "
        f"global IDs episode_{start_index}..episode_{start_index + total_eps - 1}."
    )

    episode_diagnostics: dict[int, dict] = {}

    async def run_one(order, pair):
        global_index = start_index + order
        try:
            transitions = await simulate_episode(
                global_index,
                pair,
                total_eps,
                semaphore,
                persona_tier=PERSONA_TIERS[global_index % len(PERSONA_TIERS)],
                seed=seed,
                display_number=order + 1,
                generation_run_id=generation_run_id,
                plan_id=plan_id,
                generation_started_at=generation_started_at,
                ollama_client=ollama_client,
                resume_source=resume_source,
                resume_csv_path=resume_csv_path,
                resume_categories=resume_categories,
                failure_diagnostics=episode_diagnostics,
            )
            return order, transitions, None
        except Exception as error:
            # A transient Ollama/document failure must not cancel every other
            # episode in a run that may already have consumed many GPU-hours.
            message = f"{type(error).__name__}: {error}"
            diagnostic = episode_diagnostics.setdefault(global_index, {
                "episode_id": f"episode_{global_index}",
                "display_number": order + 1,
                "resume_file": None if pair is None else pair[0],
                "jd_file": None if pair is None else pair[1],
                "rejected_question_attempts": [],
                "fallback_questions": [],
            })
            diagnostic.update({
                "status": "failed",
                "failure_stage": diagnostic.get("failure_stage", "episode_exception"),
                "failure_reason": diagnostic.get("failure_reason", message),
            })
            return order, [], message

    tasks = [
        asyncio.create_task(run_one(order, pair))
        for order, pair in enumerate(episodes_to_run)
    ]
    completed = {}
    failed = {}
    run_started = time.monotonic()
    finished_count = 0
    for future in asyncio.as_completed(tasks):
        order, episode_transitions, error = await future
        finished_count += 1
        if not episode_transitions:
            global_index = start_index + order
            diagnostic = episode_diagnostics.get(global_index, {})
            failed[order] = error or diagnostic.get(
                "failure_reason", "episode returned no valid transitions"
            )
            print(
                f"[WARN] New episode {order + 1} failed and was not appended: "
                f"{failed[order]}"
            )
            continue
        completed[order] = episode_transitions
        combined = list(existing)
        for completed_order in sorted(completed):
            combined.extend(completed[completed_order])
        _atomic_json_write(dataset_path, combined)
        elapsed = max(time.monotonic() - run_started, 1e-9)
        episodes_per_hour = finished_count / elapsed * 3600
        remaining = total_eps - finished_count
        eta = remaining / max(finished_count / elapsed, 1e-9)
        print(
            f"Checkpointed {len(completed)}/{total_eps} new episodes; "
            f"combined transitions={len(combined)}; "
            f"observed rate={episodes_per_hour:.2f} episodes/hour; "
            f"ETA={_format_duration(eta)}."
        )

    await ollama_client.aclose()

    dataset = list(existing)
    for completed_order in sorted(completed):
        dataset.extend(completed[completed_order])
    all_completed = len(completed) == total_eps and not failed
    canonical_dataset = dataset if all_completed else source_dataset
    new_fallback_transition_count = sum(
        item.get("question_generation_mode") == "deterministic_grounded_fallback"
        for episode in completed.values()
        for item in episode
    )
    working_fallback_transition_count = sum(
        item.get("question_generation_mode") == "deterministic_grounded_fallback"
        for item in dataset
    )
    combined_fallback_transition_count = sum(
        item.get("question_generation_mode") == "deterministic_grounded_fallback"
        for item in canonical_dataset
    )
    pairing_records = []
    for global_index in sorted(episode_diagnostics):
        diagnostic = episode_diagnostics[global_index]
        pairing_record = diagnostic.get("pairing_record")
        if pairing_record:
            pairing_records.append({
                "episode_id": diagnostic.get("episode_id"),
                "episode_status": diagnostic.get("status"),
                "resume_file": diagnostic.get("resume_file"),
                "jd_file": diagnostic.get("jd_file"),
                **dict(pairing_record),
            })
    partial_path = None
    if not all_completed and completed:
        partial_path = (
            dataset_path.parent / "failed_runs" /
            f"{generation_run_id}.partial.json"
        )
        _atomic_json_write(partial_path, dataset)
        if original_dataset_bytes is None:
            dataset_path.unlink(missing_ok=True)
        else:
            _atomic_bytes_write(dataset_path, original_dataset_bytes)
    canonical_dataset_hash = (
        hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        if dataset_path.exists() else None
    )
    partial_dataset_hash = (
        hashlib.sha256(partial_path.read_bytes()).hexdigest()
        if partial_path is not None and partial_path.exists() else None
    )
    run_manifest.update({
        "status": "complete" if all_completed else "failed",
        "completed_episodes": len(completed),
        "failed_episodes": {str(order): message for order, message in failed.items()},
        "failed_episode_diagnostics": {
            str(order): episode_diagnostics.get(start_index + order, {})
            for order in sorted(failed)
        },
        "successful_fallback_episode_diagnostics": {
            str(global_index): {
                "episode_id": diagnostic.get("episode_id"),
                "status": diagnostic.get("status"),
                "resume_file": diagnostic.get("resume_file"),
                "jd_file": diagnostic.get("jd_file"),
                "rejected_question_attempts": diagnostic.get(
                    "rejected_question_attempts", []
                ),
                "fallback_questions": diagnostic.get("fallback_questions", []),
            }
            for global_index, diagnostic in sorted(episode_diagnostics.items())
            if diagnostic.get("status") == "complete"
            and diagnostic.get("fallback_questions")
        },
        "combined_transition_count": len(canonical_dataset),
        "partial_combined_transition_count": (
            len(dataset) if not all_completed else None
        ),
        "deterministic_grounding_fallback_count": new_fallback_transition_count,
        "new_deterministic_grounding_fallback_count": (
            new_fallback_transition_count
        ),
        "combined_deterministic_grounding_fallback_count": (
            combined_fallback_transition_count
        ),
        "partial_combined_deterministic_grounding_fallback_count": (
            working_fallback_transition_count if not all_completed else None
        ),
        "pairing_records": pairing_records,
        "partial_dataset_path": str(partial_path) if partial_path else None,
        "canonical_dataset_hash": canonical_dataset_hash,
        "partial_dataset_hash": partial_dataset_hash,
        "canonical_dataset_restored": not all_completed,
        "generation_finished_at": datetime.now(timezone.utc).isoformat(),
    })
    _atomic_json_write(manifest_path, run_manifest)
    if not all_completed:
        print(
            f"\nGeneration failed: {len(completed)}/{total_eps} episodes completed; "
            f"canonical dataset restored. Partial checkpoint: {partial_path or 'none'}."
        )
        raise RuntimeError(
            f"Generation completed {len(completed)}/{total_eps} episodes; "
            "partial output is not eligible for training"
        )
    print(
        f"\nSimulation complete: {len(completed)}/{total_eps} new episodes saved. "
        f"Combined dataset has {len(group_transitions_into_episodes(dataset))} "
        f"episodes and {len(dataset)} transitions at {dataset_path}."
    )
    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run 1 episode for every JD-Resume permutation")
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=DEFAULT_SWEEP_EPISODES,
        help=(
            "Episodes to generate in this run (with --append, this is the "
            "number added to the existing corpus)"
        ),
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=4,
        help="Concurrent episode workers (API calls remain separately bounded)",
    )
    parser.add_argument("--candidate-request-concurrency", type=int, default=3)
    parser.add_argument("--evaluator-request-concurrency", type=int, default=2)
    parser.add_argument(
        "--identity-components",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(20, 6, 6),
        help="Independent resume/JD component targets for each split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Simulation seed")
    parser.add_argument("--dataset-file", default=str(DATASET_FILE))
    parser.add_argument(
        "--resume-source",
        choices=("csv", "pdf"),
        default="csv",
        help="Resume input source; CLI generation defaults to the cleaned CSV",
    )
    parser.add_argument(
        "--resume-csv",
        default=str(DEFAULT_CLEANED_RESUME_CSV),
        help="Path to the cleaned resume CSV when --resume-source=csv",
    )
    parser.add_argument(
        "--resume-categories",
        nargs="+",
        default=list(DEFAULT_RESUME_CATEGORIES),
        help="Allowed cleaned-CSV resume categories",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append max_episodes new episodes with backup and atomic checkpoints",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Explicitly replace an existing dataset instead of appending",
    )
    parser.add_argument("--allow-model-mix", action="store_true")
    parser.add_argument("--gpu-vram-gb", type=float, default=8.0)
    parser.add_argument("--skip-ollama-capacity-check", action="store_true")
    args = parser.parse_args()
    
    try:
        asyncio.run(run_simulation(
            sweep=args.sweep,
            max_episodes=args.max_episodes,
            max_concurrent=args.max_concurrent,
            candidate_request_concurrency=args.candidate_request_concurrency,
            evaluator_request_concurrency=args.evaluator_request_concurrency,
            identity_component_targets=tuple(args.identity_components),
            seed=args.seed,
            dataset_file=args.dataset_file,
            append=args.append,
            replace_existing=args.replace_existing,
            allow_model_mix=args.allow_model_mix,
            gpu_vram_gb=args.gpu_vram_gb,
            check_ollama_capacity=not args.skip_ollama_capacity_check,
            resume_source=args.resume_source,
            resume_csv_path=args.resume_csv,
            resume_categories=tuple(args.resume_categories),
        ))
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as error:
        parser.exit(1, f"[ERROR] {error}\n")
