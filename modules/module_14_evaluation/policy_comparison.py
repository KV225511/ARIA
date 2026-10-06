"""Matched fresh-environment comparisons for ARIA policies."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable
import numpy as np

from modules.module_07_rl.calibration_protocol import atomic_json_write, canonical_json_hash
from modules.module_07_rl.environment import ARIAInterviewEnv
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE


POLICY_COMPARISON_VERSION = "aria-policy-comparison-v2"


@dataclass(frozen=True)
class EvidenceCondition:
    name: str
    centers: tuple[float, float, float]
    jitter_width: float
    episode_bias_width: float = 0.0
    global_bias: float = 0.0
    evaluator_confidence: float = 1.0
    modality_confidence: float = 1.0


EVIDENCE_CONDITIONS = {
    "base": EvidenceCondition("base", (0.20, 0.50, 0.80), 0.16),
    "overlap": EvidenceCondition("overlap", (0.38, 0.50, 0.62), 0.50, 0.24),
    "low_confidence": EvidenceCondition(
        "low_confidence", (0.25, 0.50, 0.75), 0.30, 0.16,
        evaluator_confidence=0.55, modality_confidence=0.60,
    ),
    "positive_shift": EvidenceCondition(
        "positive_shift", (0.20, 0.50, 0.80), 0.24, 0.12, global_bias=0.10,
    ),
}


def _bootstrap_mean_ci(values, *, seed: int, resamples: int = 2000) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(resamples, array.size), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {"level": 0.95, "lower": float(lower), "upper": float(upper),
            "resamples": resamples}


def validated_iql_assets(project_root: str | Path) -> dict[str, Path]:
    """Return the repository's frozen v7 inference assets."""
    root = Path(project_root).resolve()
    base = root / "data/synthetic/v3/production-grounding-v10"
    v7 = base / "derived-calibration-v7"
    return {
        "checkpoint": root / "modules/module_07_rl/aria_iql_belief_v7.pth",
        "protocol": v7 / "protocol/calibration_protocol_v7.json",
        "protocol_state": v7 / "protocol/calibration_protocol_state_v2.json",
        "development_bundle": v7 / "manifests/development_bundle_v4.json",
        "split_manifest": base / "derived-calibration-v6/manifests/split_manifest_v4.json",
        "belief_config": v7 / "calibration/belief_model_v3.json",
    }


def load_validated_iql(project_root: str | Path):
    """Load the frozen IQL policy and its exact belief configuration fail-closed."""
    from modules.module_06_belief.belief_config import BeliefModelConfig
    from modules.module_07_rl.iql_policy import IQLPolicy

    assets = validated_iql_assets(project_root)
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("validated IQL assets are missing: " + ", ".join(missing))
    policy = IQLPolicy.load(
        assets["checkpoint"],
        protocol_file=assets["protocol"],
        protocol_state_file=assets["protocol_state"],
        development_bundle_file=assets["development_bundle"],
        split_manifest_file=assets["split_manifest"],
        belief_config_file=assets["belief_config"],
        require_validated=True,
    )
    return policy, BeliefModelConfig.load(assets["belief_config"]), assets


class _PolicyBase:
    def __init__(self, name: str, seed: int):
        self.name = name
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _decision(self, action: int, mask) -> dict:
        legal = np.asarray(mask, dtype=bool)
        if not legal[action]:
            raise ValueError(f"{self.name} selected illegal action {action}")
        probabilities = np.zeros(len(RL_ACTION_SPACE), dtype=float)
        probabilities[action] = 1.0
        return {"action_idx": action, "action_probabilities": probabilities.tolist(),
                "selected_action_probability": 1.0}


class RandomValidPolicy(_PolicyBase):
    def select_action(self, observation, mask, **kwargs):
        legal = np.flatnonzero(np.asarray(mask, dtype=bool))
        return self._decision(int(self.rng.choice(legal)), mask)


class FixedScriptPolicy(_PolicyBase):
    script = (4, 2, 0, 3, 5, 6)

    def __init__(self, seed: int):
        super().__init__("fixed_script", seed)
        self.turn = 0

    def select_action(self, observation, mask, **kwargs):
        if bool(mask[7]) and self.turn >= 10:
            action = 7
        else:
            legal = np.asarray(mask, dtype=bool)
            action = next((self.script[(self.turn + shift) % len(self.script)]
                           for shift in range(len(self.script))
                           if legal[self.script[(self.turn + shift) % len(self.script)]]), int(np.flatnonzero(legal)[0]))
        self.turn += 1
        return self._decision(action, mask)


class RuleBasedAdaptivePolicy(_PolicyBase):
    def select_action(self, observation, mask, **kwargs):
        state = np.asarray(observation, dtype=float)
        legal = np.asarray(mask, dtype=bool)
        if legal[7] and state[3] < 0.45 and state[5] >= 0.5:
            action = 7
        elif state[13] < 0.40 and legal[4]:
            action = 4
        elif state[13] > 0.75 and legal[0]:
            action = 0
        elif legal[2]:
            action = 2
        else:
            action = int(np.flatnonzero(legal)[0])
        return self._decision(action, mask)


class UncertaintyGreedyPolicy(_PolicyBase):
    def select_action(self, observation, mask, **kwargs):
        state = np.asarray(observation, dtype=float)
        legal = np.asarray(mask, dtype=bool)
        if legal[7] and state[3] < 0.35:
            action = 7
        elif state[4] > 0.55 and legal[3]:
            action = 3
        elif legal[4]:
            action = 4
        else:
            action = int(np.flatnonzero(legal)[0])
        return self._decision(action, mask)


def deterministic_evidence_provider(true_label: int, scenario_seed: int,
                                    condition: EvidenceCondition | None = None) -> Callable:
    condition = condition or EVIDENCE_CONDITIONS["base"]
    bias_token = int(sha256(f"bias|{scenario_seed}".encode()).hexdigest()[:16], 16)
    episode_bias = ((bias_token % 2001) / 2000.0 - 0.5) * condition.episode_bias_width

    def provider(*, episode_id, turn_id, action_name, target_skill, **kwargs):
        token = f"{scenario_seed}|{episode_id}|{turn_id}|{action_name}|{target_skill}"
        raw = int(sha256(token.encode()).hexdigest()[:16], 16)
        jitter = ((raw % 2001) / 2000.0 - 0.5) * condition.jitter_width
        action_bonus = 0.025 if action_name in {"ask_follow_up_same_topic", "probe_foundation"} else 0.0
        semantic = float(np.clip(condition.centers[int(true_label)] + jitter + episode_bias
                                 + condition.global_bias + action_bonus, 0.0, 1.0))
        anxiety = ((raw >> 12) % 20 == 0)
        return {"semantic_score": semantic, "behavior_score": 0.5,
                "cognitive_load": "anxiety" if anxiety else "low",
                "evaluator_confidence": condition.evaluator_confidence,
                "modality_confidence": condition.modality_confidence}

    return provider


def run_episode(env, policy, evidence_provider, *, episode_id: str, true_label: int,
                seed: int, maximum_steps: int = 30) -> dict:
    observation, _ = env.reset(seed=seed)
    rows = []
    for turn in range(maximum_steps):
        mask = np.asarray(env.get_action_mask(), dtype=int)
        decision = policy.select_action(observation, mask, deterministic=True)
        action = int(decision["action_idx"])
        if not bool(mask[action]):
            raise ValueError("policy selected an illegal action")
        action_name = RL_ACTION_SPACE[action]
        if action_name == "conclude_interview":
            next_observation, reward, terminated, truncated, info = env.step_with_scores(action, None, None, None)
        else:
            target = env.select_target_skill(action)
            evidence = evidence_provider(env=env, episode_id=episode_id, turn_id=turn,
                                         action_idx=action, action_name=action_name, target_skill=target)
            next_observation, reward, terminated, truncated, info = env.step_with_scores(
                action, evidence["semantic_score"], evidence["behavior_score"], evidence["cognitive_load"],
                target_skill=target, question_fingerprint=f"{action_name}:{target}:{turn}",
                evaluator_confidence=evidence.get("evaluator_confidence", 1.0),
                modality_confidence=evidence.get("modality_confidence", 1.0),
            )
        rows.append({"turn_id": turn, "action_idx": action, "action_name": action_name,
                     "reward": float(reward), "info_gain": float(info.get("info_gain", 0.0)),
                     "terminated": bool(terminated), "truncated": bool(truncated), "info": info})
        observe_transition = getattr(policy, "observe_transition", None)
        if observe_transition is not None:
            observe_transition(float(reward))
        observation = next_observation
        if terminated or truncated:
            break
    if not rows or not (rows[-1]["terminated"] or rows[-1]["truncated"]):
        raise RuntimeError("comparison rollout did not finish")
    final = rows[-1]["info"]
    predicted = final.get("aggregate_label")
    return {"episode_id": episode_id, "seed": seed, "true_label": int(true_label),
            "predicted_label": predicted, "correct": predicted == int(true_label),
            "turns": int(final.get("valid_evidence_count", len(rows))),
            "skills_covered": int(final.get("skills_covered", 0)),
            "reward": float(sum(row["reward"] for row in rows)),
            "information_gain": float(sum(row["info_gain"] for row in rows)),
            "termination_reason": final.get("termination_reason") or "max_turns",
            "actions": dict(Counter(row["action_name"] for row in rows)), "transitions": rows}


def _classification_metrics(episodes: list[dict]) -> dict:
    recalls, f1_scores = [], []
    for label in range(3):
        true_positive = sum(row["true_label"] == label and row["predicted_label"] == label
                            for row in episodes)
        false_negative = sum(row["true_label"] == label and row["predicted_label"] != label
                             for row in episodes)
        false_positive = sum(row["true_label"] != label and row["predicted_label"] == label
                             for row in episodes)
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_scores.append(f1)
    return {"balanced_accuracy": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1_scores)),
            "per_class_recall": {str(label): float(recalls[label]) for label in range(3)}}


def _summary(episodes: list[dict], *, seed: int) -> dict:
    def stats(values):
        array = np.asarray(values, dtype=float)
        return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "min": float(array.min()), "max": float(array.max()),
                "ci_95": _bootstrap_mean_ci(array, seed=seed)}
    correct = [row["correct"] for row in episodes]
    return {"num_episodes": len(episodes), "accuracy": float(np.mean(correct)),
            "accuracy_ci_95": _bootstrap_mean_ci(correct, seed=seed + 1),
            **_classification_metrics(episodes),
            "reward": stats([row["reward"] for row in episodes]),
            "turns": stats([row["turns"] for row in episodes]),
            "skills_covered": stats([row["skills_covered"] for row in episodes]),
            "information_gain": stats([row["information_gain"] for row in episodes]),
            "termination_reasons": dict(Counter(row["termination_reason"] for row in episodes))}


def _paired_comparisons(results: dict, reference: str, *, seed: int) -> dict:
    reference_rows = {row["episode_id"]: row for row in results[reference]["episodes"]}
    comparisons = {}
    for name, result in results.items():
        if name == reference:
            continue
        rows = {row["episode_id"]: row for row in result["episodes"]}
        if rows.keys() != reference_rows.keys():
            raise ValueError("paired comparisons require identical scenario identifiers")
        metrics = {}
        for metric in ("correct", "reward", "information_gain", "turns", "skills_covered"):
            differences = [float(reference_rows[key][metric]) - float(rows[key][metric])
                           for key in sorted(rows)]
            metrics[metric] = {
                "mean_difference_reference_minus_comparator": float(np.mean(differences)),
                "ci_95": _bootstrap_mean_ci(differences, seed=seed + len(metrics)),
                "reference_wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "reference_losses": sum(value < 0 for value in differences),
            }
        comparisons[name] = metrics
    return {"reference_policy": reference, "comparisons": comparisons}


def compare_policies(policy_factories: dict[str, Callable[[int], object]], output_file: str | Path,
                     *, episodes_per_class: int = 10, seed: int = 42,
                     env_factory: Callable[[], ARIAInterviewEnv] = ARIAInterviewEnv,
                     policy_metadata: dict | None = None,
                     evidence_condition: EvidenceCondition | None = None) -> dict:
    if episodes_per_class < 1:
        raise ValueError("episodes_per_class must be positive")
    results = {}
    evidence_condition = evidence_condition or EVIDENCE_CONDITIONS["base"]
    scenario_ids = [(label, index, seed + label * 10000 + index)
                    for label in range(3) for index in range(episodes_per_class)]
    for policy_name, factory in policy_factories.items():
        episodes = []
        for label, index, scenario_seed in scenario_ids:
            policy = factory(scenario_seed)
            episodes.append(run_episode(env_factory(), policy,
                                        deterministic_evidence_provider(label, scenario_seed,
                                                                        evidence_condition),
                                        episode_id=f"label-{label}-{index:04d}", true_label=label,
                                        seed=scenario_seed))
        policy_seed = seed + int(sha256(policy_name.encode()).hexdigest()[:8], 16)
        results[policy_name] = {"summary": _summary(episodes, seed=policy_seed), "episodes": episodes}
    reference = "aria_iql_v7" if "aria_iql_v7" in results else next(iter(results))
    saturated = [name for name, value in results.items()
                 if value["summary"]["accuracy"] >= 0.999]
    report = {"schema_version": POLICY_COMPARISON_VERSION, "seed": seed,
              "episodes_per_class": episodes_per_class, "matched_scenarios": True,
              "scenario_count": len(scenario_ids), "policies": results,
              "evidence_condition": asdict(evidence_condition),
              "paired_analysis": _paired_comparisons(results, reference, seed=seed + 9000),
              "ceiling_audit": {
                  "accuracy_saturated": len(saturated) >= 2,
                  "policies_at_or_above_0_999": saturated,
                  "interpretation": ("Accuracy does not discriminate among multiple policies under this evidence condition."
                                     if len(saturated) >= 2 else
                                     "The matched condition does not show a multi-policy accuracy ceiling."),
              },
              "policy_metadata": policy_metadata or {},
              "limitations": ["Synthetic matched rollouts do not establish real-candidate performance.",
                              "The evidence provider is a controlled simulation, not a human response model."]}
    report["report_hash"] = canonical_json_hash(report)
    atomic_json_write(Path(output_file), report)
    return report


def baseline_policy_factories() -> dict[str, Callable[[int], object]]:
    return {
        "random_valid": lambda seed: RandomValidPolicy("random_valid", seed),
        "fixed_script": lambda seed: FixedScriptPolicy(seed),
        "rule_based_adaptive": lambda seed: RuleBasedAdaptivePolicy("rule_based_adaptive", seed),
        "uncertainty_greedy": lambda seed: UncertaintyGreedyPolicy("uncertainty_greedy", seed),
    }
