import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Adjust imports to local module structure
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.module_05_ontology.graph import SkillOntologyGraph
from modules.module_06_belief.belief_config import BeliefModelConfig
from modules.module_06_belief.belief_state import BeliefStateUpdater
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE
from modules.module_07_rl.reward_model import compute_step_reward, compute_stop_reward
from modules.module_07_rl.state_builder import STATE_DIM, build_action_mask, build_policy_state

MAX_NODES = 50
MAX_TURNS = 30
MIN_INTERVIEW_TURNS = 10
MIN_SKILLS_COVERED = 5

class ARIAInterviewEnv(gym.Env):
    """
    POMDP Environment for the ARIA Interview Agent.
    Integrates the Skill Ontology Graph and the Competency Belief Updater.
    """
    
    def __init__(
        self,
        role_name="backend_developer",
        ontology=None,
        belief_config=None,
        belief_sigma=None,
    ):
        super(ARIAInterviewEnv, self).__init__()
        
        # Load Ontology
        self.ontology = ontology if ontology is not None else SkillOntologyGraph(role_name)
        self.nodes = sorted(self.ontology.get_all_skills())[:MAX_NODES]
        self.num_nodes = len(self.nodes)
        self.belief_sigma = belief_sigma
        
        # 8 Discrete Actions
        self.action_space = spaces.Discrete(len(RL_ACTION_SPACE))
        
        if belief_config is not None and belief_sigma is not None:
            raise ValueError("Pass belief_config or legacy belief_sigma, not both")
        if belief_sigma is not None:
            belief_config = BeliefModelConfig.legacy(belief_sigma)
        self.belief_config = belief_config or BeliefModelConfig()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.belief_updater = BeliefStateUpdater(self.nodes, config=self.belief_config)
        self.turn_id = 0
        self.current_node_idx = 0
        self.consecutive_turns_on_node = 0
        self.last_target_skill = None
        self.previous_evidence = {}
        self.valid_evidence_count = 0
        return self._get_obs(), {}
        
    def _get_obs(self):
        current_skill = self.nodes[self.current_node_idx] if self.nodes else None
        return build_policy_state(
            self.belief_updater,
            total_skills=self.num_nodes,
            turn_id=self.turn_id,
            current_skill=current_skill,
            consecutive_focus_turns=self.consecutive_turns_on_node,
            valid_evidence_count=self.valid_evidence_count,
            previous=self.previous_evidence,
        )
        
    def sync_ontology_nodes(self):
        """
        Re-syncs env nodes and belief updater after ontology adaptation.
        Should be called after env.ontology.adapt_to_candidate().
        """
        all_skills = self.ontology.get_all_skills()
        if len(all_skills) > MAX_NODES:
            raise ValueError(
                f"Ontology has {len(all_skills)} nodes, exceeding MAX_NODES={MAX_NODES}"
            )
        self.nodes = sorted(all_skills)
        self.num_nodes = len(self.nodes)

    def select_target_skill(self, action_idx):
        """Choose the skill the next question and belief update must share."""
        if not self.nodes:
            raise RuntimeError("Cannot select a target skill from an empty ontology")

        action_name = RL_ACTION_SPACE[action_idx]
        current_skill = self.last_target_skill or self.nodes[self.current_node_idx]

        if action_name == "ask_follow_up_same_topic" and self.last_target_skill:
            candidates = [self.last_target_skill]
        elif action_name in {"increase_difficulty"}:
            candidates = self.ontology.get_advanced(current_skill)
        elif action_name in {"decrease_difficulty", "probe_foundation"}:
            candidates = self.ontology.get_prerequisites(current_skill)
        elif action_name == "switch_topic":
            candidates = [skill for skill in self.nodes if skill != current_skill]
        else:
            candidates = [current_skill]

        candidates = [skill for skill in candidates if skill in self.belief_updater.beliefs]
        if not candidates:
            candidates = [current_skill]

        def requirement_priority(skill: str) -> int:
            metadata = self.ontology.get_skill_metadata(skill)
            return metadata.selection_priority if metadata is not None else 5

        visited = set(self.belief_updater.get_visited_skills())
        coverage_incomplete = len(visited) < self.required_skill_coverage
        # Until the conclusion contract is met, do not repeatedly prefer a
        # small set of high-priority requirements over untouched skills.
        return min(
            candidates,
            key=lambda skill: (
                0 if coverage_incomplete and skill not in visited else 1,
                requirement_priority(skill),
                self.belief_updater.get_evidence_count(skill),
                -self.belief_updater._calculate_entropy(
                    self.belief_updater.get_belief(skill)
                ),
                skill,
            ),
        )

    @property
    def required_skill_coverage(self):
        return min(
            max(MIN_SKILLS_COVERED, self.belief_config.minimum_skill_coverage),
            self.num_nodes,
        )

    def conclusion_status(self):
        assessment = self.belief_updater.get_aggregate_assessment()
        visited = len(assessment["visited_skills"])
        status = {
            "turn": {
                "actual": self.turn_id,
                "required": MIN_INTERVIEW_TURNS,
                "ready": self.turn_id >= MIN_INTERVIEW_TURNS,
            },
            "coverage": {
                "actual": visited,
                "required": self.required_skill_coverage,
                "ready": visited >= self.required_skill_coverage,
            },
            "valid_evidence": {
                "actual": self.valid_evidence_count,
                "required": 5,
                "ready": self.valid_evidence_count >= 5,
            },
            "can_conclude": False,
        }
        status["can_conclude"] = all(
            status[name]["ready"] for name in ("turn", "coverage", "valid_evidence")
        )
        return status

    def can_conclude(self):
        return self.conclusion_status()["can_conclude"]

    def get_action_mask(self):
        mask = build_action_mask(self)
        follow_up = RL_ACTION_SPACE.index("ask_follow_up_same_topic")
        if self.last_target_skill is None:
            mask[follow_up] = 0.0
        return mask

    def step(self, action_idx):
        # NOTE: do NOT increment turn_id here — step_with_scores() handles it
        action_name = RL_ACTION_SPACE[action_idx]
        
        # Simulate environment dynamics (candidate response)
        # In a real setup, this triggers the LLM to ask a question, and waits for candidate signals.
        # Here, we mock the candidate signals based on the action taken.
        
        if action_name == "conclude_interview":
            return self.step_with_scores(action_idx, None, None, None)

        semantic_score = np.random.uniform(0.3, 0.9)
        behavior_score = np.random.uniform(0.3, 0.9)
        cog_load = np.random.choice(['low', 'anxiety', 'ignorance'])
        
        target_skill = self.select_target_skill(action_idx)
        return self.step_with_scores(
            action_idx, semantic_score, behavior_score, cog_load,
            target_skill=target_skill,
        )

    def step_with_scores(
        self,
        action_idx,
        semantic_score,
        behavior_score,
        cog_load,
        target_skill=None,
        evaluator_confidence=1.0,
        stt_confidence=1.0,
        modality_confidence=1.0,
        question_fingerprint=None,
        incongruence_score=None,
    ):
        if not isinstance(action_idx, (int, np.integer)) or not 0 <= int(action_idx) < len(RL_ACTION_SPACE):
            raise ValueError(f"Invalid action index: {action_idx}")
        action_idx = int(action_idx)
        action_name = RL_ACTION_SPACE[action_idx]
        action_mask_before = self.get_action_mask().tolist()

        # Stop is a decision, not a question. It must never create evidence,
        # mutate beliefs, select a skill, or consume an interview turn.
        if action_name == "conclude_interview":
            conclusion_allowed = self.can_conclude()
            conclusion_status = self.conclusion_status()
            conclusion_status["can_conclude"] = conclusion_allowed
            assessment = self.belief_updater.get_aggregate_assessment()
            return self._get_obs(), compute_stop_reward(not conclusion_allowed), bool(
                conclusion_allowed
            ), False, {
                "info_gain": 0.0,
                "action": action_name,
                "target_skill": None,
                "conclusion_allowed": conclusion_allowed,
                "conclusion_status": conclusion_status,
                "conclude_blocked": not conclusion_allowed,
                "skills_covered": len(assessment["visited_skills"]),
                "valid_evidence_count": self.valid_evidence_count,
                "aggregate_belief": assessment["belief"].tolist(),
                "aggregate_label": assessment["label"],
                "aggregate_raw_label": assessment["raw_label"],
                "aggregate_confidence": assessment["confidence"],
                "effective_evidence": assessment["effective_evidence"],
                "assessment_status": assessment["status"],
                "termination_reason": (
                    "explicit_conclusion" if conclusion_allowed else None
                ),
                "action_mask_before": action_mask_before,
                "action_mask": self.get_action_mask().tolist(),
            }

        if semantic_score is None or behavior_score is None or cog_load is None:
            raise ValueError("Question actions require complete evaluator scores")

        self.turn_id += 1
        target_skill = target_skill or self.select_target_skill(action_idx)
        if target_skill not in self.belief_updater.beliefs:
            raise ValueError(f"Unknown target skill: {target_skill}")
        self.current_node_idx = self.nodes.index(target_skill)
        
        old_target_entropy = self.belief_updater._calculate_entropy(
            self.belief_updater.get_belief(target_skill)
        )
        old_count = self.belief_updater.get_evidence_count(target_skill)
        old_ess = self.belief_updater.get_effective_sample_size(target_skill)
        self.belief_updater.update_belief(
            target_skill,
            semantic_score,
            cog_load,
            behavior_score,
            evidence_confidence=evaluator_confidence,
            stt_confidence=stt_confidence,
            modality_confidence=modality_confidence,
            question_fingerprint=question_fingerprint,
        )
        self.valid_evidence_count += 1
        new_target_entropy = self.belief_updater._calculate_entropy(
            self.belief_updater.get_belief(target_skill)
        )
        new_ess = self.belief_updater.get_effective_sample_size(target_skill)
        effective_increment = max(new_ess - old_ess, 0.0)
        info_gain = max(0.0, old_target_entropy - new_target_entropy) * effective_increment
        reward = compute_step_reward(
            info_gain,
            first_skill_visit=old_count == 0,
            previous_skill_count=old_count,
            cognitive_load=cog_load,
            invalid_action=False,
        )

        assessment = self.belief_updater.get_aggregate_assessment()
        conclusion_allowed = self.can_conclude()
        conclusion_status = self.conclusion_status()
        conclusion_status["can_conclude"] = conclusion_allowed
        terminated = False
        termination_reason = None
        truncated = False
        if self.turn_id >= MAX_TURNS: # Hard limit
            truncated = True
            if not terminated:
                termination_reason = "max_turns"
            
        # Update current node focus for next turn based on action
        if self.last_target_skill != target_skill:
            self.consecutive_turns_on_node = 0
        else:
            self.consecutive_turns_on_node += 1
        self.last_target_skill = target_skill
        reliability = float(
            np.clip(evaluator_confidence, 0.0, 1.0)
            * np.clip(stt_confidence, 0.0, 1.0)
            * np.clip(modality_confidence, 0.0, 1.0)
        )
        self.previous_evidence = {
            "semantic_score": semantic_score,
            "evidence_reliability": reliability,
            "behavior_score": behavior_score,
            "cognitive_load": cog_load,
            "incongruence_score": incongruence_score,
            "action_idx": int(action_idx),
        }

        return self._get_obs(), reward, terminated, truncated, {
            "info_gain": info_gain,
            "action": action_name,
            "target_skill": target_skill,
            "conclusion_allowed": conclusion_allowed,
            "conclusion_status": conclusion_status,
            "conclude_blocked": action_name == "conclude_interview" and not conclusion_allowed,
            "skills_covered": len(assessment["visited_skills"]),
            "valid_evidence_count": self.valid_evidence_count,
            "aggregate_belief": assessment["belief"].tolist(),
            "aggregate_label": assessment["label"],
            "aggregate_raw_label": assessment["raw_label"],
            "aggregate_confidence": assessment["confidence"],
            "effective_evidence": assessment["effective_evidence"],
            "assessment_status": assessment["status"],
            "termination_reason": termination_reason,
            "action_mask_before": action_mask_before,
            "action_mask": self.get_action_mask().tolist(),
        }

if __name__ == "__main__":
    # Test Environment
    env = ARIAInterviewEnv()
    obs, _ = env.reset()
    print("Environment Initialized.")
    print(f"Observation Shape: {obs.shape}")
    
    # Take a random step
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"Random Step Reward: {reward:.4f}, Terminated: {terminated}")
    print(f"Info: {info}")
