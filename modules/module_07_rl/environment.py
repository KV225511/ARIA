import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Adjust imports to local module structure
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.module_05_ontology.graph import SkillOntologyGraph
from modules.module_06_belief.belief_state import BeliefStateUpdater
from modules.module_07_rl.rl_spec import RL_ACTION_SPACE, REWARD_COEFFICIENTS, TERMINATION_ENTROPY_THRESHOLD

MAX_NODES = 50
MAX_TURNS = 30
MIN_INTERVIEW_TURNS = 10
MIN_SKILLS_COVERED = 5

class ARIAInterviewEnv(gym.Env):
    """
    POMDP Environment for the ARIA Interview Agent.
    Integrates the Skill Ontology Graph and the Competency Belief Updater.
    """
    
    def __init__(self, role_name="backend_developer", ontology=None):
        super(ARIAInterviewEnv, self).__init__()
        
        # Load Ontology
        self.ontology = ontology if ontology is not None else SkillOntologyGraph(role_name)
        self.nodes = sorted(self.ontology.get_all_skills())[:MAX_NODES]
        self.num_nodes = len(self.nodes)
        
        # 8 Discrete Actions
        self.action_space = spaces.Discrete(len(RL_ACTION_SPACE))
        
        # Define Observation Space (State)
        # belief_vector (MAX_NODES * 3) + entropy (1) + turn_id (1)
        # Padded to handle variable ontology sizes across JDs
        obs_dim = (MAX_NODES * 3) + 2  
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.belief_updater = BeliefStateUpdater(self.nodes)
        self.turn_id = 0
        self.current_node_idx = 0
        self.consecutive_turns_on_node = 0
        self.last_target_skill = None
        return self._get_obs(), {}
        
    def _get_obs(self):
        belief_flat = np.concatenate([self.belief_updater.get_belief(n) for n in self.nodes])
        
        # Pad to MAX_NODES
        if self.num_nodes < MAX_NODES:
            padding = np.zeros((MAX_NODES - self.num_nodes) * 3, dtype=np.float32)
            belief_flat = np.concatenate([belief_flat, padding])
            
        # A three-class entropy is bounded by log(3). Normalize it so the
        # observation respects the declared [0, 1] Box contract.
        entropy = np.array([
            self.belief_updater.get_global_entropy() / np.log(3.0)
        ])
        # Normalize turn_id to [0, 1] assuming max ~ 30 turns
        turn = np.array([min(self.turn_id / 30.0, 1.0)]) 
        
        return np.concatenate([belief_flat, entropy, turn], dtype=np.float32)
        
    def sync_ontology_nodes(self):
        """
        Re-syncs env nodes and belief updater after ontology adaptation.
        Should be called after env.ontology.adapt_to_candidate().
        """
        all_skills = self.ontology.get_all_skills()
        # Bug #4 guard: truncate to MAX_NODES to prevent obs dimension overflow
        if len(all_skills) > MAX_NODES:
            import logging
            logging.getLogger(__name__).warning(
                f"Ontology has {len(all_skills)} nodes, exceeding MAX_NODES={MAX_NODES}. Truncating."
            )
            all_skills = all_skills[:MAX_NODES]
        self.nodes = sorted(all_skills)
        self.num_nodes = len(self.nodes)

    def select_target_skill(self, action_idx):
        """Choose the skill the next question and belief update must share."""
        if not self.nodes:
            raise RuntimeError("Cannot select a target skill from an empty ontology")

        action_name = RL_ACTION_SPACE[action_idx]
        current_skill = self.nodes[self.current_node_idx]

        if action_name in {"increase_difficulty"}:
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

        # Prefer under-observed and uncertain skills; lexical order makes ties
        # deterministic across runs despite ontology storage using sets.
        return min(
            candidates,
            key=lambda skill: (
                self.belief_updater.get_evidence_count(skill),
                -self.belief_updater._calculate_entropy(
                    self.belief_updater.get_belief(skill)
                ),
                skill,
            ),
        )

    def can_conclude(self):
        required_coverage = min(MIN_SKILLS_COVERED, self.num_nodes)
        return (
            self.turn_id >= MIN_INTERVIEW_TURNS
            and len(self.belief_updater.get_visited_skills()) >= required_coverage
        )

    def step(self, action_idx):
        # NOTE: do NOT increment turn_id here — step_with_scores() handles it
        action_name = RL_ACTION_SPACE[action_idx]
        
        # Simulate environment dynamics (candidate response)
        # In a real setup, this triggers the LLM to ask a question, and waits for candidate signals.
        # Here, we mock the candidate signals based on the action taken.
        
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
    ):
        self.turn_id += 1
        action_name = RL_ACTION_SPACE[action_idx]
        target_skill = target_skill or self.select_target_skill(action_idx)
        if target_skill not in self.belief_updater.beliefs:
            raise ValueError(f"Unknown target skill: {target_skill}")
        self.current_node_idx = self.nodes.index(target_skill)
        
        semantic_score = min(semantic_score, 1.0)
        behavior_score = min(behavior_score, 1.0)
        
        old_entropy = self.belief_updater.get_global_entropy()
        
        # Update Belief
        self.belief_updater.update_belief(
            target_skill, semantic_score, cog_load, behavior_score
        )
        new_entropy = self.belief_updater.get_global_entropy()
        
        # Calculate Reward (Information Gain - Duration Penalty + other factors)
        info_gain = max(0, old_entropy - new_entropy)
        reward = (REWARD_COEFFICIENTS["alpha"] * info_gain) - REWARD_COEFFICIENTS["beta"]
        
        # Use full reward spec
        if cog_load in ['anxiety', 'ignorance']:
            reward -= REWARD_COEFFICIENTS.get("delta", 0.0) # Distress penalty
            
        # Add basic outcome alignment (omega) if concluding with high certainty
        conclusion_allowed = self.can_conclude()
        if action_name == "conclude_interview" and not conclusion_allowed:
            reward -= REWARD_COEFFICIENTS["beta"]
        if action_name == "conclude_interview" and conclusion_allowed:
            reward += REWARD_COEFFICIENTS.get("omega", 0.0)
        
        # Check termination
        terminated = False
        if conclusion_allowed and (
            new_entropy < TERMINATION_ENTROPY_THRESHOLD
            or action_name == "conclude_interview"
        ):
            terminated = True
            
        truncated = False
        if self.turn_id >= MAX_TURNS: # Hard limit
            truncated = True
            
        # Update current node focus for next turn based on action
        if self.last_target_skill != target_skill:
            self.consecutive_turns_on_node = 0
        else:
            self.consecutive_turns_on_node += 1
        self.last_target_skill = target_skill

        assessment = self.belief_updater.get_aggregate_assessment()
        return self._get_obs(), reward, terminated, truncated, {
            "info_gain": info_gain,
            "action": action_name,
            "target_skill": target_skill,
            "conclusion_allowed": conclusion_allowed,
            "conclude_blocked": action_name == "conclude_interview" and not conclusion_allowed,
            "skills_covered": len(assessment["visited_skills"]),
            "aggregate_belief": assessment["belief"].tolist(),
            "aggregate_label": assessment["label"],
            "aggregate_confidence": assessment["confidence"],
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
