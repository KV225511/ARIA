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

class ARIAInterviewEnv(gym.Env):
    """
    POMDP Environment for the ARIA Interview Agent.
    Integrates the Skill Ontology Graph and the Competency Belief Updater.
    """
    
    def __init__(self, role_name="backend_developer", ontology=None):
        super(ARIAInterviewEnv, self).__init__()
        
        # Load Ontology
        self.ontology = ontology if ontology is not None else SkillOntologyGraph(role_name)
        self.nodes = self.ontology.get_all_skills()
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
        return self._get_obs(), {}
        
    def _get_obs(self):
        belief_flat = np.concatenate([self.belief_updater.get_belief(n) for n in self.nodes])
        
        # Pad to MAX_NODES
        if self.num_nodes < MAX_NODES:
            padding = np.zeros((MAX_NODES - self.num_nodes) * 3, dtype=np.float32)
            belief_flat = np.concatenate([belief_flat, padding])
            
        entropy = np.array([self.belief_updater.get_global_entropy()])
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
        self.nodes = all_skills
        self.num_nodes = len(self.nodes)

    def step(self, action_idx):
        # NOTE: do NOT increment turn_id here — step_with_scores() handles it
        action_name = RL_ACTION_SPACE[action_idx]
        
        # Simulate environment dynamics (candidate response)
        # In a real setup, this triggers the LLM to ask a question, and waits for candidate signals.
        # Here, we mock the candidate signals based on the action taken.
        
        semantic_score = np.random.uniform(0.3, 0.9)
        behavior_score = np.random.uniform(0.3, 0.9)
        cog_load = np.random.choice(['low', 'anxiety', 'ignorance'])
        
        return self.step_with_scores(action_idx, semantic_score, behavior_score, cog_load)
        
    def step_with_scores(self, action_idx, semantic_score, behavior_score, cog_load):
        self.turn_id += 1
        action_name = RL_ACTION_SPACE[action_idx]
        current_node = self.nodes[self.current_node_idx]
        
        semantic_score = min(semantic_score, 1.0)
        behavior_score = min(behavior_score, 1.0)
        
        old_entropy = self.belief_updater.get_global_entropy()
        
        # Update Belief
        self.belief_updater.update_belief(current_node, semantic_score, cog_load, behavior_score)
        new_entropy = self.belief_updater.get_global_entropy()
        
        # Calculate Reward (Information Gain - Duration Penalty + other factors)
        info_gain = max(0, old_entropy - new_entropy)
        reward = (REWARD_COEFFICIENTS["alpha"] * info_gain) - REWARD_COEFFICIENTS["beta"]
        
        # Use full reward spec
        if cog_load in ['anxiety', 'ignorance']:
            reward -= REWARD_COEFFICIENTS.get("delta", 0.0) # Distress penalty
            
        # Add basic outcome alignment (omega) if concluding with high certainty
        if action_name == "conclude_interview" and new_entropy < TERMINATION_ENTROPY_THRESHOLD:
            reward += REWARD_COEFFICIENTS.get("omega", 0.0)
        
        # Check termination
        terminated = False
        if new_entropy < TERMINATION_ENTROPY_THRESHOLD or action_name == "conclude_interview":
            terminated = True
            
        truncated = False
        if self.turn_id >= 30: # Hard limit
            truncated = True
            
        # Update current node focus for next turn based on action
        if action_name == "switch_topic":
            self.current_node_idx = (self.current_node_idx + 1) % self.num_nodes
            self.consecutive_turns_on_node = 0
        else:
            self.consecutive_turns_on_node += 1
            # Auto-advance after 3 consecutive turns on the same node
            if self.consecutive_turns_on_node >= 3:
                self.current_node_idx = (self.current_node_idx + 1) % self.num_nodes
                self.consecutive_turns_on_node = 0
            
        return self._get_obs(), reward, terminated, truncated, {"info_gain": info_gain, "action": action_name}

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
