import sys
import os

# Adjust imports to local module structure
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modules.module_05_ontology.graph import SkillOntologyGraph
from modules.module_06_belief.belief_state import BeliefStateUpdater
from modules.module_07_rl.environment import ARIAInterviewEnv

def test_integration():
    print("=== Testing Integration of Modules 5, 6, and 7 ===")
    
    # 1. Test Ontology (Module 5)
    print("\n--- Module 5: Ontology ---")
    ontology = SkillOntologyGraph("backend_developer")
    skills = ontology.get_all_skills()
    print(f"Loaded {len(skills)} skills for Backend Developer.")
    print(f"Dependencies for 'Docker': {ontology.get_prerequisites('Docker')}")
    
    # 2. Test Belief Updater (Module 6)
    print("\n--- Module 6: Competency Belief Updater ---")
    updater = BeliefStateUpdater(skills)
    print(f"Initial Global Entropy: {updater.get_global_entropy():.4f}")
    updater.update_belief("Docker", semantic_score=0.9, cognitive_load="low", behavior_score=0.9)
    print(f"Belief after strong response on Docker: {updater.get_belief('Docker')}")
    print(f"New Global Entropy: {updater.get_global_entropy():.4f}")
    
    # 3. Test RL Environment (Module 7)
    print("\n--- Module 7: RL Environment (POMDP) ---")
    env = ARIAInterviewEnv("backend_developer")
    obs, _ = env.reset()
    print(f"Environment initialized. Observation shape: {obs.shape}")
    
    # Simulate a full rollout
    print("Simulating rollouts...")
    terminated = False
    truncated = False
    turn = 0
    
    while not (terminated or truncated):
        turn += 1
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Turn {turn} | Action: {info['action']:>25} | Reward: {reward:+.4f} | Info Gain: {info['info_gain']:.4f}")
        
    print(f"\nEpisode concluded after {turn} turns.")
    print("Integration test passed successfully.")

if __name__ == "__main__":
    test_integration()
