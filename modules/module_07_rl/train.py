"""
Training pipeline for the ARIA RL Interview Policy Agent.
Note: The master specification requests Implicit Q-Learning (IQL) for offline RL.
Stable-Baselines3 does not support IQL out of the box in its main library. 
For this simulation phase, we use PPO on our simulated environment to validate 
the POMDP formulation. To switch to IQL, you should collect offline trajectories 
from this environment and use an offline library like d3rlpy or SB3 Contrib.
"""

import os
import sys

# Adjust imports to local module structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.module_07_rl.environment import ARIAInterviewEnv

def train_simulated_policy(role_name="backend_developer", total_timesteps=10000):
    env = ARIAInterviewEnv(role_name)
    
    try:
        from stable_baselines3 import PPO
        
        print(f"Initializing PPO Agent for {role_name} ontology...")
        model = PPO("MlpPolicy", env, verbose=1)
        
        print(f"Starting training for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps)
        
        # Save the model
        model_path = os.path.join(os.path.dirname(__file__), f"aria_policy_{role_name}.zip")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
    except ImportError:
        print("stable_baselines3 is not installed.")
        print("Run: pip install stable-baselines3")
        print("Falling back to random rollout for validation...")
        
        obs, _ = env.reset()
        for _ in range(20): # Max turns
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Episode finished. Final Info: {info}")
                break

if __name__ == "__main__":
    train_simulated_policy()
