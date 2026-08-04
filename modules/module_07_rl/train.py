"""
Native PyTorch Implementation of Implicit Q-Learning (IQL) for the ARIA RL Agent.
Replaces the previous PPO dependency to align with the master specification
without requiring external RL libraries that conflict with our PyTorch version.

This script collects simulated offline trajectories from the ARIA POMDP environment
and trains an IQL policy (Value Network, Q-Networks, and Policy Network) over them.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

# Adjust imports to local module structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.module_07_rl.environment import ARIAInterviewEnv

# Hyperparameters
TAU = 0.005          # Target network update rate
GAMMA = 0.99         # Discount factor
EXPECTILE = 0.8      # Expectile for value network (IQL hyperparam)
BETA = 3.0           # Temperature for advantage weighted regression
LR = 3e-4            # Learning rate
BATCH_SIZE = 64      # Batch size

class IQLNetworks(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Value Network (V)
        self.v_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q-Networks (Double Q-learning)
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Policy Network (Actor) - outputs logits for categorical distribution
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

def get_action_one_hot(action, action_dim):
    one_hot = np.zeros(action_dim, dtype=np.float32)
    one_hot[action] = 1.0
    return one_hot

def collect_offline_trajectories(env, num_transitions=5000):
    """Simulate collecting offline data using a random policy"""
    print(f"Collecting {num_transitions} transitions of offline data...")
    transitions = []
    obs, _ = env.reset()
    
    for _ in range(num_transitions):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        transitions.append({
            "obs": obs,
            "action": get_action_one_hot(action, env.action_space.n),
            "action_idx": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": terminated or truncated
        })
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
    return transitions

def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)

def train_iql_policy(role_name="backend_developer", total_epochs=20):
    env = ARIAInterviewEnv(role_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    nets = IQLNetworks(state_dim, action_dim).to(device)
    q1_target = copy.deepcopy(nets.q1_net).to(device)
    q2_target = copy.deepcopy(nets.q2_net).to(device)
    
    optimizer_v = optim.Adam(nets.v_net.parameters(), lr=LR)
    optimizer_q = optim.Adam(list(nets.q1_net.parameters()) + list(nets.q2_net.parameters()), lr=LR)
    optimizer_policy = optim.Adam(nets.policy_net.parameters(), lr=LR)
    
    # 1. Collect Data
    dataset = collect_offline_trajectories(env, num_transitions=5000)
    
    # 2. Offline Training Loop
    print(f"Starting IQL Offline Training for {total_epochs} epochs...")
    num_batches = len(dataset) // BATCH_SIZE
    
    for epoch in range(total_epochs):
        np.random.shuffle(dataset)
        epoch_v_loss = 0
        epoch_q_loss = 0
        epoch_pi_loss = 0
        
        for i in range(num_batches):
            batch = dataset[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            
            s = torch.tensor(np.array([b["obs"] for b in batch])).to(device)
            a_onehot = torch.tensor(np.array([b["action"] for b in batch])).to(device)
            a_idx = torch.tensor(np.array([b["action_idx"] for b in batch], dtype=np.int64)).to(device)
            r = torch.tensor(np.array([b["reward"] for b in batch], dtype=np.float32)).unsqueeze(1).to(device)
            s_next = torch.tensor(np.array([b["next_obs"] for b in batch])).to(device)
            d = torch.tensor(np.array([b["done"] for b in batch], dtype=np.float32)).unsqueeze(1).to(device)
            
            # --- Update Value Network (Expectile Regression) ---
            with torch.no_grad():
                q1 = q1_target(torch.cat([s, a_onehot], dim=1))
                q2 = q2_target(torch.cat([s, a_onehot], dim=1))
                q_target = torch.min(q1, q2)
                
            v = nets.v_net(s)
            v_loss = expectile_loss(q_target - v, EXPECTILE).mean()
            
            optimizer_v.zero_grad()
            v_loss.backward()
            optimizer_v.step()
            
            # --- Update Q Networks ---
            with torch.no_grad():
                v_next = nets.v_net(s_next)
                q_target_val = r + (1 - d) * GAMMA * v_next
                
            q1 = nets.q1_net(torch.cat([s, a_onehot], dim=1))
            q2 = nets.q2_net(torch.cat([s, a_onehot], dim=1))
            q_loss = F.mse_loss(q1, q_target_val) + F.mse_loss(q2, q_target_val)
            
            optimizer_q.zero_grad()
            q_loss.backward()
            optimizer_q.step()
            
            # --- Update Policy Network (Advantage Weighted Regression) ---
            with torch.no_grad():
                adv = q_target - v
                weight = torch.exp(BETA * adv).clamp(max=100.0)
                
            logits = nets.policy_net(s)
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, a_idx.unsqueeze(1))
            
            pi_loss = -(weight * action_log_probs).mean()
            
            optimizer_policy.zero_grad()
            pi_loss.backward()
            optimizer_policy.step()
            
            # Soft update target networks
            for param, target_param in zip(nets.q1_net.parameters(), q1_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(nets.q2_net.parameters(), q2_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                
            epoch_v_loss += v_loss.item()
            epoch_q_loss += q_loss.item()
            epoch_pi_loss += pi_loss.item()
            
        print(f"Epoch {epoch+1}/{total_epochs} | V Loss: {epoch_v_loss/num_batches:.4f} | Q Loss: {epoch_q_loss/num_batches:.4f} | Pi Loss: {epoch_pi_loss/num_batches:.4f}")
        
    # Save the native PyTorch model
    model_path = os.path.join(os.path.dirname(__file__), f"aria_iql_policy_{role_name}.pth")
    torch.save(nets.state_dict(), model_path)
    print(f"IQL Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_iql_policy()
