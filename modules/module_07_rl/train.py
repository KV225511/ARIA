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
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import copy

# Adjust imports to local module structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.module_07_rl.environment import ARIAInterviewEnv

class IQLNetworks(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Value Network (V)
        self.v_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Q-Networks (Double Q-learning)
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Policy Network (Actor) - outputs logits for categorical distribution
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
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

def train_iql_policy(
    role_name="backend_developer", 
    total_epochs=100,
    num_transitions=5000,
    batch_size=256,
    lr=1e-4,
    gamma=0.99,
    tau=0.005,
    expectile=0.8,
    beta=3.0,
    resume_file=None,
    jd_file=None
):
    from modules.module_07_rl.data_loader import get_random_pair, get_specific_pair
    from modules.module_07_rl.metrics import run_benchmark
    import re
    
    if resume_file and jd_file:
        resume_text, jd_text, r_name, j_name = get_specific_pair(resume_file, jd_file)
    else:
        resume_text, jd_text, r_name, j_name = get_random_pair()
        
    print(f"Training on Resume: {r_name} | JD: {j_name}")
    
    env = ARIAInterviewEnv(role_name)
    env.ontology.adapt_to_candidate(jd_text, resume_text)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    nets = IQLNetworks(state_dim, action_dim).to(device)
    q1_target = copy.deepcopy(nets.q1_net).to(device)
    q2_target = copy.deepcopy(nets.q2_net).to(device)
    
    optimizer_v = optim.Adam(nets.v_net.parameters(), lr=lr)
    optimizer_q = optim.Adam(list(nets.q1_net.parameters()) + list(nets.q2_net.parameters()), lr=lr)
    optimizer_policy = optim.Adam(nets.policy_net.parameters(), lr=lr)
    
    scheduler_v = CosineAnnealingLR(optimizer_v, T_max=total_epochs, eta_min=1e-5)
    scheduler_q = CosineAnnealingLR(optimizer_q, T_max=total_epochs, eta_min=1e-5)
    scheduler_pi = CosineAnnealingLR(optimizer_policy, T_max=total_epochs, eta_min=1e-5)
    
    # 1. Collect Data
    dataset_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'synthetic', 'qwen_rl_dataset.json')
    if os.path.exists(dataset_file):
        print(f"Loading high-fidelity LLM dataset from {dataset_file}...")
        import json
        with open(dataset_file, "r") as f:
            dataset = json.load(f)
    else:
        dataset = collect_offline_trajectories(env, num_transitions=num_transitions)
    
    # 2. Validate dataset dimensions to catch stale pre-padding data (Bug #5 fix)
    if dataset:
        sample_obs_len = len(dataset[0]["obs"])
        expected_obs_dim = env.observation_space.shape[0]
        if sample_obs_len != expected_obs_dim:
            print(
                f"\n[ERROR] Dataset dimension mismatch!\n"
                f"  Dataset obs length : {sample_obs_len}\n"
                f"  Expected obs length: {expected_obs_dim}\n"
                f"  Your dataset was generated before the padding fix.\n"
                f"  Please delete '{dataset_file}' and re-run llm_simulator.py to regenerate it."
            )
            return
    
    # 2.5 Soft outcome alignment and per-turn belief alignment reward shaping
    from modules.module_07_rl.rl_spec import REWARD_COEFFICIENTS
    omega = REWARD_COEFFICIENTS.get("omega", 5.0)
    zeta = REWARD_COEFFICIENTS.get("zeta", 0.5)
    
    mismatches = 0
    for b in dataset:
        if "true_label" in b and "aria_label" in b:
            distance = abs(b["true_label"] - b["aria_label"])
            # Dense per-turn alignment bonus
            alignment_bonus = zeta * (1.0 - distance / 2.0)
            b["reward"] = float(b.get("reward", 0.0)) + alignment_bonus
            
            # Graded outcome alignment on terminal transitions (prevents -5.0 crash)
            if b.get("done"):
                soft_terminal = omega * (1.0 - distance / 2.0)
                b["reward"] += soft_terminal
                if distance > 0:
                    mismatches += 1
                
    if dataset:
        print(f"Dataset fix applied: Soft reward alignment active. (Found {mismatches} imperfect belief conclusions in dataset)")

    # Pre-convert dataset to numpy arrays for much faster batch slicing
    obs_arr = np.array([b["obs"] for b in dataset], dtype=np.float32)
    action_onehot_arr = np.array([b["action"] for b in dataset], dtype=np.float32)
    action_idx_arr = np.array([b["action_idx"] for b in dataset], dtype=np.int64)
    reward_arr = np.array([b["reward"] for b in dataset], dtype=np.float32).reshape(-1, 1)
    next_obs_arr = np.array([b["next_obs"] for b in dataset], dtype=np.float32)
    done_arr = np.array([b["done"] for b in dataset], dtype=np.float32).reshape(-1, 1)
    
    # 2. Offline Training Loop
    print(f"Starting IQL Offline Training for {total_epochs} epochs...")
    import math
    num_samples = len(dataset)
    if num_samples < batch_size:
        batch_size = num_samples
    # Use ceil to ensure all samples are included (avoids dropping the last partial batch)
    num_batches = max(1, math.ceil(num_samples / batch_size)) if num_samples > 0 else 0
    
    for epoch in range(total_epochs):
        # Seed per epoch for reproducibility: shuffle is deterministic but different per epoch
        rng = np.random.default_rng(seed=epoch)
        indices = rng.permutation(num_samples)
        
        epoch_v_loss = 0
        epoch_q_loss = 0
        epoch_pi_loss = 0
        
        for i in range(num_batches):
            batch_idx = indices[i*batch_size : (i+1)*batch_size]
            
            s = torch.tensor(obs_arr[batch_idx]).to(device)
            a_onehot = torch.tensor(action_onehot_arr[batch_idx]).to(device)
            a_idx = torch.tensor(action_idx_arr[batch_idx]).to(device)
            r = torch.tensor(reward_arr[batch_idx]).to(device)
            s_next = torch.tensor(next_obs_arr[batch_idx]).to(device)
            d = torch.tensor(done_arr[batch_idx]).to(device)
            
            # --- Update Value Network (Expectile Regression) ---
            with torch.no_grad():
                q1 = q1_target(torch.cat([s, a_onehot], dim=1))
                q2 = q2_target(torch.cat([s, a_onehot], dim=1))
                q_target_val_policy = torch.min(q1, q2)
                
            v = nets.v_net(s)
            v_loss = expectile_loss(q_target_val_policy - v, expectile).mean()
            
            optimizer_v.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(nets.v_net.parameters(), max_norm=1.0)
            optimizer_v.step()
            
            # --- Update Q Networks ---
            with torch.no_grad():
                v_next = nets.v_net(s_next)
                q_target_val = r + (1 - d) * gamma * v_next
                
            q1 = nets.q1_net(torch.cat([s, a_onehot], dim=1))
            q2 = nets.q2_net(torch.cat([s, a_onehot], dim=1))
            q_loss = F.mse_loss(q1, q_target_val) + F.mse_loss(q2, q_target_val)
            
            optimizer_q.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(nets.q1_net.parameters()) + list(nets.q2_net.parameters()), max_norm=1.0)
            optimizer_q.step()
            
            # --- Update Policy Network (Advantage Weighted Regression) ---
            with torch.no_grad():
                # Re-evaluate V using the UPDATED value network to prevent using a stale value
                v_updated = nets.v_net(s)
                adv = q_target_val_policy - v_updated
                weight = torch.exp(beta * adv).clamp(max=100.0)
                
            logits = nets.policy_net(s)
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, a_idx.unsqueeze(1))
            
            pi_loss = -(weight * action_log_probs).mean()
            
            optimizer_policy.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(nets.policy_net.parameters(), max_norm=1.0)
            optimizer_policy.step()
            
            # Soft update target networks
            with torch.no_grad():
                for param, target_param in zip(nets.q1_net.parameters(), q1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(nets.q2_net.parameters(), q2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
            epoch_v_loss += v_loss.item()
            epoch_q_loss += q_loss.item()
            epoch_pi_loss += pi_loss.item()
            
        if num_batches > 0:
            print(f"Epoch {epoch+1}/{total_epochs} | V Loss: {epoch_v_loss/num_batches:.4f} | Q Loss: {epoch_q_loss/num_batches:.4f} | Pi Loss: {epoch_pi_loss/num_batches:.4f}")
        else:
            print(f"Epoch {epoch+1}/{total_epochs} | No batches to train on.")
        
        scheduler_v.step()
        scheduler_q.step()
        scheduler_pi.step()
        
        # Save a checkpoint per epoch
        model_path = os.path.join(os.path.dirname(__file__), "aria_iql_universal.pth")
        torch.save(nets.state_dict(), model_path)
        
    print(f"IQL Model successfully saved to {model_path}")
    
    # 3. Run Benchmark Metrics
    if os.path.exists(dataset_file):
        print("Evaluating policy and running benchmark metrics...")
        run_benchmark(dataset_file)

if __name__ == "__main__":
    train_iql_policy()
