# Issue #10 fix: sys.path.append must come before any relative imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.settings import TERMINATION_ENTROPY_THRESHOLD

# Shared constants for RL Formulation

# Issue #9 fix: RL_STATE_SCHEMA updated to reflect actual observation space.
# The real obs vector is: [padded_belief_flat (MAX_NODES*3=150), entropy (1), norm_turn_id (1)]
# Total = 152 dimensions. Fields below marked TODO are not yet in the obs but are planned.
RL_STATE_SCHEMA = {
    "padded_belief_vector": "float array (MAX_NODES * 3 = 150 dims, zero-padded)",
    "belief_entropy":       "float (1 dim) - global entropy across all skills",
    "normalized_turn_id":  "float (1 dim) - turn/30, clamped to [0,1]",
    # --- PLANNED (not yet in obs) ---
    "fused_vector":            "TODO: float array (fixed-dim output of fusion layer)",
    "cognitive_load_label":    "TODO: int (0=low, 1=anxiety, 2=ignorance)",
    "distress_score":          "TODO: float",
    "anti_gaming_active":      "TODO: int (0 or 1)",
    "topics_covered":          "TODO: int (count of nodes visited)",
    "consecutive_same_topic":  "TODO: int",
    "incongruence_score":      "TODO: float",
    "answer_consistency":      "TODO: float"
}

RL_ACTION_SPACE = [
    "increase_difficulty",       # 0
    "decrease_difficulty",       # 1
    "ask_follow_up_same_topic",  # 2
    "switch_topic",              # 3
    "probe_foundation",          # 4
    "ask_behavioral",            # 5
    "ask_situational",           # 6
    "conclude_interview"         # 7
]

RL_ACTION_MASKS = {
    # Define rules where certain actions are invalid
    # e.g., can't probe_foundation if node has no prerequisites
}

REWARD_COEFFICIENTS = {
    "alpha": 1.5,   # information gain weight
    "beta": 0.1,    # duration penalty weight
    "gamma": 0.5,   # signal consistency bonus weight
    "delta": 1.0,   # distress penalty weight
    "epsilon": 2.0, # integrity detection bonus weight
    "omega": 5.0,   # outcome alignment reward weight
    "zeta": 0.5     # dense belief alignment bonus weight
}
