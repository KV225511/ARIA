# Issue #10 fix: sys.path.append must come before any relative imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.settings import TERMINATION_ENTROPY_THRESHOLD

# Shared constants for RL Formulation

# State-v3 is fixed and ontology/JD permutation invariant. The canonical ordered
# names live in state_builder.STATE_FEATURE_NAMES; keeping that definition in
# one place prevents the schema documentation from drifting from replay.
RL_STATE_SCHEMA = {
    "schema_version": "aria-state-v3",
    "dimensions": 33,
    "feature_names_source": "modules.module_07_rl.state_builder.STATE_FEATURE_NAMES",
    "action_mask_in_state": False,
}

ACTION_SCHEMA_VERSION = "aria-action-v3"

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
    "zeta": 0.5     # first-visit skill coverage bonus
}
