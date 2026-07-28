# Shared constants for RL Formulation

RL_STATE_SCHEMA = {
    "belief_vector": "float array (length varies by ontology)",
    "belief_entropy": "float",
    "fused_vector": "float array (fixed-dim output of fusion layer)",
    "cognitive_load_label": "int (0=low, 1=anxiety, 2=ignorance)",
    "distress_score": "float",
    "anti_gaming_active": "int (0 or 1)",
    "turn_id": "int",
    "topics_covered": "int (count of nodes visited)",
    "consecutive_same_topic": "int",
    "incongruence_score": "float",
    "answer_consistency": "float"
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
    "omega": 5.0    # outcome alignment reward weight
}

TERMINATION_ENTROPY_THRESHOLD = 0.3
