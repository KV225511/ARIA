"""Versioned, state-conditioned behavior policy and generation targets."""

BEHAVIOR_POLICY_VERSION = "aria-behavior-policy-v2"
PAIR_PLAN_SCHEMA_VERSION = "aria-pair-plan-v2"

TARGET_SWITCH_SHARE = (0.25, 0.35)
HARD_MAX_SWITCH_SHARE = 0.40
TARGET_MAX_TURN_RATE = 0.10
HARD_MAX_TURN_RATE = 0.15
TARGET_NO_OVERLAP_RATIO = 0.30
TARGET_NO_OVERLAP_BAND = (0.25, 0.35)
HARD_NO_OVERLAP_BAND = (0.20, 0.40)
MIN_DISTRIBUTION_GATE_EPISODES = 60

RECOVERY_TURN = 20
FORCED_CONCLUSION_TURN = 25

# All weights are positive so every legal question action retains logged support.
COVERAGE_WEIGHTS = {
    "increase_difficulty": 0.10,
    "decrease_difficulty": 0.10,
    "ask_follow_up_same_topic": 0.10,
    "switch_topic": 0.35,
    "probe_foundation": 0.10,
    "ask_behavioral": 0.10,
    "ask_situational": 0.15,
}
RECOVERY_WEIGHTS = {
    "increase_difficulty": 0.075,
    "decrease_difficulty": 0.075,
    "ask_follow_up_same_topic": 0.05,
    "switch_topic": 0.60,
    "probe_foundation": 0.075,
    "ask_behavioral": 0.05,
    "ask_situational": 0.075,
}
POST_COVERAGE_WEIGHTS = {
    0: {
        "increase_difficulty": 0.05, "decrease_difficulty": 0.20,
        "ask_follow_up_same_topic": 0.15, "switch_topic": 0.15,
        "probe_foundation": 0.20, "ask_behavioral": 0.10,
        "ask_situational": 0.15,
    },
    1: {
        "increase_difficulty": 0.10, "decrease_difficulty": 0.10,
        "ask_follow_up_same_topic": 0.20, "switch_topic": 0.15,
        "probe_foundation": 0.10, "ask_behavioral": 0.15,
        "ask_situational": 0.20,
    },
    2: {
        "increase_difficulty": 0.20, "decrease_difficulty": 0.05,
        "ask_follow_up_same_topic": 0.15, "switch_topic": 0.15,
        "probe_foundation": 0.10, "ask_behavioral": 0.15,
        "ask_situational": 0.20,
    },
}
