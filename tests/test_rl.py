import pytest
import numpy as np
import random
from modules.module_07_rl.environment import ARIAInterviewEnv
from modules.module_07_rl.rl_spec import TERMINATION_ENTROPY_THRESHOLD
from modules.module_07_rl.llm_simulator import (
    ACTION_TO_INDEX,
    behavior_action_distribution,
    select_behavior_action,
)
from modules.module_07_rl.state_builder import STATE_DIM

@pytest.fixture
def env():
    return ARIAInterviewEnv("backend_developer")

def test_env_initialization(env):
    """Test that the environment initializes correctly."""
    assert env.num_nodes == 17  # Backend developer ontology has 17 nodes
    assert env.observation_space.shape[0] == STATE_DIM
    assert env.action_space.n == 8

def test_env_reset(env):
    """Test the reset function."""
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    assert env.turn_id == 0
    assert env.observation_space.contains(obs)

def test_env_step_random_action(env):
    """Test taking a random step."""
    env.reset()
    action = ACTION_TO_INDEX["switch_topic"]
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "info_gain" in info
    assert "action" in info
    assert env.turn_id == 1

def test_env_blocks_premature_conclusion(env):
    """Conclusion is invalid until minimum turns and skill coverage are met."""
    env.reset()
    obs, reward, terminated, truncated, info = env.step(7)
    assert terminated is False
    assert info["conclude_blocked"] is True
    assert info["action"] == "conclude_interview"
    assert env.turn_id == 0
    assert env.valid_evidence_count == 0
    assert info["action_mask_before"][7] == 0.0


def test_env_terminates_after_coverage_contract(env):
    env.reset()
    for skill in env.nodes[:5]:
        env.belief_updater.update_belief(skill, 0.8, "low", 0.8)
    env.turn_id = 10
    env.valid_evidence_count = 5
    beliefs_before = {
        skill: env.belief_updater.get_belief(skill).copy() for skill in env.nodes
    }
    _, _, terminated, _, info = env.step_with_scores(
        7, None, None, None
    )
    assert terminated is True
    assert info["conclusion_allowed"] is True
    assert info["termination_reason"] == "explicit_conclusion"
    assert env.turn_id == 10
    assert all(
        np.array_equal(beliefs_before[skill], env.belief_updater.get_belief(skill))
        for skill in env.nodes
    )


def test_step_updates_explicit_target_skill(env):
    env.reset()
    target = env.nodes[-1]
    env.step_with_scores(2, 0.9, 0.9, "low", target_skill=target)
    assert env.belief_updater.get_evidence_count(target) == 1
    assert env.belief_updater.get_visited_skills() == [target]


def test_behavior_policy_prioritizes_coverage(env):
    env.reset()
    action_idx, policy_name = select_behavior_action(env, random.Random(0))
    assert action_idx == ACTION_TO_INDEX["switch_topic"]
    assert policy_name == "coverage_heuristic"


def test_behavior_policy_logs_normalized_support_and_masks_stop(env):
    env.reset()
    probabilities, _ = behavior_action_distribution(env)
    assert len(probabilities) == env.action_space.n
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(probability > 0.0 for probability in probabilities[:7])
    assert probabilities[7] == 0.0


def test_question_turn_does_not_auto_terminate_on_confidence(env):
    env.reset()
    env.turn_id = 10
    env.valid_evidence_count = 5
    for skill in env.nodes[:5]:
        env.belief_updater.update_belief(skill, 0.95, "low", 0.95)
    _, _, terminated, _, info = env.step_with_scores(
        ACTION_TO_INDEX["ask_follow_up_same_topic"],
        0.95,
        0.95,
        "low",
        target_skill=env.nodes[0],
    )
    assert terminated is False
    assert info["termination_reason"] is None

def test_env_truncation_condition(env):
    """Test that the environment truncates at 30 turns."""
    env.reset()
    env.turn_id = 29
    
    # Step with a non-conclude action (e.g. 3: switch_topic)
    obs, reward, terminated, truncated, info = env.step(3)
    assert truncated is True


def test_ignorance_is_not_treated_as_distress(env):
    env.reset()
    target = env.nodes[0]
    _, ignorance_reward, *_ = env.step_with_scores(
        3, 0.1, 0.1, "ignorance", target_skill=target
    )
    env.reset()
    _, anxiety_reward, *_ = env.step_with_scores(
        3, 0.1, 0.1, "anxiety", target_skill=target
    )
    assert ignorance_reward > anxiety_reward
