import pytest
import numpy as np
from modules.module_07_rl.environment import ARIAInterviewEnv
from modules.module_07_rl.rl_spec import TERMINATION_ENTROPY_THRESHOLD

@pytest.fixture
def env():
    return ARIAInterviewEnv("backend_developer")

def test_env_initialization(env):
    """Test that the environment initializes correctly."""
    assert env.num_nodes == 17  # Backend developer ontology has 17 nodes
    assert env.observation_space.shape[0] == (50 * 3) + 2  # Padded beliefs (MAX_NODES=50) + entropy + turn = 152
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
    action = env.action_space.sample()
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


def test_env_terminates_after_coverage_contract(env):
    env.reset()
    for skill in env.nodes[:5]:
        env.belief_updater.update_belief(skill, 0.8, "low", 0.8)
    env.turn_id = 9
    target = env.nodes[0]
    _, _, terminated, _, info = env.step_with_scores(
        7, 0.8, 0.8, "low", target_skill=target
    )
    assert terminated is True
    assert info["conclusion_allowed"] is True


def test_step_updates_explicit_target_skill(env):
    env.reset()
    target = env.nodes[-1]
    env.step_with_scores(2, 0.9, 0.9, "low", target_skill=target)
    assert env.belief_updater.get_evidence_count(target) == 1
    assert env.belief_updater.get_visited_skills() == [target]

def test_env_truncation_condition(env):
    """Test that the environment truncates at 30 turns."""
    env.reset()
    env.turn_id = 29
    
    # Step with a non-conclude action (e.g. 3: switch_topic)
    obs, reward, terminated, truncated, info = env.step(3)
    assert truncated is True
