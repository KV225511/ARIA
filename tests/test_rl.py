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

def test_env_termination_condition(env):
    """Test that the environment terminates when entropy is low or conclude action is taken."""
    env.reset()
    
    # Force the conclude action (index 7 based on RL_ACTION_SPACE)
    # Action space: 0: increase_difficulty, ... 7: conclude_interview
    obs, reward, terminated, truncated, info = env.step(7)
    assert terminated is True
    assert info["action"] == "conclude_interview"

def test_env_truncation_condition(env):
    """Test that the environment truncates at 30 turns."""
    env.reset()
    env.turn_id = 29
    
    # Step with a non-conclude action (e.g. 3: switch_topic)
    obs, reward, terminated, truncated, info = env.step(3)
    assert truncated is True
