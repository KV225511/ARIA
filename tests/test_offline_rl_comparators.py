import numpy as np

from modules.module_07_rl.rl_spec import RL_ACTION_SPACE
from modules.module_07_rl.state_builder import STATE_DIM
from modules.module_14_evaluation.offline_rl_comparators import (
    ComparatorConfig,
    DecisionTransformerPolicy,
    GreedyNetworkPolicy,
    train_behavior_cloning,
    train_decision_transformer,
    train_discrete_cql,
)


def _rows(prefix: str, episode_count: int = 4, episode_length: int = 3):
    rows = []
    for episode in range(episode_count):
        for turn in range(episode_length):
            action = (episode + turn) % (len(RL_ACTION_SPACE) - 1)
            observation = np.zeros(STATE_DIM, dtype=np.float32)
            observation[action] = 1.0
            next_observation = observation.copy()
            mask = np.ones(len(RL_ACTION_SPACE), dtype=np.float32)
            done = turn == episode_length - 1
            next_mask = np.zeros_like(mask) if done else mask.copy()
            rows.append({
                "episode_id": f"{prefix}-{episode}",
                "obs": observation.tolist(),
                "next_obs": next_observation.tolist(),
                "action_idx": action,
                "reward": float(action == episode % (len(RL_ACTION_SPACE) - 1)),
                "done": done,
                "action_mask_before": mask.tolist(),
                "action_mask": next_mask.tolist(),
            })
    return rows


def test_transition_comparators_train_with_terminal_empty_masks():
    train_rows = _rows("train")
    validation_rows = _rows("validation", episode_count=2)
    config = ComparatorConfig(epochs=1, batch_size=8, hidden_dim=16,
                              context_length=4, transformer_layers=1,
                              transformer_heads=4)

    bc_model, bc_metrics = train_behavior_cloning(train_rows, validation_rows, config)
    cql_model, cql_metrics = train_discrete_cql(train_rows, validation_rows, config)

    assert np.isfinite(bc_metrics["history"][0]["train_loss"])
    assert np.isfinite(cql_metrics["history"][0]["train_loss"])
    for name, model in (("bc", bc_model), ("cql", cql_model)):
        policy = GreedyNetworkPolicy(model, name)
        decision = policy.select_action(np.zeros(STATE_DIM, dtype=np.float32),
                                        np.ones(len(RL_ACTION_SPACE), dtype=int))
        assert 0 <= decision["action_idx"] < len(RL_ACTION_SPACE)


def test_decision_transformer_trains_and_tracks_observed_reward():
    train_rows = _rows("train")
    validation_rows = _rows("validation", episode_count=2)
    config = ComparatorConfig(epochs=1, batch_size=8, hidden_dim=16,
                              context_length=4, transformer_layers=1,
                              transformer_heads=4)
    model, metrics = train_decision_transformer(train_rows, validation_rows, config)
    policy = DecisionTransformerPolicy(
        model,
        target_return=metrics["target_return"],
        return_scale=metrics["return_scale"],
    )
    before = policy.remaining_return
    decision = policy.select_action(np.zeros(STATE_DIM, dtype=np.float32),
                                    np.ones(len(RL_ACTION_SPACE), dtype=int))
    policy.observe_transition(0.25)

    assert np.isfinite(metrics["history"][0]["train_loss"])
    assert 0 <= decision["action_idx"] < len(RL_ACTION_SPACE)
    assert policy.remaining_return == before - 0.25
