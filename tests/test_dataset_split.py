from modules.module_07_rl.dataset_split import (
    apply_terminal_outcome_rewards,
    group_transitions_into_episodes,
    split_by_resume_jd_group,
)


def _episode(episode_id, resume, jd, label):
    return [
        {
            "episode_id": episode_id,
            "resume_file": resume,
            "jd_file": jd,
            "true_label": label,
            "aria_label": label,
            "done": turn == 1,
        }
        for turn in range(2)
    ]


def test_group_transitions_uses_episode_id():
    transitions = _episode("a", "r1", "j1", 0) + _episode("b", "r2", "j2", 1)
    episodes = group_transitions_into_episodes(transitions)
    assert len(episodes) == 2
    assert all(len(episode) == 2 for episode in episodes)


def test_split_keeps_resume_jd_groups_isolated():
    transitions = []
    for index in range(12):
        transitions.extend(
            _episode(
                f"ep-{index}",
                f"resume-{index // 2}",
                f"jd-{index // 2}",
                index % 3,
            )
        )

    splits = split_by_resume_jd_group(transitions, seed=7)
    group_owners = {}
    for split_name, split_transitions in splits.items():
        for transition in split_transitions:
            key = (transition["resume_file"], transition["jd_file"])
            assert key not in group_owners or group_owners[key] == split_name
            group_owners[key] = split_name
            assert transition["dataset_split"] == split_name

    assert sum(len(items) for items in splits.values()) == len(transitions)


def test_split_keeps_connected_resume_and_jd_identities_together():
    transitions = (
        _episode("a", "shared-resume", "jd-1", 0)
        + _episode("b", "shared-resume", "jd-2", 1)
        + _episode("c", "resume-3", "jd-2", 2)
        + _episode("d", "resume-4", "jd-4", 0)
    )
    splits = split_by_resume_jd_group(transitions, seed=3)
    owners = {}
    for split_name, items in splits.items():
        for transition in items:
            owners[transition["episode_id"]] = split_name

    assert owners["a"] == owners["b"] == owners["c"]


def test_outcome_reward_only_changes_terminal_training_transition():
    transitions = [
        {"done": False, "true_label": 2, "aria_label": 0, "reward": 1.0},
        {"done": True, "true_label": 2, "aria_label": 0, "reward": 1.0},
        {"done": True, "true_label": 1, "aria_label": 1, "reward": 1.0},
    ]
    mismatches = apply_terminal_outcome_rewards(transitions, omega=5.0)
    assert transitions[0]["reward"] == 1.0
    assert transitions[1]["reward"] == -4.0
    assert transitions[2]["reward"] == 6.0
    assert mismatches == 1


def test_terminal_outcome_reward_cannot_be_applied_twice():
    transitions = [
        {"done": True, "true_label": 1, "aria_label": 1, "reward": 0.5},
    ]
    apply_terminal_outcome_rewards(transitions, omega=5.0)
    try:
        apply_terminal_outcome_rewards(transitions, omega=5.0)
    except ValueError as error:
        assert "already applied" in str(error)
    else:
        raise AssertionError("Expected terminal reward double-application rejection")
