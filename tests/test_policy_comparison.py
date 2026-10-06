from modules.module_14_evaluation.policy_comparison import (
    FixedScriptPolicy,
    UncertaintyGreedyPolicy,
    compare_policies,
    validated_iql_assets,
)


def test_policy_comparison_uses_matched_scenarios(tmp_path):
    factories = {
        "fixed": lambda seed: FixedScriptPolicy(seed),
        "uncertainty": lambda seed: UncertaintyGreedyPolicy("uncertainty", seed),
    }
    report = compare_policies(factories, tmp_path / "policies.json", episodes_per_class=1, seed=5)
    assert report["matched_scenarios"] is True
    assert report["scenario_count"] == 3
    assert set(report["policies"]) == set(factories)
    assert all(value["summary"]["num_episodes"] == 3 for value in report["policies"].values())
    assert (tmp_path / "policies.json").is_file()


def test_validated_iql_asset_map_points_to_frozen_files():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    assets = validated_iql_assets(root)
    assert set(assets) == {
        "checkpoint", "protocol", "protocol_state", "development_bundle",
        "split_manifest", "belief_config",
    }
    assert all(path.is_file() for path in assets.values())
