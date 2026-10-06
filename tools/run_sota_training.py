"""Command-line entry point for reproducible ARIA SOTA experiments."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.module_07_rl.environment import ARIAInterviewEnv
from modules.module_14_evaluation.policy_comparison import (
    baseline_policy_factories,
    compare_policies,
    load_validated_iql,
)
from modules.module_14_evaluation.sota_training import run_preflight, run_text_benchmark


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run reproducible ARIA SOTA experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="audit local datasets and frozen artifacts")
    preflight.add_argument("--output", default="output/sota/preflight.json")
    text = subparsers.add_parser("text", help="run question-isolated Mohler lexical baselines")
    text.add_argument("--dataset", required=True)
    text.add_argument("--output-dir", default="output/sota/text")
    text.add_argument("--seed", type=int, default=42)
    policy = subparsers.add_parser("policy", help="run matched synthetic policy comparisons")
    policy.add_argument("--output", default="output/sota/policy_baselines.json")
    policy.add_argument("--episodes-per-class", type=int, default=10)
    policy.add_argument("--seed", type=int, default=42)
    policy.add_argument("--include-iql", action="store_true",
                        help="validate and compare the frozen v7 IQL checkpoint")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        report = run_preflight(ROOT, ROOT / args.output)
    elif args.command == "text":
        report = run_text_benchmark(args.dataset, args.output_dir, seed=args.seed)
    else:
        factories = baseline_policy_factories()
        env_factory = ARIAInterviewEnv
        metadata = {}
        if args.include_iql:
            iql, belief_config, assets = load_validated_iql(ROOT)
            factories["aria_iql_v7"] = lambda _seed: iql
            env_factory = lambda: ARIAInterviewEnv(belief_config=belief_config)
            metadata["aria_iql_v7"] = {
                "checkpoint_sha256": iql.checkpoint_sha256,
                "protocol_hash": iql.metadata["protocol_hash"],
                "protocol_status": iql.metadata["protocol_status"],
                "assets": {name: str(path.relative_to(ROOT)) for name, path in assets.items()},
            }
        report = compare_policies(
            factories, ROOT / args.output,
            episodes_per_class=args.episodes_per_class, seed=args.seed,
            env_factory=env_factory, policy_metadata=metadata,
        )
    summary = {
        "schema_version": report["schema_version"],
        "report_hash": report["report_hash"],
    }
    if args.command == "preflight":
        summary.update({"available_tasks": report["available_tasks"],
                        "missing_tasks": report["missing_tasks"]})
    elif args.command == "text":
        summary.update({
            "models": {item["model_name"]: item["test_metrics"] for item in report["models"]},
            "output_dir": str(Path(args.output_dir).resolve()),
        })
    else:
        summary.update({
            "policies": {name: value["summary"] for name, value in report["policies"].items()},
            "output": str((ROOT / args.output).resolve()),
        })
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
