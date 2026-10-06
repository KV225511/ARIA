"""Create a compact, manuscript-ready summary from full policy rollout reports."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json


DIRECT_POLICIES = (
    "aria_iql_v7",
    "behavior_cloning",
    "discrete_cql",
    "decision_transformer",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_summary(report_files: list[Path], training_report_file: Path) -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_files]
    training = json.loads(training_report_file.read_text(encoding="utf-8"))
    conditions = {}
    rows = []
    for path, report in zip(report_files, reports):
        condition = report["evidence_condition"]["name"]
        if condition in conditions:
            raise ValueError(f"duplicate evidence condition: {condition}")
        conditions[condition] = {
            "source_file": str(path),
            "source_sha256": _sha256(path),
            "report_hash": report["report_hash"],
            "scenario_count": report["scenario_count"],
            "accuracy_saturated": report["ceiling_audit"]["accuracy_saturated"],
        }
        for policy, result in report["policies"].items():
            summary = result["summary"]
            rows.append({
                "condition": condition,
                "policy": policy,
                "episodes": summary["num_episodes"],
                "accuracy": summary["accuracy"],
                "macro_f1": summary["macro_f1"],
                "reward_mean": summary["reward"]["mean"],
                "reward_ci_lower": summary["reward"]["ci_95"]["lower"],
                "reward_ci_upper": summary["reward"]["ci_95"]["upper"],
                "information_gain_mean": summary["information_gain"]["mean"],
                "turns_mean": summary["turns"]["mean"],
                "skills_covered_mean": summary["skills_covered"]["mean"],
            })
    return {
        "schema_version": "aria-policy-summary-v1",
        "development_only": True,
        "locked_test_used": False,
        "training_report": {
            "source_file": str(training_report_file),
            "source_sha256": _sha256(training_report_file),
            "report_hash": training["report_hash"],
            "test_split_used": training["test_split_used"],
            "comparators": training["comparators"],
        },
        "conditions": conditions,
        "rows": rows,
    }


def write_outputs(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "policy_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "policy_comparison_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])

    indexed = {(row["condition"], row["policy"]): row for row in summary["rows"]}
    lines = [
        "# Direct policy comparison on frozen ARIA development data",
        "",
        "These are controlled synthetic development results. The locked test split was not used. "
        "They support comparison of policy behavior under the same ARIA interface, but they do not "
        "establish real-candidate performance or a universal state-of-the-art claim.",
        "",
        "## Comparator training",
        "",
        "| Comparator | Best validation logged-action accuracy |",
        "|---|---:|",
    ]
    for name in ("behavior_cloning", "discrete_cql", "decision_transformer"):
        value = summary["training_report"]["comparators"][name]["best_validation_action_accuracy"]
        lines.append(f"| {name.replace('_', ' ')} | {value:.4f} |")
    lines.extend([
        "",
        "Validation action accuracy measures imitation of logged actions. It is a selection diagnostic, "
        "not the rollout outcome metric.",
        "",
        "## Matched rollout results",
        "",
        "Each cell is accuracy / mean designed reward / mean information gain over 45 matched episodes "
        "(15 per synthetic class).",
        "",
        "| Policy | Base | Overlap | Low confidence | Positive shift |",
        "|---|---:|---:|---:|---:|",
    ])
    condition_order = ("base", "overlap", "low_confidence", "positive_shift")
    for policy in DIRECT_POLICIES:
        cells = []
        for condition in condition_order:
            row = indexed[(condition, policy)]
            cells.append(f"{row['accuracy']:.4f} / {row['reward_mean']:.2f} / {row['information_gain_mean']:.2f}")
        lines.append(f"| {policy.replace('_', ' ')} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Base accuracy is saturated: multiple learned and non-learned policies reach 1.0 because the "
        "evidence centers are widely separated. Accuracy must therefore not be used to rank policies in "
        "that condition. Under overlap, IQL does not dominate classification accuracy: behavior "
        "cloning reaches 0.5333, discrete CQL 0.5111, Decision Transformer 0.4889, and ARIA IQL 0.4667. "
        "Across all four conditions, ARIA IQL has the largest mean designed reward and information gain "
        "among the direct learned comparators, while also covering all 17 skills. These outcomes show "
        "behavior under ARIA's designed simulator and reward; they do not demonstrate human interview quality.",
        "",
        "## Provenance",
        "",
        f"Comparator training report hash: `{summary['training_report']['report_hash']}`.",
        "",
    ])
    for condition in condition_order:
        meta = summary["conditions"][condition]
        lines.append(f"- `{condition}`: report hash `{meta['report_hash']}`; source SHA-256 `{meta['source_sha256']}`")
    (output_dir / "policy_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--reports", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = build_summary(args.reports, args.training_report)
    write_outputs(summary, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "conditions": sorted(summary["conditions"]),
                      "rows": len(summary["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
