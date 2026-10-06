"""Render a comparison table only from generated local experiment reports.

Published results belong in the paper's cited related-work table. This utility
does not copy unverified literature values or invent metrics when a run is absent.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json


def _load(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def render_local_comparison(*, text_report: str | Path | None = None,
                            policy_report: str | Path | None = None) -> str:
    rows = []
    evidence = []
    if text_report:
        report = _load(text_report)
        for model in report.get("models", []):
            metrics = model["test_metrics"]
            rows.append(("Text", model["model_name"], metrics["accuracy"],
                         metrics["macro_f1"], report["split_manifest_hash"]))
        evidence.append(f"Text report: `{Path(text_report)}`")
    if policy_report:
        report = _load(policy_report)
        for name, result in report.get("policies", {}).items():
            summary = result["summary"]
            rows.append(("Policy simulation", name, summary["accuracy"], None,
                         report["report_hash"]))
        evidence.append(f"Policy report: `{Path(policy_report)}`")
    if not rows:
        raise ValueError("provide at least one generated experiment report")
    lines = [
        "# ARIA locally reproduced comparison", "",
        "These values come from repository-generated artifacts. Policy accuracy is from a",
        "controlled synthetic response model and is not a real-candidate performance claim.", "",
        "| Task | Model or policy | Accuracy | Macro-F1 | Evidence hash |",
        "|---|---|---:|---:|---|",
    ]
    for task, name, accuracy, macro_f1, digest in rows:
        f1 = "n/a" if macro_f1 is None else f"{macro_f1:.4f}"
        lines.append(f"| {task} | {name} | {accuracy:.4f} | {f1} | `{digest}` |")
    lines.extend(["", "## Source artifacts", "", *(f"- {item}" for item in evidence)])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-report")
    parser.add_argument("--policy-report")
    parser.add_argument("--output", default="output/sota/local_comparison.md")
    args = parser.parse_args(argv)
    content = render_local_comparison(text_report=args.text_report, policy_report=args.policy_report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
