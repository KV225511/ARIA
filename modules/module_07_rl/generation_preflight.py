"""Fail-closed source validation before expensive ARIA Ollama generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.module_07_rl.data_loader import (
    DEFAULT_CLEANED_RESUME_CSV,
    DEFAULT_RESUME_CATEGORIES,
    get_resume_source_manifest,
    get_valid_jd_documents,
)


def _atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_preflight_report(
    *,
    resume_csv: str | Path,
    resume_categories: tuple[str, ...] | list[str],
    identity_components: tuple[int, int, int],
) -> dict:
    if len(identity_components) != 3 or any(value <= 0 for value in identity_components):
        raise ValueError("identity_components must contain three positive counts")
    required_components = sum(identity_components)
    resume_manifest = get_resume_source_manifest(
        "csv", resume_csv, resume_categories
    )

    _, jd_report = get_valid_jd_documents()
    unique_readable_jds = jd_report["unique_readable_content_hashes"]
    checks = {
        "enough_unique_resumes": (
            resume_manifest["unique_content_hashes"] >= required_components
        ),
        "enough_unique_readable_jds": unique_readable_jds >= required_components,
        "no_duplicate_jd_content": not jd_report["duplicate_content_groups"],
        "no_unreadable_selected_jds": not jd_report["unreadable_files"],
    }
    return {
        "schema_version": "aria-generation-preflight-v1",
        "required_identity_components": required_components,
        "identity_component_targets": list(identity_components),
        "resume_source": resume_manifest,
        "job_descriptions": jd_report,
        "shortfall": {
            "unique_resumes": max(
                required_components - resume_manifest["unique_content_hashes"], 0
            ),
            "unique_readable_jds": max(
                required_components - unique_readable_jds, 0
            ),
        },
        "checks": checks,
        "passes_preflight": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-csv", default=str(DEFAULT_CLEANED_RESUME_CSV))
    parser.add_argument(
        "--resume-categories",
        nargs="+",
        default=list(DEFAULT_RESUME_CATEGORIES),
    )
    parser.add_argument(
        "--identity-components",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(20, 6, 6),
    )
    parser.add_argument(
        "--output",
        default="data/synthetic/v3/reports/input_preflight.json",
    )
    args = parser.parse_args()
    try:
        report = build_preflight_report(
            resume_csv=args.resume_csv,
            resume_categories=tuple(args.resume_categories),
            identity_components=tuple(args.identity_components),
        )
        _atomic_json_write(Path(args.output), report)
    except (OSError, ValueError) as error:
        parser.exit(1, f"[ERROR] {error}\n")
    print(json.dumps(report, indent=2))
    if not report["passes_preflight"]:
        parser.exit(1, "[ERROR] Generation input preflight failed\n")


if __name__ == "__main__":
    main()
