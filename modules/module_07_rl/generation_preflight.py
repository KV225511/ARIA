"""Fail-closed source validation before expensive ARIA Ollama generation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from modules.module_05_ontology.grounding import build_role_profile, validate_role_profile
from modules.module_07_rl.data_loader import (
    DEFAULT_CLEANED_RESUME_CSV,
    DEFAULT_RESUME_CATEGORIES,
    get_resume_source_manifest,
    get_valid_jd_documents,
    extract_text_from_pdf,
    get_resume_documents,
)
from modules.module_07_rl.generation_policy import TARGET_NO_OVERLAP_RATIO


def _atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def get_groundable_jd_documents(directory: Path | None = None) -> tuple[list[Path], dict]:
    if directory is None:
        jd_paths, jd_report = get_valid_jd_documents()
    else:
        jd_paths, jd_report = get_valid_jd_documents(directory)
    accepted = []
    ungroundable_jds = []
    grounded_skill_counts = {}
    for path in jd_paths:
        jd_text = extract_text_from_pdf(str(path))
        try:
            profile = build_role_profile(jd_text)
            validate_role_profile(profile, jd_text)
            grounded_skill_counts[path.name] = len(profile.skills)
            accepted.append(path)
        except ValueError as error:
            ungroundable_jds.append({"file": path.name, "reason": str(error)})
    return accepted, {
        **jd_report,
        "groundable_role_profiles": len(accepted),
        "ungroundable_role_profiles": ungroundable_jds,
        "grounded_skill_counts": grounded_skill_counts,
    }


def build_preflight_report(
    *,
    resume_csv: str | Path,
    resume_categories: tuple[str, ...] | list[str],
    identity_components: tuple[int, int, int],
    max_episodes: int | None = None,
    no_evidence_overlap_target: float = TARGET_NO_OVERLAP_RATIO,
    seed: int = 42,
) -> dict:
    if len(identity_components) != 3 or any(value <= 0 for value in identity_components):
        raise ValueError("identity_components must contain three positive counts")
    if not 0.0 <= no_evidence_overlap_target <= 1.0:
        raise ValueError("no_evidence_overlap_target must be between 0.0 and 1.0")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    required_components = sum(identity_components)
    resume_manifest = get_resume_source_manifest(
        "csv", resume_csv, resume_categories
    )

    jd_paths, jd_report = get_groundable_jd_documents()
    groundable_jds = len(jd_paths)
    unique_readable_jds = jd_report["unique_readable_content_hashes"]
    checks = {
        "enough_unique_resumes": (
            resume_manifest["unique_content_hashes"] >= required_components
        ),
        "enough_unique_readable_jds": unique_readable_jds >= required_components,
        "no_duplicate_jd_content": not jd_report["duplicate_content_groups"],
        "no_unreadable_selected_jds": not jd_report["unreadable_files"],
        "enough_groundable_jds": groundable_jds >= required_components,
    }
    pairing_plan = None
    if max_episodes is not None and all(checks.values()):
        # Lazy import avoids a simulator/preflight import cycle during startup.
        from modules.module_07_rl.llm_simulator import (
            balance_pairing_classes_by_persona,
            build_pairing_class_map,
            build_split_safe_sweep_pairs,
        )
        resumes = get_resume_documents("csv", resume_csv, resume_categories)
        classes = build_pairing_class_map(resumes, jd_paths)
        compatibility_material = sorted(
            (resume, jd, pairing_class)
            for (resume, jd), pairing_class in classes.items()
        )
        compatibility_hash = hashlib.sha256(json.dumps(
            compatibility_material, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        compatibility_counts = dict(Counter(classes.values()))
        try:
            pairs = build_split_safe_sweep_pairs(
                resumes, jd_paths, max_episodes, seed=seed,
                component_targets=identity_components, pairing_classes=classes,
                no_overlap_target=no_evidence_overlap_target,
            )
            pairs = balance_pairing_classes_by_persona(pairs, classes, start_index=0)
        except ValueError as error:
            checks["pairing_mix_feasible"] = False
            pairing_plan = {
                "simulation_seed": seed,
                "pairing_compatibility_hash": compatibility_hash,
                "candidate_pairing_class_counts": compatibility_counts,
                "error": str(error),
            }
        else:
            no_overlap = sum(classes[pair] == "no_evidence_overlap" for pair in pairs)
            pairs_hash = hashlib.sha256(json.dumps(
                pairs, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            pairing_plan = {
                "simulation_seed": seed,
                "planned_pairs_hash": pairs_hash,
                "pairing_compatibility_hash": compatibility_hash,
                "candidate_pairing_class_counts": compatibility_counts,
                "planned_episodes": len(pairs),
                "planned_no_evidence_overlap_count": no_overlap,
                "planned_no_evidence_overlap_ratio": no_overlap / len(pairs),
            }
            checks["pairing_mix_feasible"] = True
    return {
        "schema_version": "aria-generation-preflight-v3",
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
            "groundable_jds": max(required_components - groundable_jds, 0),
        },
        "checks": checks,
        "pairing_plan": pairing_plan,
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
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-evidence-overlap-target", type=float,
        default=TARGET_NO_OVERLAP_RATIO,
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
            max_episodes=args.max_episodes,
            no_evidence_overlap_target=args.no_evidence_overlap_target,
            seed=args.seed,
        )
        _atomic_json_write(Path(args.output), report)
    except (OSError, ValueError) as error:
        parser.exit(1, f"[ERROR] {error}\n")
    print(json.dumps(report, indent=2))
    if not report["passes_preflight"]:
        parser.exit(1, "[ERROR] Generation input preflight failed\n")


if __name__ == "__main__":
    main()
