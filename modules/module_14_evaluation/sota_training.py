"""Reproducible local training and evaluation for ARIA's SOTA study.

The module deliberately separates locally reproduced measurements from published
comparison values.  It never substitutes fallback metrics when data or artifacts
are missing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import csv
import platform
import sys
import time

import numpy as np

from modules.module_07_rl.calibration_protocol import atomic_json_write, canonical_json_hash


SOTA_PROTOCOL_VERSION = "aria-sota-training-v1"
TEXT_REPORT_VERSION = "aria-text-benchmark-v1"
SPLIT_MANIFEST_VERSION = "aria-group-split-v1"


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependency_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


@dataclass(frozen=True)
class AssetSpec:
    key: str
    relative_candidates: tuple[str, ...]
    required_for: tuple[str, ...]
    description: str


ASSETS = (
    AssetSpec("fer2013", ("data/model1_video/fer2013",), ("vision",), "FER2013 image root"),
    AssetSpec("ravdess", ("data/model2_audio/ravdess",), ("audio",), "RAVDESS audio root"),
    AssetSpec(
        "mohler",
        ("data/model3_text/mohler_dataset.parquet", "data/model3_text/mohler_dataset.csv"),
        ("text",),
        "Mohler short-answer dataset",
    ),
    AssetSpec("mosei", ("data/model4_fusion",), ("fusion",), "CMU-MOSEI feature root"),
    AssetSpec("box_of_lies", ("data/model5_incongruence",), ("incongruence",), "Box of Lies feature root"),
    AssetSpec(
        "iql_checkpoint",
        ("modules/module_07_rl/aria_iql_belief_v7.pth",),
        ("policy",),
        "Frozen v7 IQL checkpoint",
    ),
)

HISTORICAL_RESULT_REPORTS = (
    "ARIA_updated.md",
    "ARIA_PROJECT_CONTEXT_AND_RESULTS.md",
    "ARIA_Benchmark_Diagnosis.md",
)


def run_preflight(project_root: str | Path, output_file: str | Path | None = None) -> dict:
    root = Path(project_root).resolve()
    records = []
    for spec in ASSETS:
        candidates = [root / candidate for candidate in spec.relative_candidates]
        found = next((path for path in candidates if path.exists()), None)
        records.append(
            {
                "key": spec.key,
                "description": spec.description,
                "required_for": list(spec.required_for),
                "status": "available" if found else "missing",
                "path": str(found.relative_to(root)) if found else None,
                "candidates": list(spec.relative_candidates),
                "sha256": file_sha256(found) if found and found.is_file() else None,
            }
        )
    historical_results = []
    for relative_path in HISTORICAL_RESULT_REPORTS:
        path = root / relative_path
        historical_results.append({
            "path": relative_path,
            "status": "available" if path.is_file() else "missing",
            "sha256": file_sha256(path) if path.is_file() else None,
            "evidence_class": "historical_markdown_report",
            "rerunnable_from_current_checkout": False,
        })
    report = {
        "schema_version": SOTA_PROTOCOL_VERSION,
        "project_root": str(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": {
            name: _dependency_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "torch", "transformers", "pyarrow")
        },
        "assets": records,
        "historical_results": historical_results,
        "available_tasks": sorted(
            task for record in records if record["status"] == "available" for task in record["required_for"]
        ),
        "missing_tasks": sorted(
            task for record in records if record["status"] == "missing" for task in record["required_for"]
        ),
        "policy": (
            "Missing assets are reported and never replaced by fallback scores. "
            "Historical Markdown results are inventoried separately from rerunnable experiments."
        ),
    }
    report["report_hash"] = canonical_json_hash(report)
    if output_file:
        atomic_json_write(Path(output_file), report)
    return report


def _stable_group_order(groups: Iterable[str], seed: int) -> list[str]:
    return sorted(set(map(str, groups)), key=lambda value: sha256(f"{seed}:{value}".encode()).hexdigest())


def build_group_split_manifest(
    sample_ids: list[str],
    group_ids: list[str],
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict:
    if len(sample_ids) != len(group_ids) or not sample_ids:
        raise ValueError("sample_ids and group_ids must be non-empty and equally sized")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique")
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1 - train_fraction:
        raise ValueError("invalid split fractions")
    ordered_groups = _stable_group_order(group_ids, seed)
    if len(ordered_groups) < 3:
        raise ValueError("group-isolated train/validation/test splitting requires at least three groups")
    target_train = max(1, round(len(ordered_groups) * train_fraction))
    target_validation = max(1, round(len(ordered_groups) * validation_fraction))
    if target_train + target_validation >= len(ordered_groups):
        target_train = len(ordered_groups) - 2
        target_validation = 1
    assignments = {}
    for index, group in enumerate(ordered_groups):
        assignments[group] = (
            "train" if index < target_train
            else "validation" if index < target_train + target_validation
            else "test"
        )
    samples = [
        {"sample_id": str(sample), "group_id": str(group), "split": assignments[str(group)]}
        for sample, group in zip(sample_ids, group_ids)
    ]
    manifest = {
        "schema_version": SPLIT_MANIFEST_VERSION,
        "seed": int(seed),
        "fractions": {"train": train_fraction, "validation": validation_fraction,
                      "test": 1.0 - train_fraction - validation_fraction},
        "group_assignments": assignments,
        "samples": samples,
        "counts": dict(Counter(item["split"] for item in samples)),
        "group_counts": dict(Counter(assignments.values())),
    }
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    return manifest


def _read_frame(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Text benchmarking requires pandas") from exc
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Mohler input must be .parquet or .csv")


def _find_column(columns: Iterable[str], candidates: Iterable[str], description: str) -> str:
    normalized = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(f"dataset has no {description} column; tried {list(candidates)}")


def _score_to_label(score: float) -> str:
    return "high" if float(score) >= 4.0 else "medium" if float(score) >= 2.5 else "low"


def _classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
    labels = ["low", "medium", "high"]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def run_text_benchmark(
    dataset_file: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict:
    """Train matched lexical baselines on a frozen question-isolated split."""
    started = time.time()
    dataset_path = Path(dataset_file).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    frame = _read_frame(dataset_path).copy()
    answer_col = _find_column(frame.columns, ("student_answer", "answer", "student"), "student answer")
    reference_col = _find_column(frame.columns, ("instructor_answer", "reference_answer", "reference"), "reference answer")
    score_col = _find_column(frame.columns, ("score_avg", "score", "grade"), "score")
    try:
        question_col = _find_column(
            frame.columns, ("question_id", "question", "question_text", "id"), "question/group"
        )
    except ValueError as exc:
        raise ValueError("Question-isolated evaluation requires a question ID or question text column") from exc
    frame = frame.dropna(subset=[answer_col, reference_col, score_col, question_col]).reset_index(drop=True)
    sample_ids = [f"mohler-{index:06d}" for index in range(len(frame))]
    group_ids = frame[question_col].astype(str).tolist()
    manifest = build_group_split_manifest(sample_ids, group_ids, seed=seed)
    split_by_id = {row["sample_id"]: row["split"] for row in manifest["samples"]}
    frame["_sample_id"] = sample_ids
    frame["_split"] = [split_by_id[value] for value in sample_ids]
    frame["_label"] = frame[score_col].astype(float).map(_score_to_label)
    # Student and reference text are joined identically for every model.
    frame["_text"] = (
        "student: " + frame[answer_col].astype(str) + " reference: " + frame[reference_col].astype(str)
    )
    train = frame[frame["_split"] == "train"]
    validation = frame[frame["_split"] == "validation"]
    test = frame[frame["_split"] == "test"]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("frozen group split produced an empty partition")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    candidates: list[tuple[str, Any, list[dict]]] = []
    for c_value in (0.1, 1.0, 10.0):
        candidates.append((
            "tfidf_logistic_regression",
            Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("model", LogisticRegression(C=c_value, max_iter=3000, class_weight="balanced", random_state=seed)),
            ]),
            [{"C": c_value}],
        ))
        candidates.append((
            "tfidf_linear_svm",
            Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("model", LinearSVC(C=c_value, class_weight="balanced", random_state=seed)),
            ]),
            [{"C": c_value}],
        ))

    selected: dict[str, dict] = {}
    for model_name, model, parameters in candidates:
        model.fit(train["_text"], train["_label"])
        validation_pred = model.predict(validation["_text"]).tolist()
        validation_metrics = _classification_metrics(validation["_label"].tolist(), validation_pred)
        record = {"model": model, "parameters": parameters[0], "validation_metrics": validation_metrics}
        incumbent = selected.get(model_name)
        if incumbent is None or (
            validation_metrics["macro_f1"], validation_metrics["accuracy"], -parameters[0]["C"]
        ) > (
            incumbent["validation_metrics"]["macro_f1"],
            incumbent["validation_metrics"]["accuracy"],
            -incumbent["parameters"]["C"],
        ):
            selected[model_name] = record

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_write(output / "split_manifest.json", manifest)
    predictions = []
    results = []
    for model_name in sorted(selected):
        record = selected[model_name]
        prediction = record["model"].predict(test["_text"]).tolist()
        metrics = _classification_metrics(test["_label"].tolist(), prediction)
        results.append({
            "model_name": model_name,
            "parameters": record["parameters"],
            "validation_metrics": record["validation_metrics"],
            "test_metrics": metrics,
        })
        predictions.extend(
            {"sample_id": sample_id, "group_id": group, "split": "test", "model_name": model_name,
             "y_true": truth, "y_pred": pred}
            for sample_id, group, truth, pred in zip(
                test["_sample_id"], test[question_col].astype(str), test["_label"], prediction
            )
        )
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    report = {
        "schema_version": TEXT_REPORT_VERSION,
        "protocol_version": SOTA_PROTOCOL_VERSION,
        "dataset_file": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "split_manifest_hash": manifest["manifest_hash"],
        "seed": seed,
        "task": "three-class short-answer performance classification",
        "grouping_column": question_col,
        "counts": manifest["counts"],
        "models": results,
        "selection_rule": "maximum validation macro-F1; accuracy then smaller C break ties",
        "test_used_for_selection": False,
        "elapsed_seconds": time.time() - started,
    }
    report["report_hash"] = canonical_json_hash(report)
    atomic_json_write(output / "text_benchmark_report.json", report)
    return report
