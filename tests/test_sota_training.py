import csv
import json

import pytest

from modules.module_14_evaluation.sota_training import (
    build_group_split_manifest,
    run_preflight,
    run_text_benchmark,
)


def test_group_split_is_deterministic_and_group_isolated():
    sample_ids = [f"s-{index}" for index in range(18)]
    groups = [f"q-{index // 2}" for index in range(18)]
    first = build_group_split_manifest(sample_ids, groups, seed=17)
    second = build_group_split_manifest(sample_ids, groups, seed=17)
    assert first == second
    seen = {}
    for row in first["samples"]:
        seen.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in seen.values())
    assert set(first["counts"]) == {"train", "validation", "test"}


def test_preflight_reports_missing_assets_without_scores(tmp_path):
    report = run_preflight(tmp_path, tmp_path / "preflight.json")
    assert all(item["status"] == "missing" for item in report["assets"])
    assert all("metric" not in item and "accuracy" not in item for item in report["assets"])
    assert "models" not in report
    assert all(item["status"] == "missing" for item in report["historical_results"])
    assert (tmp_path / "preflight.json").is_file()


def test_text_benchmark_freezes_split_and_writes_test_predictions(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    dataset = tmp_path / "mohler.csv"
    with dataset.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["question_id", "student_answer", "reference_answer", "score"],
        )
        writer.writeheader()
        for question in range(15):
            for label, score in (("low", 1.0), ("medium", 3.0), ("high", 5.0)):
                writer.writerow({
                    "question_id": f"question-{question}",
                    "student_answer": f"{label} {label} response for concept {question}",
                    "reference_answer": "complete reference concept explanation",
                    "score": score,
                })
    output = tmp_path / "run"
    report = run_text_benchmark(dataset, output, seed=23)
    manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    assert report["test_used_for_selection"] is False
    assert len(report["models"]) == 2
    assert (output / "test_predictions.csv").is_file()
    assignments = {}
    for row in manifest["samples"]:
        assignments.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(value) == 1 for value in assignments.values())
