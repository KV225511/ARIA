import csv
import hashlib

from modules.module_07_rl import generation_preflight


def _write_clean_csv(path, count):
    columns = [
        "ID", "Resume_str", "Resume_prompt", "Category",
        "Resume_text_hash", "Prompt_truncated",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(count):
            text = f"Engineering resume {index} " + "technical systems " * 8
            normalized = " ".join(text.split()).casefold()
            writer.writerow({
                "ID": str(index),
                "Resume_str": text,
                "Resume_prompt": text,
                "Category": "ENGINEERING",
                "Resume_text_hash": hashlib.sha256(normalized.encode()).hexdigest(),
                "Prompt_truncated": "false",
            })


def test_preflight_requires_unique_readable_jds(tmp_path, monkeypatch):
    resume_csv = tmp_path / "clean.csv"
    _write_clean_csv(resume_csv, 3)
    jds = [tmp_path / f"jd-{index}.pdf" for index in range(3)]
    for path in jds:
        path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        generation_preflight,
        "get_valid_jd_documents",
        lambda: (jds[:2], {
            "pdf_files": 3,
            "filename_excluded_files": [],
            "unreadable_files": [],
            "duplicate_content_groups": [{
                "excluded_file": jds[1].name,
                "retained_file": jds[0].name,
                "content_hash": "duplicate",
            }],
            "unique_readable_content_hashes": 2,
        }),
    )

    report = generation_preflight.build_preflight_report(
        resume_csv=resume_csv,
        resume_categories=("ENGINEERING",),
        identity_components=(1, 1, 1),
    )

    assert report["resume_source"]["unique_content_hashes"] == 3
    assert report["job_descriptions"]["unique_readable_content_hashes"] == 2
    assert report["checks"]["enough_unique_readable_jds"] is False
    assert report["checks"]["no_duplicate_jd_content"] is False
    assert report["shortfall"]["unique_readable_jds"] == 1
    assert report["passes_preflight"] is False
