import csv
import hashlib
from pathlib import Path

import pytest

from modules.module_07_rl import data_loader


FIELDNAMES = [
    "ID",
    "Resume_str",
    "Resume_prompt",
    "Category",
    "Resume_text_hash",
    "Prompt_truncated",
]


def _hash(text: str) -> str:
    normalized = " ".join(text.split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _row(record_id: str, text: str, category: str = "ENGINEERING", **updates):
    row = {
        "ID": record_id,
        "Resume_str": text,
        "Resume_prompt": text[:8_000],
        "Category": category,
        "Resume_text_hash": _hash(text),
        "Prompt_truncated": str(len(text) > 8_000).lower(),
    }
    row.update(updates)
    return row


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_cleaned_csv_loader_filters_categories_and_preserves_full_identity_hash(tmp_path):
    engineering = "Python systems engineering and API design. " * 8
    finance = "Financial planning and accounting controls. " * 8
    source = tmp_path / "resumes.csv"
    _write_csv(source, [
        _row("101", engineering),
        _row("102", finance, category="FINANCE"),
    ])

    records = data_loader.load_clean_resume_csv(source, ("ENGINEERING",))

    assert len(records) == 1
    assert records[0].source_id == "opensporks:101"
    assert records[0].name == "opensporks:101"
    assert records[0].category == "ENGINEERING"
    assert records[0].content_hash == _hash(engineering)
    assert records[0].prompt_text.endswith(data_loader.END_OF_DOCUMENT_MARKER)
    assert records[0].source_file_hash == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("text", "updates", "message"),
    [
        ("Engineering contact person@example.com " * 5, {}, "unredacted email"),
        ("Engineering contact 212-555-0100 " * 5, {}, "unredacted phone"),
        ("Engineering replacement \ufffd character " * 5, {}, "replacement character"),
        ("Engineering systems " * 8, {"Resume_prompt": "x" * 8_001}, "prompt length"),
        ("Engineering systems " * 8, {"Resume_text_hash": "0" * 64}, "content hash"),
        ("Engineering systems " * 8, {"Resume_prompt": "different safe text"}, "deterministically derived"),
    ],
)
def test_cleaned_csv_loader_rejects_tampered_or_unsafe_rows(
    tmp_path, text, updates, message
):
    source = tmp_path / "resumes.csv"
    _write_csv(source, [_row("101", text, **updates)])

    with pytest.raises(ValueError, match=message):
        data_loader.load_clean_resume_csv(source, ("ENGINEERING",))


def test_cleaned_csv_loader_rejects_duplicate_normalized_content(tmp_path):
    text = "Python systems engineering and API design. " * 8
    source = tmp_path / "resumes.csv"
    _write_csv(source, [_row("101", text), _row("102", text.upper())])

    with pytest.raises(ValueError, match="Duplicate normalized resume content"):
        data_loader.load_clean_resume_csv(source, ("ENGINEERING",))


def test_specific_csv_pair_uses_prompt_text_but_full_resume_identity_hash(
    tmp_path, monkeypatch
):
    full_text = "Python backend engineering and database design. " * 200
    prompt_text, _ = data_loader._expected_resume_prompt(full_text)
    source = tmp_path / "resumes.csv"
    _write_csv(source, [_row("101", full_text, Resume_prompt=prompt_text)])
    jd_dir = tmp_path / "jds"
    jd_dir.mkdir()
    jd_path = jd_dir / "backend.pdf"
    jd_path.write_bytes(b"placeholder")
    jd_text = "Backend developer role using Python, APIs, SQL, and cloud systems. " * 4
    jd_text += data_loader.END_OF_DOCUMENT_MARKER
    monkeypatch.setattr(data_loader, "JDS_DIR", jd_dir)
    monkeypatch.setattr(data_loader, "extract_text_from_pdf", lambda _: jd_text)

    pair = data_loader.load_specific_pair(
        "opensporks:101",
        jd_path.name,
        resume_source="csv",
        resume_csv_path=source,
        resume_categories=("ENGINEERING",),
    )

    assert pair.resume_text == prompt_text + data_loader.END_OF_DOCUMENT_MARKER
    assert pair.resume_content_hash == _hash(full_text)
    assert pair.resume_prompt_hash == hashlib.sha256(
        pair.resume_text.encode("utf-8")
    ).hexdigest()
    assert pair.resume_source_type == "opensporks_csv"
    assert pair.resume_category == "ENGINEERING"
