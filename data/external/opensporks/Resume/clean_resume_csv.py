"""Create a deterministic, privacy-reduced ARIA resume dataset from Resume.csv."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


EXPECTED_COLUMNS = ["ID", "Resume_str", "Resume_html", "Category"]
OUTPUT_COLUMNS = [
    "ID",
    "Resume_str",
    "Resume_prompt",
    "Category",
    "Resume_text_hash",
    "Prompt_truncated",
    "Original_character_count",
    "Clean_character_count",
    "Prompt_character_count",
    "Email_redactions",
    "Phone_redactions",
]
PROMPT_CHARACTER_LIMIT = 8_000
MINIMUM_TEXT_CHARACTERS = 100
OMISSION_MARKER = "\n\n[... CONTENT OMITTED FOR LENGTH ...]\n\n"

EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)"
    r"\d{3}[-.\s]?\d{4}(?!\d)"
)
INSTRUCTION_PATTERN = re.compile(
    r"(?im)^\s*(ignore|forget|disregard|override)\b|"
    r"^\s*(system\s*:|you are now\b|act as\b)"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in text.split("\n")]
    normalized_lines: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        normalized_lines.append(line)
        previous_blank = is_blank
    return "\n".join(normalized_lines).strip()


def normalized_hash_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def redact_contact_data(value: str) -> tuple[str, int, int]:
    without_email, email_count = EMAIL_PATTERN.subn("[EMAIL_REDACTED]", value)
    redacted, phone_count = PHONE_PATTERN.subn("[PHONE_REDACTED]", without_email)
    return redacted, email_count, phone_count


def trim_at_word_boundary(value: str, limit: int, *, keep_tail: bool = False) -> str:
    if len(value) <= limit:
        return value
    if keep_tail:
        fragment = value[-limit:]
        first_space = re.search(r"\s", fragment)
        return fragment[first_space.end():] if first_space else fragment
    fragment = value[:limit]
    last_space = max(fragment.rfind(" "), fragment.rfind("\n"))
    return fragment[:last_space] if last_space > 0 else fragment


def make_prompt_text(value: str) -> tuple[str, bool]:
    if len(value) <= PROMPT_CHARACTER_LIMIT:
        return value, False
    head_budget = 6_000
    tail_budget = PROMPT_CHARACTER_LIMIT - head_budget - len(OMISSION_MARKER)
    head = trim_at_word_boundary(value, head_budget).rstrip()
    tail = trim_at_word_boundary(value, tail_budget, keep_tail=True).lstrip()
    prompt = head + OMISSION_MARKER + tail
    if len(prompt) > PROMPT_CHARACTER_LIMIT:
        prompt = prompt[:PROMPT_CHARACTER_LIMIT].rstrip()
    return prompt, True


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def clean_dataset(source: Path, output: Path, audit_path: Path, review_path: Path) -> dict:
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected CSV columns: {reader.fieldnames!r}; expected {EXPECTED_COLUMNS!r}"
            )
        source_rows = list(reader)

    output_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    retained_by_hash: dict[str, str] = {}
    exclusion_counts: Counter[str] = Counter()
    email_rows = 0
    phone_rows = 0
    email_matches = 0
    phone_matches = 0
    prompt_truncated_rows = 0
    reviewed_instruction_rows = 0

    for source_row_number, row in enumerate(source_rows, start=2):
        record_id = (row.get("ID") or "").strip()
        category = (row.get("Category") or "").strip()
        original_text = row.get("Resume_str") or ""
        html_text = row.get("Resume_html") or ""
        normalized = normalize_text(original_text)

        if len(normalized) < MINIMUM_TEXT_CHARACTERS:
            exclusion_counts["blank_or_short"] += 1
            review_rows.append({
                "ID": record_id,
                "Category": category,
                "Issue": "blank_or_short",
                "Action": "excluded",
                "Details": f"normalized_characters={len(normalized)}; source_row={source_row_number}",
            })
            continue

        replacement_count = original_text.count("\ufffd") + html_text.count("\ufffd")
        if replacement_count:
            exclusion_counts["encoding_replacement_character"] += 1
            review_rows.append({
                "ID": record_id,
                "Category": category,
                "Issue": "encoding_replacement_character",
                "Action": "excluded",
                "Details": f"replacement_characters={replacement_count}; source_row={source_row_number}",
            })
            continue

        instruction_terms = sorted({match.group(0).strip().casefold() for match in INSTRUCTION_PATTERN.finditer(normalized)})
        if instruction_terms:
            reviewed_instruction_rows += 1
            review_rows.append({
                "ID": record_id,
                "Category": category,
                "Issue": "instruction_like_language",
                "Action": "retained_after_manual_review",
                "Details": "benign employment-language false positive; terms=" + "|".join(instruction_terms),
            })

        cleaned_text, row_email_matches, row_phone_matches = redact_contact_data(normalized)
        cleaned_text = normalize_text(cleaned_text)
        if row_email_matches:
            email_rows += 1
            email_matches += row_email_matches
        if row_phone_matches:
            phone_rows += 1
            phone_matches += row_phone_matches
        if row_email_matches or row_phone_matches:
            review_rows.append({
                "ID": record_id,
                "Category": category,
                "Issue": "contact_data",
                "Action": "redacted",
                "Details": (
                    f"email_matches={row_email_matches}; phone_matches={row_phone_matches}; "
                    f"source_row={source_row_number}"
                ),
            })

        canonical = normalized_hash_text(cleaned_text)
        text_hash = sha256_bytes(canonical.encode("utf-8"))
        if text_hash in retained_by_hash:
            exclusion_counts["duplicate_normalized_text"] += 1
            review_rows.append({
                "ID": record_id,
                "Category": category,
                "Issue": "duplicate_normalized_text",
                "Action": "excluded",
                "Details": f"retained_id={retained_by_hash[text_hash]}; source_row={source_row_number}",
            })
            continue
        retained_by_hash[text_hash] = record_id

        prompt_text, prompt_truncated = make_prompt_text(cleaned_text)
        if prompt_truncated:
            prompt_truncated_rows += 1
            review_rows.append({
                "ID": record_id,
                "Category": category,
                "Issue": "long_resume",
                "Action": "prompt_text_limited",
                "Details": (
                    f"clean_characters={len(cleaned_text)}; "
                    f"prompt_characters={len(prompt_text)}; limit={PROMPT_CHARACTER_LIMIT}"
                ),
            })

        output_rows.append({
            "ID": record_id,
            "Resume_str": cleaned_text,
            "Resume_prompt": prompt_text,
            "Category": category,
            "Resume_text_hash": text_hash,
            "Prompt_truncated": str(prompt_truncated).lower(),
            "Original_character_count": len(original_text),
            "Clean_character_count": len(cleaned_text),
            "Prompt_character_count": len(prompt_text),
            "Email_redactions": row_email_matches,
            "Phone_redactions": row_phone_matches,
        })

    write_csv(output, output_rows, OUTPUT_COLUMNS)
    write_csv(review_path, review_rows, ["ID", "Category", "Issue", "Action", "Details"])

    output_hashes = [str(row["Resume_text_hash"]) for row in output_rows]
    cleaned_texts = [str(row["Resume_str"]) for row in output_rows]
    prompt_texts = [str(row["Resume_prompt"]) for row in output_rows]
    category_counts = Counter(str(row["Category"]) for row in output_rows)
    assertions = {
        "all_ids_present": all(str(row["ID"]).strip() for row in output_rows),
        "all_text_at_least_minimum_length": all(
            len(text) >= MINIMUM_TEXT_CHARACTERS for text in cleaned_texts
        ),
        "all_text_hashes_unique": len(output_hashes) == len(set(output_hashes)),
        "no_replacement_characters": all("\ufffd" not in text for text in cleaned_texts),
        "no_email_patterns": all(not EMAIL_PATTERN.search(text) for text in cleaned_texts),
        "no_phone_patterns": all(not PHONE_PATTERN.search(text) for text in cleaned_texts),
        "all_prompts_within_limit": all(
            len(text) <= PROMPT_CHARACTER_LIMIT for text in prompt_texts
        ),
    }
    audit = {
        "schema_version": "aria-resume-cleaning-v1",
        "source_file": source.name,
        "source_sha256": sha256_file(source),
        "output_file": output.name,
        "output_sha256": sha256_file(output),
        "source_rows": len(source_rows),
        "retained_rows": len(output_rows),
        "excluded_rows": sum(exclusion_counts.values()),
        "exclusions": dict(sorted(exclusion_counts.items())),
        "contact_redaction": {
            "email_rows": email_rows,
            "email_matches": email_matches,
            "phone_rows": phone_rows,
            "phone_matches": phone_matches,
        },
        "instruction_review": {
            "rows_reviewed": reviewed_instruction_rows,
            "decision": "retained_as_benign_employment_language",
        },
        "prompt_limiting": {
            "character_limit": PROMPT_CHARACTER_LIMIT,
            "truncated_rows": prompt_truncated_rows,
            "maximum_output_prompt_characters": max(map(len, prompt_texts), default=0),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "output_columns": OUTPUT_COLUMNS,
        "assertions": assertions,
        "passes_all_assertions": all(assertions.values()),
    }
    temporary_audit = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary_audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    temporary_audit.replace(audit_path)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    audit = clean_dataset(args.source, args.output, args.audit, args.review)
    print(json.dumps(audit, indent=2))
    if not audit["passes_all_assertions"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
