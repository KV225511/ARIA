import csv
import hashlib
import re
import os
import random
import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JDS_DIR = DATA_DIR / "jds"
DEFAULT_CLEANED_RESUME_CSV = (
    DATA_DIR / "external" / "opensporks" / "Resume" / "Resume.cleaned.csv"
)
DEFAULT_RESUME_CATEGORIES: tuple[str, ...] = (
    "INFORMATION-TECHNOLOGY",
    "ENGINEERING",
)
RESUME_PROMPT_CHARACTER_LIMIT = 8_000
END_OF_DOCUMENT_MARKER = "\n--- END OF DOCUMENT ---"
PROMPT_OMISSION_MARKER = "\n\n[... CONTENT OMITTED FOR LENGTH ...]\n\n"

_CLEANED_CSV_COLUMNS = {
    "ID",
    "Resume_str",
    "Resume_prompt",
    "Category",
    "Resume_text_hash",
    "Prompt_truncated",
}
_EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)"
    r"\d{3}[-.\s]?\d{4}(?!\d)"
)


@dataclass(frozen=True)
class ResumeDocument:
    """Validated resume metadata used by generation planning and prompting."""

    source_id: str
    prompt_text: str
    content_hash: str
    category: str
    source_type: str
    source_file_hash: str

    @property
    def name(self) -> str:
        """Compatibility identifier used by the existing pair planner."""
        return self.source_id


@dataclass(frozen=True)
class LoadedDocumentPair:
    resume_text: str
    jd_text: str
    resume_id: str
    jd_id: str
    resume_content_hash: str
    jd_content_hash: str
    resume_source_type: str
    resume_category: str | None
    resume_source_file_hash: str | None
    resume_prompt_hash: str


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_hash_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _without_document_marker(value: str) -> str:
    return value.removesuffix(END_OF_DOCUMENT_MARKER).strip()


def _trim_at_word_boundary(value: str, limit: int, *, keep_tail: bool = False) -> str:
    if len(value) <= limit:
        return value
    if keep_tail:
        fragment = value[-limit:]
        first_space = re.search(r"\s", fragment)
        return fragment[first_space.end():] if first_space else fragment
    fragment = value[:limit]
    last_space = max(fragment.rfind(" "), fragment.rfind("\n"))
    return fragment[:last_space] if last_space > 0 else fragment


def _expected_resume_prompt(value: str) -> tuple[str, bool]:
    if len(value) <= RESUME_PROMPT_CHARACTER_LIMIT:
        return value, False
    head_budget = 6_000
    tail_budget = (
        RESUME_PROMPT_CHARACTER_LIMIT - head_budget - len(PROMPT_OMISSION_MARKER)
    )
    head = _trim_at_word_boundary(value, head_budget).rstrip()
    tail = _trim_at_word_boundary(value, tail_budget, keep_tail=True).lstrip()
    prompt = head + PROMPT_OMISSION_MARKER + tail
    return prompt[:RESUME_PROMPT_CHARACTER_LIMIT].rstrip(), True

# Prompt injection patterns to strip from PDFs (Security fix #6)
_INJECTION_PATTERNS = re.compile(
    r'(?im)^\s*(?:ignore|forget|disregard|override|as an ai|new instruction|system:|you are now|act as).{0,120}$'
)

def sanitize_pdf_text(text: str) -> str:
    """Strip potential prompt-injection lines from extracted PDF text."""
    cleaned = _INJECTION_PATTERNS.sub("", text)
    # Add a hard delimiter so LLMs cannot confuse PDF content with system instructions
    return cleaned.strip() + "\n--- END OF DOCUMENT ---"

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file, falling back to pypdf if pdfplumber fails."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return sanitize_pdf_text(text.strip())
    except Exception as e:
        logger.warning(f"pdfplumber failed for {pdf_path}: {e}. Trying fallback.")
        
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return sanitize_pdf_text(text.strip())
    except Exception as e:
        logger.error(f"Fallback PDF extraction failed for {pdf_path}: {e}")
        return ""

def get_all_pdfs(directory: Path) -> list[Path]:
    """Returns a list of all PDF paths in a given directory."""
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
    return list(directory.rglob("*.pdf"))


@lru_cache(maxsize=8)
def _load_clean_resume_csv_cached(
    resolved_path: str,
    file_size: int,
    modified_ns: int,
) -> tuple[ResumeDocument, ...]:
    """Load and validate one immutable snapshot of the cleaned resume CSV."""
    del file_size, modified_ns  # Values intentionally participate in the cache key.
    path = Path(resolved_path)
    source_file_hash = _file_sha256(path)
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _CLEANED_CSV_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Cleaned resume CSV is missing required columns: {sorted(missing)}"
            )
        rows = list(reader)

    documents = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        raw_id = (row.get("ID") or "").strip()
        category = (row.get("Category") or "").strip()
        full_text = (row.get("Resume_str") or "").strip()
        prompt_text = (row.get("Resume_prompt") or "").strip()
        declared_hash = (row.get("Resume_text_hash") or "").strip().lower()
        source_id = f"opensporks:{raw_id}"

        if not raw_id:
            raise ValueError(f"Cleaned resume CSV row {row_number} has no ID")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", raw_id):
            raise ValueError(f"Cleaned resume CSV row {row_number} has an unsafe ID")
        if source_id in seen_ids:
            raise ValueError(f"Duplicate resume ID in cleaned CSV: {source_id}")
        if len(full_text) < 100:
            raise ValueError(f"Resume {source_id} contains fewer than 100 characters")
        if not prompt_text or len(prompt_text) > RESUME_PROMPT_CHARACTER_LIMIT:
            raise ValueError(
                f"Resume {source_id} prompt length must be 1.."
                f"{RESUME_PROMPT_CHARACTER_LIMIT} characters"
            )
        if "\ufffd" in full_text or "\ufffd" in prompt_text:
            raise ValueError(f"Resume {source_id} contains a replacement character")
        if _EMAIL_PATTERN.search(full_text) or _EMAIL_PATTERN.search(prompt_text):
            raise ValueError(f"Resume {source_id} contains an unredacted email address")
        if _PHONE_PATTERN.search(full_text) or _PHONE_PATTERN.search(prompt_text):
            raise ValueError(f"Resume {source_id} contains an unredacted phone number")

        calculated_hash = _sha256_text(_normalized_hash_text(full_text))
        if declared_hash != calculated_hash:
            raise ValueError(
                f"Resume {source_id} content hash does not match Resume_text_hash"
            )
        if calculated_hash in seen_hashes:
            raise ValueError(
                f"Duplicate normalized resume content in cleaned CSV: {source_id}"
            )
        expected_prompt, expected_truncated = _expected_resume_prompt(full_text)
        declared_truncated = (row.get("Prompt_truncated") or "").strip().casefold()
        if declared_truncated not in {"true", "false"}:
            raise ValueError(f"Resume {source_id} has invalid Prompt_truncated metadata")
        if (declared_truncated == "true") != expected_truncated:
            raise ValueError(f"Resume {source_id} Prompt_truncated metadata is inconsistent")
        if prompt_text != expected_prompt:
            raise ValueError(f"Resume {source_id} Resume_prompt is not deterministically derived")

        seen_ids.add(source_id)
        seen_hashes.add(calculated_hash)
        documents.append(ResumeDocument(
            source_id=source_id,
            prompt_text=prompt_text + END_OF_DOCUMENT_MARKER,
            content_hash=calculated_hash,
            category=category,
            source_type="opensporks_csv",
            source_file_hash=source_file_hash,
        ))

    if not documents:
        raise ValueError(f"Cleaned resume CSV contains no usable records: {path}")
    return tuple(documents)


def load_clean_resume_csv(
    csv_path: str | Path = DEFAULT_CLEANED_RESUME_CSV,
    categories: tuple[str, ...] | list[str] | None = DEFAULT_RESUME_CATEGORIES,
) -> list[ResumeDocument]:
    """Return validated, optionally category-filtered cleaned CSV resumes."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned resume CSV not found: {path}")
    stat = path.stat()
    records = _load_clean_resume_csv_cached(
        str(path.resolve()), stat.st_size, stat.st_mtime_ns
    )
    if categories is None:
        selected = list(records)
    else:
        category_set = {str(value).strip() for value in categories if str(value).strip()}
        if not category_set:
            raise ValueError("At least one resume category must be selected")
        selected = [record for record in records if record.category in category_set]
    if not selected:
        raise ValueError(
            f"No cleaned resumes matched categories: {sorted(category_set)}"
        )
    return selected


def get_resume_documents(
    resume_source: str = "pdf",
    resume_csv_path: str | Path = DEFAULT_CLEANED_RESUME_CSV,
    resume_categories: tuple[str, ...] | list[str] | None = DEFAULT_RESUME_CATEGORIES,
) -> list[Path] | list[ResumeDocument]:
    """Return planning handles for the selected resume source."""
    if resume_source == "csv":
        return load_clean_resume_csv(resume_csv_path, resume_categories)
    if resume_source == "pdf":
        return [path for path in get_all_pdfs(RESUMES_DIR) if is_valid_resume(path)]
    raise ValueError(f"Unknown resume source: {resume_source}")


def get_resume_source_manifest(
    resume_source: str,
    resume_csv_path: str | Path = DEFAULT_CLEANED_RESUME_CSV,
    resume_categories: tuple[str, ...] | list[str] | None = DEFAULT_RESUME_CATEGORIES,
) -> dict:
    """Return immutable resume-source provenance for generation manifests."""
    documents = get_resume_documents(
        resume_source, resume_csv_path, resume_categories
    )
    if resume_source == "csv":
        path = Path(resume_csv_path).resolve()
        categories = sorted({document.category for document in documents})
        return {
            "source_type": "opensporks_csv",
            "source_file": _manifest_path(path),
            "source_file_hash": _file_sha256(path),
            "selected_categories": categories,
            "selected_resume_count": len(documents),
            "unique_content_hashes": len({document.content_hash for document in documents}),
        }
    return {
        "source_type": "pdf_directory",
        "source_directory": str(RESUMES_DIR.resolve()),
        "selected_categories": None,
        "selected_resume_count": len(documents),
        "unique_content_hashes": None,
    }


# --- Resume Blocklist ---
# Filename prefixes known to be image-only scans (no extractable text).
_BLOCKED_RESUME_PREFIXES: tuple[str, ...] = ("Image_", "image_", "IMG_", "img_", "Scan_", "scan_")

# --- JD Exclusion List ---
# JD filenames that are NOT relevant to BTech/Engineering technical interviews.
# These would generate irrelevant Q&A pairs that pollute the RL dataset.
_EXCLUDED_JD_NAMES: frozenset[str] = frozenset({
    "AMVI_Foods_Food_Technologist_JD.pdf",          # Food Technology
    "JD - Technical Sales and Application Specialist - Fresher - Biotechnology (1).pdf",  # Biotech Sales
    "JD - Technical Sales and Service Engineer - Freshers(0-1 year) Biomedical.pdf",      # Biomedical Sales
    "Job Description_Biomedical - eInfochips (An Arrow Company) 2027.pdf",               # Biomedical
    "Internship (Mechanical).pdf",                  # Mechanical only
    "Mechanical Intern \ufffd Design (2D_3D).pdf",   # Mechanical Design
    "JD - Assistant Manager (NEEV) (1).pdf",        # Non-technical management
    "Job Description BDA (2).pdf",                  # Business Development
    "Linux Driver Intern JD.pdf",                   # Image-only / unreadable text layer
})


def is_valid_resume(path: Path) -> bool:
    """Returns False for image-only or blocked resume filenames."""
    return not path.name.startswith(_BLOCKED_RESUME_PREFIXES)


def is_valid_jd(path: Path) -> bool:
    """Returns False for excluded non-BTech JDs."""
    return path.name not in _EXCLUDED_JD_NAMES


def get_valid_jd_documents(
    directory: Path = JDS_DIR,
) -> tuple[list[Path], dict]:
    """Return readable, content-unique JDs and a non-sensitive source report."""
    all_paths = get_all_pdfs(directory)
    name_valid = [path for path in all_paths if is_valid_jd(path)]
    accepted: list[Path] = []
    unreadable: list[str] = []
    duplicate_groups: list[dict[str, str]] = []
    first_file_by_hash: dict[str, str] = {}
    for path in name_valid:
        text = extract_text_from_pdf(str(path))
        body = _without_document_marker(text)
        if len(body) < 100:
            unreadable.append(path.name)
            continue
        content_hash = _sha256_text(_normalized_hash_text(body))
        retained = first_file_by_hash.get(content_hash)
        if retained is not None:
            duplicate_groups.append({
                "excluded_file": path.name,
                "retained_file": retained,
                "content_hash": content_hash,
            })
            continue
        first_file_by_hash[content_hash] = path.name
        accepted.append(path)
    return accepted, {
        "pdf_files": len(all_paths),
        "filename_excluded_files": sorted(
            path.name for path in all_paths if not is_valid_jd(path)
        ),
        "unreadable_files": sorted(unreadable),
        "duplicate_content_groups": duplicate_groups,
        "unique_readable_content_hashes": len(accepted),
    }


# --- Domain Keyword Sets ---
# Maps broad domain names to keywords to detect in PDF text.
# Used to filter resumes so they align with the JD's field.
_DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "software": {
        "python", "java", "javascript", "react", "node", "sql", "api", "backend",
        "frontend", "fullstack", "docker", "kubernetes", "git", "software", "developer",
        "engineer", "programming", "database", "linux", "algorithm", "data structure",
        "microservice", "rest", "graphql", "aws", "azure", "gcp", "cloud", "devops",
        "machine learning", "deep learning", "tensorflow", "pytorch", "nlp",
        "computer science", "cs", "btech", "b.tech", "information technology",
    },
    "hardware_embedded": {
        "embedded", "fpga", "vhdl", "verilog", "microcontroller", "rtos", "linux kernel",
        "driver", "firmware", "arm", "stm32", "pcb", "schematic", "circuit",
        "electronics", "electrical", "vlsi", "signal processing", "iot",
    },
    "mechanical": {
        "mechanical", "cad", "solidworks", "ansys", "catia", "autocad", "fea",
        "thermodynamics", "fluid", "manufacturing", "design", "automobile", "automotive",
        "battery", "powertrain", "motor", "dvp", "testing", "validation",
    },
    "biomedical": {
        "biomedical", "biology", "biotechnology", "pharmaceutical", "clinical",
        "medical", "health", "hospital", "lab", "specimen", "bioinformatics",
    },
    "data_analytics": {
        "data science", "analytics", "tableau", "power bi", "excel", "statistics",
        "r programming", "pandas", "numpy", "visualization", "etl", "pipeline",
        "data engineer", "snowflake", "redshift", "bigquery",
    },
    "food_tech": {
        "food", "nutrition", "fssai", "haccp", "food technology", "culinary",
        "quality control", "food safety",
    },
}


def _detect_domain(text: str) -> set[str]:
    """
    Detects which domain(s) a PDF text belongs to based on keyword frequency.
    Returns a set of matching domain names.
    """
    text_lower = text.lower()
    matched = set()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        # Require at least 3 keyword hits to claim a domain match
        if hits >= 3:
            matched.add(domain)
    return matched


def get_domain_matched_pairs(jd_path: Path, resume_paths: list[Path],
                              max_sample: int = 500) -> list[Path]:
    """
    Given a JD, returns a filtered list of resumes that match the JD's domain.
    Falls back to all resumes if fewer than 5 domain matches are found.
    Skips image-only or unreadable resumes (extracted text < 100 chars).

    Args:
        jd_path: Path to the JD PDF file.
        resume_paths: Full list of resume PDF paths.
        max_sample: Maximum number of resumes to scan per JD (random subset for speed).

    Returns:
        Filtered list of resume Paths that share domain keywords with the JD.
    """
    jd_text = extract_text_from_pdf(str(jd_path))
    jd_domains = _detect_domain(jd_text)

    if not jd_domains:
        logger.debug(f"No domain detected for JD '{jd_path.name}'. Using full resume pool.")
        return resume_paths

    # Scan a random subset for speed (8,340 resumes is too many to scan fully each time)
    sample_paths = random.sample(resume_paths, min(max_sample, len(resume_paths)))

    matched = []
    for rp in sample_paths:
        try:
            rtext = extract_text_from_pdf(str(rp))
            # Skip image-only or unreadable PDFs (less than 100 chars of real text)
            if len(rtext.strip()) < 100:
                continue
            resume_domains = _detect_domain(rtext)
            if jd_domains & resume_domains:  # at least one domain in common
                matched.append(rp)
        except Exception:
            continue

    if len(matched) < 5:
        logger.warning(
            f"Only {len(matched)} domain-matched resumes found for '{jd_path.name}'. "
            f"Falling back to readable resume pool."
        )
        # Fall back to readable resumes only (still better than image-only)
        readable = [
            rp for rp in random.sample(resume_paths, min(500, len(resume_paths)))
            if len(extract_text_from_pdf(str(rp)).strip()) >= 100
        ]
        return readable if readable else resume_paths

    logger.info(f"JD '{jd_path.name}' domains={jd_domains} -> {len(matched)}/{max_sample} resumes matched.")
    return matched


def _find_pdf(directory: Path, filename: str) -> Path | None:
    direct = directory / filename
    if direct.exists():
        return direct
    for path in directory.rglob("*.pdf"):
        if path.name == filename:
            return path
    return None


def _csv_domain_matches(
    jd_text: str,
    resumes: list[ResumeDocument],
) -> list[ResumeDocument]:
    jd_domains = _detect_domain(jd_text)
    if not jd_domains:
        return resumes
    matched = [
        resume for resume in resumes
        if jd_domains & _detect_domain(_without_document_marker(resume.prompt_text))
    ]
    if len(matched) < 5:
        logger.warning(
            "Only %s cleaned CSV resumes matched JD domains %s; using the "
            "validated category-filtered resume pool.",
            len(matched), sorted(jd_domains),
        )
        return resumes
    return matched


def load_specific_pair(
    resume_identifier: str,
    jd_filename: str,
    *,
    resume_source: str = "pdf",
    resume_csv_path: str | Path = DEFAULT_CLEANED_RESUME_CSV,
    resume_categories: tuple[str, ...] | list[str] | None = DEFAULT_RESUME_CATEGORIES,
) -> LoadedDocumentPair:
    """Load one resume/JD pair with stable identity and source provenance."""
    jd_path = _find_pdf(JDS_DIR, jd_filename)
    if not jd_path:
        raise FileNotFoundError(f"JD not found: {jd_filename}")
    jd_text = extract_text_from_pdf(str(jd_path))
    if len(_without_document_marker(jd_text)) < 100:
        raise ValueError(f"JD contains fewer than 100 extractable characters: {jd_filename}")

    if resume_source == "csv":
        records = load_clean_resume_csv(resume_csv_path, resume_categories)
        record_by_id = {record.source_id: record for record in records}
        record = record_by_id.get(resume_identifier)
        if record is None:
            raise FileNotFoundError(
                f"Resume ID not found in selected cleaned CSV pool: {resume_identifier}"
            )
        return LoadedDocumentPair(
            resume_text=record.prompt_text,
            jd_text=jd_text,
            resume_id=record.source_id,
            jd_id=jd_path.name,
            resume_content_hash=record.content_hash,
            jd_content_hash=_sha256_text(
                _normalized_hash_text(_without_document_marker(jd_text))
            ),
            resume_source_type=record.source_type,
            resume_category=record.category,
            resume_source_file_hash=record.source_file_hash,
            resume_prompt_hash=_sha256_text(record.prompt_text),
        )

    if resume_source != "pdf":
        raise ValueError(f"Unknown resume source: {resume_source}")
    resume_path = _find_pdf(RESUMES_DIR, resume_identifier)
    if not resume_path:
        raise FileNotFoundError(f"Resume not found: {resume_identifier}")
    resume_text = extract_text_from_pdf(str(resume_path))
    if len(_without_document_marker(resume_text)) < 100:
        raise ValueError(
            f"Resume contains fewer than 100 extractable characters: {resume_identifier}"
        )
    return LoadedDocumentPair(
        resume_text=resume_text,
        jd_text=jd_text,
        resume_id=resume_path.name,
        jd_id=jd_path.name,
        resume_content_hash=_sha256_text(resume_text),
        jd_content_hash=_sha256_text(
            _normalized_hash_text(_without_document_marker(jd_text))
        ),
        resume_source_type="pdf_directory",
        resume_category=None,
        resume_source_file_hash=_file_sha256(resume_path),
        resume_prompt_hash=_sha256_text(resume_text),
    )


def load_random_pair(
    *,
    resume_source: str = "pdf",
    resume_csv_path: str | Path = DEFAULT_CLEANED_RESUME_CSV,
    resume_categories: tuple[str, ...] | list[str] | None = DEFAULT_RESUME_CATEGORIES,
    domain_match: bool = True,
) -> LoadedDocumentPair:
    """Choose and load one random pair from the configured resume source."""
    jds = [path for path in get_all_pdfs(JDS_DIR) if is_valid_jd(path)]
    if not jds:
        raise ValueError("Missing valid PDFs in data/jds.")
    jd_path = random.choice(jds)
    jd_text = extract_text_from_pdf(str(jd_path))

    if resume_source == "csv":
        resumes = load_clean_resume_csv(resume_csv_path, resume_categories)
        candidates = _csv_domain_matches(jd_text, resumes) if domain_match else resumes
        selected = random.choice(candidates)
        return LoadedDocumentPair(
            resume_text=selected.prompt_text,
            jd_text=jd_text,
            resume_id=selected.source_id,
            jd_id=jd_path.name,
            resume_content_hash=selected.content_hash,
            jd_content_hash=_sha256_text(
                _normalized_hash_text(_without_document_marker(jd_text))
            ),
            resume_source_type=selected.source_type,
            resume_category=selected.category,
            resume_source_file_hash=selected.source_file_hash,
            resume_prompt_hash=_sha256_text(selected.prompt_text),
        )

    if resume_source != "pdf":
        raise ValueError(f"Unknown resume source: {resume_source}")
    resumes = [path for path in get_all_pdfs(RESUMES_DIR) if is_valid_resume(path)]
    if not resumes:
        raise ValueError("Missing valid PDFs in data/resumes.")
    candidates = get_domain_matched_pairs(jd_path, resumes) if domain_match else resumes
    return load_specific_pair(
        random.choice(candidates).name,
        jd_path.name,
        resume_source="pdf",
    )

def get_random_pair(domain_match: bool = True) -> tuple[str, str, str, str]:
    """
    Randomly selects a valid JD, then picks a domain-matched resume for that JD.
    Excludes image-only resumes and non-BTech JDs.
    Falls back to readable pairing if domain matching yields fewer than 5 results.

    Args:
        domain_match: If True (default), filters resumes by JD domain keywords.
                      Set to False for legacy purely-random behaviour.

    Returns: (resume_text, jd_text, resume_filename, jd_filename)
    """
    all_resumes = get_all_pdfs(RESUMES_DIR)
    all_jds = get_all_pdfs(JDS_DIR)

    # Apply blocklists
    resumes = [r for r in all_resumes if is_valid_resume(r)]
    jds = [j for j in all_jds if is_valid_jd(j)]

    if not resumes or not jds:
        raise ValueError("Missing valid PDFs in data/resumes or data/jds directories.")

    jd_path = random.choice(jds)

    if domain_match:
        matched_resumes = get_domain_matched_pairs(jd_path, resumes)
        resume_path = random.choice(matched_resumes)
    else:
        resume_path = random.choice(resumes)
    
    resume_text = extract_text_from_pdf(str(resume_path))
    jd_text = extract_text_from_pdf(str(jd_path))
    
    return resume_text, jd_text, resume_path.name, jd_path.name

def get_specific_pair(resume_filename: str, jd_filename: str) -> tuple[str, str, str, str]:
    """
    Loads a specific resume and JD by filename.
    """
    pair = load_specific_pair(resume_filename, jd_filename, resume_source="pdf")
    return pair.resume_text, pair.jd_text, pair.resume_id, pair.jd_id

if __name__ == "__main__":
    # Test data loader
    print("Testing data loader...")
    r_text, j_text, r_name, j_name = get_random_pair()
    print(f"Loaded Resume: {r_name} ({len(r_text)} chars)")
    print(f"Loaded JD: {j_name} ({len(j_text)} chars)")
