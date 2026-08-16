import re
import os
import random
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JDS_DIR = DATA_DIR / "jds"

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
})


def is_valid_resume(path: Path) -> bool:
    """Returns False for image-only or blocked resume filenames."""
    return not path.name.startswith(_BLOCKED_RESUME_PREFIXES)


def is_valid_jd(path: Path) -> bool:
    """Returns False for excluded non-BTech JDs."""
    return path.name not in _EXCLUDED_JD_NAMES


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
    def find_file(directory: Path, filename: str) -> Path | None:
        path = directory / filename
        if path.exists():
            return path
        for p in directory.rglob("*.pdf"):
            if p.name == filename:
                return p
        return None

    resume_path = find_file(RESUMES_DIR, resume_filename)
    jd_path = find_file(JDS_DIR, jd_filename)
    
    if not resume_path:
        raise FileNotFoundError(f"Resume not found: {resume_filename}")
    if not jd_path:
        raise FileNotFoundError(f"JD not found: {jd_filename}")
        
    resume_text = extract_text_from_pdf(str(resume_path))
    jd_text = extract_text_from_pdf(str(jd_path))
    
    return resume_text, jd_text, resume_path.name, jd_path.name

if __name__ == "__main__":
    # Test data loader
    print("Testing data loader...")
    r_text, j_text, r_name, j_name = get_random_pair()
    print(f"Loaded Resume: {r_name} ({len(r_text)} chars)")
    print(f"Loaded JD: {j_name} ({len(j_text)} chars)")
