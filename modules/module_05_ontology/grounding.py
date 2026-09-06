"""Evidence-backed role profiles and question-grounding checks for ARIA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import re
import unicodedata


ROLE_PROFILE_SCHEMA_VERSION = "aria-role-profile-v1"
GROUNDING_SCHEMA_VERSION = "aria-question-grounding-v1"
GROUNDING_POLICY_VERSION = "aria-grounding-policy-v2"
MIN_GROUNDED_SKILLS = 5


def normalize_document_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t ]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def normalized_text_hash(value: str) -> str:
    normalized = normalize_document_text(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceSpan:
    source: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    canonical_name: str
    definition: str
    domain: str
    aliases: tuple[str, ...]
    prerequisites: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundedSkill:
    skill_id: str
    canonical_name: str
    definition: str
    domain: str
    aliases: tuple[str, ...]
    priority: str
    support_type: str
    jd_evidence: tuple[EvidenceSpan, ...]
    resume_evidence: tuple[EvidenceSpan, ...]
    prerequisite_ids: tuple[str, ...]

    @property
    def selection_priority(self) -> int:
        return {
            "required": 0,
            "preferred": 1,
            "unspecified": 2,
            "prerequisite": 3,
        }.get(self.priority, 4)


@dataclass(frozen=True)
class RoleProfile:
    schema_version: str
    normalized_document_hash: str
    role_title: str
    role_domain: str
    skills: tuple[GroundedSkill, ...]
    acronym_resolutions: tuple[dict, ...]

    @property
    def profile_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)

    def skill(self, skill_id_or_name: str) -> GroundedSkill | None:
        key = str(skill_id_or_name).casefold()
        return next(
            (
                skill for skill in self.skills
                if skill.skill_id.casefold() == key
                or skill.canonical_name.casefold() == key
            ),
            None,
        )


_CATALOG: tuple[SkillDefinition, ...] = (
    SkillDefinition("electrical-circuits", "Electrical Circuits", "Circuit analysis using voltage, current, resistance, and network laws.", "electrical_electronics", ("electrical circuit", "circuits", "ohm's law", "kirchhoff")),
    SkillDefinition("electronic-components", "Electronic Components", "Selection and behaviour of resistors, capacitors, diodes, and semiconductor components.", "electrical_electronics", ("electronic components", "component selection", "resistors", "capacitors", "diodes"), ("electrical-circuits",)),
    SkillDefinition("pcb-design", "PCB Design", "Schematic capture, board layout, signal integrity, and design validation.", "electrical_electronics", ("pcb", "schematic", "board layout", "printed circuit"), ("electrical-circuits", "electronic-components")),
    SkillDefinition("emi-emc", "EMI/EMC Testing", "Electromagnetic compatibility design and pre-compliance testing.", "electrical_electronics", ("emi", "emc", "electromagnetic compatibility"), ("electrical-circuits",)),
    SkillDefinition("power-electronics", "Power Electronics", "Design and validation of switched power-conversion systems.", "electrical_electronics", ("power electronics", "power electronic", "inverter", "converter"), ("electrical-circuits", "electronic-components")),
    SkillDefinition("test-validation", "Test and Validation", "Planning, executing, and documenting engineering verification and validation.", "manufacturing_quality", ("test plan", "testing and validation", "test and validation", "validation activities", "dvp", "dvt", "pvt")),
    SkillDefinition("root-cause-analysis", "Root Cause Analysis", "Systematic isolation and verification of failure causes.", "manufacturing_quality", ("root cause analysis", "root-cause analysis", "failure analysis", "troubleshoot"), ("test-validation",)),
    SkillDefinition("quality-engineering", "Quality Engineering", "Quality planning, process control, inspection, and corrective action.", "manufacturing_quality", ("quality engineering", "quality control", "quality assurance", "quality standards", "inspection")),
    SkillDefinition("doe", "Design of Experiments", "Controlled multivariable experimentation and statistical interpretation.", "manufacturing_quality", ("design of experiments", "doe"), ("test-validation",)),
    SkillDefinition("battery-testing", "Battery Testing", "Electrical, thermal, mechanical, health, and performance validation of batteries.", "mechanical_automotive", ("battery", "dcir", "hppc", "state of charge", "soc", "battery pack"), ("test-validation",)),
    SkillDefinition("battery-management-system", "Battery Management System", "Monitoring, estimation, protection, and control of battery packs.", "mechanical_automotive", ("bms", "battery management system"), ("battery-testing", "electrical-circuits")),
    SkillDefinition("homologation", "Homologation Standards", "Regulatory and type-approval requirements for vehicles and components.", "mechanical_automotive", ("homologation", "type approval", "regulatory compliance"), ("test-validation",)),
    SkillDefinition("mechanical-testing", "Mechanical Testing", "Vibration, shock, fatigue, interference, and mechanical durability testing.", "mechanical_automotive", ("vibration", "shock", "mechanical testing", "torque", "interference checks"), ("test-validation",)),
    SkillDefinition("thermal-testing", "Thermal Testing", "Thermal cycling, measurement, and performance validation across temperature conditions.", "mechanical_automotive", ("thermal cycling", "thermal testing", "temperature testing"), ("test-validation",)),
    SkillDefinition("engineering-drawings", "Engineering Drawings", "Creation and interpretation of dimensioned technical drawings and assembly documentation.", "mechanical_automotive", ("engineering drawings", "2d drawings", "technical drawings")),
    SkillDefinition("gd-and-t", "Geometric Dimensioning and Tolerancing", "Specification and interpretation of geometric tolerances using GD&T.", "mechanical_automotive", ("gd&t", "geometric dimensioning", "geometric tolerancing"), ("engineering-drawings",)),
    SkillDefinition("mechanical-design", "Mechanical Design", "Design of mechanical parts, assemblies, interfaces, and production-ready models.", "mechanical_automotive", ("mechanical design", "3d models", "assembly design", "solidworks", "catia"), ("engineering-drawings",)),
    SkillDefinition("product-development", "Product Development", "Iterative development from requirements through design, prototype, validation, and release.", "mechanical_automotive", ("product development", "prototype development", "design improvements"), ("mechanical-design", "test-validation")),
    SkillDefinition("manufacturing-processes", "Manufacturing Processes", "Selection and analysis of processes used to manufacture mechanical components and assemblies.", "manufacturing_quality", ("manufacturing processes", "production processes"), ("engineering-drawings",)),
    SkillDefinition("python", "Python", "Python programming, packaging, testing, and runtime behaviour.", "software", ("python",)),
    SkillDefinition("java", "Java", "Java language, runtime, and application development.", "software", ("java", "jvm")),
    SkillDefinition("javascript", "JavaScript", "JavaScript language and application development.", "software", ("javascript", "typescript", "node.js", "nodejs")),
    SkillDefinition("rest-api", "REST API", "HTTP API design, resource modelling, validation, and failure handling.", "software", ("rest api", "restful", "api development", "apis")),
    SkillDefinition("authentication", "Authentication", "Identity verification, authorization boundaries, and credential handling.", "software", ("authentication", "authorization", "oauth", "openid", "jwt"), ("rest-api",)),
    SkillDefinition("sql", "SQL", "Relational querying, transactions, and data manipulation.", "software", ("sql", "relational database")),
    SkillDefinition("database-design", "Database Design", "Schema modelling, integrity constraints, normalization, and indexing.", "software", ("database design", "database schema", "data modeling", "normalization"), ("sql",)),
    SkillDefinition("oop", "Object-Oriented Programming", "Object modelling, encapsulation, composition, and polymorphism.", "software", ("object-oriented", "object oriented", "oop")),
    SkillDefinition("design-patterns", "Design Patterns", "Context-appropriate reusable software design structures and trade-offs.", "software", ("design patterns", "strategy pattern", "factory pattern"), ("oop",)),
    SkillDefinition("system-design", "System Design", "Architecture, scalability, reliability, interfaces, and trade-offs.", "software", ("system design", "distributed system", "architecture", "scalability")),
    SkillDefinition("linux", "Linux", "Linux processes, files, permissions, networking, and operations.", "software", ("linux", "unix")),
    SkillDefinition("docker", "Docker", "Container images, runtime isolation, networking, and storage.", "software", ("docker", "containerization", "containers"), ("linux",)),
    SkillDefinition("kubernetes", "Kubernetes", "Container orchestration, deployment, networking, availability, and operations.", "software", ("kubernetes", "k8s"), ("docker",)),
    SkillDefinition("ci-cd", "CI/CD", "Automated build, test, release, and deployment workflows.", "software", ("ci/cd", "continuous integration", "continuous deployment", "jenkins", "gitlab ci")),
    SkillDefinition("microservices", "Microservices", "Service boundaries, communication, resilience, and operability.", "software", ("microservices", "microservice"), ("rest-api", "system-design")),
    SkillDefinition("kafka", "Kafka", "Distributed event streaming, partitions, consumer groups, and delivery guarantees.", "data_ml", ("kafka", "kstreams", "ktable", "event streaming"), ("system-design",)),
    SkillDefinition("machine-learning", "Machine Learning", "Model development, evaluation, deployment, and monitoring.", "data_ml", ("machine learning", "deep learning", "pytorch", "tensorflow"), ("python",)),
    SkillDefinition("decision-analytics", "Decision Analytics", "Use of data and quantitative methods to support defensible business decisions.", "data_ml", ("decision analytics", "advanced analytics", "data analytics", "data-driven solutions")),
    SkillDefinition("statistical-modeling", "Statistical Modeling", "Development and validation of statistical, predictive, and simulation models.", "data_ml", ("statistical model", "predictive model", "simulation model"), ("decision-analytics",)),
    SkillDefinition("optimization", "Optimization", "Formulation and solution of constrained decision problems using quantitative optimization methods.", "data_ml", ("optimization techniques", "optimization", "optimisation"), ("decision-analytics",)),
    SkillDefinition("generative-ai", "Generative AI", "Responsible application and evaluation of generative models and large language models.", "data_ml", ("generative ai", "large language models", "prompt design")),
)

_CATALOG_BY_ID = {item.skill_id: item for item in _CATALOG}


def grounding_contract_hash() -> str:
    payload = {
        "role_profile_schema_version": ROLE_PROFILE_SCHEMA_VERSION,
        "question_grounding_schema_version": GROUNDING_SCHEMA_VERSION,
        "grounding_policy_version": GROUNDING_POLICY_VERSION,
        "minimum_grounded_skills": MIN_GROUNDED_SKILLS,
        "catalog": [asdict(item) for item in _CATALOG],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


def _evidence(text: str, aliases: tuple[str, ...], source: str) -> tuple[EvidenceSpan, ...]:
    found: list[EvidenceSpan] = []
    for alias in aliases:
        for match in _pattern(alias).finditer(text):
            found.append(EvidenceSpan(source, match.group(0), match.start(), match.end()))
    found.sort(key=lambda item: (item.start, item.end))
    return tuple(found[:5])


def _priority(text: str, spans: tuple[EvidenceSpan, ...]) -> str:
    required_terms = ("required", "must", "mandatory", "qualification", "responsibilit")
    preferred_terms = ("preferred", "desirable", "nice to have")
    for span in spans:
        context = text[max(0, span.start - 160):min(len(text), span.end + 80)].casefold()
        if any(term in context for term in required_terms):
            return "required"
        if any(term in context for term in preferred_terms):
            return "preferred"
    return "unspecified"


def _role_title(text: str, domain: str) -> str:
    candidates = text.split("\n")[:12]
    for line in candidates:
        match = re.search(r"(?:role|position|job title)\s*:\s*(.{3,100})", line, re.I)
        if match:
            return match.group(1).strip(" -:")
    if len(candidates) >= 2 and re.fullmatch(
        r"(?:associate|intern|engineer|developer|analyst|specialist|manager)",
        candidates[1],
        re.I,
    ) and 3 <= len(candidates[0]) <= 70:
        return f"{candidates[0]} {candidates[1]}"
    for line in candidates:
        if 3 <= len(line) <= 100 and not re.search(r"responsibilit|about |qualification", line, re.I):
            return line.strip(" -:")
    return {
        "electrical_electronics": "Electrical/Electronics Engineer",
        "manufacturing_quality": "Quality and Validation Engineer",
        "mechanical_automotive": "Mechanical/Automotive Engineer",
        "software": "Software Engineer",
        "data_ml": "Machine Learning Engineer",
    }.get(domain, "Technical role")


@lru_cache(maxsize=128)
def build_role_profile(jd_text: str, resume_text: str = "") -> RoleProfile:
    jd = normalize_document_text(jd_text.replace("--- END OF DOCUMENT ---", ""))
    resume = normalize_document_text(resume_text.replace("--- END OF DOCUMENT ---", ""))
    if len(jd) < 100:
        raise ValueError("A grounded role profile requires at least 100 JD characters")

    explicit: dict[str, GroundedSkill] = {}
    for definition in _CATALOG:
        jd_spans = _evidence(jd, definition.aliases, "jd")
        if not jd_spans:
            continue
        resume_spans = _evidence(resume, definition.aliases, "resume")
        explicit[definition.skill_id] = GroundedSkill(
            skill_id=definition.skill_id,
            canonical_name=definition.canonical_name,
            definition=definition.definition,
            domain=definition.domain,
            aliases=definition.aliases,
            priority=_priority(jd, jd_spans),
            support_type="explicit_jd",
            jd_evidence=jd_spans,
            resume_evidence=resume_spans,
            prerequisite_ids=definition.prerequisites,
        )
    if not explicit:
        raise ValueError("JD contains no supported, evidence-backed competencies")

    # Add reviewed prerequisites only when they support an explicit JD competency.
    for skill in tuple(explicit.values()):
        for prerequisite_id in skill.prerequisite_ids:
            if prerequisite_id in explicit or prerequisite_id not in _CATALOG_BY_ID:
                continue
            definition = _CATALOG_BY_ID[prerequisite_id]
            explicit[prerequisite_id] = GroundedSkill(
                skill_id=definition.skill_id,
                canonical_name=definition.canonical_name,
                definition=definition.definition,
                domain=definition.domain,
                aliases=definition.aliases,
                priority="prerequisite",
                support_type="reviewed_prerequisite",
                jd_evidence=skill.jd_evidence,
                resume_evidence=_evidence(resume, definition.aliases, "resume"),
                prerequisite_ids=definition.prerequisites,
            )

    domain_counts: dict[str, int] = {}
    for skill in explicit.values():
        if skill.support_type == "explicit_jd":
            domain_counts[skill.domain] = domain_counts.get(skill.domain, 0) + 1
    role_domain = min(
        domain_counts,
        key=lambda domain: (-domain_counts[domain], domain),
    )

    acronyms: list[dict] = []
    if re.search(r"(?<![A-Za-z])BMS(?![A-Za-z])", jd, re.I):
        battery_context = any(term in jd.casefold() for term in ("battery", "dcir", "hppc", "soc"))
        acronyms.append({
            "term": "BMS",
            "expansion": "Battery Management System" if battery_context else None,
            "status": "resolved" if battery_context else "unresolved",
            "support_type": "reviewed_domain_interpretation" if battery_context else "ambiguous",
        })
    if any(item["status"] == "unresolved" for item in acronyms):
        raise ValueError("JD contains unresolved ambiguous acronyms")

    skills = tuple(sorted(explicit.values(), key=lambda item: (item.selection_priority, item.skill_id)))
    if len(skills) < MIN_GROUNDED_SKILLS:
        raise ValueError(
            f"JD yields {len(skills)} evidence-backed competencies; "
            f"at least {MIN_GROUNDED_SKILLS} are required"
        )
    return RoleProfile(
        schema_version=ROLE_PROFILE_SCHEMA_VERSION,
        normalized_document_hash=normalized_text_hash(jd),
        role_title=_role_title(jd, role_domain),
        role_domain=role_domain,
        skills=skills,
        acronym_resolutions=tuple(acronyms),
    )


def validate_role_profile(profile: RoleProfile, jd_text: str) -> None:
    jd = normalize_document_text(jd_text.replace("--- END OF DOCUMENT ---", ""))
    if profile.normalized_document_hash != normalized_text_hash(jd):
        raise ValueError("Role profile JD hash does not match the supplied document")
    ids = {skill.skill_id for skill in profile.skills}
    if len(ids) != len(profile.skills):
        raise ValueError("Role profile contains duplicate skill IDs")
    for skill in profile.skills:
        if not skill.jd_evidence:
            raise ValueError(f"Skill {skill.skill_id} has no JD evidence")
        for span in skill.jd_evidence:
            if span.start < 0 or span.end <= span.start or jd[span.start:span.end] != span.text:
                raise ValueError(f"Skill {skill.skill_id} contains invalid JD evidence")
        if any(item == skill.skill_id for item in skill.prerequisite_ids):
            raise ValueError(f"Skill {skill.skill_id} has a self prerequisite")


def grounding_packet(profile: RoleProfile, target: GroundedSkill) -> dict:
    return {
        "schema_version": GROUNDING_SCHEMA_VERSION,
        "grounding_policy_version": GROUNDING_POLICY_VERSION,
        "role_title": profile.role_title,
        "role_domain": profile.role_domain,
        "role_profile_hash": profile.profile_hash,
        "target_skill_id": target.skill_id,
        "target_skill": target.canonical_name,
        "target_aliases": list(target.aliases),
        "target_definition": target.definition,
        "target_priority": target.priority,
        "support_type": target.support_type,
        "jd_evidence": [span.text for span in target.jd_evidence[:3]],
        "jd_evidence_spans": [asdict(span) for span in target.jd_evidence[:3]],
        "resume_evidence": [span.text for span in target.resume_evidence[:3]],
        "allowed_skill_ids": [skill.skill_id for skill in profile.skills],
        "acronym_resolutions": list(profile.acronym_resolutions),
    }


def build_pairing_record(profile: RoleProfile) -> dict:
    """Describe JD/resume alignment without turning it into a hidden filter."""
    explicit = [skill for skill in profile.skills if skill.support_type == "explicit_jd"]
    overlapping = [skill for skill in explicit if skill.resume_evidence]
    return {
        "schema_version": "aria-pairing-record-v1",
        "role_profile_hash": profile.profile_hash,
        "explicit_jd_skill_ids": [skill.skill_id for skill in explicit],
        "resume_overlap_skill_ids": [skill.skill_id for skill in overlapping],
        "explicit_jd_skill_count": len(explicit),
        "resume_overlap_skill_count": len(overlapping),
        "resume_overlap_ratio": len(overlapping) / max(len(explicit), 1),
        "pairing_class": "evidence_overlap" if overlapping else "no_evidence_overlap",
    }


def normalize_generated_question(value: str) -> str:
    """Remove presentation-only wrappers without repairing semantic defects."""
    text = (value or "").strip()
    fenced = re.fullmatch(
        r"```(?:text|markdown)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        text = fenced.group(1).strip()

    quote_pairs = (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"))
    for opening, closing in quote_pairs:
        if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
            text = text[len(opening):-len(closing)].strip()
            break

    text = re.sub(
        r"^(?:interview\s+)?question\s*:\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    for opening, closing in quote_pairs:
        if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
            text = text[len(opening):-len(closing)].strip()
            break
    return " ".join(text.split())


_QUESTION_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[+#.][A-Za-z0-9+#.]*)*")
_SUPPORT_STOP_WORDS = {
    "and", "application", "approach", "behaviour", "behavior", "context",
    "creation", "data", "design", "development", "execution", "for", "from",
    "handling", "implementation", "into", "model", "monitoring", "operations",
    "performance", "planning", "process", "runtime", "selection", "system",
    "technical", "testing", "that", "the", "this", "through", "use", "using",
    "validation", "with", "work",
}
_REQUEST_VERBS = (
    "analyze", "compare", "define", "describe", "design", "diagnose", "discuss",
    "evaluate", "explain", "identify", "outline", "show", "walk me through",
)


def _token_form(token: str) -> str:
    """Apply conservative inflection folding without conflating technical terms."""
    value = token.casefold().rstrip(".")
    if len(value) > 4 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 4 and value.endswith(("ches", "shes", "sses", "xes", "zes")):
        return value[:-2]
    if len(value) > 3 and value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def _support_tokens(values: list[str] | tuple[str, ...]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for match in _QUESTION_TOKEN_PATTERN.finditer(str(value)):
            raw = match.group(0)
            folded = _token_form(raw)
            is_short_technical = (
                any(character.isdigit() for character in raw)
                or any(character in "+#." for character in raw)
                or raw == "C"
            )
            if (
                folded not in _SUPPORT_STOP_WORDS
                and (len(folded) >= 3 or is_short_technical)
            ):
                tokens.add(folded)
    return tokens


def _interrogative_reasons(question: str) -> list[str]:
    if not question:
        return ["question must be a non-empty interrogative"]
    question_marks = question.count("?")
    if question_marks > 2:
        return ["question contains more than two assessment clauses"]
    if question_marks >= 1:
        return []

    lowered = question.casefold().strip()
    request_pattern = (
        r"^(?:(?:given|consider|assuming|suppose)\b.{0,300},\s*)?"
        r"(?:please\s+)?(?:"
        + "|".join(re.escape(verb) for verb in _REQUEST_VERBS)
        + r")\b"
    )
    # Imperative assessment prompts such as "Given this failure, describe ..."
    # are valid interview questions even when a local model ends them with a period.
    if re.search(request_pattern, lowered):
        return []
    return ["question must be a non-empty interrogative"]


def _is_contextual_skill(target_skill_id: str, other: SkillDefinition) -> bool:
    target = _CATALOG_BY_ID.get(target_skill_id)
    return target is not None and target.domain == other.domain


def validate_grounded_question(question: str, packet: dict, history: list[dict]) -> dict:
    normalized = " ".join((question or "").split())
    reasons = _interrogative_reasons(normalized)
    if len(normalized) > 500:
        reasons.append("question exceeds 500 characters")
    prior = {" ".join(str(item.get("q", "")).casefold().split()) for item in history}
    if normalized.casefold() in prior:
        reasons.append("question duplicates accepted history")

    target_terms = [
        packet.get("target_skill", ""),
        *packet.get("target_aliases", []),
        packet.get("target_definition", ""),
        *packet.get("jd_evidence", []),
    ]
    tokens = _support_tokens(target_terms)
    question_tokens = _support_tokens([normalized])
    overlap = sorted(tokens & question_tokens)
    target_aliases = [packet.get("target_skill", ""), *packet.get("target_aliases", [])]
    target_mentioned = any(
        alias and _pattern(str(alias)).search(normalized) for alias in target_aliases
    )
    if not target_mentioned and len(overlap) < 2:
        reasons.append("question has no lexical support from the target or JD evidence")

    allowed_skill_ids = set(packet.get("allowed_skill_ids", ()))
    generic_aliases = {
        "architecture", "validation activities", "testing and validation",
        "test and validation", "component selection", "design improvements",
    }
    off_profile_skills = []
    contextual_skills = []
    target_skill_id = str(packet.get("target_skill_id", ""))
    for definition in _CATALOG:
        if definition.skill_id in allowed_skill_ids:
            continue
        aliases = (definition.canonical_name, *definition.aliases)
        mentioned = any(
            alias.casefold() not in generic_aliases
            and _pattern(alias).search(normalized)
            for alias in aliases
        )
        if not mentioned:
            continue
        if target_mentioned and _is_contextual_skill(target_skill_id, definition):
            contextual_skills.append(definition.skill_id)
        else:
            off_profile_skills.append(definition.skill_id)
    if off_profile_skills:
        reasons.append(
            "question introduces off-profile competencies: "
            + ", ".join(sorted(set(off_profile_skills)))
        )

    for acronym in packet.get("acronym_resolutions", []):
        term = str(acronym.get("term", ""))
        expansion = acronym.get("expansion")
        if term and re.search(rf"\b{re.escape(term)}\b", normalized, re.I):
            if not expansion:
                reasons.append(f"unresolved acronym used: {term}")
            if term.casefold() == "bms" and re.search(r"building\s+management", normalized, re.I):
                reasons.append("BMS was assigned the wrong domain expansion")

    return {
        "schema_version": GROUNDING_SCHEMA_VERSION,
        "grounding_policy_version": GROUNDING_POLICY_VERSION,
        "target_skill_id": packet.get("target_skill_id"),
        "role_profile_hash": packet.get("role_profile_hash"),
        "decision": "accept" if not reasons else "reject",
        "valid": not reasons,
        "reasons": reasons,
        "supporting_terms": overlap,
        "off_profile_skill_ids": sorted(set(off_profile_skills)),
        "contextual_skill_ids": sorted(set(contextual_skills)),
    }
