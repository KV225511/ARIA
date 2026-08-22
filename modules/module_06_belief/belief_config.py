"""Versioned configuration for ARIA's competency belief model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BELIEF_SCHEMA_VERSION = "belief-v2"


@dataclass(frozen=True)
class BeliefModelConfig:
    schema_version: str = BELIEF_SCHEMA_VERSION
    class_centers: tuple[float, float, float] = (0.20, 0.50, 0.80)
    class_scales: tuple[float, float, float] = (0.22, 0.22, 0.22)
    class_prior: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    repeat_discount_power: float = 0.5
    duplicate_question_multiplier: float = 0.25
    max_skill_effective_sample_size: float = 5.0
    aggregation_temperature: float = 1.0
    posterior_floor: float = 1e-4
    minimum_assessment_confidence: float = 0.50
    minimum_effective_evidence: float = 2.0
    minimum_skill_coverage: int = 3
    raw_dataset_hash: str = ""
    split_manifest_hash: str = ""
    fit_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        vectors = {
            "class_centers": self.class_centers,
            "class_scales": self.class_scales,
            "class_prior": self.class_prior,
        }
        for name, values in vectors.items():
            if len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f"{name} must contain three finite values")
        if any(not 0.0 <= value <= 1.0 for value in self.class_centers):
            raise ValueError("class_centers must be in [0, 1]")
        if any(left > right for left, right in zip(self.class_centers, self.class_centers[1:])):
            raise ValueError("class_centers must be monotonically ordered")
        if any(value <= 0.0 for value in self.class_scales):
            raise ValueError("class_scales must be positive")
        if any(value <= 0.0 for value in self.class_prior):
            raise ValueError("class_prior values must be positive")
        if not math.isclose(sum(self.class_prior), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("class_prior must sum to 1")
        if self.repeat_discount_power < 0.0:
            raise ValueError("repeat_discount_power must be non-negative")
        if not 0.0 <= self.duplicate_question_multiplier <= 1.0:
            raise ValueError("duplicate_question_multiplier must be in [0, 1]")
        if self.max_skill_effective_sample_size <= 0.0:
            raise ValueError("max_skill_effective_sample_size must be positive")
        if self.aggregation_temperature <= 0.0:
            raise ValueError("aggregation_temperature must be positive")
        if not 0.0 < self.posterior_floor < 1.0 / 3.0:
            raise ValueError("posterior_floor must be in (0, 1/3)")
        if not 0.0 <= self.minimum_assessment_confidence <= 1.0:
            raise ValueError("minimum_assessment_confidence must be in [0, 1]")
        if self.minimum_effective_evidence < 0.0:
            raise ValueError("minimum_effective_evidence must be non-negative")
        if self.minimum_skill_coverage < 0:
            raise ValueError("minimum_skill_coverage must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def with_updates(self, **changes) -> "BeliefModelConfig":
        return replace(self, **changes)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(output)
        return output

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BeliefModelConfig":
        converted = dict(data)
        for name in ("class_centers", "class_scales", "class_prior"):
            if name in converted:
                converted[name] = tuple(float(value) for value in converted[name])
        return cls(**converted)

    @classmethod
    def load(cls, path: str | Path) -> "BeliefModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def legacy(cls, likelihood_sigma: float) -> "BeliefModelConfig":
        """Explicit adapter for historical shared-sigma callers."""
        sigma = float(likelihood_sigma)
        return cls(
            schema_version="belief-v1-legacy-adapter",
            class_scales=(sigma, sigma, sigma),
            repeat_discount_power=0.0,
            max_skill_effective_sample_size=1e9,
            posterior_floor=1e-9,
            minimum_assessment_confidence=0.0,
            minimum_effective_evidence=0.0,
            minimum_skill_coverage=0,
        )
