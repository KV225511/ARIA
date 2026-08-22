"""Semantic-only, reliability-weighted competency belief updates."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from modules.module_06_belief.belief_config import BeliefModelConfig


COGNITIVE_LOAD_LABELS = frozenset(
    {"low", "anxiety", "ignorance", "confident_ignorance"}
)


class BeliefStateUpdater:
    """Maintain per-skill competency posteriors without behavioral bias."""

    def __init__(self, skill_nodes, config: BeliefModelConfig | None = None):
        self.config = config or BeliefModelConfig()
        self.default_belief = np.asarray(self.config.class_prior, dtype=float)
        self.DEFAULT_BELIEF = self.default_belief.copy()
        self.CLASS_CENTERS = np.asarray(self.config.class_centers, dtype=float)
        self.class_scales = np.asarray(self.config.class_scales, dtype=float)
        self.likelihood_sigma = float(np.mean(self.class_scales))

        ordered_nodes = list(dict.fromkeys(str(skill) for skill in skill_nodes))
        self.beliefs = {skill: self.default_belief.copy() for skill in ordered_nodes}
        self.evidence_counts = {skill: 0 for skill in ordered_nodes}
        self.effective_sample_sizes = {skill: 0.0 for skill in ordered_nodes}
        # Backward-compatible read alias. Values are discounted effective evidence.
        self.evidence_strengths = self.effective_sample_sizes
        self.question_fingerprints = {skill: set() for skill in ordered_nodes}
        self.global_entropy_sum = sum(
            self._calculate_entropy(dist) for dist in self.beliefs.values()
        )

    @classmethod
    def from_legacy(cls, skill_nodes, likelihood_sigma: float):
        return cls(skill_nodes, config=BeliefModelConfig.legacy(likelihood_sigma))

    @staticmethod
    def _normalize(dist):
        values = np.asarray(dist, dtype=float)
        total = float(np.sum(values))
        if not np.all(np.isfinite(values)) or total <= 0.0:
            return np.ones_like(values) / len(values)
        return values / total

    @staticmethod
    def _softmax(logits):
        values = np.asarray(logits, dtype=float)
        shifted = values - np.max(values)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials)

    @staticmethod
    def _calculate_entropy(dist):
        values = np.asarray(dist, dtype=float)
        return float(-np.sum(values * np.log(values + 1e-12)))

    def _apply_floor(self, dist):
        floored = np.maximum(np.asarray(dist, dtype=float), self.config.posterior_floor)
        return self._normalize(floored)

    def get_global_entropy(self):
        """Legacy all-node entropy; use assessment entropy for termination."""
        if not self.beliefs:
            return 0.0
        return self.global_entropy_sum / len(self.beliefs)

    def get_assessment_entropy(self, skill_weights=None):
        return self._calculate_entropy(self.get_aggregate_belief(skill_weights))

    def get_mean_visited_entropy(self):
        visited = self.get_visited_skills()
        if not visited:
            return self._calculate_entropy(self.default_belief)
        return float(np.mean([
            self._calculate_entropy(self.beliefs[skill]) for skill in visited
        ]))

    def get_belief(self, skill):
        return self.beliefs.get(skill, self.default_belief.copy())

    def get_evidence_count(self, skill):
        return self.evidence_counts.get(skill, 0)

    def get_effective_sample_size(self, skill):
        return self.effective_sample_sizes.get(skill, 0.0)

    def get_visited_skills(self):
        return [skill for skill, count in self.evidence_counts.items() if count > 0]

    def ensure_skill(self, skill):
        """Add a previously unseen skill without changing existing evidence."""
        skill = str(skill)
        if skill not in self.beliefs:
            self.beliefs[skill] = self.default_belief.copy()
            self.evidence_counts[skill] = 0
            self.effective_sample_sizes[skill] = 0.0
            self.question_fingerprints[skill] = set()
            self.global_entropy_sum += self._calculate_entropy(self.default_belief)
        return skill

    def get_aggregate_belief(self, skill_weights: Mapping[str, float] | None = None):
        """Aggregate skill evidence with a normalized log-opinion pool."""
        visited = self.get_visited_skills()
        if not visited:
            return self.default_belief.copy()

        skill_weights = skill_weights or {}
        prior_log = np.log(np.maximum(self.default_belief, 1e-12))
        weighted_delta = np.zeros(3, dtype=float)
        total_weight = 0.0
        for skill in visited:
            importance = max(float(skill_weights.get(skill, 1.0)), 0.0)
            evidence_weight = min(
                self.effective_sample_sizes[skill],
                self.config.max_skill_effective_sample_size,
            )
            weight = importance * evidence_weight
            if weight <= 0.0:
                continue
            skill_log = np.log(np.maximum(self.beliefs[skill], 1e-12))
            weighted_delta += weight * (skill_log - prior_log)
            total_weight += weight

        if total_weight <= 0.0:
            return self.default_belief.copy()
        logits = prior_log + (weighted_delta / total_weight)
        logits /= self.config.aggregation_temperature
        return self._apply_floor(self._softmax(logits))

    def get_aggregate_assessment(self, skill_weights=None):
        belief = self.get_aggregate_belief(skill_weights=skill_weights)
        raw_label = int(np.argmax(belief))
        confidence = float(belief[raw_label])
        visited = self.get_visited_skills()
        effective_evidence = float(sum(
            min(value, self.config.max_skill_effective_sample_size)
            for value in self.effective_sample_sizes.values()
        ))
        sufficient = (
            confidence >= self.config.minimum_assessment_confidence
            and effective_evidence >= self.config.minimum_effective_evidence
            and len(visited) >= self.config.minimum_skill_coverage
        )
        return {
            "belief": belief,
            "label": raw_label if sufficient else None,
            "raw_label": raw_label,
            "status": "classified" if sufficient else "insufficient_evidence",
            "confidence": confidence,
            "effective_evidence": effective_evidence,
            "visited_skills": visited,
            "evidence_counts": dict(self.evidence_counts),
            "evidence_strengths": dict(self.effective_sample_sizes),
        }

    @staticmethod
    def _validate_evidence(semantic_score, cognitive_load, confidences):
        score = float(semantic_score)
        if not math.isfinite(score):
            raise ValueError("semantic_score must be finite")
        if not 0.0 <= score <= 1.0:
            raise ValueError("semantic_score must be in [0, 1]")
        load = cognitive_load or "low"
        if load not in COGNITIVE_LOAD_LABELS:
            raise ValueError(f"Unknown cognitive-load label: {load}")
        values = []
        for name, value in confidences.items():
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            values.append(number)
        return score, values

    def update_belief(
        self,
        skill,
        semantic_score,
        cognitive_load="low",
        behavior_score=None,
        evidence_confidence=1.0,
        stt_confidence=1.0,
        modality_confidence=1.0,
        question_fingerprint=None,
    ):
        """Update competency using semantic evidence only.

        ``behavior_score`` is accepted solely for legacy call compatibility and
        deliberately has no influence on the competency likelihood.
        """
        if skill not in self.beliefs:
            return self.default_belief.copy()
        score, reliability_parts = self._validate_evidence(
            semantic_score,
            cognitive_load,
            {
                "evidence_confidence": evidence_confidence,
                "stt_confidence": stt_confidence,
                "modality_confidence": modality_confidence,
            },
        )
        reliability = float(np.prod(reliability_parts))
        current_ess = self.effective_sample_sizes[skill]
        remaining = max(self.config.max_skill_effective_sample_size - current_ess, 0.0)
        discount = (1.0 + current_ess) ** self.config.repeat_discount_power
        update_weight = reliability / discount

        if question_fingerprint:
            fingerprint = str(question_fingerprint)
            if fingerprint in self.question_fingerprints[skill]:
                update_weight *= self.config.duplicate_question_multiplier
            self.question_fingerprints[skill].add(fingerprint)
        update_weight = min(update_weight, remaining)
        if update_weight <= 0.0:
            return self.beliefs[skill].copy()

        log_likelihood = (
            -0.5 * ((score - self.CLASS_CENTERS) / self.class_scales) ** 2
            - np.log(self.class_scales)
        )
        current = self.beliefs[skill]
        posterior = self._softmax(
            np.log(np.maximum(current, 1e-12)) + update_weight * log_likelihood
        )
        posterior = self._apply_floor(posterior)

        self.global_entropy_sum += (
            self._calculate_entropy(posterior) - self._calculate_entropy(current)
        )
        self.beliefs[skill] = posterior
        self.evidence_counts[skill] += 1
        self.effective_sample_sizes[skill] += update_weight
        return posterior.copy()
