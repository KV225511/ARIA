"""
Module 2 — Vision Baseline Manager

Stores per-candidate vision baselines (resting Action Units, eye contact score, blink rate)
and computes relative deviations from baseline for turns 3+. Thread-safe via lock.
"""

import threading
from typing import Any

from config.settings import BASELINE_TURNS
from modules.module_02_vision.face_mesh import AU_KEYS


class VisionBaselineManager:
    """Manages resting visual baselines per candidate across interview turns."""

    def __init__(self, baseline_turns: int = BASELINE_TURNS) -> None:
        self.baseline_turns = baseline_turns
        self.baselines: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _get_or_create_candidate_state(self, candidate_id: str) -> dict[str, Any]:
        if candidate_id not in self.baselines:
            self.baselines[candidate_id] = {
                "baseline_turns": [],
                "baseline": None,
            }
        return self.baselines[candidate_id]

    def _extract_baseline_features(self, vision_summary: dict[str, Any]) -> dict[str, Any]:
        au_acts = vision_summary.get("au_activations", {})
        return {
            "au_activations": {k: float(au_acts.get(k, 0.0)) for k in AU_KEYS},
            "eye_contact_score": float(vision_summary.get("eye_contact_score", 0.0)),
            "blink_rate": float(vision_summary.get("blink_rate", 0.0)),
        }

    def _store_baseline_turn(
        self, state: dict[str, Any], vision_summary: dict[str, Any]
    ) -> dict[str, Any]:
        bf = self._extract_baseline_features(vision_summary)
        state["baseline_turns"].append(bf)
        return bf

    def _compute_baseline(self, state: dict[str, Any]) -> dict[str, Any] | None:
        turns = state["baseline_turns"]
        if not turns:
            return None
        n = len(turns)

        au_baseline = {}
        for k in AU_KEYS:
            au_baseline[k] = float(sum(t["au_activations"].get(k, 0.0) for t in turns) / n)

        eye_baseline = float(sum(t.get("eye_contact_score", 0.0) for t in turns) / n)
        blink_baseline = float(sum(t.get("blink_rate", 0.0) for t in turns) / n)

        baseline = {
            "au_activations": au_baseline,
            "eye_contact_score": eye_baseline,
            "blink_rate": blink_baseline,
        }
        state["baseline"] = baseline
        return baseline

    def _safe_deviation(self, current_value: float, baseline_value: float) -> float:
        current_value = float(current_value or 0.0)
        baseline_value = float(baseline_value or 0.0)
        if abs(baseline_value) < 1e-6:
            return 0.0
        return float((current_value - baseline_value) / baseline_value)

    def update_with_baseline(
        self, candidate_id: str, turn_id: int, vision_summary: dict[str, Any]
    ) -> dict[str, Any]:
        """Attach baseline deviations to a turn summary."""
        with self._lock:
            state = self._get_or_create_candidate_state(candidate_id)
            summary = dict(vision_summary)

            # Case 1: Baseline collection turns
            if turn_id <= self.baseline_turns:
                self._store_baseline_turn(state, summary)
                if len(state["baseline_turns"]) >= self.baseline_turns:
                    self._compute_baseline(state)

                summary["au_deviations"] = None
                summary["eye_contact_deviation"] = None
                summary["blink_rate_deviation"] = None
                return summary

            # Case 2: Post-baseline comparison turns
            baseline = state.get("baseline")
            if baseline is None:
                baseline = self._compute_baseline(state)

            if baseline is None:
                summary["au_deviations"] = None
                summary["eye_contact_deviation"] = None
                summary["blink_rate_deviation"] = None
                return summary

            current_au = summary.get("au_activations", {})
            base_au = baseline.get("au_activations", {})
            summary["au_deviations"] = {
                k: self._safe_deviation(current_au.get(k, 0.0), base_au.get(k, 0.0))
                for k in AU_KEYS
            }
            summary["eye_contact_deviation"] = self._safe_deviation(
                summary.get("eye_contact_score", 0.0),
                baseline.get("eye_contact_score", 0.0),
            )
            summary["blink_rate_deviation"] = self._safe_deviation(
                summary.get("blink_rate", 0.0),
                baseline.get("blink_rate", 0.0),
            )
            return summary
