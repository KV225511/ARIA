"""
Module 3 — Prosody Baseline Manager

Stores per-candidate prosody baselines and computes deviations from
baseline for turns 3+.  Thread-safe via a lock around baseline state.
"""

import threading


class ProsodyBaselineManager:
    def __init__(self, baseline_turns=2):
        self.baseline_turns = baseline_turns
        self.baselines = {}
        # P2 — Protect baselines dict against concurrent access from
        # multiple FastAPI request threads handling different candidates.
        self._lock = threading.Lock()

    def _get_or_create_candidate_state(self, candidate_id):
        if candidate_id not in self.baselines:
            self.baselines[candidate_id] = {
                "baseline_turns": [],
                "baseline": None
            }
        return self.baselines[candidate_id]

    def _extract_baseline_features(self, prosody_features):
        return {
            "pitch_mean": float(prosody_features.get("pitch_mean", 0.0)),
            "speech_rate": float(prosody_features.get("speech_rate", 0.0)),
            "energy_mean": float(prosody_features.get("energy_mean", 0.0))
        }

    def _store_baseline_turn(self, state, prosody_features):
        baseline_features = self._extract_baseline_features(prosody_features)
        state["baseline_turns"].append(baseline_features)
        return baseline_features

    def _compute_baseline(self, state) -> dict[str, float] | None:
        baseline_turns = state["baseline_turns"]
        if len(baseline_turns) == 0:
            return None
        n = len(baseline_turns)
        pitch_baseline = float(sum(t.get("pitch_mean", 0.0) for t in baseline_turns) / n)
        rate_baseline = float(sum(t.get("speech_rate", 0.0) for t in baseline_turns) / n)
        energy_baseline = float(sum(t.get("energy_mean", 0.0) for t in baseline_turns) / n)

        baseline = {
            'pitch_mean': pitch_baseline,
            'speech_rate': rate_baseline,
            'energy_mean': energy_baseline
        }

        state['baseline'] = baseline
        return baseline

    def _safe_deviation(self, current_value, baseline_value) -> float:
        current_value = float(current_value or 0.0)
        baseline_value = float(baseline_value or 0.0)

        if abs(baseline_value) < 1e-6:
            return 0.0

        return float((current_value - baseline_value) / baseline_value)

    def update_with_baseline(self, candidate_id, turn_id, prosody_features):
        # P2 — Thread-safe: acquire lock for the duration of the state
        # read/write cycle so concurrent requests for different candidates
        # don't corrupt the shared baselines dict.
        with self._lock:
            state = self._get_or_create_candidate_state(candidate_id)

            features = dict(prosody_features)

            # Case 1: Turn 1 and Turn 2 are baseline turns
            if turn_id <= self.baseline_turns:
                self._store_baseline_turn(state, features)

                # Compute baseline only when enough baseline turns are collected
                if len(state["baseline_turns"]) >= self.baseline_turns:
                    self._compute_baseline(state)

                features["pitch_deviation"] = None
                features["rate_deviation"] = None
                features["energy_deviation"] = None

                return features

            # Case 2: Turn 3 onwards, compare with baseline
            baseline = state.get("baseline")

            # If baseline was not computed yet, try computing it
            if baseline is None:
                baseline = self._compute_baseline(state)

            # If still no baseline, return None deviations safely
            if baseline is None:
                features["pitch_deviation"] = None
                features["rate_deviation"] = None
                features["energy_deviation"] = None
                return features

            # Compute deviation values
            features["pitch_deviation"] = self._safe_deviation(
                features.get("pitch_mean", 0.0),
                baseline.get("pitch_mean", 0.0)
            )

            features["rate_deviation"] = self._safe_deviation(
                features.get("speech_rate", 0.0),
                baseline.get("speech_rate", 0.0)
            )

            features["energy_deviation"] = self._safe_deviation(
                features.get("energy_mean", 0.0),
                baseline.get("energy_mean", 0.0)
            )

            return features