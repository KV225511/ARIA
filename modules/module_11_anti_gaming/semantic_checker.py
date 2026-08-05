"""
Module 11 — Anti-Gaming: Coaching & Scripting Detection via Semantic Analysis

Detects two related gaming strategies:

1. **Coaching** — Someone off-camera is feeding the candidate answers.
   Signal: lateral head turns (looking off-screen) + high-quality answer
   appearing after the head turn.

2. **Scripted** — Candidate has pre-memorized answers for anticipated questions.
   Signal: high semantic similarity between answers to unrelated questions
   (same prepared script recycled) + abrupt lexical complexity shifts
   inconsistent with the candidate's natural speech baseline.

Uses TF-IDF cosine similarity (lightweight, no external model) for
cross-turn semantic comparison. BGE-M3 can be swapped in later when
Raghav's embedding pipeline is integrated.

Input:
    transcript: str                — Current turn transcript
    vision: dict                   — Module 2 per-turn summary (head_pose)
    word_timestamps: list          — Module 1 word-level timestamps
    session_history: list[dict]    — Previous turn transcripts

Output:
    {
        "flags": list,              # subset of ["coaching", "scripted"]
        "confidences": dict,        # confidence per flag
        "evidence": {
            "lateral_head_turns": int,
            "max_head_yaw": float,
            "max_cross_turn_similarity": float,
            "complexity_shift": float,
            "complexity_baseline": float,
            "complexity_current": float,
        }
    }

Owner: Krissh
"""

from __future__ import annotations

import math
import re
from typing import Any


# ── DETECTION THRESHOLDS ───────────────────────────────────────────────────

# Head yaw (degrees) beyond which a head turn is classified as lateral
# (looking off-camera, potentially toward a coach)
LATERAL_HEAD_YAW_THRESHOLD = 25.0

# Cross-turn TF-IDF cosine similarity above this value for answers to
# different questions is suspicious (recycled scripted content)
SCRIPTED_SIMILARITY_THRESHOLD = 0.65

# Very high similarity — near-identical phrasing across different questions
HIGH_SIMILARITY_THRESHOLD = 0.80

# Lexical complexity shift (ratio) that triggers scripting flag.
# If current turn complexity is >2x or <0.5x the session baseline,
# it suggests inconsistent authorship (AI-generated vs natural speech).
COMPLEXITY_SHIFT_THRESHOLD = 1.8

# Minimum turns of history needed before cross-turn comparison is meaningful
MIN_HISTORY_TURNS = 2

# Minimum transcript length (words) to analyze
MIN_WORDS_FOR_ANALYSIS = 5


class SemanticChecker:
    """
    Detects coaching and scripting patterns by analyzing head movements,
    cross-turn semantic similarity, and lexical complexity consistency.

    Stateless per-call — session history is passed in, not stored.

    Usage:
        checker = SemanticChecker()
        result = checker.detect(
            transcript="The candidate's answer...",
            vision=vision_summary,
            word_timestamps=word_timestamps,
            session_history=[{"transcript": "prev answer", "turn_id": 1}, ...],
        )
    """

    def __init__(
        self,
        lateral_yaw_threshold: float = LATERAL_HEAD_YAW_THRESHOLD,
        similarity_threshold: float = SCRIPTED_SIMILARITY_THRESHOLD,
        complexity_shift_threshold: float = COMPLEXITY_SHIFT_THRESHOLD,
        min_history_turns: int = MIN_HISTORY_TURNS,
    ) -> None:
        self.lateral_yaw_threshold = lateral_yaw_threshold
        self.similarity_threshold = similarity_threshold
        self.complexity_shift_threshold = complexity_shift_threshold
        self.min_history_turns = min_history_turns

    def detect(
        self,
        transcript: str,
        vision: dict[str, Any],
        word_timestamps: list[dict[str, Any]],
        session_history: list[dict[str, Any]],
        gaze_frames: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze current turn for coaching and scripting signals.

        Args:
            transcript: Current turn transcript text.
            vision: Module 2 per-turn summary (must include head_pose).
            word_timestamps: Word-level timestamps from Module 1.
            session_history: Previous turn dicts with at minimum
                             {"transcript": str, "turn_id": int}.

        Returns:
            Detection result dict with flags, confidences, and evidence.
        """
        transcript = (transcript or "").strip()
        vision = vision or {}
        session_history = session_history or []

        flags: list[str] = []
        confidences: dict[str, float] = {"coaching": 0.0, "scripted": 0.0}
        evidence: dict[str, Any] = {
            "lateral_head_turns": 0,
            "max_head_yaw": 0.0,
            "max_cross_turn_similarity": 0.0,
            "complexity_shift": 0.0,
            "complexity_baseline": 0.0,
            "complexity_current": 0.0,
        }

        words = transcript.split()
        if len(words) < MIN_WORDS_FOR_ANALYSIS:
            return {"flags": flags, "confidences": confidences, "evidence": evidence}

        # ── Detect lateral coaching (head turns) ───────────────────────────
        coaching_confidence, head_evidence = self._detect_coaching(vision, gaze_frames)
        evidence.update(head_evidence)
        confidences["coaching"] = coaching_confidence
        if coaching_confidence > 0:
            flags.append("coaching")

        # ── Scripting Detection: Cross-Turn Similarity + Complexity ────────
        scripted_confidence, script_evidence = self._detect_scripting(
            transcript, session_history
        )
        evidence.update(script_evidence)
        confidences["scripted"] = scripted_confidence
        if scripted_confidence > 0:
            flags.append("scripted")

        return {
            "flags": flags,
            "confidences": confidences,
            "evidence": evidence,
        }

    # ── COACHING DETECTION ─────────────────────────────────────────────────

    def _detect_coaching(
        self,
        vision: dict[str, Any],
        gaze_frames: list[dict[str, Any]] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """
        Detect lateral head turns indicating off-camera coaching.

        Returns:
            (confidence, evidence_dict)
        """
        evidence = {
            "lateral_head_turns": 0,
            "max_head_yaw": 0.0,
        }
        
        # Determine sequence of yaw values
        yaw_sequence = []
        if gaze_frames:
            for frame in gaze_frames:
                yaw = abs(float(frame.get("head_pose", {}).get("yaw", 0.0)))
                yaw_sequence.append(yaw)
        else:
            head_pose = vision.get("head_pose", {})
            yaw_sequence.append(abs(float(head_pose.get("yaw", 0.0))))
            
        if not yaw_sequence:
            return 0.0, evidence
            
        # Detect sustained turns (more than a single frame glitch)
        max_yaw = max(yaw_sequence)
        evidence["max_head_yaw"] = max_yaw
        
        # Count frames above threshold
        frames_above_thresh = sum(1 for y in yaw_sequence if y >= self.lateral_yaw_threshold)
        
        # Need at least a minimal sequence (e.g., 2 frames) to trigger coaching reliably if gaze_frames are provided
        if gaze_frames and frames_above_thresh < 2:
            return 0.0, evidence
        elif not gaze_frames and max_yaw < self.lateral_yaw_threshold:
            return 0.0, evidence

        # Head turn detected — confidence scales with severity
        evidence["lateral_head_turns"] = 1

        # Linear scale from threshold to 2x threshold
        severity = min(
            (max_yaw - self.lateral_yaw_threshold) / self.lateral_yaw_threshold,
            1.0,
        )
        confidence = 0.3 + 0.4 * severity

        return _clamp(confidence, 0.0, 1.0), evidence

    # ── SCRIPTING DETECTION ────────────────────────────────────────────────

    def _detect_scripting(
        self,
        transcript: str,
        session_history: list[dict[str, Any]],
    ) -> tuple[float, dict[str, Any]]:
        """
        Detect scripted/memorized answers via cross-turn similarity
        and lexical complexity shifts.

        Returns:
            (confidence, evidence_dict)
        """
        evidence: dict[str, Any] = {
            "max_cross_turn_similarity": 0.0,
            "complexity_shift": 0.0,
            "complexity_baseline": 0.0,
            "complexity_current": 0.0,
        }

        if len(session_history) < self.min_history_turns:
            return 0.0, evidence

        # ── Cross-turn similarity (TF-IDF cosine) ─────────────────────────
        max_similarity = 0.0
        history_transcripts = [
            h.get("transcript", "")
            for h in session_history
            if h.get("transcript", "").strip()
        ]

        for prev_transcript in history_transcripts:
            sim = _tfidf_cosine_similarity(transcript, prev_transcript)
            max_similarity = max(max_similarity, sim)

        evidence["max_cross_turn_similarity"] = max_similarity

        # ── Lexical complexity shift ───────────────────────────────────────
        current_complexity = _compute_lexical_complexity(transcript)
        evidence["complexity_current"] = current_complexity

        if history_transcripts:
            baseline_complexities = []
            for h in session_history:
                if h.get("transcript", "").strip():
                    if "complexity" in h:
                        baseline_complexities.append(float(h["complexity"]))
                    else:
                        baseline_complexities.append(_compute_lexical_complexity(h["transcript"]))
                        
            if baseline_complexities:
                baseline_avg = sum(baseline_complexities) / len(baseline_complexities)
                evidence["complexity_baseline"] = baseline_avg

                if baseline_avg > 0.01:
                    shift_ratio = current_complexity / baseline_avg
                    evidence["complexity_shift"] = shift_ratio
            else:
                shift_ratio = 1.0
                evidence["complexity_shift"] = 1.0
        else:
            shift_ratio = 1.0

        # ── Combine signals ────────────────────────────────────────────────
        similarity_suspicious = max_similarity >= self.similarity_threshold
        complexity_suspicious = (
            shift_ratio >= self.complexity_shift_threshold
            or (shift_ratio > 0 and shift_ratio <= 1.0 / self.complexity_shift_threshold)
        )

        if similarity_suspicious and complexity_suspicious:
            # Both signals — strong evidence
            sim_severity = min(
                (max_similarity - self.similarity_threshold)
                / (HIGH_SIMILARITY_THRESHOLD - self.similarity_threshold + 1e-6),
                1.0,
            )
            confidence = 0.55 + 0.35 * sim_severity
        elif similarity_suspicious:
            # Similarity alone — moderate evidence
            sim_severity = min(
                (max_similarity - self.similarity_threshold)
                / (HIGH_SIMILARITY_THRESHOLD - self.similarity_threshold + 1e-6),
                1.0,
            )
            confidence = 0.30 + 0.25 * sim_severity
        elif complexity_suspicious:
            # Complexity shift alone — weak signal (could be topic change)
            confidence = 0.15
        else:
            confidence = 0.0

        return _clamp(confidence, 0.0, 1.0), evidence


# ── TF-IDF COSINE SIMILARITY ──────────────────────────────────────────────
# Lightweight implementation matching Module 1's SemanticGrader approach.
# No external dependencies — uses simple Python math.

def _tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Compute TF-IDF-weighted cosine similarity between two texts.

    Uses unigram + bigram features. Fresh computation per call
    (no shared state, thread-safe — matches SemanticGrader Fix C2).
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    # Build unigram + bigram feature sets
    ngrams_a = _extract_ngrams(tokens_a, max_n=2)
    ngrams_b = _extract_ngrams(tokens_b, max_n=2)

    # Combined vocabulary
    vocab = set(ngrams_a.keys()) | set(ngrams_b.keys())
    if not vocab:
        return 0.0

    # Simple TF vectors (normalized term frequency)
    total_a = sum(ngrams_a.values())
    total_b = sum(ngrams_b.values())

    # IDF weights (simple two-document IDF)
    idf: dict[str, float] = {}
    for term in vocab:
        doc_count = (1 if term in ngrams_a else 0) + (1 if term in ngrams_b else 0)
        idf[term] = math.log(2.0 / doc_count) + 1.0

    # TF-IDF vectors
    vec_a: dict[str, float] = {}
    vec_b: dict[str, float] = {}
    for term in vocab:
        tf_a = ngrams_a.get(term, 0) / max(total_a, 1)
        tf_b = ngrams_b.get(term, 0) / max(total_b, 1)
        vec_a[term] = tf_a * idf[term]
        vec_b[term] = tf_b * idf[term]

    # Cosine similarity
    dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vocab)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if mag_a < 1e-10 or mag_b < 1e-10:
        return 0.0

    return _clamp(dot / (mag_a * mag_b), 0.0, 1.0)


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization with basic punctuation stripping."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _extract_ngrams(tokens: list[str], max_n: int = 2) -> dict[str, int]:
    """Extract unigram and bigram frequency counts."""
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    if max_n >= 2:
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i + 1]}"
            counts[bigram] = counts.get(bigram, 0) + 1
    return counts


def _compute_lexical_complexity(text: str) -> float:
    """
    Estimate lexical complexity as the average syllable count per word.

    A rough proxy for vocabulary sophistication — natural speech tends
    to be 1.2–1.8 syllables/word; AI-generated text is often 2.0+.
    """
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    total_syllables = sum(_estimate_syllables(w) for w in words)
    return total_syllables / len(words)


def _estimate_syllables(word: str) -> int:
    """
    Estimate syllable count using vowel group heuristic.

    Not perfectly accurate, but consistent enough for relative
    cross-turn comparison (which is what we need).
    """
    word = word.lower().strip()
    if not word:
        return 1

    # Count vowel groups
    count = len(re.findall(r"[aeiouy]+", word))

    # Adjust for silent-e
    if word.endswith("e") and count > 1:
        count -= 1

    # Adjust for common suffixes
    if word.endswith("le") and len(word) > 2 and word[-3] not in "aeiouy":
        count += 1

    return max(count, 1)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to [low, high]."""
    return max(low, min(high, value))
