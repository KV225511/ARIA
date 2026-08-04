"""
Module 1 Extension — Automated Semantic Grading Head

Evaluates candidate speech transcripts against reference rubrics and answer keys
using TF-IDF N-gram cosine similarity and keyword coverage scoring.
"""

import re
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticGrader:
    """
    Automated short-answer and interview response semantic grading engine.

    Compares candidate transcription against expected rubric answers to compute
    objective similarity scores, keyword overlap, and letter grades.
    """

    def grade_response(
        self,
        candidate_transcript: str,
        reference_answer: str,
        required_keywords: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Grade a candidate's transcript against a reference rubric answer.

        Note: A fresh TfidfVectorizer is instantiated per call (not shared on self)
        to avoid state mutation across calls and thread-safety issues.
        """
        if not candidate_transcript or not candidate_transcript.strip():
            return {
                "similarity_score": 0.0,
                "keyword_coverage": 0.0,
                "missing_keywords": list(required_keywords or []),
                "composite_score": 0.0,
                "final_grade": "F",
                "feedback": "No transcription provided.",
            }

        if not reference_answer or not reference_answer.strip():
            return {
                "similarity_score": 1.0,
                "keyword_coverage": 1.0,
                "missing_keywords": [],
                "composite_score": 1.0,
                "final_grade": "A",
                "feedback": "No reference answer required.",
            }

        # FIX C2 — Instantiate a fresh TfidfVectorizer per call (no shared mutable state,
        # no data race in multithreaded use, proper vocabulary for each document pair).
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")

        corpus = [reference_answer.lower(), candidate_transcript.lower()]
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            sim_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            sim_score = 0.0

        # FIX M4 — Use regex word-boundary matching instead of substring matching.
        # Prevents "calm" from matching "becalmed", "focused" from matching "unfocused", etc.
        kw_coverage = 1.0
        missing_kws = []
        if required_keywords:
            cand_lower = candidate_transcript.lower()
            found = 0
            for kw in required_keywords:
                pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                if re.search(pattern, cand_lower):
                    found += 1
                else:
                    missing_kws.append(kw)
            kw_coverage = float(found / len(required_keywords))

        # Composite Score: cosine similarity (70%) + keyword coverage (30%)
        composite = (0.7 * sim_score) + (0.3 * kw_coverage)

        # FIX H4 — Added D grade band (0.15 – 0.35) to avoid harsh F jump from C
        if composite >= 0.75 or sim_score >= 0.80:
            grade = "A"
            feedback = "Excellent response covering core semantic concepts."
        elif composite >= 0.55 or sim_score >= 0.60:
            grade = "B"
            feedback = "Good response, though some technical nuances or keywords were missed."
        elif composite >= 0.35:
            grade = "C"
            feedback = "Partial answer; significant concepts absent."
        elif composite >= 0.15:
            grade = "D"
            feedback = "Minimal overlap with reference rubric; key concepts largely missing."
        else:
            grade = "F"
            feedback = "Response diverges significantly from expected reference rubric."

        return {
            "similarity_score": sim_score,
            "keyword_coverage": kw_coverage,
            "missing_keywords": missing_kws,
            "composite_score": composite,
            "final_grade": grade,
            "feedback": feedback,
        }
