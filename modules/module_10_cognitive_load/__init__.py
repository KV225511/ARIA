"""
Module 10 — Cognitive Load Separator

Resolves the ambiguity between genuine knowledge gaps and anxiety-induced
performance degradation using prosody, vision, and semantic signals.

Classifies each turn into one of four cognitive load quadrants:
    - low:                  Low stress + high semantic score (optimal mastery)
    - anxiety:              High stress + high semantic score (knows but nervous)
    - ignorance:            High stress + low semantic score (doesn't know)
    - confident_ignorance:  Low stress + low semantic score (bluffing)

Owner: Krissh
"""

from modules.module_10_cognitive_load.classifier import CognitiveLoadClassifier

__all__ = [
    "CognitiveLoadClassifier",
]
