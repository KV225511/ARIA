"""Module 3 — Prosody feature extraction and baseline calibration."""

from modules.module_03_prosody.baseline import ProsodyBaselineManager
from modules.module_03_prosody.extractor import ProsodyExtractor
from modules.module_03_prosody.pipeline import process_prosody_turn

__all__ = [
    "ProsodyExtractor",
    "ProsodyBaselineManager",
    "process_prosody_turn",
]
