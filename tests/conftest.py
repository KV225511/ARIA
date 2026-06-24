"""Generate minimal test fixtures for ARIA module tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from config.settings import AUDIO_SAMPLE_RATE

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def ensure_fixtures() -> Path:
    """Create a 10-second mono 16 kHz WAV if missing."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = FIXTURES_DIR / "test_audio_10s.wav"

    if not wav_path.exists():
        duration_s = 10.0
        t = np.linspace(0, duration_s, int(AUDIO_SAMPLE_RATE * duration_s), dtype=np.float32)
        # Simple tone + amplitude modulation to mimic speech-like energy
        audio = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        sf.write(str(wav_path), audio, AUDIO_SAMPLE_RATE)

    return wav_path
