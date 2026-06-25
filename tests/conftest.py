"""Generate minimal test fixtures for ARIA module tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from config.settings import AUDIO_SAMPLE_RATE

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def ensure_fixtures() -> Path:
    """Create a 10-second mono 16 kHz WAV if missing."""
    return ensure_prosody_fixture(duration_s=10.0, filename="test_audio_10s.wav")


def ensure_prosody_fixture(
    duration_s: float = 30.0,
    filename: str | None = None,
) -> Path:
    """Create a mono 16 kHz WAV fixture for prosody tests."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"test_audio_{int(duration_s)}s.wav"
    wav_path = FIXTURES_DIR / filename

    if not wav_path.exists():
        t = np.linspace(0, duration_s, int(AUDIO_SAMPLE_RATE * duration_s), dtype=np.float32)
        # Tone + amplitude modulation — enough energy for openSMILE / librosa VAD
        audio = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        sf.write(str(wav_path), audio, AUDIO_SAMPLE_RATE)

    return wav_path
