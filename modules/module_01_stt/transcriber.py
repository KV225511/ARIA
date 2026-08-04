"""
Module 1 — Speech-to-Text (STT)

Uses faster-whisper (Whisper large-v3) to transcribe candidate audio.
Offline mode: transcribe a complete audio buffer or .wav file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import (
    ALLOWED_AUDIO_DIR,
    AUDIO_SAMPLE_RATE,
    DEVICE,
    MAX_AUDIO_DURATION_S,
    MODEL_WHISPER,
    WHISPER_COMPUTE_TYPE,
)

# Lazy-loaded singleton — models must NOT reload per call
_transcriber_instance: "Transcriber | None" = None


class Transcriber:
    """Whisper large-v3 transcriber with GPU int8 quantization."""

    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            MODEL_WHISPER,
            device=DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )

    def transcribe_sync(
        self,
        audio: np.ndarray,
        question_end_time: float | None = None,
    ) -> dict[str, Any]:
        """
        Transcribe a mono float32 audio array at 16 kHz.

        Args:
            audio: Raw audio, shape (N,) float32, 16 kHz mono.
            question_end_time: Unix timestamp when the question ended.
                Used to compute response_latency_ms from first spoken word.

        Returns:
            Dict matching the Module 1 output schema.

        Raises:
            ValueError: If audio shape is wrong or duration exceeds limit.
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected 1-D audio array, got shape {audio.shape}")

        # P0 — Reject oversized audio to prevent GPU OOM
        duration_s = len(audio) / AUDIO_SAMPLE_RATE
        if duration_s > MAX_AUDIO_DURATION_S:
            raise ValueError(
                f"Audio duration {duration_s:.1f}s exceeds maximum "
                f"allowed {MAX_AUDIO_DURATION_S}s"
            )

        segments, info = self._model.transcribe(
            audio,
            vad_filter=True,
            word_timestamps=True,
        )

        transcript_parts: list[str] = []
        word_timestamps: list[dict[str, float | str]] = []
        confidences: list[float] = []

        for segment in segments:
            transcript_parts.append(segment.text.strip())
            if segment.avg_logprob is not None:
                # Convert log-probability to 0-1 confidence (Whisper logprob ~ [-1, 0])
                confidences.append(float(min(1.0, max(0.0, 1.0 + segment.avg_logprob))))

            if segment.words:
                for word in segment.words:
                    word_timestamps.append(
                        {
                            "word": word.word.strip(),
                            "start": float(word.start),
                            "end": float(word.end),
                        }
                    )

        transcript = " ".join(part for part in transcript_parts if part).strip()
        confidence = float(np.mean(confidences)) if confidences else 0.0
        response_latency_ms = _compute_response_latency(
            word_timestamps, question_end_time
        )

        return {
            "transcript": transcript,
            "word_timestamps": word_timestamps,
            "language": info.language or "en",
            "confidence": confidence,
            "response_latency_ms": response_latency_ms,
        }

    def transcribe_file_sync(
        self,
        wav_path: str | Path,
        question_end_time: float | None = None,
    ) -> dict[str, Any]:
        """Load a .wav file and transcribe it (offline mode).

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file is outside ALLOWED_AUDIO_DIR.
        """
        import soundfile as sf

        path = Path(wav_path).resolve()

        # FIX C3 — Path traversal protection: reject paths outside allowed directory.
        # Also reject symlinks — relative_to() alone doesn't stop symlink attacks.
        allowed = Path(ALLOWED_AUDIO_DIR).resolve()
        try:
            path.relative_to(allowed)
        except ValueError:
            raise PermissionError(
                f"Access denied: audio path must be under {allowed}. "
                f"Got: {path}"
            )

        if path.is_symlink():
            raise PermissionError(
                f"Access denied: symlinked audio paths are not allowed. "
                f"Got: {path}"
            )

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio, sample_rate = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != AUDIO_SAMPLE_RATE:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=AUDIO_SAMPLE_RATE
            )

        return self.transcribe_sync(audio, question_end_time=question_end_time)


def _get_transcriber() -> Transcriber:
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = Transcriber()
    return _transcriber_instance


def _compute_response_latency(
    word_timestamps: list[dict[str, float | str]],
    question_end_time: float | None,
) -> float:
    """
    Compute ms from question end to first spoken word.

    If question_end_time is provided, latency is the difference between
    the absolute time of the first word and question_end_time.
    If question_end_time is None, returns latency from audio clip start
    to first word (offline mode).
    """
    if not word_timestamps:
        return 0.0

    first_word_start_s = float(word_timestamps[0]["start"])

    if question_end_time is not None:
        # P0 FIX: previously both branches were identical, ignoring
        # question_end_time entirely.  Now compute real absolute latency.
        # first_word_start_s is relative to clip start; question_end_time
        # is an absolute Unix timestamp.  The caller is responsible for
        # providing the absolute time of the first word via the audio
        # stream timestamp offset.  For clip-only mode (offline), the
        # question_end_time represents how far into the clip the question
        # ended, so latency = (first_word - question_end).
        latency_s = first_word_start_s - question_end_time
        return max(latency_s * 1000.0, 0.0)

    # Offline / clip-only mode: latency relative to clip start
    return first_word_start_s * 1000.0


async def transcribe(
    audio: np.ndarray,
    question_end_time: float | None = None,
) -> dict[str, Any]:
    """Async wrapper — runs transcription in a thread pool."""
    transcriber = _get_transcriber()
    return await asyncio.to_thread(
        transcriber.transcribe_sync, audio, question_end_time
    )


async def transcribe_file(
    wav_path: str | Path,
    question_end_time: float | None = None,
) -> dict[str, Any]:
    """Async wrapper for offline file transcription."""
    transcriber = _get_transcriber()
    return await asyncio.to_thread(
        transcriber.transcribe_file_sync, wav_path, question_end_time
    )
