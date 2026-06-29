"""
Module 3 — Prosody Feature Extraction

Extracts speech prosody features from a candidate's turn audio:
pitch, energy, MFCCs, pauses, disfluencies, speech rate, jitter,
shimmer, and personal-baseline deviations.

Uses openSMILE eGeMAPSv02 for acoustic features and librosa for
VAD / MFCC.  openSMILE is called exactly once for LLD and once for
Functionals per turn (cached), not per-feature.
"""

import re

import librosa
import numpy as np
import opensmile

from config.settings import AUDIO_SAMPLE_RATE, MFCC_COEFFICIENTS

# openSMILE eGeMAPS F0 is semitones relative to this reference frequency
F0_REFERENCE_HZ = 27.5

# P1 — Cap value for speech_to_silence_ratio when silence is near-zero
# instead of returning raw speech duration in seconds (wrong unit).
MAX_SPEECH_SILENCE_RATIO = 1e6


def _semitones_to_hz(semitones: np.ndarray) -> np.ndarray:
    """Convert openSMILE semitone F0 values to Hz."""
    return F0_REFERENCE_HZ * (2 ** (semitones / 12.0))


class ProsodyExtractor:
    def __init__(self):
        self.smile_functionals = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        self.smile_lld = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

        self._wavlm_model = None
        self._wavlm_extractor = None

    def extract(self, audio_clip, word_timestamps=None, response_latency_ms=None):
        audio_arr = self._validate_audio(audio_clip)

        duration = self._compute_duration(audio_arr, AUDIO_SAMPLE_RATE)

        # P1 — Process openSMILE exactly ONCE for each level and cache the
        # resulting DataFrames.  Previously LLD was called 2× and Functionals
        # 2× per turn, doubling latency for no reason.
        lld_df = self._compute_lld(audio_arr, AUDIO_SAMPLE_RATE)
        func_df = self._compute_functionals(audio_arr, AUDIO_SAMPLE_RATE)

        pitch_features = self._compute_pitch_features(lld_df)
        energy_mean = self._compute_energy(lld_df)
        mfcc_vector = self._compute_mfcc(audio_arr, AUDIO_SAMPLE_RATE)
        wavlm_embedding = self._compute_wavlm_embedding(audio_arr, AUDIO_SAMPLE_RATE)
        jitter = self._compute_jitter(func_df)
        shimmer = self._compute_shimmer(func_df)

        intervals = self._detect_speech_intervals(audio_arr)

        pause_features = self._compute_pause_features(intervals, AUDIO_SAMPLE_RATE)
        speech_to_silence_ratio = self._compute_speech_to_silence_ratio(
            intervals, duration, AUDIO_SAMPLE_RATE
        )

        speech_rate = self._compute_speech_rate(word_timestamps)
        disfluency_features = self._compute_disfluencies(word_timestamps)

        latency = self._normalize_response_latency(response_latency_ms)

        return {
            **pitch_features,
            "speech_rate": speech_rate,
            **pause_features,
            "disfluency_count": disfluency_features["disfluency_count"],
            "disfluency_timestamps": disfluency_features["disfluency_timestamps"],
            "response_latency_ms": latency,
            "energy_mean": energy_mean,
            "jitter": jitter,
            "shimmer": shimmer,
            "mfcc_vector": mfcc_vector,
            "wavlm_embedding": wavlm_embedding,
            "speech_to_silence_ratio": speech_to_silence_ratio,
        }

    # ── Audio validation ───────────────────────────────────────────────────

    def _validate_audio(self, audio_clip) -> np.ndarray:
        audio_arr = np.asarray(audio_clip)

        if len(audio_arr) == 0:
            raise ValueError("The Audio file is empty")

        if audio_arr.ndim != 1:
            raise ValueError(
                f"The ProsodyExtractor expects 1D audio but received shape {audio_arr.shape}"
            )

        audio_arr = audio_arr.astype(np.float32)
        audio_arr = np.nan_to_num(audio_arr)

        max_amp = np.max(np.abs(audio_arr))

        if max_amp > 1.0:
            audio_arr = audio_arr / max_amp

        return audio_arr

    def _compute_duration(self, audio_arr, sample_rate) -> float:
        total_sample = len(audio_arr)

        if sample_rate <= 0:
            raise ValueError("Invalid sample rate")

        if len(audio_arr) == 0:
            raise ValueError("The Audio file is empty")

        duration = total_sample / sample_rate
        return float(duration)

    # ── Cached openSMILE calls ─────────────────────────────────────────────

    def _compute_lld(self, audio_arr, sample_rate):
        """Single LLD extraction — used for pitch and energy."""
        if len(audio_arr) < int(0.1 * sample_rate) or np.max(np.abs(audio_arr)) < 1e-6:
            return None
        try:
            df = self.smile_lld.process_signal(audio_arr, sample_rate)
            return df if not df.empty else None
        except Exception:
            return None

    def _compute_functionals(self, audio_arr, sample_rate):
        """Single Functionals extraction — used for jitter and shimmer."""
        if len(audio_arr) < int(0.1 * sample_rate) or np.max(np.abs(audio_arr)) < 1e-6:
            return None
        try:
            df = self.smile_functionals.process_signal(audio_arr, sample_rate)
            return df if not df.empty else None
        except Exception:
            return None

    # ── Pitch ──────────────────────────────────────────────────────────────

    def _compute_pitch_features(self, lld_df) -> dict[str, float]:
        if lld_df is None:
            return {
                "pitch_mean": 0.0,
                "pitch_variance": 0.0,
                "pitch_range": 0.0,
            }

        pitch_col = "F0semitoneFrom27.5Hz_sma3nz"

        if pitch_col not in lld_df.columns:
            return {
                "pitch_mean": 0.0,
                "pitch_variance": 0.0,
                "pitch_range": 0.0,
            }

        pitch_semitones = lld_df[pitch_col].to_numpy(dtype=np.float32)
        valid_semitones = pitch_semitones[np.isfinite(pitch_semitones) & (pitch_semitones > 0)]

        if len(valid_semitones) == 0:
            return {
                "pitch_mean": 0.0,
                "pitch_variance": 0.0,
                "pitch_range": 0.0,
            }

        pitch_hz = _semitones_to_hz(valid_semitones)
        mean = float(np.mean(pitch_hz))
        variance = float(np.var(pitch_hz))
        range_val = float(np.max(pitch_hz) - np.min(pitch_hz))

        return {
            "pitch_mean": mean,
            "pitch_variance": variance,
            "pitch_range": range_val,
        }

    # ── Energy ─────────────────────────────────────────────────────────────

    def _compute_energy(self, lld_df) -> float:
        if lld_df is None:
            return 0.0

        energy_col = "Loudness_sma3"

        if energy_col not in lld_df.columns:
            return 0.0

        energy_values = lld_df[energy_col].to_numpy(dtype=np.float32)
        valid_energy = energy_values[np.isfinite(energy_values)]

        if len(valid_energy) == 0:
            return 0.0

        return float(np.mean(valid_energy))

    # ── MFCC ───────────────────────────────────────────────────────────────

    def _compute_mfcc(self, audio_arr, sample_rate, n_mfcc=None) -> list[float]:
        """13 MFCC coefficients via librosa (eGeMAPS does not include MFCCs)."""
        n_mfcc = n_mfcc or MFCC_COEFFICIENTS
        if len(audio_arr) < int(0.1 * sample_rate) or np.max(np.abs(audio_arr)) < 1e-6:
            return [0.0] * n_mfcc

        mfcc_mat = librosa.feature.mfcc(y=audio_arr, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfcc_mat, axis=1).astype(float).tolist()

    # ── Jitter / Shimmer (from cached Functionals) ─────────────────────────

    def _compute_jitter(self, func_df) -> float:
        if func_df is None:
            return 0.0
        features = func_df.iloc[0].to_dict()
        jitter = features.get("jitterLocal_sma3nz_amean", 0.0)
        return float(jitter)

    def _compute_shimmer(self, func_df) -> float:
        if func_df is None:
            return 0.0
        features = func_df.iloc[0].to_dict()
        shimmer = features.get("shimmerLocaldB_sma3nz_amean", 0.0)
        return float(shimmer)

    # ── Speech intervals / pauses ──────────────────────────────────────────

    def _detect_speech_intervals(self, audio_arr, top_db=30) -> np.ndarray:
        if len(audio_arr) == 0:
            return np.empty((0, 2), dtype=int)

        if np.max(np.abs(audio_arr)) < 1e-6:
            return np.empty((0, 2), dtype=int)

        intervals = librosa.effects.split(
            audio_arr,
            top_db=top_db,
        )

        return intervals

    def _compute_pause_features(self, intervals, sample_rate, min_pause_ms=250) -> dict[str, int | float]:
        if len(intervals) == 0 or len(intervals) == 1:
            return {
                "pause_count": 0,
                "pause_total_duration_ms": 0.0
            }
        count = 0
        tot_duration = 0
        for i in range(1, len(intervals)):
            prev = intervals[i - 1][1]
            curr = intervals[i][0]
            diff = curr - prev
            gap_ms = diff / sample_rate * 1000
            if gap_ms > min_pause_ms:
                count += 1
                tot_duration += gap_ms

        return {
            "pause_count": int(count),
            "pause_total_duration_ms": float(tot_duration)
        }

    def _compute_speech_to_silence_ratio(self, intervals, total_duration, sample_rate) -> float:
        if intervals is None or len(intervals) == 0:
            return 0.0
        speech_duration = 0.0
        for interval in intervals:
            start = interval[0]
            end = interval[1]
            interval_duration = (end - start) / sample_rate
            speech_duration += interval_duration
        silence_duration = total_duration - speech_duration
        silence_duration = max(silence_duration, 0.0)
        # P1 FIX: Previously returned raw speech_duration (in seconds)
        # when silence was near-zero — wrong unit (not a ratio).
        # Now returns a capped max value to keep it dimensionless.
        if silence_duration < 1e-4:
            return float(MAX_SPEECH_SILENCE_RATIO)
        ratio = speech_duration / silence_duration
        return float(ratio)

    # ── Speech rate / disfluencies ─────────────────────────────────────────

    def _estimate_syllables(self, word: str) -> int:
        word = word.lower().strip()

        # Remove punctuation / non-letters
        word = re.sub(r"[^a-z]", "", word)

        if not word:
            return 0

        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels

            if is_vowel and not previous_was_vowel:
                syllable_count += 1

            previous_was_vowel = is_vowel

        # Silent ending "e", example: make, time, code
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        # Every spoken word should count as at least 1 syllable
        return max(syllable_count, 1)

    def _compute_speech_rate(self, word_timestamps) -> float:
        if word_timestamps is None or len(word_timestamps) == 0:
            return 0.0

        first_word_start = word_timestamps[0].get("start", 0.0)
        last_word_end = word_timestamps[-1].get("end", 0.0)

        speaking_duration = last_word_end - first_word_start

        if speaking_duration <= 0:
            return 0.0

        total_syllables = 0

        for item in word_timestamps:
            word = item.get("word", "")
            total_syllables += self._estimate_syllables(word)

        speech_rate = total_syllables / speaking_duration

        return float(speech_rate)

    def _compute_disfluencies(self, word_timestamps) -> dict[str, int | list[float]]:
        if word_timestamps is None or len(word_timestamps) == 0:
            return {
                "disfluency_count": 0,
                "disfluency_timestamps": []
            }

        filler = {"um", "uh", "erm", "hmm", "ah", "like"}
        count = 0
        timestamps = []
        for item in word_timestamps:
            word = item.get("word", "")
            word = word.lower().strip()
            word = re.sub(r"[^a-z]", "", word)
            if word in filler:
                count += 1
                timestamps.append(float(item.get("start", 0.0)))

        return {
            "disfluency_count": count,
            "disfluency_timestamps": timestamps
        }

    def _normalize_response_latency(self, response_latency_ms) -> float:
        if response_latency_ms is None:
            return 0.0

        try:
            response_latency_ms = float(response_latency_ms)
        except (TypeError, ValueError):
            return 0.0

        if response_latency_ms < 0:
            return 0.0

        return float(response_latency_ms)

    def _compute_wavlm_embedding(self, audio_arr: np.ndarray, sr: int) -> list[float]:
        """Extracts 768-dim temporal pooled self-supervised embeddings using WavLM."""
        try:
            import torch
            import transformers
        except ImportError:
            return [0.0] * 768

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if getattr(self, "_wavlm_model", None) is None:
                model_name = "microsoft/wavlm-base-plus"
                self._wavlm_extractor = transformers.AutoFeatureExtractor.from_pretrained(model_name)
                self._wavlm_model = transformers.AutoModel.from_pretrained(model_name, use_safetensors=True).to(device)
                self._wavlm_model.eval()

            if sr != 16000:
                audio_arr = librosa.resample(audio_arr, orig_sr=sr, target_sr=16000)
                target_sr = 16000
            else:
                target_sr = sr

            inputs = self._wavlm_extractor(audio_arr, return_tensors="pt", sampling_rate=target_sr).input_values.to(self._wavlm_model.device)
            with torch.no_grad():
                out = self._wavlm_model(inputs).last_hidden_state
                pooled = out.mean(dim=1).squeeze(0).cpu()
                return [float(x) for x in pooled.tolist()]
        except Exception as exc:
            print(f"[!] WavLM embedding extraction failed: {exc}")
            return [0.0] * 768