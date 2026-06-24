import re

import librosa
import numpy as np
import opensmile

# Use your actual settings names here.
# According to ARIA guide:
# AUDIO_SAMPLE_RATE = 16000
# MFCC_COEFFICIENTS = 13
from config.settings import AUDIO_SAMPLE_RATE, MFCC_COEFFICIENTS


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

    def extract(self, audio_clip, turn_id, candidate_id, response_latency_ms=None):
        pass

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

    def _compute_pitch_features(self, audio_arr, sample_rate) -> dict[str, float]:
        if len(audio_arr) < int(0.1 * sample_rate) or np.max(np.abs(audio_arr)) < 1e-6:
            return {
                "pitch_mean": 0.0,
                "pitch_variance": 0.0,
                "pitch_range": 0.0,
            }

        lld_df = self.smile_lld.process_signal(audio_arr, sample_rate)

        pitch_col = "F0semitoneFrom27.5Hz_sma3nz"

        if lld_df.empty or pitch_col not in lld_df.columns:
            return {
                "pitch_mean": 0.0,
                "pitch_variance": 0.0,
                "pitch_range": 0.0,
            }

        pitch_values = lld_df[pitch_col].to_numpy(dtype=np.float32)
        valid_pitch = pitch_values[np.isfinite(pitch_values)]

        if len(valid_pitch) == 0:
            return {
                "pitch_mean": 0.0,
                "pitch_variance": 0.0,
                "pitch_range": 0.0,
            }

        mean = float(np.mean(valid_pitch))
        variance = float(np.var(valid_pitch))
        range_val = float(np.max(valid_pitch) - np.min(valid_pitch))

        return {
            "pitch_mean": mean,
            "pitch_variance": variance,
            "pitch_range": range_val,
        }

    def _compute_energy(self, audio_arr, sample_rate) -> float:
        if len(audio_arr) == 0:
            return 0.0

        if np.max(np.abs(audio_arr)) < 1e-6:
            return 0.0

        lld_df = self.smile_lld.process_signal(audio_arr, sample_rate)

        energy_col = "Loudness_sma3"

        if lld_df.empty or energy_col not in lld_df.columns:
            return 0.0

        energy_values = lld_df[energy_col].to_numpy(dtype=np.float32)
        valid_energy = energy_values[np.isfinite(energy_values)]

        if len(valid_energy) == 0:
            return 0.0

        return float(np.mean(valid_energy))

    def _compute_mfcc(self, audio_arr, sample_rate, n_mfcc=13) -> list[float]:
        if len(audio_arr) < int(0.1 * sample_rate) or np.max(np.abs(audio_arr)) < 1e-6:
            return [0.0] * n_mfcc

        func_df = self.smile_functionals.process_signal(audio_arr, sample_rate)

        if func_df.empty:
            return [0.0] * n_mfcc

        features = func_df.iloc[0].to_dict()
        mfcc_vector = []

        for i in range(1, n_mfcc + 1):
            pattern = re.compile(rf"^mfcc{i}.*_amean$")

            value = 0.0

            for feature_name, feature_value in features.items():
                if pattern.match(feature_name):
                    value = float(feature_value)
                    break

            mfcc_vector.append(value)

        return mfcc_vector

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
    def _detect_pause_intervals(self,intervals,sample_rate,min_pause_ms=250)->dict[str,int |float]:
        pass