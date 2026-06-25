import re

import librosa
import numpy as np
import opensmile

# Use your actual settings names here.
# According to ARIA guide:
# AUDIO_SAMPLE_RATE = 16000
# MFCC_COEFFICIENTS = 13
from config.settings import AUDIO_SAMPLE_RATE, MFCC_COEFFICIENTS

# openSMILE eGeMAPS F0 is semitones relative to this reference frequency
F0_REFERENCE_HZ = 27.5


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

    def extract(self, audio_clip, word_timestamps=None, response_latency_ms=None):
        audio_arr=self._validate_audio(audio_clip)
        
        duration=self._compute_duration(audio_arr,AUDIO_SAMPLE_RATE)
        
        pitch_features=self._compute_pitch_features(audio_arr,AUDIO_SAMPLE_RATE)
        energy_mean=self._compute_energy(audio_arr,AUDIO_SAMPLE_RATE)
        mfcc_vector=self._compute_mfcc(audio_arr,AUDIO_SAMPLE_RATE)
        jitter=self._compute_jitter(audio_arr,AUDIO_SAMPLE_RATE)
        shimmer=self._compute_shimmer(audio_arr,AUDIO_SAMPLE_RATE)
        
        intervals=self._detect_speech_intervals(audio_arr)
        
        pause_features=self._compute_pause_features(intervals,AUDIO_SAMPLE_RATE)
        speech_to_silence_ratio=self._compute_speech_to_silence_ratio(intervals,duration,AUDIO_SAMPLE_RATE)
        
        speech_rate=self._compute_speech_rate(word_timestamps)
        disfluency_features=self._compute_disfluencies(word_timestamps)
        
        latency=self._normalize_response_latency(response_latency_ms)
        
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
    "speech_to_silence_ratio": speech_to_silence_ratio,
}
        
        

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

    def _compute_mfcc(self, audio_arr, sample_rate, n_mfcc=None) -> list[float]:
        """13 MFCC coefficients via librosa (eGeMAPS does not include MFCCs)."""
        n_mfcc = n_mfcc or MFCC_COEFFICIENTS
        if len(audio_arr) < int(0.1 * sample_rate) or np.max(np.abs(audio_arr)) < 1e-6:
            return [0.0] * n_mfcc

        mfcc_mat = librosa.feature.mfcc(y=audio_arr, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfcc_mat, axis=1).astype(float).tolist()

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
    def _compute_pause_features(self,intervals,sample_rate,min_pause_ms=250)->dict[str,int |float]:
        if len(intervals)==0 or len(intervals)==1:
            return {
                "pause_count":0,
                "pause_total_duration_ms":0.0    
            }
        count=0
        tot_duration=0
        for i in range(1,len(intervals)):
            prev=intervals[i-1][1]
            curr=intervals[i][0]
            diff=curr-prev
            gap_ms=diff/sample_rate*1000
            if gap_ms>min_pause_ms:
                count+=1
                tot_duration+=gap_ms
        
        return {
            "pause_count":int(count),
            "pause_total_duration_ms":float(tot_duration)
        }
    def _compute_speech_to_silence_ratio(self,intervals,total_duration,sample_rate)->float:
        if intervals is None or len(intervals)==0:
            return 0.0
        speech_duration=0.0
        for interval in intervals:
            start=interval[0]
            end=interval[1]
            interval_duration=(end-start)/sample_rate
            speech_duration+=interval_duration
        silence_duration=total_duration-speech_duration
        silence_duration=max(silence_duration,0.0)
        if silence_duration<1e-4:
            return speech_duration
        ratio=speech_duration/silence_duration
        return float(ratio)
    
    def _compute_jitter(self,audio_arr,sample_rate)->float:
        if audio_arr is None or len(audio_arr)==0:
            return 0.0
        if np.max(np.abs(audio_arr)) < 1e-6:
            return 0.0
        df=self.smile_functionals.process_signal(audio_arr,sample_rate)
        
        if df is None or df.empty:
            return 0.0
        features=df.iloc[0].to_dict()
        jitter=features.get("jitterLocal_sma3nz_amean", 0.0)
            
        return float(jitter)
    
    def _compute_shimmer(self,audio_arr,sample_rate)->float:
        if audio_arr is None or len(audio_arr)==0:
            return 0.0
        if np.max(np.abs(audio_arr)) < 1e-6:
            return 0.0
        df=self.smile_functionals.process_signal(audio_arr,sample_rate)
        
        if df is None or df.empty:
            return 0.0
        features=df.iloc[0].to_dict()
        shimmer=features.get("shimmerLocaldB_sma3nz_amean", 0.0)
            
        return float(shimmer)
    
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
    
    
    def _compute_disfluencies(self,word_timestamps)->dict[str, int | list[float]]:
        if word_timestamps is None or len(word_timestamps)==0:
            return {
            "disfluency_count":0,
            "disfluency_timestamps":[]
        }
        
        filler={"um","uh","erm","hmm","ah","like"}
        count=0
        timestamps=[]
        for item in word_timestamps:
            word=item.get("word","")
            word=word.lower().strip()
            word = re.sub(r"[^a-z]", "", word)
            if word in filler:
                count+=1
                timestamps.append(float(item.get("start",0.0)))
        
        return {
            "disfluency_count":count,
            "disfluency_timestamps":timestamps
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