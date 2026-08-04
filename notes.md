# ARIA — Development Notes

> Living documentation for the ARIA project. All code changes, architecture decisions, and module internals are recorded here so future coding assistants (and team members) can pick up context without re-reading the entire codebase.

**Project:** Autonomous Reinforcement-based Interview Agent with Multimodal Adaptive Assessment  
**Reference:** [ARIA_Coding_Assistant_Guide.md](./ARIA_Coding_Assistant_Guide.md)  
**Last updated:** 2026-06-29 (Honest Benchmarking Suite, Zero-Leakage Evaluation, Actor-Independent RAVDESS Split)

---

## Table of Contents

1. [Project Status](#project-status)
2. [Environment & Setup](#environment--setup)
3. [Folder Structure (Current)](#folder-structure-current)
4. [Module 1 — Speech-to-Text (STT)](#module-1--speech-to-text-stt)
5. [Module 2 — Vision](#module-2--vision)
6. [Module 3 — Prosody](#module-3--prosody)
7. [Module 4 — Dynamic Multimodal Fusion](#module-4--dynamic-multimodal-fusion)
8. [Change Log](#change-log)
9. [Next Steps](#next-steps)

---

## Project Status

| Module | Status | Owner | Notes |
|--------|--------|-------|-------|
| Module 1 — STT | **Implemented & Verified** | Krissh | Path security verified, latency calculation bug fixed |
| Module 2 — Vision | **Implemented & Verified** | Krissh | Migrated to MediaPipe Tasks API (`FaceLandmarker`), non-lossy emotion distributions |
| Module 3 — Prosody | **Implemented & Verified** | Krissh | Cached openSMILE calls (2x speedup), thread-safe baseline locking |
| Module 4 — Fusion | **Implemented & Verified** | Krissh | Dynamic attention gating, per-modality dissonance penalization, concatenation baseline & benchmarking sandbox |
| Module 10 — Cognitive Load | **Implemented & Verified** | Krissh | Rule-based 4-quadrant classifier (low/anxiety/ignorance/confident_ignorance), composite distress scoring |
| Module 11 — Anti-Gaming | **Implemented & Verified** | Krissh | 3 parallel detectors (gaze, latency, semantic) + AntiGamingMonitor orchestrator |
| config/settings.py | **Done** | — | Global constants |
| Backend / Frontend | Not started | Raghav | — |

---

## Environment & Setup

### Requirements

```
OS:       Windows 11
Python:   3.11.9 (strict — not 3.12/3.13)
GPU:      NVIDIA RTX 4060 — 8 GB VRAM
CUDA:     13.0
Venv:     .venv (activate before every session)
```

### Activate virtual environment

```powershell
.venv\Scripts\activate
```

### Install dependencies (Module 1 & 2)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/Ahmednull/L2CS-Net.git
```

### L2CS-Net weights (Module 2 gaze)

Download `L2CSNet_gaze360.pkl` from the [official L2CS-Net repo](https://github.com/Ahmednull/L2CS-Net) and place at:

```
models/L2CSNet_gaze360.pkl
```

Or set in `.env`:

```
L2CS_WEIGHTS_PATH=C:/path/to/L2CSNet_gaze360.pkl
```

If weights are missing, gaze falls back to head-pose estimation from MediaPipe (less accurate but keeps the pipeline runnable).

### GPU memory budget

| Component | VRAM |
|-----------|------|
| faster-whisper large-v3 (int8) | ~2.5 GB |
| MediaPipe + DeepFace | ~1.5 GB |
| L2CS-Net | ~0.5 GB |
| Headroom | ~3.0 GB |

**Rule:** Models are lazy-loaded once at first use — never reload per frame or per transcription call.

---

## Folder Structure (Current)

```
ARIA/
├── ARIA_Coding_Assistant_Guide.md   # Master spec (do not deviate)
├── ARIA_Benchmark_Diagnosis.md      # Diagnosis report for empirical metrics
├── notes.md                         # This file
├── requirements.txt
├── pytest.ini                       # Integration test marker
├── .gitignore
├── config/
│   ├── __init__.py
│   └── settings.py                  # Global constants — import everywhere
├── models/
│   └── face_landmarker.task         # MediaPipe Tasks face landmark model
├── modules/
│   ├── module_01_stt/
│   │   ├── __init__.py
│   │   ├── semantic_grader.py       # TF-IDF N-Gram & keyword rubric semantic scoring
│   │   └── transcriber.py           # faster-whisper offline transcription
│   ├── module_02_vision/
│   │   ├── __init__.py
│   │   ├── baseline.py              # VisionBaselineManager (resting AU calibration)
│   │   ├── emotion.py               # DeepFace emotion analysis & remapping
│   │   ├── face_mesh.py             # MediaPipe Tasks landmarks & AU activations
│   │   ├── gaze.py                  # L2CS-Net gaze estimation
│   │   ├── temporal_au.py           # Temporal AU sequence tracking & velocity
│   │   └── vision_processor.py      # Orchestrator (parallel threads + turn summary)
│   ├── module_03_prosody/           # Prosody feature extraction & baseline
│   │   ├── __init__.py
│   │   ├── baseline.py              # ProsodyBaselineManager
│   │   ├── extractor.py             # ProsodyExtractor (openSMILE + librosa)
│   │   └── pipeline.py              # process_prosody_turn() orchestrator
│   ├── module_04_fusion/            # Multimodal Dynamic Fusion Engine
│   │   ├── __init__.py
│   │   ├── attention_fusion.py      # Dynamic softmax cross-modal attention gating
│   │   ├── concat_fusion.py         # Unweighted concatenation baseline
│   │   ├── fusion_engine.py         # Main MultimodalFusionEngine wrapper
│   │   ├── normalizer.py            # Deque historical imputation & normalization
│   │   └── schema.py                # FULL_FEATURE_SCHEMA definition
│   ├── module_10_cognitive_load/    # Cognitive Load Separator
│   │   ├── __init__.py
│   │   └── classifier.py           # 4-quadrant cognitive load classifier
│   └── module_11_anti_gaming/       # Anti-Gaming & Integrity Monitor
│       ├── __init__.py              # AntiGamingMonitor orchestrator
│       ├── gaze_scanner.py          # Note reading detection via gaze sweeps
│       ├── latency_checker.py       # AI assistance detection via latency + delivery
│       └── semantic_checker.py      # Coaching/scripting via TF-IDF + complexity
└── tests/
    ├── conftest.py
    ├── test_anti_gaming.py          # Module 11 unit tests (3 detectors + orchestrator)
    ├── test_cognitive_load.py       # Module 10 unit tests (4 quadrants + edge cases)
    ├── test_fusion.py               # Module 4 unit tests (gating, dissonance, masking)
    ├── test_prosody.py              # Module 3 unit tests
    ├── test_stt.py                  # Module 1 unit tests
    ├── test_vision.py               # Module 2 unit tests
    └── benchmarks/
        └── run_baseline_benchmarks.py # Honest zero-leakage evaluation suite
```

---

## Module 1 — Speech-to-Text (STT)

### Purpose

Convert the candidate's spoken answer into text with word-level timestamps, language detection, confidence, and response latency. This is the **text modality** input for the POMDP observation space.

### Files

| File | Role |
|------|------|
| `modules/module_01_stt/transcriber.py` | Core transcription logic |
| `modules/module_01_stt/semantic_grader.py` | Automated semantic grader (TF-IDF N-Gram & keyword rubric scoring) |
| `modules/module_01_stt/__init__.py` | Public exports: `Transcriber`, `transcribe`, `transcribe_file` |
| `config/settings.py` | `MODEL_WHISPER`, `WHISPER_COMPUTE_TYPE`, `DEVICE`, `AUDIO_SAMPLE_RATE` |

### Extension — Automated Semantic Grader (`semantic_grader.py`)

Evaluates candidate speech transcripts against reference rubrics and answer keys using TF-IDF N-gram (1–2) cosine similarity (70% weight) and regex word-boundary keyword coverage (30% weight).
- **Class:** `SemanticGrader` (`grade_response(candidate_transcript, reference_answer, required_keywords)`)
- **Key design decisions:** Instantiates a fresh `TfidfVectorizer` per call to avoid shared mutable state across threads (Fix C2); uses regex word-boundary matching `\b` (Fix M4); includes a `D` grade band (0.15–0.35) to prevent harsh C → F transitions (Fix H4).

### Tooling

- **Library:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **Model:** Whisper `large-v3`
- **Device:** CUDA with `compute_type="int8"` (VRAM budget)
- **VAD:** `vad_filter=True` prevents hallucination on silence

### Input contract

```python
audio: np.ndarray  # shape (N,), float32, 16 kHz mono
question_end_time: float | None  # optional Unix timestamp for latency calc
```

### Output contract (exact schema)

```python
{
    "transcript": str,              # full text of candidate answer
    "word_timestamps": [            # word-level timing
        {"word": str, "start": float, "end": float}
    ],
    "language": str,                # e.g. "en"
    "confidence": float,            # 0.0–1.0, from segment avg_logprob
    "response_latency_ms": float    # ms from question end to first word
}
```

### How it works (step by step)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Audio input    │────▶│  WhisperModel    │────▶│  Segment iterator   │
│  float32 16kHz  │     │  large-v3 int8   │     │  vad_filter=True    │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                            │
                    ┌───────────────────────────────────────┘
                    ▼
         ┌──────────────────────┐
         │  For each segment:   │
         │  - append text       │
         │  - collect words   │
         │  - avg logprob→conf  │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  response_latency_ms │
         │  = first_word.start  │
         │    × 1000            │
         └──────────────────────┘
```

1. **Model loading:** `Transcriber.__init__` creates a single `WhisperModel` instance. A module-level singleton (`_get_transcriber()`) ensures the model is loaded once per process.

2. **Transcription:** `transcribe_sync()` passes the numpy array directly to `model.transcribe()` with `word_timestamps=True`.

3. **Confidence:** Whisper returns `avg_logprob` per segment (typically −1 to 0). We map this to 0–1 via `1.0 + avg_logprob`, clamped.

4. **Response latency:** Time from when the question ended to when the candidate's first word starts. When only an audio clip is available (offline mode), latency = `word_timestamps[0]["start"] * 1000` ms relative to clip start.

5. **Async wrapper:** `transcribe()` and `transcribe_file()` use `asyncio.to_thread()` so the FastAPI event loop is not blocked during GPU inference.

### Public API

```python
from modules.module_01_stt import transcribe, transcribe_file, Transcriber

# Async — use in FastAPI routes
result = await transcribe(audio_array)

# Offline file mode
result = await transcribe_file("path/to/answer.wav")

# Sync — use in scripts / tests
transcriber = Transcriber()
result = transcriber.transcribe_sync(audio_array)
result = transcriber.transcribe_file_sync("path/to/answer.wav")
```

### Design decisions

| Decision | Rationale |
|----------|-----------|
| Singleton model | Avoid reloading ~2.5 GB model on every turn |
| int8 quantization | Fits RTX 4060 8 GB budget |
| VAD filter | Prevents Whisper hallucinating text during silence |
| Offline first | Streaming WebRTC integration comes in Phase 5 |
| `soundfile` + `librosa` resample | Accept any sample rate WAV, normalize to 16 kHz |

### Known limitations

- **Streaming mode not yet implemented** — currently processes complete audio buffers.
- **Tone-only test audio** produces empty transcripts — integration tests validate schema, not speech quality.
- **Latency with absolute timestamps** requires the backend to pass `question_end_time` once WebRTC timing is wired up.

### Tests

```powershell
pytest tests/test_stt.py -v
pytest tests/test_stt.py -v -m integration  # requires faster-whisper + GPU
```

---

## Module 2 — Vision

### Purpose

Analyze webcam frames to extract facial landmarks, Action Unit activations, interview-context emotion, gaze direction, eye contact, head pose, and blink patterns. Frame-level outputs are aggregated into a **per-turn summary** consumed by Module 4 (Fusion) and downstream RL state.

### Files

| File | Role |
|------|------|
| `face_mesh.py` | MediaPipe 468 landmarks, AU estimation, head pose, blink detection |
| `emotion.py` | DeepFace emotion → interview-context label remapping |
| `gaze.py` | L2CS-Net gaze (yaw/pitch) + eye contact score |
| `baseline.py` | Thread-safe `VisionBaselineManager` for dynamic candidate resting calibration |
| `temporal_au.py` | Temporal AU sequence tracking (onset velocity, variance, dynamic micro-expressions) |
| `vision_processor.py` | Parallel orchestration + per-turn summarization with dynamic deviation tracking |
| `__init__.py` | Exports `VisionProcessor` and `VisionBaselineManager` |

### Input contract

```python
frame: np.ndarray  # BGR, shape (H, W, 3), from OpenCV / WebRTC
timestamp_ms: float  # optional, for blink timing
```

Frames arrive at **2 fps** (every 500 ms) from WebRTC per the spec.

### Per-frame output contract

```python
{
    "landmarks": np.ndarray,        # (468, 3) — x, y, z
    "au_activations": {
        "AU1": float, "AU2": float, "AU4": float, "AU6": float,
        "AU12": float, "AU15": float, "AU17": float, "AU23": float, "AU25": float
    },
    "emotion_label": str,           # engaged/confused/nervous/confident/blank
    "emotion_confidence": float,
    "gaze_vector": {"yaw": float, "pitch": float},
    "eye_contact_score": float,     # 0.0 = away, 1.0 = direct
    "head_pose": {"roll": float, "pitch": float, "yaw": float},
    "blink_detected": bool,
    "blink_duration_ms": float
}
```

Returns **`None`** if no face detected — caller skips that frame in turn summary.

### Per-turn summary contract

```python
{
    "emotion_label": str,           # mode (most frequent) across frames
    "au_activations": dict,         # mean of each AU
    "gaze_vector": {"yaw": float, "pitch": float},  # mean
    "eye_contact_score": float,     # mean
    "head_pose": {"roll", "pitch", "yaw"},          # mean
    "blink_rate": float             # blinks per minute
}
```

### Architecture

```
                    ┌─────────────────────────────────────┐
                    │         VisionProcessor             │
                    │  process_frame(frame, timestamp_ms) │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │ ThreadPoolExecutor (max_workers=3)            │
              ▼                       ▼                       ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │ FaceMeshAnalyzer│   │ EmotionAnalyzer │   │  GazeEstimator  │
    │   (MediaPipe)   │   │   (DeepFace)    │   │   (L2CS-Net)    │
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                       │
             │ landmarks           │ emotion_label         │ gaze_vector
             │ au_activations      │ emotion_confidence    │ eye_contact_score
             │ head_pose           │                       │
             │ blink_*             │                       │
             └─────────────────────┴───────────────────────┘
                                      │
                                      ▼
                           Merge into frame dict
                                      │
                           (stored in _turn_frames)
                                      │
                                      ▼
                           summarize_turn(candidate_id, turn_id)
                                      │
                                      ▼
                        VisionBaselineManager calculates deviations
                        (au_deviations, eye_contact_deviation)
```

### Sub-module details

#### face_mesh.py — MediaPipe Tasks FaceLandmarker

**What it does:**

1. Converts BGR → RGB and runs modern MediaPipe Tasks `FaceLandmarker` (`models/face_landmarker.task`, max 1 face). This resolves legacy protobuf conflicts with TensorFlow/DeepFace.
2. Extracts 468 landmarks as `(x, y, z)` pixel coordinates.
3. Estimates 9 Action Units geometrically from landmark distances (normalized by interocular distance).
4. Computes head pose (roll, pitch, yaw in degrees) via `cv2.solvePnP` with a canonical 3D face model.
5. Detects blinks using Eye Aspect Ratio (EAR): when EAR drops below `EAR_BLINK_THRESHOLD` (0.21) and recovers, a blink is recorded with duration in ms. State resets between turns via `reset_state()`.

**AU estimation approach:**

MediaPipe does not output FACS Action Units natively. We derive approximate 0–1 activations from geometry:

| AU | FACS meaning | Geometric proxy |
|----|--------------|-----------------|
| AU1 | Inner brow raise | Inner brow ↔ upper lid distance |
| AU2 | Outer brow raise | Outer brow ↔ upper lid distance |
| AU4 | Brow lowerer | Inverse of brow ↔ eye proximity |
| AU6 | Cheek raiser | Cheek ↔ upper lid distance |
| AU12 | Lip corner puller (smile) | Mouth width + corner lift |
| AU15 | Lip corner depressor | Inverse of corner lift |
| AU17 | Chin raiser | Chin ↔ lower lip distance |
| AU23 | Lip tightener | Inverse of lip separation |
| AU25 | Lips part | Lip separation |

These are **relative** features for within-interview change detection, not certified FACS intensities.

#### emotion.py — DeepFace + label remapping

**What it does:**

1. Calls `DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)`.
2. Takes the dominant generic emotion and maps it to interview-context labels:

| DeepFace label | ARIA label |
|----------------|------------|
| happy | engaged |
| neutral | blank (or confident if high confidence) |
| fear | nervous |
| surprise | confused |
| sad, disgust, angry | nervous |

3. Valid output labels: `engaged`, `confused`, `nervous`, `confident`, `blank`.

**Pin:** `deepface==0.0.79` — newer versions change output schema (see guide Section 12).

#### gaze.py — L2CS-Net

**What it does:**

1. Lazy-loads L2CS-Net `Pipeline` with ResNet50 backbone and Gaze360 weights.
2. Runs `pipeline.step(frame)` → pitch/yaw in radians → converted to degrees.
3. Computes `eye_contact_score` from gaze angles:

```
score = 1.0 - 0.6 × min(|yaw|/30, 1) - 0.4 × min(|pitch|/25, 1)
```

4. **Fallback:** If weights file missing, uses head pose yaw/pitch from `face_mesh.py` as a proxy.

#### temporal_au.py — Temporal AU Sequence Tracking

**What it does:**
1. Evaluates frame-by-frame Action Unit time-series across an interview turn ($T \times 15$ matrix) using `TemporalAUTracker`.
2. Computes temporal velocity (first derivative across frames) and variance across time for 15 key AUs (`brow_inner_up`, `brow_lower`, `lip_corner_pull`, `lip_press`, etc.).
3. Classifies dynamic micro-expressions into temporal emotion predictions (`blank`, `confused`, `nervous`, `confident`, `engaged`) with evidence-based confidence scaling (Fix H2).
4. Resolves still-frame `BLANK` false-negatives by tracking facial onset velocities over time.

#### vision_processor.py — Orchestrator

**What it does:**

1. **`start_turn(timestamp_ms)`** — clears frame buffer for a new Q&A turn.
2. **`process_frame(frame, timestamp_ms)`** — runs mesh, emotion, and gaze in parallel threads; merges results; appends to `_turn_frames`.
3. **`summarize_turn(turn_duration_ms)`** — aggregates:
   - `emotion_label`: mode (most frequent)
   - `au_activations`: element-wise mean
   - `gaze_vector`, `eye_contact_score`, `head_pose`: mean
   - `blink_rate`: `(blink_count / turn_duration_minutes)`

### Public API

```python
from modules.module_02_vision import VisionProcessor

processor = VisionProcessor()
processor.start_turn(timestamp_ms=0.0)

# Called every 500 ms during candidate answer
frame_result = processor.process_frame(bgr_frame, timestamp_ms=500.0)
# frame_result is None if no face — skip in aggregation

# At turn end
vision_summary = processor.summarize_turn(turn_duration_ms=15000.0)
```

### Design decisions

| Decision | Rationale |
|----------|-----------|
| Parallel threads per frame | Mesh, DeepFace, L2CS are independent — reduces per-frame latency |
| Singleton analyzers | Models loaded once, not per frame (guide Section 13) |
| Return None on no face | Avoid polluting turn summary with zeros |
| Geometric AUs | No extra AU model — keeps VRAM budget; sufficient for relative signals |
| L2CS fallback | Pipeline remains testable without manual weight download |
| Mode emotion for turn | Robust to single-frame misclassification at 2 fps |

### Known limitations

- **AU values are geometric approximations**, not OpenFace/BP4D ground truth.
- **DeepFace is slow** (~100–300 ms per frame on GPU) — may need frame skipping under load.
- **L2CS requires manual weight download** — not pip-installable as weights.
- **Blank test frames return None** — integration test skips face validation on synthetic black images.

### Tests

```powershell
pytest tests/test_vision.py -v
pytest tests/test_vision.py -v -m integration  # requires mediapipe, deepface
```

---

## Module 3 — Prosody

### Purpose

Extract speech prosody features from a candidate's turn audio — pitch, energy, MFCCs, pauses, disfluencies, speech rate, jitter, shimmer, and personal-baseline deviations. Output feeds Module 4 (Fusion) and Module 10 (Cognitive Load).

### Files

| File | Role | Status |
|------|------|--------|
| `modules/module_03_prosody/extractor.py` | `ProsodyExtractor` — raw audio → prosody dict (openSMILE + librosa + WavLM) | **Done** |
| `modules/module_03_prosody/baseline.py` | `ProsodyBaselineManager` — baseline store + deviations | **Done** |
| `modules/module_03_prosody/pipeline.py` | `process_prosody_turn()` — wires extractor + baseline | **Done** |
| `modules/module_03_prosody/__init__.py` | Public exports: `ProsodyExtractor`, `ProsodyBaselineManager`, `process_prosody_turn` | **Done** |
| `tests/test_prosody.py` | Unit + integration tests | **Done** |

> **Naming note:** Folder is correctly named `module_03_prosody/`, matching guide specification. All imports in `pipeline.py` and `__init__.py` match this path.

### Tooling

| Feature | Implementation |
|---------|----------------|
| Pitch, jitter, shimmer, loudness | **openSMILE** eGeMAPSv02 (LLD + Functionals) |
| Self-supervised embeddings | **HuggingFace transformers** `microsoft/wavlm-base-plus` (768-dim temporal average-pooled) |
| Pause detection, speech intervals | **librosa** `effects.split` |
| Speech rate, disfluencies | Heuristic syllable count + STT `word_timestamps` |
| Baseline calibration | In-memory `ProsodyBaselineManager` |
| Config | `AUDIO_SAMPLE_RATE`, `MFCC_COEFFICIENTS` from `config/settings.py` |

SpeechBrain is listed in the guide but **not used** — openSMILE covers jitter/shimmer/pitch. Acceptable deviation from guide tooling.

### Input contract

**Guide minimum:**

```python
audio_clip: np.ndarray  # 16 kHz mono float32
turn_id: int
candidate_id: str
```

**`ProsodyExtractor.extract()` — audio features only:**

```python
audio_clip: np.ndarray
word_timestamps: list | None   # from Module 1 STT
response_latency_ms: float | None
```

**`process_prosody_turn()` — full turn with baseline:**

```python
audio_clip: np.ndarray
turn_id: int
candidate_id: str
word_timestamps: list | None
response_latency_ms: float | None
```

### Output contract

**Raw output from `ProsodyExtractor.extract()`** (no deviation fields):

```python
{
    "pitch_mean": float,
    "pitch_variance": float,
    "pitch_range": float,
    "speech_rate": float,
    "pause_count": int,
    "pause_total_duration_ms": float,
    "disfluency_count": int,
    "disfluency_timestamps": [float],
    "response_latency_ms": float,
    "energy_mean": float,
    "jitter": float,
    "shimmer": float,
    "mfcc_vector": list,            # length 13
    "wavlm_embedding": list,        # length 768
    "speech_to_silence_ratio": float,
}
```

**Full output from `process_prosody_turn()` or `baseline.update_with_baseline()`:**

Adds deviation fields per guide:

```python
{
    # ... all raw fields above ...
    "pitch_deviation": float | None,   # None on turns 1–2
    "rate_deviation": float | None,
    "energy_deviation": float | None,
}
```

### Architecture

```
Module 1 (STT)
  word_timestamps ──┐
  response_latency ─┤
                    ▼
audio_clip ──▶ ProsodyExtractor.extract() ──▶ raw prosody dict
                    │                              │
                    │ openSMILE eGeMAPSv02         │
                    │ librosa VAD/split            │
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ProsodyBaselineManager.update_with_baseline()
                      turn_id <= 2 → store baseline, deviations = None
                      turn_id >  2 → deviation = (current - baseline) / baseline
                                   │
                                   ▼
                           final prosody dict → turn_signal["prosody"]
```

**Recommended entry point for backend:**

```python
from modules.module_3_prosody.pipeline import process_prosody_turn

prosody = process_prosody_turn(
    audio_clip=turn_audio,
    turn_id=turn_id,
    candidate_id=candidate_id,
    word_timestamps=stt_result["word_timestamps"],
    response_latency_ms=stt_result["response_latency_ms"],
)
```

### `ProsodyExtractor` — method reference

| Method | Source | Output |
|--------|--------|--------|
| `_validate_audio` | numpy | Normalized float32 mono array |
| `_compute_pitch_features` | openSMILE LLD → semitones → **Hz** | pitch mean/var/range |
| `_compute_energy` | openSMILE LLD `Loudness_sma3` | mean loudness |
| `_compute_mfcc` | librosa `feature.mfcc` (13 coeffs, time-averaged) | 13-element list |
| `_compute_jitter` | Functionals `jitterLocal_sma3nz_amean` | float |
| `_compute_shimmer` | Functionals `shimmerLocaldB_sma3nz_amean` | float |
| `_detect_speech_intervals` | librosa `effects.split(top_db=30)` | `(N, 2)` sample intervals |
| `_compute_pause_features` | gaps between intervals > 250 ms | pause_count, pause_total_duration_ms |
| `_compute_speech_to_silence_ratio` | speech duration / silence duration | float |
| `_compute_speech_rate` | syllables / speaking duration from word_timestamps | syllables/sec |
| `_compute_disfluencies` | filler words um/uh/erm/hmm/ah/like | count + timestamps |
| `_normalize_response_latency` | passthrough with validation | float ≥ 0 |

### `ProsodyBaselineManager` — baseline logic

Matches guide spec:

```
Turn 1–2 (turn_id <= baseline_turns):
  → append {pitch_mean, speech_rate, energy_mean} to baseline_turns[]
  → after 2 turns: compute mean baseline across both turns
  → pitch_deviation, rate_deviation, energy_deviation = None

Turn 3+:
  → deviation = (current - baseline) / baseline  (safe: returns 0.0 if baseline ≈ 0)
```

State stored in memory: `self.baselines[candidate_id]`.

### Guide compliance checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| All output schema keys present | ✅ | Via extractor + baseline pipeline |
| Pauses > 250 ms | ✅ | Strict `>` threshold |
| Disfluency fillers um/uh/erm/like | ✅ | Also detects hmm, ah |
| Baseline turns 1–2, deviation turn 3+ | ✅ | `ProsodyBaselineManager` |
| openSMILE | ✅ | eGeMAPSv02 |
| SpeechBrain | ❌ | Not used — openSMILE sufficient |
| pitch_mean in Hz | ✅ | Semitones converted via `27.5 * 2^(st/12)` |
| energy_mean as RMS | ⚠️ | Returns openSMILE loudness (sone) — optional to change |
| mfcc_vector (13 coeffs) | ✅ | librosa MFCC (eGeMAPS has no MFCCs) |
| wavlm_embedding (768 dims) | ✅ | HuggingFace `microsoft/wavlm-base-plus` self-supervised embeddings |
| Folder name `module_03_prosody` | ✅ | Correctly named `module_03_prosody/` |
| `extract()` API | ✅ | Audio-only; `turn_id`/`candidate_id` on pipeline/baseline |

### Suggested improvements (optional — code works without these)

1. ~~**Convert pitch semitones → Hz**~~ — **Done**

2. ~~**Fix MFCC extraction**~~ — **Done** (librosa for MFCC; openSMILE for jitter/shimmer/pitch)

3. ~~**Cache openSMILE calls in `extract()`**~~ — **Done** (cached model instance across invocations for 2x speedup)

4. **Wire `BASELINE_TURNS` from settings** — optional; `pipeline.py` hardcodes `baseline_turns=2`

5. **Resample non-16 kHz audio** — optional if all input is guaranteed 16 kHz from STT/WebRTC

6. ~~**Rename folder** to `module_03_prosody`~~ — **Done** (folder and imports match `module_03_prosody/`)

7. ~~**Remove unused `turn_id`/`candidate_id` from `extract()`**~~ — **Done** (baseline params stay on `process_prosody_turn()`)

8. ~~**Add `__init__.py`**~~ — **Done** (`__init__.py` exports key classes and pipeline functions)

9. **Use RMS for `energy_mean`** — optional; current loudness (sone) works for baseline deviation

### Tests

```powershell
pytest tests/test_prosody.py -v                  # unit tests (mocked openSMILE)
pytest tests/test_prosody.py -v -m integration   # real openSMILE on 30s fixture
```

**Coverage:**

| Test group | What it validates |
|------------|-------------------|
| `TestProsodyBaselineManager` | Deviations None on turns 1–2; float on turn 3; safe division |
| `TestProsodyExtractorHelpers` | Validation, pauses, speech rate, disfluencies, latency |
| `TestProsodyExtractorExtract` | Full raw schema keys (mocked openSMILE) |
| `TestProsodyPipeline` | End-to-end extractor + baseline via `process_prosody_turn` |
| `TestProsodyIntegration` | 30s audio clip; mfcc length 13; baseline flow with real openSMILE |

---

## Module 4 — Dynamic Multimodal Fusion

Module 4 (`modules/module_04_fusion`) merges heterogeneous multimodal outputs from STT, Vision, and Prosody into a single non-lossy turn-level signal (`FUSED_VECTOR_DIM` = 72).

### Files

| File | Role |
|------|------|
| `modules/module_04_fusion/schema.py` | Defines `FULL_FEATURE_SCHEMA` (72-dim non-lossy feature order across text, vision, and prosody) |
| `modules/module_04_fusion/normalizer.py` | `FeatureNormalizer` — historical deque imputation (`history_size=5`) & soft non-linear scaling |
| `modules/module_04_fusion/attention_fusion.py` | `DynamicAttentionFusion` — gated softmax cross-modal attention & dissonance penalization |
| `modules/module_04_fusion/concat_fusion.py` | `ConcatFusion` / `ConcatenationFusionEngine` — unweighted uniform baseline ($1 / N_{active}$) |
| `modules/module_04_fusion/fusion_engine.py` | `MultimodalFusionEngine` — main orchestrator wrapping normalizer & attention fusion |
| `modules/module_04_fusion/__init__.py` | Public exports: `MultimodalFusionEngine`, `ConcatenationFusionEngine`, `FeatureNormalizer`, etc. |

### Schema Breakdown (`FULL_FEATURE_SCHEMA` = 72 dims)

The fused vector maintains a strict, non-lossy order across all three active modalities:
1. **Text Modality (11 dims):** STT confidence/latency/word count/duration (4), semantic scalar features (similarity, relevance, completeness, confidence) (4), and competency distribution probabilities (`beginner`, `mid`, `expert`) (3).
2. **Vision Modality (33 dims):** Vision/emotion confidence, eye contact, blink rate & deviation (5), emotion probability distribution (`blank`, `nervous`, `confused`, `engaged`, `confident`) (5), gaze yaw/pitch (2), head pose roll/pitch/yaw (3), AU activations for 9 AUs (9), and AU resting baseline deviations (9).
3. **Prosody Modality (28 dims):** Prosody scalar features (pitch mean/var/range, rate, pauses, disfluencies, latency, energy, jitter, shimmer, speech-to-silence ratio) (12), prosody baseline deviations (pitch, rate, energy) (3), and MFCC coefficients 1–13 (13).

### Architecture & Components

1. **Schema (`schema.py`)**: Defines fixed non-lossy order (`FULL_FEATURE_SCHEMA`) preserving competency distributions, emotion probability vectors, Action Unit (AU) activations/deviations, and acoustic MFCC/scalar features.
2. **Normalizer (`normalizer.py`)**: Converts raw module dictionaries into aligned numeric vectors. Employs per-candidate historical deques (`history_size=5`) to impute missing modality features using recent historical means rather than zero-filling. Uses soft non-linear compression (`math.tanh`, rational curves) for unbounded metrics.
3. **Dynamic Attention Fusion (`attention_fusion.py`)**: Gated softmax cross-modal attention fusion.
   - **Per-Modality Dissonance Penalization**: Calculates mean pairwise cosine distance between each active modality and other active modalities. Selective penalization lowers the weight of conflicting modalities without penalizing agreeing ones.
   - **Scale-Preserving Gating**: Gating multipliers are scaled by the number of active modalities ($g_m = w_m \times N_{active}$), ensuring features maintain average 100% scale regardless of how many modalities are present.
4. **Unweighted Concatenation Baseline (`concat_fusion.py`)**: Baseline engine (`ConcatenationFusionEngine`) providing uniform weights ($1 / N_{active}$) and zero dissonance for direct ablation evaluation against attention fusion.
5. **Honest Benchmarking Pipeline (`tests/benchmarks/run_baseline_benchmarks.py`)**: Automated evaluation suite enforcing strict zero-leakage evaluation (`y_true = ground-truth label only`, `y_pred = model inference output only`). Evaluates FER2013 via DeepFace blind inference (64.76% accuracy), RAVDESS via an actor-independent RandomForest classifier (47.22% accuracy on held-out Actors 19-24), and Mohler inter-annotator baseline (78.40% agreement). Includes mandatory engineering diagnostic verification checks.

---

## Change Log

### 2026-08-04 — Module 10 (Cognitive Load) & Module 11 (Anti-Gaming) Implementation

**Created:**
- **Module 10 — Cognitive Load Separator (`modules/module_10_cognitive_load/classifier.py`)**: Rule-based 4-quadrant classifier that separates `low`, `anxiety`, `ignorance`, and `confident_ignorance` cognitive states. Uses composite distress scoring from 7 weighted physiological signals (pitch/rate/energy deviation, disfluency rate, gaze instability, jitter, pause density) with sigmoid normalization and automatic weight redistribution for missing signals. Baseline-aware — defaults to `low` with low confidence on turns 1–2.
- **Module 11 — Anti-Gaming & Integrity Monitor**: Three parallel integrity detectors plus `AntiGamingMonitor` orchestrator:
  - **GazeScanner (`gaze_scanner.py`)**: Detects note reading via sustained horizontal gaze sweeps (>2s monotonic left-to-right yaw movement with constrained pitch variation).
  - **LatencyChecker (`latency_checker.py`)**: Detects AI assistance via suspicious response latency (>5s) combined with unnaturally uniform speech delivery (coefficient of variation analysis on local speech rate segments).
  - **SemanticChecker (`semantic_checker.py`)**: Detects coaching (lateral head turns >25° yaw) and scripted answers (cross-turn TF-IDF cosine similarity >0.65 + lexical complexity shifts >1.8x baseline). Uses lightweight TF-IDF (no BGE-M3 dependency).
- **Tests**: `tests/test_cognitive_load.py` (24 tests) and `tests/test_anti_gaming.py` (25 tests) — 49/49 passing.

**Design decisions:**
- All modules are rule-based (no ML models needed at this stage) — matches Phase 4 build order.
- No dependency on Raghav's modules (5, 6, 7, 8, 12, 13, 14). Fully standalone.
- Thread-safe and stateless per-call — compatible with FastAPI async concurrency.
- Output contracts match the `turn_signal` interface spec exactly.

### 2026-07-08 — Comprehensive Module 1–4 Documentation & Schema Alignment Audit

**Updated in `notes.md`:**
- **Section 3 (Folder Structure)**: Synchronized file tree with actual workspace contents across `module_01_stt`, `module_02_vision`, `module_03_prosody`, and `module_04_fusion`. Added `semantic_grader.py` (Module 1) and `temporal_au.py` (Module 2).
- **Section 4 (Module 1 — STT)**: Added `semantic_grader.py` to file reference table and added a dedicated subsection documenting TF-IDF N-Gram similarity and keyword rubric scoring.
- **Section 5 (Module 2 — Vision)**: Added `temporal_au.py` to file reference table and added a dedicated subsection detailing temporal onset velocity tracking, AU variance evaluation, and dynamic micro-expression classification.
- **Section 6 (Module 3 — Prosody)**: Updated folder path references to correctly read `module_03_prosody/`, updated `__init__.py` status to **Done**, removed obsolete naming warning, added `microsoft/wavlm-base-plus` 768-dim self-supervised embeddings to tooling/schema tables, and updated checklist/improvement statuses.
- **Section 7 (Module 4 — Fusion)**: Corrected `FUSED_VECTOR_DIM` from 65 to 72 to match actual `FULL_FEATURE_SCHEMA` size in `schema.py`. Added comprehensive file reference table and exact 72-dim feature breakdown across text (11), vision (33), and prosody (28).

### 2026-06-29 — Critical Audit & Security Hardening (22 Defects Fixed)

**Critical (C) Fixes:**
- **C1 — Dead Code + Uncaught Emotion Crash (`vision_processor.py`)**: Removed unreachable `if not face_detected` guard block (dead code). Wrapped `emotion_future.result()` in `try/except` so DeepFace OOM/crashes fall back to `blank` emotion instead of propagating uncaught.
- **C2 — Shared TF-IDF Vectorizer State Race (`semantic_grader.py`)**: Removed instance-level `self.vectorizer` — each `grade_response()` call now instantiates a fresh `TfidfVectorizer` locally, eliminating state mutation and thread-safety issues across concurrent benchmark runs.
- **C3 — Path Traversal Security (`settings.py` + `transcriber.py`)**: Added `path.is_symlink()` rejection to `transcribe_file_sync` — symlink attacks bypass `relative_to()` alone.
- **C4 — Private `_turn_frames` Encapsulation (`vision_processor.py`)**: Added public `get_turn_frames()` method; `run_live_calibration.py` now uses stable public API.
- **C5 — Dissonance Threshold Miscalibrated (`run_live_calibration.py`)**: Raised threshold from `0.35` (fired on every calm session) to `0.65`. Added intermediate `MODERATE` band at `0.45`–`0.65`.

**High (H) Fixes:**
- **H1 — Benchmark Threshold Overfitting (`run_baseline_benchmarks.py`)**: Mohler grader now uses proper 80/20 train/test split (`seed=42`). Thresholds tuned only on 80% training set; accuracy reported on unseen 20% test rows.
- **H2 — Magic Number Confidence Values (`temporal_au.py`)**: All AU tracker confidence values replaced with evidence-based computation (e.g., `0.50 + 0.40 × variance_flatness`).
- **H3 — Landmark Memory Accumulation (`vision_processor.py`)**: Stripped raw 478-point landmark arrays from `_turn_frames` storage — saves ~214K floats per 15-second session.
- **H5 — ThreadPoolExecutor Resource Leak (`run_live_calibration.py`)**: `vision_module.close()` now called at end of `main()`.
- **H6 — Fake Empirical Numbers in `compare_sota.py`**: Fully rewritten to dynamically compute accuracy figures by calling the live benchmark runner — no more hardcoded strings.
- **H7 — Silent Synthetic Audio Fallback (`run_live_calibration.py`)**: Prominent 3-line `!!! WARNING: SYNTHETIC AUDIO IN USE !!!` banner printed when microphone capture fails.

**Medium / Low (M/L) Fixes:**
- **M2** — AU key access guarded with `.get(key, 0.0)` to prevent `KeyError` on partial frame dicts.
- **M4** — Keyword matching upgraded from substring (`in`) to `re.search(r'\b...\b')` word-boundary matching.
- **M5** — Hardcoded personal name removed; `--name` CLI argument added to `run_live_calibration.py`.
- **M6** — Blink rate key mismatch fixed: scorecard now reads `blink_rate` (correct schema key).
- **H4** — `D` grade band added to `SemanticGrader` (composite 0.15–0.35) to eliminate harsh `C → F` jump.
- **L3** — `sounddevice` and `accelerate` added to `requirements.txt`.

**Verification**: All **31 pytest tests pass** after all fixes (`31 passed in 50.77s`).

### 2026-06-29 — SOTA Benchmark Extensions & Live Calibration Tool

**Created / Implemented (Plug-and-Play Extensions):**
- **Live Calibration Tool (`tools/run_live_calibration.py`)**: Standalone interactive diagnostic runner connecting webcam and microphone to ARIA Modules 1-4. Outputs live latency, emotion confidence, WavLM vector verification, dynamic attention allocation, and cross-modal dissonance scores without touching core module files.
- **Temporal AU Sequence Tracker (`modules/module_02_vision/temporal_au.py`)**: Implemented frame-by-frame Action Unit onset velocity and temporal derivative tracking to resolve still-frame `BLANK` false-negatives (Task 2).
- **Automated Semantic Grader (`modules/module_01_stt/semantic_grader.py`)**: Built TF-IDF N-Gram cosine similarity and keyword rubric scoring engine for candidate transcription turns. Wired directly into `run_baseline_benchmarks.py`, replacing the human inter-annotator baseline and achieving **72.64% automated grading accuracy** across 2,273 Mohler dataset rows.
- **SOTA Comparison Engine (`tests/benchmarks/compare_sota.py`)**: Evaluates ARIA models against published academic benchmarks, generating `SOTA_Benchmark_Comparison.md`.

### 2026-06-29 — WavLM Self-Supervised Audio Embeddings & SOTA RAVDESS SER Boost

**Upgraded / Completed (Task 1):**
- **Prosody Extractor (`modules/module_03_prosody/extractor.py`)**: Integrated HuggingFace `microsoft/wavlm-base-plus` self-supervised transformer embeddings alongside openSMILE features. Extracted 768-dim temporal average-pooled hidden representations with automatic GPU CUDA acceleration.
- **Audio Benchmarking Pipeline (`run_baseline_benchmarks.py`)**: Upgraded feature extraction to capture 793-dim vectors (25 scalar openSMILE + 768 WavLM). Replaced unscaled random forest classifier with a scaled `StandardScaler + LogisticRegression` pipeline trained on strict actor-independent splits (Actors 01-18 train, Actors 19-24 unseen test).
- **Benchmark Results**: Unseen speaker Speech Emotion Recognition (SER) accuracy increased by over **20.5 percentage points**, jumping from **49.17%** to **69.72%** (Macro F1: 0.7084).

### 2026-06-29 — Honest Benchmarking Pipeline & Zero-Leakage Evaluation Suite

**Fixed / Upgraded:**
- **Benchmarking Pipeline (`run_baseline_benchmarks.py`)**: Completely rewrote the benchmarking pipeline to eliminate target label leakage and synthetic classification accuracy reporting. Enforced strict rule: `y_true = ground-truth label only`, `y_pred = output from ARIA model inference only`.
- **Vision FER2013 Blind Inference**: Integrated blind inference via `EmotionAnalyzer.process_frame()`, mapping FER2013 ground truth folders into interview emotion categories (`engaged, confused, nervous, confident, blank`). Achieved **66.67%** accuracy across representative test images without label copying.
- **Audio RAVDESS Actor-Independent Classifier**: Extracted acoustic feature vectors across all 1,440 RAVDESS files and cached to disk (`ravdess_features.pkl`). Enforced a strict **actor-independent train/test split** (Actors 01-18 for training, Actors 19-24 for held-out evaluation).
- **Text Mohler Baseline**: Segregated under human inter-annotator evaluation (`Grader 1 vs Consensus score_avg`), confirming **78.40%** agreement.
- **Engineering Diagnostics**: Isolated synthetic stress testing under dedicated engineering verification checks confirming vector dimensions (72-dim), missing modality zero-weight recovery, attention weight summation (1.0), and dynamic cross-modal dissonance penalization.

### 2026-06-29 — Module 4 Dynamic Fusion Fixes, Concatenation Baseline & Benchmarking Sandbox

**Fixed / Upgraded:**
- **Module 4 (Fusion)**: Fixed cross-modal dissonance penalization in `attention_fusion.py` (`_compute_modality_weights`) to calculate per-modality cosine distance against other active modalities, selectively punishing conflicting signals rather than applying a global uniform scalar.
- **Module 4 (Gating Scale)**: Updated `_apply_modality_weights` to scale softmax weights by active modality count ($w_m \times N_{active}$), preventing feature values from shrinking to 33% scale when all 3 modalities are active.
- **Module 4 (Baseline)**: Implemented `concat_fusion.py` (`ConcatFusion` and `ConcatenationFusionEngine`) for unweighted baseline comparison. Exported engines in `__init__.py`.
- **Benchmarking & Testing**: Created unit test suite `tests/test_fusion.py` (31/31 tests passing across repo). Created `tests/benchmarks/run_baseline_benchmarks.py` replicating the baseline comparison metrics table across FER2013, RAVDESS, Mohler, CMU-MOSEI, and Box of Lies.

### 2026-06-29 — Modules 1-3 Critical Review Fixes & Dynamic Architecture Migration

**Fixed / Upgraded:**
- **Module 1 (STT)**: Fixed response latency bug (`None` vs provided timestamps); added path traversal security protection (`ALLOWED_AUDIO_DIR = PROJECT_ROOT`); added input size validation (`MAX_AUDIO_DURATION_S`).
- **Module 2 (Vision)**: Migrated `face_mesh.py` from legacy `mp.solutions` to modern MediaPipe Tasks API (`FaceLandmarker` with `models/face_landmarker.task`), eliminating protobuf C++ descriptor conflicts with TensorFlow/DeepFace. Added `TF_USE_LEGACY_KERAS=1` default for Keras 2 compatibility. Added non-lossy probability distribution remapping (`emotion_distribution`). Added blink state reset between turns (`reset_state()`). Resolved vulnerability 2B by creating `VisionBaselineManager` (`baseline.py`) for thread-safe dynamic candidate resting calibration across Turns 1+2 and deviation computation (`au_deviations`, `eye_contact_deviation`, `blink_rate_deviation`) on Turns 3+.
- **Module 3 (Prosody)**: Cached `openSMILE` model instance across extractor invocations (2x speedup). Added thread-safe `RLock` for baseline storage. Fixed pitch conversion test fixture. Created `__init__.py`.
- **Module 4 (Fusion Roadmap)**: Updated `ARIA_Coding_Assistant_Guide.md` with dynamic multimodal optimization specification (dynamic missing modality imputation, continuous feature alignment, and cross-modal gating/attention weights).
- **Verification**: Ran all test suites (`test_stt.py`, `test_vision.py`, `test_prosody.py`) — 100% passing (28/28 tests).

### 2026-06-24 — Module 3 fixes: pitch Hz, librosa MFCC, extract API

**Fixed in `extractor.py`:**

- Pitch: semitones → Hz before mean/variance/range
- MFCC: librosa instead of openSMILE (eGeMAPS has no MFCCs)
- `extract()` signature: removed unused `turn_id` / `candidate_id` (baseline stays in pipeline)

**Updated:** `pipeline.py`, `tests/test_prosody.py`, `notes.md`

### 2026-06-24 — Module 3 complete: tests + notes update

**Added:**

- `tests/test_prosody.py` — 13 unit tests + 2 integration tests (openSMILE)
- `pytest.ini` — registers `integration` marker
- `tests/conftest.py` — `ensure_prosody_fixture()` for 30s audio

**Fixed:**

- `pipeline.py` import path: `module_03_prosody` → `module_3_prosody` (was broken)

**Updated:**

- `notes.md` — full Module 3 documentation, compliance checklist, improvement suggestions

### 2026-06-24 — Module 3 prosody implementation (user)

**Observed in repo (not modified by assistant):**

- `modules/module_3_prosody/extractor.py` — full `ProsodyExtractor` with openSMILE eGeMAPS + librosa
- `modules/module_3_prosody/baseline.py` — `ProsodyBaselineManager`
- `modules/module_3_prosody/pipeline.py` — `process_prosody_turn()` orchestrator

**Updated:**

- `notes.md` — full Module 3 section, status table, folder structure, open issues list

### 2026-06-24 — Initial Phase 1 foundation (Modules 1 & 2)

**Created:**

- `config/settings.py` — global constants, paths, emotion map, model settings
- `modules/module_01_stt/transcriber.py` — faster-whisper offline transcription with async wrappers
- `modules/module_02_vision/face_mesh.py` — MediaPipe landmarks, geometric AUs, head pose, blink EAR
- `modules/module_02_vision/emotion.py` — DeepFace with interview-context remapping
- `modules/module_02_vision/gaze.py` — L2CS-Net with head-pose fallback
- `modules/module_02_vision/vision_processor.py` — parallel frame processing + turn summarization
- `tests/test_stt.py`, `tests/test_vision.py`, `tests/conftest.py`
- `requirements.txt`, `.gitignore`, this `notes.md`

**Not created yet (per build order):**

- Module 4 Fusion
- `config/rl_spec.py`
- Backend / Frontend
- Streaming STT

---

## Next Steps & Dynamic Optimization Roadmap

Per [ARIA_Coding_Assistant_Guide.md](./ARIA_Coding_Assistant_Guide.md) dynamic architecture specifications:

1. **Module 4 — Dynamic Multimodal Fusion Engine** (`fusion_engine.py` / `attention_fusion.py`): ✅ Implemented cross-modal attention gating, scale-preserving gating, per-modality dissonance penalization, concatenation baseline, and evaluation sandbox.
2. **System-Wide Dynamic Enhancements**:
   - **Temporal Warmup Weighting**: Apply higher weights to later turns so candidate anxiety during icebreakers does not disproportionately lower session competency scores.
   - **Closed-Loop Adaptive Questioning**: Link cognitive load anomalies ($Z$-scores) directly to Module 8 (LLM Question Generator) to dynamically scaffold difficulty or trigger clarification probes during cross-modal dissonance.
3. **End-to-End Dynamic Integration Test**: Simulate a multi-turn session confirming dynamic missing modality recovery and baseline deviation flow.

### AI Agent Execution Guide: Boosting Multimodal Benchmark Accuracies to SOTA

When instructing AI coding agents to upgrade ARIA's baseline accuracies toward State-of-the-Art (SOTA), follow this modular execution roadmap. Give the agent one task at a time and reference the target files.

#### Task 1: Upgrade Audio & Prosody to Self-Supervised Embeddings ✅ COMPLETED
- **Objective**: Replace or augment static openSMILE eGeMAPS/MFCC scalars with pre-trained acoustic transformer embeddings.
- **Implementation Status**:
  1. Integrated `microsoft/wavlm-base-plus` (`AutoFeatureExtractor` & `AutoModel`) with automatic CUDA acceleration into `ProsodyExtractor` (`modules/module_03_prosody/extractor.py`).
  2. Extracted pooled 768-dim temporal self-supervised embeddings alongside existing openSMILE scalars and MFCCs.
  3. Upgraded `run_baseline_benchmarks.py` to use a `StandardScaler + LogisticRegression` pipeline on the combined 793-dim representations.
- **Empirical Benchmark Result**: Actor-independent unseen speaker speech emotion recognition (SER) accuracy boosted from **49.17%** (static openSMILE + RandomForest) to **69.72%** (+20.55 percentage points | Macro F1: 0.7084).

#### Task 2: Upgrade Vision to Temporal AU Sequence Tracking (FER2013 Target: 75%+)
- **Objective**: Capture dynamic micro-expressions over time rather than averaging static per-frame probabilities.
- **Target Files**: `modules/module_02_vision/face_mesh.py`, `modules/module_02_vision/emotion.py`
- **Agent Instructions**:
  1. Modify `face_mesh.py` to buffer frame-by-frame Action Unit (AU) time-series across the interview turn ($T \times 15$ matrix).
  2. Implement a 1D Temporal Convolutional Network (TCN) or Bi-LSTM head to classify dynamic AU sequence transitions (e.g., transient AU4 brow lower or AU12 smile suppression).

#### Task 3: Upgrade Fusion to Word-Level Cross-Attention (CMU-MOSEI Target: 83%+)
- **Objective**: Capture fine-grained asynchronous alignment across speech words, facial cues, and vocal tone.
- **Target Files**: `modules/module_04_fusion/attention_fusion.py`
- **Agent Instructions**:
  1. Extend `DynamicAttentionFusion` to support sub-turn cross-attention matrices ($Q_{\text{text}} K_{\text{vision}}^T$).
  2. Compute directional attention weights allowing text tokens to attend directly to synchronized video frames and pitch contours before turn-level pooling.

#### Task 4: Sharpen Deception & Incongruence Penalty (Box of Lies Target: 68%+)
- **Objective**: Penalize temporal lag between verbal sentiment and physical micro-expressions.
- **Target Files**: `modules/module_04_fusion/attention_fusion.py`
- **Agent Instructions**:
  1. Calculate latency asymmetry between verbal emotional words and physical facial expression onset.
  2. Incorporate temporal lag asymmetry directly into `_compute_cross_modal_dissonance`.

---

## Interface Contract Reminder

At turn end, Krissh's modules must produce this structure for Raghav's backend (full schema in guide Section 6):

```python
turn_signal = {
    "session_id": str,
    "turn_id": int,
    "transcript": str,                    # Module 1
    "word_timestamps": list,              # Module 1
    "language": str,                      # Module 1
    "response_latency_ms": float,         # Module 1
    "vision": { ... },                    # Module 2 per-turn summary
    "prosody": { ... },                   # Module 3 ✅ via process_prosody_turn()
    "fused_vector": list,                 # Module 4 ✅ via fuse_turn()
    "cognitive_load_label": str,          # Module 10 ✅ via CognitiveLoadClassifier.classify()
    "distress_score": float,              # Module 10 ✅ via CognitiveLoadClassifier.classify()
    "anti_gaming_flags": list,            # Module 11 ✅ via AntiGamingMonitor.evaluate_turn()
}
```

All Krissh-owned perception modules (1, 2, 3, 4, 10, 11) now supply their respective fields in the interface contract.

---

*When making changes, append to the [Change Log](#change-log) and update the relevant module section above.*
