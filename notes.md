# ARIA — Development Notes

> Living documentation for the ARIA project. All code changes, architecture decisions, and module internals are recorded here so future coding assistants (and team members) can pick up context without re-reading the entire codebase.

**Project:** Autonomous Reinforcement-based Interview Agent with Multimodal Adaptive Assessment  
**Reference:** [ARIA_Coding_Assistant_Guide.md](./ARIA_Coding_Assistant_Guide.md)  
**Last updated:** 2026-06-24

---

## Table of Contents

1. [Project Status](#project-status)
2. [Environment & Setup](#environment--setup)
3. [Folder Structure (Current)](#folder-structure-current)
4. [Module 1 — Speech-to-Text (STT)](#module-1--speech-to-text-stt)
5. [Module 2 — Vision](#module-2--vision)
6. [Change Log](#change-log)
7. [Next Steps](#next-steps)

---

## Project Status

| Module | Status | Owner | Notes |
|--------|--------|-------|-------|
| Module 1 — STT | **Implemented** (offline) | Krissh | Streaming mode deferred |
| Module 2 — Vision | **Implemented** (single frame + turn summary) | Krissh | L2CS fallback via head pose if weights missing |
| Module 3 — Prosody | Not started | Krissh | — |
| Module 4 — Fusion | Not started | Krissh | — |
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
├── notes.md                         # This file
├── requirements.txt
├── .gitignore
├── config/
│   ├── __init__.py
│   └── settings.py                  # Global constants — import everywhere
├── modules/
│   ├── module_01_stt/
│   │   ├── __init__.py
│   │   └── transcriber.py
│   └── module_02_vision/
│       ├── __init__.py
│       ├── face_mesh.py
│       ├── emotion.py
│       ├── gaze.py
│       └── vision_processor.py      # Orchestrator (parallel threads + turn summary)
├── models/                          # gitignored — L2CS weights go here
├── data/                            # gitignored datasets
└── tests/
    ├── conftest.py
    ├── test_stt.py
    └── test_vision.py
```

---

## Module 1 — Speech-to-Text (STT)

### Purpose

Convert the candidate's spoken answer into text with word-level timestamps, language detection, confidence, and response latency. This is the **text modality** input for the POMDP observation space.

### Files

| File | Role |
|------|------|
| `modules/module_01_stt/transcriber.py` | Core transcription logic |
| `modules/module_01_stt/__init__.py` | Public exports: `Transcriber`, `transcribe`, `transcribe_file` |
| `config/settings.py` | `MODEL_WHISPER`, `WHISPER_COMPUTE_TYPE`, `DEVICE`, `AUDIO_SAMPLE_RATE` |

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
| `vision_processor.py` | Parallel orchestration + per-turn summarization |
| `__init__.py` | Exports `VisionProcessor` |

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
                           summarize_turn() at turn end
```

### Sub-module details

#### face_mesh.py — MediaPipe Face Mesh

**What it does:**

1. Converts BGR → RGB and runs MediaPipe Face Mesh (`refine_landmarks=True`, max 1 face).
2. Extracts 468 landmarks as `(x, y, z)` pixel coordinates.
3. Estimates 9 Action Units geometrically from landmark distances (normalized by interocular distance).
4. Computes head pose (roll, pitch, yaw in degrees) via `cv2.solvePnP` with a canonical 3D face model.
5. Detects blinks using Eye Aspect Ratio (EAR): when EAR drops below `EAR_BLINK_THRESHOLD` (0.21) and recovers, a blink is recorded with duration in ms.

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

## Change Log

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

- Module 3 Prosody
- Module 4 Fusion
- `config/rl_spec.py`
- Backend / Frontend
- Streaming STT

---

## Next Steps

Per [ARIA_Coding_Assistant_Guide.md](./ARIA_Coding_Assistant_Guide.md) Section 9 build order:

1. **Module 3 — Prosody** (`extractor.py`, `baseline.py`) — SpeechBrain + openSMILE on turn audio
2. **Module 2 enhancement** — validate per-turn summary with real webcam footage at 2 fps
3. **Module 4 — Fusion V1** (`concat_fusion.py`) — concatenate text + vision + prosody features
4. **End-to-end test** — one simulated turn through Modules 1→2→3→4, print `turn_signal`
5. **Download L2CS weights** to `models/` for production gaze accuracy
6. **Streaming STT** — integrate with WebRTC audio chunks in Phase 5

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
    "prosody": { ... },                   # Module 3 (TODO)
    "fused_vector": list,                 # Module 4 (TODO)
    "cognitive_load_label": str,          # Module 10 (TODO)
    "distress_score": float,              # Module 10 (TODO)
    "anti_gaming_flags": list,            # Module 11 (TODO)
}
```

Modules 1 and 2 currently supply: `transcript`, `word_timestamps`, `language`, `response_latency_ms`, and `vision`.

---

*When making changes, append to the [Change Log](#change-log) and update the relevant module section above.*
