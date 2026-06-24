# ARIA — Coding Assistant Master Guide
### Autonomous Reinforcement-based Interview Agent with Multimodal Adaptive Assessment

---

## CRITICAL INSTRUCTIONS FOR CODING ASSISTANT

You are building ARIA — a multimodal AI interview system. Read this entire document before writing a single line of code. Every module, interface contract, folder path, and data schema is defined here. Do not deviate from the structure. Do not add dependencies not listed here. Do not rename files or folders.

---

## 1. Environment Specification

```
OS:              Windows 11
Python:          3.11.9 (MUST use this version, not 3.12 or 3.13)
GPU:             NVIDIA RTX 4060 — 8GB VRAM
CUDA:            13.0
Package Manager: pip only (no conda)
Virtual Env:     .venv (already created, always activate before running)
```

**Activate environment before every session:**
```bash
.venv\Scripts\activate
```

**GPU memory budget (must stay under 8GB total):**
```
faster-whisper large-v3 (int8):  ~2.5GB
MediaPipe + DeepFace:            ~1.5GB
L2CS-Net:                        ~0.5GB
Fusion layer:                    ~0.5GB
Headroom:                        ~3.0GB
```

---

## 2. Exact Folder Structure

Create this structure exactly. Do not add or rename folders.

```
ARIA/
│
├── .venv/                          # Python 3.11.9 virtual environment
├── .env                            # API keys, model paths (never commit)
├── .gitignore
├── requirements.txt
├── README.md
│
├── config/
│   ├── __init__.py
│   ├── settings.py                 # Global constants, model names, paths
│   └── rl_spec.py                  # RL state/action/reward specification (Krissh owns)
│
├── modules/
│   ├── module_01_stt/
│   │   ├── __init__.py
│   │   └── transcriber.py          # faster-whisper streaming transcription
│   │
│   ├── module_02_vision/
│   │   ├── __init__.py
│   │   ├── face_mesh.py            # MediaPipe 468 landmarks + AU activations
│   │   ├── emotion.py              # DeepFace emotion classifier (interview-context)
│   │   └── gaze.py                 # L2CS-Net gaze vector (yaw, pitch)
│   │
│   ├── module_03_prosody/
│   │   ├── __init__.py
│   │   ├── extractor.py            # SpeechBrain + openSMILE feature extraction
│   │   └── baseline.py             # Per-candidate personal baseline calibration
│   │
│   ├── module_04_fusion/
│   │   ├── __init__.py
│   │   ├── concat_fusion.py        # V1 — simple concatenation (build first)
│   │   └── attention_fusion.py     # V2 — cross-modal attention transformer (build after V1 works)
│   │
│   ├── module_05_ontology/
│   │   ├── __init__.py
│   │   └── skill_graph.py          # NetworkX directed skill graph (Raghav owns)
│   │
│   ├── module_06_belief/
│   │   ├── __init__.py
│   │   └── updater.py              # Bayesian belief state updater (Raghav owns)
│   │
│   ├── module_07_rl/
│   │   ├── __init__.py
│   │   └── policy.py               # IQL policy training and inference (Raghav owns)
│   │
│   ├── module_08_llm/
│   │   ├── __init__.py
│   │   └── question_generator.py   # Llama/Qwen question generation (Raghav owns)
│   │
│   ├── module_09_tts/
│   │   ├── __init__.py
│   │   ├── synthesizer.py          # Coqui XTTS-v2 TTS (Raghav owns)
│   │   └── avatar.py               # SadTalker talking head (Raghav owns)
│   │
│   ├── module_10_cognitive_load/
│   │   ├── __init__.py
│   │   └── classifier.py           # 3-class: low / anxiety / ignorance (Krissh owns)
│   │
│   └── module_11_anti_gaming/
│       ├── __init__.py
│       ├── gaze_scanner.py         # Note reading detection via gaze patterns
│       ├── latency_checker.py      # AI assistance detection via latency uniformity
│       └── semantic_checker.py     # Coaching detection via semantic similarity
│
├── rl/
│   ├── __init__.py
│   ├── reward.py                   # Reward function implementation + unit tests (Krissh owns)
│   ├── state_builder.py            # Assembles RL state vector from all module outputs
│   └── action_masker.py            # Action validity rules per state (Krissh owns)
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── components/
│       │   ├── LandingPage.jsx
│       │   ├── InterviewRoom.jsx
│       │   └── ResumeUpload.jsx
│       └── utils/
│           └── webrtc.js
│
├── backend/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app entry point (Raghav owns)
│   ├── routes/
│   │   ├── session.py              # /start-session, /end-session
│   │   └── turn.py                 # /process-turn
│   └── queue/
│       └── redis_handler.py        # Redis message queue
│
├── data/
│   ├── datasets/                   # Raw downloaded datasets (gitignored)
│   ├── processed/                  # Preprocessed versions
│   └── synthetic/                  # LLM-generated synthetic training data
│
├── tests/
│   ├── test_stt.py
│   ├── test_vision.py
│   ├── test_prosody.py
│   ├── test_fusion.py
│   └── test_reward.py
│
└── notebooks/
    └── exploration/                # EDA notebooks — not production code
```

---

## 3. Project Overview

ARIA is an autonomous AI interviewer. A candidate joins, turns on webcam and microphone, uploads resume, and speaks naturally. ARIA listens, watches, analyses, and decides what to ask next — all in real time. No scripts. No fixed question banks. Every interview is unique and adaptive.

**The core technical idea:** A job interview is modelled as a POMDP (Partially Observable Markov Decision Process). The AI agent never directly observes the candidate's true competency — it infers it from noisy multimodal signals. A reinforcement learning policy decides which interview action to take next to maximally reduce uncertainty about true competency.

**End result:** A full PDF evaluation report covering technical competency, communication analysis, confidence estimation, bluffing detection, cognitive load profile, integrity flags, and hiring recommendation.

---

## 4. POMDP Formulation

This is the mathematical foundation of ARIA. Every module maps to one of these components.

```
State (S):       Candidate's TRUE competency per skill node — hidden, never directly observed
Observation (O): Multimodal signals per turn (transcript + facial + prosody)
Action (A):      What the agent asks next — 8 discrete options
Reward (R):      Information gain − penalties (see reward function below)
Belief (b):      P(skill_level | observations so far) — updated after every turn
Transition (T):  How belief updates after each observed response
```

**Belief state per skill node:**
```python
b[node] = [P(beginner), P(mid), P(expert)]
# Initialised: [0.33, 0.33, 0.33]
# Updated via Bayesian inference after every turn
# Interview ends when entropy(b) < TERMINATION_THRESHOLD across all nodes
```

---

## 5. Module-by-Module Specification

### MODULE 1 — Speech-to-Text (STT)
**File:** `modules/module_01_stt/transcriber.py`
**Owner:** Krissh
**Tool:** faster-whisper with Whisper large-v3

**Input:**
```python
audio: np.ndarray  # raw audio array, 16kHz mono float32
```

**Output (must match this schema exactly):**
```python
{
    "transcript": str,              # full text of the candidate's answer
    "word_timestamps": [            # list of word-level timestamps
        {"word": str, "start": float, "end": float}
    ],
    "language": str,                # detected language code e.g. "en"
    "confidence": float,            # average segment confidence 0.0-1.0
    "response_latency_ms": float    # time from question end to first word
}
```

**Implementation notes:**
- Use `compute_type="int8"` to stay within GPU memory budget
- Use `device="cuda"` — GPU is available
- Enable VAD filter (`vad_filter=True`) to prevent hallucination on silence
- Model size: `large-v3`
- Run as async function — do not block the main thread
- Offline mode first (transcribe a complete .wav file), streaming mode later

**Install:**
```bash
pip install faster-whisper
```

---

### MODULE 2 — Vision Module
**Files:** `modules/module_02_vision/face_mesh.py`, `emotion.py`, `gaze.py`
**Owner:** Krissh
**Tools:** MediaPipe Face Mesh + DeepFace + L2CS-Net

**Input:**
```python
frame: np.ndarray  # BGR frame from OpenCV, shape (H, W, 3)
```

**Output per frame (must match this schema exactly):**
```python
{
    "landmarks": np.ndarray,        # shape (468, 3) — x, y, z per landmark
    "au_activations": {             # Action Unit activations
        "AU1": float,   # inner brow raise
        "AU2": float,   # outer brow raise
        "AU4": float,   # brow lowerer
        "AU6": float,   # cheek raiser
        "AU12": float,  # lip corner puller (smile)
        "AU15": float,  # lip corner depressor
        "AU17": float,  # chin raiser
        "AU23": float,  # lip tightener
        "AU25": float   # lips part
    },
    "emotion_label": str,           # one of: engaged/confused/nervous/confident/blank
    "emotion_confidence": float,    # 0.0-1.0
    "gaze_vector": {
        "yaw": float,               # horizontal gaze direction in degrees
        "pitch": float              # vertical gaze direction in degrees
    },
    "eye_contact_score": float,     # 0.0 = looking away, 1.0 = direct eye contact
    "head_pose": {
        "roll": float,
        "pitch": float,
        "yaw": float
    },
    "blink_detected": bool,         # True if blink in this frame
    "blink_duration_ms": float      # duration if blink detected, else 0.0
}
```

**Per-turn summary (average of all frames in turn):**
```python
{
    "emotion_label": str,           # most frequent emotion label across turn
    "au_activations": dict,         # mean AU values across turn
    "gaze_vector": {"yaw": float, "pitch": float},  # mean gaze
    "eye_contact_score": float,     # mean eye contact score
    "head_pose": {"roll": float, "pitch": float, "yaw": float},  # mean pose
    "blink_rate": float             # blinks per minute
}
```

**Implementation notes:**
- Frames arrive every 500ms from WebRTC (2 fps)
- Run MediaPipe, DeepFace, L2CS-Net in parallel threads per frame
- DeepFace emotion labels must be remapped: map generic labels to interview-context labels
  - `happy` → `engaged`
  - `neutral` → `blank`
  - `fear` → `nervous`
  - `surprise` → `confused`
  - `sad` / `disgust` / `angry` → `nervous`
- Store all frame outputs in a list, summarise at turn end
- L2CS-Net weights: download from official repo (open source)

**Install:**
```bash
pip install mediapipe deepface opencv-python
```

---

### MODULE 3 — Prosody Module
**Files:** `modules/module_03_prosody/extractor.py`, `baseline.py`
**Owner:** Krissh
**Tools:** SpeechBrain + openSMILE

**Input:**
```python
audio_clip: np.ndarray  # full audio of one candidate turn, 16kHz mono float32
turn_id: int            # used to check if this is a baseline turn (turn_id <= 2)
candidate_id: str       # to retrieve personal baseline
```

**Output (must match this schema exactly):**
```python
{
    "pitch_mean": float,            # mean F0 in Hz
    "pitch_variance": float,        # variance of F0
    "pitch_range": float,           # max F0 - min F0
    "speech_rate": float,           # syllables per second
    "pause_count": int,             # number of within-answer pauses > 250ms
    "pause_total_duration_ms": float,
    "disfluency_count": int,        # count of um/uh/erm/like
    "disfluency_timestamps": [float],  # timestamps of each disfluency
    "response_latency_ms": float,   # already computed by STT module
    "energy_mean": float,           # mean RMS energy
    "jitter": float,                # cycle-to-cycle F0 variation
    "shimmer": float,               # cycle-to-cycle amplitude variation
    "mfcc_vector": list,            # 13 MFCC coefficients, list of floats
    "speech_to_silence_ratio": float,
    # Deviation from personal baseline (None for first 2 turns)
    "pitch_deviation": float,
    "rate_deviation": float,
    "energy_deviation": float
}
```

**Baseline logic:**
- Turns 1 and 2: store features as personal baseline, set deviation fields to None
- Turns 3+: compute deviation = (current - baseline) / baseline for pitch, rate, energy
- Baseline stored in memory dict keyed by candidate_id

**Install:**
```bash
pip install speechbrain opensmile
```

---

### MODULE 4 — Multimodal Fusion Layer
**Files:** `modules/module_04_fusion/concat_fusion.py`, `attention_fusion.py`
**Owner:** Krissh
**Build V1 (concat) first. V2 (attention) only after all three input modules work.**

**Input:**
```python
transcript: str                     # from Module 1
vision_summary: dict                # from Module 2 per-turn summary
prosody_features: dict              # from Module 3
turn_id: int
```

**V1 Output (concat_fusion.py):**
```python
{
    "fused_vector": list,           # concatenated feature vector, fixed dimension
    "fusion_method": "concat",
    "vector_dim": int               # document this value once computed
}
```

**V2 Output (attention_fusion.py):**
```python
{
    "fused_vector": list,           # attention-weighted fused vector
    "fusion_method": "cross_modal_attention",
    "modality_weights": {           # how much each modality contributed this turn
        "text": float,
        "vision": float,
        "prosody": float
    },
    "vector_dim": int
}
```

**V2 Architecture:**
- Three separate encoders: text encoder (BERT-small), vision encoder (linear projection), prosody encoder (linear projection)
- Cross-modal attention: each modality attends to the other two
- Output: weighted sum → single fixed-dim vector
- Trained with session-level competency labels as supervision signal

---

### MODULE 10 — Cognitive Load Separation
**File:** `modules/module_10_cognitive_load/classifier.py`
**Owner:** Krissh

**Input:**
```python
prosody: dict       # from Module 3
vision: dict        # from Module 2 per-turn summary
turn_id: int
candidate_id: str
```

**Output:**
```python
{
    "cognitive_load_label": str,    # one of: "low" / "anxiety" / "ignorance"
    "distress_score": float,        # continuous 0.0-1.0
    "confidence": float,            # classifier confidence
    "signals_used": list            # which signals drove this prediction
}
```

**Classification logic:**
- `low`: low disfluency, normal speech rate, stable gaze, low pitch variance
- `anxiety`: high disfluency + high pitch variance + gaze breaks + normal/correct answer content
- `ignorance`: long latency + low speech rate + shallow semantic content + looking away

**Key distinction:** anxiety = candidate knows but is stressed. ignorance = candidate does not know. Both can look similar — the separator is semantic content quality from STT combined with physiological stress signals.

---

### MODULE 11 — Anti-Gaming
**Files:** `modules/module_11_anti_gaming/gaze_scanner.py`, `latency_checker.py`, `semantic_checker.py`
**Owner:** Krissh

**Output:**
```python
{
    "flags": list,                  # empty list if clean
    # possible values: "note_reading" / "ai_assist" / "coaching" / "scripted"
    "flag_confidences": dict,       # confidence per flag
    "is_flagged": bool
}
```

**Detection logic per flag:**

`note_reading` — gaze scan pattern shows horizontal left-to-right sweeps (reading pattern) rather than natural thinking movement (upward/random). Detected from gaze_vector sequence across frames.

`ai_assist` — response latency is suspiciously short AND delivery rate is unnaturally uniform (low variance in words-per-second across the answer). Humans have natural rhythm variation; AI-read answers are flat.

`coaching` — head turns sharply during answer (looking off-camera) + semantic similarity between this answer and a previous unrelated answer is above threshold (scripted language).

`scripted` — vocabulary complexity and sentence structure are inconsistent with candidate's baseline (detected via BGE-M3 embeddings comparing phrasing style across turns).

---

### RL SPECIFICATION
**File:** `config/rl_spec.py`
**Owner:** Krissh — must deliver this before Raghav begins IQL training

**State vector:**
```python
RL_STATE_SCHEMA = {
    # From Module 6 (Raghav)
    "belief_vector": list,          # flattened [P(beg), P(mid), P(exp)] per skill node
    "belief_entropy": float,        # overall entropy across all nodes

    # From Module 4 (Krissh)
    "fused_vector": list,           # fixed-dim output of fusion layer

    # From Module 10 (Krissh)
    "cognitive_load_label": int,    # 0=low, 1=anxiety, 2=ignorance
    "distress_score": float,

    # From Module 11 (Krissh)
    "anti_gaming_active": bool,

    # Session context
    "turn_id": int,
    "topics_covered": list,
    "consecutive_same_topic": int
}
```

**Action space:**
```python
RL_ACTION_SPACE = [
    "increase_difficulty",
    "decrease_difficulty",
    "ask_follow_up_same_topic",
    "switch_topic",
    "probe_foundation",
    "ask_behavioral",
    "ask_situational",
    "conclude_interview"
]
```

**Action masks (when each action is INVALID):**
```python
RL_ACTION_MASKS = {
    "conclude_interview":        "belief_entropy > TERMINATION_ENTROPY_THRESHOLD",
    "probe_foundation":          "no prerequisite node exists in ontology graph",
    "increase_difficulty":       "consecutive_same_topic > 3",
    "decrease_difficulty":       "turn_id <= 2",  # baseline turns, no adaptation yet
}
```

**Reward function:**
```python
# File: rl/reward.py
def compute_reward(state_before, state_after, action, flags):
    R = (
        alpha * information_gain(state_before, state_after)    # KL divergence on belief
        - beta * duration_penalty()                             # small negative per turn
        + gamma * signal_consistency_bonus(state_after)        # modality agreement
        - delta * distress_penalty(state_after)                # high distress = bad
        + epsilon * integrity_bonus(flags)                     # terminal: if flag confirmed
        + omega * outcome_alignment_reward                     # terminal: from Dataset 5
    )
    return R

# Default coefficients (tune during training):
REWARD_COEFFICIENTS = {
    "alpha": 1.0,    # information gain weight
    "beta": 0.05,    # duration penalty weight
    "gamma": 0.3,    # signal consistency weight
    "delta": 0.5,    # distress penalty weight
    "epsilon": 0.8,  # integrity detection weight
    "omega": 2.0     # outcome alignment weight (highest — this is the ground truth)
}

TERMINATION_ENTROPY_THRESHOLD = 0.3
```

---

## 6. Interface Contract

This is the exact data structure Krissh's modules must output per turn. Raghav's modules consume this. Both parties must agree on this schema before coding begins.

```python
# Produced at end of every conversational turn
turn_signal = {
    "session_id": str,
    "turn_id": int,

    # Module 1 output
    "transcript": str,
    "word_timestamps": list,
    "language": str,
    "response_latency_ms": float,

    # Module 2 output (per-turn summary)
    "vision": {
        "emotion_label": str,
        "au_activations": {
            "AU1": float, "AU2": float, "AU4": float, "AU6": float,
            "AU12": float, "AU15": float, "AU17": float, "AU23": float, "AU25": float
        },
        "gaze_vector": {"yaw": float, "pitch": float},
        "eye_contact_score": float,
        "head_pose": {"roll": float, "pitch": float, "yaw": float},
        "blink_rate": float
    },

    # Module 3 output
    "prosody": {
        "pitch_mean": float,
        "pitch_variance": float,
        "speech_rate": float,
        "pause_count": int,
        "disfluency_count": int,
        "response_latency_ms": float,
        "jitter": float,
        "shimmer": float,
        "mfcc_vector": list,
        "pitch_deviation": float,
        "rate_deviation": float
    },

    # Module 4 output
    "fused_vector": list,

    # Module 10 output
    "cognitive_load_label": str,    # "low" / "anxiety" / "ignorance"
    "distress_score": float,

    # Module 11 output
    "anti_gaming_flags": list       # [] if clean
}
```

---

## 7. Installation Order

Install in this exact order to avoid dependency conflicts:

```bash
# Step 1 — Core ML
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2 — STT
pip install faster-whisper

# Step 3 — Vision
pip install mediapipe deepface opencv-python

# Step 4 — Prosody
pip install speechbrain opensmile

# Step 5 — NLP + Embeddings
pip install transformers sentence-transformers

# Step 6 — Graph
pip install networkx

# Step 7 — RL
pip install stable-baselines3

# Step 8 — Backend
pip install fastapi uvicorn redis

# Step 9 — Report
pip install reportlab pymupdf spacy
python -m spacy download en_core_web_sm

# Step 10 — Utilities
pip install numpy scipy librosa soundfile python-dotenv pytest
```

---

## 8. config/settings.py — Global Constants

```python
# All model paths, device settings, and global constants go here
# Import this file in every module — never hardcode values

MODEL_WHISPER = "large-v3"
WHISPER_COMPUTE_TYPE = "int8"
DEVICE = "cuda"

VIDEO_FRAME_INTERVAL_MS = 500      # capture frame every 500ms
AUDIO_SAMPLE_RATE = 16000          # 16kHz mono
MFCC_COEFFICIENTS = 13

BASELINE_TURNS = 2                  # first N turns used for personal baseline
TERMINATION_ENTROPY_THRESHOLD = 0.3

EMOTION_LABEL_MAP = {
    "happy": "engaged",
    "neutral": "blank",
    "fear": "nervous",
    "surprise": "confused",
    "sad": "nervous",
    "disgust": "nervous",
    "angry": "nervous"
}

DATA_DIR = "data/"
DATASETS_DIR = "data/datasets/"
PROCESSED_DIR = "data/processed/"
SYNTHETIC_DIR = "data/synthetic/"
```

---

## 9. Build Order (Follow Exactly)

Build in this sequence. Do not jump ahead. Each step depends on the previous.

```
Phase 1 — Foundation (Week 1)
├── 1. Environment setup + folder structure + requirements.txt
├── 2. config/settings.py
├── 3. Module 1 STT — offline mode (transcribe a .wav file)
├── 4. Module 2 Vision — single frame processing
└── 5. Module 3 Prosody — feature extraction on a .wav file

Phase 2 — Per-Turn Pipeline (Week 2)
├── 6. Module 2 Vision — per-turn summary (aggregate frames)
├── 7. Module 3 Prosody — personal baseline calibration
├── 8. Module 4 Fusion — V1 concatenation
└── 9. Test: feed one simulated turn through Modules 1→2→3→4, print turn_signal

Phase 3 — Intelligence Layer (Week 3, Raghav)
├── 10. Module 5 Skill Ontology Graph
├── 11. Module 6 Belief Updater (rule-based first)
├── 12. Module 8 LLM Question Generator
└── 13. End-to-end loop: one complete question-answer turn

Phase 4 — Novel Modules (Week 4)
├── 14. Module 10 Cognitive Load Classifier
├── 15. Module 11 Anti-Gaming
├── 16. rl/reward.py with unit tests
├── 17. config/rl_spec.py (Krissh delivers to Raghav)
└── 18. Module 7 RL Policy — simulated rollout training

Phase 5 — Integration (Week 5)
├── 19. FastAPI backend — /start-session, /process-turn, /end-session
├── 20. Redis queue setup
├── 21. React frontend — WebRTC audio/video capture
├── 22. Module 4 Fusion V2 — attention transformer
└── 23. End-to-end full interview test
```

---

## 10. Testing Requirements

Every module must have a test in `tests/`. Tests run with:
```bash
pytest tests/
```

**Minimum tests per module:**

`test_stt.py`:
- Transcribe a 10-second audio clip, assert transcript is non-empty string
- Assert word_timestamps list is non-empty
- Assert response_latency_ms is a positive float

`test_vision.py`:
- Process a single frame, assert all keys present in output dict
- Assert emotion_label is one of the 5 valid labels
- Assert gaze_vector has yaw and pitch keys

`test_prosody.py`:
- Extract features from a 30-second audio clip
- Assert mfcc_vector has 13 elements
- Assert speech_rate is positive float

`test_reward.py`:
- Unit test each reward term independently
- Test that conclude_interview action is masked when entropy is above threshold
- Test that reward is higher when belief entropy decreases more

---

## 11. Datasets — Real World Sources

Use these exact datasets. Download and place in `data/datasets/`.

### For Module 2 (Vision) Training:
| Dataset | What it provides | Access |
|---|---|---|
| **AFEW-VA** | Valence/arousal video in-the-wild | Email authors |
| **BP4D+** | AU labels, facial action, 2D+3D video | Request form (academic) |
| **MAHNOB-HCI** | Multimodal affect, gaze, AU synchronized | Free academic download |
| **RECOLA** | Multimodal affect in real dyadic interactions | Free academic download |

### For Module 3 (Prosody) Training:
| Dataset | What it provides | Access |
|---|---|---|
| **MSP-PODCAST** | Natural speech prosody, emotion labels | Free academic |
| **IEMOCAP** | Emotional speech, prosody annotated | Free academic (USC) |
| **RAVDESS** | Acted emotional speech + facial video | Zenodo direct download |
| **DAIC-WOZ** | Clinical interview speech, stress, depression | Request form |

### For Module 4 (Fusion) + Interview Context:
| Dataset | What it provides | Access |
|---|---|---|
| **MIT Interview Dataset** | Real job interview video, hirability scores | Email MIT Media Lab |
| **ChaLearn APA** | Personality, interview, multimodal | Direct download |
| **FG-NET Talking Face** | Talking head video | Direct download |

### For Module 10 (Cognitive Load):
| Dataset | What it provides | Access |
|---|---|---|
| **CLAS** | Cognitive load during tasks, physiological | Free academic |
| **DEAP** | EEG + video, arousal/valence | Request form |
| **Stroop Task Dataset** | Cognitive load speech + video | Various Zenodo entries |

### For Module 11 (Anti-Gaming / Deception):
| Dataset | What it provides | Access |
|---|---|---|
| **MU3D** | Deception detection video | Request form |
| **Real-life Trial Dataset** | Deception in real scenarios | Direct download |
| **DOLOS** | Online exam cheating gaze patterns | Email authors |

---

## 12. Common Errors and Fixes

```
Error: CUDA out of memory
Fix:   Reduce batch size. Use compute_type="int8" for whisper. 
       Do not load all models simultaneously at import time — lazy load.

Error: MediaPipe no face detected
Fix:   Add face detection confidence threshold check. 
       If no face: return None for that frame. Skip frame in summary.

Error: DeepFace AttributeError on emotion
Fix:   Pin deepface==0.0.79. Later versions change the output schema.

Error: faster-whisper hallucination on silence
Fix:   Ensure vad_filter=True in transcribe() call.

Error: SpeechBrain model download fails
Fix:   Set SPEECHBRAIN_DIR in .env to a local cache path.
       Models download on first use — needs internet access.

Error: torch not finding CUDA
Fix:   Confirm: torch.cuda.is_available() returns True
       If False: reinstall torch with correct CUDA version flag.
```

---

## 13. What NOT to Do

- Do NOT use Python 3.12 or 3.13 — use 3.11.9 only
- Do NOT hardcode file paths — use config/settings.py
- Do NOT load models inside functions that are called per-frame — load once at module init
- Do NOT use paid APIs (OpenAI, Google Vision, etc.) — everything runs locally
- Do NOT change the turn_signal schema without updating both Krissh and Raghav's modules
- Do NOT commit .env or data/datasets/ to git
- Do NOT build the attention fusion (V2) before the concat fusion (V1) works end-to-end
- Do NOT start RL training before rl_spec.py is finalised and agreed upon

---

## 14. Key Terminology

| Term | Meaning in ARIA context |
|---|---|
| Turn | One complete question-answer exchange |
| Belief state | Probability distribution over competency levels per skill node |
| Entropy | Measure of uncertainty in belief state — low entropy = confident assessment |
| Fused vector | Fixed-dimension representation of one turn combining all 3 modalities |
| Cognitive load | Mental effort level — separated into anxiety vs ignorance |
| Information gain | KL divergence between belief state before and after a turn |
| Ontology graph | Directed prerequisite graph of skills for a job role |
| IQL | Implicit Q-Learning — offline RL algorithm used for policy training |
| POMDP | Partially Observable Markov Decision Process — the math framework |
| Action mask | Rule that prevents invalid actions in certain states |

---

*Guide prepared: June 2026*
*Project: ARIA — Autonomous Reinforcement-based Interview Agent*
*Team: Krissh (Perception + RL Formulation) & Raghav Sejpal (Intelligence + Backend)*
*Institution: VIT Vellore*
*Patent deadline: July 17, 2026 — DO NOT publicly share before filing*
