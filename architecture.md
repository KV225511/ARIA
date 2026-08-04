# ARIA — System Architecture & Technical Specification
### Autonomous Reinforcement-based Interview Agent with Multimodal Adaptive Assessment

This document provides the definitive, comprehensive architectural specification for **ARIA (Advanced Real-world Incongruence & Affect / Autonomous Reinforcement-based Interview Agent)**. It details the theoretical foundations, real-time data flows, module responsibilities, hardware constraints, and integration contracts across the entire system.

---

## Table of Contents
1. [Executive Summary & System Vision](#1-executive-summary--system-vision)
2. [Mathematical Foundation: The POMDP Formulation](#2-mathematical-foundation-the-pomdp-formulation)
3. [The 15-Module Architecture Deep Dive](#3-the-15-module-architecture-deep-dive)
   - [Perception & Signal Layer (Modules 1–4)](#31-perception--signal-layer-modules-14)
   - [Intelligence & Control Layer (Modules 5–8)](#32-intelligence--control-layer-modules-58)
   - [Synthesis & Interaction Layer (Module 9)](#33-synthesis--interaction-layer-module-9)
   - [Cognitive & Behavioral Analysis Layer (Modules 10–13)](#34-cognitive--behavioral-analysis-layer-modules-1013)
   - [Evaluation & Feedback Layer (Modules 14–15)](#35-evaluation--feedback-layer-modules-1415)
4. [Real-Time Data Flow & Turn Orchestration](#4-real-time-data-flow--turn-orchestration)
5. [The 72-Dimension Multimodal Fusion Schema](#5-the-72-dimension-multimodal-fusion-schema)
6. [Hardware Budget & VRAM Allocation](#6-hardware-budget--vram-allocation)

---

## 1. Executive Summary & System Vision

ARIA is an autonomous, real-time multimodal AI interviewer designed to replace rigid, scripted post-hoc video screening tools (such as HireVue) with an adaptive, conversational screening agent. 

Unlike traditional platforms that simply record answers for offline human evaluation, ARIA treats the interview itself as a **real-time closed-loop inference problem**. As a candidate speaks, ARIA simultaneously captures audio, video, and text streams via WebRTC. It extracts linguistic meaning, vocal tone, facial micro-expressions, gaze vectors, and physiological stress signals.

### Key Architectural Differentiators:
- **Decoupling WHAT to ask from HOW to ask:** An offline-trained Reinforcement Learning (RL) policy decides *what* action to take next (e.g., increase difficulty, probe foundational concepts, switch topic), while an LLM (Llama 3.1 70B / Qwen2.5-72B) dynamically crafts *how* to phrase the question naturally.
- **POMDP Competency Tracking:** Candidate competency is treated as a hidden state over an occupational skill graph, updated via Bayesian inference after every conversational turn.
- **Cognitive Load Separation:** Differentiates between a candidate who *does not know* an answer (ignorance) and a candidate who *knows the answer but is experiencing high anxiety*.
- **Cross-Modal Incongruence Detection:** Real-time detection of bluffing by comparing vocal/behavioral confidence against semantic depth.
- **Anti-Gaming Integrity Monitoring:** Tracks off-camera eye sweeps (note reading), latency uniformity (AI assistance), and acoustic/semantic scripting.

---

## 2. Mathematical Foundation: The POMDP Formulation

A job interview is formally modeled as a **Partially Observable Markov Decision Process (POMDP)**. Because an interviewer can never directly observe a candidate's true underlying technical expertise, it must infer competency through noisy, multimodal behavioral and linguistic observations.

### The 6 POMDP Tuple Components:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O} \rangle$$

1. **State Space ($\mathcal{S}$):** The candidate's true competency level across a directed graph of domain skills. Each skill node $k$ exists in a hidden latent state $s_k \in \{\text{Beginner}, \text{Intermediate}, \text{Expert}\}$.
2. **Action Space ($\mathcal{A}$):** A discrete set of 8 interview control actions executed by the policy agent:
   - `increase_difficulty`: Probe deeper into the current topic at a higher technical tier.
   - `decrease_difficulty`: Simplify the question or ask for basic definitions.
   - `ask_follow_up_same_topic`: Clarify or request elaboration on the previous answer.
   - `switch_topic`: Move to an adjacent or unvisited node in the skill graph.
   - `probe_foundation`: Traverse backward along prerequisite graph edges to check foundational knowledge.
   - `ask_behavioral`: Switch to STAR-format situational/behavioral assessment.
   - `ask_situational`: Present a hypothetical architectural or workplace scenario.
   - `conclude_interview`: Terminate the session once diagnostic certainty is achieved.
3. **Observation Space ($\Omega$):** The multi-sensor turn signal $o_t$ produced at the end of turn $t$, comprising the 72-dimension fused feature vector, transcript semantic score, cognitive load label, and integrity flags.
4. **Belief State ($\mathcal{B}$):** The probability distribution over true competency states for each skill node $k$:
   $$b_t(k) = [P(\text{Beginner}), P(\text{Intermediate}), P(\text{Expert})]$$
   At session initialization ($t=0$), all nodes start at maximum uncertainty: $b_0(k) = [0.333, 0.333, 0.333]$.
5. **Transition Probability ($\mathcal{T}$):** The Bayesian belief update mapping $b_{t+1} = \tau(b_t, a_t, o_{t+1})$ executed by Module 6.
6. **Reward Function ($\mathcal{R}$):** The scalar reward driven primarily by **information gain** (reduction in belief entropy):
   $$\mathcal{R}_t = \alpha \cdot \text{D}_{\text{KL}}(b_{t+1} \parallel b_t) - \beta \cdot \text{cost}(a_t) + \gamma \cdot \text{consistency}(o_{t+1}) - \delta \cdot \text{distress}(o_{t+1}) + \Omega \cdot \text{outcome\_alignment}$$

### Novel Termination Condition:
Rather than conducting a fixed number of questions (e.g., 10 questions), ARIA evaluates the Shannon entropy across all skill node belief distributions:
$$H(\mathcal{B}_t) = -\frac{1}{K}\sum_{k=1}^{K} \sum_{j \in \{\text{Beg, Mid, Exp}\}} b_t(k)_j \log_2 b_t(k)_j$$
When global entropy falls below the termination threshold ($H(\mathcal{B}_t) < 0.3$), the system triggers `conclude_interview`.

---

## 3. The 15-Module Architecture Deep Dive

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                               CANDIDATE INTERFACE                                 │
│              WebRTC Real-Time Audio (16kHz) & Video (500ms / 2 FPS)               │
└────────────────────────┬────────────────────────────────┬─────────────────────────┘
                         │ Audio Stream                   │ Video Frames
                         ▼                                ▼
┌─────────────────────────────────────────┐    ┌────────────────────────────────────┐
│      MODULE 1: SPEECH-TO-TEXT (STT)     │    │       MODULE 2: VISION ENGINE      │
│  faster-whisper (large-v3, int8, CUDA)  │    │  MediaPipe Face Mesh + DeepFace +  │
│  • Word timestamps & response latency   │    │  L2CS-Net (Gaze, AUs, Emotion)     │
│  • Automated TF-IDF Semantic Grader     │    │  • Temporal AU Sequence Tracker    │
└────────────────────┬────────────────────┘    └─────────────────┬──────────────────┘
                     │                                           │
                     ▼                                           │
┌─────────────────────────────────────────┐                      │
│       MODULE 3: PROSODY EXTRACTOR       │                      │
│  SpeechBrain + openSMILE + WavLM        │                      │
│  • F0 pitch, speech rate, jitter, MFCC  │                      │
│  • Personal baseline calibration (T≤2)  │                      │
└────────────────────┬────────────────────┘                      │
                     │                                           │
                     └─────────────────┬─────────────────────────┘
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                   MODULE 4: MULTIMODAL DYNAMIC FUSION ENGINE                      │
│  • Fuses Text (11d) + Vision (33d) + Prosody (28d) → 72-Dimension Canonical Vector│
│  • Cross-Modal Attention Gating Network (GMU) & Unweighted Concatenation Baseline │
│  • Dynamic Missing Modality Imputation (Historical Deque Means)                   │
└──────────────────────────────────────┬────────────────────────────────────────────┘
                                       │ Fused Vector (72d)
         ┌─────────────────────────────┼──────────────────────────────┐
         ▼                             ▼                              ▼
┌───────────────────┐       ┌─────────────────────┐        ┌────────────────────────┐
│ MODULE 10:        │       │ MODULE 11:          │        │ MODULE 12:             │
│ COGNITIVE LOAD    │       │ ANTI-GAMING MONITOR │        │ INCONGRUENCE DETECTOR  │
│ • Low Load        │       │ • Note Reading      │        │ • Vocal Confidence vs. │
│ • Anxiety         │       │ • AI Assistance     │        │   Semantic Depth Delta │
│ • Ignorance       │       │ • Coaching/Scripted │        │ • Bluffing Penalties   │
└────────┬──────────┘       └──────────┬──────────┘        └───────────┬────────────┘
         │                             │                               │
         └─────────────────────────────┼───────────────────────────────┘
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                   MODULE 6: COMPETENCY BELIEF UPDATER                             │
│  • Bayesian updating over Module 5 Skill Ontology Graph (NetworkX + O*NET)        │
│  • Evaluates belief entropy H(B) against termination threshold (0.3)              │
└──────────────────────────────────────┬────────────────────────────────────────────┘
                                       │ Updated Belief State B(t)
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                   MODULE 7: RL INTERVIEW POLICY AGENT                             │
│  • Implicit Q-Learning (IQL via Stable-Baselines3) trained on offline rollouts    │
│  • Selects discrete action a(t) ∈ {increase_diff, probe_found, switch_topic...}   │
└──────────────────────────────────────┬────────────────────────────────────────────┘
                                       │ Action Target & Skill Node
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                   MODULE 8: LLM QUESTION GENERATOR                                │
│  • Llama 3.1 70B / Qwen2.5-72B via Ollama (Instruction-Following)                 │
│  • Context: Resume NER + Action Target + Belief State + Conversation History      │
└──────────────────────────────────────┬────────────────────────────────────────────┘
                                       │ Natural Language Question String
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                   MODULE 9: TTS + AVATAR SYNTHESIS                                │
│  • Coqui XTTS-v2 / Kokoro Audio Synthesis + SadTalker Talking Head Video         │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1 Perception & Signal Layer (Modules 1–4)

#### Module 1: Speech-to-Text (STT) & Semantic Grader
- **File:** `modules/module_01_stt/transcriber.py`, `semantic_grader.py`
- **Owner:** Krissh
- **Tooling:** `faster-whisper` (`large-v3`, `int8` quantization, `cuda` device, VAD filtering enabled), `scikit-learn`.
- **Responsibilities:**
  - Converts incoming 16kHz mono audio into transcripts with word-level timestamps and response latency tracking.
  - **Automated Semantic Grader (`semantic_grader.py`):** Evaluates candidate answers against domain reference answers using a hybrid rubric:
    - **70% Weight:** TF-IDF N-Gram (1–2) cosine similarity between candidate response and expert reference.
    - **30% Weight:** Regex word-boundary keyword coverage against mandatory technical concepts.
  - **Thread-Safety:** Instantiates a fresh `TfidfVectorizer` per grading call to prevent race conditions during concurrent turn evaluations.

#### Module 2: Vision Engine & Temporal AU Tracking
- **File:** `modules/module_02_vision/vision_processor.py`, `face_mesh.py`, `emotion.py`, `gaze.py`, `temporal_au.py`, `baseline.py`
- **Owner:** Krissh
- **Tooling:** `MediaPipe Tasks` (Face Landmarker), `DeepFace` (pinned to `0.0.79`), `L2CS-Net` (PyTorch).
- **Responsibilities:**
  - Processes video frames captured every 500ms (2 FPS). Uses `ThreadPoolExecutor` to run Mesh, Emotion, and Gaze analyzers in parallel.
  - **Landmarks & AUs:** Extracts 468 3D facial landmarks and 9 Action Units (`AU1, AU2, AU4, AU6, AU12, AU15, AU17, AU23, AU25`).
  - **Emotion Remapping:** Maps generic DeepFace emotion outputs to interview-specific behavioral states:
    - `happy` $\rightarrow$ `engaged` | `neutral` $\rightarrow$ `blank` | `fear`/`sad`/`angry` $\rightarrow$ `nervous` | `surprise` $\rightarrow$ `confused`
  - **Gaze & Pose:** Estimates 3D head pose (`roll, pitch, yaw`) and eye gaze angles (`yaw, pitch`) via L2CS-Net to compute eye-contact scores.
  - **Temporal AU Sequence Tracking (`temporal_au.py`):** Evaluates dynamic micro-expressions across the $T \times 15$ turn matrix. Computes onset velocities ($\Delta \text{AU} / \Delta t$) and AU variance across frames to differentiate genuine facial expressions from rigid or rehearsed responses.

#### Module 3: Prosody Extractor & Baseline Calibration
- **File:** `modules/module_03_prosody/extractor.py`, `baseline.py`, `pipeline.py`
- **Owner:** Krissh
- **Tooling:** `openSMILE` (eGeMAPSv02 feature set), `librosa` (MFCCs), `transformers` (`microsoft/wavlm-base-plus`).
- **Responsibilities:**
  - Extracts acoustic features from speech waveforms: mean/variance F0 pitch, speech rate (syllables/sec), pause frequency/duration, disfluency counts (um/uh/erm timestamps), RMS energy, jitter, shimmer, and 13 MFCC coefficients.
  - **WavLM Embeddings:** Generates 768-dimension temporal average-pooled self-supervised acoustic embeddings to capture subtle vocal timbre and stress nuances.
  - **Personal Baseline Calibration (`baseline.py`):** During conversational turns $T \in \{1, 2\}$ (icebreaker/introductory questions), stores candidate acoustic metrics as a personal baseline. For turns $T \ge 3$, computes relative deviations: $\Delta x = (x_{\text{current}} - x_{\text{baseline}}) / x_{\text{baseline}}$.

#### Module 4: Multimodal Dynamic Fusion Engine
- **File:** `modules/module_04_fusion/fusion_engine.py`, `attention_fusion.py`, `concat_fusion.py`, `schema.py`, `normalizer.py`
- **Owner:** Krissh
- **Tooling:** PyTorch, Custom Gated Multimodal Units (GMU).
- **Responsibilities:**
  - **Canonical Alignment (`schema.py`):** Aligns heterogeneous sensor outputs into a strict **72-dimension** feature vector (`FUSED_VECTOR_DIM = 72`).
  - **Dynamic Attention Gating (`attention_fusion.py`):** Replaces static concatenation with a learnable cross-modal attention network. Evaluates pairwise cosine dissonance across active modalities; lowers attention logits for conflicting sensors without penalizing agreeing channels.
  - **Scale-Preserving Gating:** Multiplies softmax weights by active modality count ($g_m = w_m \times N_{\text{active}}$) so that feature magnitudes remain consistent regardless of sensor dropouts.
  - **Missing Modality Imputation (`normalizer.py`):** Maintains per-candidate historical deques (`history_size=5`). If a sensor disconnects (e.g., webcam failure or audio silence), dynamically imputes missing feature dimensions using historical means rather than zero-padding.
  - **Ablation Baseline (`concat_fusion.py`):** Provides unweighted concatenation ($w_m = 1/N_{\text{active}}$) for benchmarking against attention fusion.

---

### 3.2 Intelligence & Control Layer (Modules 5–8)

#### Module 5: Skill Ontology Graph
- **File:** `modules/module_05_ontology/skill_graph.py`
- **Owner:** Raghav
- **Tooling:** `NetworkX`, O*NET Occupational Database.
- **Responsibilities:**
  - Constructs a directed acyclic graph (DAG) representing domain skills and prerequisite dependencies (e.g., `SQL` $\rightarrow$ `Query Optimization` $\rightarrow$ `Database Indexing`).
  - Loaded dynamically at session start based on the candidate's target job role and spaCy NER resume parsing.
  - Guides RL exploration: when a candidate exhibits weakness in a node, the graph allows backward prerequisite traversal (`probe_foundation`).

#### Module 6: Competency Belief Updater
- **File:** `modules/module_06_belief/updater.py`
- **Owner:** Raghav & Krissh
- **Tooling:** Bayesian Inference Engine.
- **Responsibilities:**
  - Receives the 72-d fused vector, semantic grade, cognitive load label, and anti-gaming flags at the end of each turn.
  - Performs Bayesian updates across all touched nodes in the Module 5 ontology graph:
    $$P(S_k \mid o_1, \dots, o_t) = \frac{P(o_t \mid S_k) P(S_k \mid o_1, \dots, o_{t-1})}{\sum_{j} P(o_t \mid S_j) P(S_j \mid o_1, \dots, o_{t-1})}$$
  - Computes global belief entropy and signals session termination when uncertainty drops below threshold.

#### Module 7: RL Interview Policy Agent
- **File:** `modules/module_07_rl/policy.py`, `rl/reward.py`, `rl/state_builder.py`, `rl/action_masker.py`, `config/rl_spec.py`
- **Owner:** Raghav (Training) & Krissh (Formulation/Specs)
- **Tooling:** `Stable-Baselines3`, Implicit Q-Learning (IQL).
- **Responsibilities:**
  - Maps the current RL state vector (belief distribution, belief entropy, 72-d fused vector, cognitive load label, distress score, anti-gaming status, turn ID, and topic history) to one of 8 discrete actions.
  - **Action Masking (`action_masker.py`):** Prevents invalid actions (e.g., masking `conclude_interview` if entropy $> 0.3$; masking `probe_foundation` if no prerequisite exists; masking `decrease_difficulty` during baseline turns $T \le 2$).
  - **Offline RL Training:** Trained via IQL on synthetic LLM rollouts (300+ episodes across diverse candidate personas) and calibrated against MIT Interview Dataset ratings.

#### Module 8: LLM Question Generator
- **File:** `modules/module_08_llm/question_generator.py`
- **Owner:** Raghav
- **Tooling:** `Ollama` running `Llama 3.1 70B` or `Qwen2.5-72B`.
- **Responsibilities:**
  - Decoupled natural language generation. Translates the abstract RL action (e.g., `increase_difficulty` on node `JWT Authentication`) into an empathetic, context-aware interview question.
  - Integrates candidate resume context, past conversation turns, and current belief states into the system prompt to ensure continuity and avoid repetition.

---

### 3.3 Synthesis & Interaction Layer (Module 9)

#### Module 9: TTS & Avatar Synthesis
- **File:** `modules/module_09_tts/synthesizer.py`, `avatar.py`
- **Owner:** Raghav
- **Tooling:** `Coqui XTTS-v2` / `Kokoro` (Speech Synthesis), `SadTalker` (Talking Head Animation).
- **Responsibilities:**
  - Converts the LLM-generated question string into natural audio waveforms with consistent interviewer voice persona.
  - Generates lip-synchronized talking-head video frames streamed back to the candidate frontend via WebRTC.

---

### 3.4 Cognitive & Behavioral Analysis Layer (Modules 10–13)

#### Module 10: Cognitive Load Separator
- **File:** `modules/module_10_cognitive_load/classifier.py`
- **Owner:** Krissh
- **Responsibilities:**
  - Resolves the ambiguity between genuine knowledge gaps and anxiety-induced performance degradation.
  - Evaluates latency, speech rate deviations, vocal jitter, gaze breaks, and eventual semantic score:
    - **High Load + High Semantic Score** $\rightarrow$ `anxiety` (Candidate knows the material but is nervous; downstream behavioral penalties are suppressed).
    - **High Load + Low Semantic Score** $\rightarrow$ `ignorance` (Candidate lacks technical competency; standard scoring applies).
    - **Low Load + Low Semantic Score** $\rightarrow$ `confident_ignorance` (Bluffing or incorrect assumptions).
    - **Low Load + High Semantic Score** $\rightarrow$ `low` (Optimal mastery).

#### Module 11: Anti-Gaming & Integrity Monitor
- **File:** `modules/module_11_anti_gaming/gaze_scanner.py`, `latency_checker.py`, `semantic_checker.py`
- **Owner:** Krissh
- **Responsibilities:**
  - **Note Reading (`gaze_scanner.py`):** Detects horizontal left-to-right eye saccades across consecutive frames exceeding 2.0 seconds.
  - **AI Assistance (`latency_checker.py`):** Identifies suspicious acoustic delivery characterized by prolonged pre-answer latency followed by unnaturally uniform speech rate (low variance in words/sec).
  - **Coaching/Scripting (`semantic_checker.py`):** Detects off-camera lateral head turns coupled with high semantic similarity between answers to unrelated questions (via BGE-M3 embeddings) or abrupt lexical complexity shifts.

#### Module 12: Cross-Modal Incongruence Detector
- **File:** `modules/module_12_incongruence/` (or integrated in intelligence layer)
- **Owner:** Raghav
- **Tooling:** Cross-Modal Gated Attention Network (`BGE-M3` + `openSMILE`).
- **Responsibilities:**
  - Directly compares vocal/prosodic confidence (speech rate, pitch stability, energy) against linguistic depth (cosine distance to expert reference).
  - Identifies bluffing (high vocal confidence + shallow semantic depth) and outputs an incongruence penalty penalty signal to the RL policy agent.

#### Module 13: Interview Fairness Auditor
- **File:** `modules/module_13_fairness/`
- **Owner:** Raghav
- **Responsibilities:**
  - Runs asynchronously across completed sessions to monitor policy exploration fairness.
  - Audits difficulty assignment distributions against demographic-adjacent acoustic traits (accent, pitch range, speaking rate) to ensure the RL policy does not systematically bias difficulty trajectories.

---

### 3.5 Evaluation & Feedback Layer (Modules 14–15)

#### Module 14: Evaluation Engine & Report Generator
- **File:** `modules/module_14_eval/`
- **Owner:** Raghav
- **Tooling:** `ReportLab` (PDF generation), `Llama 3.1` (Narrative synthesis).
- **Responsibilities:**
  - Aggregates turn-by-turn belief histories, semantic grades, cognitive load profiles, and integrity flags at session conclusion.
  - Compiles an 11-section PDF evaluation report: Technical Competency, Communication Analysis, Behavioral Assessment, Confidence Estimation, Emotional Timeline, Cognitive Load Breakdown, Integrity Audit, Fairness Summary, Narrative Strengths/Weaknesses, Hiring Recommendation, and Full Transcript.

#### Module 15: Self-Improving Feedback Loop
- **File:** Integrated in RL training pipeline.
- **Owner:** Raghav & Krissh
- **Responsibilities:**
  - Ingests real-world hiring outcomes (hired / rejected / job performance metrics) as terminal reward signals ($\Omega$).
  - Periodically triggers offline policy retraining on accumulated session databases, allowing ARIA's interview strategy to continuously refine itself through real-world deployment.

---

## 4. Real-Time Data Flow & Turn Orchestration

The interview execution loop proceeds through a strict synchronous/asynchronous turn orchestration cycle:

```
[Candidate Speaking via WebRTC]
       │
       ├─► Audio Stream (16kHz Mono) ──► Faster-Whisper STT ──► Transcript & Latency
       │                                        │
       │                                        ▼
       │                                 Semantic Grader ─────► TF-IDF / Keyword Score (Mod 1)
       │                                        │
       ├─► Audio Stream (16kHz Mono) ──► Prosody Extractor ───► F0, MFCC, WavLM, Deviations (Mod 3)
       │
       └─► Video Frames (500ms/2fps) ──► Vision Processor ────► AUs, Emotion, Gaze, Eye Contact (Mod 2)
                                                │
                                                ▼
[Turn End Triggered by Silence / VAD]           │
       │                                        ▼
       └───────────────────────────────► Multimodal Fusion ───► 72-d Fused Vector (Mod 4)
                                                │
                                                ├──────────────────────────────┐
                                                ▼                              ▼
                                         Cognitive Load (Mod 10)     Anti-Gaming Monitor (Mod 11)
                                                │                              │
                                                └──────────────┬───────────────┘
                                                               ▼
                                                    Belief Updater (Mod 6)
                                                               │
                                                               ▼
                                                    RL Policy Agent (Mod 7) ──► Selects Action a(t)
                                                               │
                                                               ▼
                                                    LLM Question Gen (Mod 8) ─► Natural Text Question
                                                               │
                                                               ▼
                                                    TTS & Avatar (Mod 9) ─────► Video/Audio Response
```

1. **Continuous Capture:** While the candidate answers, audio is buffered in 16kHz float32 chunks and video is sampled at 2 FPS.
2. **Parallel Feature Extraction:** Upon Voice Activity Detection (VAD) silence detection indicating answer completion:
   - Module 1 transcribes text and computes response latency.
   - Module 2 aggregates frame-level landmark arrays into turn-average AU activations, gaze vectors, and emotion distributions.
   - Module 3 extracts acoustic features and computes relative percentage deviations against turns 1–2 baseline.
3. **Multimodal Fusion:** Module 4 maps the outputs into `schema.FULL_FEATURE_SCHEMA` and executes cross-modal attention gating to generate the 72-dimension vector.
4. **State Analysis:** Modules 10, 11, and 12 evaluate cognitive load, integrity flags, and incongruence simultaneously.
5. **Belief & Policy Update:** Module 6 updates competency probabilities on the ontology graph. Module 7 evaluates entropy; if $> 0.3$, it selects the next interview action.
6. **Question Synthesis:** Module 8 generates the contextual question string, which Module 9 synthesizes into audio and talking-head video for the candidate.

---

## 5. The 72-Dimension Multimodal Fusion Schema

Defined in `modules/module_04_fusion/schema.py`, the canonical feature vector rigorously standardizes all sensor inputs into **72 dimensions**:

$$\text{FUSED\_VECTOR\_DIM} = 11_{\text{text}} + 33_{\text{vision}} + 28_{\text{prosody}} = 72$$

| Modality | Index Range | Dim | Feature Components & Schema Mapping |
| :--- | :---: | :---: | :--- |
| **Text / Linguistic** | `[0 : 11]` | **11** | `transcript_length`, `word_count`, `response_latency_ms`, `semantic_similarity_score`, `keyword_coverage_score`, `complexity_index`, `sentiment_valence`, `sentiment_arousal`, `question_relevance`, `coherence_score`, `technical_depth` |
| **Vision / Behavioral** | `[11 : 44]` | **33** | `eye_contact_score`, `blink_rate`, `head_pose_roll/pitch/yaw`, `gaze_yaw/pitch`, **9 AU Activations** (`AU1, AU2, AU4, AU6, AU12, AU15, AU17, AU23, AU25`), **9 AU Relative Deviations**, **5 Emotion Probabilities** (`engaged, confused, nervous, confident, blank`), `facial_microexpression_variance`, `visual_stability` |
| **Prosody / Vocal** | `[44 : 72]` | **28** | `pitch_mean/variance/range`, `speech_rate`, `pause_count`, `pause_duration_ms`, `disfluency_count`, `energy_mean`, `jitter`, `shimmer`, `speech_to_silence_ratio`, **3 Baseline Deviations** (`pitch_dev, rate_dev, energy_dev`), `vocal_stress_index`, **13 MFCC Coefficients** (`mfcc_0` through `mfcc_12`) |

---

## 6. Hardware Budget & VRAM Allocation

ARIA is engineered to run locally on a single consumer-grade workstation without relying on paid external cloud APIs.

### Target Environment:
- **OS:** Windows 11
- **Python:** 3.11.9 (Strict requirement; incompatible with 3.12/3.13 due to PyTorch/MediaPipe C++ bindings)
- **GPU:** NVIDIA GeForce RTX 4060 — **8GB VRAM**
- **CUDA:** 13.0 / PyTorch CU121
- **Package Management:** `pip` only within `.venv` virtual environment

### Strict 8GB VRAM Budget Breakdown:
To prevent out-of-memory (OOM) exceptions during real-time multi-model execution, GPU VRAM is strictly budgeted:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA RTX 4060 — 8GB VRAM BUDGET                     │
├────────────────────────────────────────┬─────────────────────────────────────┤
│ Model / Component                      │ VRAM Allocation                     │
├────────────────────────────────────────┼─────────────────────────────────────┤
│ faster-whisper (large-v3, int8)        │ ~2.5 GB                             │
│ MediaPipe Face Mesh + DeepFace (0.0.79)│ ~1.5 GB                             │
│ L2CS-Net Gaze Estimation               │ ~0.5 GB                             │
│ Multimodal Attention Fusion Layer (GMU)│ ~0.5 GB                             │
│ System Headroom / PyTorch Workspace    │ ~3.0 GB                             │
├────────────────────────────────────────┼─────────────────────────────────────┤
│ TOTAL ALLOCATED                        │ ~5.0 GB / 8.0 GB (37.5% Free Buffer)│
└────────────────────────────────────────┴─────────────────────────────────────┘
```

**Optimization Rules:**
1. **INT8 Quantization:** `faster-whisper` must instantiate with `compute_type="int8"`.
2. **Lazy Loading:** Models are loaded into memory asynchronously on first use rather than globally at import time.
3. **Ollama LLM / TTS Offloading:** In standalone testing, Llama 3.1 70B and XTTS-v2 execute via Ollama/local server endpoints or dedicated background threads utilizing system RAM and CPU inference when VRAM headroom is constrained.

---
*End of System Architecture & Technical Specification.*
