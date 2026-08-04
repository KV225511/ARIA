# ARIA — Complete Master Document
### Autonomous Reinforcement-based Interview Agent with Multimodal Adaptive Assessment
**Author:** Raghav Sejpal (23BAI0095), VIT Vellore | B.Tech AI & ML, Year 3
**Team:** Raghav Sejpal + Krissh
**Target:** IEEE Publication + Provisional Patent Filing (before July 17, 2026)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Core Novelty and Differentiators](#3-core-novelty-and-differentiators)
4. [POMDP Formulation](#4-pomdp-formulation)
5. [System Architecture — All 13 Modules](#5-system-architecture--all-13-modules)
6. [Architecture Diagram](#6-architecture-diagram)
7. [Reinforcement Learning — Deep Dive](#7-reinforcement-learning--deep-dive)
8. [The 6 Baseline Models](#8-the-6-baseline-models)
9. [Real-World Datasets — All 6 Models](#9-real-world-datasets--all-6-models)
10. [Baseline Model Results](#10-baseline-model-results)
11. [Open Source Tech Stack](#11-open-source-tech-stack)
12. [Work Division](#12-work-division)
13. [Interface Contract (Krissh → Raghav)](#13-interface-contract-krissh--raghav)
14. [Step-by-Step Build Guide](#14-step-by-step-build-guide)
15. [Where Training Happens](#15-where-training-happens)
16. [Project Schedule](#16-project-schedule)
17. [Patent Strategy](#17-patent-strategy)
18. [Research Contributions](#18-research-contributions)

---

## 1. Project Overview

ARIA is an autonomous AI interviewer capable of conducting realistic, adaptive job interviews using real-time voice, video, behavioral analysis, and conversational reasoning — without any human intervention.

A candidate joins the platform, turns on their webcam and microphone, uploads their resume, and speaks naturally with the AI. The AI asks questions verbally, listens to responses, watches facial expressions and body language, analyzes tone and confidence from voice, understands the semantic meaning of answers using NLP, and continuously decides what to ask next.

The interview is not scripted. The AI dynamically adapts based on candidate performance. If the candidate answers confidently and correctly, the AI increases difficulty and asks deeper follow-up questions. If the candidate struggles, it simplifies questions, switches topics, or probes foundational understanding. The system maintains memory of the entire conversation and reasons over the candidate's strengths, weaknesses, confidence level, hesitation patterns, communication skills, and technical understanding.

At the end, the platform generates a full evaluation report covering technical competency, communication analysis, behavioral assessment, confidence estimation, emotional consistency, integrity flags, fairness audit, and a hiring recommendation.

**One-line description:**
An autonomous AI interviewer capable of conducting realistic adaptive interviews using real-time voice, video, behavioral analysis, and conversational reasoning to evaluate candidates intelligently without human intervention.

---

## 2. Problem Statement

Existing AI interview platforms perform post-hoc analysis — they record, then evaluate. None treat the interview itself as a dynamic inference problem. Human recruiters adapt in real-time based on what they observe; current systems do not.

Problems with existing systems:
- Rigid, scripted assessments that fail to measure true competency
- Easily gamed by candidates who know the fixed question structure
- No separation between anxiety-induced errors and genuine knowledge gaps
- No real-time incongruence detection between behavioral confidence and semantic depth
- No auditing of AI decision-making for fairness

**Prior art that must be differentiated from:**
- HireVue: post-hoc analysis, no adaptive control, no RL, no POMDP
- Karat / Interviewing.io: human-assisted, not autonomous
- Academic AVEC-style papers: analysis only, no adaptive interview control loop

---

## 3. Core Novelty and Differentiators

The central innovation is formalising a job interview as a **Partially Observable Markov Decision Process (POMDP)** where the AI agent's goal is to minimise uncertainty over a candidate's true competency distribution — not simply to ask questions — using a reinforcement-learning-trained policy over real-time multimodal observations.

**7 Specific Novel Contributions:**

1. First formalisation of a job interview as a POMDP with belief state inference over a skill ontology graph
2. RL policy trained via information-gain reward — using belief entropy reduction as the primary reward signal
3. Skill ontology graph traversal driven by belief state updates to find root competency gaps
4. Cross-modal incongruence detection — comparing prosodic confidence against semantic depth to catch bluffing
5. Cognitive load separation — distinguishing anxiety-induced errors from genuine knowledge gaps
6. Real-time multimodal integrity monitoring — detecting note reading, AI assistance, off-camera coaching
7. Self-improving closed-loop system — hiring outcomes as terminal RL reward, enabling continuous policy improvement

---

## 4. POMDP Formulation

A POMDP is the right formalization because the interviewer never directly observes the candidate's true competency — it only observes noisy, partial signals. This mirrors exactly what a human interviewer does: infer true ability from imperfect behavioral observations.

```
State (S)       — candidate's true competency level per skill node (hidden, never directly observed)
Observation (O) — multimodal signals per turn: transcript, facial expression, gaze, prosody features
Action (A)      — interview control actions: what to ask next, how hard, which topic
Reward (R)      — information gain about true competency + efficiency + signal consistency - distress
Belief (b)      — running probability distribution P(skill_level | observations so far)
Transition (T)  — how belief state updates after each candidate turn
```

**Belief State:**
For every node in the skill ontology graph, the system maintains:
```
b(node) = [P(beginner), P(mid), P(expert)]
```
All three initialised to [0.33, 0.33, 0.33] at session start. Updated after every turn using Bayesian inference over the fused multimodal signal.

**Novel Termination Condition:**
The interview ends when entropy across all belief distributions falls below a threshold — meaning the system has sufficient confidence in its assessment. The interview ends when the AI knows enough, not after a fixed number of questions.

---

## 5. System Architecture — All 13 Modules

### Module 1 — STT (Speech-to-Text)
**Tool:** faster-whisper running Whisper large-v3 in streaming mode
**Input:** raw audio stream from candidate microphone
**Output:** text transcript per conversational turn
**Notes:** Streaming mode means partial transcripts are available before the candidate finishes speaking. Runs as an async task. Owned by Krissh.

---

### Module 2 — Vision Module
**Tools:** MediaPipe Face Mesh + DeepFace + L2CS-Net
**Input:** video frames at 500ms intervals from candidate webcam
**Outputs:**
- 468 facial landmark coordinates per frame (MediaPipe)
- Action Unit activations: AU1, AU2, AU4, AU6, AU12, AU15, AU17, AU23, AU25 (DeepFace)
- Emotion classification: interview-context labels (engaged, confused, nervous, confident, blank)
- Gaze direction vector: yaw and pitch per eye (L2CS-Net)
- Eye contact score: binary per frame, averaged per turn
- Head pose: roll, pitch, yaw
- Blink rate and duration
**Owned by Krissh.**

---

### Module 3 — Prosody Module
**Tools:** SpeechBrain / openSMILE
**Input:** raw audio per candidate turn
**Outputs:**
- Pitch (F0) contour and variance
- Speech rate in syllables per second
- Pause duration and frequency
- Energy and loudness envelope
- Disfluency markers with timestamps (um, uh, erm counts)
- Voice jitter and shimmer
- MFCCs (13–40 coefficients)
- Response latency: time from question end to answer start
**Notes:** First two turns establish personal baseline — all subsequent turns compared against it. Owned by Krissh.

---

### Module 4 — Multimodal Fusion Layer
**Architecture:** Cross-modal attention transformer
**Input:** transcript + vision snapshot + prosody vector per conversational turn
**Output:** single fused signal vector per turn
**Notes:** Initially implemented as simple concatenation. Upgraded to cross-attention transformer once individual pipelines are stable. The transformer learns to weight each modality differently depending on interview phase. Owned by Krissh.

---

### Module 5 — Skill Ontology Graph
**Tool:** NetworkX + O*NET occupational data
**Structure:** Directed graph where nodes are skills, edges represent prerequisite relationships

**Example — Backend Developer:**
```
REST API → Authentication → JWT → Session Management → Security Architecture
SQL → Query Optimization → Indexing → Database Design
Functions → OOP → Design Patterns → System Design
Linux Basics → Docker → Kubernetes
Docker → CI/CD
```

**Example — Data Scientist:**
```
Statistics → Hypothesis Testing → A/B Testing
Linear Algebra → ML Fundamentals → Supervised Learning → Ensemble Methods
Python → Pandas → Data Cleaning → Feature Engineering → Model Training → Model Evaluation → MLOps
```

**Notes:** Loaded at session start based on job role. RL agent navigates this graph — weak nodes trigger backward traversal to find root gaps. Built from O*NET data + resume-extracted skills. Owned by Raghav.

---

### Module 6 — Competency Belief Updater *(Novel Component 1)*
**Approach:** Bayesian inference
**Input:** fused signal vector per turn + semantic evaluation of answer
**Output:** updated probability distribution [P(beginner), P(mid), P(expert)] for every skill node touched in that turn
**Initial state:** [0.33, 0.33, 0.33] for all nodes

**Update logic:**
- Strong semantic answer + confident prosody + stable gaze → shift toward expert
- Shallow semantic + high pause + gaze breaks → shift toward beginner
- High cognitive load flag → weight semantic score more than behavioral signals

**Termination signal:** Global entropy across all belief distributions — interview terminates when this falls below threshold (0.3 in baseline).
**Owned by Raghav and krissh.**

---

### Module 7 — RL Interview Policy Agent *(Core Novel Contribution)*
**Input:** current belief state vector + fused signal vector
**Output:** one of 8 discrete interview actions
**Training:** Offline RL using IQL on LLM-simulated candidate rollouts, then fine-tuned on real interview data via self-improving loop
**Owned by Raghav and krissh.**
*(See Section 7 for full RL deep dive)*

---

### Module 8 — LLM Question Generator
**Model:** Llama 3.1 70B or Qwen2.5-72B (via Ollama)
**Input:** RL action + belief state + full conversation history + resume context
**Output:** natural language question tailored to the candidate
**Critical design point:** The LLM decides HOW to ask. The RL agent decides WHAT to ask. These are explicitly decoupled — this is architecturally important and highlighted in the paper.
**System prompt includes:** extracted resume, current belief state per skill node, instruction from RL agent, full conversation history.
**Owned by Raghav.**

---

### Module 9 — TTS + Avatar Layer
**Tools:** Coqui XTTS-v2 or Kokoro (TTS) + SadTalker (avatar)
**Input:** question text from LLM question generator
**Output:** audio speech + animated talking head video
**Notes:** Consistent voice and face persona throughout session. pyttsx3 used as baseline placeholder. Owned by Raghav.

---

### Module 10 — Cognitive Load Separation Module *(Novel Component 2)*
**Problem solved:** Separates two failure modes that look identical on the surface — candidate does not know the answer, and candidate knows but is anxious.

**Input per turn:**
- Response latency (ms from question end to answer start)
- Disfluency count (um, uh frequency)
- Gaze break frequency and duration during the answer
- Speech rate drop compared to personal baseline
- Eventual answer quality (semantic score after full answer)

**Output:** cognitive load tag: {low_load, high_load_anxiety, high_load_ignorance}

**Logic:**
- High load + high eventual semantic score → anxiety tag
- High load + low eventual semantic score → ignorance tag
- Low load + low semantic score → confident ignorance

**Effect on scoring:** Anxiety-tagged turns scored differently — behavioral penalties reduced, semantic content weighted more heavily. Owned by Krissh.

---

### Module 11 — Anti-Gaming / Integrity Module *(Novel Component 3)*
**Three parallel detectors:**

**Detector A — Note Reading:**
Signal: gaze off-screen with horizontal scan pattern for more than 2 seconds

**Detector B — AI Assistance:**
Signal 1: unusually long pre-answer latency
Signal 2: unusually polished answer following the long pause
Signal 3: vocabulary complexity inconsistency across session

**Detector C — Scripted / Coaching:**
Signal 1: lateral head turns + micro-pause before answering
Signal 2: semantic similarity between answers to unrelated questions is suspiciously high
Signal 3: delivery rate too uniform and too fast

**Output:** per-session integrity report with flags, confidence scores, and evidence. None auto-reject — all disclosed in report for hiring manager review.
**Owned by Krissh.**

---

### Module 12 — Cross-Modal Incongruence Detector *(Novel Component 4)*
**Input per turn:**
- Prosodic confidence score (high speech rate, low pause, stable pitch → confident)
- Semantic depth score (cosine similarity between answer embedding and expert reference embedding via BGE-M3)

**Output:** incongruence flag + magnitude

**Logic:** When prosody says confident but semantic analysis says shallow, the delta is the bluffing signal.

**Effect:** Incongruence flag fed back to RL policy agent as additional penalty signal, triggering deeper probing rather than progression.
**Owned by Raghav.**

---

### Module 13 — Interview Fairness Auditor *(Novel Component 5)*
**Runs as:** background audit process across sessions, not per-turn

**What it monitors:**
- Distribution of difficulty levels assigned by RL agent across candidate segments
- Whether difficulty assignments correlate with demographic-adjacent signals (speech accent, pace, pitch range)
- Whether certain session characteristics consistently produce lower scores regardless of semantic quality

**Output:** per-session fairness report showing action distribution and any detected correlation between speech pattern features and difficulty assignment.
**Owned by Raghav.**

---

### Module 14 — Evaluation Engine + Report Generator
**Input:** all module outputs accumulated across the full session
**Processing:** Llama 3.1 generates narrative sections; rule-based scoring aggregates per-skill-node beliefs into section scores

**Output report sections:**
1. Technical Competency Score — per skill node + aggregate
2. Communication Analysis — coherence, fluency, vocabulary
3. Behavioral Assessment — engagement, consistency, adaptability
4. Confidence Estimation — prosody-derived, gaze-derived, combined
5. Emotional Consistency Timeline
6. Cognitive Load Profile — anxiety vs ignorance breakdown
7. Integrity Assessment — flags with evidence and confidence scores
8. Fairness Audit Summary
9. Strengths and Weaknesses — LLM-generated narrative
10. Hiring Recommendation with confidence interval
11. Full Interview Transcript

**Rendered as:** structured JSON + PDF (ReportLab). Owned by Raghav.

---

### Module 15 (bonus) — Self-Improving Feedback Loop *(Novel Component 6)*
**How it works:**
- After each real interview, hiring outcome fed back: hired / rejected / performed well / underperformed
- This becomes the terminal reward signal in the RL training pipeline
- Every N sessions, the RL policy is retrained on accumulated real-world session data
- System generates its own training data through deployment

---

## 6. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     CANDIDATE INTERFACE                     │
│          WebRTC (real-time audio + video stream)            │
│          Resume uploaded → parsed at session start          │
│          Job role selected → ontology graph loaded          │
└────────────┬────────────────────────┬───────────────────────┘
             │ audio stream           │ video frames (500ms)
    ┌────────▼──────────┐    ┌────────▼──────────────┐
    │  MODULE 1 — STT   │    │  MODULE 2 — VISION     │
    │  faster-whisper   │    │  MediaPipe + DeepFace  │
    │  Whisper large-v3 │    │  + L2CS-Net            │
    │  → transcript     │    │  → emotion, gaze,      │
    │    per turn       │    │    eye contact, pose   │
    └────────┬──────────┘    └────────┬──────────────┘
             │                        │
    ┌────────▼──────────┐    ┌────────▼──────────────┐
    │  MODULE 3         │    │  Gaze + Microexpression│
    │  PROSODY          │    │  Classifier            │
    │  SpeechBrain +    │    │  → interview-context   │
    │  openSMILE        │    │    labels              │
    │  → pitch, rate,   │    └────────┬──────────────┘
    │    pauses, stress │             │
    └────────┬──────────┘             │
             └──────────┬─────────────┘
                        │
    ┌───────────────────▼────────────────────────────┐
    │          MODULE 4 — MULTIMODAL FUSION          │
    │   Cross-modal attention transformer            │
    │   Start: simple concatenation                  │
    │   → single fused signal vector per turn        │
    └───────────────────┬────────────────────────────┘
                        │
    ┌───────────────────▼────────────────────────────┐
    │        MODULE 6 — COMPETENCY BELIEF UPDATER    │
    │   Bayesian update P(skill_level) per node      │
    │   in skill ontology graph after every turn     │
    │   Tracks belief entropy as termination signal  │
    └───────────────────┬────────────────────────────┘
                        │
    ┌───────────────────▼────────────────────────────┐
    │     MODULE 12 — INCONGRUENCE DETECTOR          │
    │   prosody confidence vs semantic depth delta   │
    │   Flags bluffing → penalty signal to RL agent  │
    └───────────────────┬────────────────────────────┘
                        │
    ┌───────────────────▼────────────────────────────┐
    │      MODULE 10 — COGNITIVE LOAD SEPARATOR      │
    │   low_load / high_load_anxiety / ignorance     │
    │   Adjusts scoring for anxious candidates       │
    └───────────────────┬────────────────────────────┘
                        │
    ┌───────────────────▼────────────────────────────┐
    │       MODULE 11 — ANTI-GAMING MODULE           │
    │   Note reading / AI assistance / coaching      │
    │   → integrity flags stored per turn            │
    └───────────────────┬────────────────────────────┘
                        │
    ┌───────────────────▼────────────────────────────┐
    │      MODULE 7 — RL INTERVIEW POLICY AGENT      │
    │   IQL-trained policy (Stable-Baselines3)       │
    │   Input: belief state + fused signal           │
    │   Output: next interview action (8 options)    │
    │   Reward: info gain - penalties                │
    └───────────────────┬────────────────────────────┘
                        │
    ┌───────────────────▼────────────────────────────┐
    │      MODULE 13 — FAIRNESS AUDITOR              │
    │   Background: monitors RL action distribution  │
    │   Detects bias correlated with speech patterns │
    └───────────────────┬────────────────────────────┘
                        │
    ┌───────────────────▼────────────────────────────┐
    │        MODULE 8 — LLM QUESTION GENERATOR       │
    │   Llama 3.1 70B / Qwen2.5-72B via Ollama      │
    │   Input: RL action + belief + history + resume │
    │   Output: natural language question            │
    └───────────────────┬────────────────────────────┘
                        │
    ┌───────────────────▼────────────────────────────┐
    │         MODULE 9 — TTS + AVATAR                │
    │   Coqui XTTS-v2 / Kokoro → speech             │
    │   SadTalker → talking head avatar              │
    └───────────────────┬────────────────────────────┘
                        │
                [end of session]
                        │
    ┌───────────────────▼────────────────────────────┐
    │      MODULE 14 — EVALUATION ENGINE + REPORT    │
    │   Aggregates all session module outputs        │
    │   Llama generates narrative sections           │
    │   ReportLab renders PDF                        │
    └───────────────────┬────────────────────────────┘
                        │
    ┌───────────────────▼────────────────────────────┐
    │       MODULE 15 — SELF-IMPROVING LOOP          │
    │   Hiring outcome → terminal RL reward          │
    │   Periodic policy retraining on real data      │
    └────────────────────────────────────────────────┘
```

---

## 7. Reinforcement Learning — Deep Dive

### Why RL and Not Just an LLM Deciding

An LLM deciding what to ask next is reactive. It sees the last answer and generates the next question. It has no objective, no optimisation target, and no memory of what information it still needs to gather.

The RL policy has an explicit goal: **minimise uncertainty about the candidate's true competency as efficiently as possible.** That is a fundamentally different mechanism.

### State Space

```python
state = {
    "belief_vector":       # P(skill_level) for every node in the ontology graph
                           # e.g., [P(JWT=beginner), P(JWT=mid), P(JWT=expert), ...]
    "belief_entropy":      # overall entropy across all skill nodes
    "fused_vector":        # fixed-dim output of fusion layer (from Krissh)
    "cognitive_load_label":# 0=low, 1=anxiety, 2=ignorance
    "distress_score":      # continuous distress estimate
    "anti_gaming_active":  # any flag currently raised
    "turn_id":             # integer session progress
    "topics_covered":      # skill nodes visited so far
    "consecutive_same_topic": # turns on current topic
    "incongruence_score":  # current prosody-semantic delta
    "answer_consistency":  # variance of semantic quality across turns
}
```

### Action Space (8 discrete actions)

```python
actions = [
    "increase_difficulty",       # same topic, harder question
    "decrease_difficulty",       # same topic, simpler question
    "ask_follow_up_same_topic",  # dig deeper into current topic
    "switch_topic",              # move to a different skill node
    "probe_foundation",          # go backward in ontology graph to prerequisite
    "ask_behavioral",            # switch to behavioral question (STAR format)
    "ask_situational",           # switch to hypothetical scenario question
    "conclude_interview"         # terminate — sufficient belief reduction achieved
]
```

### Reward Function

```
R(t) = α × information_gain(t)
     - β × duration_penalty(t)
     + γ × signal_consistency_bonus(t)
     - δ × candidate_distress_penalty(t)
     + ε × integrity_detection_bonus(t)     [terminal — only if flag confirmed]
     + Ω × outcome_alignment_reward         [terminal — from Dataset 5]
```

- **information_gain(t):** KL divergence between belief state before and after turn t. Primary reward driver.
- **duration_penalty(t):** Small negative per turn. Prevents unnecessary questions.
- **signal_consistency_bonus(t):** Positive when all three modalities agree on the candidate's state.
- **candidate_distress_penalty(t):** Negative when cognitive load and distress signals are simultaneously very high.
- **outcome_alignment_reward:** Terminal reward. Difference between system recommendation and actual hiring outcome. Ground truth the policy ultimately optimises toward.

### RL Coefficients (initial values from rl_spec.py)

```python
REWARD_COEFFICIENTS = {
    "alpha": float,   # information gain weight
    "beta": float,    # duration penalty weight
    "gamma": float,   # signal consistency bonus weight
    "delta": float,   # distress penalty weight
    "epsilon": float, # integrity detection bonus weight
    "omega": float    # outcome alignment reward weight
}
TERMINATION_ENTROPY_THRESHOLD = 0.3
```

### Training Strategy

**Stage 1 — Simulation (offline RL):**
- Generate thousands of interview rollouts using LLM-simulated candidate personas
- Personas include: junior/mid/senior skill levels, confident/anxious behavioral types, bluffer/genuine, native/non-native speakers
- Use Implicit Q-Learning (IQL) via Stable-Baselines3
- Competency distribution calibrated from real MIT Interview Dataset ratings

**Stage 2 — Fine-tuning (real data):**
- Once deployed, real session data accumulates
- Hiring outcomes from Dataset 5 provide terminal rewards
- Policy periodically retrained

**Why IQL (Implicit Q-Learning):**
- Cannot have a real candidate sitting while agent explores randomly
- All training data collected offline (simulated or historical)
- IQL handles offline RL stably
- Available in Stable-Baselines3

### Policy Comparison Results (Synthetic Simulation, N=300 episodes)

| Policy | Avg Turns | Avg Final Entropy | Avg Info Gain | Avg Reward | Avg Skill Accuracy |
|---|---|---|---|---|---|
| Random Policy | 6.77 | 1.4316 | 0.2621 | -1.1559 | 0.4933 |
| Fixed-Script Policy | 10.00 | 1.2792 | 0.4104 | -1.4683 | 0.6393 |
| Rule-Based Adaptive | 19.00 | 1.0865 | 0.7407 | -1.6446 | 0.6320 |
| Greedy Entropy Policy | 19.00 | 1.4103 | 0.6264 | -2.0814 | 0.4767 |
| ARIA Reward-Heuristic | 19.00 | **1.0531** | **0.7666** | -1.6901 | **0.6887** |

**Key findings:**
- ARIA Reward-Heuristic achieves lowest final entropy (most confident assessment) and highest info gain
- Greedy Entropy (probe_foundation 95% of time) has poor skill accuracy — exploiting one action is suboptimal
- Random Policy wins on reward (fewer turns = less duration penalty) — suggests duration penalty needs retuning
- Rule-Based Adaptive and ARIA Heuristic are closest to what IQL should learn to exceed

---

## 8. The 6 Baseline Models

These are the 6 models that make up ARIA's core processing pipeline, each evaluated independently with real-world datasets.

### Model 1 — Video Emotion Classifier
**Modality:** Video
**What it does:** Watches the candidate's face and classifies emotional/behavioral state — confident, nervous, confused, engaged, or blank
**How it works:** Every webcam frame gets analyzed for 468 facial landmarks. A CNN looks at patterns in these points and predicts interview-context emotional state. DeepFace gives generic emotions (happy, sad, angry, etc.) which are mapped to interview-specific states via a lookup table at baseline level.
**Architecture:** Small 3-layer CNN (conv1→16 filters, conv2→32 filters, conv3→64 filters, FC head)
**Why the baseline uses a mapping table:** Public datasets only label generic emotions. Real interview-context labels (nervous, confident, etc.) require interview footage with those specific labels to train on directly.

### Model 2 — Audio Cognitive Load Classifier
**Modality:** Audio
**What it does:** Listens to HOW the candidate speaks (not what they say) — pitch, pauses, speech rate — and estimates stress/cognitive load level
**How it works:** Speech has measurable physical properties. Extracted as MFCC features (numerical representation of voice "shape") + pitch, energy, jitter, shimmer. Deep learning audio model trained on these features.
**Architecture:** Deep learning audio model on MFCC + prosodic features
**Limitation:** RAVDESS has general emotion labels, not "anxiety vs ignorance" ground truth. The 3-class cognitive load separation (low / high-anxiety / high-ignorance) is a separate downstream classification, not trained on RAVDESS directly.

### Model 3 — Text Semantic Competency Classifier
**Modality:** Text
**What it does:** Reads what the candidate actually said and judges how good the answer is compared to what an expert would say
**How it works:** Both candidate answer and expert reference answer are converted to embeddings (numerical fingerprints that capture meaning, not just words). Cosine similarity measures how close the candidate's answer is to the expert's. Score mapped to beginner/mid/expert.
**Multiple baselines run:**
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM (best Macro F1 overall: 0.6030)
- SBERT Cosine Threshold
- SBERT Feature Classifier
- ARIA Hybrid Semantic-Rubric (best Accuracy: 0.7516)

**What TF-IDF is:** Term Frequency-Inverse Document Frequency. A word gets a high score only if it appears a lot in one document but not in most others. Much older and simpler than SBERT embeddings — SBERT understands meaning, TF-IDF only counts word overlap. Notable finding: TF-IDF+SVM outperformed SBERT approaches on Macro F1.

### Model 4 — Multimodal Fusion Model
**Modality:** Video + Audio + Text (combined)
**What it does:** Takes outputs of Models 1, 2, and 3 and merges them into one combined signal per turn
**How it works:** Mean-pools each modality across the time axis to get fixed-dim vectors, then concatenates all three (baseline concatenation fusion) and trains a Logistic Regression classifier on top.
**Ablation finding:** Text-only (0.8015) outperformed fused (0.7976) — notable finding worth discussing in the paper. Suggests the baseline fusion is not yet learning to leverage the additional modalities effectively.

### Model 5 — Cross-Modal Incongruence Detector
**Modality:** Audio + Text (cross-modal)
**What it does:** Catches bluffing — when someone sounds confident but their answer is actually shallow
**How it works:** Takes prosodic confidence score (from audio) and semantic depth score (from text), compares the delta. High tone confidence + thin content = incongruence flag. Architecture: deep cross-modal model with learned modality gate weights (Text/Semantic branch: 0.4457, Behavior/Nonverbal branch: 0.5543)
**Finding:** Model learned to weight behavioral branch slightly more than semantic branch for deception detection.

### Model 6 — RL Policy Agent
**Modality:** Sequential decision model (not a perceptual model)
**What it does:** Decides what question to ask next based on the current belief state about the candidate
**How it works:** Not trained like the other 5 — trained via simulated interview rollouts. Rule-based policy used as baseline before IQL training. Evaluated on 300-episode simulation calibrated with real MIT Interview Dataset ratings.
**Key distinction:** This is the brain of the system. Everything else feeds this one decision-maker.

---

## 9. Real-World Datasets — All 6 Models

| Model | Dataset | Access | Key Parameters |
|---|---|---|---|
| 1 — Video | FER2013 (Kaggle: jonathanoheix) | Direct download | 28,709 train / 7,178 val images, 7 emotion classes, 48x48 grayscale |
| 1 — Video (alt) | AffectNet (Kaggle: fatihkgg) | Direct download | Larger, 8 classes including contempt, continuous valence-arousal |
| 2 — Audio | RAVDESS (Zenodo) | Direct download — CC BY-NC-SA 4.0 | 1,440 .wav files, 24 actors, 8 emotion classes, emotion code in filename |
| 3 — Text | Mohler Dataset (ASAG) | Direct download | 2,273 real student answers to 80 CS questions, scored 0-5 by human graders |
| 4 — Fusion | CMU-MOSEI | Download via CMU SDK | 16,326 train / 4,659 test, text (768-dim) + audio (74-dim) + vision (35-dim) |
| 5 — Incongruence | Box of Lies (Multimodal Deception in Dialogs) | Obtained | 225 samples across 25 EAF annotation files, 144 deceptive / 81 truthful |
| 6 — RL | MIT Interview Dataset (Naim et al.) | Request required | 138 mock interviews, human hireability ratings — used for environment calibration |
| All | O*NET Database | Direct download | Skill taxonomy + occupational data — used for ontology graph construction |

**Data folder structure:**
```
ARIA/data/
├── model1_video/fer2013/
│   ├── train/<emotion>/*.jpg
│   └── validation/<emotion>/*.jpg
├── model2_audio/ravdess/
│   └── Actor_01/ ... Actor_24/*.wav
├── model3_text/
│   └── mohler_dataset.parquet
├── model4_fusion/
│   └── unaligned_50.pkl
├── model5_incongruence/
│   └── *.eaf annotation files + transcripts
└── model6_rl/
    └── mit_interview_ratings.csv
```

---

## 10. Baseline Model Results

### Dataset 1 — FER2013 (Facial Expression Recognition 2013)

| Model | Precision | Recall | F1-score | Accuracy |
|---|---|---|---|---|
| CNN — ARIA Video Emotion Baseline | 0.56 (macro) | 0.48 (macro) | 0.48 (macro) | 0.5548 |

**Per-class breakdown:**
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.47 | 0.42 | 0.44 | 958 |
| disgust | 0.70 | 0.13 | 0.21 | 111 |
| fear | 0.39 | 0.23 | 0.29 | 1024 |
| happy | 0.74 | 0.80 | 0.77 | 1774 |
| neutral | 0.51 | 0.57 | 0.54 | 1233 |
| sad | 0.41 | 0.51 | 0.45 | 1247 |
| surprise | 0.67 | 0.70 | 0.68 | 831 |

**Notes:**
- 55.48% accuracy is in line with published FER2013 vanilla CNN baselines (human-level agreement on FER2013 is only 65-68%)
- happy and surprise are strong — most visually distinct facial signatures
- disgust has severe class imbalance (111 vs 1774 for happy) causing 0.13 recall
- fear most confused with sad (302 misclassifications) and surprise (148)
- Interview-context label distribution: nervous 29.8%, confident 26.6%, blank 19.2%, confused 12.3%, engaged 12.1%

---

### Dataset 2 — RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

| Model | Precision | Recall | F1-score | Accuracy |
|---|---|---|---|---|
| Deep Learning Audio Model | 0.55 (macro) | 0.55 (macro) | 0.52 (macro) | 0.5590 |

**Per-class breakdown:**
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.96 | 0.66 | 0.78 | 38 |
| calm | 0.48 | 0.87 | 0.62 | 38 |
| disgust | 0.56 | 0.89 | 0.69 | 38 |
| fearful | 0.63 | 0.62 | 0.62 | 39 |
| happy | 0.40 | 0.10 | 0.16 | 39 |
| neutral | 0.25 | 0.42 | 0.31 | 19 |
| sad | 0.37 | 0.18 | 0.25 | 38 |
| surprised | 0.79 | 0.67 | 0.72 | 39 |

**Notes:**
- Cognitive load proxy distribution: high_load 61.5%, low_load 38.5%
- 1,440 real .wav files from 24 actors, balanced classes except neutral (96 vs 192)

---

### Dataset 3 — Mohler Dataset (Short Answer Grading)

**Dataset stats:** 2,273 Q&A pairs, score range 0.00–5.00, labels: expert 1639, mid 554, beginner 80

| Model | Precision (macro) | Recall (macro) | F1-score (macro) | Accuracy |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.50 | 0.61 | 0.5167 | 0.6681 |
| TF-IDF + Linear SVM | 0.62 | 0.59 | **0.6030** | 0.7495 |
| SBERT Cosine Threshold | 0.46 | 0.48 | 0.4629 | 0.6791 |
| SBERT Feature Classifier | 0.52 | 0.58 | 0.5410 | 0.7165 |
| ARIA Hybrid Semantic-Rubric | **0.78** | 0.55 | 0.5886 | **0.7516** |

**Best model by Macro F1:** TF-IDF + Linear SVM (0.6030)
**Best model by Accuracy:** ARIA Hybrid Semantic-Rubric (0.7516)

**Notable finding:** TF-IDF+SVM outperformed SBERT approaches on Macro F1 despite being a much simpler method — worth a sentence in the paper.

---

### Dataset 4 — CMU-MOSEI

**Dataset stats:** 16,326 train / 1,871 valid / 4,659 test, fused feature dim 877, label balance: 11,588 positive / 4,738 negative

| Model | Precision (macro) | Recall (macro) | F1-score (macro) | Accuracy |
|---|---|---|---|---|
| Text-only | — | — | — | 0.8015 |
| Audio-only | — | — | — | 0.6098 |
| Vision-only | — | — | — | 0.5808 |
| Fused — ARIA Multimodal Fusion | 0.76 | 0.79 | 0.77 | 0.7976 |

**Notable finding:** Text-only (0.8015) outperformed fused (0.7976). Baseline concatenation fusion is not yet effectively leveraging additional modalities. The cross-attention transformer upgrade should address this.

---

### Dataset 5 — Box of Lies (Multimodal Deception Detection in Dialogs)

**Dataset stats:** 225 samples, 25 EAF files, 144 deceptive / 81 truthful, 221/225 transcripts non-empty

| Model | Precision (macro) | Recall (macro) | F1-score (macro) | Accuracy |
|---|---|---|---|---|
| Deep Cross-Modal Incongruence Model | 0.56 | 0.55 | 0.55 | 0.5965 |

**Learned gate weights:** Text/Semantic 0.4457, Behavior/Nonverbal 0.5543
**Notes:** Severe overfitting observed — train accuracy reached 1.0 by epoch 13 but test accuracy plateaued at ~0.51. Best test accuracy (0.5965) occurred at epoch 2. Dataset is very small (168 train / 57 test) — dropout and early stopping critical for next iteration.

---

### Model 6 — RL Policy Agent (Synthetic Simulation, N=300 episodes)

*(Separate evaluation — no precision/recall/F1 metrics; uses RL-specific metrics)*

| Policy | Avg Turns | Avg Final Entropy | Avg Info Gain | Avg Reward | Avg Skill Accuracy |
|---|---|---|---|---|---|
| Random Policy | 6.77 | 1.4316 | 0.2621 | -1.1559 | 0.4933 |
| Fixed-Script Policy | 10.00 | 1.2792 | 0.4104 | -1.4683 | 0.6393 |
| Rule-Based Adaptive | 19.00 | 1.0865 | 0.7407 | -1.6446 | 0.6320 |
| Greedy Entropy | 19.00 | 1.4103 | 0.6264 | -2.0814 | 0.4767 |
| ARIA Reward-Heuristic | 19.00 | **1.0531** | **0.7666** | -1.6901 | **0.6887** |

---

## 11. Open Source Tech Stack

| Component | Tool | Notes |
|---|---|---|
| STT | faster-whisper (Whisper large-v3) | Streaming mode |
| Facial expression | DeepFace + MediaPipe Face Mesh | Emotion + landmark detection |
| Gaze | L2CS-Net | Eye contact quantification, pretrained on MPIIGaze |
| Prosody | SpeechBrain + openSMILE | Pause, stress, rate, pitch |
| Semantic eval | BGE-M3 + all-MiniLM-L6-v2 | Meaning depth, coherence |
| RL framework | Stable-Baselines3 (IQL) | Policy training |
| LLM | Llama 3.1 70B or Qwen2.5-72B | Via Ollama, instruction-following |
| TTS | Coqui XTTS-v2 / Kokoro | Natural voice, open weights |
| Avatar | SadTalker | Lip-sync talking head |
| Resume parsing | PyMuPDF + spaCy NER | Extract skills, experience |
| Skill ontology | NetworkX + O*NET data | Free occupational data |
| Backend | FastAPI + Redis | Async orchestration |
| Frontend | React + WebRTC (native) | No paid services |
| Report | ReportLab | PDF output |
| Video processing | OpenCV | Frame capture |

**Zero paid APIs required in the core pipeline.**

---

## 12. Work Division

### Raghav — Intelligence, Training & Infrastructure Layer

**Modules owned:**
- Module 5 — Skill Ontology Graph (NetworkX + O*NET)
- Module 6 — Competency Belief Updater
- Module 7 — RL Training Pipeline (IQL, Stable-Baselines3)
- Module 8 — LLM Question Generator (Llama 3.1 70B via Ollama)
- Module 9 — TTS + Avatar (Coqui XTTS-v2 + SadTalker)
- Module 12 — Cross-Modal Incongruence Detector
- Module 13 — Interview Fairness Auditor
- Module 14 — Evaluation Engine + Report Generator
- Step 17 — FastAPI + Redis Backend

**Training steps owned:**
- Bayesian belief classifier (Dataset 2)
- RL policy Stage 1 — offline IQL on synthetic rollouts
- RL policy Stage 2 — fine-tuning on Dataset 5
- Incongruence classifier (Dataset 1 + Dataset 2)
- Evaluation score calibration (Dataset 5)

**Literature review:** 23 papers — POMDP/RL for dialogue (10), multimodal interview analysis (8), adaptive question generation (5)

---

### Krissh — Perception, Signal Layer & RL Formulation

**Modules owned:**
- Module 1 — STT
- Module 2 — Vision
- Module 3 — Prosody
- Module 4 — Multimodal Fusion Layer
- Module 10 — Cognitive Load Separation
- Module 11 — Anti-Gaming
- Step 16 — Frontend (React + WebRTC)

**RL Formulation ownership:**
Krissh defines the mathematical specification — reward function, state/action space, action masking rules. Raghav implements the IQL training pipeline on top.

**Delivers to Raghav by June 5:** `rl_spec.py` — shared constants file with RL_STATE_SCHEMA, RL_ACTION_SPACE, RL_ACTION_MASKS, REWARD_COEFFICIENTS, TERMINATION_ENTROPY_THRESHOLD

**Training steps owned:**
- Emotion classifier fine-tune (Dataset 1)
- Cognitive load classifier (Dataset 3)
- Cross-modal attention fusion transformer (Dataset 1 all modalities)
- Cognitive load 3-class classifier (Dataset 3)
- Anti-gaming anomaly detectors (Dataset 4)

**Literature review:** 27 papers — gaze/facial expression (8), prosody/cognitive load (8), fairness in AI hiring (7), RL reward shaping (4)

---

### Shared Responsibilities

| Task | Notes |
|---|---|
| `rl_spec.py` | Krissh writes, Raghav signs off before training |
| Step 18 — End-to-end integration | Both together |
| Architecture diagram final | Both, Week of June 5 |
| Paper writing | Each writes sections matching module ownership |
| Ablation studies | Each runs ablations on own modules |
| Patent provisional filing | Raghav — contact VIT IP cell immediately |

---

## 13. Interface Contract (Krissh → Raghav)

### Per conversational turn — turn_signal dict

```python
turn_signal = {
    "session_id": str,
    "turn_id": int,
    "transcript": str,                     # Module 1
    "vision": {
        "emotion_label": str,              # engaged / confused / nervous / confident / blank
        "au_activations": dict,            # AU1, AU2, AU4, AU6, AU12, AU15, AU17, AU23, AU25
        "gaze_vector": {"yaw": float, "pitch": float},
        "eye_contact_score": float,        # 0.0 – 1.0
        "head_pose": {"roll": float, "pitch": float, "yaw": float},
        "blink_rate": float
    },
    "prosody": {
        "pitch_mean": float,
        "pitch_variance": float,
        "speech_rate": float,
        "pause_count": int,
        "disfluency_count": int,
        "response_latency_ms": float,
        "jitter": float,
        "shimmer": float,
        "mfcc_vector": list
    },
    "fused_vector": list,                  # Module 4 — fixed-dim
    "cognitive_load_label": str,           # low / anxiety / ignorance
    "distress_score": float,
    "anti_gaming_flags": list
}
```

### rl_spec.py (Krissh delivers to Raghav by June 5)

```python
RL_STATE_SCHEMA = { ... }
RL_ACTION_SPACE = [ ... ]
RL_ACTION_MASKS = { ... }
REWARD_COEFFICIENTS = {
    "alpha": float,   "beta": float,
    "gamma": float,   "delta": float,
    "epsilon": float, "omega": float
}
TERMINATION_ENTROPY_THRESHOLD = float
```

---

## 14. Step-by-Step Build Guide

**Rule: Build in this order, no skipping steps.**

1. Create project folder structure with separate subfolders per module
2. Get core conversation loop working: Whisper → Llama → TTS (terminal test first, no UI)
3. Add resume parsing (PyMuPDF + Llama JSON extraction)
4. Build skill ontology graph (NetworkX, Backend Developer first)
5. Add vision pipeline (MediaPipe + DeepFace + L2CS-Net on background thread)
6. Add prosody pipeline (SpeechBrain/openSMILE per turn)
7. Build fusion layer (start with concatenation)
8. Build competency belief updater (Bayesian, rule-based first)
9. Build RL policy agent (rule-based baseline first, IQL later)
10. Wire RL agent to LLM question generator (decoupled — RL picks what, LLM generates how)
11. Build incongruence detector (BGE-M3 cosine similarity vs prosody confidence delta)
12. Build cognitive load module (latency + disfluency + gaze → 3-class output)
13. Build anti-gaming module (3 passive detectors)
14. Build evaluation engine + report generator
15. Build fairness auditor (action distribution logging + correlation checks)
16. Build React + WebRTC frontend
17. Connect everything via FastAPI + Redis
18. End-to-end test (full interview, both topics you know and topics you don't)

**Build order dependency:**
```
Model 3 (Text) → Model 1 (Video) → Model 2 (Audio)
       ↓              ↓              ↓
       └──────→ Model 4 (Fusion) ←───┘
                      ↓
             Model 5 (Incongruence)
                      ↓
             Model 6 (RL Agent)
```

---

## 15. Where Training Happens

**These components are pretrained — load and run inference only:**
Whisper large-v3, MediaPipe Face Mesh, L2CS-Net, SpeechBrain base models, BGE-M3, Llama 3.1 70B, Coqui XTTS-v2, SadTalker, PyMuPDF + spaCy

**These components require training:**

| Step | What You Train | Dataset |
|---|---|---|
| Step 5 | Emotion classifier fine-tune on DeepFace | FER2013 / AffectNet (Dataset 1) |
| Step 6 | Stress / cognitive load classifier | RAVDESS proxy + Dataset 3 |
| Step 7 | Cross-attention fusion transformer | CMU-MOSEI (Dataset 1 full) |
| Step 8 | Bayesian belief update classifier | Mohler Dataset (Dataset 2) |
| Step 9 Stage 1 | RL policy — offline IQL | Synthetic LLM-simulated rollouts |
| Step 9 Stage 2 | RL policy — fine-tuning | MIT Interview Dataset ratings (Dataset 5) |
| Step 11 | Incongruence binary classifier | Dataset 1 + Dataset 2 combined |
| Step 12 | Cognitive load 3-class classifier | Dataset 3 (anxiety-vs-ignorance label) |
| Step 13 | Anti-gaming anomaly detector | Dataset 4 |
| Step 14 | Evaluation score calibration | Dataset 5 |

---

## 16. Project Schedule

| Deadline | Raghav | Krissh |
|---|---|---|
| May 15 | Problem statement, contribution list | Problem statement, contribution list |
| May 22 | Datasets 2, 5 schema; synthetic pipeline planned | Datasets 1, 3, 4 sourced and documented |
| May 29 | 23 papers reviewed (RL / dialogue / adaptive QGen) | 27 papers reviewed (gaze / prosody / fairness / reward shaping) |
| June 5 | Core loop: Whisper → Llama → TTS; resume parser | STT + Vision pipeline; WebRTC base; **rl_spec.py delivered** |
| June 12 | Belief updater (rule-based), ontology graphs (2 roles), incongruence v1 | Prosody pipeline, fusion (concat), cognitive load module |
| June 19 | RL policy Stage 1 trained, fairness auditor, FastAPI + Redis, full pipeline | Anti-gaming module, fusion upgrade, frontend complete, reward function with unit tests |
| June 26 | RL vs baseline comparison, belief convergence curves, all figures ready | Perception experiments, distress/cognitive load metrics, reward validation |
| July 3 | Ablation: remove RL / incongruence / cognitive load | Ablation: remove fusion / vision / prosody / reward components |
| July 10 | Paper: Intro, POMDP, RL Training, Belief State, Evaluation Engine, Related Work | Paper: Vision, Prosody, Fusion, Cognitive Load, Anti-Gaming, RL Formulation, Datasets |
| July 17 | **Patent provisional filed**, LaTeX compile, submission ready | Final proofread, formatting pass |

---

## 17. Patent Strategy

### What Is Patentable

The patentable core is NOT "AI interview platform." The specific technical mechanisms that are novel:

1. POMDP formalisation of interview control — treating job interview as a POMDP with belief state inference over a skill ontology graph
2. RL policy trained via information-gain reward — using belief entropy reduction as primary reward signal
3. Cross-modal incongruence detection — comparing prosodic confidence against semantic depth to detect bluffing
4. Cognitive load separation pipeline — separating anxiety-induced errors from genuine knowledge gaps
5. Skill ontology graph traversal — navigating prerequisite skill relationships driven by belief state updates
6. Self-improving loop via hiring outcome feedback — hiring outcomes as terminal RL reward
7. Embedded fairness auditing — monitoring interview policy action distribution for demographic-correlated bias

### Patent Statement (for claims drafting)

"A system that treats candidate assessment as a belief-state inference problem and solves it via reinforcement learning over multimodal observations."

### Critical: What to Avoid Before Filing

- Do NOT open-source this repository publicly
- Do NOT present at any conference
- Do NOT post the architecture online
- Do NOT upload a preprint anywhere

Any of the above constitutes prior public disclosure and kills novelty in most jurisdictions.

### Practical Steps (India)

1. Contact VIT IP cell immediately — they co-file and often cover costs for student innovations
2. File Provisional Patent Application at Indian Patent Office (ipindia.gov.in) — approximately Rs.1,750 for individuals — locks priority date, gives 12 months
3. Follow up with Complete Specification within 12 months
4. Filing must happen before July 17, 2026 (before paper submission)

---

## 18. Research Contributions

1. **First POMDP formalisation of a job interview** — belief state inference over a skill ontology graph, updated via multimodal Bayesian inference after each conversational turn
2. **RL-trained interview policy with information-gain reward** — first application of information-theoretic reward shaping to interview control
3. **Skill ontology graph traversal driven by belief state** — structured method of navigating prerequisite skill relationships to identify root competency gaps
4. **Cross-modal incongruence detection** — novel signal for bluffing detection by measuring delta between prosodic confidence and semantic depth
5. **Cognitive load separation in interview scoring** — first explicit method for separating anxiety-induced performance degradation from genuine knowledge gaps in automated assessment
6. **Real-time multimodal integrity monitoring** — passive, non-intrusive framework for detecting note reading, AI assistance, off-camera coaching, scripted memorisation
7. **Self-improving closed-loop assessment system** — real-world hiring outcomes as terminal RL reward enabling continuous policy improvement through deployment

---

*Document compiled: July 2026*
*Project: ARIA — Autonomous Reinforcement-based Interview Agent*
*Team: Raghav Sejpal (23BAI0095) + Krissh, VIT Vellore*
