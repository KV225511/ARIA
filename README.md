# ARIA - Autonomous Reinforcement-Based Interview Agent

## Executive Summary & System Vision
ARIA is an autonomous, real-time multimodal AI interviewer designed to replace rigid, scripted post-hoc video screening tools with an adaptive, conversational screening agent. It treats the interview as a real-time closed-loop inference problem (POMDP). As a candidate speaks, ARIA simultaneously captures audio, video, and text streams to extract linguistic meaning, vocal tone, facial micro-expressions, gaze vectors, and physiological stress signals.

## System Architecture

The system is composed of 15 modules distributed across five core layers:

1. **Perception & Signal Layer (Modules 1–4):**
   - **Speech-to-Text & Semantic Grader:** Transcribes speech and evaluates answers against domain references.
   - **Vision Engine:** Tracks facial landmarks, emotion, gaze, and micro-expressions.
   - **Prosody Extractor:** Extracts acoustic features and calibrates personal baselines.
   - **Multimodal Dynamic Fusion Engine:** Fuses Text, Vision, and Prosody into a 72-dimension canonical feature vector.

2. **Intelligence & Control Layer (Modules 5–8):**
   - **Skill Ontology Graph:** Maps domain skills and prerequisite dependencies.
   - **Competency Belief Updater:** Performs Bayesian belief updates based on observed behavior and knowledge.
   - **RL Interview Policy Agent:** Determines the optimal next interview action based on the belief state and fused features.
   - **LLM Question Generator:** Dynamically crafts the natural language question based on the RL policy's chosen action.

3. **Synthesis & Interaction Layer (Module 9):**
   - **TTS & Avatar Synthesis:** Converts the LLM-generated question into natural audio and lip-synchronized video.

4. **Cognitive & Behavioral Analysis Layer (Modules 10–13):**
   - **Cognitive Load Separator:** Differentiates between genuine knowledge gaps and anxiety-induced performance drops.
   - **Anti-Gaming & Integrity Monitor:** Detects note-reading, AI assistance, and scripted responses.
   - **Cross-Modal Incongruence Detector:** Identifies bluffing by comparing vocal confidence to semantic depth.
   - **Interview Fairness Auditor:** Monitors policy exploration fairness.

5. **Evaluation & Feedback Layer (Modules 14–15):**
   - **Evaluation Engine:** Generates comprehensive PDF interview reports.
   - **Self-Improving Feedback Loop:** Ingests real-world hiring outcomes for continuous RL training.

## Current Progress & Status

**Raghav has successfully implemented the system architecture up to the RL Policy.** 

The current implementation status is as follows:
- **Completed:** Modules 1 through 7, including the perception pipelines, multimodal fusion engine, cognitive load separation, anti-gaming monitors, Bayesian belief updating, and the RL policy agent for action selection.
- **Pending (Work in Progress):** Only the LLM Question Generator (Module 8) and the final integration/synthesis layer (Module 9: TTS & Avatar Synthesis) are left to be completed before the full end-to-end orchestration loop is finished.

## Hardware Budget
ARIA is engineered to run locally without relying on paid external APIs. It targets an NVIDIA GeForce RTX 4060 (8GB VRAM) environment, employing strict VRAM budgeting, lazy loading, and INT8 quantization to operate efficiently.
