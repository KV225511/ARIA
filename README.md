# ARIA — Autonomous Reinforcement-Based Interview Agent

ARIA is a local-first research prototype for adaptive, multimodal technical interviews. Instead of following a fixed question list, ARIA models an interview as a partially observable decision process: it gathers evidence from each response, updates a probabilistic competency model, and selects the next interview strategy based on what remains uncertain.

The project combines speech, vision, prosody, skill ontologies, Bayesian belief tracking, offline reinforcement learning, local LLM question generation, interview integrity signals, fairness auditing, reporting, and feedback logging in a modular 15-part architecture.

> **Project status:** all 15 module areas have prototype implementations. A polished React interface and FastAPI/WebSocket interview loop are available, but the full multimodal and learned-policy stack is not yet connected end to end. ARIA is a research system, not a production hiring or automated decision product.

## How ARIA Works

```text
Job description + résumé
          |
          v
Dynamic skill ontology
          |
          v
Candidate audio + video + text
          |
          +--> speech / semantic evidence
          +--> facial, gaze, and temporal signals
          +--> prosody and response behavior
          |
          v
72-dimensional multimodal representation
          |
          +--> cognitive-load context
          +--> integrity and incongruence signals
          +--> fairness monitoring
          |
          v
Bayesian competency belief over skill nodes
          |
          v
Interview policy action
          |
          v
Locally generated question --> speech/UI --> next turn
```

ARIA separates **what to ask** from **how to ask it**. The policy chooses one of eight interview actions—such as probing a foundation, following up, switching topics, or changing difficulty—while a local Ollama model turns that action and the current evidence into a natural question.

## Current Implementation

| Layer | Modules | Implemented | Current integration state |
|---|---:|---|---|
| Perception and signal | 1–4 | Speech-to-text, semantic grading, facial landmarks, emotion, gaze, temporal action units, prosody, normalization, concatenation, and attention fusion | Implemented and tested as modules; only speech transcription is currently used by the live API |
| Intelligence and control | 5–8 | Dynamic skill ontology, belief-v2 competency model, replay-v2 data pipeline, IQL training, environment, audits, calibration, and Ollama question generation | Ontology, belief updates, and question generation are live; the trained IQL checkpoint is not yet loaded by the web app |
| Synthesis and interaction | 9 | Local TTS/avatar baseline plus browser speech synthesis | Browser TTS is live; lip-synchronized avatar output remains a future integration |
| Cognitive and behavioral analysis | 10–13 | Cognitive-load classification, anti-gaming signals, cross-modal incongruence detection, and fairness auditing | Implemented and tested independently; not yet orchestrated in every live turn |
| Evaluation and feedback | 14–15 | Structured report generation and SQLite trajectory/hiring-outcome logging | Implemented as services; not yet exposed as a complete post-interview product flow |

### Live Interview Application

The current application includes:

- a responsive React 19 and Vite interface;
- drag-and-drop PDF setup for the job description and candidate résumé;
- FastAPI session creation and PDF text extraction;
- candidate-specific ontology adaptation;
- a real-time WebSocket interview channel;
- streaming local-LLM question generation;
- typed and recorded candidate answers;
- `ffmpeg` audio conversion and local speech transcription;
- browser speech synthesis, camera preview, recording state, and a live transcript;
- explicit connection, media, privacy, and interview-strategy states.

The live orchestrator currently uses a simplified belief update after each answer. Multimodal fusion, calibrated semantic evidence, the learned IQL action policy, fairness signals, and final reporting still need to be wired into that loop before the application represents the complete architecture.

## The 15 Modules

1. **Speech-to-Text and Semantic Grader** — transcription, timestamps, response latency, and answer scoring.
2. **Vision Engine** — face landmarks, emotion, gaze, blinks, and temporal action-unit features.
3. **Prosody Extractor** — pitch, speech rate, energy, jitter, MFCCs, and personal baselines.
4. **Multimodal Fusion Engine** — canonical 72-feature schema, normalization, missing-modality handling, concatenation, and attention fusion.
5. **Skill Ontology Graph** — role templates, prerequisite relationships, and résumé/JD adaptation.
6. **Competency Belief Updater** — per-skill Bayesian evidence accumulation and an aggregate competency verdict.
7. **RL Interview Policy** — eight-action environment, replay data preparation, Implicit Q-Learning, checkpoint selection, and policy evaluation gates.
8. **LLM Question Generator** — contextual question generation and streaming through Ollama.
9. **TTS and Avatar Baseline** — local speech synthesis and avatar-frame interface.
10. **Cognitive Load Separator** — distinguishes low load, anxiety, and likely knowledge gaps.
11. **Anti-Gaming Monitor** — gaze, latency, semantic-template, and assistance indicators.
12. **Incongruence Detector** — compares semantic depth with vocal and behavioral confidence.
13. **Fairness Auditor** — monitors exploration and treatment across interview trajectories.
14. **Evaluation Engine** — competency scoring and structured narrative reports.
15. **Feedback Logger** — stores trajectories and later hiring outcomes for controlled learning.

## Belief-v2 and Offline RL

The RL pipeline has been redesigned around immutable raw evidence, identity-safe dataset splits, deterministic replay, and locked evaluation.

Key safeguards now include:

- split isolation by connected résumé/JD identity components and content hashes;
- an immutable, hash-addressed `BeliefModelConfig`;
- training-only fitting of competency emission centers and scales;
- validation-only tuning of repetition, effective-sample-size, aggregation, and abstention parameters;
- explicit abstention when coverage, confidence, or evidence is insufficient;
- a fixed 32-dimensional `aria-state-v2` representation;
- replay of belief, state, reward, termination, labels, and action masks from raw transitions;
- rejection of non-finite evidence and schema/configuration mismatches;
- separate quality gates for raw evidence, validation calibration, locked testing, offline-RL support, and fresh learned-policy rollouts;
- deterministic IQL training and atomic, metadata-rich checkpoint output;
- a locked-test command that refuses to overwrite an existing report.

### Benchmark Status

The earlier benchmark produced roughly **0.39 micro-F1** and collapsed most terminal predictions into the Beginner class. That result was a metric over stored belief verdicts, not a valid evaluation of the learned IQL policy.

The current target is **0.80–0.90 micro-F1**, but that result has **not** yet been established. The repository does not contain the raw transition dataset needed to produce a new measured result. Calibration cannot manufacture class information if evaluator scores are compressed or invalid, and fixed offline actions cannot honestly measure counterfactual policy performance.

A defensible result requires:

1. a sufficiently large, balanced, de-identified résumé/JD corpus;
2. valid raw rollouts generated with distinct candidate and evaluator models;
3. passing raw-evidence and offline-support gates;
4. belief-v2 preparation and validation-only calibration;
5. IQL training and checkpoint selection without reading test labels;
6. fresh learned-policy rollouts tied to the selected checkpoint hash;
7. one post-freeze evaluation on the locked test set.

See [ARIA_RL_IMPROVEMENT_CHANGES.md](ARIA_RL_IMPROVEMENT_CHANGES.md) for the complete belief-v2 and replay-v2 design.

## Quick Start

### Prerequisites

- Python 3.10 or newer;
- Node.js and npm compatible with Vite 8;
- [Ollama](https://ollama.com/) with the configured local model;
- `ffmpeg` available on `PATH` for recorded audio answers;
- optional NVIDIA CUDA support for the full speech, vision, and prosody stack.

### Install

```powershell
git clone https://github.com/KV225511/ARIA.git
cd ARIA

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

ollama pull llama3.1

Set-Location frontend
npm ci
Set-Location ..
```

### Run the Application

The launcher starts the FastAPI backend on port `8000` and the Vite frontend on port `5173`:

```powershell
.\run.ps1
```

Or run the services separately:

```powershell
# Terminal 1 — repository root
.\.venv\Scripts\uvicorn.exe app:app --reload --port 8000

# Terminal 2
Set-Location frontend
npm run dev
```

Open `http://localhost:5173`. API documentation is available at `http://localhost:8000/docs`.

### Configuration

ARIA reads the following environment variables, including values placed in a root `.env` file:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama service used by ontology, question, and report generation |
| `OLLAMA_MODEL` | `llama3.1` | Live ontology/question/report model |
| `ARIA_DEVICE` | `cuda` | Primary ML execution device; use `cpu` when CUDA is unavailable |
| `ARIA_CANDIDATE_MODEL` | `qwen2.5:7b` | Synthetic candidate model used during rollout generation |
| `ARIA_EVALUATOR_MODEL` | `llama3.1` | Independent semantic evaluator used during rollout generation |
| `L2CS_WEIGHTS_PATH` | `models/L2CSNet_gaze360.pkl` | Gaze-estimation checkpoint path |

Candidate and evaluator models must remain distinct when generating benchmark evidence.

## Testing

Run the Python test suite from the repository root:

```powershell
pytest -q
```

Check the frontend separately:

```powershell
Set-Location frontend
npm run lint
npm run build
```

Tests cover speech, vision, prosody, fusion, ontology adaptation, beliefs, cognitive load, anti-gaming, final modules, dataset splitting, calibration, replay, metrics, locked evaluation, and IQL training contracts.

## RL Data and Training Workflow

Raw benchmark inputs are intentionally not committed. Place valid PDF documents in:

```text
data/resumes/
data/jds/
```

Generate a balanced synthetic rollout log with distinct local models:

```powershell
$env:ARIA_CANDIDATE_MODEL = "qwen2.5:7b"
$env:ARIA_EVALUATOR_MODEL = "llama3.1"

python -m modules.module_07_rl.llm_simulator `
  --sweep `
  --max_episodes 300 `
  --max_concurrent 5 `
  --seed 42
```

Prepare the identity-safe split, calibration, and replay-v2 artifacts:

```powershell
python -m modules.module_07_rl.prepare_belief_pipeline `
  data/synthetic/qwen_rl_dataset.json `
  data/synthetic/derived
```

Train and select the IQL checkpoint using only train and validation data:

```powershell
python -m modules.module_07_rl.train `
  --train-file data/synthetic/derived/splits/train.json `
  --validation-file data/synthetic/derived/splits/validation.json `
  --belief-config data/synthetic/derived/belief_model_v2.json
```

Only after calibration and checkpoint selection are frozen, write the locked stored-belief report once:

```powershell
python -m modules.module_07_rl.evaluate_locked_test `
  data/synthetic/derived/splits/test.json `
  data/synthetic/derived/belief_model_v2.json `
  data/synthetic/derived/locked_test_report_v2.json `
  --confirm-config-frozen
```

This stored-belief report does not replace fresh learned-policy rollouts.

## Repository Layout

```text
ARIA/
├── app.py                         # FastAPI and WebSocket orchestrator
├── frontend/                      # React/Vite candidate interface
├── config/                        # Runtime and model settings
├── modules/                       # Modules 1–15
│   ├── module_01_stt/
│   ├── module_02_vision/
│   ├── ...
│   └── module_15_feedback/
├── tests/                         # Unit, integration, RL, and benchmark tests
├── tools/                         # Live calibration utilities
├── docs/examples/belief_v2/      # Non-benchmark artifact schema examples
├── architecture.md               # Detailed system design reference
└── ARIA_RL_IMPROVEMENT_CHANGES.md
```

## Hardware Target

ARIA is designed to operate locally on a system with an NVIDIA GeForce RTX 4060 and 8 GB of VRAM. The architecture uses quantized models, staged loading, and modular execution to stay within that budget. CPU execution is possible for selected modules but will be slower, and the full perception stack requires its external model weights to be present under `models/` or configured through environment variables.

## Next Milestones

1. Build a balanced, de-identified source corpus and regenerate valid raw evidence.
2. Run belief-v2 preparation, quality gates, IQL training, and fresh policy rollouts.
3. Wire the selected IQL policy and calibrated semantic evidence into `app.py`.
4. Connect vision, prosody, fusion, cognitive-load, integrity, and fairness signals to every live turn.
5. Complete the report/feedback user flow and replace the avatar baseline with synchronized output.
6. Perform privacy, accessibility, fairness, adversarial, and human-review validation before any real hiring use.

## Further Documentation

- [System architecture](architecture.md)
- [RL benchmark diagnosis](ARIA_Benchmark_Diagnosis.md)
- [Belief-v2 and replay-v2 changes](ARIA_RL_IMPROVEMENT_CHANGES.md)
- [Frontend guide](frontend/README.md)
- [Coding assistant and module contracts](ARIA_Coding_Assistant_Guide.md)
