# ARIA — interview answer anchors

Use this pattern: **what it does → why we chose it → one implementation detail → limitation or next step.** Expand only when the interviewer asks.

## Project pitch

- **Name:** Autonomous Reinforcement-Based Interview Agent.
- **Problem:** fixed interview scripts do not adapt to a candidate's answers or uncertainty.
- **Approach:** track skill evidence and choose the next interview strategy.
- **Architecture:** React interface + FastAPI orchestration + local models + separate offline learning pipeline.
- **Status:** research prototype; the live app and complete learned/multimodal loop are not fully integrated.

## Four responsibilities to remember

| Component | Answer anchor |
|---|---|
| Skill graph | What skill should we investigate? |
| Belief model | What do we know, and how uncertain are we? |
| Policy | What strategy should we use next? |
| LLM | How should we word the question? |

**Example:** SQL answer → update SQL belief → choose harder question → select advanced skill → generate wording.

## Live request flow

- Upload résumé + JD PDFs → extract text → build grounded skill graph.
- Create UUID session → store graph, beliefs, history and turn count in memory.
- Open WebSocket → generate first question.
- Typed answer goes directly to text; recorded audio goes **WebM → ffmpeg WAV → faster-whisper transcript**.
- Append history → update belief → choose strategy/skill → generate and validate next question.
- React displays the question and speaks it using browser TTS.
- Disconnect currently deletes the session.

## Technology choices

| Choice | Why / tradeoff |
|---|---|
| React + Vite | Component-based interface and development tooling. |
| FastAPI | Python API layer near the ML modules; async support. |
| WebSocket | Repeated bidirectional messages; requires connection recovery and flow control. |
| PyMuPDF | Extract text from uploaded PDFs; scanned PDFs may need OCR. |
| ffmpeg + faster-whisper | Normalize recorded audio, then transcribe locally. |
| Ollama | Run local language models; model latency and memory become local constraints. |
| In-memory sessions | Simple prototype; no persistence or sharing across server processes. |

## Skill graph and grounding

- Nodes = skills; edges = prerequisite relationships.
- Predecessors support foundation questions; successors support harder topics.
- Adjacency maps make neighbor retrieval straightforward; cycle validation protects the graph.
- Grounding ties the target/question to JD evidence.
- Question checks include duplicates, length and lexical support; up to three generation attempts.
- **Tradeoff:** explicit control and testability, but extra modeling; lexical checks do not prove correctness.

## Belief model

- Per skill: probabilities over **Beginner / Mid / Expert**.
- New semantic evidence updates the prior into a posterior.
- Confidence affects evidence weight; repetition is discounted and effective evidence capped.
- Aggregate visited-skill evidence; abstain when confidence, coverage or evidence is insufficient.
- Entropy measures uncertainty; certainty does not guarantee correctness.
- Current competency likelihood uses semantic evidence; the behavior-score argument does not affect it.

## Policy and RL

- **POMDP idea:** competence is hidden; answers are imperfect observations.
- **State:** 33 features summarizing beliefs, uncertainty, coverage and previous signals/actions.
- **Actions:** harder, easier, follow-up, switch topic, foundation, behavioral, situational, conclude.
- **Reward:** information gain + coverage, minus time/repetition/distress/invalid-action costs; terminal classification shaping.
- **Masks:** exclude illegal actions, including premature conclusion.
- **Offline RL:** learn from recorded interviews instead of experimenting on live candidates.
- **IQL:** two Q networks, a value network and a policy; favor recorded actions with higher estimated advantage.
- **Limitation:** depends on dataset quality and action coverage.

## Data and evaluation

- Separate candidate and evaluator models during synthetic generation.
- Split connected résumé/JD identities together to reduce leakage.
- Keep raw evidence immutable; replay derived beliefs, states and rewards after calibration changes.
- Use versioned configurations, hashes and checkpoint compatibility checks for reproducibility.
- Freeze choices before locked testing.
- Stored-answer evaluation tests belief classification; different policy questions require fresh responses.

## All 15 modules, at a glance

| Modules | Purpose | Live status |
|---|---|---|
| 1: Speech + semantics | Transcribe and grade answers | Transcription live; real grading not wired into live updates. |
| 2–4: Vision, prosody, fusion | Combine visual, acoustic and semantic features | Standalone; 72-feature fusion differs from the 33-feature policy state. |
| 5–6: Ontology + belief | Organize skills and accumulate evidence | Live; evidence inputs currently placeholders. |
| 7–8: Policy + LLM | Choose strategy and write questions | LLM live; learned policy separate. |
| 9: TTS/avatar | Deliver questions | Browser TTS live; server/avatar prototype separate. |
| 10–13: Cognitive load, anti-gaming, incongruence, fairness | Context and auditing signals | Not orchestrated in every live turn. |
| 14–15: Evaluation + feedback | Reports and SQLite trajectory/outcome logging | Separate services; incomplete live product flow. |

## Current limitations — know these exactly

- Live semantic score is fixed at **0.9**; answers are not truly graded there yet.
- Live strategies cycle through **foundation → harder → switch topic**.
- The live app does not load the trained IQL checkpoint or calibrated v7 belief configuration.
- Camera preview does not mean vision analysis is running.
- Current live delivery sends a complete validated question; audio uploads after recording stops.

## Results you can defend

- Saved v7 synthetic test: **91 interviews; 91.2% accuracy/micro-F1; 91.7% macro-F1**.
- This measures **stored-evidence competency classification**, not live learned-policy performance.
- Offline policy diagnostic and fresh-rollout execution code exist; their existence is not proof of policy benefit.
- Human-interview generalization remains a separate validation question.

## Engineering follow-ups

| Topic | Answer anchors |
|---|---|
| Scale | Shared session storage; durable records; separate inference workers; bounded GPU queue. |
| Async | `async def` does not make blocking work nonblocking; offload it appropriately. |
| Reliability | Timeouts, structured errors, validated fallbacks, reconnect support, idempotent turn IDs. |
| Latency | Measure upload, conversion, transcription, generation and retry time before optimizing. |
| Testing | Graph/belief unit tests; replay/checkpoint contract tests; API integration with model stubs. |
| Hardening | Authentication, input limits, subprocess checks, guaranteed temp-file cleanup. |
| Next milestone | Real semantic evidence + calibrated beliefs → validated policy → stopping/reporting → more modalities. |

## Five answers to rehearse

1. What problem does ARIA solve, and what was your own contribution?
2. What happens from an audio answer to the next question?
3. Why separate the graph, belief, policy and LLM?
4. How would you fix one reliability or scaling limitation?
5. What exactly does the 91.2% result measure?

The combined `ARIA-all-diagrams.excalidraw` contains all four diagrams on one canvas. Based on the local code and saved reports inspected on 25 September 2026; no new benchmark was run.
