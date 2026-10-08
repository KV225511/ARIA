# Repository Guidelines

> **IMPORTANT — Living guide:** At the start of every session, review this file against the current task and repository state. Update it when goals, architecture, workflows, or constraints change; ask the user focused questions when those changes cannot be resolved from the repository or prior instructions. Do not let outdated guidance persist between sessions.

## Goal and Current State

ARIA helps students and job seekers practice role-specific technical interviews and receive evidence-based feedback on strengths and gaps. The immediate deliverable is a reliable application: upload a résumé and job description, answer by text or audio, follow session progress, receive adaptive questions, recover from failures, and finish with a meaningful report. Educators and coaches are secondary users. Complete and integrate existing modules before starting new benchmark campaigns. ARIA is not an automated hiring system.

Today, the FastAPI/React path uses Google OpenID Connect followed by ARIA JWT/refresh sessions, owner-scoped PostgreSQL records, private PDF storage, durable document/session jobs, persisted questions and answers, versioned runtime checkpoints, history, explicit completion/cancellation, and an authenticated WebSocket with a single-writer lease. Recorded WebM audio is transient and only its transcript is retained. The live loop still uses placeholder evidence scores and a fixed three-action cycle; it does not load the saved calibrated belief configuration or trained policy checkpoint. Several multimodal modules and final reporting remain disconnected. Document this behavior honestly until integration is verified.

## Specification Approval Gate

Authentication and durable interview ownership follow the approved `docs/specs/001-auth-and-interview-persistence.md` version 1.0. Google is the initial login provider; local PostgreSQL and private filesystem storage are the development defaults, with Neon and private object storage proposed for a later hosted pilot. Record implementation and acceptance evidence against requirement IDs; amend the approved specification before changing its contracts.

The served ASGI application is `backend.api` re-exported from `app.py`; the former in-memory routes remain only as unserved compatibility code in `app.py` and must not regain authority. Module 15's standalone SQLite trajectory logger remains separate from the live PostgreSQL account/history store. Preserve existing offline datasets and model artifacts; do not infer owners for legacy data.

## Architecture and Boundaries

Target flow: résumé and job description → grounded skill ontology → question → candidate response → perception and scored evidence → per-skill competency beliefs → **33-feature policy state and valid-action mask** → frozen eight-action policy → grounded next question → frontend. At completion, recorded evidence and the interview trajectory feed evaluation, feedback, and permitted persistence. The **72-feature multimodal fusion vector is separate** from the policy state.

Integrate the existing calibrated belief configuration with its matching frozen trained policy checkpoint from the saved release artifacts. Load the checkpoint with the corresponding protocol, protocol state, development bundle, and split manifest; preserve their hashes, feature order, action definitions, and mask rules. The saved synthetic classification result does not establish learned-policy benefit or human-interview performance.

The frontend owns inputs, media permissions, recording, transcript, progress, connection state, and results. The backend owns authoritative session state, orchestration, streaming, timeouts, recovery, and cleanup. Perception returns evidence with reliability and availability indicators; missing audio or video must not count as weak performance. Beliefs retain calibrated evidence semantics. The policy selects a legal strategy; question generation expresses it without overriding the action or inventing evidence. Reports must show observed evidence, coverage, and uncertainty. Keep offline data preparation, training, and evaluation separate from live operation.

## Project Structure and Priorities

`app.py` hosts the FastAPI/WebSocket orchestrator; `frontend/src/` holds the React/Vite UI. `modules/module_01_*` through `module_15_*` contain the perception, ontology, belief, policy, generation, analysis, reporting, and feedback components. Settings are in `config/`, pytest coverage in `tests/`, and architecture details in `architecture.md` and `docs/`.

Work in this order: define module and frontend/backend contracts; connect real evidence, calibrated beliefs, state, masks, provenance-checked checkpoint inference, and grounded questions; complete setup, recording, streaming, interruption, progress, and results; integrate remaining modalities, auxiliary analysis, fairness, reporting, storage, and speech output; then verify complete sessions, concurrency isolation, reconnects, cancellation, model failures, and cleanup. Optional modules need availability checks and explicit fallbacks. A standalone demonstration does not complete a user feature. Loading the frozen checkpoint is integration work; changing its schema, objective, or weights is a separate model change. Measure latency and concurrency before setting targets.

## Commands, Style, and Tests

From the root: `pip install -r requirements.txt`, `uvicorn app:app --reload --port 8000`, and `pytest -q`. From `frontend/`: `npm ci`, `npm run dev`, `npm run lint` (Oxlint), and `npm run build`. On Windows, `./run.ps1` launches both services and clears port 8000. Use four-space Python and two-space JSX indentation, `snake_case` Python names, and `PascalCase.jsx` components. No repository-wide Python formatter or coverage threshold is configured.

Name pytest files and functions `test_*`; mark tests requiring GPU models or openSMILE `integration`. Add focused regression coverage for changed contracts and complete session paths: real evidence reaching beliefs and policy, legal actions and stopping, frontend/backend state agreement, fallback, recovery, and report consistency.

## Commits and Pull Requests

Commit regularly during development: make a separate commit for each completed, coherent feature, fix, chore, or meaningful improvement to an existing feature. Keep unrelated changes apart and avoid splitting one logical change into commits that cannot stand on their own. Run relevant checks before committing. Stage only the intended files, then review `git status` and `git diff --cached` so private data, generated artifacts, and unrelated work stay out of the commit.

Use a short, imperative subject describing the outcome, such as `feat: stream interview questions`, `fix: retain session on reconnect`, or `chore: update setup guide`. Add a body when the reason or tradeoff is not obvious. History already uses `feat:` and `chore:`; use `fix:` for bug repairs and `feat:` for user-visible improvements. In pull requests, describe the behavior change, link relevant issues, report test and lint results, and include screenshots for UI changes.

## Data and Model Safeguards

Preserve the existing dataset and trained model by default. Regeneration, retraining, or changed research assumptions require explicit authorization. Preserve feature ordering, actions, masks, calibration, and checkpoint compatibility; document intentional contract changes. Maintain training, validation, and locked-test separation. Keep processing local-first with configurable Ollama models; external processing of candidate data requires an explicit decision. Keep secrets and private candidate material out of commits and routine logs. Define consent, retention, and deletion for interview data. Treat behavioral, fairness, and anti-gaming signals within their limits; never silently turn them into definitive competence or honesty judgments. Never fabricate scores, transcripts, reports, or successful responses to hide failures.
