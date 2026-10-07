# ARIA Repository Evidence Audit

Audit date: 2026-09-15  
Repository: `C:\Users\Raghav Sejpal\Documents\ChatGPT\ARIA`  
Git branch and head: `main`, `99fd7550a235f97fa950c3037eb4d71d76ffca43`  
Working tree at audit start: clean tracked tree; untracked `.agents/` and `skills-lock.json` only.

## Authority hierarchy

1. Latest immutable experimental artifact with provenance, schema version, and hashes.
2. Current executable code and tests.
3. Current architecture documentation and active runbook.
4. Git history and completed Codex task evidence.
5. Current Word report.
6. Older Markdown reports and design proposals.
7. Aspirational specifications and planned architecture.

Conflicting values are not averaged. Lower-ranked sources are retained as historical context only.

## Verified repository state

- The current RL observation contract is `aria-state-v4`, 33 dimensions. The action schema exposes eight actions, including a legality-gated `conclude_interview` action.
- The production synthetic corpus contains 600 episodes, 9,018 transitions, and 33 connected resume/job-description identity components. Terminal labels are balanced at 200 per class.
- The full raw-evidence audit passed. It found no invalid evaluations, malformed grounding, missing provenance, within-episode duplicate questions, or resume/JD identity leakage across splits.
- Generation distribution gates passed: switch-topic share 0.29924, maximum-turn episode rate 0, and no-evidence-overlap rate 0.30.
- Split populations are 422 train, 87 validation, and 91 locked test episodes.
- The v7 locked artifact is `aria-locked-test-evaluation-v1`, produced by `aria-belief-calibration-v7`, with final status `VALIDATED_SYNTHETIC`. It explicitly sets `evaluates_learned_policy: false`.
- On 91 locked synthetic episodes, stored-belief classification achieved accuracy/micro-F1 0.9120879, macro-F1 0.9167659, balanced accuracy 0.9226551, minimum class recall 0.8285714, ECE 0.0870087, and Brier score 0.1395080.
- The v7 result is an improvement over the paired v6 belief configuration on the same 91 episodes, but the paired component bootstrap is explicitly diagnostic.
- The selected IQL checkpoint is `aria-iql-checkpoint-v4`, uses `aria-state-v4`, and has SHA-256 `6a9c604bbaf6af987ec631ce27574766b85594af5f248ca88a920fde06c99029`.
- Validation-only clipped weighted importance sampling evaluates the learned policy diagnostically over 1,405 transitions: weighted reward estimate 0.4617542, effective sample size 1003.54, target-policy action entropy 1.63315, and no ratio clipping. The artifact sets `release_gate: false` and `requires_fresh_rollouts_for_causal_claims: true`.
- No current fresh learned-policy rollout result was found. Consequently, the repository does not establish causal improvement from the learned interview policy.
- The live FastAPI/WebSocket application is a partial integration. It performs PDF ingestion, role/ontology adaptation, grounded local question generation, typed/audio answer handling, transcription, and belief updates. Each answer currently supplies fixed semantic and behavior scores of 0.9, cycles only `probe_foundation`, `increase_difficulty`, and `switch_topic`, and does not load the trained IQL policy or the full multimodal/fairness stack.

## Verification commands and outcomes

- Frontend production build: verified, Vite 8 build completed successfully.
- Focused Module 7/calibration suite: verified, 341 tests passed in 18.95 seconds.
- Unscoped repository-wide pytest collection: partially verified. Collection was polluted by tests under `.tmp/review2/node_modules`; project collection also exposed an outdated root `test_ontology.py` call and missing optional `pyttsx3`, `librosa`, and `mediapipe` dependencies. These are reported rather than converted into a pass.
- Complete raw dataset audit: verified and passed.
- Direct audit of the older `derived/qwen_rl_dataset_belief_v3.json`: verified and failed historical calibration gates (micro-F1 0.5633, macro-F1 0.5384, ECE 0.2025, 125 missing/abstained predictions). This file is not the authoritative v7 locked result.
- Existing DOCX structural inspection: completed for both Project I reports. Neither contains comments, tracked insertions/deletions, or field instructions. The original report has 18 pages and six media files; the humanized report has 22 pages and seven media files.
- Existing DOCX visual inspection: completed from the repository's 18-page and 22-page rendered page sets. The original is visibly a template with highlighted sample/placeholding text; the humanized report is readable and substantially complete, but contains stale v2/32-feature/53-episode claims.
- Fresh DOCX rendering was attempted but blocked because LibreOffice `soffice.exe` is unavailable to the managed renderer on Windows. Existing repository renders were therefore used; the limitation remains recorded.
- Repository PDFs: all 33 job-description PDFs and the excluded PDF were parsed end to end. Two are image-only/unreadable by text extraction (`Campus Category Manager...pdf` and `Linux Driver Intern JD.pdf`), and one has an empty second page. The remaining PDFs yielded extractable text.
- Resume evidence: the raw CSV has 2,484 rows; the cleaned CSV has 2,475 rows with no missing fields or duplicate rows. The audit workbook records nine exclusions, 36 email matches redacted, 158 phone matches redacted, 340 prompts limited to 8,000 characters, and all assertions passed.

## Reproducibility classification

| Item | Status | Basis |
|---|---|---|
| v7 artifact identity and metrics | Verified | Immutable JSON fields, hashes, and paired metadata |
| Production corpus integrity | Verified | Complete programmatic audit of all 600 episodes |
| Module 7 implementation contracts | Verified | Current code plus 341 passing focused tests |
| Frontend build | Verified | Successful production build |
| Full Python suite | Partially verified | Collection pollution and missing optional dependencies |
| Learned-policy benefit | Unsupported | Diagnostic WIS only; no fresh selected-policy rollouts |
| End-to-end multimodal live operation | Unsupported | Live loop does not invoke the complete implemented stack |
| Real-candidate validity and fairness | Unsupported | Synthetic corpus; no human-subject deployment study |

