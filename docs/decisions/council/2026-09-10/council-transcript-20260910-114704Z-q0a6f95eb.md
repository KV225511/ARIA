# LLM Council Transcript

MODE: Standard (user invoked council)
Timestamp: 2026-09-10T11:47:04Z

## Framed question

DECISION: Whether the ARIA calibration-repair plan should be approved unchanged or corrected before Terra-light implementation.

CONTEXT: Preserve the exact 600-episode, 9,018-transition raw dataset; preserve locked-test assignments; correct abstention-aware metrics, split allocation, calibration selection, audit gates, provenance, and fail-closed training.

STAKES: A flawed repair could leak locked-test information, overfit six validation components, or produce unsupported end-to-end validity claims.

## Bias audit

Flags: false precision in policy thresholds; validation-selection Goodhart risk; solution fixation on calibration; complexity bias from bundling changes; dataset immutability becoming dogmatic if defects prove irreparable.

## Advisor synthesis

- Red Team: hashes preserve membership but do not prevent test-label access; the 1,350-candidate search is too broad for 21 components.
- First Principles: treat observed validation as development evidence; select solely through grouped training resampling and reserve the locked test for one release evaluation.
- Expansionist: retain Pareto candidates and select for robust margin; use leave-one-component-out only as a diagnostic; ordinal decision policy is a later fallback.
- Outsider: preregister the protocol and use FAILED, PROVISIONAL, and VALIDATED states.
- Executor: implement metrics, audit, split migration, existing calibration, and optional expanded tuning as separate checkpoints.

## Peer review

Consensus scores: 4, 5, 5, 4, 4. Average: 4.4/5.

Shared concerns: validation is no longer pristine; access isolation is stronger than hashes; synthetic internal validity is not real-world competency validity; environment and schema compatibility require explicit contracts.

## Debate round

Prosecutor: the pipeline may only learn simulator-persona reconstruction; exact class balance and common synthetic mechanisms limit construct validity.

Defender: controlled balance is appropriate for internal infrastructure validation, and the clean provenance/action/grounding audits are useful evidence, but the result must remain PROVISIONAL—SYNTHETIC until external human evaluation.

## Final verdict

Approve only after correction. Implement in independently verified stages. Isolate locked-test inputs architecturally; freeze and hash the analysis protocol before calibration; rerun the current calibrator before expanding the search; compute ECE from three-class probabilities and report abstention separately; preserve schema/environment provenance; and limit the claim to internal synthetic-pipeline validation.

## Dissent ledger

- Passing validates reproducibility within the simulator, not real competency.
- ECE semantics for abstentions must be preregistered.
- Schema compatibility and dependency pinning are release requirements.
- The protocol, code revision, lockfile, manifest, gates, search space, and release command should all be hashed before test access.
