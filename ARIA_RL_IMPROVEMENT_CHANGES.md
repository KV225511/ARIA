# ARIA RL Benchmark Improvement Changes

## Goal

Improve the ARIA RL benchmark from roughly 40% micro-F1 toward a credible 80-90% heldout micro-F1.

The key point is that the original low score was not caused by a single bug. It came from a combination of weak belief aggregation, noisy evaluator evidence, random-policy data, unsafe splits, and evaluation that reported stored belief labels rather than a clearly separated learned-policy result.

## Why These Changes Were Needed

The benchmark showed a strong prediction collapse:

```text
aria_label_counts: {0: 423, 1: 76, 2: 1}
true_label_counts: {0: 180, 1: 165, 2: 155}
```

That means ARIA mostly predicted Beginner, almost never predicted Expert, and therefore had poor recall for Mid and Expert candidates. Since this is a single-label three-class task, micro-F1 is effectively the same as accuracy, so the collapse directly limited the score to around 40%.

The target improvement requires an honest pipeline first. Raising the score by leaking the candidate tier into the simulator or evaluation would make the benchmark meaningless. The changes below focus on improving the system without using the true label as evidence.

## Changes Made

### 1. Repaired the Belief Pipeline

Files changed:

- `modules/module_06_belief/belief_state.py`
- `modules/module_07_rl/environment.py`
- `modules/module_07_rl/llm_simulator.py`
- `modules/module_08_llm/generator.py`
- `modules/module_07_rl/metrics.py`

What changed:

- Replaced fragile single-node terminal verdicts with aggregate belief across visited skills.
- Added deterministic node ordering so runs are reproducible.
- Normalized entropy by `log(3)` so the observation space remains inside the declared `[0, 1]` bounds.
- Passed an explicit target skill into question generation and belief updates.
- Increased the minimum interview length and required broader skill coverage before conclusion.
- Made benchmark reporting explicit that stored `aria_label` values are belief verdicts, not learned-policy rollouts.

Why:

The old terminal verdict could be based on one noisy skill node. If that node received a conservative score, the final label often collapsed to Beginner even when other evidence existed. Aggregating visited skill beliefs makes the final assessment more stable and better aligned with the whole interview.

### 2. Improved Synthetic Data Quality

Files changed:

- `modules/module_07_rl/llm_simulator.py`
- `modules/module_06_belief/belief_state.py`
- `modules/module_07_rl/environment.py`
- `modules/module_07_rl/dataset_audit.py`

What changed:

- Added separate candidate and evaluator model configuration:
  - `ARIA_CANDIDATE_MODEL`
  - `ARIA_EVALUATOR_MODEL`
- Removed persona leakage from interviewer prompts.
- Added structured evaluator output with semantic score, behavior score, cognitive load, confidence, and rubric evidence.
- Added evaluator retries and invalid-evaluation rejection.
- Added balanced persona cycling across Beginner, Mid, and Expert candidates.
- Replaced pure random data collection with a mixed behavior policy:
  - exploration
  - coverage-oriented questioning
  - belief-oriented questioning
  - confidence-aware conclusion
- Added dataset auditing for label balance, prediction collapse, score spread, reward spread, invalid evaluator outputs, model overlap, and split leakage.

Why:

The old dataset was collected with a nearly random policy and weak, low-variance rewards. Offline RL cannot learn a good interview policy from random actions and flat rewards. The improved simulator produces more useful trajectories while still avoiding true-label leakage.

### 3. Added Leakage-Safe Dataset Splits

Files changed:

- `modules/module_07_rl/dataset_split.py`
- `modules/module_07_rl/dataset_audit.py`
- `modules/module_07_rl/train.py`
- `modules/module_07_rl/metrics.py`

What changed:

- Added connected-component dataset splitting across resumes and job descriptions.
- Ensured any shared resume or job description stays entirely within one split.
- Added audit checks for resume leakage and job-description leakage.
- Split data into train, validation, and test sets before training.
- Trained only on the train split.
- Reported heldout stored-belief metrics separately from policy evaluation.
- Added terminal-only supervised outcome reward shaping for training transitions.

Why:

If the same resume or JD appears in both train and test, high scores can come from memorization instead of generalization. Leakage-safe splitting is required before treating any 80-90% micro-F1 result as real.

### 4. Added Validation-Only Belief Calibration

Files changed:

- `modules/module_07_rl/belief_calibration.py`
- `modules/module_06_belief/belief_state.py`
- `modules/module_07_rl/environment.py`

What changed:

- Added a calibration script that replays validation episodes and tests candidate belief likelihood sigmas.
- Selects sigma using validation micro-F1, with macro-F1 as a tie-breaker to avoid class collapse.
- Allows runtime configuration through `ARIA_BELIEF_SIGMA`.

Why:

The Gaussian likelihood width controls how strongly semantic scores separate Beginner, Mid, and Expert. Too wide, and classes bleed together. Too tight, and individual noisy scores dominate. Calibrating on validation data gives a principled way to tune this without touching the test set.

## Tests Added or Updated

Tests were added or expanded for:

- belief aggregation and confidence handling
- simulator behavior and evaluator parsing
- dataset audit checks
- leakage-safe splitting
- metrics reporting
- validation-only belief calibration
- RL environment behavior

The targeted test suite passed after the changes:

```text
32 passed
```

## What Still Needs To Happen

The implementation work is in place, but the score cannot be honestly raised to 80-90% until a clean dataset is regenerated and evaluated.

Current blocker:

- `data/resumes/` has no resume PDFs.
- `data/jds/` has job-description PDFs.
- The simulator requires both valid resumes and valid job descriptions.
- Old resumes were intentionally removed from the repository and ignored by `.gitignore`, so they should not be restored without explicit approval.

Next required steps:

1. Add valid resume PDFs under `data/resumes/`.
2. Regenerate the RL dataset with distinct candidate and evaluator models.
3. Run the dataset audit and reject the dataset if it shows collapse, leakage, or flat rewards.
4. Calibrate belief sigma on the validation split only.
5. Retrain IQL on the train split.
6. Evaluate on the heldout test split.
7. Report both stored-belief metrics and actual learned-policy rollout metrics.

## Expected Impact

These changes should improve micro-F1 by fixing the main structural causes of Beginner collapse:

- final predictions now use global evidence instead of one node
- interviews collect broader skill evidence before conclusion
- evaluator confidence affects belief updates
- synthetic data is more balanced and informative
- train/test leakage is guarded against
- belief sharpness can be tuned on validation data

The remaining jump to 80-90% depends on regenerating a high-quality dataset and confirming the result on a leakage-safe heldout test split.
