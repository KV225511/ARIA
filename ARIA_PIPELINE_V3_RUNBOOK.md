# ARIA Pipeline v3 Runbook

## Scope

This runbook describes the implemented v3 data-generation, replay, and
training contract. Existing v2 synthetic data is evidence for historical
belief-model experiments only; it must not be used to train the v3 policy.

## Required inputs

- The cleaned OpenSporks resume source at
  `data/external/opensporks/Resume/Resume.cleaned.csv`. It must contain unique
  `Resume_text_hash` values and bounded `Resume_prompt` values.
- At least 32 independently sourced, valid job descriptions for the default
  `20/6/6` train/validation/test component plan. The current filename
  exclusion policy is applied before this count; generation fails closed if
  fewer than 32 remain.
- Ollama with distinct candidate and evaluator models installed.
- Enough disk space for the raw corpus, immutable manifests, derived replay,
  checkpoints, and reports.
- A frozen random seed and a recorded source-document inventory.

## 1. Generate raw v3 evidence

Run the input preflight before contacting Ollama:

```powershell
python -m modules.module_07_rl.generation_preflight --resume-csv data/external/opensporks/Resume/Resume.cleaned.csv --resume-categories INFORMATION-TECHNOLOGY ENGINEERING --identity-components 20 6 6 --output data/synthetic/v3/reports/input_preflight.json
```

Do not start generation unless `passes_preflight` is true. The preflight
validates cleaned-resume hashes, selected categories, extractable JD text,
duplicate JD content, and the unique-document counts required by the component
plan.

Before the production run, use a three-component canary:

```powershell
python -m modules.module_07_rl.generation_preflight --resume-csv data/external/opensporks/Resume/Resume.cleaned.csv --resume-categories INFORMATION-TECHNOLOGY ENGINEERING --identity-components 1 1 1 --output data/synthetic/v3/reports/input_preflight_canary.json
python -m modules.module_07_rl.llm_simulator --sweep --max_episodes 3 --max_concurrent 3 --candidate-request-concurrency 3 --evaluator-request-concurrency 2 --identity-components 1 1 1 --seed 42 --resume-source csv --resume-csv data/external/opensporks/Resume/Resume.cleaned.csv --resume-categories INFORMATION-TECHNOLOGY ENGINEERING --replace-existing --dataset-file data/synthetic/v3/canary/qwen_rl_dataset.json
```

Audit the canary structurally before scaling. Canary metrics are diagnostic and
do not satisfy the production episode-count or held-out evaluation gates.

Run the simulator with an explicit replacement path for a new corpus. The
production target is 600 episodes and 32 independent identity components.

```powershell
python -m modules.module_07_rl.llm_simulator --sweep --max_episodes 600 --max_concurrent 4 --candidate-request-concurrency 3 --evaluator-request-concurrency 2 --identity-components 20 6 6 --seed 42 --resume-source csv --resume-csv data/external/opensporks/Resume/Resume.cleaned.csv --resume-categories INFORMATION-TECHNOLOGY ENGINEERING --replace-existing --dataset-file data/synthetic/v3/qwen_rl_dataset.json
```

CSV generation uses `Resume_prompt` for the model context and
`Resume_text_hash` for leakage-safe identity grouping. The generation manifest
records the cleaned CSV hash, selected categories, selected row count, and
unique content-hash count. `Resume_html` is never loaded into model prompts.

The episode-worker limit controls document/environment work. The two request
limits control actual Ollama calls. The shared client permits parallel calls
for one model, but never overlaps candidate and evaluator model phases. This
avoids dual residency and repeated model swapping on an 8 GB GPU.

Every accepted question transition must contain:

- transition, action, state, reward, and generator schema versions;
- pre-action legality mask;
- full behavior-policy probability vector and selected-action probability;
- content hashes for resume and JD;
- prompt hashes and model identities;
- raw question, candidate answer, evaluator scores, confidence, and rubric
  evidence;
- atomic run-manifest provenance.

Every stop transition must be terminal, have action index 7, have identical
`obs` and `next_obs`, and contain no question, answer, target skill, evaluator
scores, or new belief evidence.

## 2. Audit raw evidence

```powershell
python -m modules.module_07_rl.dataset_audit data/synthetic/v3/qwen_rl_dataset.json --stage raw --min-episodes 600 --output data/synthetic/v3/reports/raw_evidence.json
python -m modules.module_07_rl.dataset_audit data/synthetic/v3/qwen_rl_dataset.json --stage offline_rl --output data/synthetic/v3/reports/offline_support.json
```

Stop immediately if either report fails. In particular, do not repair illegal
actions by relabeling them. Regenerate the affected episodes from their saved
run manifest.

## 3. Freeze the split before calibration

The replay stage creates `split_manifest_v3.json`. It records the raw dataset
hash, per-episode assignment, content-hash identities, independent component
counts, and a locked-test assignment hash. Replay recomputes and verifies both
the manifest hash and raw dataset hash before deriving any state.

The default component targets are:

- train: 20 components;
- validation: 6 components;
- locked test: 6 components.

No resume or JD content hash may appear in more than one split.

## 4. Calibrate beliefs and replay transitions

Use `prepare_calibrate_replay` from
`modules.module_07_rl.replay_dataset` to split raw evidence, fit calibration on
train/validation only, and create v3 replay artifacts. The locked test remains
unreported during this stage.

Required outputs are:

- `belief_model_v2.json` (belief model remains version 2);
- `split_manifest_v3.json`;
- `qwen_rl_dataset_belief_v3.json`;
- `replay_comparison_v3.json`;
- `calibration_report_v3.json`;
- `splits/train.json`, `splits/validation.json`, and `splits/test.json`.

Replay rejects legacy transitions because exact behavior propensities cannot
be reconstructed after the fact.

## 5. Train the IQL policy

```powershell
python -m modules.module_07_rl.train --train-file data/synthetic/v3/derived/splits/train.json --validation-file data/synthetic/v3/derived/splits/validation.json --belief-config data/synthetic/v3/derived/belief_model_v2.json --output modules/module_07_rl/aria_iql_belief_v3.pth --epochs 100 --batch-size 256 --seed 42
```

Training fails closed when schema versions, state semantics, hashes, masks,
propensities, selected-action legality, stop invariants, raw-evidence quality,
offline action support, or validation belief gates fail.

## 6. Evaluation and release status

The stored-belief locked-test evaluator is implemented and writes a v3 report.
It is not a learned-policy evaluation. The next implementation phase must add:

1. weighted importance sampling using logged v3 propensities;
2. doubly robust evaluation with an independently fit evaluator;
3. fitted-Q evaluation with component-bootstrap confidence intervals;
4. fresh deterministic rollouts in which the trained policy chooses each
   action and action masks are enforced;
5. baseline comparisons for return, terminal accuracy, ordinal MAE, interview
   length, invalid-action rate, coverage, and distress rate;
6. a release manifest that binds checkpoint, configuration, split, dataset,
   and evaluation-report hashes.

Do not describe the pipeline as end-to-end policy validated until that phase
passes and the locked test is opened exactly once after configuration freeze.
