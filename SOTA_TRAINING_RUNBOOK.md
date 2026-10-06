# ARIA SOTA experiment runbook

This pipeline produces repository-backed evidence for the paper. It separates
locally reproduced measurements from values cited from published papers and never
uses fallback scores when a dataset, dependency, or model artifact is missing.

## 1. Audit the available evidence

```powershell
python tools/run_sota_training.py preflight --output output/sota/preflight.json
```

The report records dependency versions, asset paths, file hashes, and the tasks
that can run. A missing dataset remains `missing`; it does not become a cached or
estimated result.

## 2. Train text baselines

```powershell
python tools/run_sota_training.py text `
  --dataset data/model3_text/mohler_dataset.csv `
  --output-dir output/sota/text `
  --seed 42
```

The runner groups records by question, freezes train/validation/test assignments,
selects hyperparameters using validation macro-F1, and evaluates the held-out test
partition once. It writes the split manifest, predictions, and a hashed JSON report.

## 3. Compare interview policies

Train the same-data comparator checkpoints without changing ARIA's dataset or
frozen IQL checkpoint:

```powershell
python tools/run_sota_training.py train-policy-comparators `
  --output-dir output/sota/direct_comparators `
  --epochs 20 `
  --seed 42
```

This trains a behavior-cloning baseline, discrete Conservative Q-Learning, and a
Decision Transformer adaptation. CQL and Decision Transformer are published
architectures; the repository implementations are compact ARIA-specific
adaptations rather than the authors' original code. The registry in
`config/sota_comparator_registry.json` records the verified paper sources and
comparison scope.

Run the scripted baselines alone:

```powershell
python tools/run_sota_training.py policy `
  --output output/sota/policy_baselines.json `
  --episodes-per-class 10
```

Include the repository's validated v7 IQL checkpoint:

```powershell
python tools/run_sota_training.py policy `
  --include-iql `
  --comparator-dir output/sota/direct_comparators `
  --output output/sota/policy_comparison.json `
  --episodes-per-class 10
```

The IQL command validates the checkpoint against the frozen protocol, protocol
state, development bundle, split manifest, and belief configuration before the
first rollout. Every policy receives the same scenario seeds and response model.
These rollouts are controlled simulations and must be reported as such.

Run prespecified robustness conditions with `--condition overlap`,
`--condition low_confidence`, and `--condition positive_shift`. Keep the same
seed and number of episodes for every condition. The runner adds bootstrap
confidence intervals, paired differences relative to IQL, and an accuracy-ceiling
audit. A saturated accuracy result is diagnostic evidence and must not be used to
rank policies.

Create a compact manuscript-ready record from the full reports:

```powershell
python tools/summarize_policy_comparisons.py `
  --training-report output/sota/direct_comparators/training_report.json `
  --reports output/sota/direct_policy_base.json output/sota/direct_policy_overlap.json output/sota/direct_policy_low_confidence.json output/sota/direct_policy_positive_shift.json `
  --output-dir paper/generated_sota
```

## 4. Render a local comparison table

```powershell
python tests/benchmarks/compare_sota.py `
  --text-report output/sota/text/text_benchmark_report.json `
  --policy-report output/sota/policy_comparison.json `
  --output output/sota/local_comparison.md
```

Published SOTA values should be added to the manuscript only after citation
verification. They should not be inserted into the generated local-results table.
