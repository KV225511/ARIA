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
  --output output/sota/policy_comparison.json `
  --episodes-per-class 10
```

The IQL command validates the checkpoint against the frozen protocol, protocol
state, development bundle, split manifest, and belief configuration before the
first rollout. Every policy receives the same scenario seeds and response model.
These rollouts are controlled simulations and must be reported as such.

## 4. Render a local comparison table

```powershell
python tests/benchmarks/compare_sota.py `
  --text-report output/sota/text/text_benchmark_report.json `
  --policy-report output/sota/policy_comparison.json `
  --output output/sota/local_comparison.md
```

Published SOTA values should be added to the manuscript only after citation
verification. They should not be inserted into the generated local-results table.
