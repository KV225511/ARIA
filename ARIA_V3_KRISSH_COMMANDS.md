# ARIA v3 Commands for Krissh

Run the following sections in order from PowerShell. Do not use the legacy v2
dataset for v3 training, and do not proceed past a failed gate.

## 1. Add six technical job descriptions

Place at least six independently sourced, text-extractable technical JD PDFs in:

```text
C:\Users\kriss\github\ARIA\data\jds
```

Every JD must:

- contain at least 100 extractable characters;
- represent an IT or engineering role;
- be meaningfully different from every other JD;
- not be a renamed or copied duplicate.

## 2. Configure PowerShell

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "C:\Users\kriss\github\ARIA"
$PY = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$ResumeCsv = "data\external\opensporks\Resume\Resume.cleaned.csv"

& $PY --version
```

The interpreter must be Python 3.11 with the ARIA dependencies installed. If
the virtual environment is unavailable, repair it or assign `$PY` to the full
path of a working Python 3.11 executable.

## 3. Run the production input preflight

```powershell
& $PY -m modules.module_07_rl.generation_preflight `
  --resume-csv $ResumeCsv `
  --resume-categories INFORMATION-TECHNOLOGY ENGINEERING `
  --identity-components 20 6 6 `
  --output "data\synthetic\v3\reports\input_preflight.json"

if ($LASTEXITCODE -ne 0) {
    throw "Production input preflight failed."
}

$Preflight = Get-Content -Raw `
  "data\synthetic\v3\reports\input_preflight.json" |
  ConvertFrom-Json

if (-not $Preflight.passes_preflight) {
    throw "passes_preflight is false."
}

if ($Preflight.shortfall.unique_readable_jds -ne 0) {
    throw "The production JD pool still has a shortfall."
}
```

Proceed only when `passes_preflight` is `true` and
`shortfall.unique_readable_jds` is `0`.

## 4. Verify Ollama

```powershell
ollama list
ollama ps
```

The installed-model list must include:

```text
qwen2.5:7b
gemma3:4b
```

Install either missing model with:

```powershell
ollama pull qwen2.5:7b
ollama pull gemma3:4b
```

Configure the generation process:

```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
$env:ARIA_CANDIDATE_MODEL = "qwen2.5:7b"
$env:ARIA_EVALUATOR_MODEL = "gemma3:4b"
$env:ARIA_OLLAMA_NUM_CTX = "4096"
$env:ARIA_OLLAMA_TIMEOUT = "300"
$env:ARIA_OLLAMA_KEEP_ALIVE = "0"
```

Keep the candidate and evaluator models different.

## 5. Run the canary preflight

```powershell
& $PY -m modules.module_07_rl.generation_preflight `
  --resume-csv $ResumeCsv `
  --resume-categories INFORMATION-TECHNOLOGY ENGINEERING `
  --identity-components 1 1 1 `
  --output "data\synthetic\v3\reports\input_preflight_canary.json"

if ($LASTEXITCODE -ne 0) {
    throw "Canary input preflight failed."
}

$CanaryPreflight = Get-Content -Raw `
  "data\synthetic\v3\reports\input_preflight_canary.json" |
  ConvertFrom-Json

if (-not $CanaryPreflight.passes_preflight) {
    throw "Canary passes_preflight is false."
}
```

## 6. Generate the three-episode canary

The following command intentionally replaces the fixed canary dataset. Copy it
first if an earlier canary must be retained.

```powershell
$CanaryDataset = "data\synthetic\v3\canary\qwen_rl_dataset.json"

& $PY -m modules.module_07_rl.llm_simulator `
  --sweep `
  --max_episodes 3 `
  --max_concurrent 3 `
  --candidate-request-concurrency 3 `
  --evaluator-request-concurrency 2 `
  --identity-components 1 1 1 `
  --seed 42 `
  --resume-source csv `
  --resume-csv $ResumeCsv `
  --resume-categories INFORMATION-TECHNOLOGY ENGINEERING `
  --replace-existing `
  --dataset-file $CanaryDataset

if ($LASTEXITCODE -ne 0) {
    throw "Canary generation failed."
}
```

## 7. Audit the canary

```powershell
& $PY -m modules.module_07_rl.dataset_audit `
  $CanaryDataset `
  --stage raw `
  --min-episodes 3 `
  --output "data\synthetic\v3\canary\reports\raw_evidence.json"

if ($LASTEXITCODE -ne 0) {
    throw "Canary audit command failed."
}

$CanaryAudit = Get-Content -Raw `
  "data\synthetic\v3\canary\reports\raw_evidence.json" |
  ConvertFrom-Json

if (-not $CanaryAudit.passes_quality_gates) {
    throw "Canary quality gates failed."
}
```

Before proceeding, confirm the report shows:

- three completed episodes and three identity components;
- no missing masks or propensities;
- no illegal actions or invalid stop transitions;
- no empty questions, answers, or evaluations;
- `passes_quality_gates: true`.

## 8. Generate the 600-episode production corpus

This command intentionally replaces the fixed v3 production dataset. Preserve
any existing file before running it if that file is needed.

```powershell
$ProductionDataset = "data\synthetic\v3\qwen_rl_dataset.json"

& $PY -m modules.module_07_rl.llm_simulator `
  --sweep `
  --max_episodes 600 `
  --max_concurrent 4 `
  --candidate-request-concurrency 3 `
  --evaluator-request-concurrency 2 `
  --identity-components 20 6 6 `
  --seed 42 `
  --resume-source csv `
  --resume-csv $ResumeCsv `
  --resume-categories INFORMATION-TECHNOLOGY ENGINEERING `
  --replace-existing `
  --dataset-file $ProductionDataset

if ($LASTEXITCODE -ne 0) {
    throw "Production generation failed."
}
```

Do not interrupt Ollama or change either model during this run.

## 9. Audit the production corpus

```powershell
& $PY -m modules.module_07_rl.dataset_audit `
  $ProductionDataset `
  --stage raw `
  --min-episodes 600 `
  --output "data\synthetic\v3\reports\raw_evidence.json"

if ($LASTEXITCODE -ne 0) {
    throw "Production raw-evidence audit failed."
}

& $PY -m modules.module_07_rl.dataset_audit `
  $ProductionDataset `
  --stage offline_rl `
  --output "data\synthetic\v3\reports\offline_support.json"

if ($LASTEXITCODE -ne 0) {
    throw "Production offline-RL audit failed."
}

$RawAudit = Get-Content -Raw `
  "data\synthetic\v3\reports\raw_evidence.json" |
  ConvertFrom-Json

$OfflineAudit = Get-Content -Raw `
  "data\synthetic\v3\reports\offline_support.json" |
  ConvertFrom-Json

if (-not $RawAudit.passes_quality_gates) {
    throw "Production raw-evidence quality gates failed."
}

if (-not $OfflineAudit.passes_quality_gates) {
    throw "Production offline-RL quality gates failed."
}
```

If either report fails, stop. Do not manually repair, relabel, or reconstruct
transitions.

## 10. Create leakage-safe splits and replay beliefs

```powershell
$DerivedRoot = "data\synthetic\v3\derived"

& $PY -m modules.module_07_rl.prepare_belief_pipeline `
  $ProductionDataset `
  $DerivedRoot `
  --split-seed 42 `
  --bootstrap-samples 1000

if ($LASTEXITCODE -ne 0) {
    throw "Split, calibration, or belief replay failed."
}

$RequiredFiles = @(
  "$DerivedRoot\belief_model_v2.json",
  "$DerivedRoot\split_manifest_v3.json",
  "$DerivedRoot\qwen_rl_dataset_belief_v3.json",
  "$DerivedRoot\replay_comparison_v3.json",
  "$DerivedRoot\calibration_report_v3.json",
  "$DerivedRoot\splits\train.json",
  "$DerivedRoot\splits\validation.json",
  "$DerivedRoot\splits\test.json"
)

$MissingFiles = $RequiredFiles | Where-Object { -not (Test-Path $_) }

if ($MissingFiles) {
    $MissingFiles
    throw "Required derived artifacts are missing."
}
```

Do not inspect or evaluate `test.json` yet.

## 11. Train the IQL policy

```powershell
& $PY -m modules.module_07_rl.train `
  --train-file "$DerivedRoot\splits\train.json" `
  --validation-file "$DerivedRoot\splits\validation.json" `
  --belief-config "$DerivedRoot\belief_model_v2.json" `
  --output "modules\module_07_rl\aria_iql_belief_v3.pth" `
  --epochs 100 `
  --batch-size 256 `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    throw "IQL training failed."
}
```

Training must pass the raw-evidence, offline-support, and validation-belief
gates. Do not bypass a failed gate.

## 12. Stop before unlocking the test set

Do not run `evaluate_locked_test` yet. Complete and freeze the following first:

1. weighted importance sampling (WIS);
2. doubly robust evaluation;
3. fitted-Q evaluation (FQE);
4. component-bootstrap confidence intervals;
5. fresh trained-policy rollouts;
6. baseline comparisons;
7. the final release manifest.

After those evaluators and all configuration are frozen, open the locked test
exactly once.
