# ARIA v3 Operator Runbook

This runbook preserves the v2 corpus, generates an isolated v3 canary, applies
fail-closed validation gates, and only then permits production generation and
training.

## Current prerequisites and blockers

- The existing `.venv` points to a missing Microsoft Store Python installation.
- `data/resumes` currently has no PDFs.
- `data/jds` has 32 PDFs, but the loader excludes several; the production plan
  requires at least 32 valid resumes and 32 valid JDs.
- The current learned-policy evaluation is incomplete. Training a checkpoint is
  not equivalent to end-to-end policy validation.
- Do not modify, append to, or convert the v2 corpus into v3 policy data.

## 1. Open PowerShell in the repository

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "C:\Users\Raghav Sejpal\Documents\ChatGPT\ARIA"
```

Record the v2 corpus hash:

```powershell
$V2Dataset = "data\synthetic\qwen_rl_dataset_499.json"
Get-FileHash -Algorithm SHA256 $V2Dataset
```

Never use `$V2Dataset` as the destination of a generation command.

## 2. Repair Python and create a clean environment

Check the installed Python interpreters:

```powershell
py -0p
py -3.11 --version
```

If Python 3.11 does not start, install or repair it:

```powershell
winget install --exact --id Python.Python.3.11
```

Close and reopen PowerShell after installation. Then create a new environment
without overwriting the broken `.venv`:

```powershell
Set-Location "C:\Users\Raghav Sejpal\Documents\ChatGPT\ARIA"

py -3.11 --version
py -3.11 -m venv .venv311

$PY = (Resolve-Path ".\.venv311\Scripts\python.exe").Path

& $PY -m pip install --upgrade pip setuptools wheel
& $PY -m pip install -r requirements.txt
```

Verify the required imports:

```powershell
& $PY -c "import torch, numpy, sklearn, httpx, pdfplumber, pypdf; print('imports OK'); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

## 3. Run the regression suite

```powershell
& $PY -m pytest -q `
  tests\test_llm_simulator.py `
  tests\test_dataset_split.py `
  tests\test_dataset_audit.py `
  tests\test_belief_calibration.py `
  tests\test_replay_dataset.py `
  tests\test_train_v2.py `
  tests\test_metrics.py `
  tests\test_locked_test_evaluation.py

if ($LASTEXITCODE -ne 0) {
    throw "Regression suite failed. Do not generate the production dataset."
}
```

The locked-test fixture and its v3 contract must pass before final evaluation.

## 4. Supply valid, de-identified source documents

Create the resume directory:

```powershell
New-Item -ItemType Directory -Force "data\resumes" | Out-Null
```

Place at least 32 independently sourced, extractable resume PDFs in
`data/resumes` and at least 32 valid technical JD PDFs in `data/jds`.

Use opaque resume names:

```text
resume_0001.pdf
resume_0002.pdf
...
resume_0032.pdf
```

Check the effective pools through the same filters used by the simulator:

```powershell
& $PY -c "from modules.module_07_rl.data_loader import RESUMES_DIR,JDS_DIR,get_all_pdfs,is_valid_resume,is_valid_jd; r=[p for p in get_all_pdfs(RESUMES_DIR) if is_valid_resume(p)]; j=[p for p in get_all_pdfs(JDS_DIR) if is_valid_jd(p)]; print('valid_resumes=',len(r)); print('valid_jds=',len(j))"
```

Do not start production generation unless both values are at least 32.

Check resume filenames:

```powershell
$BadNames = Get-ChildItem "data\resumes" -Recurse -File -Filter "*.pdf" |
    Where-Object { $_.BaseName -notmatch '^resume_[0-9]{4}$' }

$BadNames

if ($BadNames) {
    throw "Resume filenames are not fully de-identified."
}
```

## 5. Prepare Ollama

Start the Ollama application or service, then install the two distinct models:

```powershell
ollama list
ollama pull qwen2.5:7b
ollama pull gemma3:4b
```

Verify the local server:

```powershell
Invoke-RestMethod "http://localhost:11434/api/tags" |
    ConvertTo-Json -Depth 5
```

Configure the run:

```powershell
$env:OLLAMA_HOST = "http://localhost:11434"
$env:ARIA_CANDIDATE_MODEL = "qwen2.5:7b"
$env:ARIA_EVALUATOR_MODEL = "gemma3:4b"
$env:ARIA_OLLAMA_NUM_CTX = "4096"
$env:ARIA_OLLAMA_TIMEOUT = "300"
$env:ARIA_OLLAMA_KEEP_ALIVE = "0"
```

`ARIA_OLLAMA_KEEP_ALIVE=0` prevents a completed model phase from remaining
resident. It is safer for an 8 GB GPU, though model swapping may reduce
throughput.

## 6. Generate an isolated 24-episode v3 canary

Use a unique directory. Do not use `--append` or `--replace-existing`.

```powershell
$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$CanaryRoot = "data\synthetic\v3\canary-$RunId"
$CanaryRaw = "$CanaryRoot\qwen_rl_dataset.json"
$CanaryReports = "$CanaryRoot\reports"
$CanaryDerived = "$CanaryRoot\derived"

New-Item -ItemType Directory -Force $CanaryReports | Out-Null
ollama list | Out-File -Encoding utf8 "$CanaryRoot\ollama-models.txt"
```

Generate the canary with eight identity components:

```powershell
& $PY -m modules.module_07_rl.llm_simulator `
  --sweep `
  --max_episodes 24 `
  --max_concurrent 2 `
  --candidate-request-concurrency 1 `
  --evaluator-request-concurrency 1 `
  --identity-components 4 2 2 `
  --seed 42 `
  --gpu-vram-gb 8 `
  --dataset-file $CanaryRaw

if ($LASTEXITCODE -ne 0) {
    throw "Canary generation failed."
}
```

If generation fails, start another uniquely named canary. Do not append to the
partial run.

## 7. Verify canary completeness

```powershell
$ManifestFile = Get-ChildItem "$CanaryRoot\manifests\*.json" |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1

$Manifest = Get-Content -Raw $ManifestFile.FullName | ConvertFrom-Json
$FailedEpisodes = @($Manifest.failed_episodes.PSObject.Properties).Count

$Manifest | Select-Object `
    status, `
    planned_episodes, `
    completed_episodes, `
    failed_episodes, `
    candidate_model, `
    evaluator_model

if ($Manifest.completed_episodes -ne $Manifest.planned_episodes) {
    throw "Canary is partial: completed episode count differs from planned count."
}

if ($FailedEpisodes -ne 0) {
    throw "Canary contains failed episodes."
}
```

This check is mandatory because the current simulator can describe a partially
completed run as complete.

## 8. Audit the raw canary

Run the raw-evidence gate:

```powershell
& $PY -m modules.module_07_rl.dataset_audit `
  $CanaryRaw `
  --stage raw `
  --min-episodes 24 `
  --output "$CanaryReports\raw_evidence.json"

if ($LASTEXITCODE -ne 0) {
    throw "Raw audit command failed."
}

$RawAudit = Get-Content -Raw "$CanaryReports\raw_evidence.json" |
    ConvertFrom-Json

$RawAudit | Format-List

if (-not $RawAudit.passes_quality_gates) {
    throw "Raw evidence gate failed."
}
```

Run the offline-RL gate separately. Do not rely on the composite audit:

```powershell
& $PY -m modules.module_07_rl.dataset_audit `
  $CanaryRaw `
  --stage offline_rl `
  --output "$CanaryReports\offline_support.json"

$OfflineAudit = Get-Content -Raw "$CanaryReports\offline_support.json" |
    ConvertFrom-Json

$OfflineAudit | Format-List
```

Weak action support may occur in a 24-episode canary. Mask, propensity,
legality, and stop failures are never acceptable:

```powershell
$HardFailures =
    $OfflineAudit.invalid_action_masks +
    $OfflineAudit.illegal_selected_actions +
    $OfflineAudit.invalid_behavior_propensities +
    $OfflineAudit.inconsistent_selected_propensities +
    $OfflineAudit.invalid_stop_transitions

if ($HardFailures -ne 0) {
    throw "Canary contains mask, propensity, legality, or stop failures."
}
```

## 9. Scan the canary for obvious PII

```powershell
$PiiPattern = '\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|\b(?:\+?91[- ]?)?[6-9][0-9]{9}\b'

$PiiHits = Select-String `
    -Path $CanaryRaw `
    -Pattern $PiiPattern `
    -CaseSensitive:$false

$PiiHits | Select-Object -First 20

if ($PiiHits) {
    throw "Potential email address or phone number found in generated data."
}
```

This is only a basic scan. Production still needs an explicit PII policy and a
stronger entity-based detector.

## 10. Calibrate and replay the canary

The combined calibration/replay function does not currently have a dedicated
CLI, so invoke it directly:

```powershell
& $PY -c "import json; from modules.module_07_rl.replay_dataset import prepare_calibrate_replay; result=prepare_calibrate_replay(r'$CanaryRaw', r'$CanaryDerived', split_seed=42, bootstrap_samples=100); print(json.dumps(result, indent=2, default=str))"

if ($LASTEXITCODE -ne 0) {
    throw "Canary calibration/replay failed."
}
```

Inspect the split manifest:

```powershell
$CanarySplit = Get-Content -Raw "$CanaryDerived\split_manifest_v3.json" |
    ConvertFrom-Json

$CanarySplit.summary | ConvertTo-Json -Depth 5
```

Enforce the requested component contract:

```powershell
$ActualComponents = @(
    [int]$CanarySplit.summary.train.identity_components
    [int]$CanarySplit.summary.validation.identity_components
    [int]$CanarySplit.summary.test.identity_components
)

if (($ActualComponents -join ",") -ne "4,2,2") {
    throw "Split planner did not preserve the requested 4/2/2 component contract."
}
```

The current implementation may fail this check. If it does, fix the split
contract before generating 600 episodes. Do not train on the 24-episode canary.

## 11. Generate the 600-episode production corpus

Only enter this section after the complete canary passes.

```powershell
$ProductionId = Get-Date -Format "yyyyMMdd-HHmmss"
$ProdRoot = "data\synthetic\v3\production-$ProductionId"
$ProdRaw = "$ProdRoot\qwen_rl_dataset.json"
$ProdReports = "$ProdRoot\reports"
$ProdDerived = "$ProdRoot\derived"

New-Item -ItemType Directory -Force $ProdReports | Out-Null
ollama list | Out-File -Encoding utf8 "$ProdRoot\ollama-models.txt"
```

Generate with the increased concurrency limits:

```powershell
& $PY -m modules.module_07_rl.llm_simulator `
  --sweep `
  --max_episodes 600 `
  --max_concurrent 4 `
  --candidate-request-concurrency 3 `
  --evaluator-request-concurrency 2 `
  --identity-components 20 6 6 `
  --seed 42 `
  --gpu-vram-gb 8 `
  --dataset-file $ProdRaw

if ($LASTEXITCODE -ne 0) {
    throw "Production generation failed."
}
```

If Ollama produces timeouts or GPU-memory failures, start a new production run
with these lower limits:

```powershell
--max_concurrent 2 `
--candidate-request-concurrency 1 `
--evaluator-request-concurrency 1
```

Do not append to a failed production run.

## 12. Apply strict production audits

```powershell
& $PY -m modules.module_07_rl.dataset_audit `
  $ProdRaw `
  --stage raw `
  --min-episodes 600 `
  --output "$ProdReports\raw_evidence.json"

& $PY -m modules.module_07_rl.dataset_audit `
  $ProdRaw `
  --stage offline_rl `
  --output "$ProdReports\offline_support.json"

$RawGate = Get-Content -Raw "$ProdReports\raw_evidence.json" |
    ConvertFrom-Json

$OfflineGate = Get-Content -Raw "$ProdReports\offline_support.json" |
    ConvertFrom-Json

if (-not $RawGate.passes_quality_gates) {
    throw "Production raw-evidence gate failed."
}

if (-not $OfflineGate.passes_quality_gates) {
    throw "Production offline-RL gate failed."
}
```

## 13. Freeze, calibrate, and replay production data

```powershell
& $PY -c "import json; from modules.module_07_rl.replay_dataset import prepare_calibrate_replay; result=prepare_calibrate_replay(r'$ProdRaw', r'$ProdDerived', split_seed=42, bootstrap_samples=1000); print(json.dumps(result, indent=2, default=str))"

if ($LASTEXITCODE -ne 0) {
    throw "Production calibration/replay failed."
}
```

Verify the production split:

```powershell
$Split = Get-Content -Raw "$ProdDerived\split_manifest_v3.json" |
    ConvertFrom-Json

$ActualEpisodes = @(
    [int]$Split.summary.train.episodes
    [int]$Split.summary.validation.episodes
    [int]$Split.summary.test.episodes
)

$ActualComponents = @(
    [int]$Split.summary.train.identity_components
    [int]$Split.summary.validation.identity_components
    [int]$Split.summary.test.identity_components
)

if (($ActualEpisodes -join ",") -ne "420,90,90") {
    throw "Expected 420/90/90 episodes; got $($ActualEpisodes -join '/')."
}

if (($ActualComponents -join ",") -ne "20,6,6") {
    throw "Expected 20/6/6 identity components; got $($ActualComponents -join '/')."
}
```

Do not open or report locked-test metrics during calibration, selection, or
training.

## 14. Train IQL

Do not enter this section until policy training applies action masks to policy
logits and that behavior has a passing regression test.

```powershell
$Checkpoint = "$ProdRoot\aria_iql_belief_v3.pth"

& $PY -m modules.module_07_rl.train `
  --train-file "$ProdDerived\splits\train.json" `
  --validation-file "$ProdDerived\splits\validation.json" `
  --belief-config "$ProdDerived\belief_model_v2.json" `
  --output $Checkpoint `
  --epochs 100 `
  --batch-size 256 `
  --seed 42 `
  --early-stopping-patience 10 `
  --early-stopping-min-delta 0.0001

if ($LASTEXITCODE -ne 0) {
    throw "IQL training failed."
}

Get-FileHash -Algorithm SHA256 $Checkpoint
```

Never use `--allow-belief-gate-failure` for a production checkpoint.

## 15. Mandatory stopping point

Do not use `evaluate_locked_test.py` as evidence that the IQL policy works. Its
current CLI accepts no policy checkpoint and evaluates the stored belief verdict,
not actions selected by the learned policy.

The following still require implementation before a complete release command
exists:

1. mask-enforced policy inference;
2. weighted importance sampling;
3. doubly robust evaluation;
4. fitted-Q evaluation;
5. fresh policy-controlled rollouts;
6. baseline comparisons;
7. component-bootstrap confidence intervals;
8. one-time locked-test enforcement;
9. a release manifest binding dataset, split, configuration, checkpoint, model,
   and evaluation-report hashes.

The currently valid stopping point is:

```text
v3 corpus generated
  -> raw and offline gates passed
  -> split frozen
  -> beliefs calibrated
  -> transitions replayed
  -> IQL checkpoint trained
```

It is not yet:

```text
learned policy independently evaluated and release validated
```
