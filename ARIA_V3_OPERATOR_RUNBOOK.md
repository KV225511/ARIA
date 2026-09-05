# ARIA v3 Operator Runbook

This runbook generates and validates a new v3 synthetic dataset from the
cleaned OpenSporks resume CSV. It preserves the legacy v2 corpus and keeps the
locked test unavailable until model selection is complete.

## Current gate

The cleaned dataset provides 238 unique resumes from the `ENGINEERING` and
`INFORMATION-TECHNOLOGY` categories. The latest production preflight found 26
unique readable and permitted JDs. The `20/6/6` production plan requires 32,
so add at least six independently sourced technical JDs before production.

The three-component canary can run with the current inputs. Do not treat a
passing canary as production validation.

## 1. Open the repository

Run all commands from PowerShell:

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "C:\Users\kriss\github\ARIA"
$PY = (Resolve-Path ".\.venv\Scripts\python.exe").Path

& $PY --version
```

The interpreter must be Python 3.11 with the project dependencies installed.
If `.venv` does not work, repair or recreate it before continuing.

## 2. Verify the cleaned resume source

```powershell
$ResumeCsv = "data\external\opensporks\Resume\Resume.cleaned.csv"
$ResumeAudit = "data\external\opensporks\Resume\Resume.cleaning_audit.json"

Test-Path $ResumeCsv
Test-Path $ResumeAudit
Get-FileHash -Algorithm SHA256 $ResumeCsv
```

The expected cleaned CSV hash is:

```text
6F6782A90E9FFC52A407575A904282D9104C9C625E38C31A58CCD03FB14DD169
```

Stop if the file is missing or its hash differs. If the source was
intentionally regenerated, review the new cleaning audit before accepting the
new hash.

## 3. Run regression tests

```powershell
& $PY -m pytest -q `
  tests\test_resume_csv_loader.py `
  tests\test_generation_preflight.py `
  tests\test_llm_simulator.py `
  tests\test_dataset_split.py `
  tests\test_dataset_audit.py `
  tests\test_belief_calibration.py `
  tests\test_replay_dataset.py `
  tests\test_train_v2.py `
  tests\test_metrics.py `
  tests\test_locked_test_evaluation.py

if ($LASTEXITCODE -ne 0) {
    throw "Regression suite failed. Do not generate data."
}
```

## 4. Add the missing job descriptions

Place at least six new PDF files in:

```text
C:\Users\kriss\github\ARIA\data\jds
```

Each new JD must:

- contain at least 100 extractable characters;
- describe an IT or engineering position;
- have content distinct from every existing JD;
- come from an independent source;
- contain no prompt-like instructions or unnecessary personal data.

Renaming or copying an existing JD does not create a new identity.

## 5. Run the production input preflight

```powershell
& $PY -m modules.module_07_rl.generation_preflight `
  --resume-csv $ResumeCsv `
  --resume-categories INFORMATION-TECHNOLOGY ENGINEERING `
  --identity-components 20 6 6 `
  --output "data\synthetic\v3\reports\input_preflight.json"

$Preflight = Get-Content -Raw `
  "data\synthetic\v3\reports\input_preflight.json" |
  ConvertFrom-Json

$Preflight | ConvertTo-Json -Depth 8

if (-not $Preflight.passes_preflight) {
    throw "Production input preflight failed. Do not contact Ollama."
}

if ($Preflight.shortfall.unique_readable_jds -ne 0) {
    throw "The production JD pool is still too small."
}
```

Required production values:

```text
passes_preflight: true
required_identity_components: 32
resume_source.unique_content_hashes: at least 32
job_descriptions.unique_readable_content_hashes: at least 32
shortfall.unique_readable_jds: 0
```

## 6. Prepare Ollama

```powershell
ollama list
ollama pull qwen2.5:7b
ollama pull gemma3:4b

Invoke-RestMethod "http://localhost:11434/api/tags" |
  ConvertTo-Json -Depth 5
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

The candidate and evaluator models must remain different.

## 7. Run the canary preflight

```powershell
& $PY -m modules.module_07_rl.generation_preflight `
  --resume-csv $ResumeCsv `
  --resume-categories INFORMATION-TECHNOLOGY ENGINEERING `
  --identity-components 1 1 1 `
  --output "data\synthetic\v3\reports\input_preflight_canary.json"

$CanaryPreflight = Get-Content -Raw `
  "data\synthetic\v3\reports\input_preflight_canary.json" |
  ConvertFrom-Json

if (-not $CanaryPreflight.passes_preflight) {
    throw "Canary input preflight failed."
}
```

## 8. Generate a new three-episode canary

Use a timestamped destination so an earlier canary is never overwritten:

```powershell
$CanaryId = Get-Date -Format "yyyyMMdd-HHmmss"
$CanaryRoot = "data\synthetic\v3\canary-$CanaryId"
$CanaryRaw = "$CanaryRoot\qwen_rl_dataset.json"
$CanaryReports = "$CanaryRoot\reports"

New-Item -ItemType Directory -Force $CanaryReports | Out-Null
ollama list | Out-File -Encoding utf8 "$CanaryRoot\ollama-models.txt"

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
  --gpu-vram-gb 8 `
  --dataset-file $CanaryRaw

if ($LASTEXITCODE -ne 0) {
    throw "Canary generation failed."
}
```

Do not append to a partial or failed canary. Start a new timestamped run.

## 9. Verify canary completion

```powershell
$CanaryManifestFile = Get-ChildItem "$CanaryRoot\manifests\*.json" |
  Sort-Object LastWriteTimeUtc -Descending |
  Select-Object -First 1

$CanaryManifest = Get-Content -Raw $CanaryManifestFile.FullName |
  ConvertFrom-Json

$FailedCanaryEpisodes = @(
  $CanaryManifest.failed_episodes.PSObject.Properties
).Count

if ($CanaryManifest.completed_episodes -ne 3) {
    throw "Canary did not complete all three episodes."
}

if ($FailedCanaryEpisodes -ne 0) {
    throw "Canary contains failed episodes."
}

if ($CanaryManifest.resume_source.source_file_hash -ne `
    "6f6782a90e9ffc52a407575a904282d9104c9c625e38c31a58ccd03fb14dd169") {
    throw "Canary used an unexpected resume source."
}
```

## 10. Audit the canary

```powershell
& $PY -m modules.module_07_rl.dataset_audit `
  $CanaryRaw `
  --stage raw `
  --min-episodes 3 `
  --output "$CanaryReports\raw_evidence.json"

$CanaryRawAudit = Get-Content -Raw "$CanaryReports\raw_evidence.json" |
  ConvertFrom-Json

if (-not $CanaryRawAudit.passes_quality_gates) {
    throw "Canary raw-evidence gate failed."
}
```

Run the offline-support audit as a diagnostic:

```powershell
& $PY -m modules.module_07_rl.dataset_audit `
  $CanaryRaw `
  --stage offline_rl `
  --output "$CanaryReports\offline_support.json"

$CanaryOfflineAudit = Get-Content -Raw `
  "$CanaryReports\offline_support.json" |
  ConvertFrom-Json

$HardFailures =
  $CanaryOfflineAudit.invalid_action_masks +
  $CanaryOfflineAudit.illegal_selected_actions +
  $CanaryOfflineAudit.invalid_behavior_propensities +
  $CanaryOfflineAudit.inconsistent_selected_propensities +
  $CanaryOfflineAudit.invalid_stop_transitions

if ($HardFailures -ne 0) {
    throw "Canary contains mask, propensity, legality, or stop failures."
}
```

Weak action support can occur with only three episodes. Structural failures
are never acceptable.

## 11. Scan the canary for contact data

```powershell
$PiiPattern = '\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)'

$PiiHits = Select-String `
  -Path $CanaryRaw `
  -Pattern $PiiPattern `
  -CaseSensitive:$false

if ($PiiHits) {
    $PiiHits | Select-Object -First 20
    throw "Potential contact data found in the canary."
}
```

## 12. Generate the 600-episode production corpus

Run this section only after the production preflight and canary gates pass:

```powershell
$ProductionId = Get-Date -Format "yyyyMMdd-HHmmss"
$ProdRoot = "data\synthetic\v3\production-$ProductionId"
$ProdRaw = "$ProdRoot\qwen_rl_dataset.json"
$ProdReports = "$ProdRoot\reports"
$ProdDerived = "$ProdRoot\derived"

New-Item -ItemType Directory -Force $ProdReports | Out-Null
ollama list | Out-File -Encoding utf8 "$ProdRoot\ollama-models.txt"

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
  --gpu-vram-gb 8 `
  --dataset-file $ProdRaw

if ($LASTEXITCODE -ne 0) {
    throw "Production generation failed."
}
```

If the GPU reports memory errors or repeated timeouts, start a new timestamped
production run with:

```powershell
--max_concurrent 2 `
--candidate-request-concurrency 1 `
--evaluator-request-concurrency 1
```

Do not append to a failed production run.

## 13. Verify and audit production

Verify that the production manifest records all 600 episodes, no failures, and
the expected immutable resume source:

```powershell
$ProdManifestFile = Get-ChildItem "$ProdRoot\manifests\*.json" |
  Sort-Object LastWriteTimeUtc -Descending |
  Select-Object -First 1

$ProdManifest = Get-Content -Raw $ProdManifestFile.FullName |
  ConvertFrom-Json

$FailedProductionEpisodes = @(
  $ProdManifest.failed_episodes.PSObject.Properties
).Count

if ($ProdManifest.completed_episodes -ne 600) {
    throw "Production did not complete all 600 episodes."
}

if ($FailedProductionEpisodes -ne 0) {
    throw "Production contains failed episodes."
}

if ($ProdManifest.resume_source.source_file_hash -ne `
    "6f6782a90e9ffc52a407575a904282d9104c9c625e38c31a58ccd03fb14dd169") {
    throw "Production used an unexpected resume source."
}
```

Run both production dataset audits:

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
    throw "Production offline-support gate failed."
}
```

Scan the complete production corpus for contact data before calibration:

```powershell
$ProductionPiiHits = Select-String `
  -Path $ProdRaw `
  -Pattern $PiiPattern `
  -CaseSensitive:$false

if ($ProductionPiiHits) {
    $ProductionPiiHits | Select-Object -First 20
    throw "Potential contact data found in the production corpus."
}
```

Do not relabel invalid actions or reconstruct missing propensities. Regenerate
the affected episodes from their recorded source configuration.

## 14. Freeze, calibrate, and replay

```powershell
& $PY -m modules.module_07_rl.prepare_belief_pipeline `
  $ProdRaw `
  $ProdDerived `
  --split-seed 42 `
  --bootstrap-samples 1000

if ($LASTEXITCODE -ne 0) {
    throw "Production calibration and replay failed."
}
```

Confirm the following artifacts exist:

```powershell
$RequiredDerivedFiles = @(
  "$ProdDerived\belief_model_v2.json",
  "$ProdDerived\split_manifest_v3.json",
  "$ProdDerived\qwen_rl_dataset_belief_v3.json",
  "$ProdDerived\replay_comparison_v3.json",
  "$ProdDerived\calibration_report_v3.json",
  "$ProdDerived\splits\train.json",
  "$ProdDerived\splits\validation.json",
  "$ProdDerived\splits\test.json"
)

$MissingDerivedFiles = $RequiredDerivedFiles |
  Where-Object { -not (Test-Path $_) }

if ($MissingDerivedFiles) {
    $MissingDerivedFiles
    throw "Required derived artifacts are missing."
}
```

Do not inspect locked-test metrics at this stage.

## 15. Train IQL

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

## 16. Mandatory stopping point

Do not unlock the test split or describe the policy as end-to-end validated
until the following are implemented and frozen:

1. mask-enforced learned-policy inference;
2. weighted importance sampling;
3. doubly robust evaluation;
4. fitted-Q evaluation;
5. component-bootstrap confidence intervals;
6. fresh learned-policy rollouts;
7. baseline comparisons;
8. one-time locked-test enforcement;
9. a release manifest binding all source, dataset, split, configuration,
   checkpoint, model, and evaluation-report hashes.

The currently valid stopping point is a trained v3 checkpoint backed by
passing raw-data, offline-support, split, calibration, and replay gates. That
checkpoint is not yet a release-validated interview policy.
