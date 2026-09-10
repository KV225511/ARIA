# ARIA additive calibration protocol v5 operator runbook

Protocol v5 is a narrow follow-up to the failed v4 training-CV search. It keeps
the v4 best failing candidate fixed and tests only lower values of the existing
`repeat_discount_power` field. It evaluates at most four candidates and does
not read validation unless one passes every training-CV gate.

Use a new `derived-calibration-v5` directory. Do not delete, overwrite, or
reuse the v4 protocol directory. The raw corpus is read-only and checked byte
for byte before and after split freezing.

Run these commands from PowerShell:

```powershell
Set-Location "C:\Users\Raghav Sejpal\Documents\ChatGPT\ARIA"

$PY = (Resolve-Path ".venv\Scripts\python.exe").Path
$ROOT = (Resolve-Path "data\synthetic\v3\production-grounding-v10").Path
$RAW = Join-Path $ROOT "qwen_rl_dataset.json"
$BASE = Join-Path $ROOT "derived\split_manifest_v3.json"
$OUT = Join-Path $ROOT "derived-calibration-v5"
$V4REPORT = (Resolve-Path "C:\Users\Raghav Sejpal\Downloads\aria-v4-candidate-summary.json").Path

& $PY -m modules.module_07_rl.prepare_belief_pipeline_v5 $OUT `
  --raw-file $RAW `
  --base-split-manifest $BASE `
  --dependency-lock (Resolve-Path "requirements.txt").Path `
  --parent-v4-report $V4REPORT

& $PY -m modules.module_07_rl.prepare_belief_pipeline_v5 $OUT `
  --development-train "$OUT\raw-splits\train.json" `
  --development-validation "$OUT\raw-splits\validation.json" `
  --split-manifest "$OUT\manifests\split_manifest_v4.json" `
  --protocol "$OUT\protocol\calibration_protocol_v5.json" `
  --bootstrap-samples 1000
```

The second command is the one permitted validation execution. Do not rerun it.
Review `calibration/training_cv_report_v2.json` and
`calibration/development_calibration_report_v5.json`. Continue only when the
protocol status is `PROVISIONAL_SYNTHETIC`:

```powershell
$P = Get-Content "$OUT\protocol\calibration_protocol_v5.json" -Raw | ConvertFrom-Json
$P | Select-Object protocol_schema_version, protocol_status, validation_executions, belief_config_hash

& $PY -m modules.module_07_rl.train `
  --train-file "$OUT\replayed\train.json" `
  --validation-file "$OUT\replayed\validation.json" `
  --belief-config "$OUT\calibration\belief_model_v2.json" `
  --development-bundle "$OUT\manifests\development_bundle_v2.json" `
  --calibration-protocol "$OUT\protocol\calibration_protocol_v5.json" `
  --output "modules\module_07_rl\aria_iql_belief_v5.pth"
```

The following command is the sole action that reads the locked test. Run it
only after the configuration and release decision are final:

```powershell
& $PY -m modules.module_07_rl.locked_test_evaluator `
  --locked-test "$OUT\raw-splits\locked\test.json" `
  --belief-config "$OUT\calibration\belief_model_v2.json" `
  --protocol "$OUT\protocol\calibration_protocol_v5.json" `
  --split-manifest "$OUT\manifests\split_manifest_v4.json" `
  --output-dir $OUT
```

`release/release_attempt_v1.json` is an application-level one-attempt guard,
not cryptographic enforcement. Never delete it to retry. A failed or
interrupted attempt requires a new protocol version.
