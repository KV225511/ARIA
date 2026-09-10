# ARIA calibration protocol v4 operator runbook

This workflow keeps calibration selection training-only, consumes validation
once, and leaves the locked test unread until the explicit release command.
The raw `qwen_rl_dataset.json` file is opened read-only and its byte hash is
checked before and after split freezing.

The release attempt file is an application-level one-attempt guard. It makes an
accidental rerun fail closed, including after an interrupted attempt, but it is
not cryptographic enforcement against someone manually deleting or editing
local artifacts.

## Production commands

Run from the repository root in PowerShell:

```powershell
Set-Location "C:\Users\kriss\github\ARIA"

$PY = (Resolve-Path ".venv\Scripts\python.exe").Path
$RAW = (Resolve-Path "data\synthetic\v3\production-grounding-v10\qwen_rl_dataset.json").Path
$BASE = (Resolve-Path "data\synthetic\v3\production-grounding-v10\derived\split_manifest_v3.json").Path
$OUT = Join-Path (Get-Location) "data\synthetic\v3\production-grounding-v10\derived-calibration-v4"

& $PY -m modules.module_07_rl.prepare_belief_pipeline $OUT `
  --raw-file $RAW `
  --base-split-manifest $BASE `
  --dependency-lock (Resolve-Path "requirements.txt").Path

& $PY -m modules.module_07_rl.prepare_belief_pipeline $OUT `
  --development-train "$OUT\raw-splits\train.json" `
  --development-validation "$OUT\raw-splits\validation.json" `
  --split-manifest "$OUT\manifests\split_manifest_v4.json" `
  --protocol "$OUT\protocol\calibration_protocol_v4.json" `
  --bootstrap-samples 1000
```

Continue only if `protocol/calibration_protocol_v4.json` has status
`PROVISIONAL_SYNTHETIC`:

```powershell
& $PY -m modules.module_07_rl.train `
  --train-file "$OUT\replayed\train.json" `
  --validation-file "$OUT\replayed\validation.json" `
  --belief-config "$OUT\calibration\belief_model_v2.json" `
  --development-bundle "$OUT\manifests\development_bundle_v1.json" `
  --calibration-protocol "$OUT\protocol\calibration_protocol_v4.json" `
  --output "modules\module_07_rl\aria_iql_belief_v4.pth"
```

The following is the sole command that reads the locked test. Run it only once,
after the configuration and release decision are final:

```powershell
& $PY -m modules.module_07_rl.locked_test_evaluator `
  --locked-test "$OUT\raw-splits\locked\test.json" `
  --belief-config "$OUT\calibration\belief_model_v2.json" `
  --protocol "$OUT\protocol\calibration_protocol_v4.json" `
  --split-manifest "$OUT\manifests\split_manifest_v4.json" `
  --output-dir $OUT
```

If `release/release_attempt_v1.json` already exists, do not delete it to retry.
A failed validation or locked-test attempt requires a new protocol version.
