# ARIA calibration protocol v6 operator runbook

Protocol v6 is a narrow successor to the failed v5 training-only search. It
reproduces the v5 best-failing anchor and varies only the existing
`minimum_effective_evidence` setting: `2.00`, `1.75`, then `1.50`. All other
behavior and fitted-emission settings are frozen. The search evaluates no more
than three candidates and cannot read validation unless training-only grouped
cross-validation first passes every gate.

The immutable protocol JSON and mutable protocol-state JSON are separate. Never
edit either file. Use a new `derived-calibration-v6` directory; never delete or
reuse the v4 or v5 directories. The commands below read the raw corpus but do
not rewrite it.

## 1. Define production paths and record the raw byte hash

```powershell
Set-Location "C:\Users\kriss\github\ARIA"

$PY = (Resolve-Path ".venv\Scripts\python.exe").Path
$ROOT = (Resolve-Path "data\synthetic\v3\production-grounding-v10").Path
$RAW = Join-Path $ROOT "qwen_rl_dataset.json"
$BASE = Join-Path $ROOT "derived\split_manifest_v3.json"
$V5 = Join-Path $ROOT "derived-calibration-v5"
$OUT = Join-Path $ROOT "derived-calibration-v6"
$RAW_BEFORE = (Get-FileHash -LiteralPath $RAW -Algorithm SHA256).Hash.ToLowerInvariant()

"raw_file_sha256_before=$RAW_BEFORE"
if (Test-Path -LiteralPath $OUT) {
  throw "Refusing to reuse an existing v6 output directory: $OUT"
}
```

## 2. Freeze v6 and reproduce the exact v5 split assignments

This step creates the v6 protocol, state, manifest, authenticated split
inventory, and raw split files. It validates that v5 is `FAILED`, that v5 never
used validation, and that all supplied v5 artifacts match their frozen hashes.

```powershell
& $PY -m modules.module_07_rl.prepare_belief_pipeline_v6 $OUT `
  --raw-file $RAW `
  --base-split-manifest $BASE `
  --parent-v5-protocol "$V5\protocol\calibration_protocol_v5.json" `
  --parent-v5-report "$V5\calibration\training_cv_report_v2.json" `
  --parent-v5-split-manifest "$V5\manifests\split_manifest_v4.json" `
  --parent-v5-inventory "$V5\manifests\raw_split_inventory_v2.json" `
  --dependency-lock (Resolve-Path "requirements.txt").Path

$RAW_AFTER_FREEZE = (Get-FileHash -LiteralPath $RAW -Algorithm SHA256).Hash.ToLowerInvariant()
if ($RAW_AFTER_FREEZE -ne $RAW_BEFORE) {
  throw "Raw dataset bytes changed; stop immediately."
}

$STATE = Get-Content "$OUT\protocol\calibration_protocol_state_v1.json" -Raw | ConvertFrom-Json
$STATE | Select-Object current_status, validation_executions, revision, protocol_hash
```

Expected state: `FROZEN_FOR_DEVELOPMENT`, zero validation executions. Review
the generated protocol and manifest before proceeding. Confirm that the v6
manifest assignments equal v5 exactly:

```powershell
$M5 = Get-Content "$V5\manifests\split_manifest_v4.json" -Raw | ConvertFrom-Json
$M6 = Get-Content "$OUT\manifests\split_manifest_v4.json" -Raw | ConvertFrom-Json
$A5 = $M5.assignments | ConvertTo-Json -Compress
$A6 = $M6.assignments | ConvertTo-Json -Compress
if ($A5 -cne $A6) { throw "v6 split assignments differ from v5" }
```

## 3. Run training-only selection and the sole validation attempt

The next command may evaluate validation exactly once. Do not rerun it,
regardless of success, failure, or interruption. A durable validation-attempt
file is created before validation bytes are read.

```powershell
& $PY -m modules.module_07_rl.prepare_belief_pipeline_v6 $OUT `
  --development-train "$OUT\raw-splits\train.json" `
  --development-validation "$OUT\raw-splits\validation.json" `
  --split-manifest "$OUT\manifests\split_manifest_v4.json" `
  --inventory "$OUT\manifests\raw_split_inventory_v3.json" `
  --protocol "$OUT\protocol\calibration_protocol_v6.json" `
  --protocol-state "$OUT\protocol\calibration_protocol_state_v1.json" `
  --parent-v5-report "$V5\calibration\training_cv_report_v2.json" `
  --bootstrap-samples 1000

$STATE = Get-Content "$OUT\protocol\calibration_protocol_state_v1.json" -Raw | ConvertFrom-Json
$STATE | Select-Object current_status, validation_executions, revision, belief_config_hash

$CV = Get-Content "$OUT\calibration\training_cv_report_v3.json" -Raw | ConvertFrom-Json
$CV | Select-Object selection_status, candidate_count, candidate_limit, validation_used_for_selection

$RAW_AFTER_VALIDATION = (Get-FileHash -LiteralPath $RAW -Algorithm SHA256).Hash.ToLowerInvariant()
if ($RAW_AFTER_VALIDATION -ne $RAW_BEFORE) {
  throw "Raw dataset bytes changed; stop immediately."
}
if (Test-Path "$OUT\replayed\test.json") {
  throw "Development unexpectedly produced replayed test data."
}
```

Continue only if state is `PROVISIONAL_SYNTHETIC`, validation executions is
exactly `1`, selection status is `ELIGIBLE`, and candidate count is at most `3`.
If state is `FAILED`, v6 is permanently closed; create a new protocol version
for any further calibration change.

## 4. Train from the authenticated development bundle

```powershell
& $PY -m modules.module_07_rl.train `
  --train-file "$OUT\replayed\train.json" `
  --validation-file "$OUT\replayed\validation.json" `
  --belief-config "$OUT\calibration\belief_model_v2.json" `
  --development-bundle "$OUT\manifests\development_bundle_v3.json" `
  --calibration-protocol "$OUT\protocol\calibration_protocol_v6.json" `
  --calibration-protocol-state "$OUT\protocol\calibration_protocol_state_v1.json" `
  --output "modules\module_07_rl\aria_iql_belief_v6.pth"
```

Training validates all artifact hashes, schema compatibility, split names,
belief-config hashes, raw/offline-RL/calibration gates, and rejects test rows.
It has no locked-test input.

## 5. Optional final one-attempt locked-test release evaluation

This is the only production command that reads the locked test. Run it only
after the configuration and release decision are final. It creates the attempt
claim before reading test bytes and refuses every second attempt.

```powershell
& $PY -m modules.module_07_rl.locked_test_evaluator `
  --locked-test "$OUT\raw-splits\locked\test.json" `
  --belief-config "$OUT\calibration\belief_model_v2.json" `
  --protocol "$OUT\protocol\calibration_protocol_v6.json" `
  --protocol-state "$OUT\protocol\calibration_protocol_state_v1.json" `
  --split-manifest "$OUT\manifests\split_manifest_v4.json" `
  --raw-split-inventory "$OUT\manifests\raw_split_inventory_v3.json" `
  --output-dir $OUT
```

`release/release_attempt_v1.json` is an application-level one-attempt guard,
not absolute cryptographic enforcement. Never delete or edit it to retry. A
failed or interrupted attempt permanently sets v6 to `FAILED`.
