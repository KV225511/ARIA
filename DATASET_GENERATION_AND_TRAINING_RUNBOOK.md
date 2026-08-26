# ARIA Dataset Generation and Training Runbook

This runbook restores the clean 200-episode corpus, verifies the Ollama fix,
appends 300 valid episodes using `qwen2.5:7b` and `gemma3:4b`, calibrates the
belief model, trains IQL, and evaluates the locked test split.

Run every command from PowerShell on the laptop containing the dataset.

## 1. Update the repository

```powershell
Set-Location "C:\Users\kriss\github\ARIA"

git status
git pull --ff-only origin main
git log -1 --oneline
```

The latest commit should be:

```text
bc20520 Fail closed on Ollama generation errors
```

Activate the virtual environment if applicable:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify the simulator fix:

```powershell
python -m pytest -q tests/test_llm_simulator.py
```

Expected result: `177 passed`.

## 2. Quarantine the corrupted dataset

```powershell
$rawDataset = "data\synthetic\qwen_rl_dataset.json"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Copy-Item `
  $rawDataset `
  "data\synthetic\qwen_rl_dataset.corrupted_$timestamp.json"
```

This preserves the corrupted dataset for diagnosis.

## 3. Restore the clean 200-episode dataset

```powershell
$cleanBackup = "data\synthetic\backups\qwen_rl_dataset.pre_append.93d5bb947ec5.json"

if (-not (Test-Path $cleanBackup)) {
    throw "Clean backup was not found: $cleanBackup"
}

Copy-Item $cleanBackup $rawDataset -Force
```

Verify the restored count:

```powershell
python -c "import json; from pathlib import Path; d=json.loads(Path(r'data/synthetic/qwen_rl_dataset.json').read_text(encoding='utf-8')); print('Transitions:',len(d)); print('Episodes:',len({x['episode_id'] for x in d}))"
```

It should report approximately 200 episodes. Do not continue if it still
reports roughly 500 episodes.

## 4. Configure Ollama

Verify the installed models:

```powershell
ollama list
```

Required models:

```text
qwen2.5:7b
gemma3:4b
```

Install them if needed:

```powershell
ollama pull qwen2.5:7b
ollama pull gemma3:4b
```

Configure the Ollama server for an 8 GB RTX 4060 Laptop GPU:

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "2", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH", "4096", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "5m", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")
```

Completely exit Ollama from the Windows system tray and restart it so the
server settings take effect.

Set ARIA's current-terminal configuration:

```powershell
$env:ARIA_CANDIDATE_MODEL = "qwen2.5:7b"
$env:ARIA_EVALUATOR_MODEL = "gemma3:4b"
$env:ARIA_OLLAMA_KEEP_ALIVE = "5m"
$env:ARIA_OLLAMA_NUM_CTX = "4096"
$env:ARIA_OLLAMA_TIMEOUT = "300"
```

`"5m"` is recommended. Commit `bc20520` also safely converts environment value
`"-1"` to numeric JSON `-1` before sending it to Ollama.

## 5. Generate a six-episode pilot

```powershell
python -m modules.module_07_rl.llm_simulator `
  --sweep `
  --append `
  --max_episodes 6 `
  --max_concurrent 2 `
  --seed 43 `
  --dataset-file $rawDataset `
  --gpu-vram-gb 8
```

The pilot must have:

- no `400 Bad Request` responses;
- no repeated candidate or evaluator failures;
- six successfully checkpointed episodes;
- semantic scores that differ across competency classes.

Monitor Ollama in another terminal:

```powershell
ollama ps
```

If available, monitor the GPU:

```powershell
nvidia-smi -l 2
```

## 6. Inspect the pilot episodes

```powershell
python -c "import json; from pathlib import Path; d=json.loads(Path(r'data/synthetic/qwen_rl_dataset.json').read_text(encoding='utf-8')); g={}; [g.setdefault(x['episode_id'],[]).append(x) for x in d]; ids=sorted(g,key=lambda x:int(x.replace('episode_','').replace('episode-','')))[-6:]; [print(e,'class=',r[0].get('true_label'),'turns=',len(r),'mean=',round(sum(x['semantic_score'] for x in r)/len(r),3),'range=',(min(x['semantic_score'] for x in r),max(x['semantic_score'] for x in r)),'valid=',all(x.get('evaluation_valid') for x in r)) for e in ids for r in [g[e]]]"
```

Expected pattern:

- Beginner scores are generally lower.
- Mid scores are generally intermediate.
- Expert scores are generally higher.
- Every episode reports `valid=True`.
- Scores are not all exactly `0.1`.

If all three classes still score approximately `0.1`, stop. Do not generate
the remaining episodes.

## 7. Generate the remaining 294 episodes

The following batches produce 300 new episodes in total:

```text
6 pilot + 94 + 100 + 100 = 300 new episodes
200 existing + 300 new = 500 total episodes
```

### Batch 1: 94 episodes

```powershell
python -m modules.module_07_rl.llm_simulator `
  --sweep `
  --append `
  --max_episodes 94 `
  --max_concurrent 2 `
  --seed 44 `
  --dataset-file $rawDataset `
  --gpu-vram-gb 8
```

### Batch 2: 100 episodes

```powershell
python -m modules.module_07_rl.llm_simulator `
  --sweep `
  --append `
  --max_episodes 100 `
  --max_concurrent 2 `
  --seed 45 `
  --dataset-file $rawDataset `
  --gpu-vram-gb 8
```

### Batch 3: 100 episodes

```powershell
python -m modules.module_07_rl.llm_simulator `
  --sweep `
  --append `
  --max_episodes 100 `
  --max_concurrent 2 `
  --seed 46 `
  --dataset-file $rawDataset `
  --gpu-vram-gb 8
```

Each successful episode is atomically checkpointed. API failures discard the
affected episode instead of generating artificial `0.1` evidence.

## 8. Verify the final episode count

```powershell
python -c "import json; from pathlib import Path; d=json.loads(Path(r'data/synthetic/qwen_rl_dataset.json').read_text(encoding='utf-8')); print('Transitions:',len(d)); print('Episodes:',len({x['episode_id'] for x in d}))"
```

Target: at least 500 episodes.

If failures leave the corpus below 500, generate the missing number with a new
seed. Each append must request at least three episodes.

```powershell
python -m modules.module_07_rl.llm_simulator `
  --sweep `
  --append `
  --max_episodes 3 `
  --max_concurrent 2 `
  --seed 47 `
  --dataset-file $rawDataset `
  --gpu-vram-gb 8
```

## 9. Audit the combined raw dataset

Use a new run name so corrupted derived artifacts are never reused:

```powershell
$runTag = "500ep_clean_v2"
$derived = "data\synthetic\derived_$runTag"

python -m modules.module_07_rl.dataset_audit `
  $rawDataset `
  --stage raw `
  --min-episodes 500 `
  --output "data\synthetic\raw_audit_$runTag.json"
```

Do not continue unless the raw-evidence gate passes and the class score
distributions remain separated.

## 10. Calibrate and replay the belief pipeline

Do not reuse the old corrupted `derived_500ep` directory.

```powershell
python -m modules.module_07_rl.prepare_belief_pipeline `
  $rawDataset `
  $derived `
  --split-seed 42 `
  --bootstrap-samples 1000
```

This stage performs:

1. Leakage-safe train, validation, and locked-test splitting.
2. Belief-model calibration on training evidence.
3. Hyperparameter selection using validation data.
4. Belief-state replay for every transition.
5. Versioned configuration, manifest, split, and report generation.

Do not bypass calibration on the first attempt.

## 11. Audit offline-RL support

```powershell
python -m modules.module_07_rl.dataset_audit `
  "$derived\splits\train.json" `
  --stage offline_rl `
  --output "$derived\offline_rl_audit.json"
```

Confirm that all eight interview actions have adequate support.

## 12. Train IQL

```powershell
$checkpoint = "modules\module_07_rl\aria_iql_$runTag.pth"

python -m modules.module_07_rl.train `
  --train-file "$derived\splits\train.json" `
  --validation-file "$derived\splits\validation.json" `
  --belief-config "$derived\belief_model_v2.json" `
  --output $checkpoint `
  --epochs 100 `
  --batch-size 256 `
  --seed 42 `
  --early-stopping-patience 10 `
  --early-stopping-min-delta 0.0001
```

The trainer atomically retains the best validation checkpoint and stops after
ten epochs without meaningful validation improvement.

Do not use `--allow-belief-gate-failure` unless calibration genuinely fails and
the resulting run is explicitly labelled experimental and uncalibrated.

## 13. Run the locked-test evaluation

Run this only after calibration, hyperparameters, and checkpoint selection are
frozen:

```powershell
python -m modules.module_07_rl.evaluate_locked_test `
  "$derived\splits\test.json" `
  "$derived\belief_model_v2.json" `
  "$derived\locked_test_report_v2.json" `
  --confirm-config-frozen
```

This evaluates stored belief verdicts on the locked test split. Fixed offline
trajectories cannot honestly measure counterfactual learned-policy returns
without fresh rollouts or logged behavior propensities.

## Failure policy

If the raw audit, belief calibration, or offline-RL support gate fails:

1. Stop the pipeline.
2. Preserve the generated JSON report.
3. Inspect class distributions, identity components, evaluator validity, and
   failed episode logs.
4. Do not immediately bypass the gate or reuse corrupted derived artifacts.
