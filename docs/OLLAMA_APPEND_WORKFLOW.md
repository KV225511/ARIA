# Append-only dataset generation on an 8 GB RTX 4060 Laptop GPU

## Hardware conclusion

The quality pair is:

- candidate/interviewer: `qwen2.5:7b` (approximately 4.7 GB model file);
- independent evaluator: `gemma3:4b` (approximately 3.3 GB model file).

Their weight files alone total approximately 8.0 GB. An 8 GB laptop GPU also
needs VRAM for Windows/display use, CUDA and Ollama runtime buffers, attention
workspace, and each request's KV cache. Consequently, both models cannot remain
fully resident in VRAM simultaneously. `keep_alive=-1` prevents time-based
unloading, but it cannot create memory; Ollama still unloads an idle model when
the next model must be loaded.

Do not force `OLLAMA_MAX_LOADED_MODELS=2` for this pair. Ollama's documented
concurrent-load requirement is that the models completely fit in available
VRAM. On this GPU, use one loaded model at a time and two parallel requests per
model phase. Concurrent episodes naturally group Qwen question/answer work and
Gemma evaluation work, amortizing model swaps across two episodes.

The faster pair `qwen2.5:1.5b` + `gemma3:4b` is likely to fit, but the 1.5B
candidate previously produced weaker persona separation. It is a speed-first
alternative, not the recommended benchmark configuration.

## One-time Ollama server configuration on Windows

Run these PowerShell commands, then completely quit and restart Ollama:

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "2", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH", "4096", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "-1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")
```

Why:

- `MAX_LOADED_MODELS=1` prevents futile 7B+4B co-residency attempts;
- `NUM_PARALLEL=2` processes two requests during each model phase;
- 4096 context is Ollama's normal sub-24-GiB setting and is sufficient for the
  simulator prompts;
- Flash Attention and `q8_0` KV cache reduce context memory. Ollama documents
  q8 KV as roughly half the f16 cache with usually negligible quality impact,
  although Qwen-family output should still be spot-checked for regressions;
- `keep_alive=-1` avoids an unnecessary time-based unload between batches.

Verify actual placement while generation runs:

```powershell
ollama ps
nvidia-smi -l 2
```

`ollama ps` should show the active model as `100% GPU`. Alternation between Qwen
and Gemma is expected; repeated unloading between every individual episode is
not.

## Preserve and extend the existing corpus

Set the per-process simulator configuration:

```powershell
$env:ARIA_CANDIDATE_MODEL = "qwen2.5:7b"
$env:ARIA_EVALUATOR_MODEL = "gemma3:4b"
$env:ARIA_OLLAMA_KEEP_ALIVE = "-1"
$env:ARIA_OLLAMA_NUM_CTX = "4096"
```

The simulator normalizes the environment string `"-1"` to numeric JSON `-1`
before calling Ollama. Duration values such as `"5m"` remain strings. This is
important because a unitless JSON string `"-1"` is rejected by some Ollama
versions even though numeric `-1` is valid.

Synthetic generation fails closed: question, candidate, or evaluator API
failures cannot be converted into fallback answers or valid low semantic
scores. After three consecutive failures the episode is discarded while other
concurrent episodes continue, leaving the existing corpus intact.

## Measure before committing to a multi-day run

The slow operation is synthetic episode generation, not IQL training. At an
average of about 21 turns, every episode needs roughly 63 model generations:
one interviewer question, one candidate answer, and one evaluator result per
turn. The earlier 200-episode run taking 36 hours therefore measured about 10.8
minutes per episode at concurrency 1.

Concurrency 2 has an optimistic lower-bound estimate of 5.4 minutes per
episode. Model swapping and unequal response lengths can make it slower:

| New episodes | Optimistic estimate from the 36-hour baseline |
|---:|---:|
| 6 pilot episodes | about 32 minutes |
| 100 | about 9 hours |
| 300 | about 27 hours |
| 500 | about 45 hours |

Run a six-episode append pilot first. These are valid episodes and remain in the
corpus, so the pilot does not waste generated data:

```powershell
python -m modules.module_07_rl.llm_simulator `
  --sweep `
  --append `
  --max_episodes 6 `
  --max_concurrent 2 `
  --seed 43 `
  --gpu-vram-gb 8
```

Each successful checkpoint now prints the observed episodes/hour and ETA. Let
all six finish before trusting the estimate: model warm-up and the first swap
make the first one or two samples pessimistic. If two-way concurrency causes
host RAM pressure, GPU fallback, or request timeouts, use concurrency 1; a
higher number is not recommended on this 8 GB GPU.

If possible, add new de-identified resumes and JDs before appending. More turns
over the same identities increase sample count but not the number of independent
identity components. New documents give a more credible validation/test result.

If the goal is 500 episodes **in total** and the existing corpus has 200, append
300, not 500. Prefer three separately checkpointed batches of about 100 with
different seeds so a machine interruption has a small recovery scope. Account
for the six pilot episodes when choosing the final batch size.

Example 100-episode batch:

```powershell
python -m modules.module_07_rl.llm_simulator `
  --sweep `
  --append `
  --max_episodes 100 `
  --max_concurrent 2 `
  --seed 43 `
  --gpu-vram-gb 8
```

Repeat with seeds 44 and 45, then calculate the last batch from the actual
number of unique episode IDs. Failed episodes are deliberately not added, so do
not assume requested count equals successful count:

```powershell
python -c "import json; from pathlib import Path; d=json.loads(Path('data/synthetic/qwen_rl_dataset.json').read_text()); print(len({x['episode_id'] for x in d}))"
```

Important behavior:

- `--max_episodes N` means N additional episodes, not a final total of N;
- the prior JSON is copied once to `data/synthetic/backups/` by content hash;
- every completed episode is atomically checkpointed into the combined file;
- an unexpected failure in one episode is isolated instead of cancelling the
  remaining concurrent tasks;
- IDs continue after the highest existing ID, e.g. `episode_200` onward;
- existing model provenance must match Qwen 7B + Gemma 4B;
- unused documents form new disconnected identity components when possible;
- otherwise generation stays within the existing train/validation/test identity
  pools and prints a warning about limited independent diversity;
- plain `--sweep` now refuses to erase an existing dataset. Replacement requires
  the explicit `--replace-existing` flag.

Do not use seed 42 for the append. Seeds 43 onward make added selection and
policy randomness distinct while retaining reproducibility.

## Rebuild calibration and train after append

Use a new derived directory so the 200-episode artifacts remain recoverable:

```powershell
python -m modules.module_07_rl.dataset_audit `
  data/synthetic/qwen_rl_dataset.json `
  --stage raw `
  --output data/synthetic/raw_audit_500ep.json

python -m modules.module_07_rl.prepare_belief_pipeline `
  data/synthetic/qwen_rl_dataset.json `
  data/synthetic/derived_500ep
```

Do not bypass calibration on the first attempt. With roughly 75 validation
episodes in a 500-episode corpus, the collapse gate is much more informative
than it was with nine.

Then train to a new checkpoint path:

```powershell
python -m modules.module_07_rl.train `
  --train-file data/synthetic/derived_500ep/splits/train.json `
  --validation-file data/synthetic/derived_500ep/splits/validation.json `
  --belief-config data/synthetic/derived_500ep/belief_model_v2.json `
  --output modules/module_07_rl/aria_iql_belief_v2_500ep.pth `
  --epochs 100 `
  --batch-size 256 `
  --seed 43 `
  --early-stopping-patience 10
```

One hundred epochs is only a ceiling. The trainer atomically retains the best
validation epoch and stops when the objective has not improved for ten epochs.
The earlier run peaked at epoch 5, so this avoids spending the remaining 85+
epochs memorizing training actions after the useful checkpoint is saved.

After configuration and checkpoint selection are frozen, unlock the stored
belief test report once:

```powershell
python -m modules.module_07_rl.evaluate_locked_test `
  data/synthetic/derived_500ep/splits/test.json `
  data/synthetic/derived_500ep/belief_model_v2.json `
  data/synthetic/derived_500ep/locked_test_report_v2.json `
  --confirm-config-frozen
```

This evaluates stored belief verdicts. Fixed offline trajectories still cannot
honestly evaluate counterfactual IQL actions without fresh rollouts or logged
behavior propensities.

The report deliberately names fixed-trajectory fields
`logged_action_entropy`, `logged_avg_cumulative_reward`, and
`logged_avg_episode_length`. They describe the behavior data and must not be
interpreted as actions or returns produced by the trained IQL policy.

## If calibration still fails

Do not delete the combined raw corpus. Keep its audit and calibration failure
report, inspect Low-class validation episodes, identity-component counts, and
abstention thresholds, and only use `--allow-belief-gate-failure` for an
explicitly experimental policy-training run. A bypassed run must not be reported
as a calibrated benchmark result.
