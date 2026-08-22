# ARIA belief-v2 and replay-v2 design

## Outcome

ARIA now has a replay-first migration path for the existing evaluator evidence.
It does not require or permit LLM question/answer regeneration. The raw JSON is
read-only; calibration, state, reward, labels, and termination are emitted to a
new versioned derived directory.

The redesign fixes the structural reason calibration previously had little
effect: old observations, rewards, and `aria_label` values remained embedded in
the offline transitions after belief parameters changed. Replay-v2 rebuilds
every derived field from the frozen configuration.

## Data and control flow

```text
immutable raw transitions
        |
        v
identity components (resume/JD content hash preferred)
        |
        +--> train ------> fit class emissions + bootstrap stability
        |
        +--> validation -> tune repetition/ESS/aggregation/abstention
        |
        +--> test -------- LOCKED
        |
        v
frozen BeliefModelConfig (hash-addressed)
        |
        v
deterministic episode replay
        |
        +--> versioned train.json ------+
        +--> versioned validation.json -+--> IQL training/model selection
        +--> versioned test.json ---------> explicit post-freeze evaluation only
```

## Design changes

### Belief model

- `BeliefModelConfig` is immutable, validated, deterministically serialized,
  and addressed by a SHA-256 configuration hash.
- Technical competency likelihood uses only `semantic_score`.
- Evaluator, STT, and modality confidence scale evidence reliability; they do
  not move the score toward a competency class.
- Behavior, prosody, anxiety, accent, incongruence, and presentation style are
  policy/context features only.
- Per-skill evidence uses Gaussian log-likelihoods, log-space normalization,
  repeated-question discount, sublinear repeated evidence, and an ESS cap.
- Global belief uses a normalized log-opinion pool over visited skills only.
- Abstention is explicit when coverage, ESS, or confidence is insufficient.
- Non-finite and out-of-range evidence is rejected rather than silently
  poisoning a posterior.
- Legacy shared-sigma behavior is available only through the explicit
  `BeliefModelConfig.legacy(...)` adapter.

### Calibration and split isolation

- Resume/JD identity components are split before fitting. Content hashes take
  priority over filenames, so renamed duplicates stay together.
- Centers and scales are fitted on training labels only using episode-balanced
  robust statistics. Small-class scales shrink toward a pooled scale.
- Centers are projected into monotonic order without imposing artificial
  spacing.
- Validation labels tune only repeat discount, skill ESS cap, aggregation
  temperature, and the abstention threshold.
- Collapsed validation candidates are rejected before ranking by macro-F1,
  ordinal MAE, and expected calibration error.
- Parameter stability is bootstrapped by episode. Test metrics are not computed
  by the prepare or training commands.

### Replay and stable state

- Replay preserves every source field and stores `raw_*` copies for fields it
  replaces.
- It recomputes belief, verdict/abstention, confidence, ESS, information gain,
  observation, next observation, reward, termination, and action mask.
- Every replayed transition carries raw dataset, split manifest, and belief
  config hashes plus belief/state/reward/replay schema versions.
- State-v2 has 32 named, fixed-position features. It contains global belief,
  entropy, coverage, turn, focus belief/ESS, and previous evidence/context/action
  values with availability flags. It never depends on JD-specific padding.
- The action mask is a separate field and is not smuggled into the state.
- `obs` is constructed before the current action/evidence and `next_obs` after
  it, preventing future-transition leakage.

### Reward, termination, and gates

- Information gain is measured on the target skill, not untouched uniform
  ontology nodes.
- Ignorance is not treated as anxiety/distress.
- Step reward includes coverage, duration, redundancy, anxiety, and invalid
  action terms. There is no unconditional conclusion bonus.
- Terminal correctness has explicit correct/adjacent/opposite/abstain costs and
  is applied exactly once; a second shaping attempt is an error.
- Conclusion requires minimum turns, coverage, ESS, and calibrated certainty,
  and produces a structured termination reason.
- Gates are separate: raw evidence, calibration validation, locked test,
  offline-RL support, and learned-policy rollout. Relaxing the validation
  belief gate does not bypass raw or offline-support failures.
- Stored-belief reports explicitly set `evaluates_learned_policy: false`.

### Training

- Training accepts only replay-v2 train and validation splits. It has no test
  input.
- Every transition is checked for split, configuration hash, state schema,
  feature names, dimensions, finite values, actions, and rewards.
- Python, NumPy, Torch, and CUDA randomness is seeded; deterministic algorithms
  and evaluation-mode target inference are used.
- The best validation checkpoint is written atomically to a new v2 filename.
  The prior accepted checkpoint is never overwritten.
- Checkpoints include model, belief config/hash, state schema/names, split hash,
  seed, hyperparameters, and selection metadata.

## Commands

Prepare calibration and immutable replay from the existing raw log:

```powershell
python -m modules.module_07_rl.prepare_belief_pipeline `
  data/synthetic/qwen_rl_dataset.json `
  data/synthetic/derived
```

This produces:

- `belief_model_v2.json`
- `calibration_report_v2.json`
- `split_manifest_v2.json`
- `qwen_rl_dataset_belief_v2.json`
- `replay_comparison_v2.json`
- `splits/train.json`, `splits/validation.json`, and locked `splits/test.json`

Train and select the best checkpoint without loading test:

```powershell
python -m modules.module_07_rl.train `
  --train-file data/synthetic/derived/splits/train.json `
  --validation-file data/synthetic/derived/splits/validation.json `
  --belief-config data/synthetic/derived/belief_model_v2.json
```

Only after calibration and model selection are frozen, unlock the stored-belief
test result once:

```powershell
python -m modules.module_07_rl.evaluate_locked_test `
  data/synthetic/derived/splits/test.json `
  data/synthetic/derived/belief_model_v2.json `
  data/synthetic/derived/locked_test_report_v2.json `
  --confirm-config-frozen
```

The command refuses to overwrite an existing locked-test report.

## Quality interpretation

The raw evidence gate may still fail if the fixed log has genuine semantic
score compression, insufficient identity components, invalid evidence, model
overlap, or leakage. Calibration can improve the mapping of informative scores;
it cannot create class information absent from the evidence. A failed raw gate
must not be bypassed to claim a benchmark result.

Most importantly, fixed offline trajectories cannot honestly evaluate
counterfactual IQL actions without fresh rollouts or logged behavior
propensities. Validation loss is useful for checkpoint selection, but it is not
learned-policy performance. A learned-policy gate accepts only reports from
fresh rollouts tied to a checkpoint hash.

## Artifact examples

Illustrative, non-benchmark schema examples live in
`docs/examples/belief_v2/`. Real reports are intentionally absent because the
raw transition log is not available in this workspace. The examples must never
be quoted as measured ARIA performance.
