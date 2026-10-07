# ARIA Claim Conflict Register

| Conflict | Competing sources | Authoritative resolution | Paper action |
|---|---|---|---|
| Latest score | Project I reports and `ARIA_TRAINING_AND_DEBUGGING_REPORT.md`: 53 episodes, about 96% | v7 locked artifact: 91 episodes, accuracy 0.9121, macro-F1 0.9168 | Report v7; label 53-episode result historical |
| State size | Humanized report: 32 features; current code and v7 checkpoint: 33 | `state_builder.py`, `rl_spec.py`, checkpoint metadata: `aria-state-v4`, 33 | Use 33 and enumerate contract |
| Belief schema | Older docs: belief-v2 | v7 belief model and checkpoint: belief-v3 | Use belief-v3 for current results; explain lineage |
| Learned-policy effectiveness | Informal language around IQL training implies performance | Locked artifact says `evaluates_learned_policy: false`; WIS report is diagnostic and requires fresh rollouts | Make no causal policy-effectiveness claim |
| Production readiness | Aspirational architecture and UI wording imply integrated adaptive policy | `app.py` uses fixed 0.9 evidence and a three-action cycle | Call system a partial research prototype |
| Full-suite health | Older reports count test functions or imply broad readiness | Focused suite 341/341; unscoped collection fails for temporary/vendor tests, stale root test, and optional dependencies | Report focused pass and full-suite limitation separately |
| Dataset belief quality | Older derived belief dataset fails gates | v7 frozen configuration and locked report pass | Do not blend the two; retain failed audit as calibration history |
| Project framing | `ARIA_PROJECT_CONTEXT_AND_RESULTS.md` describes an older benchmark/problem | Current repository name, code, architecture, v7 artifacts | Exclude incompatible benchmark narrative from central claims |
| Zotero authority | Requested as bibliography authority | Local Zotero API unavailable; no profile detected | Reconcile using repository references, Scite, Undermind, and verified identifiers; disclose limitation |

