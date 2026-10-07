# ARIA Evidence Ledger

| ID | Claim | Evidence | Scope | Status |
|---|---|---|---|---|
| E1 | Current state contract is 33-dimensional `aria-state-v4` | `rl_spec.py`, `state_builder.py`, v7 checkpoint metadata | Offline policy | Verified |
| E2 | Action space contains eight actions | `RL_ACTION_SPACE`, environment action space | Offline and intended live policy | Verified |
| E3 | Production corpus has 600 episodes and 9,018 transitions | Complete raw audit | Synthetic only | Verified |
| E4 | Identity components do not cross train/validation/test | Complete raw audit and split manifest | Production synthetic corpus | Verified |
| E5 | Generation distribution gates pass | `generation_distribution.json`, complete raw audit | Production synthetic corpus | Verified |
| E6 | v7 locked stored-belief metrics are 0.9121 accuracy and 0.9168 macro-F1 | `locked_test_evaluation_v1.json` | 91 locked synthetic episodes | Verified |
| E7 | v7 locked evaluation is not learned-policy evaluation | `evaluates_learned_policy: false` | Interpretation boundary | Verified |
| E8 | Validation WIS evaluates learned policy diagnostically | `offline_policy_evaluation_v1.json` | 1,405 validation transitions | Verified, diagnostic |
| E9 | Fresh learned-policy rollouts support causal improvement | No current artifact | Learned policy | Gap |
| E10 | Live app uses full multimodal and IQL pipeline | `app.py` contradicts claim | Live application | Rejected |
| E11 | Frontend compiles | Vite production build | Current checkout | Verified |
| E12 | Module 7 contracts pass tests | 341 focused tests | Current checkout | Verified |
| E13 | Full repository test suite is clean | Collection errors and missing optional dependencies | Current checkout | Gap |
| E14 | Resume cleaning preserved 2,475 usable rows with passed assertions | CSV-wide inspection and XLSX audit | External resume input | Verified |
| E15 | ARIA is valid or fair for real hiring | No human-subject, subgroup, or deployment evidence | Real-world use | Unsupported |

