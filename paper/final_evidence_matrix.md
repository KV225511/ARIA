# Final Evidence Matrix

| Manuscript claim | Direct repository evidence | Literature role | Alternative explanation / risk | Narrow wording used | Status |
|---|---|---|---|---|---|
| State/action contract is current | `rl_spec.py`, `state_builder.py`, v7 checkpoint | IQL defines algorithmic basis | Older documents describe 32 features | 33-dimensional `aria-state-v4`, eight actions | Confirmed |
| Corpus is identity-isolated | Raw corpus, generation distribution, split manifest, full audit | Synthetic-data and RL design guidance | Shared prompt/model can still create distributional dependence | No linked identity component crosses partitions | Confirmed |
| Belief-v3 performs well on locked synthetic test | `locked_test_evaluation_v1.json` | Calibration literature motivates probability metrics | Synthetic labels may be easier or model-conditioned | Reports metrics on 91 synthetic episodes only | Confirmed |
| Paired v7 belief metrics improve over v6 | `locked_test_evaluation_v1.json` paired comparison and component bootstrap | Calibration literature motivates multi-metric reporting | Only six identity components; synthetic beliefs, not policy behavior | Accuracy +0.1209 and macro-F1 +0.1158 on the same 91 episodes | Confirmed |
| IQL checkpoint is reproducible | Checkpoint metadata and hash | Kostrikov et al. | Objective value may not translate to behavior | Versioned checkpoint with recorded hyperparameters | Confirmed |
| WIS is numerically stable on validation logs | OPE JSON | Thomas; Voloshin; Patterson | Overlap diagnostics do not give causal performance | Diagnostic estimate only; release gate false | Confirmed |
| Learned policy improves interviewing | No fresh rollout artifact | OPE literature requires stronger design | Logged-policy mismatch, estimator bias, reward misspecification | Explicitly not claimed | Gap |
| Live application executes evaluated policy | `app.py` contradicts | Systems literature supplies comparators | UI appearance can imply more integration than backend provides | Live demo uses fixed evidence and three-action cycle | Rejected |
| ARIA is first adaptive multimodal interview system | PolyInterview and other precedents | Closest prior work occupies broad claim | Additional recent systems may exist | Auditable integration is differentiated; no first claim | Rejected |
| ARIA is valid/fair for hiring | No human criterion or subgroup study | Psychometric, fairness, disability, ASR, and reaction literature defines requirements | Synthetic accuracy can be mistaken for validity | Research/mock-interview prototype only | Gap |
| Focused implementation is tested | 341 focused tests; frontend build | Empirical RL reproducibility guidance | Unscoped collection fails | Focused pass, full-suite limitation | Confirmed |

## Functional-completeness retrospective

- **Scope covered:** repository state, offline data and policy lineage, live application, literature, claims, companion artifacts, and release boundaries.
- **Authority and locks:** current code and frozen artifacts govern; stale reports are historical; claim wording is locked to synthetic/prototype scope.
- **Alignment:** each reported result has a method and artifact; unsupported policy and hiring claims are recorded as gaps.
- **Adapters and overlays:** computational/simulation, AI/LLM, evidence-synthesis, ethics, citation, document, PDF, and spreadsheet checks were applied.
- **Change propagation:** corrected state size, belief version, sample size, and policy-evaluation scope were propagated through the abstract, results, discussion, conclusion, ledgers, and dossier.
- **Deterministic and visual checks:** schema, 70/70 citation coverage, test, build, file, and render checks are recorded; Zotero remains unavailable.
- **Open issues:** two S4 evidence gaps (real-person validity/fairness; learned-policy effect), one S3 live-integration gap, and one S2 full-suite collection issue.
- **Readiness:** ready as an internal, venue-neutral research-prototype manuscript; not ready as evidence for real hiring deployment or learned-policy superiority.
