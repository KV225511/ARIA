# Topic Decision Dossier — Auditable Adaptive Interview Research

## 1. Executive Decision Summary

| Field | Value |
|---|---|
| Area | Auditable adaptive mock interviewing with explicit skill beliefs and offline question-policy learning |
| Compiled | 2026-09-15 |
| Verdict grade | Screening-grade — assembles evidence; does not decide worth |
| Search confidence | Medium — 249 records from one deep-search service, checked with Scite; local Zotero was unavailable |

One candidate topic was evaluated retrospectively against the repository’s implemented evidence.

| Candidate 1 | Auditable belief-state and offline-policy architecture for adaptive mock interviews |
|---|---|
| **Verdict** | **Worth pursuing — only if its open conditions hold** |
| **Reason** | Prior work covers multimodal scoring, answer-aware follow-ups, adaptive testing, and offline reinforcement learning separately. ARIA’s auditable integration remains differentiable, but real-person validity, subgroup safety, and fresh learned-policy rollouts are missing. |

**Key uncertainty.** PolyInterview is a close 2026 integration precedent, and the search cannot prove that no still-closer system exists.

## 2. Candidate Definition

**Candidate 1 — Auditable belief-state and offline-policy architecture for adaptive mock interviews.** The study would characterize a reproducible system that maps question-linked evidence to explicit multidimensional skill beliefs and selects subsequent questions with a conservatively learned offline policy. *Why it could be a gap:* Current methods fall short — close systems combine adaptive follow-ups and multimodal feedback, but the retrieved literature did not expose ARIA’s combination of a versioned state/action contract, identity-isolated synthetic trajectories, locked belief calibration, and diagnostic off-policy evaluation.

This is partially attempted, not a wholly unoccupied area. HireNet and Naim et al. establish multimodal interview scoring; PolyInterview adds answer-aware follow-ups; deep computerized adaptive testing connects Bayesian latent states with learned item selection; and implicit Q-learning supplies the offline algorithm. No verified source in the screened set establishes real-world hiring validity for the integrated ARIA design.

## 3. Decision Scorecard

| Gate | Score | Rationale |
|---|---|---|
| Gate 1 — Gap still open | 3/5 neutral | The precise audit-oriented integration appears only partially occupied, but PolyInterview substantially narrows the systems gap and search recall is incomplete. |
| Gate 2 — Real contribution | 3/5 neutral | The contribution is an incremental systems and evaluation artifact, not a new interview or RL primitive. Its value depends on auditable contracts and negative evidence being reported faithfully. |
| Gate 3 — Feasible | 4/5 agree | The current repository supports synthetic-corpus, calibration, and diagnostic-OPE work. Human-subject validation is feasible only with ethics approval, consent, accessibility planning, and an appropriate sample. |
| **Verdict** | **Worth pursuing — only if its open conditions hold** | Proceed as a research-prototype and reproducibility paper; do not frame it as a validated hiring product. |

## 4. Evidence Base

**Search funnel.**

| Stage | Count |
|---|---|
| Retrieved and ranked by Undermind deep search | 249 |
| Inspected in the leading relevance tranche | 50 |
| Selected into the comparison matrix | 21 |
| Identifier-verified through Scite or arXiv metadata | 21 |

**Prior-art classification.**

| By evidence type | By role in the decision |
|---|---|
| 10 empirical interview or applicant studies; 4 reviews/conceptual analyses; 5 algorithm or methods papers; 2 recent system/adaptive-testing preprints | 8 define closest technical precedents; 8 constrain validity, fairness, or acceptance; 5 constrain calibration and policy evaluation |

**Closest prior work.**

- **Automated interview scoring:** HireNet (primary study) and Naim et al. (primary study) show that sequential, multimodal interview data can predict human ratings. Their fixed-question designs and proxy labels do not establish adaptive skill diagnosis.
- **Integrated practice systems:** PolyInterview (recent preprint and close analogue) combines role-conditioned questions, answer-aware follow-ups, and multimodal feedback. It weakens a broad novelty claim; it does not report criterion validity or an explicit offline-RL belief-state contract.
- **Adaptive policy learning:** Deep computerized adaptive testing (methods/empirical preprint) combines multidimensional latent states and learned selection, while implicit Q-learning (algorithm paper) supplies a conservative offline method. Neither validates an interview policy.
- **Audit and harm:** Rhea et al., Booth et al., Tippins et al., and the multidisciplinary survey (audit studies/reviews) show why predictive performance alone is insufficient.

## 5. Gate-by-Gate Assessment

### Gate 1 — Gap still open

- **Score:** 3/5 neutral.
- **Evidence:** The screened set contains every major component and one close integrated system, but no verified paper with the same explicit 33-dimensional state contract, identity-isolated synthetic generation, frozen belief calibration, IQL checkpoint provenance, and diagnostic WIS evaluation.
- **Interpretation:** The broad topic is occupied; a narrow audit-and-reproducibility question leans open.
- **Risk:** Search recall is bounded by a single deep-search workspace, Scite coverage, and unavailable Zotero metadata.
- **Action needed:** Repeat the exact integration query in additional bibliographic indexes and inspect forward citations to PolyInterview and deep CAT before making a priority claim.

### Gate 2 — Real contribution

- **Score:** 3/5 neutral.
- **Evidence:** Each technical primitive is established. ARIA’s defensible contribution is the versioned integration and evidence trail, including results that prevent stronger claims: the locked test does not evaluate the learned policy and the live application does not execute the full pipeline.
- **Interpretation:** This is incremental systems research. “Incremental” is descriptive, not a quality judgment.
- **Risk:** A paper that hides the missing rollout or live-integration evidence would reduce the contribution to unsupported architecture prose.
- **Action needed:** Treat auditability as an evaluated property: publish schemas, hashes, split rules, failure histories, and explicit release gates; compare against simpler question-selection baselines.

### Gate 3 — Feasible

- **Score:** 4/5 agree for a prototype paper; 2/5 disagree for a hiring-validity claim.
- **Evidence:** The repository contains a 600-episode synthetic corpus, a locked 91-episode belief evaluation, a versioned IQL checkpoint, 341 passing focused tests, and a diagnostic validation-split WIS estimate. It contains no fresh learned-policy rollouts or real-applicant study.
- **Interpretation:** A reproducibility and prototype-characterization paper is feasible now. A criterion-validity or fairness paper is a separate human-subject study.
- **Risk:** Consent, data retention, disability access, ASR disparities, label validity, and adequate subgroup sample sizes bind the later study.
- **Action needed:** Freeze the current synthetic release, then preregister a human pilot with job-analysis-grounded outcomes, human and heuristic baselines, accessibility accommodations, and explicit stopping criteria.

## 6. Risks and Upgrade / Kill Tests

**Named risks.**

- **Construct-validity risk:** Synthetic low/medium/high skill labels and fixed semantic/behavioral values in the live demo are not evidence that ARIA measures occupational competence.
- **Dataset-constraint risk:** Identity isolation protects the synthetic split but does not create population representativeness.
- **Novelty risk:** PolyInterview and deep CAT may be extended by work outside the current search corpus.
- **Reproducibility risk:** Candidate and evaluator LLMs can change; prompts, versions, raw outputs, hashes, and parsers must remain frozen.
- **Accessibility and fairness risk:** Visual, prosodic, and ASR-derived evidence can burden disabled, neurodivergent, accented, or non-native-speaking users unequally.

**Upgrade / kill test — auditable adaptive mock interviewing.** Worth pursuing once all of these hold:

1. a forward/backward search around PolyInterview, HireNet, and deep CAT finds no prior system with the same auditable belief-state and offline-policy contract, or the claim is narrowed to a reproducibility contribution;
2. fresh rollouts compare the frozen IQL policy with fixed-cycle, random-valid-action, and uncertainty-greedy baselines on held-out identities, with uncertainty intervals and failure analysis;
3. a human-study protocol identifies job-analysis-grounded outcomes, consent and retention rules, accessibility accommodations, and sample targets adequate for planned subgroup analyses;
4. the live application loads the frozen belief and policy artifacts or is explicitly excluded from the evaluated system.

The topic should be reframed or stopped if the learned policy fails to improve a predeclared rollout outcome without worsening coverage, repetition, or subgroup burden, or if the proposed human outcomes cannot support the intended construct claim.

## 7. Recommended Next Steps

Do not pursue “an AI hiring system that accurately and fairly evaluates candidates” as the current topic. The repository has no real-applicant criterion validity, no subgroup fairness evaluation, and no evidence that the current live application executes the evaluated offline policy.

The narrower audit-oriented prototype is conditionally promising. Freeze the v7 artifact set, run fresh held-out policy rollouts against simple baselines, complete a broader priority search, and design a separate ethics-reviewed human pilot. These steps would turn the paper’s present contribution from a careful repository characterization into a test of adaptive policy behavior and user-facing validity.

## Appendix A. Search and Screening Protocol

| Field | Value |
|---|---|
| Search date | 2026-09-15 |
| Sources | Undermind deep search; Scite metadata, abstracts, full text, and bibliography formatting; repository references |
| Query families | Integrated adaptive interview architecture; multimodal interview validation; belief-state adaptive testing; offline RL and OPE; fairness, disability, ASR, and applicant reaction; failure and limitation searches |
| Number retrieved | 249 ranked records |
| Deduplication | DOI/arXiv identity, then title and author-year comparison |
| Inclusion | Direct technical precedent, validity/fairness constraint, or evaluation method needed to bound an ARIA claim |
| Exclusion | Generic interview coaching, unrelated multimodal models, commercial pages, and papers without a claim-level role |
| Screening | Automated ranking followed by manual title/abstract/full-text judgment; the literature matrix records retained sources |
| Known limitations | Zotero local API unavailable; one retrieved Raghavan PDF was incomplete; no claim of exhaustive recall |
| Recall confidence | Medium; adequate for scoped wording, insufficient for an absolute first-system claim |

## Appendix B. Deliverable File List

| File | Purpose |
|---|---|
| `topic_dossier.md` / `.docx` | Human-readable decision memo |
| `topic_dossier.bib` | Verified screening bibliography |
| `literature_matrix.md` | Cross-paper evidence comparison |
| `gaps.yml` | Machine-readable decision record |

