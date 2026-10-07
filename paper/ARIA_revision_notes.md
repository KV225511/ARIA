# ARIA manuscript revision — 2 October 2026

Status: review draft. The scientific rewrite and baseline consolidation are implemented. Exact matching to the requested ScienceDirect reference remains pending access to its PDF. No new experiments were run.

## Deliverables

- `paper/ARIA_revised_journal_draft.tex`: authoritative editable standalone LaTeX manuscript, with inline vector diagrams and bibliography.
- `output/pdf/ARIA_revised_journal_draft.pdf`: 14-page independently rendered review PDF; five vector figures, eight tables, six equations.
- `paper/ARIA_revision_source_hashes.json`: SHA-256 hashes for the eleven inspected local authority sources.
- Original manuscript files are preserved.

The native LaTeX editor was opened. Its compiler failed before checking the source with `Unable to find standard directories for platform`. The PDF was therefore rendered with ReportLab from the same LaTeX prose, captions, tables, and bibliography, with equivalent vector figures and separately typeset equations. It is not evidence that the LaTeX source compiled successfully. The native editor remains the place for subsequent source edits.

## Results recovered

| Repository source | Recovered material | Treatment |
|---|---|---|
| `ARIA_updated.md`, Section 10 | FER2013 and RAVDESS results, five Mohler text baselines, four MOSEI modality conditions, Box of Lies result | Reported results retained, with task boundaries and reproducibility limitations |
| `ARIA_updated.md`, Section 10, Model 6 | Five historical policies in a reported 300-episode simulation | Restored as historical policy baselines; reward heuristic is not IQL |
| `ARIA_PROJECT_CONTEXT_AND_RESULTS.md`, Section 3 | 25 model rows across five domains, accuracy, macro-F1, elapsed times | Complete appendix; unequal training budgets preclude a controlled SOTA ranking |
| `data/synthetic/v3/production-grounding-v10/derived-calibration-v7/release/locked_test_evaluation_v1.json` | Current and preceding belief metrics, confusion matrix, component bootstrap | Primary numerical source for current paired belief comparison |
| `data/synthetic/v3/production-grounding-v10/derived-calibration-v7/audits/offline_policy_evaluation_v1.json` | Weighted reward diagnostic and false release gate | Retained as transition-level diagnostic, not discounted return improvement |

The named `combined_models_report.csv`, `run_all_models_report` driver, and `models/.../train_real.py` implementations were not found in the inspected checkout, including the focused search of ignored files. The presence of a summary is not treated as independent reproduction. No outside folder was searched for missing experiment outputs.

Conflicts are preserved explicitly: FER2013 has two different run summaries; the text pair 0.6681/0.5167 has inconsistent model labels; Box of Lies baseline F1 differs across summaries. The old diagnosis document contains suspicious near-perfect benchmark results and requests leakage checks; those values were not promoted to current evidence. The comparison script also contains fallback constants and unsupported SOTA targets, which were not copied into the paper.

## Scientific and editorial changes

- Rebuilt the abstract, introduction, research questions, method, results, discussion, and conclusion around baseline evidence and the current offline release.
- Added a dedicated state-of-the-art analysis with a qualitative comparison table. Contemporary preprints are labeled; metrics from incompatible tasks are not ranked together.
- Corrected the state groups to match the 33 features in `state_builder.py`; the action mask is separate.
- Explained the semantic-only competency likelihood. Behavioral inputs are not described as direct evidence of skill in the current update.
- Distinguished the historical reward heuristic, the current IQL checkpoint, and the live application's fixed evidence/three-action cycle.
- Corrected interpretation of the importance-weighted transition-reward diagnostic by inspecting its implementation.
- Retained the adverse ECE change alongside improved classification and Brier metrics.
- Reduced the original broad 70-source inventory to nine sources directly used and checked for this focused draft. Original bibliography and historical papers are preserved.

## Citation verification

The following identifiers and abstracts were read from primary sources during this revision. The manuscript uses abstract-supported method summaries, not external numerical leaderboard claims. The Kim PDF abstract and introductory material were inspected; this was not an end-to-end review of that paper. The requested ScienceDirect style reference was inaccessible, so its content and length were not inferred.

| Key | Verified identifier / primary reading source | Claim supported |
|---|---|---|
| hirenet | [10.1609/aaai.v33i01.3301573](https://ojs.aaai.org/index.php/AAAI/article/view/3832) | Hierarchical interview representation and recruiter-judgment target |
| poly | [arXiv:2607.10310](https://arxiv.org/abs/2607.10310) | Role-conditioned interaction and multimodal feedback |
| cat | [arXiv:2502.19275](https://arxiv.org/abs/2502.19275), record links published DOI 10.1017/psy.2026.10106 | Multivariate trait estimation and learned item selection |
| kim | [10.1109/ACCESS.2023.3325891](https://yoonjongyeon.github.io/assets/pdf/fairness.pdf) | Distributional regularization and fairness/accuracy trade-off |
| mag | [10.1109/ACCESS.2024.3473314](https://scholar.ui.ac.id/en/publications/mag-bert-arl-for-fair-automated-video-interview-assessment/) | Multimodal representation and adversarial reweighting |
| avi | [arXiv:2608.25316](https://arxiv.org/abs/2608.25316) | Trait-targeted dataset and strong text baseline |
| iql | [arXiv:2110.06169](https://arxiv.org/abs/2110.06169) | Expectile learning and advantage-weighted policy extraction; implementation details separately checked against ARIA code |
| ope | [10.1609/aaai.v29i1.9541](https://ojs.aaai.org/index.php/AAAI/article/view/9541) | Confidence-aware policy evaluation |
| calibration | [PMLR 70, Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) | Distinction between predictive correctness and calibrated confidence |

## Verification and limits

Completed four substantive review passes: argument/structure; evidence/scope; prose/terminology; delivery consistency. Source-derived generation checked all 25 rapid-run rows. The locked confusion matrix sums to 91, with 83 current and 72 preceding correct decisions. Checked metric differences against unrounded JSON values. Checked all nine citation keys and figure/table references. Rendered and visually inspected all 14 PDF pages in contact sheets, then inspected the architecture, equations, and dense appendix at higher resolution. Repaired missing math glyphs and orphaned appendix headings/caption. Final PDF text has no stray LaTeX backslashes, unsupported font characters, or horizontally out-of-page text.

The prose audit on the exact extracted reading text flags repeated source labels, URL forms, and four metric-reporting constructions. These are retained as necessary source attribution and parallel reporting of distinct metrics. Raw-LaTeX audit findings concerning repeated environments and hyphenated architecture names are markup/table artifacts, not prose duplication. No claim of AI-authorship detection is made.

Functional coverage: the rewrite, baseline consolidation, SOTA analysis, and architecture figures are delivered. Matching the requested paper's exact format, explanation pattern, and length is still blocked by reference access. The native TeX compilation remains unverified. Historical result provenance is report-level; fresh current-IQL rollouts and human validation remain scientific gaps. This is not a submission-ready certification.
