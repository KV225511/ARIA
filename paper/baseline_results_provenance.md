# Baseline results provenance audit

The repository contains two historical baseline-result reports added together in
commit `23b6b84cbe6b5a94f28727e9ab87adc6a43788dd`. These results may be described in
the manuscript as previously recorded project experiments. They are not currently
rerunnable because the referenced `combined_models_report.csv`, training scripts,
model directory, checkpoints, and raw benchmark datasets are absent from the
current checkout and from the reachable Git object history.

## Detailed baseline table

Source: `ARIA_updated.md`, Section 10.

| Dataset | Model | Accuracy | Macro F1 | Evidence status |
|---|---|---:|---:|---|
| FER2013 | CNN video-emotion baseline | 0.5548 | 0.48 | Historical report with per-class breakdown |
| RAVDESS | Deep-learning audio model | 0.5590 | 0.52 | Historical report with per-class breakdown |
| Mohler | TF-IDF + logistic regression | 0.6681 | 0.5167 | Historical report only |
| Mohler | TF-IDF + linear SVM | 0.7495 | 0.6030 | Historical report only |
| Mohler | SBERT cosine threshold | 0.6791 | 0.4629 | Historical report only |
| Mohler | SBERT feature classifier | 0.7165 | 0.5410 | Historical report only |
| Mohler | ARIA hybrid semantic-rubric | 0.7516 | 0.5886 | Historical report only |
| CMU-MOSEI | Text-only | 0.8015 | Not reported | Historical report only |
| CMU-MOSEI | Audio-only | 0.6098 | Not reported | Historical report only |
| CMU-MOSEI | Vision-only | 0.5808 | Not reported | Historical report only |
| CMU-MOSEI | Early-concatenation fusion | 0.7976 | 0.77 | Historical report only |
| Box of Lies | Cross-modal incongruence model | 0.5965 | 0.55 | Historical report only |

The same file records a 300-episode synthetic comparison: random policy skill
accuracy 0.4933, fixed-script 0.6393, rule-based adaptive 0.6320, greedy entropy
0.4767, and ARIA reward-heuristic 0.6887. This predates the current v7 IQL
protocol and should be labeled as a historical simulation.

## Fast 25-model matrix

`ARIA_PROJECT_CONTEXT_AND_RESULTS.md` reports 25 successful `--fast` executions.
The strongest reported non-baseline entries were Mohler gradient-boosted trees
(accuracy 0.7833, macro F1 0.5201) and the Box of Lies multimodal transformer
(accuracy 0.6800, macro F1 0.5614). The report states that detailed outputs were
stored in `combined_models_report.csv`; that CSV is not present.

The FER2013 baseline differs slightly between the reports: accuracy 0.5548 and
macro F1 0.48 in the detailed table, versus accuracy 0.5552 and macro F1 0.4600
in the 25-model fast-run table. They should be treated as separate recorded runs,
not merged or averaged.

## Excluded result set

`ARIA_Benchmark_Diagnosis.md` records earlier values of 0.9946 for FER2013,
0.9917 for RAVDESS, and 1.0000 for fusion, then explicitly diagnoses those values
as suspicious or invalid because of leakage and simplified prediction logic.
These figures must not appear as valid baseline results.

## Manuscript wording

Use wording such as: “Historical project logs report …” or “In an earlier fast-run
benchmark recorded in the repository …”. Do not describe these numbers as newly
reproduced, independently verified, or directly comparable with published SOTA
results until the raw predictions, split manifests, and executable training
artifacts are restored.
