# Documentation index

This index organizes ARIA's documentation by purpose. Existing documents retain their original contents and filenames. Documents required at their original paths by links or scripts remain there; independent root documents are grouped below. Commands in runbooks continue to run from the repository root.

Historical reports and plans describe the repository at the time they were written. Their placement here does not establish current implementation status or validate their results. Start with the repository guidelines and approved specification for current application constraints.

## Project entry points and requirements

| Document | Purpose |
| --- | --- |
| [Project README](../README.md) | Application overview, setup, commands, and module status |
| [Repository guidelines](../AGENTS.md) | Current goals, architecture boundaries, and development safeguards |
| [Authentication and interview persistence specification](specs/001-auth-and-interview-persistence.md) | Approved authentication, ownership, and persistence contracts |
| [Frontend guide](../frontend/README.md) | Frontend setup and development |
| [Application launcher](../run.ps1) | Root PowerShell launcher; retained at its documented executable path |

## Architecture and module contracts

| Document or collection | Purpose |
| --- | --- |
| [System architecture](../architecture.md) | Live, offline, and intended integration paths |
| [Architecture diagrams and source files](architecture/README.md) | Complete architecture PNG, SVG, Excalidraw, and build sources |
| [System overview SVG](aria-system-architecture.svg) / [PNG](aria-system-architecture.png) | Existing overview diagram assets |
| [Coding assistant guide](../ARIA_Coding_Assistant_Guide.md) | Historical module specifications and interface contracts |
| [Belief-v2 and replay-v2 design](../ARIA_RL_IMPROVEMENT_CHANGES.md) | Replay, calibration, state, reward, and training design changes |
| [Belief-v2 schema examples](examples/belief_v2/) | Illustrative JSON contracts; not measured benchmark results |

## Operational runbooks

### Dataset generation

These documents share a directory so the legacy runbook's link to its replacement remains valid.

| Document | Purpose |
| --- | --- |
| [Pipeline v3 runbook](runbooks/dataset-generation/ARIA_PIPELINE_V3_RUNBOOK.md) | Generation inputs, audits, replay, training, and release gates |
| [v3 operator runbook](runbooks/dataset-generation/ARIA_V3_OPERATOR_RUNBOOK.md) | PowerShell production and canary workflow |
| [v3 commands for Krissh](runbooks/dataset-generation/ARIA_V3_KRISSH_COMMANDS.md) | Operator-specific command sequence |
| [Legacy dataset generation and training runbook](runbooks/dataset-generation/DATASET_GENERATION_AND_TRAINING_RUNBOOK.md) | Explicitly superseded v2 workflow, preserved for reference |
| [Ollama append workflow](OLLAMA_APPEND_WORKFLOW.md) | Existing GPU residency and append-only generation guidance |

### Calibration protocols

| Document | Purpose |
| --- | --- |
| [Calibration v4](runbooks/calibration/ARIA_CALIBRATION_V4_RUNBOOK.md) | Training-only selection, validation, and locked release workflow |
| [Calibration v5](runbooks/calibration/ARIA_CALIBRATION_V5_RUNBOOK.md) | Additive repeat-discount calibration workflow |
| [Calibration v6](runbooks/calibration/ARIA_CALIBRATION_V6_RUNBOOK.md) | Effective-evidence search, authenticated bundles, and protocol state |

### Benchmark experiments

| Document | Purpose |
| --- | --- |
| [SOTA experiment runbook](runbooks/benchmarks/SOTA_TRAINING_RUNBOOK.md) | Evidence preflight, text baselines, policy comparators, and comparison reports |

## Development history and benchmark evidence

| Document | Purpose |
| --- | --- |
| [Development notes](../notes.md) | Historical perception-module implementation details and change log |
| [Complete master document](../ARIA_updated.md) | Historical project plan, POMDP design, baseline results, and work division |
| [Project context and results](../ARIA_PROJECT_CONTEXT_AND_RESULTS.md) | Historical 25-model rapid-run matrix and execution context |
| [Benchmark diagnosis](../ARIA_Benchmark_Diagnosis.md) | Historical benchmark validity and leakage investigation |
| [Training and debugging report](reports/training/ARIA_TRAINING_AND_DEBUGGING_REPORT.md) | Historical dataset, calibration, and training diagnostic report |
| [Training output log](reports/training/train_output.txt) | Existing empty training log, preserved without adding output |

## Council decisions

Each dated directory contains the original HTML report and matching Markdown transcript. The report-to-transcript relative links remain intact.

| Decision | Report | Transcript |
| --- | --- | --- |
| 2026-08-30: pipeline v3 generation authorization | [Report](decisions/council/2026-08-30/council-report-20260830-201134Z-q9ec87cb9.html) | [Transcript](decisions/council/2026-08-30/council-transcript-20260830-201134Z-q9ec87cb9.md) |
| 2026-09-10: calibration repair plan | [Report](decisions/council/2026-09-10/council-report-20260910-114704Z-q0a6f95eb.html) | [Transcript](decisions/council/2026-09-10/council-transcript-20260910-114704Z-q0a6f95eb.md) |

## Academic reports and research workspace

| Document or collection | Purpose |
| --- | --- |
| [Project I report](reports/academic/project-i/ARIA_BCSE497J_Project_I_Report.docx) | Original BCSE497J report |
| [Project I humanized report](reports/academic/project-i/ARIA_BCSE497J_Project_I_Report_Humanized.docx) | Existing alternate report version, preserved separately |
| [Report logo](reports/academic/project-i/assets/vit-logo.jpg) | Associated report image, preserved unchanged |
| [Manuscripts and revision evidence](../paper/) | Existing research papers, LaTeX, bibliography, figures, review records, and generated comparisons |
| [Research dossiers and evidence](../.research/) | Existing literature matrix, source inventory, evidence ledger, and topic dossier |
| [Paper memory](../.paper/) | Existing manuscript state and reusable writing records |

The two Project I reports and logo remain ignored by Git under the existing filename patterns. They are organized locally and are not added to version control.

## Study and interview preparation

| Document or collection | Purpose |
| --- | --- |
| [Software engineering study guide](interview-prep/ARIA-software-engineering-study-guide.md) | Repository-oriented engineering study material |
| [Key points](interview-prep/ARIA-key-points.md) | Concise project preparation notes |
| [Interview preparation diagrams](interview-prep/) | System map, live turn, belief/policy, offline pipeline, and diagram sources |

## Original-path compatibility and preservation

The following documents remain in the root intentionally:

| Retained file | Reason |
| --- | --- |
| `README.md` | Repository entry point |
| `AGENTS.md` | Repository instruction discovery |
| `architecture.md` | Linked by the unchanged root README and repository guidelines; its image path also depends on its location |
| `ARIA_Coding_Assistant_Guide.md` | Linked by the unchanged README and development notes |
| `ARIA_RL_IMPROVEMENT_CHANGES.md` | Linked by the unchanged README |
| `ARIA_Benchmark_Diagnosis.md` | Linked by the README and read by SOTA preflight at its root path |
| `ARIA_PROJECT_CONTEXT_AND_RESULTS.md` | Read by SOTA preflight and the paper revision script at its root path |
| `ARIA_updated.md` | Read by SOTA preflight and the paper revision script; recorded in research provenance |
| `notes.md` | Its unchanged relative links require the root coding assistant guide beside it |

[Relocation manifest](organization/root-document-moves.json) records original and organized paths, SHA-256 hashes, byte sizes, categories, and Git tracking status for the moved files and retained root documents. Existing document text, scripts, datasets, model artifacts, research records, and configuration were not edited. Historical filename mentions in prose remain verbatim; use this index and manifest to locate moved files.

[Training log relocation record](organization/root-log-move.json) records the subsequent move of the empty root training log. `.codex-finalizer-review2/`, `.codex-finalizer-review2-expanded/`, and `.aria-private/` are covered by the existing Git ignore rules and remain untouched.
