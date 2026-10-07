# ARIA Source Inventory

## Scope

The research-relevant tree contains 469 files after excluding `.git`, dependency trees, skill packages, caches, and temporary render/finalizer folders. It includes 108 Python source files, 84 JSON artifacts, 33 repository PDFs, 21 project Markdown files, five PyTorch checkpoints, four CSV files, two DOCX reports, and one XLSX audit workbook. Git tracks 276 files; only the attached skill directory and `skills-lock.json` were untracked at audit start.

## Primary evidence families

| Family | Main locations | Authority and use |
|---|---|---|
| v7 locked release | `data/synthetic/v3/production-grounding-v10/derived-calibration-v7/release/` | Highest authority for stored-belief locked metrics and release status |
| v7 protocol and manifests | `derived-calibration-v7/protocol/`, `manifests/` | Hashes, schema versions, one-attempt state, split identity |
| v7 calibration | `derived-calibration-v7/calibration/` | Training-only selection, frozen belief configuration, development metrics |
| v7 policy diagnostics | `derived-calibration-v7/audits/offline_policy_evaluation_v1.json` | Diagnostic learned-policy estimate only |
| Raw production corpus | `production-grounding-v10/qwen_rl_dataset.json` | 600-episode synthetic source evidence |
| Data audits | `production-grounding-v10/raw_evidence.json`, `generation_distribution.json`, reports | Complete integrity and distribution checks |
| Current code | `modules/`, `app.py`, `frontend/src/` | Authoritative implementation contracts and live/offline distinction |
| Tests | `tests/` | Contract evidence; focused relevant suite passed |
| Current documentation | `README.md`, `architecture.md`, current runbooks | Interpretation and operator intent, subordinate to code/artifacts |
| Historical reports | Project I DOCX files and older Markdown reports | Background and conflict discovery only |
| Resume inputs | `data/external/opensporks/Resume/` | Cleaned input provenance and cleaning audit |
| Job descriptions | `data/jds/`, `data/excluded_jds/` | Role-grounding inputs and unreadable-file evidence |

## Document inspection notes

- `ARIA_BCSE497J_Project_I_Report.docx`: creator Raghav Sejpal, last modified by Parthasarathy G, modified 2026-08-24, 18 pages, 2,761 words, template placeholders and highlighted instructions remain visible.
- `ARIA_BCSE497J_Project_I_Report_Humanized.docx`: creator and last modifier Raghav Sejpal, modified 2026-09-09, 22 pages, 4,658 words, no tracked changes/comments, readable layout, but stale research values.
- Human-authored Markdown was treated as design/history unless corroborated by current code or immutable artifacts. `ARIA_updated.md` is substantially aspirational; `ARIA_PROJECT_CONTEXT_AND_RESULTS.md` concerns an older benchmark framing and is not authoritative for the current v7 agent.

## External research capability

- Undermind: connected; dedicated ARIA workspace and deep novelty/evidence search created.
- Scite: connected; used for DOI/title resolution, abstracts/full text, and citation-context inspection.
- Zotero: helper installed but local API unavailable at `127.0.0.1:23119`; no library mutation occurred.

