# Complete journal redraft — 7 October 2026

## Active deliverables

- `paper/ARIA_revised_journal_draft.tex`: editable, self-contained manuscript; authoritative prose and tables.
- `output/pdf/ARIA_revised_journal_draft.pdf`: reviewed PDF, mirrored to `paper/ARIA_revised_journal_draft.pdf`.
- `paper/ARIA_revision_checks.json`: artifact hashes and numerical/structural checks.
- `paper/citation_verification_redraft.md`: primary-source verification of the active bibliography.

Other manuscript variants and old audit files are historical unless explicitly identified here. The prior active TeX was preserved in `output/journal_redraft/previous_draft.tex` before editing.

## Evidence decisions

The redraft uses the canonical comparison summary pulled at repository revision `b373fb1`, including low-confidence behavior-cloning return **8.2989255124168**, displayed as **8.30**. The training report hash is `1c988a6935e5833aca60805f9a6fdf5277685e46e88786efcf84ad213b4c03c0`.

| Condition | Canonical report hash | Source-file SHA-256 |
|---|---|---|
| Base | e6d1a126a02cf151ff3958afef6e8c1184766ec64d7885ef2edd5513a0532594 | 81a9602afd1903bc7c9663737423f85de2d62439ea00864a36c0e5ef8cc39f53 |
| Overlap | 09ccb855120ac35a0f9bcad42234f678c1a864f63e494571c59a58398bbe68ed | 4770b6d733ed5ab7c959178f814dd3597b9f597900d6c53c4d41fe401e355a3a |
| Low confidence | e712784aaed2437fa67a00abdf1d994e84af84936e5df2293e539c2cb2fe9626 | c9eb0d8a2c9ce9b87847a9a16daf9e6e7c5de519f7a75cb2945e6341ad350708 |
| Positive shift | 510b815517ee07483880481c4cf299c9a05e134a59792e964d5a888569e069aa | b4a80a780f228edfcf7c4f1652b957ff240c033a932704e10f71001f3d328dbc |

Local detailed rollout files have different hashes. They were not mixed with this summary to calculate new paired confidence intervals. No experiment was run during the manuscript rewrite, and no dataset, checkpoint, or training implementation was changed.

The locked v7 report evaluates beliefs, not the learned policy. The direct rollout study is development-only and uses a constructed score generator after training on stored ARIA transitions. Historical component baselines remain summary-level evidence.

## Scientific corrections

- Included all eight policies in accuracy, cost/coverage, and reward-interval tables.
- Separated classification from designed return and information gain. IQL's overlap accuracy is lower than several baselines.
- Labeled the sequence comparator **DT-inspired**. Its summed embeddings, last-position training loss, and inference return-history handling differ from the published Decision Transformer.
- Described compact discrete CQL as an adaptation, not an author-supplied system or exact reproduction.
- Distinguished replay terminal outcome shaping from rollout environment reward, which omits that correctness bonus.
- Described stress conditions as exploratory, selected after the base ceiling.
- Retained the slightly worse v7 expected calibration error rather than asserting uniform calibration improvement.
- Kept external interview systems in the literature discussion; none is falsely presented as trained on ARIA.

## Verification and remaining author work

The checker verifies every displayed entry in the direct policy tables against the canonical JSON, all seven locked belief metrics against their report, citation keys, labels, and PDF text boundaries. All pages are visually inspected after rendering. The candidate prose audit is recorded under `output/journal_redraft`; repetition diagnostics are reviewed in context rather than treated as automatic failures.

The built-in TeX compiler returned `Unable to find standard directories for platform`. The PDF is produced by the existing ReportLab fallback, with vector diagrams and equivalent equation typesetting. Native TeX output is unverified; the editable source remains available in the built-in editor.

The ScienceDirect reference is inaccessible, so an exact format/length match is not claimed. The new paper is a research draft, not a certified submission-ready manuscript. Authors still need to finalize target-journal formatting, funding, competing interests, author contributions, contact details, and a permanent release link. Human validity/fairness, multi-seed comparator robustness, and faithful original-method reproduction have not been supplied by this rewrite.

## Rebuild in the activated virtual environment

```powershell
python -m pip install reportlab pdfplumber
python paper/render_revision_pdf.py
if ($LASTEXITCODE -ne 0) { throw 'PDF rendering failed' }
python paper/check_revision.py
if ($LASTEXITCODE -ne 0) { throw 'Manuscript checks failed' }
Copy-Item output/pdf/ARIA_revised_journal_draft.pdf paper/ARIA_revised_journal_draft.pdf -Force
```

The source is the active editing surface. The temporary one-off builder under `output/journal_redraft` is an intermediate artifact and must not be used to overwrite subsequent source edits.
