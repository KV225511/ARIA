# ARIA complete project architecture

`ARIA_complete_architecture.excalidraw` is native, editable Excalidraw JSON. It contains only rectangles, text and arrows. No AI image generator, embedded bitmap, or image element is used.

## Insert into Excalidraw

**Recommended:** open https://excalidraw.com and drag `ARIA_complete_architecture.excalidraw` onto the canvas, or use its Open command to load the file.

**Copy/paste code:** open `ARIA_complete_architecture.clipboard.json` in a text editor, select its entire JSON contents and copy them, then paste onto the Excalidraw canvas. This uses the native `excalidraw/clipboard` format. Do not paste it into the Mermaid dialog.

Format reference: https://docs.excalidraw.com/docs/codebase/json-schema

## Read the diagram

1. **Live interview:** candidate interface, API/session orchestration, grounded skills, current live belief updates, question generation and browser speech. The current fixed-score / three-action shortcut is explicitly identified.
2. **Evidence to action:** all perception and auxiliary modules, the separate 72-feature fusion representation, calibrated beliefs, the 33-feature policy state, and the eight-action IQL policy. Dashed routes are integration work, not claims that the live app already executes them.
3. **Reporting and output:** fairness audit, narrative report, SQLite feedback storage and the local TTS/avatar prototype. Connections are intended post-interview integration; avatar functionality is marked as a placeholder.
4. **Offline learning:** synthetic evidence, identity-safe splits, calibration and replay, IQL/comparators, and separate belief/policy evaluations. The checkpoint route points to the offline policy, not directly into the live application.

Blue-green identifies live functionality, purple identifies the offline learning/evaluation path, and amber identifies independently implemented modules. Solid arrows show implemented paths in the indicated context; dashed arrows indicate pending integration. Explanations and abbreviation expansions are also inside the diagram.

## Source and verification

Checked against `app.py`, the 15 module directories, `architecture.md`, the fusion feature schema, the current state schema, and the evaluation/comparator implementation. The current code's 33-feature state takes precedence over the older 32-feature description in README.

The auxiliary context route goes to the policy state; it does not imply that gaze or emotion is used as a calibrated competence likelihood. The current belief likelihood uses semantic evidence. Fairness and anti-gaming modules are heuristic services, not validated claims about candidates.

The JSON was structurally checked for unique IDs, valid references, module coverage and supported native element types. The PNG is a deterministic layout preview, not a screenshot of an Excalidraw import. It is generated from the same geometry and text as the native scene. The SVG is a vector preview.

To rebuild with Python and Pillow:

```powershell
python docs/architecture/build_complete_architecture.py
```

All 15 modules appear on the canvas. M6 appears twice to distinguish its simplified live use from the calibrated offline path. M1's live transcription and separate semantic grading are explicitly distinguished.
