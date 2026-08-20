# ARIA Frontend

The React interface for ARIA's adaptive interview workflow. It provides two primary experiences:

- **Interview setup:** accepts a job description and candidate résumé, then starts an ontology-backed session through the FastAPI service.
- **Live interview:** presents adaptive questions, candidate video, recording controls, typed responses, the live transcript, session state, and developer traces.

## Run locally

```bash
npm ci
npm run dev
```

The frontend expects the ARIA API at `http://localhost:8000` and connects to interview sessions through `ws://localhost:8000/ws/interview/{session_id}`.

## Quality checks

```bash
npm run lint
npm run build
```

The interface is responsive across desktop, tablet, and mobile layouts. It uses native system typography and inline SVG controls, so it has no external font or icon requests.
