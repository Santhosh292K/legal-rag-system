# Nyaya — Legal RAG, Full Stack

A citation-grounded Q&A system over Indian statute law, with optional
per-case document fusion (FIRs, charge sheets, evidence). Three parts:

```
.
├── rag/        Python RAG pipeline — hybrid retrieval, IRAC reranking,
│               temporal validity filtering, grounded answer generation.
│               CLI-first, no web layer of its own.
├── backend/    FastAPI service that imports rag/ in-process and exposes
│               it over HTTP (query, case upload, health).
└── frontend/   Next.js app — the chat UI: citation-grounded answers,
                IRAC coverage, case management, document upload.
```

Each has its own README with full detail: [`rag/README.md`](rag/README.md),
[`backend/README.md`](backend/README.md), [`frontend/README.md`](frontend/README.md).

## Quick start

**1. The pipeline's data** — dataset indexed into Qdrant, BM25 index built,
and a Gemini key or local Ollama model for answer generation. See
[`rag/README.md`](rag/README.md) Setup/Usage; this repo ships with
`rag/data/final_dataset.json`, `rag/qdrant_db`, and `rag/.env` already
populated for local dev, so you likely only need to confirm Ollama is
running.

**2. Backend:**
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```
Wait for `GET http://localhost:8000/api/health` to report `"status": "ready"`
(model loading happens once, in the background, and can take a while).

**3. Frontend:**
```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```
Open http://localhost:3000.

## Design notes

- The backend imports `rag/` directly (not a subprocess) — one process,
  one set of loaded models, one Qdrant client. See
  `backend/app/config.py:bootstrap_rag()` for how it wires `rag/`'s
  CWD-relative paths.
- Requests to the pipeline are serialized behind a lock (`backend/app/state.py`)
  — it's synchronous, model-bound code, not built for concurrent calls
  into one instance.
- The frontend never talks to `rag/` directly; all access goes through
  the backend's `/api/*` routes (`frontend/src/lib/api.ts`).
