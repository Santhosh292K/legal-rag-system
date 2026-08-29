# Legal RAG — Backend

FastAPI service that wraps the pipeline in [`../rag`](../rag) (imported
in-process, not shelled out to) and exposes it over HTTP for the
[`../frontend`](../frontend) Next.js app.

## Setup

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pulls in ../rag/requirements.txt too

cp .env.example .env                   # backend service settings (CORS, port)
```

The pipeline itself still needs its own configuration — see
[`../rag/README.md`](../rag/README.md): a `../rag/.env` with `GEMINI_API_KEY`,
the dataset indexed into `../rag/qdrant_db`, and (for answer generation)
[Ollama](https://ollama.com/) running locally with the model set in
`rag/config.py` (`OLLAMA_ANSWER_MODEL`) pulled.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Model loading (bge-large, bge-reranker-large, Qdrant, the legal KG) happens
once in a background thread on startup — `GET /api/health` reports
`"loading"` until it's done, then `"ready"` (or `"error"` with detail).
`POST /api/query` returns `503` until the system is ready.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/api/health` | Pipeline load status |
| `POST` | `/api/query` | `{query, case_id?}` → citation-grounded answer |
| `POST` | `/api/cases/{case_id}/upload` | Multipart file upload (PDF or scanned image) → ingest + index into that case |
| `GET`  | `/api/cases` | List cases uploaded to this session and their documents |
| `GET`  | `/api/cases/{case_id}` | One case's uploaded documents |

Interactive docs at `/docs` once running.

## Notes

- Requests are serialized through a process-wide lock (`app/state.py`) —
  the underlying pipeline is synchronous, model-bound code, not designed
  for concurrent calls into one instance. Fine for a single-instance
  deployment; scale by running multiple processes behind a load balancer,
  not by relying on in-process concurrency.
- The case registry (`GET /api/cases`) is in-memory and resets on
  restart — it's a convenience index for the UI, not the source of truth.
  The source of truth is `rag/qdrant_db`, which is persistent.
