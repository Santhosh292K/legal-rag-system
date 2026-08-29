"""
backend/app/main.py
FastAPI service fronting the Legal RAG pipeline in ../rag.

Run:
    uvicorn app.main:app --reload --port 8000     (from backend/)

The heavy pipeline (embedding model, cross-encoder, Qdrant, KG) loads once
in a background thread kicked off on startup — see app/state.py. Until it
finishes, /api/health reports "loading" and /api/query returns 503.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app import state
from app.routers import health, query, cases

app = FastAPI(
    title="Legal RAG API",
    description="Citation-grounded Q&A over Indian statutes, with optional "
                 "per-case document fusion (FIRs, charge sheets, evidence).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(cases.router)


@app.on_event("startup")
def _on_startup():
    state.start_loading()


@app.get("/")
def root():
    return {"service": "legal-rag-api", **state.get_status()}
