"""
backend/app/config.py
Backend service settings. Also responsible for wiring the ``rag/`` package
onto sys.path and pinning the process working directory to it, since the
pipeline's own config (rag/config.py) and several pipeline modules resolve
their data/index paths (``./data/...``, ``./qdrant_db``) relative to the
current working directory rather than to their own file location. This
must happen before anything under ``rag/`` is imported, so `bootstrap_rag()`
is called once at the top of app/state.py.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# backend/app/config.py -> parents[0]=app, [1]=backend, [2]=repo root
REPO_ROOT    = Path(__file__).resolve().parents[2]
RAG_DIR      = REPO_ROOT / "rag"
BACKEND_DIR  = REPO_ROOT / "backend"

# Explicit absolute path: this must load backend/.env regardless of CWD,
# and specifically *before* bootstrap_rag() below chdirs into rag/ (which
# has its own .env, loaded independently by rag/config.py).
load_dotenv(BACKEND_DIR / ".env")

# CORS — the Next.js dev server (and optionally a deployed frontend origin).
CORS_ORIGINS = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",") if o.strip()
]

HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT = int(os.getenv("BACKEND_PORT", "8000"))

# Reject uploads above this size before they ever touch the ingestion pipeline.
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))

_bootstrapped = False


def bootstrap_rag() -> None:
    """Make ``rag/`` importable as a set of top-level modules (``config``,
    ``main``, ``legal_rag_system``, ``pipeline.*``, ``data.*``) exactly as
    they import each other internally, and pin CWD there so their relative
    default paths resolve correctly regardless of where uvicorn was
    launched from. Idempotent — safe to call more than once."""
    global _bootstrapped
    if _bootstrapped:
        return
    if not RAG_DIR.is_dir():
        raise RuntimeError(
            f"rag/ package not found at {RAG_DIR}. The backend expects the "
            f"repo layout: <root>/rag, <root>/backend, <root>/frontend."
        )
    if str(RAG_DIR) not in sys.path:
        sys.path.insert(0, str(RAG_DIR))
    os.chdir(RAG_DIR)
    _bootstrapped = True
