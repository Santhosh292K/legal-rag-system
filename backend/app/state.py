"""
backend/app/state.py
Process-wide state: the (lazily, background-loaded) RAG system singleton,
a lock serializing calls into it, and an in-memory case registry.

The underlying LegalRAGSystem loads several large models (bge-large,
bge-reranker-large) plus an embedded Qdrant client — that's slow (seconds
to minutes) and not something every request should pay for or race to
do. It's loaded once in a background thread kicked off at app startup;
`get_status()` reports progress so the API can answer /api/health
immediately instead of hanging while models load.

The pipeline is CPU/GPU-model-bound synchronous Python, not designed for
concurrent calls into the same instance — a global lock serializes actual
inference so requests queue instead of racing on shared model/GPU state.
Route handlers await `run_in_threadpool` around every call so the lock
doesn't block the asyncio event loop.
"""
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

from app.config import bootstrap_rag

_lock = threading.Lock()
_load_lock = threading.Lock()

_state = {
    "status": "loading",   # "loading" | "ready" | "error"
    "detail": "Starting up…",
    "system": None,
}

# case_id -> list[dict] of uploaded-document summaries, newest last.
# In-memory only — restarts clear it. The underlying vector index
# (rag/qdrant_db) is what's actually persistent; this registry just lets
# the frontend show "what's in this case" without re-deriving it.
_cases: dict[str, list[dict]] = {}


def start_loading() -> None:
    """Kick off the (slow) model/pipeline load in a background thread.
    Safe to call more than once — only the first call does anything."""
    if not _load_lock.acquire(blocking=False):
        return

    def _load():
        try:
            bootstrap_rag()
            # Imported here, not at module load time: bootstrap_rag() must
            # run first so `rag/` is on sys.path and CWD before anything
            # under it is imported.
            from legal_rag_system import LegalRAGSystem
            t0 = time.time()
            system = LegalRAGSystem(verbose=True)
            _state["system"] = system
            _state["status"] = "ready"
            _state["detail"] = f"Loaded in {time.time() - t0:.1f}s"
        except Exception as e:
            _state["status"] = "error"
            _state["detail"] = f"{e}\n{traceback.format_exc()}"

    threading.Thread(target=_load, daemon=True, name="rag-loader").start()


def get_status() -> dict:
    return {"status": _state["status"], "detail": _state["detail"]}


def get_system():
    if _state["status"] != "ready" or _state["system"] is None:
        raise RuntimeError(
            f"RAG system is not ready (status={_state['status']}: {_state['detail']})"
        )
    return _state["system"]


def get_lock() -> threading.Lock:
    return _lock


def record_upload(case_id: str, summary: dict) -> None:
    _cases.setdefault(case_id, []).append({
        **summary,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    })


def list_cases() -> list[dict]:
    return [{"case_id": cid, "documents": docs} for cid, docs in _cases.items()]


def get_case(case_id: str) -> Optional[list[dict]]:
    return _cases.get(case_id)
