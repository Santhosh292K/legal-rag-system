import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────
OLLAMA_FAST_MODEL     = "qwen2.5:3b"
OLLAMA_ANSWER_MODEL   = "qwen2.5:14b" 

# ── Embeddings ───────────────────────────────────────
EMBEDDING_MODEL       = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM         = 1024
RERANKER_MODEL        = "BAAI/bge-reranker-large"

# ── Qdrant ───────────────────────────────────────────
QDRANT_PATH           = "./qdrant_db"
COLLECTION_NAME       = "legal_sections"

# ── Retrieval ────────────────────────────────────────
BM25_TOP_K    = 25
DENSE_TOP_K   = 25
HYBRID_TOP_K  = 35
RERANK_TOP_K  = 20
# diagnose_recall.py showed IPC_386, IPC_468, and SRA_006 all reached
# raw_pool/RERANK_TOP_K but were cut by FINAL_TOP_K=8. Bumped to 10 as a
# targeted, modest change to recover those specific truncation losses.
# This is a recall/precision trade-off, not a free win — it hands the
# answer generator 2 more sections per query to reason over, which can
# dilute precision if the IRAC reranker's ordering is often wrong near the
# cutoff. Re-run evaluate.py's precision metric after this change; if
# precision drops more than you're willing to trade, revert to 8 and fix
# this via IRAC_WEIGHTS / reranker scoring instead, per the original
# priority note that reranker/top-k needed a different fix than query
# expansion.
FINAL_TOP_K   = 10

# ── IRAC Scoring weights ─────────────────────────────
IRAC_WEIGHTS = {
    "issue":       0.30,
    "rule":        0.35,
    "application": 0.25,
    "conclusion":  0.10,
}

# ── Pipeline ─────────────────────────────────────────
INTENT_LABELS         = ["statute", "case_law", "definition", "procedural", "punitive"]
STATUS_ACTIVE         = "active"

# ── Data paths ───────────────────────────────────────
CSV_PATH              = "./data/dataset.csv"
JSON_PATH             = "./data/final_dataset.json"

"""
config_additions.py
Append these to the END of your existing config.py.
Nothing here modifies or removes any existing constant —
it only adds the settings Phase 1 (evidence ingestion) needs.
"""

# ── Case documents (Track B) ─────────────────────────────────
CASE_DOCUMENTS_COLLECTION = "case_documents"

DOC_TYPES = [
    "FIR",
    "Charge Sheet",
    "Medical Report",
    "Witness Statement",
    "Forensic Report",
    "Contract",
    "Email",
    "Court Order",
    "Affidavit",
    "Other",
]

# Doc types with a dedicated structural chunker (Phase 2).
# Anything not listed here falls back to GenericChunker.
STRUCTURED_CHUNKER_TYPES = {"FIR", "Charge Sheet", "Medical Report"}

# ── OCR ───────────────────────────────────────────────────────
OCR_MIN_CHARS_PER_PAGE = 40      # below this, treat the page as scanned and OCR it
OCR_LANG               = "eng"

# ── Chunking ─────────────────────────────────────────────────
GENERIC_CHUNK_SIZE     = 800     # characters
GENERIC_CHUNK_OVERLAP  = 150

# ── Case vector store ───────────────────────────────────────
CASE_QDRANT_PATH        = "./qdrant_db"   # same instance, different collection