"""
pipeline/adaptive_chunkers/generic_chunker.py
Phase 2 — Fallback chunker.

Used for: Witness Statement, Forensic Report, Contract, Email,
Court Order, Affidavit, Other — and as the safety-net fallback for
FIR / Charge Sheet / Medical Report when their expected labels
aren't found (e.g. a free-form or non-standard document).

Sliding-window paragraph chunking, no assumed structure.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import GENERIC_CHUNK_SIZE, GENERIC_CHUNK_OVERLAP
from .base_chunker import BaseChunker, Chunk


class GenericChunker(BaseChunker):
    def __init__(self, doc_type: str = "Other"):
        self.doc_type = doc_type

    def chunk(self, text: str, document_id: str, case_id: str,
              entities: dict | None = None) -> list[Chunk]:
        # Prefer paragraph boundaries; fall back to a hard sliding window
        # only when a "paragraph" would blow past the chunk size.
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks, buffer, idx = [], "", 0
        for para in paragraphs:
            if len(buffer) + len(para) <= GENERIC_CHUNK_SIZE:
                buffer = f"{buffer}\n\n{para}".strip()
                continue

            if buffer:
                chunks.append(self._make("narrative", buffer, document_id, case_id, idx,
                                          metadata={"entities": entities} if entities else None))
                idx += 1

            if len(para) <= GENERIC_CHUNK_SIZE:
                buffer = para
            else:
                # Single paragraph longer than the chunk size — hard-window it.
                for start in range(0, len(para), GENERIC_CHUNK_SIZE - GENERIC_CHUNK_OVERLAP):
                    window = para[start:start + GENERIC_CHUNK_SIZE]
                    chunks.append(self._make("narrative", window, document_id, case_id, idx,
                                              metadata={"entities": entities} if entities else None))
                    idx += 1
                buffer = ""

        if buffer:
            chunks.append(self._make("narrative", buffer, document_id, case_id, idx,
                                      metadata={"entities": entities} if entities else None))

        return chunks
