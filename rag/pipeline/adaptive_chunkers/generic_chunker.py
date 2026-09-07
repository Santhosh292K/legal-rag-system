"""
pipeline/adaptive_chunkers/generic_chunker.py
Fallback chunker — sliding-window over paragraphs, no assumed structure.

Used for Witness Statement, Forensic Report, Contract, Email, Court Order,
Affidavit and Other, and as the safety net when a structural chunker finds
none of its expected labels.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import GENERIC_CHUNK_SIZE, GENERIC_CHUNK_OVERLAP
from .base_chunker import BaseChunker, Chunk, enforce_max_size


class GenericChunker(BaseChunker):
    def __init__(self, doc_type: str = "Other"):
        self.doc_type = doc_type

    def chunk(self, text: str, document_id: str, case_id: str,
              entities: dict | None = None) -> list[Chunk]:
        sections = [("narrative", body) for body in self._windows(text)]
        return self._chunks_from_sections(sections, document_id, case_id, entities)

    @staticmethod
    def _windows(text: str) -> list[str]:
        """Group paragraphs up to GENERIC_CHUNK_SIZE, with a real overlap
        between consecutive windows.

        BUGFIX: GENERIC_CHUNK_OVERLAP was previously applied ONLY inside
        the hard-window fallback for a single oversized paragraph. On the
        normal paragraph path — which is the path essentially every real
        document takes — consecutive chunks shared nothing at all, so a
        fact spanning a paragraph boundary (an FIR narrative that names the
        accused in one paragraph and the weapon in the next) was split
        across two chunks with no context bridging them, and neither chunk
        retrieved well for a question about the pair.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            stripped = text.strip()
            return [stripped] if stripped else []

        windows: list[str] = []
        buffer = ""

        def flush():
            nonlocal buffer
            if buffer.strip():
                windows.append(buffer.strip())
            buffer = ""

        for para in paragraphs:
            # +2 accounts for the "\n\n" separator, which the old size test
            # ignored — chunks could exceed GENERIC_CHUNK_SIZE by the
            # number of joins in them.
            candidate_len = len(buffer) + (2 if buffer else 0) + len(para)
            if candidate_len <= GENERIC_CHUNK_SIZE:
                buffer = f"{buffer}\n\n{para}" if buffer else para
                continue

            flush()
            # Carry the tail of the previous window forward so consecutive
            # chunks overlap.
            if windows and GENERIC_CHUNK_OVERLAP > 0:
                buffer = windows[-1][-GENERIC_CHUNK_OVERLAP:].lstrip()
            buffer = f"{buffer}\n\n{para}".strip() if buffer else para

        flush()

        # A single paragraph can still exceed the size on its own; the
        # shared size guard windows those with the same overlap.
        return [body for _role, body in
                enforce_max_size([("narrative", w) for w in windows])]
