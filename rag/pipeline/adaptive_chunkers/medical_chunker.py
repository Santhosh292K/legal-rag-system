"""
pipeline/adaptive_chunkers/medical_chunker.py
Phase 2 — Medical report chunker: Patient -> Injuries -> Cause -> Opinion
"""
from .base_chunker import BaseChunker, Chunk, split_by_labels
from .generic_chunker import GenericChunker

LABEL_PATTERNS = {
    "patient":  [r"patient\s*(?:name|details?)?\s*[:\-]"],
    "injuries":  [r"injur(?:y|ies)\s*(?:noted|found|observed)?\s*[:\-]",
                  r"wounds?\s*[:\-]"],
    "cause":      [r"cause\s+of\s+(?:death|injury)\s*[:\-]", r"mechanism\s+of\s+injury\s*[:\-]"],
    "opinion":      [r"(?:medical\s+)?opinion\s*[:\-]", r"conclusion\s*[:\-]"],
}

class MedicalReportChunker(BaseChunker):
    doc_type = "Medical Report"

    def chunk(self, text: str, document_id: str, case_id: str,
              entities: dict | None = None) -> list[Chunk]:
        sections = split_by_labels(text, LABEL_PATTERNS)
        chunks = self._chunks_from_sections(sections, document_id, case_id, entities)

        # No labels matched -> this document doesn't follow the expected
        # template (a free-form or handwritten transcript, say). Don't drop
        # it; fall back to generic sliding-window chunking.
        if not chunks:
            return GenericChunker(doc_type=self.doc_type).chunk(
                text, document_id, case_id, entities)
        return chunks
