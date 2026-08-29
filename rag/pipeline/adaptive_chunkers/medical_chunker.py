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

        chunks = []
        for i, (role, content) in enumerate(sections.items()):
            if content:
                chunks.append(self._make(role, content, document_id, case_id, i,
                                          metadata={"entities": entities} if entities else None))

        if not chunks:
            return GenericChunker(doc_type=self.doc_type).chunk(text, document_id, case_id, entities)

        return chunks
