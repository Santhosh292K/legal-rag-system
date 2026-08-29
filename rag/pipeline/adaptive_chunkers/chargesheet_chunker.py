"""
pipeline/adaptive_chunkers/chargesheet_chunker.py
Phase 2 — Chargesheet chunker: Charges -> Evidence list -> Witness statements -> IO remarks
"""
from .base_chunker import BaseChunker, Chunk, split_by_labels
from .generic_chunker import GenericChunker

LABEL_PATTERNS = {
    "charges":            [r"charges?\s+framed\s*[:\-]", r"offen[cs]es?\s+charged\s*[:\-]"],
    "evidence_list":       [r"list\s+of\s+(?:evidence|exhibits?)\s*[:\-]",
                              r"evidence\s+collected\s*[:\-]"],
    "witness_statements":   [r"(?:list\s+of\s+)?witness(?:es)?\s+(?:statements?|examined)\s*[:\-]"],
    "io_remarks":            [r"investigating\s+officer.{0,30}remarks?\s*[:\-]",
                               r"conclusion\s+of\s+investigation\s*[:\-]"],
}


class ChargesheetChunker(BaseChunker):
    doc_type = "Charge Sheet"

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
