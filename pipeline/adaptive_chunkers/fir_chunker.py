"""
pipeline/adaptive_chunkers/fir_chunker.py
Phase 2 — FIR chunker: Complainant -> Accused -> Incident -> Time -> Sections -> Relief
"""
from .base_chunker import BaseChunker, Chunk, split_by_labels
from .generic_chunker import GenericChunker

LABEL_PATTERNS = {
    "complainant": [r"complainant\s*(?:name)?\s*[:\-]"],
    "accused":      [r"accused\s*(?:name)?\s*[:\-]"],
    "incident":      [r"(?:incident|occurrence)\s+details?\s*[:\-]",
                       r"brief\s+facts\s*[:\-]", r"narration\s*[:\-]"],
    "time":           [r"(?:date|time)\s+of\s+(?:occurrence|incident)\s*[:\-]",
                        r"occurred\s+on\s*[:\-]?"],
    "sections":        [r"(?:sections?|offen[cs]es?)\s+(?:invoked|applied)\s*[:\-]",
                         r"under\s+section"],
    "relief":           [r"(?:relief|action)\s+(?:sought|requested|taken)\s*[:\-]",
                          r"prayer\s*[:\-]"],
}


class FIRChunker(BaseChunker):
    doc_type = "FIR"

    def chunk(self, text: str, document_id: str, case_id: str,
              entities: dict | None = None) -> list[Chunk]:
        sections = split_by_labels(text, LABEL_PATTERNS)

        chunks = []
        for i, (role, content) in enumerate(sections.items()):
            if content:
                chunks.append(self._make(role, content, document_id, case_id, i,
                                          metadata={"entities": entities} if entities else None))

        # No labels matched at all -> this FIR doesn't follow the expected
        # template (e.g. free-form handwritten transcript). Don't drop the
        # document; fall back to generic chunking instead.
        if not chunks:
            return GenericChunker(doc_type=self.doc_type).chunk(text, document_id, case_id, entities)

        return chunks


if __name__ == "__main__":
    sample = """
    Complainant Name: Ramesh Kumar
    Accused Name: Suresh Yadav
    Date of Occurrence: 12/05/2024
    Brief Facts: The accused attacked the complainant with a knife near
    Andheri Station Road causing grievous injury.
    Sections invoked: 103, 118 BNS
    Relief Sought: Strict legal action against the accused.
    """
    for c in FIRChunker().chunk(sample, document_id="doc1", case_id="case1"):
        print(f"[{c.chunk_role}] {c.text[:60]}...")
