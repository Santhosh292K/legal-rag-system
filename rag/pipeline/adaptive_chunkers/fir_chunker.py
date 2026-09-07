"""
pipeline/adaptive_chunkers/fir_chunker.py
Phase 2 — FIR chunker: Complainant -> Accused -> Incident -> Time -> Sections -> Relief
"""
from .base_chunker import BaseChunker, Chunk, split_by_labels
from .generic_chunker import GenericChunker

# Patterns are matched with re.IGNORECASE | re.MULTILINE semantics via
# split_by_labels; ^ therefore anchors to a line start.
LABEL_PATTERNS = {
    "complainant": [r"complainant\s*(?:name)?\s*[:\-]"],
    "accused":      [r"accused\s*(?:name)?\s*[:\-]"],
    "incident":      [r"(?:incident|occurrence)\s+details?\s*[:\-]",
                       r"brief\s+facts\s*[:\-]", r"narration\s*[:\-]"],
    "time":           [r"(?:date|time)\s+of\s+(?:occurrence|incident)\s*[:\-]",
                        r"occurred\s+on\s*[:\-]?"],
    # NOTE: a bare r"under\s+section" used to be listed here. That phrase
    # appears constantly inside FIR narrative prose ("the accused was
    # charged under section 302..."), so the first narrative mention became
    # the split point and everything after it — the real "Sections
    # Invoked" block included — collapsed into one mislabelled chunk.
    # Anchored forms only.
    "sections":        [r"(?:sections?|offen[cs]es?)\s+(?:invoked|applied|registered)\s*[:\-]",
                         r"^\s*(?:u/s|under\s+sections?)\s*[:\-]",
                         r"^\s*sections?\s*[:\-]"],
    "relief":           [r"(?:relief|action)\s+(?:sought|requested|taken)\s*[:\-]",
                          r"prayer\s*[:\-]"],
}

class FIRChunker(BaseChunker):
    doc_type = "FIR"

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
