"""
pipeline/adaptive_chunkers/base_chunker.py
Phase 2 — shared Chunk type + chunker interface.

Every doc-type-specific chunker returns a list[Chunk] with the same
shape, so the case indexer and retrieval/fusion layers don't need to
know which chunker produced them.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id:    str                 # unique within the document, e.g. "fir_incident_0"
    case_id:     str
    document_id: str
    doc_type:    str
    chunk_role:  str                 # e.g. "incident", "diagnosis", "witness_statement"
    text:        str
    metadata:    dict = field(default_factory=dict)   # entities relevant to this chunk


def split_by_labels(text: str, label_patterns: dict[str, list[str]]) -> dict[str, str]:
    """
    Shared helper for label-driven structural chunkers (FIR, Chargesheet,
    Medical Report). Finds the first occurrence of each role's label
    patterns, in document order, and slices the text between consecutive
    labels. Roles with no match get an empty string — the caller decides
    whether to skip them or fall back to a generic chunk.
    """
    import re

    hits = []  # (start_pos, role)
    for role, patterns in label_patterns.items():
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                hits.append((m.start(), role))
                break

    hits.sort(key=lambda h: h[0])
    result = {role: "" for role in label_patterns}

    for i, (start, role) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        result[role] = text[start:end].strip()

    return result


class BaseChunker(ABC):
    """Subclass and implement chunk() for a specific document type."""

    doc_type: str = "Other"

    @abstractmethod
    def chunk(self, text: str, document_id: str, case_id: str,
              entities: dict | None = None) -> list[Chunk]:
        ...

    def _make(self, role: str, text: str, document_id: str, case_id: str,
               index: int, metadata: dict | None = None) -> Chunk:
        return Chunk(
            chunk_id=f"{self.doc_type.lower().replace(' ', '_')}_{role}_{index}",
            case_id=case_id,
            document_id=document_id,
            doc_type=self.doc_type,
            chunk_role=role,
            text=text.strip(),
            metadata=metadata or {},
        )
