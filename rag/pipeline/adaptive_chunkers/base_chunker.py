"""
pipeline/adaptive_chunkers/base_chunker.py
Shared Chunk type, the label-splitting helper, and the size guard every
chunker runs its output through.

Every doc-type-specific chunker returns list[Chunk] with the same shape,
so the case indexer and the fusion layer don't need to know which chunker
produced them.
"""
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import GENERIC_CHUNK_SIZE, GENERIC_CHUNK_OVERLAP


@dataclass
class Chunk:
    chunk_id:    str                 # unique within the document, e.g. "fir_incident_0"
    case_id:     str
    document_id: str
    doc_type:    str
    chunk_role:  str                 # e.g. "incident", "diagnosis", "witness_statement"
    text:        str
    metadata:    dict = field(default_factory=dict)   # entities relevant to this chunk


# Text that appears before the first recognised label. In a real FIR this
# is the letterhead, the FIR number, the police station and the date — the
# document's identifying header. It is emitted as its own chunk rather than
# discarded.
PREAMBLE_ROLE = "preamble"


def split_by_labels(
    text: str,
    label_patterns: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Split a labelled document into ``(role, text)`` sections, in document
    order.

    Returns a LIST, not a dict, because a role can legitimately occur more
    than once — a charge sheet with two accused has two "accused" blocks,
    an FIR can list several witnesses. The previous dict-based version kept
    only the FIRST match of each role's patterns, so every later occurrence
    was swallowed into whichever section preceded it.

    It also emits everything before the first label as a PREAMBLE_ROLE
    section. The old version started its first slice AT the first label
    hit, so ``text[:first_hit]`` — the entire document header — was
    silently dropped from every structured document.
    """
    hits: list[tuple[int, int, str]] = []   # (start, end, role)
    for role, patterns in label_patterns.items():
        for pattern in patterns:
            # MULTILINE so a pattern can anchor a label to the start of
            # its own line (^\s*sections?\s*[:-]), which is how a real
            # form field is written and how it is distinguished from the
            # same words appearing mid-sentence in narrative prose.
            for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                hits.append((m.start(), m.end(), role))

    if not hits:
        return []

    # Sort by position; where two patterns match at the same place, prefer
    # the longer match so a specific label wins over a generic one.
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))

    # Drop overlapping hits, keeping the first.
    deduped: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, role in hits:
        if start >= last_end:
            deduped.append((start, end, role))
            last_end = end

    sections: list[tuple[str, str]] = []

    preamble = text[: deduped[0][0]].strip()
    if preamble:
        sections.append((PREAMBLE_ROLE, preamble))

    for i, (start, _end, role) in enumerate(deduped):
        stop = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        body = text[start:stop].strip()
        if body:
            sections.append((role, body))

    return sections


def enforce_max_size(
    sections: list[tuple[str, str]],
    max_chars: int = GENERIC_CHUNK_SIZE,
    overlap:   int = GENERIC_CHUNK_OVERLAP,
) -> list[tuple[str, str]]:
    """Window any section longer than the embedding model can actually read.

    bge-large truncates at 512 tokens silently, so a long narrative section
    — an FIR's "brief facts" is routinely the longest part of the document
    — was being embedded from its opening lines alone, with the remainder
    contributing nothing to retrieval and never surfacing as its own chunk.
    """
    out: list[tuple[str, str]] = []
    step = max(1, max_chars - overlap)
    for role, body in sections:
        if len(body) <= max_chars:
            out.append((role, body))
            continue
        for start in range(0, len(body), step):
            window = body[start:start + max_chars].strip()
            if window:
                out.append((role, window))
            if start + max_chars >= len(body):
                break
    return out


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

    def _chunks_from_sections(
        self,
        sections:    list[tuple[str, str]],
        document_id: str,
        case_id:     str,
        entities:    dict | None,
    ) -> list[Chunk]:
        """Shared tail of every structural chunker: size-guard the sections
        and turn them into Chunks with unique ids."""
        meta = {"entities": entities} if entities else None
        return [
            self._make(role, body, document_id, case_id, i, metadata=meta)
            for i, (role, body) in enumerate(enforce_max_size(sections))
            if body.strip()
        ]
