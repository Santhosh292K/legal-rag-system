"""
pipeline/fusion.py
The piece that actually connects Track B (case documents) to Track A
(the existing statute pipeline) — retrieves case chunks, pulls out any
sections they cite or imply, feeds those into LegalRAGPipeline.query()
via its extra_sections parameter, and returns both sets of chunks
together for the answer generator.

This does NOT modify LegalRAGPipeline's own retrieval/reranking —
Track A runs exactly as it does for a general query, just with extra
sections pinned going in.
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.alea import ALEA, entities_to_facts, SectionScore
from pipeline.section_pinner import SectionPinner
from pipeline.keyword_index import KeywordIndex

SECTION_ID_PATTERN = re.compile(r"\b[A-Z]{2,6}_[A-Z0-9]+(?:_[A-Z0-9]+)*\b")

# Broad questions where the query text itself is too vague to reliably
# match specific chunks via dense similarity — 'summarize this case' isn't
# semantically close to any one fact, so top-k search against it would
# return a somewhat arbitrary slice rather than the full picture.
BROAD_QUERY_PATTERNS = [
    r"\b(case\s+details|summar\w+\s+(this\s+)?case|overview\s+of\s+(this\s+)?case)\b",
    r"\btell\s+me\s+about\s+(this|the)\s+case\b",
    r"\bwhat\s+(happened|is\s+this\s+case\s+about)\b",
    r"\ball\s+(the\s+)?(facts|details|evidence)\s+(in|of)\s+(this|the)\s+case\b",
]


def _is_broad_query(query: str) -> bool:
    q = query.lower()
    return any(re.search(p, q) for p in BROAD_QUERY_PATTERNS)


@dataclass
class FusedResult:
    query:           str
    case_id:         str
    case_chunks:     list[dict] = field(default_factory=list)   # from CaseIndexer.search()
    evidence_sections: list[str] = field(default_factory=list)  # sections pulled from case chunks
    statute_answer:  object = None    # LegalAnswer from LegalRAGPipeline.query()
    alea_scores:     list[SectionScore] = field(default_factory=list)   # Phase 3/4


def _sections_from_case_narrative(case_chunks: list[dict], pinner: SectionPinner,
                                   keyword_index: KeywordIndex | None = None) -> list[str]:
    """Runs BOTH deterministic section-finding layers against the case
    document's own text: section_pinner's semantic search against the
    indexed statute corpus (catches phrasing variants — 'abducted'/
    'kidnapped'/'taken away forcibly' all land near the same sections in
    embedding space), and the dataset-driven keyword index (catches exact
    dataset-authored terms no embedding match is guaranteed to surface,
    like IPC_364A's 'kidnapping for ransom' — see keyword_index.py for why
    this matters). Neither replaces the other; they catch different gaps.
    Scans every retrieved chunk's text, not just 'incident'-role ones —
    an FIR's 'Offences Invoked' line (a goldmine of exactly these words)
    lands in the chunker's 'sections' role, not 'incident'."""
    text = " ".join(c.get("text", "") for c in case_chunks)
    if not text.strip():
        return []

    pinned = pinner.pin(text).section_ids
    keyword_matched = keyword_index.section_ids(text) if keyword_index else []

    # Semantic pins first (highest precision), then keyword-index matches,
    # deduped preserving order.
    return list(dict.fromkeys(pinned + keyword_matched))


def _sections_from_case_chunks(case_chunks: list[dict]) -> list[str]:
    """Case chunk metadata carries entities.sections_cited (from Phase 1's
    entity_timeline_extractor) — e.g. '103 BNS', '118 BNS'. Convert those
    into the dataset's own section_id format ('BNS_103') so they can be fed
    straight into the pinner's output via extra_sections. Falls back to
    scanning the raw chunk text for an already-correct ID format, in case
    the caller passes pre-formatted chunks."""
    sections = []

    for chunk in case_chunks:
        metadata = chunk.get("metadata") or {}
        entities = metadata.get("entities") or {}
        for cited in entities.get("sections_cited", []):
            # "103 BNS" -> "BNS_103"; "420" (no act) is skipped, too ambiguous
            # to guess an act for — better to miss it than pin the wrong act.
            parts = cited.strip().split()
            if len(parts) == 2:
                number, act = parts
                # BUGFIX: this passed `number` through unchanged — no
                # uppercasing (a lettered cite like "120b bns" stayed
                # lowercase) and no zero-padding (final_dataset.json zero-
                # pads the numeric part of section_id to 3 digits for every
                # act except CRPC — e.g. "5 BNS" needs to become "BNS_005",
                # not "BNS_5"). Since this feeds fetch_by_ids(), which does
                # an exact single-value match per id with no fuzzy fallback
                # (pipeline/hybrid_retriever.py), a mismatch here silently
                # dropped the cited section rather than pinning it — a real
                # document citing a low-numbered or lettered section (common:
                # IPC_498A, IPC_120B) would lose that citation entirely.
                # Emit both the zero-padded and raw-width candidates (like
                # hybrid_retriever.py's direct lookup already does for the
                # same CRPC-is-unpadded reason) — fetch_by_ids() silently
                # skips whichever one doesn't exist, so this is safe either way.
                num_norm = number.strip().upper()
                m = re.match(r'^(\d+)([A-Z]?)$', num_norm)
                if m:
                    digits, letter = m.groups()
                    sections.append(f"{act.upper()}_{digits.zfill(3)}{letter}")
                    sections.append(f"{act.upper()}_{digits}{letter}")
                else:
                    sections.append(f"{act.upper()}_{num_norm}")

        # Also catch already-formatted IDs anywhere in the chunk text itself.
        sections.extend(SECTION_ID_PATTERN.findall(chunk.get("text", "")))

    # Dedupe, preserve order.
    seen, ordered = set(), []
    for s in sections:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def _facts_from_case_chunks(case_chunks: list[dict]) -> list:
    """Aggregates EvidenceFact objects across every retrieved case chunk's
    metadata.entities (Phase 1 output), for ALEA to score against.
    Phase 1 stores a document's FULL entity set on every one of that
    document's chunks (not just the chunk it was found in) — so naively
    aggregating across chunks would count the same fact once per chunk.
    Dedupe by (text, document_id) to count each fact once per document."""
    facts, seen = [], set()
    for chunk in case_chunks:
        metadata = chunk.get("metadata") or {}
        entities = metadata.get("entities") or {}
        if not entities:
            continue
        document_id = chunk.get("document_id", "")
        for fact in entities_to_facts(entities, doc_type=chunk.get("doc_type", "Other"),
                                       document_id=document_id):
            key = (fact.text, document_id)
            if key not in seen:
                seen.add(key)
                facts.append(fact)
    return facts


class CaseStatuteFusion:
    """Depends on an existing LegalRAGPipeline instance and CaseIndexer
    instance — doesn't own or reload either, so callers control lifecycle
    and both stay loaded once, same as the rest of the system.
    alea is optional — pass an ALEA instance (built with a shared embed_fn)
    to also get evidence-to-law coverage scoring; omit to run fusion without
    it (extra_sections-based pinning still works either way)."""

    def __init__(self, statute_pipeline, case_indexer, alea: ALEA | None = None):
        self.pipeline = statute_pipeline
        self.indexer  = case_indexer
        self.alea     = alea
        self.keyword_index = KeywordIndex()   # pure Python, loads instantly — no model needed

    def _retrieve_case_chunks(self, query: str, case_id: str, case_top_k: int) -> list[dict]:
        """Shared by answer() and answer_document_only() — routes broad
        summary-style questions to a full-case fetch instead of vague-query
        similarity search (see _is_broad_query / BROAD_QUERY_PATTERNS)."""
        if _is_broad_query(query):
            return self.indexer.get_all_chunks(case_id=case_id)
        return self.indexer.search(query, case_id=case_id, top_k=case_top_k)

    def answer(self, query: str, case_id: str, case_top_k: int = 8) -> FusedResult:
        case_chunks = self._retrieve_case_chunks(query, case_id, case_top_k)

        cited_sections     = _sections_from_case_chunks(case_chunks)
        narrative_sections = _sections_from_case_narrative(case_chunks, self.pipeline.pinner, self.keyword_index)
        # Preserve order, dedupe — narrative-pinned sections appended after
        # explicit citations, since an explicit "Section 302 IPC" in the
        # document is a stronger signal than a plain-English pattern match.
        evidence_sections = list(dict.fromkeys(cited_sections + narrative_sections))

        statute_answer = self.pipeline.query(
            query, extra_sections=evidence_sections or None,
        )

        alea_scores = []
        if self.alea is not None:
            facts = _facts_from_case_chunks(case_chunks)
            if facts:
                # Score the evidence-cited sections plus whatever the statute
                # pipeline itself retrieved — a candidate a query alone
                # wouldn't have pinned, but that shares real overlap with
                # the case's own evidence, is still worth surfacing.
                statute_ids = [c.section_id for c in statute_answer.citations] if statute_answer.citations else []
                candidates = list(dict.fromkeys(evidence_sections + statute_ids))
                alea_scores = self.alea.score_sections(facts, candidate_section_ids=candidates or None)

        return FusedResult(
            query=query,
            case_id=case_id,
            case_chunks=case_chunks,
            evidence_sections=evidence_sections,
            statute_answer=statute_answer,
            alea_scores=alea_scores,
        )

    def answer_document_only(self, query: str, case_id: str, case_top_k: int = 8) -> FusedResult:
        """For 'document'-routed queries — no statute call at all, just the
        case chunks. Kept separate from answer() so a document-grounded
        question never accidentally pulls in a statute pipeline generation
        call it doesn't need."""
        case_chunks = self._retrieve_case_chunks(query, case_id, case_top_k)
        return FusedResult(query=query, case_id=case_id, case_chunks=case_chunks)