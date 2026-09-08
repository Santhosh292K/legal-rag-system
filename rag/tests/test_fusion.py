"""Track B -> Track A hand-off: which sections an uploaded case nominates."""
import pytest

from pipeline.fusion import (
    _sections_from_case_narrative, _sections_from_case_chunks,
    MAX_NARRATIVE_SECTIONS,
)
from pipeline.section_pinner import PinResult


class StubPinner:
    """Records exactly what it was asked to embed."""

    def __init__(self, by_text=None):
        self.calls: list[str] = []
        self.by_text = by_text or {}

    def pin(self, text: str) -> PinResult:
        self.calls.append(text)
        for needle, ids in self.by_text.items():
            if needle in text:
                return PinResult(section_ids=list(ids),
                                 matched_rules=["dense sim=0.90"] * len(ids))
        return PinResult()


def _chunk(text, **kw):
    base = {"text": text, "doc_type": "FIR", "chunk_role": "incident",
            "document_id": "d1", "metadata": {}}
    base.update(kw)
    return base


def test_pins_each_chunk_separately():
    """BUGFIX under test: this used to join every chunk into ONE string and
    embed it once. bge-large truncates at 512 tokens silently, so for any
    real FIR + charge sheet the pin decision came from roughly the first
    2 KB and the rest of the case was discarded."""
    chunks = [_chunk("a" * 3000), _chunk("b" * 3000), _chunk("c" * 3000)]
    pinner = StubPinner()
    _sections_from_case_narrative(chunks, pinner)

    assert len(pinner.calls) == 3, "each chunk must be embedded on its own"
    assert all(len(c) == 3000 for c in pinner.calls)


def test_sections_from_every_chunk_are_merged():
    chunks = [_chunk("the accused stabbed the victim"),
              _chunk("a bribe was demanded by the officer")]
    pinner = StubPinner({"stabbed": ["IPC_302"], "bribe": ["PCA_007"]})
    got = _sections_from_case_narrative(chunks, pinner)
    assert got == ["IPC_302", "PCA_007"]


def test_late_chunk_contributions_are_not_lost():
    """The section only present in the LAST chunk is exactly what a single
    truncated embedding used to drop."""
    chunks = [_chunk("x" * 4000)] * 3 + [_chunk("a bribe was demanded")]
    pinner = StubPinner({"bribe": ["PCA_007"]})
    assert "PCA_007" in _sections_from_case_narrative(chunks, pinner)


def test_output_is_capped():
    """Every nominated section becomes a reserved slot downstream, so an
    uncapped list crowds the reranker out of the answer context."""
    chunks = [_chunk(f"chunk {i} stabbed") for i in range(30)]
    pinner = StubPinner({"stabbed": [f"IPC_{i:03d}" for i in range(30)]})
    got = _sections_from_case_narrative(chunks, pinner)
    assert len(got) <= MAX_NARRATIVE_SECTIONS


def test_deduplicates_across_chunks():
    chunks = [_chunk("stabbed once"), _chunk("stabbed twice")]
    pinner = StubPinner({"stabbed": ["IPC_302"]})
    assert _sections_from_case_narrative(chunks, pinner) == ["IPC_302"]


def test_empty_chunks():
    assert _sections_from_case_narrative([], StubPinner()) == []
    assert _sections_from_case_narrative([_chunk("  ")], StubPinner()) == []


class TestCitedSections:
    def test_entity_citations_are_normalised(self, known_ids):
        """Phase 1 records these as "103 BNS"; section_id is "BNS_103".
        A low-numbered or lettered cite ("5 BNS", "120b IPC") needs
        zero-padding and uppercasing or fetch_by_ids drops it silently."""
        chunks = [_chunk("", metadata={"entities": {
            "sections_cited": ["103 BNS", "5 BNS", "120b IPC"]}})]
        got = _sections_from_case_chunks(chunks, known_ids)
        assert "BNS_103" in got
        assert "BNS_005" in got
        assert "IPC_120B" in got

    def test_ids_in_chunk_text_are_found(self, known_ids):
        chunks = [_chunk("registered under IPC_302 and section 420 of the IPC")]
        got = _sections_from_case_chunks(chunks, known_ids)
        assert {"IPC_302", "IPC_420"} <= set(got)

    def test_unknown_sections_are_dropped(self, known_ids):
        chunks = [_chunk("", metadata={"entities": {"sections_cited": ["9999 IPC"]}})]
        assert _sections_from_case_chunks(chunks, known_ids) == []

    def test_order_is_preserved_without_duplicates(self, known_ids):
        chunks = [_chunk("", metadata={"entities": {
            "sections_cited": ["302 IPC", "420 IPC", "302 IPC"]}})]
        assert _sections_from_case_chunks(chunks, known_ids) == ["IPC_302", "IPC_420"]
