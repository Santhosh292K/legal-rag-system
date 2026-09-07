"""Case-document chunking must not lose text or exceed the encoder window."""
import pytest

from config import GENERIC_CHUNK_SIZE, GENERIC_CHUNK_OVERLAP
from pipeline.adaptive_chunkers import get_chunker
from pipeline.adaptive_chunkers.base_chunker import (
    split_by_labels, enforce_max_size, PREAMBLE_ROLE,
)
from pipeline.adaptive_chunkers.fir_chunker import FIRChunker, LABEL_PATTERNS
from pipeline.adaptive_chunkers.generic_chunker import GenericChunker


FIR = """POLICE STATION ANDHERI, MUMBAI
F.I.R No: 0142/2025    Dated: 12/05/2025

Complainant Name: Ramesh Kumar
Accused Name: Suresh Yadav
Accused Name: Mahesh Patil
Date of Occurrence: 12/05/2024
Brief Facts: The accused attacked the complainant with a knife near Andheri
Station Road causing grievous injury. The complainant states he was charged
under section 302 previously in an unrelated matter, which is irrelevant.
Sections invoked: 103, 118 BNS
Relief Sought: Strict legal action against the accused.
"""


@pytest.fixture(scope="module")
def fir_chunks():
    return FIRChunker().chunk(FIR, document_id="d1", case_id="c1")


def test_preamble_is_kept(fir_chunks):
    """Everything before the first label — letterhead, FIR number, police
    station, date — used to be dropped from every structured document."""
    roles = [c.chunk_role for c in fir_chunks]
    assert PREAMBLE_ROLE in roles
    preamble = next(c for c in fir_chunks if c.chunk_role == PREAMBLE_ROLE)
    assert "POLICE STATION ANDHERI" in preamble.text
    assert "0142/2025" in preamble.text


def test_repeated_roles_are_all_kept(fir_chunks):
    """Two accused means two 'accused' blocks. The dict-based splitter kept
    only the first match of each role."""
    accused = [c for c in fir_chunks if c.chunk_role == "accused"]
    assert len(accused) == 2
    joined = " ".join(c.text for c in accused)
    assert "Suresh Yadav" in joined and "Mahesh Patil" in joined


def test_narrative_mention_does_not_hijack_the_split(fir_chunks):
    """A bare 'under section' pattern used to match inside the narrative,
    so the first prose mention became the split point and the real
    'Sections invoked' block collapsed into it."""
    sections = [c for c in fir_chunks if c.chunk_role == "sections"]
    assert len(sections) == 1
    assert "103, 118 BNS" in sections[0].text
    incident = next(c for c in fir_chunks if c.chunk_role == "incident")
    assert "knife" in incident.text


def test_no_text_is_lost(fir_chunks):
    joined = " ".join(c.text for c in fir_chunks)
    for fragment in ["POLICE STATION ANDHERI", "Ramesh Kumar", "Mahesh Patil",
                     "knife", "103, 118 BNS", "Strict legal action"]:
        assert fragment in joined, f"lost: {fragment!r}"


def test_unlabelled_document_falls_back_to_generic():
    text = "\n\n".join(f"Free-form paragraph number {i}." for i in range(5))
    chunks = FIRChunker().chunk(text, document_id="d", case_id="c")
    assert chunks
    assert all(c.chunk_role == "narrative" for c in chunks)
    assert all(c.doc_type == "FIR" for c in chunks)


def test_chunk_ids_are_unique(fir_chunks):
    ids = [c.chunk_id for c in fir_chunks]
    assert len(ids) == len(set(ids))


class TestSizeGuard:
    def test_long_section_is_windowed(self):
        long_body = "x" * (GENERIC_CHUNK_SIZE * 3)
        out = enforce_max_size([("incident", long_body)])
        assert len(out) > 1
        assert all(len(body) <= GENERIC_CHUNK_SIZE for _role, body in out)
        assert all(role == "incident" for role, _ in out)

    def test_structural_chunks_are_size_capped(self):
        text = "Brief Facts: " + ("word " * 1000)
        chunks = FIRChunker().chunk(text, document_id="d", case_id="c")
        assert all(len(c.text) <= GENERIC_CHUNK_SIZE for c in chunks)

    def test_short_sections_are_untouched(self):
        assert enforce_max_size([("a", "short")]) == [("a", "short")]


class TestGenericChunker:
    def test_consecutive_windows_overlap(self):
        paras = "\n\n".join(
            " ".join(f"p{i}-{w}" for w in "alpha bravo charlie delta echo".split()) * 5
            for i in range(6)
        )
        windows = GenericChunker()._windows(paras)
        assert len(windows) > 1
        for a, b in zip(windows, windows[1:]):
            tail = a[-GENERIC_CHUNK_OVERLAP:]
            assert any(b.startswith(tail[-n:]) for n in range(1, len(tail) + 1)), (
                "consecutive windows share no text — a fact spanning a "
                "paragraph boundary would be split with nothing bridging it"
            )

    def test_respects_size_including_separators(self):
        paras = "\n\n".join("y" * 200 for _ in range(20))
        for w in GenericChunker()._windows(paras):
            assert len(w) <= GENERIC_CHUNK_SIZE

    def test_empty_input(self):
        assert GenericChunker().chunk("   ", document_id="d", case_id="c") == []

    def test_single_oversized_paragraph(self):
        chunks = GenericChunker().chunk("z" * 3000, document_id="d", case_id="c")
        assert len(chunks) > 1
        assert all(len(c.text) <= GENERIC_CHUNK_SIZE for c in chunks)


def test_registry_tags_unknown_types_with_their_own_doc_type():
    chunker = get_chunker("Witness Statement")
    chunks = chunker.chunk("A statement.", document_id="d", case_id="c")
    assert chunks and chunks[0].doc_type == "Witness Statement"


def test_split_by_labels_returns_document_order():
    sections = split_by_labels(FIR, LABEL_PATTERNS)
    positions = [FIR.index(body.split("\n")[0]) for _role, body in sections]
    assert positions == sorted(positions)
