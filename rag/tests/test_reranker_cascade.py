"""The reranker's three-stage cascade, with the models stubbed out.

What matters here is the wiring: every candidate must reach the
cross-encoder (the old code only re-scored an arbitrary Stage-1 top-8),
the LLM must refine the head of an already-good ordering, the validity
penalty must survive to the final score, and an explicitly-named section
must never be truncated away.
"""
import pytest

from pipeline.chunk_structurer import StructuredChunk
from pipeline.intent_classifier import QueryIntent
from pipeline.irac_reranker import IRACReranker
from pipeline.temporal_filter import PENALTY


class StubCrossEncoder:
    """Returns a logit per pair, driven by a keyword, and records the
    batch it was asked to score."""

    def __init__(self, keyword="murder"):
        self.keyword = keyword
        self.seen_batches = []

    def predict(self, pairs, batch_size=None):
        self.seen_batches.append(list(pairs))
        return [6.0 if self.keyword in passage.lower() else -6.0
                for _query, passage in pairs]


def _chunk(sid, content, validity="active", conclusion="penal"):
    return StructuredChunk(
        section_id=sid, content=content, act_name="Indian Penal Code, 1860",
        act_code="IPC", chapter="", category="", validity_label=validity,
        warning="", penalized_score=0.5, rule_summary=content,
        issue_tags=[], conclusion_type=conclusion, enriched_context=content,
    )


def _reranker(cross_encoder):
    r = IRACReranker(llm_top_n=0, cross_encoder=cross_encoder, max_workers=1)
    return r


INTENT = QueryIntent(label="punitive", confidence=0.9, act_hint="IPC",
                     temporal="unspecified", labels=["punitive"])


def test_every_candidate_reaches_the_cross_encoder():
    ce = StubCrossEncoder()
    chunks = [_chunk(f"IPC_{i:03d}", f"filler text {i}") for i in range(30)]
    _reranker(ce).rerank("punishment for murder", INTENT, chunks, top_k=10)
    assert len(ce.seen_batches) == 1, "must be a single batched call"
    assert len(ce.seen_batches[0]) == 30, "all candidates must be scored"


def test_cross_encoder_drives_the_ordering():
    ce = StubCrossEncoder()
    chunks = [_chunk("IPC_001", "unrelated contract provision"),
              _chunk("IPC_302", "punishment for murder is death")]
    out = _reranker(ce).rerank("punishment for murder", INTENT, chunks, top_k=2)
    assert out[0].chunk.section_id == "IPC_302"


def test_cross_encoder_scores_are_squashed_into_0_1():
    """Raw predict() is an unbounded logit; (x+1)/2 let a +6 become 3.5 and
    corrupted every downstream consumer that assumes ~[0,1]."""
    ce = StubCrossEncoder()
    chunks = [_chunk("IPC_302", "punishment for murder"),
              _chunk("IPC_001", "unrelated")]
    out = _reranker(ce).rerank("murder", INTENT, chunks, top_k=2)
    for rc in out:
        assert 0.0 <= rc.cross_enc_score <= 1.0
        assert 0.0 <= rc.final_score <= 1.0


def test_validity_penalty_reaches_the_final_score():
    ce = StubCrossEncoder()
    active   = _chunk("IPC_302", "punishment for murder", validity="active")
    repealed = _chunk("IPC_303", "punishment for murder", validity="repealed")
    out = _reranker(ce).rerank("murder", INTENT, [active, repealed], top_k=2)
    by_id = {rc.chunk.section_id: rc for rc in out}
    assert by_id["IPC_303"].final_score < by_id["IPC_302"].final_score
    assert by_id["IPC_303"].final_score == pytest.approx(
        by_id["IPC_302"].final_score * PENALTY["repealed"], rel=1e-6)


def test_explicitly_named_section_survives_truncation():
    ce = StubCrossEncoder()
    chunks = [_chunk(f"IPC_{i:03d}", "punishment for murder") for i in range(20)]
    chunks.append(_chunk("IPC_497", "adultery, wholly unrelated to the query"))
    out = _reranker(ce).rerank("what does IPC 497 say", INTENT, chunks, top_k=5)
    assert "IPC_497" in {rc.chunk.section_id for rc in out}
    assert out[0].chunk.section_id == "IPC_497"


def test_named_repealed_section_is_not_scored_as_good_law():
    """A flat 0.85 floor erased the validity penalty for any section the
    user named, so a struck-down provision came back looking confident."""
    ce = StubCrossEncoder()
    chunks = [_chunk("IPC_497", "adultery", validity="repealed"),
              _chunk("IPC_302", "punishment for murder")]
    out = _reranker(ce).rerank("what does IPC 497 say", INTENT, chunks, top_k=2)
    named = next(rc for rc in out if rc.chunk.section_id == "IPC_497")
    assert named.final_score <= 0.85 * PENALTY["repealed"] + 1e-9


def test_falls_back_gracefully_without_a_cross_encoder():
    # use_cross_encoder=False must not attempt to load a model. Before this
    # existed, cross_encoder=None unconditionally meant "download
    # bge-reranker-large", so this test hung for minutes.
    r = IRACReranker(llm_top_n=0, cross_encoder=None, max_workers=1,
                     use_cross_encoder=False)
    assert r.cross_encoder is None and r.use_cross_enc is False
    chunks = [_chunk("IPC_302", "punishment for murder"),
              _chunk("IPC_001", "unrelated contract provision")]
    out = r.rerank("punishment for murder", INTENT, chunks, top_k=2)
    assert len(out) == 2
    assert all(rc.final_score >= 0 for rc in out)


def test_empty_input():
    assert _reranker(StubCrossEncoder()).rerank("q", INTENT, [], top_k=5) == []


def test_respects_top_k():
    ce = StubCrossEncoder()
    chunks = [_chunk(f"IPC_{i:03d}", "murder") for i in range(30)]
    assert len(_reranker(ce).rerank("murder", INTENT, chunks, top_k=7)) == 7
