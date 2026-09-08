"""Confidence must actually discriminate.

results_full.json (17 queries) landed 16 in "medium" and 1 in "low", with
zero "high" — a signal that never fired its top band. Part of that was a
scoring bug: the cross-encoder's raw logit was mapped with (x+1)/2 instead
of a sigmoid, so scores landed outside [0,1] and the thresholds here were
being compared against a corrupted scale. These tests assert every band is
reachable with realistic inputs.
"""
import pytest

from pipeline.answer_generator import AnswerGenerator, Citation, LOW_RELEVANCE_THRESHOLD
from pipeline.chunk_structurer import StructuredChunk
from pipeline.irac_reranker import RankedChunk


def _ranked(sid, score):
    chunk = StructuredChunk(
        section_id=sid, content="c", act_name="Indian Penal Code, 1860",
        act_code="IPC", chapter="", category="", validity_label="active",
        warning="", penalized_score=0.0, rule_summary="", issue_tags=[],
        conclusion_type="penal",
    )
    return RankedChunk(chunk=chunk, final_score=score, irac_score=score)


def _citations(*sids):
    return [Citation(section_id=s, act_name="", category="", content="",
                     validity="active", warning="") for s in sids]


def test_high_band_is_reachable():
    """Strong retrieval AND most of the offered context cited."""
    ranked = [_ranked(f"IPC_{i:03d}", 0.80) for i in range(4)]
    verdict = AnswerGenerator()._assess_confidence(
        ranked, top_k=4, citations=_citations("IPC_000", "IPC_001", "IPC_002"))
    assert verdict == "high"


def test_low_band_is_reachable():
    ranked = [_ranked(f"IPC_{i:03d}", 0.05) for i in range(4)]
    verdict = AnswerGenerator()._assess_confidence(ranked, top_k=4, citations=[])
    assert verdict == "low"


def test_medium_band_is_reachable():
    ranked = [_ranked(f"IPC_{i:03d}", 0.50) for i in range(4)]
    verdict = AnswerGenerator()._assess_confidence(
        ranked, top_k=4, citations=_citations("IPC_000"))
    assert verdict == "medium"


def test_citing_one_section_is_not_enough_for_medium():
    """The old test was `avg >= 0.40 or (citations and len(citations) > 0)`.
    The prompt tells the model to cite everything, so the second disjunct
    was true for essentially every query."""
    ranked = [_ranked(f"IPC_{i:03d}", 0.05) for i in range(10)]
    verdict = AnswerGenerator()._assess_confidence(
        ranked, top_k=10, citations=_citations("IPC_000"))
    assert verdict == "low"


def test_strong_answer_is_not_dragged_down_by_padding():
    """A narrow query has few genuinely relevant sections; the rest of the
    context is filler the model correctly ignores. Averaging over the
    filler used to report "low" for a right answer."""
    ranked = [_ranked("IPC_498A", 0.80), _ranked("IPC_304B", 0.75),
              _ranked("BNS_085", 0.70)]
    ranked += [_ranked(f"PAD_{i:03d}", 0.05) for i in range(7)]
    verdict = AnswerGenerator()._assess_confidence(
        ranked, top_k=10, citations=_citations("IPC_498A", "IPC_304B", "BNS_085"))
    assert verdict in ("high", "medium")


def test_bands_are_ordered_by_score():
    seen = []
    for score in (0.05, 0.50, 0.85):
        ranked = [_ranked(f"IPC_{i:03d}", score) for i in range(4)]
        seen.append(AnswerGenerator()._assess_confidence(
            ranked, top_k=4, citations=_citations("IPC_000", "IPC_001", "IPC_002")))
    assert seen == ["low", "medium", "high"], seen


def test_empty_ranked_is_low():
    assert AnswerGenerator()._assess_confidence([], top_k=10, citations=[]) == "low"


def test_threshold_constant_is_shared():
    """generate() picks the cautious prompt off the same number the medium
    band uses, so the two cannot drift apart."""
    assert 0.0 < LOW_RELEVANCE_THRESHOLD < 0.65
