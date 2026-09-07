"""Which sections reach the generator's prompt.

Three channels can inject a section the reranker didn't rank highly
(pinner, Rocchio, KG). A positional slice dropped all of them; reserving
slots without a cap let them evict every reranked result instead.
"""
import pytest

from config import FINAL_TOP_K
from pipeline.answer_generator import AnswerGenerator
from pipeline.chunk_structurer import StructuredChunk
from pipeline.irac_reranker import RankedChunk
from pipeline.section_pinner import (
    PIN_EXPLANATION, ROCCHIO_EXPLANATION, KG_EXPLANATION_PREFIX,
)


def _chunk(sid: str) -> StructuredChunk:
    return StructuredChunk(
        section_id=sid, content=f"content of {sid}", act_name="Indian Penal Code, 1860",
        act_code="IPC", chapter="", category="", validity_label="active", warning="",
        penalized_score=0.0, rule_summary="", issue_tags=[], conclusion_type="penal",
    )


def _ranked(sid: str, score: float, explanation: str = "") -> RankedChunk:
    return RankedChunk(chunk=_chunk(sid), final_score=score, irac_score=score,
                       explanation=explanation)


def test_pinned_sections_are_not_dropped_by_position():
    """main.py appends re-injected pins to the END of `ranked`."""
    ranked = [_ranked(f"IPC_{i:03d}", 0.9) for i in range(FINAL_TOP_K)]
    ranked.append(_ranked("IPC_498A", 0.2, PIN_EXPLANATION))
    selected = {r.chunk.section_id for r in AnswerGenerator._select_top(ranked, FINAL_TOP_K)}
    assert "IPC_498A" in selected


def test_rescue_channels_are_not_dropped_by_score():
    """Rocchio and KG additions carry a flat discounted score that never
    beats real IRAC scores, so they lost their slot right after winning it."""
    ranked = [_ranked(f"IPC_{i:03d}", 0.9) for i in range(FINAL_TOP_K)]
    ranked.append(_ranked("IPC_505", 0.35, ROCCHIO_EXPLANATION))
    ranked.append(_ranked("BNS_103", 0.30, f"{KG_EXPLANATION_PREFIX} (supersession)"))
    selected = {r.chunk.section_id for r in AnswerGenerator._select_top(ranked, FINAL_TOP_K)}
    assert {"IPC_505", "BNS_103"} <= selected


def test_reserved_channels_cannot_take_the_whole_context():
    """The failure mode this cap exists for: pinner (6) + Rocchio (3) +
    KG (4) = 13 reserved against FINAL_TOP_K=10, leaving zero slots for
    anything the reranker actually scored. The case-fusion path could
    nominate 20+."""
    ranked = [_ranked(f"IPC_{i:03d}", 0.95) for i in range(FINAL_TOP_K)]
    for i in range(20):
        ranked.append(_ranked(f"PIN_{i:03d}", 0.10, PIN_EXPLANATION))

    selected = AnswerGenerator._select_top(ranked, FINAL_TOP_K)
    assert len(selected) == FINAL_TOP_K
    reserved = [r for r in selected if r.explanation == PIN_EXPLANATION]
    assert len(reserved) <= FINAL_TOP_K // 2
    ranked_kept = [r for r in selected if r.explanation == ""]
    assert len(ranked_kept) >= FINAL_TOP_K // 2, (
        "the reranker's own results must keep at least half the context"
    )


def test_reserved_slots_are_ranked_among_themselves():
    ranked = [_ranked(f"IPC_{i:03d}", 0.95) for i in range(FINAL_TOP_K)]
    ranked.append(_ranked("WEAK", 0.10, PIN_EXPLANATION))
    ranked.append(_ranked("STRONG", 0.80, PIN_EXPLANATION))
    selected = AnswerGenerator._select_top(ranked, 4)
    pins = [r.chunk.section_id for r in selected if r.explanation == PIN_EXPLANATION]
    assert pins[0] == "STRONG"


def test_budget_is_filled_when_ranked_pool_is_small():
    ranked = [_ranked("IPC_001", 0.9)]
    ranked += [_ranked(f"PIN_{i}", 0.2, PIN_EXPLANATION) for i in range(5)]
    selected = AnswerGenerator._select_top(ranked, FINAL_TOP_K)
    assert len(selected) == 6, "must not return a short context when candidates exist"


def test_ordinary_ranking_is_by_score():
    ranked = [_ranked("LOW", 0.1), _ranked("HIGH", 0.9), _ranked("MID", 0.5)]
    got = [r.chunk.section_id for r in AnswerGenerator._select_top(ranked, 3)]
    assert got == ["HIGH", "MID", "LOW"]


def test_no_duplicates_in_selection():
    ranked = [_ranked("IPC_001", 0.9, PIN_EXPLANATION), _ranked("IPC_002", 0.5)]
    selected = AnswerGenerator._select_top(ranked, FINAL_TOP_K)
    assert len(selected) == len({id(r) for r in selected})
