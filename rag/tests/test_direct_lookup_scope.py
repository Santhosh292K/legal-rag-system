"""The "you named this section" guarantee must mean the USER named it.

Stage 0b's offline synonym rules emit literal section numbers as recall
hints: "stole" expands to "... IPC 378 IPC 379 IPC 390 IPC 392". Scanning
that expanded text for explicit references made the retriever pin those
guesses to rank 1 (weight 2.0) and the reranker floor them at 0.85, as
though the user had asked for them by name.

Measured before the fix, "A hacker stole my OTP and drained my account"
returned IPC_378/379/390/392 at ranks 1-4, above every IT Act section.
"""
import pytest

from data.section_ref import extract_section_refs
from pipeline.universal_translator import quick_expand
from pipeline.chunk_structurer import StructuredChunk
from pipeline.intent_classifier import QueryIntent
from pipeline.irac_reranker import IRACReranker


LAY_QUERY = "A hacker stole my OTP and drained my account"


def test_offline_expansion_injects_section_numbers(known_ids):
    """The premise: expansion really does put section numbers into the
    query text, and the user's own words really don't."""
    assert extract_section_refs(LAY_QUERY, known_ids) == []
    injected = extract_section_refs(quick_expand(LAY_QUERY), known_ids)
    assert injected, "expected the synonym rules to inject section numbers"
    assert "IPC_378" in injected


def _chunk(sid):
    return StructuredChunk(
        section_id=sid, content=f"content {sid}", act_name="", act_code="IPC",
        chapter="", category="", validity_label="active", warning="",
        penalized_score=0.0, rule_summary="", issue_tags=[],
        conclusion_type="penal", enriched_context=f"content {sid}",
    )


INTENT = QueryIntent(label="punitive", confidence=0.9, act_hint=None,
                     temporal="unspecified", labels=["punitive"])


def _reranker():
    return IRACReranker(llm_top_n=0, cross_encoder=None, max_workers=1,
                        use_cross_encoder=False)


def test_reranker_ignores_sections_the_user_did_not_name():
    """direct_query is the user's words; `query` is the reranking text,
    which concatenates the LLM's translation and routinely invents section
    numbers."""
    chunks = [_chunk("IPC_378"), _chunk("ITA_066")]
    out = _reranker().rerank(
        query="theft IPC 378 " + LAY_QUERY,   # translated text, with a guess
        intent=INTENT, chunks=chunks, top_k=2,
        direct_query=LAY_QUERY,               # what the user actually typed
    )
    guessed = next(rc for rc in out if rc.chunk.section_id == "IPC_378")
    assert guessed.final_score < 0.85, (
        "a section the user never named must not get the 0.85 floor"
    )


def test_reranker_still_honours_a_real_user_reference():
    chunks = [_chunk("IPC_302"), _chunk("IPC_001")]
    out = _reranker().rerank(
        query="murder punishment", intent=INTENT, chunks=chunks, top_k=2,
        direct_query="What does IPC 302 say?",
    )
    assert out[0].chunk.section_id == "IPC_302"
    assert out[0].final_score >= 0.85


def test_direct_query_defaults_to_the_reranking_query():
    """Backward compatible: callers that don't pass direct_query keep the
    old behaviour rather than silently losing the guarantee."""
    chunks = [_chunk("IPC_302"), _chunk("IPC_001")]
    out = _reranker().rerank(query="what does IPC 302 say", intent=INTENT,
                             chunks=chunks, top_k=2)
    assert out[0].chunk.section_id == "IPC_302"
