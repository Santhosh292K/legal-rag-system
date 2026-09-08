"""Reciprocal Rank Fusion, including the cross-variant weighting."""
import pytest

from pipeline.hybrid_retriever import RetrievedChunk, reciprocal_rank_fusion


def _chunk(sid):
    return RetrievedChunk(section_id=sid, content="", score=0.0)


def _ids(merged):
    return [c.section_id for c in merged]


def test_agreement_between_lists_wins():
    dense  = [_chunk("A"), _chunk("B"), _chunk("C")]
    sparse = [_chunk("C"), _chunk("A"), _chunk("D")]
    assert _ids(reciprocal_rank_fusion([dense, sparse]))[0] == "A"


def test_scores_are_written_back_onto_chunks():
    merged = reciprocal_rank_fusion([[_chunk("A")]])
    assert merged[0].rrf_score > 0


def test_weights_scale_a_list_contribution():
    """Cross-variant fusion decays later paraphrases. Without weighting,
    a section found by ten near-duplicate expansions outranks one found by
    a single distinctive query — that measures how redundantly the query
    was expanded, not retriever agreement."""
    primary  = [_chunk("PRIMARY")]
    variants = [[_chunk("ECHO")] for _ in range(5)]

    unweighted = _ids(reciprocal_rank_fusion([primary] + variants))
    assert unweighted[0] == "ECHO", "sanity: unweighted RRF rewards repetition"

    weights = [1.0] + [0.1] * 5
    weighted = _ids(reciprocal_rank_fusion([primary] + variants, weights=weights))
    assert weighted[0] == "PRIMARY"


def test_weight_length_is_validated():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[_chunk("A")], [_chunk("B")]], weights=[1.0])


def test_k_controls_rank_discrimination():
    """k=60 leaves almost no signal across a 25-item list; k=20 is why the
    default was lowered.

    Note the fresh chunk objects per call: reciprocal_rank_fusion writes
    rrf_score onto the chunks it is given, so reusing them across two
    calls would have the second silently overwrite the first's scores.
    """
    spread = lambda m: m[0].rrf_score - m[-1].rrf_score
    wide   = reciprocal_rank_fusion([[_chunk(f"S{i}") for i in range(10)]], k=60)
    narrow = reciprocal_rank_fusion([[_chunk(f"S{i}") for i in range(10)]], k=20)
    assert spread(narrow) > spread(wide) * 2


def test_scores_are_written_onto_the_input_chunks():
    """Documented, relied-upon side effect: TemporalFilter reads
    chunk.rrf_score to compute penalized_score. It also means a second
    fusion over the same objects overwrites the first — which is what
    retrieve() wants (the cross-variant score is the final one), but is a
    trap for any caller comparing two fusions."""
    chunks = [_chunk("A"), _chunk("B")]
    merged = reciprocal_rank_fusion([chunks])
    assert merged[0] is chunks[0]
    assert chunks[0].rrf_score > 0


def test_empty_and_single_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    assert _ids(reciprocal_rank_fusion([[_chunk("A")]])) == ["A"]


def test_first_occurrence_of_a_chunk_is_kept():
    first, second = _chunk("A"), _chunk("A")
    first.content = "from the priority list"
    merged = reciprocal_rank_fusion([[first], [second]])
    assert merged[0].content == "from the priority list"
