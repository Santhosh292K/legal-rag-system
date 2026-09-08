"""ALEA evidence-to-element scoring."""
import numpy as np
import pytest

from pipeline.alea import (
    ALEA, EvidenceFact, entities_to_facts, SIM_THRESHOLD,
    SOURCE_WEIGHTS, DEFAULT_WEIGHT, _band,
)


class CountingEmbedder:
    """Deterministic hashed embeddings, and counts how much it was asked
    to embed."""

    def __init__(self):
        self.calls = 0
        self.texts = 0

    def __call__(self, texts):
        self.calls += 1
        self.texts += len(texts)
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=32)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


@pytest.fixture
def embedder():
    return CountingEmbedder()


def _facts():
    return [
        EvidenceFact(text="weapon used: knife", fact_type="weapon",
                     source_doc_type="Medical Report"),
        EvidenceFact(text="injury: stab wound", fact_type="injury",
                     source_doc_type="FIR"),
    ]


def test_ontology_is_embedded_once_across_calls(embedder):
    """These descriptions are static. They used to be re-embedded inside
    the per-section loop, on every query."""
    alea = ALEA(embed_fn=embedder)
    ids = list(alea.ontology)[:3]
    alea.score_sections(_facts(), candidate_section_ids=ids)
    after_first = embedder.texts
    alea.score_sections(_facts(), candidate_section_ids=ids)
    second_call_texts = embedder.texts - after_first
    assert second_call_texts == len(_facts()), (
        "only the evidence facts should need embedding on a repeat call"
    )


def test_source_weight_ordering():
    """Forensic/medical evidence must outweigh an FIR narrative."""
    assert SOURCE_WEIGHTS["Forensic Report"] > SOURCE_WEIGHTS["FIR"]
    assert SOURCE_WEIGHTS["FIR"] > DEFAULT_WEIGHT
    assert EvidenceFact("x", "t", "Unknown Type").weight == DEFAULT_WEIGHT


def test_no_facts_returns_nothing(embedder):
    assert ALEA(embed_fn=embedder).score_sections([]) == []


def test_unknown_sections_are_skipped(embedder):
    alea = ALEA(embed_fn=embedder)
    assert alea.score_sections(_facts(), candidate_section_ids=["NOPE_001"]) == []


def test_scores_are_bounded_and_banded(embedder):
    alea = ALEA(embed_fn=embedder)
    for score in alea.score_sections(_facts(), candidate_section_ids=list(alea.ontology)[:5]):
        assert 0.0 <= score.coverage <= 1.0
        assert 0.0 <= score.applicability <= 1.0
        assert score.band in ("Strong", "Partial", "Weak", "Missing")
        assert score.band == _band(score.coverage)


def test_results_are_sorted_by_applicability(embedder):
    alea = ALEA(embed_fn=embedder)
    scores = alea.score_sections(_facts(), candidate_section_ids=list(alea.ontology))
    assert scores == sorted(scores, key=lambda s: s.applicability, reverse=True)


def test_retrieval_prior_scales_applicability(embedder):
    alea = ALEA(embed_fn=embedder)
    sid = list(alea.ontology)[0]
    plain = alea.score_sections(_facts(), candidate_section_ids=[sid])[0]
    scaled = alea.score_sections(_facts(), candidate_section_ids=[sid],
                                 retrieval_priors={sid: 0.5})[0]
    assert scaled.applicability == pytest.approx(plain.coverage * 0.5)


def test_entities_to_facts_covers_every_entity_kind():
    entities = {"weapons": ["knife"], "injuries": ["stab wound"],
                "amounts": ["Rs. 50,000"], "complainants": ["A"], "accused": ["B"]}
    facts = entities_to_facts(entities, doc_type="FIR", document_id="d1")
    kinds = {f.fact_type for f in facts}
    assert kinds == {"weapon", "injury", "amount", "party"}
    assert all(f.source_document_id == "d1" for f in facts)


def test_band_boundaries():
    assert _band(0.0) == "Missing"
    assert _band(0.2) == "Weak"
    assert _band(0.40) == "Partial"
    assert _band(0.75) == "Strong"
