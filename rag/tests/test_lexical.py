"""IDF-weighted overlap must not punish short fields the way Jaccard did."""
import pytest

from pipeline.lexical import LexicalScorer, STOP_WORDS


@pytest.fixture(scope="module")
def scorer():
    return LexicalScorer.shared()


QUERY = ("wrongful arrest public servant causing hurt a police officer "
         "arrested Ravi without warrant and beat him in custody")


def test_short_relevant_field_beats_unrelated_text(scorer):
    """Jaccard scored a short but perfectly relevant field near zero purely
    because the query around it is longer — the measured median over a
    realistic candidate set was 0.000, which made the reranker's Stage-1
    ordering mostly ties resolved by retrieval order."""
    relevant  = scorer.overlap(QUERY, "wrongful arrest")
    unrelated = scorer.overlap(QUERY, "registration of a sale deed")
    assert relevant > 0.2
    assert relevant > unrelated * 5


def test_stemming_bridges_morphological_variants(scorer):
    """"wrongful confinement" (query) vs "wrongfully confines" (statute)
    share no surface token at all."""
    assert scorer.overlap(QUERY, "Whoever wrongfully confines any person") > 0.0


def test_unrelated_text_scores_zero(scorer):
    query = "punishment for murder with intention to cause death"
    other = "registration of a sale deed for immovable property"
    assert scorer.overlap(query, other) < 0.2


def test_rare_terms_outweigh_common_ones(scorer):
    """Matching a rare, diagnostic term must count for more than matching a
    common one. The overlap coefficient |A∩B|/min(|A|,|B|) cannot express
    this — it saturates at 1.0 for BOTH whenever the field is a subset of
    the query — which is why this uses IDF-weighted cosine instead."""
    query = "dacoity committed by five or more persons"
    assert scorer.overlap(query, "dacoity") > scorer.overlap(query, "persons")


def test_symmetry(scorer):
    a, b = "criminal breach of trust by a servant", "breach of trust servant"
    assert scorer.overlap(a, b) == pytest.approx(scorer.overlap(b, a))


def test_bounded(scorer):
    for a, b in [("murder", "murder"), ("", "murder"), ("murder", ""),
                 ("the of and", "the of and")]:
        assert 0.0 <= scorer.overlap(a, b) <= 1.0


def test_coverage_is_asymmetric(scorer):
    needle, haystack = "dowry death", "cruelty dowry death by husband or relative"
    assert scorer.coverage(needle, haystack) > scorer.coverage(haystack, needle)


def test_stopwords_are_excluded(scorer):
    assert scorer.informative_tokens("the act of the section") == set()
    assert "section" in STOP_WORDS and "act" in STOP_WORDS


def test_stopwords_are_stored_stemmed():
    """An unstemmed stop list stops filtering the moment the tokenizer
    stems — "any" becomes "ani" and sails through."""
    from data.bm25_tokenizer import tokenize
    for word in STOP_WORDS:
        assert tokenize(word) == [word], f"{word!r} is not in stemmed form"
