"""The BM25 decomposition: document side x query side must equal Okapi BM25."""
import math

import pytest

from data.bm25 import bm25_document_weights, compute_idf, build_vocab
from data.bm25_tokenizer import tokenize, build_search_text
from config import BM25_K1, BM25_B


def _okapi(query_tokens, doc_tokens, idf_by_token, avgdl, k1=BM25_K1, b=BM25_B):
    """Textbook Okapi BM25, written independently of the implementation."""
    dl = len(doc_tokens)
    tf = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    score = 0.0
    for t in set(query_tokens):
        f = tf.get(t, 0)
        if f:
            score += idf_by_token[t] * (f * (k1 + 1)) / (
                f + k1 * (1 - b + b * dl / avgdl)
            )
    return score


def test_sparse_dot_product_equals_okapi_bm25(records):
    """The whole point of the sparse decomposition. The previous
    implementation stored tf*idf on BOTH sides, which applied IDF twice
    and had no length normalization at all."""
    from pipeline.hybrid_retriever import BM25_K3

    sample = records[:400]
    corpus = [tokenize(build_search_text(r)) for r in sample]
    vocab  = build_vocab(corpus)
    idf    = compute_idf(corpus, vocab)
    avgdl  = sum(len(t) for t in corpus) / len(corpus)
    idf_by_token = {tok: idf[i] for tok, i in vocab.items()}

    query = "punishment for murder with intention to cause death"
    qtokens = [t for t in tokenize(query) if t in vocab]

    # Query side, exactly as HybridRetriever builds it, with qtf = 1 for
    # each distinct term so it reduces to plain idf.
    qw = {}
    for t in qtokens:
        idx = vocab[t]
        qw[idx] = qw.get(idx, 0.0) + 0.0
    qtf = {}
    for t in qtokens:
        qtf[t] = qtf.get(t, 0) + 1
    query_weights = {
        vocab[t]: idf_by_token[t] * (f * (BM25_K3 + 1.0)) / (BM25_K3 + f)
        for t, f in qtf.items()
    }

    for tokens in corpus[:50]:
        doc_weights = bm25_document_weights(tokens, vocab, avgdl)
        dot = sum(w * doc_weights.get(i, 0.0) for i, w in query_weights.items())
        # With every query term appearing once, the k3 factor is 1.0 and
        # the dot product must equal textbook BM25 to floating-point noise.
        expected = _okapi(qtokens, tokens, idf_by_token, avgdl)
        assert dot == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_document_weights_are_length_normalized(records):
    """A long document must not out-score a short one purely on length.
    Before the fix, corpus-vector L2 norm correlated 0.81 with length."""
    corpus = [tokenize(build_search_text(r)) for r in records[:400]]
    vocab  = build_vocab(corpus)
    avgdl  = sum(len(t) for t in corpus) / len(corpus)

    short = ["murder"] * 2 + ["fine"]
    long_ = ["murder"] * 2 + ["fine"] + ["filler"] * 200
    v = {"murder": 0, "fine": 1, "filler": 2}

    w_short = bm25_document_weights(short, v, avgdl)
    w_long  = bm25_document_weights(long_, v, avgdl)
    assert w_short[0] > w_long[0], (
        "the same term frequency in a much longer document must score lower"
    )


def test_idf_is_near_zero_for_ubiquitous_terms(records):
    corpus = [tokenize(build_search_text(r)) for r in records]
    vocab  = build_vocab(corpus)
    idf    = compute_idf(corpus, vocab)
    by_token = {tok: idf[i] for tok, i in vocab.items()}
    assert by_token["section"] < 1.0
    assert all(v >= 0 for v in idf.values()), "Okapi IDF must never go negative"
    rare = [t for t in ("dacoity", "pocso", "abetment") if t in by_token]
    for t in rare:
        assert by_token[t] > by_token["section"] * 3
