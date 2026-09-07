"""End-to-end BM25 retrieval quality against the real benchmark.

This is the regression test the vocabulary-mismatch bug needed. It runs
the actual scoring path — same tokenizer, same vocabulary construction,
same weighting — over the whole corpus and the whole benchmark, and
asserts a floor on recall.

No models required: it measures the sparse half in isolation, which is
exactly the half that was silently returning noise.
"""
import collections
import json

import pytest

from data.bm25 import build_vocab, compute_idf, bm25_document_weights
from data.bm25_tokenizer import tokenize, build_search_text
from pipeline.hybrid_retriever import BM25_K3
from tests.conftest import RAG_ROOT

BENCHMARK_PATH = RAG_ROOT / "evaluation" / "benchmark_scenarios.json"


@pytest.fixture(scope="module")
def bm25_index(records):
    corpus = [tokenize(build_search_text(r)) for r in records]
    vocab  = build_vocab(corpus)
    idf    = compute_idf(corpus, vocab)
    avgdl  = sum(len(t) for t in corpus) / len(corpus)
    docs   = [bm25_document_weights(t, vocab, avgdl) for t in corpus]
    return {
        "ids":   [r["section"] for r in records],
        "vocab": vocab, "idf": idf, "docs": docs,
    }


def _search(index, query, k):
    vocab, idf = index["vocab"], index["idf"]
    qtf = collections.Counter(t for t in tokenize(query) if t in vocab)
    weights = {
        vocab[t]: idf[vocab[t]] * (f * (BM25_K3 + 1.0)) / (BM25_K3 + f)
        for t, f in qtf.items()
    }
    if not weights:
        return []
    scored = [
        (sum(w * doc.get(i, 0.0) for i, w in weights.items()), n)
        for n, doc in enumerate(index["docs"])
    ]
    scored.sort(reverse=True)
    return [index["ids"][n] for _s, n in scored[:k]]


@pytest.fixture(scope="module")
def benchmark():
    if not BENCHMARK_PATH.exists():
        pytest.skip("benchmark_scenarios.json not present")
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        items = json.load(f)
    return [i for i in items if i.get("gold_sections")]


@pytest.mark.parametrize("query,gold", [
    ("what is the punishment for murder",            "IPC_302"),
    ("anticipatory bail application",                "CRPC_438"),
    ("punishment for criminal breach of trust by a servant", "IPC_408"),
    ("cheating and dishonestly inducing delivery of property", "IPC_420"),
])
def test_obvious_queries_rank_their_obvious_section_first(bm25_index, query, gold):
    """These are the smoke tests that would have failed loudly on the
    shipped index, where "anticipatory bail application" returned a postal
    delivery section and "punishment for murder" returned IPC_155."""
    top = _search(bm25_index, query, 5)
    assert gold in top, f"{gold} missing from top-5: {top}"


def test_recall_floor_on_the_benchmark(bm25_index, benchmark):
    """BM25 alone, no dense, no reranking. The floor is deliberately below
    the current measurement so ordinary tuning doesn't trip it, but far
    above what a vocabulary mismatch produces."""
    recall_at_10, recall_at_50, reciprocal_ranks = [], [], []
    for item in benchmark:
        gold = set(item["gold_sections"])
        ranked = _search(bm25_index, item["query"], 50)
        recall_at_10.append(len(gold & set(ranked[:10])) / len(gold))
        recall_at_50.append(len(gold & set(ranked)) / len(gold))
        rr = 0.0
        for rank, sid in enumerate(ranked, 1):
            if sid in gold:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    r10 = sum(recall_at_10) / len(recall_at_10)
    r50 = sum(recall_at_50) / len(recall_at_50)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"\nBM25-only: recall@10={r10:.3f} recall@50={r50:.3f} MRR={mrr:.3f} "
          f"over {len(benchmark)} queries")

    assert r10 > 0.18, f"recall@10 collapsed to {r10:.3f}"
    assert r50 > 0.28, f"recall@50 collapsed to {r50:.3f}"
    assert mrr > 0.18, f"MRR collapsed to {mrr:.3f}"


def test_every_query_retrieves_something(bm25_index, benchmark):
    """A query that tokenizes to nothing in the vocabulary returns an empty
    list — the symptom of a tokenizer/vocabulary mismatch."""
    empty = [i["query"] for i in benchmark if not _search(bm25_index, i["query"], 5)]
    assert empty == [], f"{len(empty)} queries retrieved nothing: {empty[:3]}"
