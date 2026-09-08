"""Live retrieval against the real Qdrant index and the real embedding model.

Skipped automatically when either is unavailable, so the default `pytest
tests/` run stays model-free and fast. Run it after any re-index:

    pytest tests/test_integration_retrieval.py -v

These are the queries that exposed the shipped vocabulary mismatch. With
query-side and corpus-side vocabularies in different index spaces,
"anticipatory bail application" returned a postal-delivery section and
"what is the punishment for murder" returned IPC_155.
"""
import sys
from pathlib import Path

import pytest

RAG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_ROOT))

from config import QDRANT_PATH, EMBEDDING_MODEL, COLLECTION_NAME
from data.index_manifest import read_manifest

pytestmark = pytest.mark.integration


def _index_ready() -> bool:
    return (read_manifest(RAG_ROOT / "data") is not None
            and (RAG_ROOT / "qdrant_db").exists())


@pytest.fixture(scope="module")
def retriever():
    if not _index_ready():
        pytest.skip("no built index — run: python data/indexer.py data/final_dataset.json")
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from pipeline.hybrid_retriever import HybridRetriever
    except ImportError as e:                                  # pragma: no cover
        pytest.skip(f"retrieval dependencies unavailable: {e}")

    client = QdrantClient(path=QDRANT_PATH)
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:                                    # pragma: no cover
        client.close()
        pytest.skip(f"embedding model unavailable offline: {e}")
    r = HybridRetriever(client=client, embed_model=model)
    yield r
    client.close()


def _ids(chunks):
    return [c.section_id for c in chunks]


@pytest.mark.parametrize("query,gold", [
    ("anticipatory bail application",                 "CRPC_438"),
    ("what is the punishment for murder",             "IPC_302"),
    ("punishment for hacking under the IT Act",       "ITA_066"),
    ("criminal breach of trust by an employee",       "IPC_408"),
])
def test_obvious_query_ranks_its_section_first(retriever, query, gold):
    assert _ids(retriever.retrieve([query], top_k=5))[0] == gold


@pytest.mark.parametrize("query,gold", [
    ("What does IPC 302 say?",       "IPC_302"),
    ("section 420 of the IPC",       "IPC_420"),
    ("u/s 138 and IPC 494",          "IPC_494"),
])
def test_explicitly_named_sections_are_pinned_first(retriever, query, gold):
    """The direct-lookup fast path must win even when BM25 and dense both
    miss — "section 420 of the IPC" is a phrasing neither retriever
    handles, and the old act-first-only regex didn't parse it at all."""
    assert _ids(retriever.retrieve([query], top_k=5))[0] == gold


def test_guessed_sections_are_not_treated_as_named(retriever):
    """Stage 0b's synonym rules emit section numbers as recall hints;
    scanning them for "explicit references" pinned theft provisions above
    every IT Act section for a hacking query."""
    from pipeline.universal_translator import quick_expand
    query    = "A hacker stole my OTP and drained my account"
    variants = [query, quick_expand(query)]
    top = _ids(retriever.retrieve(variants, top_k=6, direct_lookup_query=query))
    assert top[0].startswith("ITA_"), top
    assert "IPC_378" not in top[:4], "theft sections pinned from a regex guess"


def test_sparse_and_dense_both_contribute(retriever):
    query  = "punishment for criminal conspiracy"
    sparse = set(_ids(retriever._sparse_retrieve(query, top_k=10)))
    dense  = set(_ids(retriever._dense_retrieve(query,  top_k=10)))
    assert sparse and dense
    assert sparse != dense, "the two retrievers should not be interchangeable"


def test_no_query_returns_an_empty_pool(retriever):
    for query in ["punishment for murder", "eviction of a tenant",
                  "bail", "what is a contract"]:
        assert retriever.retrieve([query], top_k=10), f"empty pool for {query!r}"


def test_section_store_matches_the_collection(retriever):
    manifest = read_manifest(RAG_ROOT / "data")
    assert len(retriever.sections) == manifest["n_docs"]


def test_startup_guard_rejects_a_tampered_vocabulary(retriever):
    """The check that would have caught the shipped mismatch."""
    from data.index_manifest import IndexMismatchError
    from pipeline.hybrid_retriever import HybridRetriever

    original = dict(retriever.vocab)
    try:
        with pytest.raises(IndexMismatchError):
            r = HybridRetriever.__new__(HybridRetriever)
            r.vocab = {tok: idx + 1 for tok, idx in original.items()}
            r._verify_index_consistency(RAG_ROOT / "data")
    finally:
        retriever.vocab = original
