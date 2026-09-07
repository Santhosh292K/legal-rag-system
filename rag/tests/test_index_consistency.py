"""The guard that would have caught the shipped index/vocabulary mismatch.

The repository shipped with Qdrant sparse vectors built from one
tokenizer, a bm25_vocab.json built from a second, and running code
implying a third. Measured index-set agreement between the query-side
vocabulary and the stored corpus vectors was ~15%: BM25 was searching for
the wrong terms on every query, with no error anywhere.
"""
import json

import pytest

from data.index_manifest import (
    corpus_fingerprint, vocab_fingerprint, build_manifest,
    read_manifest, write_manifest, IndexMismatchError, INDEX_FORMAT_VERSION,
)
from data.bm25_tokenizer import tokenize, build_search_text, TOKENIZER_VERSION


def test_vocab_fingerprint_detects_reordered_ids():
    """Reordering ids while keeping the same tokens is exactly the failure
    that made stored vectors unreadable, and it must not hash the same."""
    a = {"murder": 0, "theft": 1}
    b = {"murder": 1, "theft": 0}
    assert vocab_fingerprint(a) != vocab_fingerprint(b)


def test_vocab_fingerprint_is_order_independent_for_identical_mappings():
    assert vocab_fingerprint({"a": 0, "b": 1}) == vocab_fingerprint({"b": 1, "a": 0})


def test_corpus_fingerprint_detects_text_definition_change():
    """Changing what build_search_text() includes must force a re-index."""
    ids = ["IPC_001", "IPC_002"]
    before = corpus_fingerprint(ids, ["act section one", "act section two"])
    after  = corpus_fingerprint(ids, ["act section one keywords", "act section two"])
    assert before != after


def test_manifest_roundtrip(tmp_path):
    manifest = build_manifest(
        section_ids=["IPC_001"], search_texts=["text"], vocab={"text": 0},
        tokenizer_version=TOKENIZER_VERSION, n_docs=1, avgdl=1.0,
        k1=1.2, b=0.75, collection_name="legal_sections",
    )
    write_manifest(manifest, tmp_path)
    assert read_manifest(tmp_path) == manifest


def test_read_manifest_missing_returns_none(tmp_path):
    assert read_manifest(tmp_path) is None


def test_mismatch_error_names_the_fix():
    err = str(IndexMismatchError("vocab drifted"))
    assert "data/indexer.py" in err, "the error must tell you how to repair it"


class TestShippedArtifacts:
    """These run against whatever is actually on disk in data/."""

    def test_manifest_exists(self, records):
        from data.index_manifest import MANIFEST_FILENAME
        from tests.conftest import RAG_ROOT
        path = RAG_ROOT / "data" / MANIFEST_FILENAME
        if not path.exists():
            pytest.skip("index not built yet — run data/indexer.py")

    def test_vocab_matches_manifest(self, records):
        from tests.conftest import RAG_ROOT
        manifest = read_manifest(RAG_ROOT / "data")
        vocab_path = RAG_ROOT / "data" / "bm25_vocab.json"
        if manifest is None or not vocab_path.exists():
            pytest.skip("index not built yet — run data/indexer.py")
        with open(vocab_path) as f:
            vocab = json.load(f)
        assert manifest["vocab_fingerprint"] == vocab_fingerprint(vocab)
        assert manifest["tokenizer_version"] == TOKENIZER_VERSION
        assert manifest["index_format_version"] == INDEX_FORMAT_VERSION

    def test_corpus_matches_manifest(self, records):
        from tests.conftest import RAG_ROOT
        manifest = read_manifest(RAG_ROOT / "data")
        if manifest is None:
            pytest.skip("index not built yet — run data/indexer.py")
        ids   = [r["section"] for r in records]
        texts = [build_search_text(r) for r in records]
        assert manifest["corpus_fingerprint"] == corpus_fingerprint(ids, texts), (
            "final_dataset.json has changed since the index was built"
        )
