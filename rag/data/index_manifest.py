"""
data/index_manifest.py

The contract between what data/indexer.py WROTE into Qdrant and what
pipeline/hybrid_retriever.py READS back at query time.

Why this module exists
----------------------
BM25 sparse vectors are keyed by integer token ids that come from a
vocabulary built at index time. Nothing about a sparse vector is
self-describing: a vector whose indices mean {0: "murder", 1: "theft"}
is structurally identical to one meaning {0: "the", 1: "of"}. If the
vocabulary is rebuilt without re-writing the stored vectors, queries and
corpus silently end up in two different index spaces and retrieval
degrades to noise — with no error, no warning, and no failing test.

That is not hypothetical; it is the state this repository shipped in.
The stored vectors were built from one tokenizer/text definition, the
vocabulary file on disk from a second, and the running code implied a
third. Measured index-set agreement between the query-side vocabulary
and the stored corpus vectors was ~15%.

The fix is to make the coupling explicit and checked:

  * indexer.py records a manifest describing exactly what it built —
    tokenizer version, the text definition, the vocabulary, and the
    corpus — as content hashes.
  * hybrid_retriever.py recomputes those hashes at startup and refuses
    to serve if any of them disagree, naming the command that repairs it.
  * a live probe additionally reads one real stored sparse vector back
    and checks it decodes against the current vocabulary, which catches
    a stale Qdrant collection even if the on-disk files agree with each
    other.

Any change to tokenization, to build_search_text(), or to the dataset
changes a hash and therefore forces a re-index instead of silently
corrupting retrieval.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

MANIFEST_FILENAME = "bm25_manifest.json"

# Bump when the *meaning* of a stored vector changes: tokenization,
# build_search_text(), or the BM25 term-weighting scheme. Bumping forces
# every consumer to reject an index built before the change.
INDEX_FORMAT_VERSION = 2


def _sha256(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def corpus_fingerprint(section_ids: list[str], search_texts: list[str]) -> str:
    """Identifies the corpus AND the text definition applied to it. Changes
    if a section is added/removed/edited, or if build_search_text() starts
    including different fields."""
    return _sha256(
        f"{sid}\x1f{text}"
        for sid, text in sorted(zip(section_ids, search_texts))
    )


def vocab_fingerprint(vocab: dict[str, int]) -> str:
    """Identifies the exact token -> id mapping. Changes if a token is
    added, removed, or assigned a different id — the failure that made
    stored vectors unreadable."""
    return _sha256(f"{tok}\x1f{idx}" for tok, idx in sorted(vocab.items()))


def build_manifest(
    *,
    section_ids:  list[str],
    search_texts: list[str],
    vocab:        dict[str, int],
    tokenizer_version: int,
    n_docs:  int,
    avgdl:   float,
    k1:      float,
    b:       float,
    collection_name: str,
) -> dict:
    return {
        "index_format_version": INDEX_FORMAT_VERSION,
        "tokenizer_version":    tokenizer_version,
        "collection_name":      collection_name,
        "corpus_fingerprint":   corpus_fingerprint(section_ids, search_texts),
        "vocab_fingerprint":    vocab_fingerprint(vocab),
        "vocab_size":           len(vocab),
        "n_docs":               n_docs,
        "avgdl":                round(avgdl, 6),
        "bm25_k1":              k1,
        "bm25_b":               b,
        "built_at":             datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_manifest(manifest: dict, data_dir: str | Path) -> Path:
    path = Path(data_dir) / MANIFEST_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


def read_manifest(data_dir: str | Path) -> dict | None:
    path = Path(data_dir) / MANIFEST_FILENAME
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


class IndexMismatchError(RuntimeError):
    """Raised when the artifacts a retriever loaded cannot have been
    produced by the same indexing run. Always names the fix."""

    REMEDY = (
        "Re-index from scratch:\n"
        "    python3 data/indexer.py data/final_dataset.json\n"
        "This rebuilds the vocabulary, the IDF table, the Qdrant vectors "
        "and the manifest together, which is the only way they can be "
        "guaranteed consistent."
    )

    def __init__(self, what: str, detail: str = ""):
        msg = f"BM25 index is inconsistent: {what}."
        if detail:
            msg += f"\n  {detail}"
        super().__init__(f"{msg}\n\n{self.REMEDY}")
