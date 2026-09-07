"""
data/build_bm25_idf.py — now a READ-ONLY verifier.

This script used to rebuild data/bm25_idf.json from data/bm25_vocab.json
independently of the indexer. That is a footgun, and it fired: BM25 sparse
vectors are keyed by integer token ids that only mean anything relative to
the vocabulary they were built with, so regenerating vocabulary or IDF
without also rewriting every stored vector leaves queries and corpus in
two different index spaces. Retrieval then degrades to noise with no error
anywhere.

That is exactly the state this repository shipped in — stored vectors from
one tokenizer, a vocabulary file from a second, running code implying a
third, with measured index-set agreement of ~15%.

So it no longer writes anything. data/indexer.py writes the vocabulary,
the IDF table, the Qdrant vectors and the manifest in one run; this script
only checks that what is on disk is self-consistent and tells you what to
run if it isn't.

Usage:
    python3 data/build_bm25_idf.py        # verify; exit 1 on mismatch
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from data.bm25_tokenizer import tokenize, build_search_text, TOKENIZER_VERSION
from data.index_manifest import (
    read_manifest, corpus_fingerprint, vocab_fingerprint,
    INDEX_FORMAT_VERSION, IndexMismatchError,
)

HERE       = Path(__file__).parent
VOCAB_PATH = HERE / "bm25_vocab.json"
IDF_PATH   = HERE / "bm25_idf.json"
DATA_PATH  = HERE / "final_dataset.json"


def verify() -> list[str]:
    """Returns a list of problems; empty means consistent."""
    problems: list[str] = []

    manifest = read_manifest(HERE)
    if manifest is None:
        return ["data/bm25_manifest.json is missing — the index predates "
                "consistency checking, or was never built."]

    if manifest.get("index_format_version") != INDEX_FORMAT_VERSION:
        problems.append(
            f"index format v{manifest.get('index_format_version')} on disk, "
            f"code expects v{INDEX_FORMAT_VERSION}")
    if manifest.get("tokenizer_version") != TOKENIZER_VERSION:
        problems.append(
            f"index built with tokenizer v{manifest.get('tokenizer_version')}, "
            f"code tokenizes as v{TOKENIZER_VERSION}")

    if not VOCAB_PATH.exists():
        problems.append(f"{VOCAB_PATH.name} is missing")
    else:
        with open(VOCAB_PATH) as f:
            vocab = json.load(f)
        if vocab_fingerprint(vocab) != manifest.get("vocab_fingerprint"):
            problems.append(f"{VOCAB_PATH.name} does not match the manifest")
        if not IDF_PATH.exists():
            problems.append(f"{IDF_PATH.name} is missing")
        else:
            with open(IDF_PATH) as f:
                idf = json.load(f)
            if len(idf) != len(vocab):
                problems.append(
                    f"{IDF_PATH.name} has {len(idf)} entries for a "
                    f"{len(vocab)}-token vocabulary")

    if DATA_PATH.exists():
        with open(DATA_PATH, encoding="utf-8") as f:
            records = json.load(f)
        ids   = [r["section"] for r in records]
        texts = [build_search_text(r) for r in records]
        if corpus_fingerprint(ids, texts) != manifest.get("corpus_fingerprint"):
            problems.append(
                "final_dataset.json has changed since the index was built")

    return problems


def main() -> int:
    problems = verify()
    if not problems:
        manifest = read_manifest(HERE) or {}
        print("[verify] BM25 index is consistent.")
        print(f"  built      : {manifest.get('built_at', '?')}")
        print(f"  vocabulary : {manifest.get('vocab_size', '?')} tokens")
        print(f"  documents  : {manifest.get('n_docs', '?')}  "
              f"avgdl={manifest.get('avgdl', '?')}")
        print(f"  bm25       : k1={manifest.get('bm25_k1')} b={manifest.get('bm25_b')}")
        return 0

    print("[verify] BM25 index is INCONSISTENT:")
    for problem in problems:
        print(f"  - {problem}")
    print()
    print(IndexMismatchError.REMEDY)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
