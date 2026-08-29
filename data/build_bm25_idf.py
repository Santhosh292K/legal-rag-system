"""
data/build_bm25_idf.py

Computes REAL BM25 IDF weights from document frequency across the corpus,
replacing the encounter-order approximation that used to live inline in
HybridRetriever (see the old "Gap 5" comment in pipeline/hybrid_retriever.py).

The old approximation assumed vocab.json's insertion order correlates with
term rarity ("tokens built into the vocab earlier tend to be more common").
That's a coincidence of how the indexer happened to iterate records, not a
real signal — it doesn't reflect how many *sections* actually contain each
token, which is what document frequency needs to measure.

This script computes true df(token) = number of sections whose text
contains that token, over the same corpus + tokenization the vocab was
built from, then writes a token_index -> idf mapping to
data/bm25_idf.json. HybridRetriever loads that file if present and only
falls back to the old approximation if it's missing, so this is a
non-breaking, opt-in improvement.

Usage:
    python3 data/build_bm25_idf.py
"""
import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from data.bm25_tokenizer import tokenize as _tokenize

HERE        = Path(__file__).parent
VOCAB_PATH  = HERE / "bm25_vocab.json"
DATA_PATH   = HERE / "final_dataset.json"
OUT_PATH    = HERE / "bm25_idf.json"


def tokenize(text: str) -> set[str]:
    # Shared tokenizer (data/bm25_tokenizer.py) so query-side, corpus-side,
    # and this idf computation all agree on the same token boundaries —
    # previously this used a local `text.lower().split()` copy that, like
    # the other two call sites, split off punctuation-fused tokens instead
    # of matching the real word.
    return set(_tokenize(text))


def main():
    with open(VOCAB_PATH, "r") as f:
        vocab: dict[str, int] = json.load(f)

    with open(DATA_PATH, "r") as f:
        records = json.load(f)

    n_docs = len(records)
    df = [0] * len(vocab)

    for rec in records:
        # embedding_text is the richer field (includes section/act name +
        # keywords), matching what the vocab was almost certainly built
        # from; fall back to content if it's ever missing.
        text = rec.get("embedding_text") or rec.get("content", "")
        for tok in tokenize(text):
            idx = vocab.get(tok)
            if idx is not None:
                df[idx] += 1

    # Standard Okapi BM25 IDF: log(1 + (N - df + 0.5) / (df + 0.5))
    # - For df=N (a term in every section), this -> ~log(1 + tiny) ≈ 0. That
    #   matters a lot in a corpus this homogeneous: legal boilerplate like
    #   "section", "act", "provision" appears in most records, and needs to
    #   contribute close to nothing to the score.
    # - For rare terms (small df), this stays high (log(1 + ~N/df)).
    # - Always >= 0 for df in [1, N], so no separate floor/clipping needed.
    #
    # (An earlier version of this script used the generic TF-IDF smoothing
    # log((N+1)/(df+1)) + 1, which floors EVERY term's weight at 1.0 even
    # when it appears in 100% of sections — that floor turned out to matter:
    # it kept boilerplate terms contributing real score in a corpus this
    # repetitive, and measurably hurt retrieval quality end-to-end. The
    # formula below is the one actually used in Okapi BM25, not a
    # substitute — use this one.)
    max_idf = math.log(1 + (n_docs - 0 + 0.5) / (0 + 0.5))  # df=0 fallback: treat as maximally rare
    idf = {}
    for tok, idx in vocab.items():
        d = df[idx]
        idf[str(idx)] = math.log(1 + (n_docs - d + 0.5) / (d + 0.5)) if d > 0 else max_idf

    with open(OUT_PATH, "w") as f:
        json.dump(idf, f)

    # Sanity spot-check: common legal filler vs a rare term, so you can eyeball
    # that the ordering makes sense before trusting it downstream.
    sample_common = vocab.get("section")
    sample_rare   = next((v for k, v in vocab.items() if k in ("dacoity", "pocso")), None)
    print(f"[build_bm25_idf] {n_docs} docs, {len(vocab)} vocab tokens -> {OUT_PATH}")
    if sample_common is not None:
        print(f"  idf('section') = {idf[str(sample_common)]:.3f}  (should be LOW — appears in most sections)")
    if sample_rare is not None:
        print(f"  idf(rare term) = {idf[str(sample_rare)]:.3f}  (should be HIGH — appears in few sections)")


if __name__ == "__main__":
    main()