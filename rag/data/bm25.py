"""
data/bm25.py
The Okapi BM25 term-weighting used by the sparse index.

    score(q,d) = SUM_t  idf(t) * qtf_w(t)
                        * (tf_d(t)*(k1+1)) / (tf_d(t) + k1*(1 - b + b*|d|/avgdl))

Split across a sparse dot product:

    document weight  w_d(t) = (tf*(k1+1)) / (tf + k1*(1 - b + b*|d|/avgdl))
    query weight     w_q(t) = idf(t) * (qtf*(k3+1)) / (k3 + qtf)

so <w_d, w_q> == BM25 exactly. This module owns the document side and the
IDF table; pipeline/hybrid_retriever.py owns the query side.

Separate from data/indexer.py so the weighting can be imported — and
tested — without pulling in sentence-transformers and torch.

What this replaces: the previous index stored `tf * idf` on BOTH sides,
which applied IDF twice (squaring it), had no term-frequency saturation,
and no document-length normalization at all. Corpus-vector L2 norm
correlated 0.81 with document length as a result.
"""
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import BM25_K1, BM25_B


def build_vocab(tokenized_corpus: list[list[str]]) -> dict[str, int]:
    """Token -> id. Ids are assigned in first-encounter order, which is
    stable for a fixed corpus + tokenizer; data/index_manifest.py hashes
    the resulting mapping so any drift is detected rather than tolerated."""
    vocab: dict[str, int] = {}
    for tokens in tokenized_corpus:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def compute_idf(
    tokenized_corpus: list[list[str]],
    vocab: dict[str, int],
) -> dict[int, float]:
    """Okapi BM25 IDF: log(1 + (N - df + 0.5)/(df + 0.5)).

    Non-negative over df in [1, N], and -> ~0 for a term present in every
    section, which matters in a corpus this homogeneous: "section", "act"
    and "provision" appear nearly everywhere and must contribute almost
    nothing.
    """
    n_docs = len(tokenized_corpus)
    df = [0] * len(vocab)
    for tokens in tokenized_corpus:
        for tok in set(tokens):
            df[vocab[tok]] += 1
    return {
        idx: math.log(1 + (n_docs - df[idx] + 0.5) / (df[idx] + 0.5))
        for idx in vocab.values()
    }


def bm25_document_weights(
    tokens: list[str],
    vocab:  dict[str, int],
    avgdl:  float,
    k1:     float = BM25_K1,
    b:      float = BM25_B,
) -> dict[int, float]:
    """Document side of the BM25 decomposition — see the module docstring.
    Length normalization lives here; IDF does NOT (it belongs to the query
    side, applied exactly once)."""
    doc_len = len(tokens)
    norm    = k1 * (1.0 - b + b * (doc_len / avgdl if avgdl else 1.0))

    tf: dict[int, int] = {}
    for token in tokens:
        idx = vocab.get(token)
        if idx is not None:
            tf[idx] = tf.get(idx, 0) + 1

    return {idx: (f * (k1 + 1.0)) / (f + norm) for idx, f in tf.items()}
