"""
data/bm25_tokenizer.py

The single definition of what a token IS, shared by every place that
builds or queries the BM25 index: data/indexer.py (vocabulary, IDF and
corpus sparse vectors), pipeline/hybrid_retriever.py (query vectors) and
pipeline/lexical.py (the reranker's lexical overlap).

There is exactly one implementation on purpose. All three call sites used
to do their own `text.lower().split()`, which is a whitespace split with
no punctuation handling, so "law." / "law," / "law" / "law?" were four
different vocabulary entries — 3,950 of 9,286 tokens (42.5%) in the
resulting vocabulary had punctuation fused to them. Since a query rarely
carries the exact trailing punctuation a corpus token happened to acquire,
"Is this legal?" tokenized to "legal?" and could not match "legal" at all.

Changing anything in this file changes what every stored sparse vector
MEANS. TOKENIZER_VERSION below is stamped into data/bm25_manifest.json by
the indexer and checked at startup by pipeline/hybrid_retriever.py, so a
change here forces a full re-index instead of silently corrupting
retrieval:

    python3 data/indexer.py data/final_dataset.json

(There is deliberately no partial rebuild. Regenerating the vocabulary
without rewriting the stored vectors repoints every query at the wrong
terms — see data/index_manifest.py.)
"""
import re

# Bump whenever tokenize() or build_search_text() changes what a token IS.
# data/indexer.py stamps this into data/bm25_manifest.json and
# pipeline/hybrid_retriever.py refuses to serve an index built under a
# different value — see data/index_manifest.py for why that check exists.
#   v2: punctuation-stripping + British/American spelling normalization
#   v3: + Snowball (Porter2) stemming
TOKENIZER_VERSION = 3

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# ── Stemming ────────────────────────────────────────────────────────────
# Suffix stripping, so morphological variants of the same legal term match.
# Statute text and the queries asked about it inflect differently by
# nature: a section says "whoever wrongfully CONFINES", the user asks
# about "wrongful CONFINEMENT"; a section prescribes what is "PUNISHABLE",
# the query asks for the "PUNISHMENT". Without stemming those are
# unrelated tokens and contribute nothing to either BM25 or the reranker's
# lexical overlap.
#
# Measured on evaluation/benchmark_scenarios.json (50 queries with gold
# sections), BM25-only, corpus and query tokenized identically:
#
#     recall@10   0.217 -> 0.245   (+0.028)
#     recall@20   0.253 -> 0.273   (+0.020)
#     recall@50   0.303 -> 0.343   (+0.040)
#     MRR         0.222 -> 0.252   (+0.030)
#     vocabulary  6109  -> 3896
#
# Snowball (Porter2) rather than a hand-rolled suffix stripper: it is the
# standard algorithm, it is already a declared dependency via nltk, and
# hand-rolled rules conflate unrelated legal terms in ways that are hard
# to notice.
#
# Deliberately NOT wrapped in a try/except fallback. Silently tokenizing
# differently when a dependency is missing is precisely the class of bug
# this file's history is made of — a stemmed corpus queried with unstemmed
# tokens is worse than either choice made consistently. Fail loudly
# instead; the manifest check would catch it at startup anyway.
try:
    from nltk.stem.snowball import SnowballStemmer as _SnowballStemmer
except ImportError as exc:                                  # pragma: no cover
    raise ImportError(
        "nltk is required for tokenization (Snowball stemming). "
        "Install it with: pip install nltk"
    ) from exc

_STEMMER = _SnowballStemmer("english")
_STEM_CACHE: dict[str, str] = {}


def _stem(token: str) -> str:
    """Memoized — the corpus has ~3.4k documents and the same tokens recur
    constantly, and SnowballStemmer.stem is pure Python."""
    stemmed = _STEM_CACHE.get(token)
    if stemmed is None:
        stemmed = _STEMMER.stem(token)
        _STEM_CACHE[token] = stemmed
    return stemmed

# ── British/American spelling normalization ──────────────────────────────
# BUG this closes: this corpus is written in Indian English (British
# spelling throughout — "labour", "defence", "offence", "practise",
# "organisation" all appear in the statute text), but nothing stops a user
# from typing the American spelling ("labor", "defense", "offense"). Since
# tokenize() has no stemming and does exact alphanumeric matching, those are
# two completely different vocabulary entries to BM25 — a query for "child
# labor" gets ZERO lexical overlap with every corpus section that says
# "labour" (BNS_146: "compels any person to labour against that person will
# ..."), even though they're the same word. This silently drops BM25 signal
# on this token for exactly the users most likely to type American English.
#
# This also affects pipeline/irac_reranker.py's _token_overlap(), which
# imports this same tokenize() function for its Stage-1 Jaccard filter — so
# the same spelling mismatch can knock a genuinely relevant section out of
# consideration before the LLM/cross-encoder ever sees it, not just weaken
# its BM25 score.
#
# Mapped to the corpus's own (British) spelling, which is the canonical
# form on the corpus side, so no re-indexing of the CONTENT is needed —
# only the query-side token changes. Kept to a short, high-confidence list
# of whole-word American→British pairs actually likely to appear in legal
# queries, rather than a general spelling-normalization library, to avoid
# conflating unrelated words.
#
# Mapped to the corpus's own (British) spelling. Applied BEFORE stemming,
# because Snowball does not conflate the variants by itself
# ("organization" -> "organ" but "organisation" -> "organis").
_US_TO_UK: dict[str, str] = {
    "labor": "labour", "labors": "labours", "labored": "laboured", "laboring": "labouring",
    "defense": "defence", "defenses": "defences",
    "offense": "offence", "offenses": "offences",
    "practise": "practice", "practised": "practiced", "practising": "practicing",
    "organization": "organisation", "organizations": "organisations",
    "recognize": "recognise", "recognized": "recognised", "recognizes": "recognises",
    "license": "licence", "licenses": "licences",  # noun form only; corpus uses "licence"
    "program": "programme", "programs": "programmes",
    "judgment": "judgement", "judgments": "judgements",
    "authorize": "authorise", "authorized": "authorised", "authorizes": "authorises",
    "fulfill": "fulfil", "fulfilled": "fulfilled", "fulfilling": "fulfilling",
    "enrollment": "enrolment", "installment": "instalment", "installments": "instalments",
    "counselor": "counsellor", "counselors": "counsellors",
    "civilization": "civilisation", "specialize": "specialise", "specialized": "specialised",
    "color": "colour", "colored": "coloured",
    "modeling": "modelling", "traveling": "travelling", "traveled": "travelled",
    "canceled": "cancelled", "canceling": "cancelling",
}


def tokenize(text: str) -> list[str]:
    """Lowercase -> punctuation-stripped -> spelling-normalized -> stemmed.

    Order matters: spelling normalization must run BEFORE stemming,
    because Snowball does not conflate the variants itself
    ("organization" -> "organ" but "organisation" -> "organis").

    Returns a list (repeats preserved, so callers computing raw term
    frequency don't need to change); wrap in set(...) where only
    membership is needed.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [_stem(_US_TO_UK.get(t, t)) for t in tokens]


def build_search_text(record: dict) -> str:
    """The text actually used for BOTH BM25 vocabulary/sparse-vector
    construction AND dense embedding — by data/indexer.py at index time and
    pipeline/hybrid_retriever.py's query side, so all three agree.

    BUGFIX: every record carries a hand-curated `meta.keywords` list
    specifically meant to bridge lay-language query vocabulary to the
    section's actual (often narrower/older) statutory wording — e.g.
    IPC_498A ("Husband or relative of husband subjecting a woman to
    cruelty...") is tagged with the keyword "domestic violence", a phrase
    that never appears anywhere in the section's own text. But indexer.py
    only ever built the BM25 vocab and dense embeddings from
    record["embedding_text"] — which does NOT include keywords (verified
    directly against final_dataset.json: IPC_498A's embedding_text is just
    the act/section header + content, no keyword list) — so that curated
    bridge vocabulary was copied into the Qdrant payload for display/
    filtering only and never once influenced retrieval. A query using
    "domestic violence" (the dataset's own suggested vocabulary for this
    exact section) had no lexical anchor to IPC_498A at all and had to
    clear the bar on dense semantic similarity alone — which it did not,
    in production, even though a same-topic query using "dowry" (a word
    that DOES appear in the section's own text) retrieved it easily.
    Appends keywords to embedding_text/content for indexing purposes only
    — does not mutate the record, so record["embedding_text"] still stores
    the clean act/section/content text wherever it's used for display.
    """
    meta     = record.get("meta") or {}
    keywords = meta.get("keywords") or []
    base     = record.get("embedding_text") or record.get("content", "")
    if not keywords:
        return base
    return f"{base} {' '.join(keywords)}"
