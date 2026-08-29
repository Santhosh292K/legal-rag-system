"""
data/bm25_tokenizer.py

Single shared tokenizer for every place that builds or queries the BM25
sparse index: data/indexer.py (vocab + corpus sparse vectors),
data/build_bm25_idf.py (document frequency), and
pipeline/hybrid_retriever.py (query-side sparse vectors).

BUG this replaces: all three call sites used to do their own
`text.lower().split()`. That's a whitespace split with no punctuation
stripping, so "law." / "law," / "law" / "law?" are four different vocab
entries. A spot check on the actual bm25_vocab.json found 3,950 of 9,286
tokens (42.5%) had punctuation stuck to them ('applicability.', 'india.',
'age.', 'company,', ...). Two concrete costs of that:
  1. Corpus-side term frequency and document frequency are fragmented
     across punctuation variants of the same word, so IDF is computed
     over the wrong counts and TF is undercounted for any token that
     ever appears before a comma/period in the source text.
  2. Query-side tokens almost never carry the exact trailing punctuation
     a corpus token happened to end up with, so a query like "Is this
     legal?" tokenizes to "legal?", which cannot match the vocab entry
     "legal" (or "legal." from some other sentence) at all — the token is
     silently dropped from the BM25 query vector entirely.
Benchmark queries in this project are full sentences ending in "?" or
containing embedded clauses ("... at 17 years of age. Is this legal?"),
so this bug was silently zeroing out BM25 signal on exactly the kind of
query the benchmark uses.

Fix: tokenize on runs of alphanumerics only, so punctuation never attaches
to a token on either the corpus or the query side.

IMPORTANT: changing this changes vocab token IDs (via data/indexer.py's
build_vocab), so after pulling this fix you must fully re-run
`python3 data/indexer.py` (not just --rebuild-vocab) — it rebuilds
bm25_vocab.json, bm25_idf.json, AND re-upserts every point's sparse
vector so they all agree on the same token->id mapping. Running only
--rebuild-vocab would leave Qdrant's stored sparse vectors keyed to the
OLD vocab ids while queries encode against the NEW ids, which is worse
than not fixing this at all.
"""
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

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
# IMPORTANT: like the punctuation-stripping fix above, this changes which
# token id a normalized word maps to versus the currently-built
# bm25_vocab.json/bm25_idf.json (an existing corpus token "labor" — if any
# ever occurs verbatim — would now normalize to "labour" and merge with
# that entry). Re-run `python3 data/indexer.py` (full rebuild, not
# --rebuild-vocab alone) after pulling this change, for the same reason
# described above.
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
    """Lowercase, punctuation-stripped, spelling-normalized tokenization.
    Returns a list (preserves repeats, so callers doing raw term-frequency
    counts don't need to change); wrap in set(...) where only membership is
    needed."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [_US_TO_UK.get(t, t) for t in tokens]
