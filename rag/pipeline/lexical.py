"""
pipeline/lexical.py
IDF-weighted lexical overlap, shared by the IRAC reranker.

Why not Jaccard
---------------
The reranker's Stage 1 used raw Jaccard, |A∩B| / |A∪B|, to compare a
query against a section's issue_tags / rule_summary / content. That is
the wrong shape for this comparison in two ways:

  1. It is length-biased. The query (after Stage 0b translation is
     concatenated with the user's own words) runs 40-80 tokens; issue_tags
     median 3 words and rule_summary median 18. The union is dominated by
     the long side, so even a PERFECT topical match scores near zero.
     Measured over 800 sections with a realistic query, the median score
     for all three components was 0.000 and the maximum 0.11-0.23.
  2. It treats every token as equally informative, so a shared "person"
     counts the same as a shared "dacoity".

Both matter because Stage 1 is not cosmetic — it chooses which handful of
candidates get an LLM and cross-encoder look. With a median of 0.000 the
ordering was mostly ties, resolved by retrieval order rather than by
relevance.

What this uses instead
----------------------
The IDF-weighted overlap coefficient:

    overlap(A,B) = SUM_{t in A∩B} idf(t) / min(SUM_{t in A} idf(t),
                                               SUM_{t in B} idf(t))

Normalizing by the SMALLER side makes a short, fully-matched field score
high instead of being punished for brevity, and IDF weighting means rare
legal vocabulary dominates shared filler. The IDF table is the same one
BM25 uses, so "informative" means the same thing in both stages.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import BM25_VOCAB_PATH
from data.bm25_tokenizer import tokenize

# Common English plus legal filler that appears in nearly every section.
# IDF already drives most of these toward zero; removing them outright
# also keeps them out of the denominator.
#
# Stored STEMMED, because that is what tokenize() emits — an unstemmed
# list would silently stop filtering the moment stemming was introduced
# ("any" -> "ani", "following" -> "follow"), which is the same
# tokenizer-drift failure mode this codebase already had once.
_STOP_WORD_SOURCES = (
    "a an the and or of in to is are was were be been being have has had "
    "do does did will would could should may might shall by with for on "
    "at from as into through that this his her its their he she it they "
    "who which what any all not no "
    # Legal filler
    "act section under such thereof thereto herein aforesaid said above following"
).split()

STOP_WORDS = frozenset(tokenize(" ".join(_STOP_WORD_SOURCES)))

# Weight given to a token absent from the BM25 vocabulary. A token the
# corpus has never seen is by definition rare, so it gets the same weight
# as the rarest known term rather than being ignored.
_UNKNOWN_TOKEN_IDF = None   # resolved lazily to max(idf)


class LexicalScorer:
    """Loads the BM25 vocabulary/IDF once and scores overlaps with it."""

    _shared: "LexicalScorer | None" = None

    def __init__(self, vocab_path: str = BM25_VOCAB_PATH):
        vocab_file = Path(vocab_path)
        idf_file   = vocab_file.parent / "bm25_idf.json"
        self._idf_by_token: dict[str, float] = {}
        self._default_idf = 1.0

        if vocab_file.exists() and idf_file.exists():
            with open(vocab_file) as f:
                vocab: dict[str, int] = json.load(f)
            with open(idf_file) as f:
                idf_by_index = {int(k): v for k, v in json.load(f).items()}
            self._idf_by_token = {
                tok: idf_by_index.get(idx, 1.0) for tok, idx in vocab.items()
            }
            if self._idf_by_token:
                self._default_idf = max(self._idf_by_token.values())
        else:
            # No index yet (fresh checkout, unit tests). Degrade to
            # unweighted overlap rather than failing — this scorer is a
            # ranking heuristic, not a correctness boundary.
            print("[LexicalScorer] BM25 vocab/idf not found — "
                  "falling back to unweighted overlap.")

    @classmethod
    def shared(cls) -> "LexicalScorer":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def idf(self, token: str) -> float:
        return self._idf_by_token.get(token, self._default_idf)

    def informative_tokens(self, text: str) -> set[str]:
        return set(tokenize(text or "")) - STOP_WORDS

    def _mass(self, tokens: set[str]) -> float:
        return sum(self.idf(t) for t in tokens)

    def overlap(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between IDF-weighted term-presence vectors.

        Bounded [0,1], symmetric, and — unlike Jaccard — not dominated by
        whichever side is longer, so a 3-word issue_tags string that
        genuinely matches is not scored near zero just because the query
        around it is 60 tokens.

        Cosine rather than the overlap coefficient |A∩B|/min(|A|,|B|):
        the overlap coefficient saturates at exactly 1.0 whenever one side
        is a subset of the other, so a section tagged with a single common
        word scored identically to one tagged with a rare, highly
        diagnostic term. Cosine keeps the IDF signal in the denominator and
        separates those cases.
        """
        a = self.informative_tokens(text_a)
        b = self.informative_tokens(text_b)
        if not a or not b:
            return 0.0
        shared = a & b
        if not shared:
            return 0.0
        num   = sum(self.idf(t) ** 2 for t in shared)
        denom = math.sqrt(self._sq_mass(a)) * math.sqrt(self._sq_mass(b))
        return min(num / denom, 1.0) if denom else 0.0

    def _sq_mass(self, tokens: set[str]) -> float:
        return sum(self.idf(t) ** 2 for t in tokens)

    def coverage(self, needle: str, haystack: str) -> float:
        """Share of `needle`'s informative mass present in `haystack`.
        Asymmetric — use when one side is definitionally the thing being
        looked for."""
        a = self.informative_tokens(needle)
        b = self.informative_tokens(haystack)
        if not a:
            return 0.0
        denom = self._mass(a)
        return (self._mass(a & b) / denom) if denom else 0.0
