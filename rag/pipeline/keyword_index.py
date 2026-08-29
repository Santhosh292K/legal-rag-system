"""
pipeline/keyword_index.py
A dataset-driven complement to section_pinner's semantic search.

section_pinner.py now finds sections by dense similarity against the
indexed corpus rather than a hand-maintained regex table (see
section_pinner.py's docstring), which already fixed most of the
"nobody wrote a pattern for this phrasing" gaps a purely regex-based
version used to have. But dense similarity can still miss an exact
dataset-authored term that doesn't happen to sit close to the query in
embedding space — e.g. IPC_364A ("kidnapping for ransom") and BNS_140 are
tagged with keywords like "kidnapping for ransom" and "ransom demand" in
meta.keywords, which a keyword-exact match will always catch regardless
of how the embedding model happens to score that particular phrasing.

This module builds an inverted index — keyword phrase -> section_ids —
directly from every section's own meta.keywords (and category, as a
lower-precision fallback), once, at load time. Zero hand-maintenance: add
a new section to the dataset with good keywords, and it's immediately
matchable, no code change required. Used as a SECOND matching pass
alongside section_pinner, not a replacement for it — the two catch
different things (semantic search catches phrasing variants like
"abducted"/"kidnapped"/"taken away forcibly" that mean the same thing;
the keyword index catches exact dataset-authored terms like "ransom
demand" regardless of embedding distance).
"""
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

MIN_KEYWORD_LEN = 4   # skip very short/generic keywords that would over-match


@dataclass
class KeywordMatch:
    section_id: str
    matched_keywords: list[str] = field(default_factory=list)
    weighted_score: float = 0.0

    @property
    def score(self) -> int:
        return len(self.matched_keywords)


class KeywordIndex:
    def __init__(self, json_path: str = "./data/final_dataset.json"):
        self._index: dict[str, list[str]] = defaultdict(list)   # keyword -> [section_ids]
        self._weight: dict[str, float] = {}                      # keyword -> specificity weight
        self._loaded_from = json_path
        self._build(json_path)

    def _build(self, json_path: str):
        with open(json_path) as f:
            records = json.load(f)

        for r in records:
            section_id = r.get("section")
            meta = r.get("meta") or {}
            keywords = meta.get("keywords") or []
            for kw in keywords:
                kw_norm = kw.strip().lower()
                if len(kw_norm) >= MIN_KEYWORD_LEN:
                    self._index[kw_norm].append(section_id)

        # Weight = how specific a keyword is. Two factors, same idea BM25's
        # IDF captures: (1) multi-word phrases are inherently more precise
        # than single common words ('kidnapping for ransom' vs 'person'),
        # and (2) a keyword shared by many sections is less discriminating
        # than one that's nearly unique — divide by how many sections use
        # it. Without this, a match on 'person' (appears in hundreds of
        # sections) scored the same as a match on 'kidnapping for ransom'
        # (appears in one) — which is exactly backwards.
        for kw, section_ids in self._index.items():
            word_count = len(kw.split())
            doc_freq = len(set(section_ids))
            self._weight[kw] = (word_count ** 2) / (doc_freq ** 0.5)

    def match(self, text: str, max_sections: int = 15) -> list[KeywordMatch]:
        """Scans text for every indexed keyword phrase as a substring match
        (not regex — keywords are dataset-authored phrases, not patterns).
        Ranks by cumulative specificity-weighted score, not raw match
        count — see _build for why that matters."""
        text_norm = re.sub(r"\s+", " ", text.lower())

        hits: dict[str, list[str]] = defaultdict(list)
        for keyword, section_ids in self._index.items():
            if keyword in text_norm:
                for sid in section_ids:
                    hits[sid].append(keyword)

        matches = [KeywordMatch(section_id=sid, matched_keywords=kws) for sid, kws in hits.items()]
        for m in matches:
            m.weighted_score = sum(self._weight.get(kw, 0.0) for kw in m.matched_keywords)
        matches.sort(key=lambda m: m.weighted_score, reverse=True)
        return matches[:max_sections]

    def section_ids(self, text: str, max_sections: int = 15) -> list[str]:
        return [m.section_id for m in self.match(text, max_sections)]


if __name__ == "__main__":
    idx = KeywordIndex()
    print(f"Indexed {len(idx._index)} distinct keywords from {idx._loaded_from}")
    print()

    text = ("unidentified persons intercepted the complainant's vehicle and abducted the "
            "company's CFO, demanding 25 crore through encrypted messages as ransom")
    matches = idx.match(text)
    for m in matches[:8]:
        print(f"  {m.section_id}  (score={m.score})  matched: {m.matched_keywords}")