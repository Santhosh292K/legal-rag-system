"""
pipeline/intent_classifier.py
Novel component #1 — Legal Intent Classifier

CHANGE FROM PREVIOUS VERSION:
The old classifier ran keyword-regex scoring first and only called the LLM
when confidence was BELOW 0.6 — and confidence could hit 0.6+ off just two
coincidental keyword matches, so most queries never reached the LLM at all.
A query phrased in a way that trips two unrelated keywords could get
confidently misclassified without any semantic check.

This version is a three-tier cascade, cheapest-and-most-precise first:
  1. Rule-based (regex)   — fast, free, precise for queries that use the
                             exact statute vocabulary. Confidence bar to
                             accept it outright is now HIGHER (0.75, was
                             implicit 0.6) since it's cheap to fall through.
  2. Embedding-based       — SemanticMatcher over a small set of canonical
     (semantic_classify)     example phrasings per intent label. Catches
                             paraphrases the regex list never enumerated,
                             without needing an LLM call. New phrasing
                             patterns can be added via a JSON data file
                             instead of editing regex.
  3. LLM-based             — full flexibility, used last since it's the
                             slowest/costliest, and as the final fallback
                             if the embedding tier isn't confident either.

act_hint detection keeps a small regex layer for literal abbreviation/act
mentions (genuinely closed vocabulary — "IPC", "CRPC" don't need fuzzy
matching) but the large "behavioural evidence" keyword lists (murder, kill,
theft, assault, ...) are now ALSO available to the embedding tier via
ACT_EXAMPLES, so a query that never says "IPC" but clearly describes an
IPC-flavoured scenario can still get a confident act_hint.
"""
import re
import json
from datetime import date
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.semantic_matcher import SemanticMatcher
from config import OLLAMA_FAST_MODEL
import ollama


@dataclass
class QueryIntent:
    label:      str
    confidence: float
    act_hint:   str | None
    temporal:   str
    keywords:   list[str] = field(default_factory=list)
    reasoning:  str       = ""
    # Gap 2: multi-label support — every intent whose score reaches >=50% of
    # the top score is included. The IRAC reranker uses this to score chunks
    # against the CLOSEST matching label rather than a single global label.
    labels:     list[str] = field(default_factory=list)
    source:     str       = "rule"   # "rule" | "semantic" | "llm" — which tier answered
    # Gap: temporal cutoff. `temporal` above is a coarse historical/current/
    # comparative/unspecified LABEL — it never told the temporal filter WHEN.
    # A query like "a theft that happened in 2019" or "before BNS came into
    # force" names an actual point in time that determines which of IPC
    # (pre-2024) vs BNS (enacted 2023, effective 2024-07-01) applies, but
    # nothing extracted that year anywhere — TemporalFilter.filter()'s
    # cutoff_year parameter existed but no caller ever computed one, so it
    # was always None and the year-based exclusion branch was dead code.
    # cutoff_year is the latest year a provision could have been enacted and
    # still have governed the query's incident (None = no date found).
    cutoff_year:   int | None = None
    # cutoff_date (ISO 'YYYY-MM-DD'): set only when the query gives
    # day/month precision, not just a bare year. Matters because BNS/BNSS/
    # BSA took effect mid-year (2024-07-01) — "a theft in January 2024" is
    # unambiguously pre-BNS, but collapsing it to just cutoff_year=2024
    # can't tell that apart from "a theft in December 2024" (post-BNS).
    # cutoff_year alone is the best available answer for a genuinely
    # date-less "2024" — see intent_classifier.py's _extract_cutoff.
    cutoff_date:   str | None = None
    cutoff_reason: str        = ""


# ── Small, closed-vocabulary literal act abbreviations ──────────────────────
# Kept as regex because these ARE a fixed, closed set — "IPC" is always
# "IPC", fuzzy matching adds no value here.
ACT_ABBREVIATION_PATTERNS = {
    "IPC":   [r"\bipc\b", r"indian penal", r"penal code"],
    "CPC":   [r"\bcpc\b", r"civil procedure"],
    "ITA":   [r"\bita\b", r"information technology act"],
    "CRPC":  [r"\bcrpc\b", r"criminal procedure"],
    "IEA":   [r"\biea\b", r"evidence act"],
    "POCSO": [r"\bpocso\b"],
    "NDPS":  [r"\bndps\b"],
    "BNS":   [r"\bbns\b", r"bharatiya nyaya"],
    "BNSS":  [r"\bbnss\b", r"bharatiya nagarik"],
}

# ── Embedding examples for act detection by scenario, not just abbreviation ─
# Replaces the old large per-act behavioural regex lists (murder, kill,
# theft, assault, hack, drug, ... ~20 patterns per act) with a handful of
# representative canonical scenarios per act.
ACT_SCENARIO_EXAMPLES = {
    "IPC": [
        "someone was murdered", "a robbery took place", "he assaulted her",
        "a doctor's negligence caused a patient's death", "fraud and cheating for money",
        "a fake certificate was used", "someone was kidnapped",
    ],
    "ITA": [
        "a hacker accessed my computer without permission", "online fraud through a website",
        "someone stole my data digitally", "phishing scam", "identity theft online",
    ],
    "CRPC": [
        "the police arrested someone", "applying for bail", "filing an FIR",
        "the investigation procedure", "custody and remand",
    ],
    "IEA": [
        "is this evidence admissible in court", "witness testimony reliability",
        "hearsay evidence in a trial",
    ],
    "POCSO": [
        "a child was sexually abused", "a minor was assaulted",
    ],
    "NDPS": [
        "someone was caught with drugs", "narcotics trafficking",
    ],
}

# ── Intent label canonical examples (embedding tier) ─────────────────────────
INTENT_EXAMPLES = {
    "definition": [
        "what does this section mean", "define this legal term",
        "explain what this law says", "what is the meaning of this provision",
    ],
    "punitive": [
        "what is the punishment for this", "can he be arrested for this",
        "is this a crime", "what charges apply here", "is this person guilty",
        "what happens if someone does this",
    ],
    "procedural": [
        "how do I file this", "what is the procedure to apply",
        "what are the steps to appeal", "how to register a complaint",
    ],
    "case_law": [
        "is there a court ruling on this", "what did the court decide in a similar case",
        "precedent for this situation",
    ],
    "statute": [
        "which section covers this", "what does the act say about this",
        "which law applies here",
    ],
}

TEMPORAL_RULES = {
    # BUGFIX: the historical list used to include a bare
    # `was|were|had|used to|earlier` alternation — ordinary past-tense
    # English, not a signal about which law-era applies. Virtually every
    # real case narrative ("the accused WAS arrested...", "she HAD filed a
    # complaint...") is written in the past tense, so this tripped
    # temporal="historical" on most real queries even with zero date
    # mentioned anywhere. That mattered a lot downstream: TemporalFilter's
    # `is_valid = ... or historical_query` (both the old code and — for
    # chunks that aren't chronologically future — the current code) treats
    # a historical query as "accept every validity label", so the supposedly
    # "strict active-only" filter silently did nothing for most queries,
    # letting amended/repealed/whatever-matched-by-meaning sections straight
    # through unfiltered. Kept only the patterns that are actually specific
    # to wanting OLD/pre-BNS law, not just past-tense narration.
    "historical": [
        r"\bbefore\s+\d{4}\b", r"\bold\s+law\b",
        r"\b(before\s+(bns|bnss|bsa)\b)",
        r"\b(ipc|crpc|iea)\s+(before|prior|replaced|superseded)\b",
        r"\b(replaced|superseded)\s+by\s+(bns|bnss|bsa)\b",
        r"\bunder\s+(the\s+)?old\s+(ipc|crpc|penal\s+code)\b",
        r"\bpre[- ]bns\b",
        r"\bwhen\s+(ipc|crpc)\s+was\s+in\s+force\b",
    ],
    "current": [
        # BUGFIX: this used to hardcode "2023|2024|2025" as "current"
        # signal words. Two problems: (1) it goes stale by construction —
        # already missing 2026 as of this writing, and needs indefinite
        # manual upkeep; (2) it fought directly with cutoff extraction
        # below — "a murder happened in 2024" was mislabeled
        # temporal="current" purely because "2024" is in this list, even
        # though the query is unambiguously describing a PAST incident.
        # Genuine "current law" intent is relative ("current", "now",
        # "still in force", ...), never a specific calendar year — a
        # specific year is exactly what cutoff extraction below is for.
        r"\b(current|now|today|present|latest|new law)\b",
        r"\b(bns|bnss|bsa)\s+(now|currently|today|replaced|enacted)\b",
        r"\bstill\s+in\s+force\b",
        r"\bis\s+(ipc|crpc|ipc\s+\d+)\s+still\b",
        r"\bpost[- ]bns\b",
        r"\bafter\s+(bns|bnss|bsa)\s+(was\s+)?enacted\b",
        r"\bwhat\s+changed\s+when\s+(bns|bnss)\b",
    ],
    "comparative": [
        r"\b(ipc|crpc)\s+(vs\.?|versus|compared\s+to|and)\s+(bns|bnss)\b",
        r"\b(bns|bnss)\s+(vs\.?|versus|compared\s+to|and)\s+(ipc|crpc)\b",
        r"\bwhat\s+changed\b",
        r"\bdifference\s+between\s+(ipc|crpc)\s+and\s+(bns|bnss)\b",
    ],
}

# BNS, BNSS and BSA all took effect 2024-07-01 (final_dataset.json's own
# effective_date field — see is_chronologically_future's module note in
# temporal_filter.py; also matches the actual notified date). A query that
# names this changeover qualitatively ("under the OLD IPC", "before BNS")
# without giving a specific year still deserves a precise cutoff — this is
# that one well-established constant, not a per-query guess.
_BNS_CHANGEOVER = date(2024, 7, 1)
_PRE_BNS_PHRASE_RE = re.compile(
    r"\bbefore\s+(?:bns|bnss|bsa)\b"
    r"|\b(?:ipc|crpc|iea)\s+(?:before|prior|replaced|superseded)\b"
    r"|\b(?:replaced|superseded)\s+by\s+(?:bns|bnss|bsa)\b"
    r"|\bunder\s+(?:the\s+)?old\s+(?:ipc|crpc|penal\s+code)\b"
    r"|\bpre[- ]bns\b"
    r"|\bwhen\s+(?:ipc|crpc)\s+was\s+in\s+force\b"
)

INTENT_EXAMPLES_PATH = str(
    Path(__file__).parent.parent / "data" / "intent_examples.json"
)
ACT_EXAMPLES_PATH = str(
    Path(__file__).parent.parent / "data" / "act_scenario_examples.json"
)


def _detect_temporal(q_lower: str) -> str:
    for t_label, t_patterns in TEMPORAL_RULES.items():
        if any(re.search(p, q_lower) for p in t_patterns):
            return t_label
    return "unspecified"


# ── Cutoff-date/year extraction ─────────────────────────────────────────────
# TEMPORAL_RULES above only classifies the query into a coarse bucket
# (historical/current/comparative/unspecified) and never looks at *which*
# date. "before 2023", "an FIR filed in 2019", "a theft in January 2024" all
# name an actual point in time — the thing that determines whether IPC/CRPC/
# IEA or their July-2024 successors BNS/BNSS/BSA govern.
#
# BUGFIX: an earlier version of this only ever captured a bare YEAR, even
# when the query gave a full date ("occurred on 15 March 2020" only kept
# "2020"). That's harmless for any year outside 2024 (BNS/BNSS/BSA were all
# enacted 2023, so whole-year comparison already separates them cleanly from
# IPC/CRPC/IEA) — but BNS/BNSS/BSA didn't take EFFECT until 2024-07-01, a
# mid-year changeover, so "a theft in January 2024" (unambiguously pre-BNS)
# and "a theft in December 2024" (unambiguously post-BNS) both collapsed to
# the same cutoff_year=2024 and became indistinguishable. Now captures a
# full date whenever the query gives one (day+month, or just month — see
# _find_date_in), and only falls back to a bare year when it genuinely
# doesn't (a plain "2024" with no month IS genuinely ambiguous re: the
# July-2024 changeover — TemporalFilter treats that honestly rather than
# guessing a side).
_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20[0-4]\d)\b")

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))   # longest-first

# "1 june 2024", "15th march, 2020"
_DATE_DMY_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b")
# "june 1 2024", "march 15th, 2020"
_DATE_MDY_RE = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b")
# ISO numeric: 2024-06-01
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
# Day-first numeric (Indian convention, matching how this dataset's own
# CRPC/BNSS effective_date fields are written): 1/6/2024, 01-06-2024
_DATE_DMY_NUM_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
# Month + year, no day: "june 2024" — still resolves month-level precision
# against a changeover date even without a day.
_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{4}})\b")


def _find_date_in(text: str):
    """Look for an explicit day/month-bearing date in `text`. Does NOT fall
    back to a bare year — callers gate that separately (see
    _INCIDENT_WORD_RE below), since a bare year alone is a much weaker,
    higher-false-positive signal than an actual date. Returns
    (date_or_None, year_or_None, matched_text)."""
    for rx, build in (
        (_DATE_DMY_RE,     lambda m: date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))),
        (_DATE_MDY_RE,     lambda m: date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))),
        (_DATE_ISO_RE,     lambda m: date(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        (_DATE_DMY_NUM_RE, lambda m: date(int(m.group(3)), int(m.group(2)), int(m.group(1)))),
        (_MONTH_YEAR_RE,   lambda m: date(int(m.group(2)), _MONTHS[m.group(1)], 1)),
    ):
        m = rx.search(text)
        if m:
            try:
                d = build(m)
                return d, d.year, m.group(0)
            except ValueError:
                continue   # e.g. day 32 — not a real date, try the next pattern
    return None, None, ""


# "before/prior to/until 2023" -> the incident predates 2023, so the cutoff
# is the day/year BEFORE it (2023 itself may not yet have been in force).
_EXCLUSIVE_TRIGGER_RE = re.compile(
    r"\b(?:before|prior to|preceding|earlier than|until|till)\b"
)
# "in/on/during/dated 2020" or a bare year right next to it -> the incident
# IS anchored to that year, so the cutoff is that date/year itself (inclusive).
_INCLUSIVE_TRIGGER_RE = re.compile(
    r"\b(?:in|on|during|as of|dated|back in|around|circa|of)\b"
)
# A bare year only counts as an incident date (not, say, the "1860" in
# "IPC, 1860" or the "2023" in "BNS 2023") when it sits near a word that
# actually describes an event happening — otherwise plenty of legal
# queries would trip a spurious cutoff just for naming an act's year.
_INCIDENT_WORD_RE = re.compile(
    r"\b(?:case|incident|fir|complaint|crime|offence|offense|happened|"
    r"occurred|committed|took place|filed|arrested|accused)\b"
)


def _extract_cutoff(q_lower: str, window: int = 40) -> tuple:
    """Find the date/year anchoring the query to a point in time. Returns
    (cutoff_date: date|None, cutoff_year: int|None, matched_text).
    cutoff_date is set only when the query gave day/month precision (see
    module note above for why bare-year isn't just rounded to Jan 1);
    cutoff_year is always set whenever cutoff_date is (its own year), and
    may be set alone when only a bare year was found."""
    from datetime import timedelta

    # 1) Exclusive trigger ("before"/"prior to"/...) — look for a date or
    #    year right after it.
    m = _EXCLUSIVE_TRIGGER_RE.search(q_lower)
    if m:
        tail = q_lower[m.end(): m.end() + window]
        full_date, year, matched = _find_date_in(tail)
        if full_date is not None:
            return full_date - timedelta(days=1), full_date.year, f"{m.group(0)} {matched}"
        ym = _YEAR_RE.search(tail)
        if ym:
            return None, int(ym.group(1)) - 1, f"{m.group(0)} {ym.group(0)}"

    # 2) An explicit day/month-bearing date ANYWHERE is a strong, low-false-
    #    positive signal on its own — acts are never cited with a month
    #    name or a D/M/Y numeric date — so it doesn't need the incident-word
    #    gating that a bare year needs below.
    full_date, year, matched = _find_date_in(q_lower)
    if full_date is not None:
        return full_date, full_date.year, matched

    # 3) Bare year: only near an incident word or inclusive trigger.
    for ym in _YEAR_RE.finditer(q_lower):
        yr = int(ym.group(1))
        lo, hi = max(0, ym.start() - window), min(len(q_lower), ym.end() + window)
        context = q_lower[lo:hi]
        preceding = q_lower[max(0, ym.start() - 8): ym.start()]
        if _INCIDENT_WORD_RE.search(context) or _INCLUSIVE_TRIGGER_RE.search(preceding):
            return None, yr, context.strip()

    # 4) Qualitative pre-BNS phrasing with no year at all ("under the old
    #    IPC", "before BNS was enacted") still names a real point in time —
    #    the day before the known 2024-07-01 changeover.
    pm = _PRE_BNS_PHRASE_RE.search(q_lower)
    if pm:
        from datetime import timedelta
        cutoff = _BNS_CHANGEOVER - timedelta(days=1)
        return cutoff, cutoff.year, pm.group(0)

    return None, None, ""


def _detect_act_hint_literal(q_lower: str) -> tuple[str | None, dict[str, int]]:
    scores: dict[str, int] = {}
    for act, patterns in ACT_ABBREVIATION_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, q_lower))
        if score > 0:
            scores[act] = score
    best = max(scores, key=scores.get) if scores else None
    return best, scores


def rule_based_classify(query: str) -> QueryIntent:
    """Fast literal pass: only matches queries that use exact statute-name
    vocabulary. Deliberately conservative — anything short of clear evidence
    should fall through to the embedding or LLM tier rather than guess."""
    q = query.lower()

    act_hint, act_scores = _detect_act_hint_literal(q)

    INTENT_LITERAL_HINTS = {
        "definition":  [r"\bwhat is\b", r"\bdefine\b", r"\bmeaning of\b", r"\bexplain\b"],
        "punitive":    [r"\bpunishment\b", r"\bpenalty\b", r"\bimprisonment\b", r"\bfine\b"],
        "procedural":  [r"\bhow to\b", r"\bprocedure\b", r"\bsteps?\b", r"\bfile\b"],
        "case_law":    [r"\bjudgment\b", r"\bruling\b", r"\bprecedent\b", r"\bcourt held\b"],
        "statute":     [r"\bsection\b", r"\bact\b", r"\bprovision\b", r"\bstatute\b"],
    }
    scores = {intent: 0 for intent in INTENT_LITERAL_HINTS}
    for intent, patterns in INTENT_LITERAL_HINTS.items():
        for p in patterns:
            if re.search(p, q):
                scores[intent] += 1

    best_intent = max(scores, key=scores.get)
    best_score  = scores[best_intent]
    # Raised bar vs. previous version (was reachable at 2 hits / 0.67) —
    # rule tier now requires stronger literal evidence before short-circuiting
    # the semantic and LLM tiers.
    confidence  = min(best_score / 3.0, 0.9) if best_score > 0 else 0.0

    threshold    = max(1, best_score * 0.5)
    multi_labels = [intent for intent, sc in scores.items() if sc >= threshold and sc > 0]
    if not multi_labels:
        multi_labels = [best_intent if confidence > 0.3 else "statute"]

    keywords = [w for w in q.split() if len(w) > 4][:6]

    return QueryIntent(
        label      = best_intent if confidence > 0.3 else "statute",
        confidence = confidence,
        act_hint   = act_hint,
        temporal   = _detect_temporal(q),
        keywords   = keywords,
        labels     = multi_labels,
        source     = "rule",
    )


def semantic_classify(
    query: str,
    intent_matcher: SemanticMatcher,
    act_matcher: SemanticMatcher,
    fallback_temporal: str,
) -> Optional[QueryIntent]:
    """Embedding tier — catches paraphrases the literal regex tier misses.
    Returns None (never guesses) if no embed_fn is wired up."""
    if not intent_matcher.embed_fn:
        return None

    intent_matches = intent_matcher.match(query, top_k=3)
    if not intent_matches:
        return None

    best_label, best_score = intent_matches[0]
    multi_labels = [lbl for lbl, sc in intent_matches if sc >= best_score * 0.85]

    act_hint = None
    act_matches = act_matcher.match(query, top_k=1)
    if act_matches and act_matches[0][1] >= 0.5:
        act_hint = act_matches[0][0]

    return QueryIntent(
        label      = best_label,
        confidence = round(float(best_score), 3),
        act_hint   = act_hint,
        temporal   = fallback_temporal,
        keywords   = [],
        labels     = multi_labels,
        source     = "semantic",
    )


CLASSIFY_PROMPT = """You are a legal query classifier for Indian law.
Classify the following legal query and return a JSON object:
{{
  "label": "<statute | case_law | definition | procedural | punitive>",
  "confidence": <float 0.0-1.0>,
  "act_hint": "<IPC/CPC/ITA/CRPC/IEA/BNS/BNSS/POCSO/NDPS/SCST or null>",
  "temporal": "<current | historical | comparative | unspecified>",
  "keywords": ["2-5 key legal terms"],
  "reasoning": "<one sentence>"
}}
IMPORTANT: If label is 'case_law', add a note in reasoning that no case law index
exists — the system only has statutory sections and will return statute results.
Query: {query}
Return ONLY the JSON. No markdown."""


def llm_classify(query: str) -> QueryIntent:
    response = ollama.chat(
        model=OLLAMA_FAST_MODEL,
        format="json",
        # Determinism — see irac_reranker.py's llm_irac_score for why.
        options={"temperature": 0},
        messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(query=query)}],
    )
    data = json.loads(response["message"]["content"])
    primary_label = data.get("label", "statute")
    llm_labels = data.get("labels", [primary_label])
    if primary_label not in llm_labels:
        llm_labels = [primary_label] + llm_labels
    return QueryIntent(
        label      = primary_label,
        confidence = float(data.get("confidence", 0.5)),
        act_hint   = data.get("act_hint"),
        temporal   = data.get("temporal", "unspecified"),
        keywords   = data.get("keywords", []),
        reasoning  = data.get("reasoning", ""),
        labels     = [l for l in llm_labels if isinstance(l, str)][:5],
        source     = "llm",
    )


class IntentClassifier:
    """
    embed_fn is optional. Pass the shared embedding model's encode function
    to enable the semantic tier — without it, this behaves as a two-tier
    rule-then-LLM classifier (same as before, but with a higher rule
    confidence bar so more queries reach the LLM).
    """

    RULE_ACCEPT_THRESHOLD     = 0.75
    SEMANTIC_ACCEPT_THRESHOLD = 0.62

    def __init__(self, embed_fn: Optional[Callable[[list[str]], object]] = None):
        self.embed_fn = embed_fn
        self._intent_matcher = SemanticMatcher(
            label_examples=INTENT_EXAMPLES, embed_fn=embed_fn,
            examples_path=INTENT_EXAMPLES_PATH,
        )
        self._act_matcher = SemanticMatcher(
            label_examples=ACT_SCENARIO_EXAMPLES, embed_fn=embed_fn,
            examples_path=ACT_EXAMPLES_PATH,
        )

    def classify(self, query: str, raw_query: Optional[str] = None) -> QueryIntent:
        """
        query:     the text actually classified for label/act_hint/temporal —
                   in main.py this is translation.primary_query when
                   available, the LLM-rewritten "legal terminology" version
                   of the user's question.
        raw_query: BUGFIX — cutoff/date extraction below used to run against
                   `query` too, but Stage 0b's Universal Translator exists
                   specifically to strip narrative framing down to legal
                   terminology (its own prompt's few-shot examples show
                   this explicitly — a whole sentence of facts becomes
                   "wrongful arrest public servant causing hurt IPC 342
                   330"), and a date like "this incident happened on
                   January 2024" is exactly the kind of narrative detail
                   that rewrite discards. Once that happened, cutoff
                   extraction had nothing left to find, and a query that
                   very obviously named a date silently got no cutoff at
                   all — visible in production as "January 2024" showing
                   BNS_146 anyway, even though the same query passed
                   directly (bypassing translation) extracted the date
                   correctly. Defaults to `query` for any caller (or test)
                   that doesn't pass one, so this is non-breaking.
        """
        if raw_query is None:
            raw_query = query
        fast = rule_based_classify(query)
        if fast.confidence >= self.RULE_ACCEPT_THRESHOLD:
            result = fast
        else:
            semantic = semantic_classify(
                query, self._intent_matcher, self._act_matcher, fast.temporal,
            )
            if semantic and semantic.confidence >= self.SEMANTIC_ACCEPT_THRESHOLD:
                # Literal act hint (if any) is more precise than the embedding
                # guess, so prefer it when both are available.
                if fast.act_hint and not semantic.act_hint:
                    semantic.act_hint = fast.act_hint
                result = semantic
            else:
                try:
                    result = llm_classify(query)
                except Exception:
                    result = semantic or fast

        # Cutoff extraction runs regardless of which tier answered — it's a
        # deterministic regex pass over the ORIGINAL user text (raw_query),
        # not `query` — see the raw_query parameter note above for why that
        # distinction matters here specifically.
        cutoff_date, cutoff_year, cutoff_reason = _extract_cutoff(raw_query.lower())
        if cutoff_year is not None:
            result.cutoff_year   = cutoff_year
            result.cutoff_date   = cutoff_date.isoformat() if cutoff_date else None
            result.cutoff_reason = cutoff_reason
            # BUGFIX: this used to only fill in when temporal=="unspecified"
            # — but TEMPORAL_RULES's "current" list used to hardcode
            # "2023|2024|2025" (fixed above), so a query naming an actual
            # past date could ALSO trip that "current" keyword match (e.g.
            # "a murder happened in 2024" hit both the extracted-cutoff path
            # here AND the literal "2024" in the current-list, landing on
            # temporal="current" — the wrong label for a query describing a
            # past incident). An explicit extracted date/year is a more
            # specific, deliberate signal than a coincidental keyword hit,
            # so it always wins, overriding whatever the regex/semantic/LLM
            # tier guessed — not just when they came back "unspecified".
            result.temporal = "historical"
        return result


if __name__ == "__main__":
    clf = IntentClassifier()  # no embed_fn here — see main.py wiring note
    for q in ["What does IPC 512 state?", "Punishment for hacking under IT Act",
              "he passed away because the surgeon was careless"]:
        r = clf.classify(q)
        print(f"Q: {q}\n   label={r.label} conf={r.confidence:.2f} "
              f"act={r.act_hint} source={r.source}\n")