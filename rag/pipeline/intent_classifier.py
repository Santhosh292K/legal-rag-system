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
    "historical": [
        r"\b(was|were|had|used to|earlier|before\s+\d{4}|old law)\b",
        r"\b(before\s+(bns|bnss|bsa)\b)",
        r"\b(ipc|crpc|iea)\s+(before|prior|replaced|superseded)\b",
        r"\b(replaced|superseded)\s+by\s+(bns|bnss|bsa)\b",
        r"\bunder\s+(the\s+)?old\s+(ipc|crpc|penal\s+code)\b",
        r"\bpre[- ]bns\b",
        r"\bwhen\s+(ipc|crpc)\s+was\s+in\s+force\b",
    ],
    "current": [
        r"\b(current|now|today|present|latest|2023|2024|2025|new law)\b",
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

    def classify(self, query: str) -> QueryIntent:
        fast = rule_based_classify(query)
        if fast.confidence >= self.RULE_ACCEPT_THRESHOLD:
            return fast

        semantic = semantic_classify(
            query, self._intent_matcher, self._act_matcher, fast.temporal,
        )
        if semantic and semantic.confidence >= self.SEMANTIC_ACCEPT_THRESHOLD:
            # Literal act hint (if any) is more precise than the embedding
            # guess, so prefer it when both are available.
            if fast.act_hint and not semantic.act_hint:
                semantic.act_hint = fast.act_hint
            return semantic

        try:
            return llm_classify(query)
        except Exception:
            return semantic or fast


if __name__ == "__main__":
    clf = IntentClassifier()  # no embed_fn here — see main.py wiring note
    for q in ["What does IPC 512 state?", "Punishment for hacking under IT Act",
              "he passed away because the surgeon was careless"]:
        r = clf.classify(q)
        print(f"Q: {q}\n   label={r.label} conf={r.confidence:.2f} "
              f"act={r.act_hint} source={r.source}\n")