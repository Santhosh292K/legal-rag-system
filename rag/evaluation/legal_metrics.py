"""
evaluation/legal_metrics.py
Fills two gaps in evaluate.py:

  1. Answer Coverage Score  — was named in evaluate.py's own docstring
     ("Legal: Hallucination Rate, Answer Coverage Score") but never
     implemented. Citation-recall (in evaluate.py) only checks the
     LLM's *parsed* citation objects. It says nothing about whether the
     generated prose actually explains each gold section — a model can
     emit a correct citations=[...] list while the answer text itself
     never discusses one of those sections, or vice versa (discusses a
     section in prose without a clean citation the regex/parser catches).
     Answer Coverage Score checks the prose directly.

  2. Semantic similarity — ROUGE/Token-F1/EM in evaluate.py are all
     lexical. A correct legal answer that paraphrases the gold answer
     ("up to two years' imprisonment" vs "punishable with imprisonment
     for a term which may extend to two years") scores low on ROUGE
     despite being right. This adds an embedding-cosine metric that
     reuses whatever embed_fn the pipeline already has loaded (bge-large
     via HybridRetriever) — no new model/download required. Falls back
     to a bag-of-words cosine if no embed_fn is supplied, clearly
     labelled as a degraded fallback rather than silently pretending to
     be semantic.
"""
import re
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Answer Coverage Score
# ═══════════════════════════════════════════════════════════════════════════════

# "IPC_304A" -> act="IPC", num="304A". Extend this if the dataset grows
# more act prefixes than IPC/BNS/CrPC/PCA/NI/POCSO/SC_ST etc.
_SECTION_ID_RE = re.compile(r"^([A-Z]{2,8})_([A-Z0-9]+)$")


def _section_mention_patterns(section_id: str) -> list[re.Pattern]:
    """Builds regexes matching the ways a section is naturally written in
    prose: 'Section 304A IPC', 'IPC 304A', 'IPC Section 304A', '304A IPC',
    the bare 'Section 304A' (act omitted when unambiguous in context), OR
    the model's actual bracket-citation format '[IPC_304A]'.

    BUGFIX: the pipeline's own ANSWER_PROMPT (pipeline/answer_generator.py)
    explicitly instructs the LLM to "Cite every factual claim with
    [SECTION_ID] immediately after it" and to close with
    "Based on: [list all cited section IDs]" — i.e. the model is told, and
    reliably does, cite as the literal bracketed id "[IPC_304A]", not as
    natural-language prose like "Section 304A IPC". The old pattern set
    only covered the prose forms and never matched "[ACT_NUM]" (the
    underscore isn't in the separator class, and there's no bracket
    pattern at all), so answer_coverage_score() silently scored 0.0 for
    every well-formed, correctly-cited answer — confirmed against
    results_full.json, where citation precision/recall (Table 3, computed
    from a *different* code path — the `\\[([A-Z]+(?:_[A-Z0-9]+)+)\\]`
    extractor in answer_generator.py) is ~0.3 while Ans-Coverage
    (Table 3b) was 0.000 across all 17 queries. Adding the bracket pattern
    fixes the metric to actually reflect what's in the answer text."""
    m = _SECTION_ID_RE.match(section_id)
    if not m:
        # Unknown format — fall back to a literal substring match on the
        # raw id (handles ids that already look like "304A" alone).
        return [re.compile(re.escape(section_id), re.IGNORECASE)]

    act, num = m.group(1), m.group(2)
    num_esc, act_esc, sid_esc = re.escape(num), re.escape(act), re.escape(section_id)
    patterns = [
        rf"\[{sid_esc}\b",                       # "[IPC_304A]" or "[IPC_304A, ...]"
        rf"\bsection\s+{num_esc}\s*{act_esc}\b",
        rf"\b{act_esc}\s+section\s+{num_esc}\b",
        rf"\b{act_esc}[\s/_-]*{num_esc}\b",        # underscore now included as a valid separator
        rf"\b{num_esc}\s*{act_esc}\b",
        rf"\bsection\s+{num_esc}\b",   # bare mention — weaker signal, still counted
    ]
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def answer_coverage_score(answer_text: str, gold_sections: list[str]) -> float:
    """Fraction of gold_sections that are actually *discussed* in the
    generated answer's prose (not just present in a parsed citations
    list). Returns 0.0 if there are no gold sections to check against.

    This is deliberately independent of evaluate.py's citation_metrics(),
    which measures the model's formal citation list against gold/
    retrieved ids. Coverage measures the free-text answer itself.
    """
    if not gold_sections or not answer_text:
        return 0.0

    covered = 0
    for sid in gold_sections:
        patterns = _section_mention_patterns(sid)
        if any(p.search(answer_text) for p in patterns):
            covered += 1
    return covered / len(gold_sections)


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic similarity (embedding cosine, with a labelled lexical fallback)
# ═══════════════════════════════════════════════════════════════════════════════

EmbedFn = Callable[[list[str]], "np.ndarray"]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _bow_cosine_fallback(prediction: str, gold: str) -> float:
    """Unweighted bag-of-words cosine — NOT a substitute for real semantic
    similarity, just a dependency-free fallback so the metric degrades
    gracefully instead of crashing when no embed_fn is available."""
    from collections import Counter
    p_toks = Counter(prediction.lower().split())
    g_toks = Counter(gold.lower().split())
    vocab = set(p_toks) | set(g_toks)
    if not vocab:
        return 0.0
    p_vec = np.array([p_toks.get(t, 0) for t in vocab], dtype=float)
    g_vec = np.array([g_toks.get(t, 0) for t in vocab], dtype=float)
    return _cosine(p_vec, g_vec)


def semantic_similarity(
    prediction: str,
    gold: str,
    embed_fn: Optional[EmbedFn] = None,
) -> float:
    """Cosine similarity between prediction and gold. Pass the pipeline's
    already-loaded embed_fn (e.g. `lambda texts: retriever.embed_model.encode(
    texts, normalize_embeddings=True)`) to get real semantic similarity;
    without it, falls back to bag-of-words cosine (see _bow_cosine_fallback).
    """
    if not prediction or not gold:
        return 0.0
    if embed_fn is None:
        return _bow_cosine_fallback(prediction, gold)
    vecs = np.array(embed_fn([prediction, gold]))
    return _cosine(vecs[0], vecs[1])


@dataclass
class LegalMetrics:
    """New metrics block — merge into evaluate.py's EvalResult."""
    answer_coverage:    float = 0.0
    semantic_similarity: float = 0.0
    semantic_similarity_is_fallback: bool = False   # True if embed_fn wasn't available
    # LLM-judge faithfulness (see faithfulness_judge.py) — only populated
    # when evaluate.py is run with --llm-judge, since it costs an extra
    # LLM call per query. 0.0/False by default, not "unfaithful" — check
    # faithfulness_judged before reading these as real scores.
    faithfulness_score:    float = 0.0
    contradiction_rate:    float = 0.0
    faithfulness_n_claims: float = 0.0   # avg claims/answer, sanity-check the score isn't from empty answers
    faithfulness_judged:   bool  = False