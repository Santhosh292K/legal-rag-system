"""
evaluation/faithfulness_judge.py

evaluate.py has always imported `judge_faithfulness` and
`context_from_citations` from this module (see evaluate.py's --llm-judge
flag and its Table 3b "Faithfulness" / "Contra %" columns), but the
module itself was never checked in — any run of evaluate.py failed at
import time, whether or not --llm-judge was actually passed. This file
fills that gap.

What "faithfulness" means here, precisely:
  Citation-recall/precision (evaluate.py's citation_metrics) only checks
  whether the LLM's *parsed* [SECTION_ID] citation list matches gold/
  retrieved ids. It says nothing about whether the generated PROSE's
  individual factual claims actually follow from the retrieved section
  content — a model can cite the right section and then still assert
  something that section doesn't say (wrong punishment range, wrong
  procedural step, an invented exception). Faithfulness judging is the
  only mechanism in this eval suite that checks that.

  This is judged against the retrieved CONTEXT (answer.citations content),
  NOT against gold_answer — a faithful answer can legitimately disagree
  with gold_answer's phrasing/scope, but it must never claim something
  its own cited sources don't support. Conflating this with gold_answer
  comparison would penalize a model for being faithful to sources that
  happen to phrase something differently than the benchmark's reference
  answer.

Method: single ollama call per query.
  1. Decompose ANSWER into atomic factual claims.
  2. For each claim, judge it against CONTEXT as one of:
       supported     — CONTEXT directly backs the claim
       contradicted   — CONTEXT states something that conflicts with it
       unsupported    — CONTEXT neither confirms nor conflicts (untethered)
  faithfulness_score = supported / n_claims
  contradiction_rate = contradicted / n_claims

This is a *local* LLM judge (via ollama, same as the rest of the
pipeline — see pipeline/answer_generator.py's ANSWER_PROMPT for the
sibling pattern) — it is not a ground-truth oracle. Treat
faithfulness_score as directional signal, not an exact number: rerun
with a stronger OLLAMA_ANSWER_MODEL if borderline claims matter, and
consider spot-checking a sample of judged claims manually before citing
this in a paper table.

Usage (already wired into evaluate.py):
    python3 evaluation/evaluate.py evaluation/benchmark_scenarios.json --llm-judge
"""
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import ollama

sys.path.append(str(Path(__file__).parent.parent))
from config import OLLAMA_ANSWER_MODEL

if TYPE_CHECKING:
    from pipeline.answer_generator import Citation


# ═══════════════════════════════════════════════════════════════════════════════
# Context assembly
# ═══════════════════════════════════════════════════════════════════════════════

# Per-citation content cap. Mirrors answer_generator.py's
# _build_context (2000 chars/section) — same rationale: enough for the
# judge to see the substantive text without blowing the local model's
# context window across 5-10 cited sections.
_MAX_CONTENT_CHARS = 2000


def context_from_citations(citations: "list[Citation]") -> str:
    """Builds the same kind of [SECTION_ID]/Act/Category/Content block
    answer_generator.py's _build_context feeds the answer-generation
    prompt, but from the LLM's own parsed `citations` list rather than
    the reranker's RankedChunk list — i.e. exactly the source material
    the answer claims to be grounded in, which is what faithfulness
    should be judged against (not the full retrieved set, and not
    gold_answer).
    """
    if not citations:
        return ""

    parts = []
    for c in citations:
        content = (getattr(c, "content", "") or "")[:_MAX_CONTENT_CHARS]
        validity = getattr(c, "validity", "") or ""
        warning = f" [WARNING: {validity.upper()}]" if validity and validity != "active" else ""
        parts.append(
            f"[{c.section_id}]{warning}\n"
            f"Act: {getattr(c, 'act_name', '')}\n"
            f"Category: {getattr(c, 'category', '')}\n"
            f"Content: {content}\n"
        )
    return "\n---\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-judge faithfulness
# ═══════════════════════════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """You are auditing a legal RAG system's generated answer for faithfulness to its retrieved source material. You are NOT judging whether the answer is a good answer, and you are NOT comparing it to any reference answer — only whether each claim it makes follows from the CONTEXT given below.

CONTEXT (the retrieved legal section text the answer must be grounded in):
{context}

ANSWER (the system's generated answer, to be audited):
{answer}

Task:
1. Break the ANSWER into its individual atomic factual claims (discrete statements of fact — skip hedges, transitions, and section-list restatements like "Based on: [...]").
2. For each claim, decide against CONTEXT ONLY:
   - "supported": CONTEXT directly backs this claim
   - "contradicted": CONTEXT states something that conflicts with this claim
   - "unsupported": CONTEXT neither confirms nor conflicts with this claim

Return ONLY a JSON object of exactly this form, no markdown fences, no commentary:
{{
  "claims": [
    {{"claim": "<claim text, verbatim or lightly trimmed from the answer>", "verdict": "supported"}},
    {{"claim": "<claim text>", "verdict": "contradicted"}},
    {{"claim": "<claim text>", "verdict": "unsupported"}}
  ]
}}
"""

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_VALID_VERDICTS = {"supported", "contradicted", "unsupported"}


@dataclass
class FaithfulnessResult:
    faithfulness_score: float = 0.0
    contradiction_rate: float = 0.0
    n_claims:            int  = 0
    claims:              list = field(default_factory=list)  # [{"claim":..., "verdict":...}, ...]
    judge_failed:         bool = False
    error:                str  = ""


def _parse_claims(raw_content: str) -> list[dict]:
    """Parses the judge's JSON response, tolerating the common ways a
    local model deviates from strict JSON (markdown fences, an
    unwrapped list instead of {"claims": [...]})."""
    cleaned = _JSON_FENCE_RE.sub("", raw_content).strip()
    data = json.loads(cleaned)

    if isinstance(data, list):
        claims = data
    else:
        claims = data.get("claims", [])

    out = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        verdict = str(c.get("verdict", "")).strip().lower()
        if verdict not in _VALID_VERDICTS:
            continue
        out.append({"claim": str(c.get("claim", "")).strip(), "verdict": verdict})
    return out


def judge_faithfulness(
    answer_text: str,
    context: str,
    model: str = OLLAMA_ANSWER_MODEL,
) -> FaithfulnessResult:
    """Judges whether each atomic claim in `answer_text` is supported by
    `context`. Returns judge_failed=True (not a silent 0.0) on any
    failure — empty inputs, a malformed/non-JSON model response, an
    ollama connection error, or a response with zero parseable claims —
    so callers (evaluate.py) can distinguish "genuinely unfaithful
    answer" from "the judge call itself didn't work" and skip failed
    judgments rather than averaging in a wrong zero.
    """
    if not answer_text or not answer_text.strip():
        return FaithfulnessResult(judge_failed=True, error="empty answer_text")
    if not context or not context.strip():
        return FaithfulnessResult(judge_failed=True, error="empty context (no citations to judge against)")

    try:
        response = ollama.chat(
            model=model,
            format="json",
            messages=[{
                "role": "user",
                "content": FAITHFULNESS_PROMPT.format(context=context, answer=answer_text),
            }],
        )
        raw_content = response["message"]["content"]
        claims = _parse_claims(raw_content)
    except Exception as e:
        return FaithfulnessResult(judge_failed=True, error=f"{type(e).__name__}: {e}")

    if not claims:
        # Judge ran but extracted nothing usable — treat as a failed
        # judgment, not a perfect (or zero) faithfulness score. A
        # genuinely empty/non-factual answer ("I don't know") should
        # not silently register as faithfulness_score=0.0 in the average.
        return FaithfulnessResult(judge_failed=True, error="judge returned zero parseable claims")

    n = len(claims)
    supported = sum(1 for c in claims if c["verdict"] == "supported")
    contradicted = sum(1 for c in claims if c["verdict"] == "contradicted")

    return FaithfulnessResult(
        faithfulness_score=round(supported / n, 4),
        contradiction_rate=round(contradicted / n, 4),
        n_claims=n,
        claims=claims,
        judge_failed=False,
    )