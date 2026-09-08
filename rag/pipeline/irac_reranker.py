"""
pipeline/irac_reranker.py
Novel component #4 — Lightweight IRAC-Based Reranker
"""
import re
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.chunk_structurer import StructuredChunk
from pipeline.intent_classifier import QueryIntent
from pipeline.temporal_filter import PENALTY as _VALIDITY_PENALTY
from pipeline.lexical import LexicalScorer, STOP_WORDS
from pipeline.irac_taxonomy import conclusion_score
from data.section_ref import extract_section_refs

from config import OLLAMA_FAST_MODEL, RERANKER_MODEL, IRAC_WEIGHTS, RERANK_TOP_K
import ollama

@dataclass
class RankedChunk:
    chunk:             StructuredChunk
    issue_score:       float = 0.0
    rule_score:        float = 0.0
    application_score: float = 0.0
    conclusion_score:  float = 0.0
    irac_score:        float = 0.0
    cross_enc_score:   float = 0.0
    final_score:       float = 0.0
    explanation:       str   = ""


# STOP_WORDS and the overlap metric now live in pipeline/lexical.py and are
# shared with anything else that needs them. The metric changed from raw
# Jaccard to an IDF-weighted overlap coefficient — see that module for the
# measurement showing why Jaccard scored 0.000 at the median here, which
# made this stage's ordering effectively arbitrary.
_SCORER = LexicalScorer.shared()


def _token_overlap(text_a: str, text_b: str) -> float:
    """IDF-weighted overlap coefficient in [0, 1]. Kept under the original
    name so call sites and tests read the same."""
    return _SCORER.overlap(text_a, text_b)


def metadata_irac_score(
    query:  str,
    intent: QueryIntent,
    chunk:  StructuredChunk,
) -> tuple[float, float, float, float]:
    """Cheap, deterministic IRAC component scores for one candidate.

    Every comparison here is IDF-weighted lexical overlap (see
    pipeline/lexical.py) plus the conclusion-type taxonomy (see
    pipeline/irac_taxonomy.py). This is Stage 1 of a cascade — it exists
    to give the cross-encoder and the LLM a sane starting order, not to be
    the final word.
    """
    active_labels = list(getattr(intent, "labels", None) or [])
    if intent.label and intent.label not in active_labels:
        active_labels.insert(0, intent.label)

    # ── Issue ────────────────────────────────────────────────────────────
    issue_score = 0.0
    if chunk.issue_tags:
        issue_score = _token_overlap(query, " ".join(chunk.issue_tags))

    # ── Rule ─────────────────────────────────────────────────────────────
    rule_score = 0.0
    if chunk.rule_summary:
        rule_score = _token_overlap(query, chunk.rule_summary)
    # BUGFIX: this used to test `intent.act_hint in chunk.act_name.upper()`.
    # act_hint is an act CODE ("IPC") while act_name is the full title
    # ("Indian Penal Code, 1860"), so the bonus never fired for 16 of the
    # 18 acts — and fired on the WRONG act for the other two, because
    # "ITA" is a substring of "Bharatiya Nyaya SanhITA" and "LimITAtion
    # Act", and "LA" of "UnLAwful Activities". An IT Act query therefore
    # boosted all 358 BNS sections and none of the 113 ITA ones. Compare
    # against act_code, which is exactly what act_hint is.
    if intent.act_hint and chunk.act_code and intent.act_hint.upper() == chunk.act_code.upper():
        rule_score = min(rule_score + 0.15, 1.0)

    # ── Application ──────────────────────────────────────────────────────
    app_score = _token_overlap(query, chunk.enriched_context or chunk.content)
    if chunk.issue_tags:
        app_score = min(app_score + _token_overlap(query, " ".join(chunk.issue_tags)) * 0.3, 1.0)

    # ── Conclusion ───────────────────────────────────────────────────────
    # Family matching against the dataset's own free-text vocabulary. The
    # previous exact-equality test against lowercase literals could never
    # match anything this corpus contains — see pipeline/irac_taxonomy.py.
    conc_score = conclusion_score(active_labels, chunk.conclusion_type)

    return issue_score, rule_score, app_score, conc_score


LLM_IRAC_PROMPT = """You are a legal relevance evaluator for Indian law.
Score how relevant this legal section is for answering the query.

Query: {query}
Section ({section_id}): {content}
Rule Summary: {rule_summary}

Return ONLY JSON:
{{
  "issue":       <0.0-1.0>,
  "rule":        <0.0-1.0>,
  "application": <0.0-1.0>,
  "conclusion":  <0.0-1.0>,
  "explanation": "<one sentence>"
}}
No markdown. No explanation outside JSON."""


def llm_irac_score(query: str, chunk: StructuredChunk) -> tuple[float, float, float, float, str]:
    # Gap 12 fix: use chunk.content (the section itself) for scoring,
    # not enriched_context truncated to 600 chars. The punishment clause
    # is usually in the second half of a section — 600 chars often cuts
    # it off entirely. 1400 chars captures most IPC/BNS sections in full.
    content_for_scoring = (chunk.enriched_context or chunk.content)[:1400]
    response = ollama.chat(
        model=OLLAMA_FAST_MODEL,
        format="json",
        # Determinism: Ollama's default temperature (0.8) meant this
        # scoring call — which directly decides what gets truncated —
        # produced different results on every run of diagnose_recall.py
        # over identical code, making it impossible to tell whether a
        # score change came from a real fix or from sampling noise.
        # temperature=0 makes reruns comparable.
        options={"temperature": 0},
        messages=[{"role": "user", "content": LLM_IRAC_PROMPT.format(
            query        = query,
            section_id   = chunk.section_id,
            content      = content_for_scoring,
            rule_summary = chunk.rule_summary or "",
        )}],
    )
    data = json.loads(response["message"]["content"])
    return (
        float(data.get("issue",       0.5)),
        float(data.get("rule",        0.5)),
        float(data.get("application", 0.5)),
        float(data.get("conclusion",  0.5)),
        data.get("explanation", ""),
    )


class IRACReranker:
    """Cheap lexical scoring -> cross-encoder -> LLM, in that order.

    Latency note (this stage used to dominate it). The original design ran
    llm_top_n=15 SEQUENTIAL ollama.chat calls per query here, on top of ~5
    other single-call LLM stages upstream and the final generation call —
    ~21 blocking round-trips to a local model, which is what produced the
    observed 192s avg / 305s p95. Three things address that:

      1. The cross-encoder now does the heavy ranking in ONE batched
         forward pass over all candidates, which is both cheaper and more
         accurate than the per-chunk LLM calls it displaces.
      2. llm_top_n is small (8) because it now refines an ordering that is
         already good, rather than rescuing an arbitrary one.
      3. Those calls run concurrently — they are independent I/O-bound
         requests with no shared state. The full win needs the Ollama
         server's OLLAMA_NUM_PARALLEL raised above its default of 1.
    """

    def __init__(self, llm_top_n: int = 8, cross_encoder=None, max_workers: int = 4,
                 ce_batch_size: int = 32, use_cross_encoder: bool = True):
        """use_cross_encoder=False skips Stage 2 entirely and does NOT try
        to load a model. Without it, passing cross_encoder=None always
        meant "load bge-reranker-large from disk or the network", so an
        ablation variant or a test that wanted the lexical-only path had no
        way to say so — it just silently downloaded 1.3 GB."""
        self.llm_top_n     = llm_top_n
        self.max_workers   = max_workers
        self.ce_batch_size = ce_batch_size
        # Accept an already-loaded CrossEncoder. evaluate.py's ablation
        # study builds five use_irac=True variants in a loop; each loading
        # its own copy of bge-reranker-large is what used to exhaust an 8GB
        # GPU and CUDA-OOM every variant.
        self.cross_encoder = cross_encoder if use_cross_encoder else None
        self.use_cross_enc = self.cross_encoder is not None
        if use_cross_encoder and self.cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(RERANKER_MODEL)
                self.use_cross_enc = True
            except Exception as e:
                # Loud, not silent: without the cross-encoder this stage
                # falls back to lexical-only ordering, which is a real and
                # measurable quality drop, not a neutral configuration.
                print(f"[IRACReranker] WARNING: cross-encoder {RERANKER_MODEL} "
                      f"unavailable ({e}). Falling back to lexical-only "
                      f"reranking — expect degraded precision.")

    def _weighted_irac(self, i, r, a, c) -> float:
        w = IRAC_WEIGHTS
        return w["issue"]*i + w["rule"]*r + w["application"]*a + w["conclusion"]*c

    def rerank(
        self,
        query:  str,
        intent: QueryIntent,
        chunks: list[StructuredChunk],
        top_k:  int = RERANK_TOP_K,
        direct_query: str | None = None,
    ) -> list[RankedChunk]:
        """Three-stage cascade: cheap lexical -> cross-encoder -> LLM.

        The ordering matters. Previously the LLM/cross-encoder slate was
        chosen by Stage 1 alone, which was raw Jaccard against very short
        metadata fields — median score 0.000 over a realistic candidate
        set, so the "top 8" were largely decided by ties resolved in
        retrieval order. Running the cross-encoder over EVERY candidate
        first fixes that at the root: it is one batched forward pass
        (cheaper than the 8 separate calls it replaces) and it is a far
        stronger relevance signal than any lexical heuristic. The LLM then
        refines only the head of an already-good ordering.
        """
        if not chunks:
            return []

        # Sections the user named outright ("IPC 302", "section 302 of the
        # IPC"). Parsing lives in data/section_ref.py so this agrees with
        # the retriever's own direct-lookup fast path instead of keeping a
        # second, subtly different regex.
        # direct_query is the user's own words. `query` is the reranking
        # text, which concatenates the LLM's translated query — and that
        # routinely contains invented section numbers, which would then be
        # floored at 0.85 and forced into the answer as though the user
        # had asked for them by name.
        direct_ids = set(extract_section_refs(
            direct_query if direct_query is not None else query,
            {c.section_id for c in chunks},
        ))

        # ── Stage 1: cheap metadata scoring (all candidates) ──────────────
        ranked: list[RankedChunk] = []
        for chunk in chunks:
            i, r, a, c = metadata_irac_score(query, intent, chunk)
            irac = self._weighted_irac(i, r, a, c)
            ranked.append(RankedChunk(
                chunk=chunk, issue_score=i, rule_score=r,
                application_score=a, conclusion_score=c,
                irac_score=irac, final_score=irac,
            ))

        # ── Stage 2: cross-encoder over ALL candidates (one batch) ────────
        if self.use_cross_enc and self.cross_encoder:
            self._apply_cross_encoder(query, ranked)
            for rc in ranked:
                rc.final_score = rc.irac_score * 0.35 + rc.cross_enc_score * 0.65
        else:
            for rc in ranked:
                rc.final_score = rc.irac_score

        ranked.sort(key=lambda x: x.final_score, reverse=True)

        # ── Stage 3: LLM refinement of the head ──────────────────────────
        top_n, rest = ranked[:self.llm_top_n], ranked[self.llm_top_n:]
        if top_n:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(top_n))) as pool:
                top_n = list(pool.map(lambda rc: self._llm_refine(query, rc), top_n))

        all_ranked = top_n + rest

        # ── Validity penalty, applied once, after all blending ───────────
        # Stage 4's temporal verdict has to reach the final ordering
        # somewhere: filter_active_only can deliberately re-admit an
        # AMENDED section when too few active ones survive, and without
        # this that section would compete on pure text match as though it
        # were current law. Derived from validity_label because
        # penalized_score itself is on the RRF scale, not this one.
        for rc in all_ranked:
            rc.final_score *= _VALIDITY_PENALTY.get(rc.chunk.validity_label, 1.0)

        all_ranked.sort(key=lambda x: x.final_score, reverse=True)

        # ── Guarantee explicitly-named sections survive the cutoff ────────
        if direct_ids:
            pinned = [rc for rc in all_ranked if rc.chunk.section_id in direct_ids]
            others = [rc for rc in all_ranked if rc.chunk.section_id not in direct_ids]
            for rc in pinned:
                # Scaled by the validity penalty rather than a flat floor:
                # a user naming a repealed section ("IPC 497") must still
                # get it back, but it must not be scored as though it were
                # good law — final_score also drives the confidence label.
                penalty = _VALIDITY_PENALTY.get(rc.chunk.validity_label, 1.0)
                rc.final_score = max(rc.final_score, 0.85 * penalty)
            pinned.sort(key=lambda x: x.final_score, reverse=True)
            return pinned + others[:max(0, top_k - len(pinned))]

        return all_ranked[:top_k]

    # ── Stage helpers ────────────────────────────────────────────────────

    def _apply_cross_encoder(self, query: str, ranked: list[RankedChunk]) -> None:
        """Score every candidate in ONE batched predict() call.

        bge-reranker-large has a classification head, so raw predict()
        returns an UNBOUNDED logit — BAAI's own model card says to apply a
        sigmoid. The previous code used (x+1)/2, the transform for a
        bipolar cosine score, which let a confidently-relevant pair at
        logit +6 become 3.5 and corrupt every downstream consumer that
        assumes ~[0,1] (confidence thresholds, the IRAC bars).
        """
        pairs = [
            (query, (rc.chunk.enriched_context or rc.chunk.content)[:512])
            for rc in ranked
        ]
        try:
            scores = self.cross_encoder.predict(pairs, batch_size=self.ce_batch_size)
        except Exception as e:
            print(f"[IRACReranker] cross-encoder unavailable, falling back to "
                  f"metadata ordering: {e}")
            self.use_cross_enc = False
            return
        for rc, raw in zip(ranked, scores):
            rc.cross_enc_score = _sigmoid(float(raw))

    def _llm_refine(self, query: str, rc: RankedChunk) -> RankedChunk:
        try:
            li, lr, la, lc, expl = llm_irac_score(query, rc.chunk)
        except Exception:
            return rc
        # Keep the LLM's own per-component judgments rather than collapsing
        # them into one scalar and discarding them: answer_generator's
        # "IRAC coverage" bars average exactly these fields, and showing
        # the pre-LLM estimate there made them read near-zero for sections
        # the reranker was actually confident about.
        rc.issue_score, rc.rule_score = li, lr
        rc.application_score, rc.conclusion_score = la, lc
        rc.explanation = expl

        llm_irac = self._weighted_irac(li, lr, la, lc)
        rc.final_score = rc.final_score * 0.6 + llm_irac * 0.4
        return rc


def _sigmoid(x: float) -> float:
    """Numerically stable — math.exp(-x) overflows for very negative x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
