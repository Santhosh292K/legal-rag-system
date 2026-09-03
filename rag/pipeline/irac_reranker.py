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
from data.bm25_tokenizer import tokenize as _bm25_tokenize

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


# Gap 11: stop-words to strip before Jaccard token overlap.
# Common English words and legal filler terms inflate Jaccard between
# unrelated pairs since every section shares words like "the", "of", "act".
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "to", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "by", "with", "for", "on", "at", "from", "as", "into", "through",
    "that", "this", "his", "her", "its", "their", "he", "she", "it",
    "they", "who", "which", "what", "any", "all", "not", "no",
    # Legal filler terms
    "act", "section", "under", "such", "thereof", "thereto", "herein",
    "aforesaid", "said", "above", "following",
})


def _token_overlap(text_a: str, text_b: str) -> float:
    # Gap 11 fix: strip stop-words before computing Jaccard so legal filler
    # words don't inflate the score between semantically unrelated pairs.
    #
    # BUGFIX: this used to split on whitespace only (text.lower().split()),
    # the same punctuation-fusing bug found and fixed in the BM25 tokenizer
    # (data/bm25_tokenizer.py — see that file's docstring for the full
    # writeup: 42.5% of BM25 vocab tokens had punctuation stuck to them).
    # It hits this function even harder: benchmark queries are full
    # sentences ("...at 17 years of age. Is this legal?"), so the LAST
    # WORD of every query lost all overlap credit ("legal?" can never
    # equal "legal"), and this score is what Stage 1 uses to pick which
    # RERANK_TOP_K candidates even get an LLM/cross-encoder look — a
    # candidate that loses Jaccard credit here can get cut before it's
    # ever properly scored. Reuse the same shared tokenizer so query and
    # section text are tokenized identically.
    a = set(_bm25_tokenize(text_a)) - STOP_WORDS
    b = set(_bm25_tokenize(text_b)) - STOP_WORDS
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def metadata_irac_score(
    query:  str,
    intent: QueryIntent,
    chunk:  StructuredChunk,
) -> tuple[float, float, float, float]:

    q_lower = query.lower()

    # Issue score
    issue_score = 0.0
    if chunk.issue_tags:
        tag_text    = " ".join(chunk.issue_tags)
        issue_score = _token_overlap(q_lower, tag_text)
        if intent.label and chunk.conclusion_type:
            if intent.label in chunk.conclusion_type or chunk.conclusion_type in intent.label:
                issue_score = min(issue_score + 0.2, 1.0)

    # Rule score
    rule_score = 0.0
    if chunk.rule_summary:
        rule_score = _token_overlap(q_lower, chunk.rule_summary.lower())
        if intent.act_hint and intent.act_hint in chunk.act_name.upper():
            rule_score = min(rule_score + 0.15, 1.0)

    # Application score — use enriched_context
    rich_text = (chunk.enriched_context or chunk.content).lower()
    app_score = _token_overlap(q_lower, rich_text)
    if chunk.issue_tags:
        overlap   = _token_overlap(q_lower, " ".join(chunk.issue_tags))
        app_score = min(app_score + overlap * 0.3, 1.0)

    # Gap 2: conclusion score uses ALL active intent labels, picks best match.
    # Previously used only intent.label (single label), which gave punitive
    # weight=0.3 (base) to chunks that perfectly match a secondary label.
    active_labels = getattr(intent, "labels", [intent.label]) or [intent.label]
    best_conc = 0.3
    for lbl in active_labels:
        c = 0.3
        if lbl == "punitive"   and chunk.conclusion_type == "punitive":    c = 1.0
        elif lbl == "definition" and chunk.conclusion_type == "definitional": c = 1.0
        elif lbl == "procedural" and chunk.conclusion_type == "procedural":  c = 1.0
        elif chunk.conclusion_type: c = 0.5
        best_conc = max(best_conc, c)
    conc_score = best_conc

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
    # ROOT CAUSE (latency, Table 4): with the old default of llm_top_n=15,
    # every single query paid for 15 SEQUENTIAL ollama.chat calls in this
    # stage alone — on top of ~5 more single-call LLM stages earlier in
    # main.py's pipeline (translator, intent classifier, query router,
    # scenario rewriter, query expander) and 1 more for the final answer.
    # That's ~21 blocking round-trips to a local model per query, which is
    # exactly what produced the observed 192s average / 305s p95 latency in
    # Table 4 — none of it is retrieval or embedding cost, it's serial LLM
    # calls. Two independent fixes, both applied here:
    #   1. Lower llm_top_n: RERANK_TOP_K=20 and FINAL_TOP_K=10 mean only
    #      ~10 chunks ever reach the answer; the cheap metadata_irac_score
    #      pass already sorts candidates before this stage runs, so LLM
    #      refinement only needs to cover a modest margin above
    #      FINAL_TOP_K, not the full RERANK_TOP_K pool. 8 preserves that
    #      margin while cutting this stage's call count nearly in half.
    #   2. Run those (now fewer) calls concurrently via a thread pool —
    #      they're independent I/O-bound requests to the Ollama server
    #      with no shared state, so they don't need to be serial. The
    #      speedup depends on the Ollama server's OLLAMA_NUM_PARALLEL
    #      setting (concurrent.futures still helps even at 1, since it
    #      overlaps this process's own overhead, but the full win needs
    #      that server-side setting raised above its default of 1).
    def __init__(self, llm_top_n: int = 8, cross_encoder=None, max_workers: int = 4):
        self.llm_top_n      = llm_top_n
        self.max_workers    = max_workers
        # BUGFIX: same double-loading bug as HybridRetriever/CaseIndexer,
        # different model. This used to always load a fresh CrossEncoder
        # (bge-reranker-large) even when a caller already had one loaded —
        # e.g. evaluate.py's ablation_study builds 5 of its 7 variants with
        # use_irac=True, each constructing its own IRACReranker, each
        # loading a second/third/... full copy of the reranker on top of
        # whatever HybridRetriever's embed model already reused. Accept an
        # already-loaded CrossEncoder and only load a fresh one as fallback.
        self.cross_encoder  = cross_encoder
        self.use_cross_enc  = cross_encoder is not None
        if self.cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(RERANKER_MODEL)
                self.use_cross_enc = True
            except Exception:
                pass

    def _weighted_irac(self, i, r, a, c) -> float:
        w = IRAC_WEIGHTS
        return w["issue"]*i + w["rule"]*r + w["application"]*a + w["conclusion"]*c

    def rerank(
        self,
        query:  str,
        intent: QueryIntent,
        chunks: list[StructuredChunk],
        top_k:  int = RERANK_TOP_K,
    ) -> list[RankedChunk]:

        if not chunks:
            return []

        # Identify direct-hit chunks (pinned by retriever — section_id matches
        # an explicit "ACT N" reference in the query).  These must survive the
        # top_k cutoff even if the IRAC scorer ranks them low.
        import re as _re
        direct_ids: set[str] = set()
        # Gap 10 fix: expanded ACT_CODE_MAP to all 18 acts (was only 6).
        ACT_CODE_MAP = {
            "ipc":   "IPC",   "cpc":   "CPC",   "crpc":  "CRPC",
            "ita":   "ITA",   "iea":   "IEA",   "coi":   "COI",
            "bns":   "BNS",   "bnss":  "BNSS",  "bsa":   "BSA",
            "sra":   "SRA",   "tpa":   "TPA",   "ica":   "ICA",
            "ndps":  "NDPS",  "pca":   "PCA",   "pocso": "POCSO",
            "scst":  "SCST",  "uapa":  "UAPA",  "la":    "LA",
        }
        for act_abbr, sec_num in _re.findall(
            r'\b(ipc|cpc|crpc|ita|iea|coi|bns|bnss|bsa|sra|tpa|ica|ndps|pca|pocso|scst|uapa|la)'
            r'\s+(?:section\s+)?(\d+[a-z]?)\b',
            query.lower()
        ):
            act_code = ACT_CODE_MAP.get(act_abbr, "")
            if act_code:
                # BUGFIX: `sec_num.upper().zfill(3)` pads by TOTAL string
                # length, not the numeric part specifically — for a section
                # like "29a" this becomes "29A" (already 3 chars, zfill is a
                # no-op) instead of the actually-indexed "029A" (confirmed
                # against final_dataset.json: section_id zero-pads the
                # numeric prefix to 3 digits, then appends the letter suffix
                # — e.g. IPC_029A, IPC_120B). Split digits from any letter
                # suffix and zero-pad only the digits. Also try the
                # unpadded form: every act pads to 3 digits in section_id
                # EXCEPT CRPC, whose section_ids are a genuine mix of
                # 1/2/3-digit widths (unpadded, as authored) — rather than
                # special-case CRPC, add both candidates for every act.
                sec_norm = sec_num.upper()
                m = re.match(r'^(\d+)([A-Z]?)$', sec_norm)
                if m:
                    digits, letter = m.groups()
                    direct_ids.add(f"{act_code}_{digits.zfill(3)}{letter}")
                    direct_ids.add(f"{act_code}_{digits}{letter}")
                else:
                    direct_ids.add(f"{act_code}_{sec_norm}")

        # Stage 1: fast metadata scoring
        ranked = []
        for chunk in chunks:
            i, r, a, c = metadata_irac_score(query, intent, chunk)
            irac        = self._weighted_irac(i, r, a, c)
            ranked.append(RankedChunk(
                chunk=chunk, issue_score=i, rule_score=r,
                application_score=a, conclusion_score=c,
                irac_score=irac, final_score=irac,
            ))
        ranked.sort(key=lambda x: x.irac_score, reverse=True)

        # Stage 2: LLM scoring for top-N
        top_n, rest = ranked[:self.llm_top_n], ranked[self.llm_top_n:]

        def _score_one(rc: "RankedChunk") -> "RankedChunk":
            try:
                li, lr, la, lc, expl = llm_irac_score(query, rc.chunk)
                llm_irac  = self._weighted_irac(li, lr, la, lc)
                final     = rc.irac_score * 0.5 + llm_irac * 0.5

                # BUGFIX: li/lr/la/lc — the LLM's own issue/rule/application/
                # conclusion judgments — used to be folded into llm_irac
                # (a single scalar for final_score) and then discarded.
                # rc.issue_score/rule_score/application_score/
                # conclusion_score stayed at their Stage-1 metadata-only
                # values FOREVER, even for chunks that just got a real LLM
                # look. Those Stage-1 values are plain Jaccard token overlap
                # between the query and a typically-short rule_summary/
                # issue_tags string — structurally biased near zero (a
                # 9-token rule_summary against a 30+ token query, even with
                # perfect topical overlap, caps Jaccard's numerator well
                # below its denominator). answer_generator.py's
                # _build_irac_summary() averages exactly these fields into
                # the "IRAC coverage" bars shown to the user — so those bars
                # were showing the crude pre-LLM estimate, not what the
                # reranker actually concluded, which is why they read as
                # near-zero (0.01-0.08) even for chunks confidently cited in
                # the final answer. Overwrite with the LLM's own per-
                # component scores so the displayed breakdown matches the
                # judgment that actually produced final_score.
                rc.issue_score       = li
                rc.rule_score        = lr
                rc.application_score = la
                rc.conclusion_score  = lc

                if self.use_cross_enc and self.cross_encoder:
                    ce        = float(self.cross_encoder.predict(
                        [(query, (rc.chunk.enriched_context or rc.chunk.content)[:512])]
                    )[0])
                    # BUGFIX: RERANKER_MODEL (BAAI/bge-reranker-large) is a
                    # classification-head cross-encoder — its raw .predict()
                    # output is an UNBOUNDED logit (BAAI's own model card:
                    # apply sigmoid to get a [0,1] relevance score), not a
                    # value already bounded to [-1, 1]. `(ce + 1) / 2` is the
                    # right transform for a cosine-similarity-style bipolar
                    # score, which this isn't — it let ce_norm land far
                    # outside [0, 1] (e.g. a confidently-relevant pair at raw
                    # score +6 became ce_norm=3.5), corrupting final_score
                    # and everything downstream that assumes it's ~[0,1]
                    # (answer_generator.py's confidence thresholds, the IRAC
                    # bars). Sigmoid is the model's documented mapping.
                    # Numerically-stable sigmoid — math.exp(-ce) can overflow
                    # for a very negative logit if computed the naive way.
                    ce_norm   = (1.0 / (1.0 + math.exp(-ce)) if ce >= 0
                                 else math.exp(ce) / (1.0 + math.exp(ce)))
                    final     = final * 0.7 + ce_norm * 0.3
                    rc.cross_enc_score = ce_norm

                rc.final_score = final
                rc.explanation = expl
            except Exception:
                rc.final_score = rc.irac_score
            return rc

        # Concurrent, not sequential — see the __init__ docstring note on
        # why this stage dominated per-query latency. Order doesn't matter
        # here since everything is re-sorted by final_score right after.
        if top_n:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(top_n))) as pool:
                llm_done = list(pool.map(_score_one, top_n))
        else:
            llm_done = []

        all_ranked = llm_done + rest

        # BUGFIX: chunk.penalized_score (Stage 4's temporal-validity penalty
        # — see temporal_filter.py) was computed and threaded all the way
        # through StructuredChunk but never once READ here — this stage
        # re-ranks purely by textual/semantic match to the query, with zero
        # weight given to how valid/current a section actually is. Harmless
        # for the common case (filter_active_only's default output is all
        # penalty=1.0 "active"/historically-valid chunks), but its own
        # "fewer than 3 valid results -> rescue amended sections" fallback
        # deliberately lets a penalty=0.70 AMENDED section back in — and
        # without this, that section competed for the final answer on pure
        # text overlap, exactly as if it weren't amended at all. Re-derive
        # the penalty ratio from validity_label (penalized_score itself is
        # RRF-scaled, not on final_score's ~0-1 scale, so it can't be
        # multiplied in directly) and apply it once, after all blending, so
        # it can't get diluted by being applied before the LLM/cross-encoder
        # stages average it back out.
        for rc in all_ranked:
            rc.final_score *= _VALIDITY_PENALTY.get(rc.chunk.validity_label, 1.0)

        all_ranked.sort(key=lambda x: x.final_score, reverse=True)

        # Always include direct-hit chunks (explicit section references like "ipc 1")
        # even if they scored low on IRAC — the user asked for them specifically.
        if direct_ids:
            pinned  = [rc for rc in all_ranked if rc.chunk.section_id in direct_ids]
            others  = [rc for rc in all_ranked if rc.chunk.section_id not in direct_ids]
            # Give pinned chunks a score boost so they appear at top of the window.
            #
            # BUGFIX: this used to floor final_score at a flat 0.85 regardless
            # of validity_label — but the validity penalty (_VALIDITY_PENALTY,
            # applied just above, before this block) had ALREADY been
            # multiplied into final_score by this point. A flat max(score,
            # 0.85) completely erased that penalty for any section the user
            # named explicitly by number: a repealed/expired/not-yet-enacted
            # section (e.g. "IPC 497" — adultery, struck down) came back
            # scored as if it were fully valid current law. The [WARNING]
            # tag still renders in the answer text, but final_score also
            # feeds answer_generator.py's confidence averaging — so a query
            # about a repealed section could get reported as HIGH confidence,
            # exactly backwards for the case that most needs a low-confidence
            # flag. Scale the floor by the same validity penalty instead:
            # still guarantees a directly-named section is included and
            # ranked near the top of its own validity band, without
            # pretending invalid law is as trustworthy as active law.
            for rc in pinned:
                penalty = _VALIDITY_PENALTY.get(rc.chunk.validity_label, 1.0)
                rc.final_score = max(rc.final_score, 0.85 * penalty)
            result = pinned + others[:max(0, top_k - len(pinned))]
        else:
            result = all_ranked[:top_k]

        return result