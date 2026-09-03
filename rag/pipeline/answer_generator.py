"""
pipeline/answer_generator.py
Grounded answer generation with sentence-level citation mapping.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
import sys

import ollama

sys.path.append(str(Path(__file__).parent.parent))
from config import FINAL_TOP_K, RERANK_TOP_K
from pipeline.irac_reranker     import RankedChunk
from pipeline.intent_classifier import QueryIntent
from pipeline.section_pinner    import PIN_EXPLANATION

from config import OLLAMA_ANSWER_MODEL


@dataclass
class Citation:
    section_id: str
    act_name:   str
    category:   str
    content:    str
    validity:   str
    warning:    str


# Shared between _assess_confidence's "medium" band and generate()'s prompt
# selection, so both agree on what "weak retrieval" means rather than two
# independently-tuned magic numbers drifting apart over time.
LOW_RELEVANCE_THRESHOLD = 0.40


@dataclass
class LegalAnswer:
    query:        str
    answer:       str
    citations:    list[Citation] = field(default_factory=list)
    warnings:     list[str]      = field(default_factory=list)
    intent:       str            = ""
    confidence:   str            = "medium"
    irac_summary: dict           = field(default_factory=dict)
    # Gap 23: section IDs actually in the reranker's top-K list, not just
    # those the LLM chose to cite. Evaluators use this for true retrieval recall.
    retrieved_section_ids: list[str] = field(default_factory=list)


ANSWER_PROMPT = """You are a precise legal assistant specializing in Indian law.
You will be given legal sections retrieved from a database. Your job is to answer the query using the content of those sections DIRECTLY and LITERALLY.

CRITICAL RULES:
- The sections provided ARE the answer source. Trust them completely.
- NEVER say a section "is not listed" or "not found" — if it appears below, it IS found.
- NEVER express uncertainty about whether a section exists — it was retrieved for you.
- Read the Content field of each section carefully and state what it says plainly.
- Do NOT add disclaimers like "additional sections would be required" or "none matches precisely".
- Cite every factual claim with [SECTION_ID] immediately after it.
- If a section is marked AMENDED or REPEALED, note it.
- End with: "Based on: [list all cited section IDs]"

Query: {query}
Query type: {intent}

Retrieved Legal Sections:
{context}

Answer directly and factually using the section content above:"""


# Used instead of ANSWER_PROMPT when either (a) an upstream stage already
# knows a specific act central to this query isn't in the database at all
# (domain_router's regex signals / universal_translator's LLM-flagged
# gaps — see main.py's `all_gaps`), or (b) retrieval quality itself is
# weak (LOW_RELEVANCE_THRESHOLD, now measured off the best few candidates
# — see generate()'s weak_retrieval). ANSWER_PROMPT's "trust it completely
# / never express uncertainty" rules are right when retrieval actually
# found the answer, but applied unconditionally they forced the model to
# confidently answer e.g. "the punishment for child labour is X" using
# only tangentially-related sections (child kidnapping/trafficking, not
# child *employment* law) — correct-sounding, wrong. This variant is
# explicitly allowed, and instructed, to say so instead of stretching a
# weak match to fit.
#
# BUGFIX: this used to instruct the model to lead with the KNOWN GAPS
# disclaimer UNCONDITIONALLY whenever known_gaps was non-empty, with no
# instruction to first check whether the retrieved sections actually
# answer the question anyway. A "domestic violence... dowry" query
# correctly flags the dedicated Protection of Women from Domestic
# Violence Act as a gap (it genuinely isn't indexed) — but the CRIMINAL
# PUNISHMENT the query actually asked about has always been governed by
# IPC 498A/304B (now BNS 85/80), which this database DOES have and did
# retrieve. The model dutifully opened with "this database does not
# contain specific sections... that directly addresses domestic violence"
# and then went on to correctly cite exactly the sections that do — a
# self-contradicting answer. A section under a differently-named act is
# still the direct, correct legal answer, not a "related" one, and the
# prompt now asks the model to make that call itself before choosing
# which framing to use, rather than always leading with the disclaimer.
ANSWER_PROMPT_CAUTIOUS = """You are a precise legal assistant specializing in Indian law.
You will be given legal sections retrieved from a database, for a query where either
retrieval confidence is LOW or a law the query is really about is KNOWN to be missing
from this database entirely (see KNOWN GAPS below).

CRITICAL RULES:
- First, judge for yourself whether the Retrieved Legal Sections below actually,
  substantively answer the query — state the specific rule/punishment/procedure asked
  about, not just touch the same general topic. A section often lives under a
  DIFFERENT act than a layperson would expect (e.g. "domestic violence" is usually
  answered by cruelty/dowry-death provisions in the penal code, not a dedicated
  Domestic Violence Act) — that is still a correct, DIRECT answer, even when KNOWN
  GAPS also names a differently-titled act as missing.
- If they DO substantively answer it: answer directly and confidently, exactly as you
  would with no gap at all. Only add a brief closing note about a KNOWN GAPS act if it
  would meaningfully change the answer (e.g. it covers a different remedy — a civil
  protection order, say — that these sections don't). Do not lead with a disclaimer
  that undersells a correct, on-point answer.
- If they do NOT substantively answer it (only tangentially related — e.g. cover
  related conduct from a different angle): THEN say so plainly as your FIRST sentence —
  e.g. "This database does not include the {{act}}, so it cannot answer this directly."
  — and present what you found as RELATED provisions, not the direct answer. Do not
  phrase content from a section about a different offence as if it were "the punishment
  for X" when the query asked about X specifically.
- Still cite every claim you do make with [SECTION_ID], and never invent a section or
  content not present below.
- If none of the retrieved sections are meaningfully related to the query, say plainly
  that no relevant provision was found in this database, instead of stretching one to fit.
- End with: "Based on: [list all cited section IDs]" (omit this line if you cited none).

Query: {query}
Query type: {intent}

KNOWN GAPS (acts relevant to this query that this database does not contain):
{gap_notice}

Retrieved Legal Sections:
{context}

Answer, being explicit about what this database does and doesn't cover:"""


FUSED_ANSWER_PROMPT = """You are a precise legal assistant specializing in Indian law.
You are answering a question about a specific uploaded case, using two
DIFFERENT kinds of source material. Keep them clearly separated — never
blend a case fact and a legal rule into one unattributed sentence.

CRITICAL RULES:
- Everything under "CASE DOCUMENT EXCERPTS" is a FACT ABOUT THIS CASE — state it as
  what the document says, not as established truth beyond the document.
- Everything under "APPLICABLE LAW" is the general legal rule — cite every
  legal claim with [SECTION_ID] immediately after it.
- Structure your answer in two clear parts: what the document states, then
  what the law says applies to those facts.
- NEVER invent a case fact that isn't in the excerpts below.
- NEVER cite a section that isn't in the applicable law list below.
- If the case documents don't address part of the question, say so plainly
  instead of guessing.
- End with: "Based on: [list all cited section IDs]"

Query: {query}

CASE DOCUMENT EXCERPTS:
{case_context}

APPLICABLE LAW:
{statute_context}

Answer, keeping case facts and legal rules clearly separated:"""


DOCUMENT_ONLY_PROMPT = """You are answering a question using ONLY the uploaded case
document excerpts below — no external legal knowledge, no statute law.

CRITICAL RULES:
- Answer the specific question asked, directly and concisely.
- Only state what the excerpts actually say — never infer or assume a fact
  that isn't explicitly present.
- If the excerpts don't contain the answer, say so plainly instead of
  guessing or padding with unrelated content from the excerpts.
- Reference which document/section the answer came from, e.g. "(from the FIR's complainant section)".

Query: {query}

CASE DOCUMENT EXCERPTS:
{case_context}

Answer the question directly:"""


class AnswerGenerator:

    @staticmethod
    def _select_top(ranked: list[RankedChunk], top_k: int) -> list[RankedChunk]:
        """Replaces the old `ranked[:top_k]` positional slice, which silently
        dropped every pinned/re-injected section: main.py appends pinned
        sections to the END of `ranked` (see 'Pinned section rescue' there),
        so a plain slice of the first top_k items never even considered
        them, regardless of how many were correctly pinned. Pinned sections
        are guaranteed a spot here — the whole point of pinning is that
        they're not supposed to be droppable by a slot-count cap — with any
        remaining budget filled by the next highest-scoring non-pinned
        chunks. If pinned sections alone exceed top_k, all of them still
        get included; the context just runs a bit longer than top_k for
        that query, which is the correct tradeoff over silently dropping a
        deterministically-identified relevant section.

        BUGFIX: Rocchio feedback (main.py Stage 6.5) and KG augmentation
        (Stage 6.75) exist for the same reason pinning does — to rescue a
        section the earlier stages missed — but they were appended to
        `ranked` with a flat discounted score (0.30 / 0.35) and then made
        to compete purely on final_score against a list that, by this
        point, is usually already sitting at RERANK_TOP_K candidates whose
        scores were mostly earned via real IRAC/LLM/cross-encoder scoring.
        A flat 0.30-0.35 rarely clears that bar, so these two stages could
        successfully find a missed gold section (visible in the
        'post_rocchio'/'post_kg' debug_trace stages) and then lose it again
        right here — confirmed against the diagnose_recall.py run this was
        written from (IPC_505: present at post_kg, absent from final).
        Running the stage at all only pays off if a slot here can't be
        stolen back by score alone, so — same reasoning as pinning — give
        Rocchio/KG additions a reserved slot too, ranked among themselves
        by their (still-discounted) score so a stronger rescued match still
        wins over a weaker one when both are competing for reserved space."""
        def _is_reserved(r: "RankedChunk") -> bool:
            return (r.explanation == PIN_EXPLANATION
                    or r.explanation == "Rocchio pseudo-relevance feedback"
                    or r.explanation.startswith("KG-augmented"))

        pinned      = [r for r in ranked if r.explanation == PIN_EXPLANATION]
        rescued     = sorted(
            (r for r in ranked if _is_reserved(r) and r.explanation != PIN_EXPLANATION),
            key=lambda r: r.final_score, reverse=True,
        )
        reserved    = pinned + rescued
        others      = sorted(
            (r for r in ranked if not _is_reserved(r)),
            key=lambda r: r.final_score, reverse=True,
        )
        remaining   = max(0, top_k - len(reserved))
        return reserved + others[:remaining]

    def _build_context(self, ranked: list[RankedChunk], top_k: int) -> str:
        parts = []
        for rc in self._select_top(ranked, top_k):
            c       = rc.chunk
            warning = f" [WARNING: {c.validity_label.upper()}]" if c.validity_label != "active" else ""
            # Use enriched_context (parent + self + children) at full length —
            # Stage 5 already assembled it; slicing to 600 chars discards most
            # of the hierarchy context. Raise to 2000 so the LLM sees the full window.
            context_text = (c.enriched_context or c.content)[:2000]
            parts.append(
                f"[{c.section_id}]{warning}\n"
                f"Act: {c.act_name}\n"
                f"Category: {c.category}\n"
                f"Content: {context_text}\n"
                f"Rule summary: {c.rule_summary}\n"
            )
        return "\n---\n".join(parts)

    def _normalize_citations(self, answer_text: str, known_ids: frozenset[str] = frozenset()) -> str:
        """Gap 19: normalize alternate citation styles to [ACT_NNN] bracket form
        before extraction so they're not silently lost.

        Handles:
          - (Section 302 IPC) / (Sec. 302 IPC)
          - u/s 302 IPC / u/s. 302
          - IPC Section 302 / under Section 302 of IPC
          - Section 66B of IT Act
        All are converted to [IPC_302] / [ITA_066B] canonical form.

        known_ids: the real section_ids actually in play for this answer
        (ranked's chunk.section_id set) — see _fmt_sec for why this matters
        beyond just the usual zero-pad guess.
        """
        import re as _re
        ACT_ALIAS = {
            "ipc": "IPC", "indian penal code": "IPC",
            "cpc": "CPC", "civil procedure code": "CPC",
            "crpc": "CRPC", "cr.p.c": "CRPC",
            "ita": "ITA", "it act": "ITA", "information technology act": "ITA",
            "iea": "IEA", "evidence act": "IEA",
            "bns": "BNS", "bnss": "BNSS", "bsa": "BSA",
            "pocso": "POCSO", "ndps": "NDPS", "scst": "SCST",
            "coi": "COI", "constitution": "COI",
        }

        def _map_act(name: str) -> str:
            return ACT_ALIAS.get(name.lower().strip(), name.upper().strip())

        def _fmt_sec(sec: str, act: str) -> str:
            # BUGFIX: this left a lettered section (e.g. "29a" from "Section
            # 29a IPC") lowercase and unpadded — sec_clean.isdigit() is False
            # whenever a letter is present, so the zfill branch never ran for
            # exactly the sections that need it most. The real indexed IDs
            # (final_dataset.json) are uppercase with the numeric part
            # zero-padded to 3 digits BEFORE the letter suffix — "IPC_029A",
            # not "IPC_29a" or "IPC_29A". _extract_citations matches this
            # normalized bracket form against chunk.section_id verbatim, so a
            # mismatch here silently dropped an otherwise-correct citation
            # the model wrote in prose form instead of bracket form.
            sec_clean = (sec.strip().lstrip("0") or "0").upper()
            m = _re.match(r'^(\d+)([A-Z]?)$', sec_clean)
            if m:
                digits, letter = m.groups()
                padded, raw = f"{digits.zfill(3)}{letter}", f"{digits}{letter}"
                # Every act zero-pads to 3 digits in section_id EXCEPT CRPC,
                # whose section_ids are a genuine mix of 1/2/3-digit widths
                # (confirmed against final_dataset.json — unpadded, as
                # authored). Rather than hardcode that one exception, prefer
                # whichever candidate is an ACTUAL section_id among this
                # answer's retrieved chunks; only guess (zero-padded) when
                # neither is known, same as before this fix.
                if f"{act}_{raw}" in known_ids:
                    sec_clean = raw
                else:
                    sec_clean = padded
            return f"[{act}_{sec_clean}]"

        # Pattern: (Section 302 IPC) or (Sec 302 IPC)
        text = _re.sub(
            r'\((?:sec(?:tion)?\.?\s+)(\d+[a-z]?)\s+([a-z .]+?)\)',
            lambda m: _fmt_sec(m.group(1), _map_act(m.group(2))),
            answer_text, flags=_re.IGNORECASE
        )
        # Pattern: under Section 302 of the IPC / under IPC Section 302
        text = _re.sub(
            r'(?:under\s+)?(?:section\s+)(\d+[a-z]?)\s+(?:of\s+(?:the\s+)?)?([a-z .]+?)(?=[,;. ])',
            lambda m: _fmt_sec(m.group(1), _map_act(m.group(2))),
            text, flags=_re.IGNORECASE
        )
        # Pattern: u/s 302 IPC / u/s. 66B ITA
        text = _re.sub(
            r'u/s\.?\s+(\d+[a-z]?)\s+([A-Za-z]+)',
            lambda m: _fmt_sec(m.group(1), _map_act(m.group(2))),
            text, flags=_re.IGNORECASE
        )
        return text

    def _extract_citations(self, answer_text: str, shown: list[RankedChunk]) -> list[Citation]:
        # BUGFIX: this used to be called with the full `ranked` list (up to
        # RERANK_TOP_K=20 candidates) as the "known valid" universe, not
        # `shown` (the exact top_k subset _build_context actually put in
        # the LLM's prompt). That let a hallucinated citation for a real
        # section_id sitting lower in `ranked` — never shown to the model
        # this call — pass verification as if it were grounded, because it
        # existed *somewhere* in the broader candidate pool. The citation
        # map here must only ever contain what the model could actually
        # have read, or "verified" citation doesn't mean grounded.
        chunk_map = {rc.chunk.section_id: rc.chunk for rc in shown}
        # Gap 19: normalize alternate citation formats first, then extract
        normalized_text = self._normalize_citations(answer_text, known_ids=frozenset(chunk_map))
        # Matches any section ID of the form: LETTERS_ALPHANUMERIC(_ALPHANUMERIC)*
        # e.g. IPC_302, IPC_302A, CPC_1_A, ITA_66B, COI_21_1
        cited_ids = set(re.findall(r"\[([A-Z]+(?:_[A-Z0-9]+)+)\]", normalized_text))
        citations = []
        for sid in cited_ids:
            if sid in chunk_map:
                c = chunk_map[sid]
                citations.append(Citation(
                    section_id = sid,
                    act_name   = c.act_name,
                    category   = c.category,
                    content    = c.content[:300],
                    validity   = c.validity_label,
                    warning    = c.warning,
                ))
        return citations

    def _assess_confidence(
        self,
        ranked:    list[RankedChunk],
        top_k:     int,
        citations: list | None = None,
    ) -> str:
        """Gap 18 fix: confidence now measures citation recall (answer quality),
        not just retrieval quality (reranker scores).

        A query can have high-scoring relevant sections but still produce a
        hallucinated or incomplete answer; blending citation recall into the
        confidence score surfaces that mismatch.

          high   = retrieval looks good AND >=50% of offered sections were cited
          medium = retrieval decent OR a meaningful fraction of citations verified
          low    = neither retrieval nor citations clear a meaningful bar

        BUGFIX: the old medium-band test was
            `avg >= 0.40 or (citations and len(citations) > 0)`
        — the second disjunct fires whenever the model cites *anything at
        all*, with no floor on how much of the offered context that
        citation actually covers. Since ANSWER_PROMPT instructs the model
        to "cite every factual claim" and it reliably emits at least one
        `[SECTION_ID]`, `len(citations) > 0` was true for nearly every
        query regardless of whether retrieval was any good — the result
        (confirmed against results_full.json's Table 5) was every one of
        17 eval queries landing in "medium" and zero in "high" or "low",
        i.e. a confidence signal that doesn't discriminate between a
        strong answer and a weak one. Recall is now computed unconditionally
        (0.0 when citations weren't supplied) and used as a real, graded
        threshold in the medium band instead of a boolean "cited something"
        check.

        BUGFIX 2: `avg` used to be computed over ALL of `top` — every one
        of the up to FINAL_TOP_K candidates OFFERED to the LLM, not just
        the ones it actually cited. For a narrow query this corpus doesn't
        have 10 genuinely relevant sections for, the remaining slots get
        filled with much weaker filler just to pad out the context window
        — averaging over that filler could drag a genuinely solid 3-
        citation answer down to "low" even though every citation it made
        was good (observed directly: a domestic-violence/dowry query cited
        3 correct BNS sections at final_score ~0.7-0.8 each, but 7 other
        offered-and-correctly-ignored candidates at ~0.1 pulled the
        average to 0.26 — "low", despite the answer being right). `avg` is
        now computed over the CITED subset when citations exist (what the
        answer actually relied on), falling back to the full offered set
        only when there's nothing else to measure from (no citations at
        all — e.g. the "no relevant sections" case).
        """
        if not ranked:
            return "low"
        top = self._select_top(ranked, top_k)

        cited_ids = {c.section_id for c in citations} if citations else set()
        scored    = [r for r in top if r.chunk.section_id in cited_ids] or top
        avg       = sum(r.final_score for r in scored) / len(scored)
        retrieval_ok = avg >= 0.65

        # Citation recall gate (Gap 18) — always computed as a graded value,
        # 0.0 when citations weren't supplied, rather than short-circuiting
        # to a boolean that always passes. Still measured against the FULL
        # offered set (not just cited_ids) — recall is specifically "how
        # much of what was offered got used", so it needs the full
        # denominator; only `avg` above changes to the cited-only numerator.
        if citations is not None:
            offered_ids  = {r.chunk.section_id for r in top}
            recall       = len(cited_ids & offered_ids) / max(len(offered_ids), 1)
        else:
            recall       = 0.0
        citation_ok = recall >= 0.5

        if retrieval_ok and citation_ok:
            return "high"
        elif avg >= LOW_RELEVANCE_THRESHOLD or recall >= 0.25:
            return "medium"
        return "low"

    def _build_irac_summary(
        self, ranked: list[RankedChunk], top_k: int,
        citations: list | None = None,
    ) -> dict:
        # Same fix as _assess_confidence's BUGFIX 2: average over what was
        # actually cited, not every candidate offered — otherwise these
        # "coverage" bars read as uniformly bad for any answer that (quite
        # correctly) ignored most of a padded-out top_k, even when the
        # sections it did cite were a strong match.
        top = self._select_top(ranked, top_k)
        if not top:
            return {}
        cited_ids = {c.section_id for c in citations} if citations else set()
        top = [r for r in top if r.chunk.section_id in cited_ids] or top
        return {
            "issue_coverage":   round(sum(r.issue_score       for r in top) / len(top), 2),
            "rule_coverage":    round(sum(r.rule_score        for r in top) / len(top), 2),
            "application_fit":  round(sum(r.application_score for r in top) / len(top), 2),
            "conclusion_match": round(sum(r.conclusion_score  for r in top) / len(top), 2),
        }

    def generate(
        self,
        query:      str,
        intent:     QueryIntent,
        ranked:     list[RankedChunk],
        top_k:      int = FINAL_TOP_K,
        # Acts main.py's domain_router/universal_translator stages already
        # determined are relevant to this query but absent from the
        # database (main.py's `all_gaps`) — not re-detected here, just
        # threaded through so the answer itself can be honest about them
        # instead of the LLM writing as if the retrieved sections are a
        # direct answer to a law it was never given.
        known_gaps: list[str] | None = None,
    ) -> LegalAnswer:

        if not ranked:
            return LegalAnswer(
                query      = query,
                answer     = "No relevant legal sections found for this query.",
                intent     = intent.label,
                confidence = "low",
            )

        top             = self._select_top(ranked, top_k)
        # BUGFIX: this used to average final_score across ALL of `top` —
        # every one of the up to FINAL_TOP_K=10 candidates about to be
        # shown to the LLM, most of which are padding when (as is typical
        # for a narrow query in this corpus) only 2-4 sections are
        # genuinely relevant. That diluted average routinely fell under
        # LOW_RELEVANCE_THRESHOLD even when the best few candidates were
        # strong, which is exactly backwards for what this flag is meant
        # to decide: "should we hedge, or is what we found good enough to
        # answer confidently" is a question about the BEST matches, not
        # the average of everything offered as padding. This is the
        # pre-generation twin of the same fix already applied to
        # _assess_confidence/_build_irac_summary (which average over what
        # was actually CITED) — this one can't do that, since nothing has
        # been cited yet at this point, so it uses the best few candidates
        # by score instead as the next-best proxy for "what we actually
        # found", rather than diluting across the whole padded context.
        best            = sorted(top, key=lambda r: r.final_score, reverse=True)[:3]
        avg_score       = sum(r.final_score for r in best) / len(best) if best else 0.0
        weak_retrieval  = avg_score < LOW_RELEVANCE_THRESHOLD
        use_cautious    = bool(known_gaps) or weak_retrieval

        context  = self._build_context(ranked, top_k)
        if use_cautious:
            prompt = ANSWER_PROMPT_CAUTIOUS.format(
                query      = query,
                intent     = intent.label,
                gap_notice = "; ".join(known_gaps) if known_gaps
                             else "(none specifically flagged — retrieval confidence was simply low)",
                context    = context,
            )
        else:
            prompt = ANSWER_PROMPT.format(query=query, intent=intent.label, context=context)

        response = ollama.chat(
            model    = OLLAMA_ANSWER_MODEL,
            messages = [{"role": "user", "content": prompt}],
        )
        answer_text = response["message"]["content"].strip()

        # BUGFIX: pass `top` (the exact top_k subset _build_context put in
        # the prompt above), not the full `ranked` — see _extract_citations'
        # docstring note. `top` was already computed above for the
        # weak_retrieval check, so this reuses it rather than re-deriving.
        citations  = self._extract_citations(answer_text, top)
        warnings   = [c.warning for c in citations if c.warning]
        # Gap 18: pass citations to confidence assessor so recall gate fires
        confidence = self._assess_confidence(ranked, top_k, citations=citations)
        # A query answered from weakly-related sections shouldn't read as
        # "medium" just because the model dutifully cited every section it
        # was handed — those citations being present doesn't mean they were
        # the right ones.
        #
        # BUGFIX: this used to only fire when `known_gaps` was also set —
        # i.e. only when the pipeline had already identified a NAMED act
        # missing from the dataset entirely (e.g. "Negotiable Instruments
        # Act is not indexed"). That left exactly the more dangerous case
        # uncovered: the law IS in the dataset but retrieval simply failed
        # to surface it (wrong query vocabulary, a missing QUICK_SYNONYMS
        # rule, embedding drift, etc.) — no known_gaps gets set for that,
        # because as far as the pipeline can tell nothing is "missing".
        # weak_retrieval alone (best-3 avg final_score < LOW_RELEVANCE_
        # THRESHOLD) is already what triggers the cautious/hedging prompt
        # above — a query that scores too low to answer confidently should
        # never come back labeled "medium" regardless of whether a gap was
        # named, so drop the known_gaps requirement and gate on
        # weak_retrieval by itself. Confirmed against a real query
        # ("punishments for having two wives at same time" — IPC_494/495,
        # BNS_082 are in the dataset but retrieval missed them): answer text
        # correctly hedged ("does not include any specific provisions...")
        # while confidence still reported "medium" next to ~0.08 IRAC
        # coverage scores — the exact contradiction this closes.
        if weak_retrieval:
            confidence = "low"
        irac_sum   = self._build_irac_summary(ranked, top_k, citations=citations)
        # Gap 23: capture the actual reranker top-K section IDs (not just
        # what the LLM cited) so the evaluator can measure true retrieval recall.
        #
        # BUGFIX: this used to slice with `top_k` (== FINAL_TOP_K == 8), the
        # same cutoff used to build the LLM's answer context. That meant
        # retrieved_section_ids never had more than 8 entries, so every
        # "@10" metric downstream (recall_at_k/precision_at_k/ndcg_at_k with
        # k=10 in evaluation/evaluate.py) was silently evaluated against an
        # 8-item list — recall@10 could never differ from recall@8, and
        # precision@10 was deflated by dividing by a k the list could never
        # reach. Report against the fuller RERANK_TOP_K pool (the output of
        # the IRAC reranker, before the answer-generation cutoff) instead,
        # so retrieval metrics reflect what the reranker actually surfaced.
        # This does NOT change what the LLM sees or cites — only what's
        # exposed for evaluation — so answer quality/latency are unaffected.
        report_k      = max(top_k, RERANK_TOP_K)
        retrieved_ids = [r.chunk.section_id for r in self._select_top(ranked, report_k)]

        return LegalAnswer(
            query        = query,
            answer       = answer_text,
            citations    = citations,
            warnings     = warnings,
            intent       = intent.label,
            confidence   = confidence,
            irac_summary = irac_sum,
            retrieved_section_ids = retrieved_ids,
        )

    def _build_case_context(self, case_chunks: list[dict]) -> str:
        parts = []
        for chunk in case_chunks:
            parts.append(
                f"[{chunk.get('document_id', 'doc')} / {chunk.get('chunk_role', 'text')}]\n"
                f"{chunk.get('text', '')[:1000]}"
            )
        return "\n---\n".join(parts) if parts else "(no matching case document excerpts found)"

    def generate_fused(
        self,
        query:          str,
        case_chunks:    list[dict],
        statute_answer: "LegalAnswer",
        top_k:          int = FINAL_TOP_K,
        alea_scores:    list | None = None,   # Gap 20: list[SectionScore] from ALEA
    ) -> LegalAnswer:
        """Combines case document excerpts with the statute pipeline's own
        answer/citations into one dual-source, citation-verified response.
        Reuses statute_answer's already-generated citations rather than
        re-deriving them, so this stays consistent with what Track A
        itself considered relevant.

        Gap 20 fix: when alea_scores are provided (a list of ALEA SectionScore
        objects from the Track B ALEA scorer), each section's coverage band
        (Strong/Partial/Weak/Missing) and applicability score are blended into
        the irac_summary, replacing the statute-only IRAC bars which mislead
        by showing statute alignment rather than case-evidence coverage.
        """
        if not case_chunks and not statute_answer.citations:
            return LegalAnswer(
                query      = query,
                answer     = "No relevant case documents or legal sections found for this query.",
                intent     = statute_answer.intent,
                confidence = "low",
            )

        case_context    = self._build_case_context(case_chunks)
        statute_context = "\n---\n".join(
            f"[{c.section_id}]\nAct: {c.act_name}\nContent: {c.content}"
            for c in statute_answer.citations
        ) or "(no applicable statute sections retrieved)"

        response = ollama.chat(
            model    = OLLAMA_ANSWER_MODEL,
            messages = [{"role": "user", "content": FUSED_ANSWER_PROMPT.format(
                query           = query,
                case_context    = case_context,
                statute_context = statute_context,
            )}],
        )
        answer_text = response["message"]["content"].strip()

        # Citation verification gate — only keep citations for sections that
        # were actually offered to the model, dropping anything hallucinated.
        offered_ids = {c.section_id for c in statute_answer.citations}
        cited_ids   = set(re.findall(r"\[([A-Z]+(?:_[A-Z0-9]+)+)\]", answer_text))
        verified    = [c for c in statute_answer.citations if c.section_id in cited_ids & offered_ids]

        confidence = "medium" if case_chunks and verified else \
                     statute_answer.confidence if verified else "low"

        # Gap 20: build an ALEA-enhanced irac_summary when evidence coverage
        # scores are available. The fused answer represents a case + statute
        # synthesis, so its IRAC bars should reflect EVIDENCE coverage (from
        # ALEA) rather than statute proximity (from the IRAC reranker).
        if alea_scores:
            alea_by_section = {s.section_id: s for s in alea_scores}
            enhanced_irac   = dict(statute_answer.irac_summary or {})
            coverage_list   = []
            for c in verified:
                alea_sec = alea_by_section.get(c.section_id)
                if alea_sec:
                    coverage_list.append({
                        "section_id":    c.section_id,
                        "band":          alea_sec.band,
                        "coverage":      round(alea_sec.coverage, 3),
                        "applicability": round(alea_sec.applicability, 3),
                    })
            if coverage_list:
                enhanced_irac["alea_coverage"] = coverage_list
                # Upgrade confidence if ALEA says Strong evidence for top section
                top_alea = alea_by_section.get(verified[0].section_id) if verified else None
                if top_alea and top_alea.band == "Strong" and confidence != "high":
                    confidence = "high"
            irac_for_response = enhanced_irac
        else:
            irac_for_response = statute_answer.irac_summary

        return LegalAnswer(
            query                 = query,
            answer                = answer_text,
            citations             = verified,
            warnings              = statute_answer.warnings,
            intent                = statute_answer.intent,
            confidence            = confidence,
            irac_summary          = irac_for_response,
            retrieved_section_ids = statute_answer.retrieved_section_ids,
        )

    def generate_document_only(self, query: str, case_chunks: list[dict]) -> LegalAnswer:
        """Actually answers the question from case chunks via the LLM,
        instead of returning the raw retrieved text — 'who is the victim'
        should come back as a direct answer ('Priya Sharma, per the FIR's
        complainant section'), not a wall of unrelated chunk dumps."""
        if not case_chunks:
            return LegalAnswer(
                query=query, answer="No matching case document content found for this query.",
                intent="document", confidence="low",
            )

        case_context = self._build_case_context(case_chunks)

        response = ollama.chat(
            model    = OLLAMA_ANSWER_MODEL,
            messages = [{"role": "user", "content": DOCUMENT_ONLY_PROMPT.format(
                query=query, case_context=case_context,
            )}],
        )
        answer_text = response["message"]["content"].strip()

        return LegalAnswer(
            query=query, answer=answer_text, intent="document", confidence="medium",
        )