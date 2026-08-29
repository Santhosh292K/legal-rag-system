"""
main.py
Legal RAG Pipeline — Universal orchestrator

Pipeline:
  User Query
    → Stage 0a  Domain Router           — maps query to act cluster, detects type
    → Stage 0b  Universal Translator    — rewrites query in legal terminology
    → Stage 0c  Section Pinner          — deterministically injects known sections
    → Stage 1   Intent Classifier       — labels intent for IRAC reranker
    → Stage 2   Multi-Query Expander    — generates search variants
    → Stage 3   Hybrid Retrieval        — BM25 + BGE + RRF (NO single-act lock)
    → Stage 4   Temporal Filter         — removes repealed/amended sections
    → Stage 5   Chunk Structurer        — enriches with parent/child context
    → Stage 6   IRAC Reranker           — scores for issue/rule/application/conclusion
    → Stage 6.5 Rocchio Feedback        — issue_tags from top-2 re-query BM25
    → Stage 6.75 KG Augmentation        — expand via READ_WITH / SUPERSEDES edges
    → Stage 7   Answer Generator        — citation-grounded answer
"""
import time
from dotenv import load_dotenv

load_dotenv()

from pipeline.intent_classifier    import IntentClassifier
from pipeline.query_expander       import QueryExpander
from pipeline.hybrid_retriever     import HybridRetriever
from pipeline.temporal_filter      import TemporalFilter
from pipeline.chunk_structurer     import ChunkStructurer
from pipeline.irac_reranker        import IRACReranker
from pipeline.answer_generator     import AnswerGenerator, LegalAnswer
from pipeline.domain_router        import DomainRouter
from pipeline.universal_translator import UniversalTranslator
from pipeline.section_pinner       import SectionPinner, PIN_EXPLANATION
from pipeline.legal_kg             import LegalKnowledgeGraph, kg_augment_ranked
from config import HYBRID_TOP_K, RERANK_TOP_K, FINAL_TOP_K

DOMAIN_TO_INTENT = {
    "civil":          "statute",
    "constitutional": "statute",
    "procedural":     "procedural",
    "comparative":    "definition",
}


class LegalRAGPipeline:

    def __init__(self, json_path="./data/final_dataset.json",
                 vocab_path="./data/bm25_vocab.json", verbose=True, qdrant_client=None,
                 # BUGFIX: accept an already-loaded embedding model, same
                 # pattern as qdrant_client above. Without this, every extra
                 # LegalRAGPipeline built on top of one already in memory
                 # (e.g. evaluate.py's ablation_study, which builds 7 of
                 # these in a loop) loaded a second full copy of bge-large
                 # via HybridRetriever, which is what blew past the 8GB GPU
                 # and CUDA OOM'd on every single ablation variant.
                 embed_model=None,
                 # BUGFIX: same reasoning as embed_model above, for the
                 # cross-encoder reranker instead of the embedding model.
                 # 5 of the 7 ablation variants have use_irac=True, and each
                 # was loading its own full copy of bge-reranker-large —
                 # fixing the embed_model duplication alone wasn't enough
                 # for those variants to fit in 8GB.
                 cross_encoder=None,
                 # Gap 22: ablation flags — disable individual components to measure
                 # their contribution. Default True = full pipeline.
                 use_irac:      bool = True,
                 use_hierarchy: bool = True,
                 use_temporal:  bool = True,
                 use_pinner:    bool = True,
                 use_kg:        bool = True):
        self.verbose       = verbose
        self.use_irac      = use_irac
        self.use_hierarchy = use_hierarchy
        self.use_temporal  = use_temporal
        self.use_pinner    = use_pinner
        self.use_kg        = use_kg

        self._log("Loading pipeline components...")
        self.retriever  = HybridRetriever(vocab_path=vocab_path, client=qdrant_client,
                                           embed_model=embed_model)

        # Shared embed_fn: reuses HybridRetriever's already-loaded bge-large
        # model rather than every embedding-aware component loading its own
        # copy. Built once here and handed to every component whose
        # semantic tier depends on it (DomainRouter, IntentClassifier,
        # QueryExpander) — without this, those components silently fall
        # back to their regex-only / rule-then-LLM paths and never exercise
        # the embedding tier they were written for.
        #
        # This is a SYMMETRIC embed_fn (no bge-large query instruction
        # prefix) — correct for these three, because they all compare the
        # query against a small curated set of example phrases via
        # SemanticMatcher (query-vs-example, both sides encoded the same
        # way), not against the passage-embedded Qdrant corpus.
        embed_fn = lambda texts: self.retriever.embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False)

        self.router     = DomainRouter(embed_fn=embed_fn)
        self.translator = UniversalTranslator()
        self.classifier = IntentClassifier(embed_fn=embed_fn)
        self.expander   = QueryExpander(embed_fn=embed_fn)

        # BUGFIX: SectionPinner is NOT the same kind of task as the three
        # components above — it dense-searches the real "legal_sections"
        # Qdrant collection (self.retriever.client.query_points against the
        # same passage vectors HybridRetriever's own dense retrieval
        # searches), i.e. it's an asymmetric query-vs-passage task exactly
        # like HybridRetriever's. Wiring it up with the plain symmetric
        # embed_fn above silently skipped the bge-large query instruction
        # prefix, which systematically depresses cosine similarity against
        # passage vectors (which WERE encoded correctly, with no prefix, by
        # data/indexer.py). That pushes genuine matches below
        # MIN_PIN_SIMILARITY and defeats the one job this stage has: catch
        # lay-language queries ("a tribal farmer's land was forcibly
        # acquired...") that BM25's exact-vocabulary matching can't reach.
        # Use HybridRetriever's embed_query_vector (same prefix, same
        # cache) instead.
        pin_embed_fn = lambda texts: [self.retriever.embed_query_vector(t) for t in texts]

        # SectionPinner reuses HybridRetriever's already-open Qdrant client
        # and bge-large model rather than opening a second client against
        # the same file-locked ./qdrant_db folder or loading a second copy
        # of the embedding model.
        self.pinner = SectionPinner(
            client=self.retriever.client,
            embed_fn=pin_embed_fn,
        ) if use_pinner else None

        self.t_filter   = TemporalFilter()
        # Gap 15: pass the shared Qdrant client so ChunkStructurer never
        # loads final_dataset.json into memory (the data is already in Qdrant).
        self.structurer = ChunkStructurer(
            json_path  = json_path,   # fallback if client is None (tests / legacy)
            client     = self.retriever.client if use_hierarchy else None,
        )
        # llm_top_n lowered 15 -> 8 and calls run concurrently — see the
        # ROOT CAUSE note in IRACReranker.__init__ (pipeline/irac_reranker.py):
        # this stage's 15 sequential ollama.chat calls per query were the
        # single largest contributor to the 192s avg / 305s p95 latency in
        # Table 4. If you re-run evaluate.py and precision drops more than
        # you're willing to trade for the latency win, raise this back up
        # (RERANK_TOP_K=20 is the ceiling) before reverting the concurrency change.
        self.reranker   = IRACReranker(llm_top_n=8, max_workers=4, cross_encoder=cross_encoder) if use_irac else None
        self.generator  = AnswerGenerator()

        # KG: build from Qdrant at init time (reuses shared client).
        # Building is fast (scroll whole collection once, ~1-2s).
        self.kg: LegalKnowledgeGraph | None = None
        if use_kg:
            try:
                self.kg = LegalKnowledgeGraph()
                self.kg.build_from_qdrant(
                    client          = self.retriever.client,
                    collection_name = "legal_sections",
                )
                stats = self.kg.stats()
                self._log(f"KG ready: {stats['nodes']} nodes, {stats['edges']} edges")
            except Exception as e:
                self._log(f"KG build failed (non-fatal, KG disabled): {e}")
                self.kg = None

        self._log("Pipeline ready.")

    def _log(self, msg):
        if self.verbose:
            print(f"[LegalRAG] {msg}")

    def query(self, user_query, act_filter=None, strict_temporal=True, extra_sections=None,
              debug_trace: dict | None = None):
        """
        debug_trace: optional dict. If passed, this call fills it in-place with
        the section_id list present at each pipeline stage boundary:
          'raw_pool'     — after Stage 3 hybrid retrieval + pinned merge, before temporal filter
          'post_rerank'  — after Stage 6 IRAC reranking (or ablation fallback)
          'post_rocchio' — after Stage 6.5 Rocchio pseudo-relevance feedback
          'post_kg'      — after Stage 6.75 KG augmentation
          'final'        — the same list evaluate.py measures recall against
        Use this to see, per query, which stage first surfaces a gold section
        (or whether it never appears at all) — see evaluation/diagnose_recall.py.
        Zero cost when debug_trace is None (the default) — every write below is
        gated on it, so normal query() calls are unaffected.
        """
        t0 = time.time()

        # ── Stage 0a: Domain routing ──────────────────────────────────────────
        self._log("Stage 0a: Domain routing...")
        routing = self.router.route(user_query)
        self._log(f"  type={routing.query_type}  domains={routing.domains}")
        self._log(f"  primary_acts={routing.primary_acts}")
        for m in routing.missing_acts:
            self._log(f"  ⚠ NOT IN DATASET: {m}")

        # ── Stage 0b: Universal translation ──────────────────────────────────
        self._log("Stage 0b: Universal legal translation...")
        translation = self.translator.translate(user_query)
        self._log(f"  domain={translation.domain}")
        self._log(f"  primary={translation.primary_query[:75]}")
        self._log(f"  predicted={translation.predicted_sections[:6]}")
        for gap in translation.dataset_gaps:
            self._log(f"  ⚠ GAP: {gap}")

        all_gaps = list(set(routing.missing_acts + translation.dataset_gaps))

        # ── Stage 0c: Section pinning (deterministic, LLM-free) ──────────────
        self._log("Stage 0c: Section pinning...")
        from pipeline.section_pinner import PinResult
        if self.use_pinner and self.pinner:
            pin_result = self.pinner.pin(user_query)
        else:
            pin_result = PinResult()   # ablation: skip pinning

        # Fusion (Track B) may already know which sections apply — e.g. a
        # section explicitly cited in an uploaded FIR, or one implied by
        # extracted evidence. Merge those in here so they flow through the
        # exact same pinned-chunk fetch path as query-text pins, rather than
        # needing a separate code path downstream.
        if extra_sections:
            existing = set(pin_result.section_ids)
            for sid in extra_sections:
                if sid not in existing:
                    pin_result.section_ids.append(sid)
                    existing.add(sid)

        self._log(f"  Pinned {len(pin_result.section_ids)} sections: {pin_result.section_ids[:6]}")

        # Fetch pinned sections directly from Qdrant (bypasses BM25/dense)
        pinned_chunks = []
        if pin_result.section_ids:
            pinned_chunks = self.retriever.fetch_by_ids(pin_result.section_ids)
            self._log(f"  Fetched {len(pinned_chunks)} pinned chunks from Qdrant")

        # ── Stage 1: Intent classification ───────────────────────────────────
        self._log("Stage 1: Classifying intent...")
        intent = self.classifier.classify(translation.primary_query or user_query)
        self._log(f"  Intent={intent.label}  Conf={intent.confidence:.2f}  Act={intent.act_hint}")

        if routing.primary_acts and (not intent.act_hint or intent.confidence < 0.5):
            intent.act_hint = routing.primary_acts[0]
        if translation.domain in DOMAIN_TO_INTENT:
            intent.label = DOMAIN_TO_INTENT[translation.domain]

        # Override intent to 'punitive' when pinner identified death/accident/crime sections
        # This prevents "procedural" label from killing IRAC scores for punitive sections
        PUNITIVE_PIN_PREFIXES = {
            "BNS_106", "IPC_304A", "IPC_304", "IPC_300", "IPC_302",
            "BNS_261", "IPC_279", "IPC_392", "IPC_390", "BNS_199",
            "IPC_166A", "PCA_013", "PCA_014", "IPC_498A", "IPC_304B",
        }
        if pin_result.section_ids:
            has_punitive_pin = any(s in PUNITIVE_PIN_PREFIXES for s in pin_result.section_ids)
            if has_punitive_pin and intent.label not in ("punitive",):
                intent.label = "punitive"
                self._log(f"  Intent overridden → punitive (punitive pin detected)")

        # Act filter: only lock to one act for single-ACT high-confidence queries.
        # NEVER lock when pinned sections span multiple acts.
        #
        # Bug fix: this used to key off len(routing.domains) == 1, but a
        # single DOMAIN can still span many acts (e.g. "criminal" alone
        # covers IPC, BNS, NDPS, POCSO, UAPA, PCA, SCST). That let a
        # confident-but-generic intent (which resolves act_hint via Tier 2
        # domain-priority fallback, favouring common acts like IPC thanks
        # to GENERAL_CODE_BONUS) lock retrieval to IPC and silently exclude
        # the correct act — e.g. SCST, COI, TPA — even though it was
        # correctly identified as part of the activated domain. Locking is
        # now gated on the act cluster itself being singular, OR on the
        # chosen act having direct (non-fallback) evidence — i.e. the query
        # actually named/matched that specific act, not just its domain.
        effective_act = act_filter
        if effective_act is None:
            pinned_acts = set(c.act_code for c in pinned_chunks)
            if len(pinned_acts) > 1:
                # Pinned sections span multiple acts — no filter
                self._log(f"  No act filter (pinned spans {pinned_acts})")
            elif (
                intent.confidence >= 0.75 and intent.act_hint
                and (len(routing.acts) == 1 or routing.primary_act_has_direct_evidence)
            ):
                effective_act = intent.act_hint
                self._log(f"  Single-act filter → {effective_act} "
                          f"({'sole act in cluster' if len(routing.acts) == 1 else 'direct evidence'})")
            else:
                self._log("  No act filter (multi-act domain, no direct act evidence)")

        # ── Stage 2: Query expansion ──────────────────────────────────────────
        # Gap 1 fix: use the translated (legal) query as the primary base, not the
        # raw user query — paraphrases should stay in legal terminology.
        self._log("Stage 2: Expanding queries...")
        legal_base = translation.primary_query or user_query
        base = list(translation.search_queries or [legal_base])
        for sec in translation.predicted_sections[:4]:
            act, _, num = sec.partition("_")
            base.append(f"section {num} {act} punishment meaning")
        if len(routing.domains) > 1:
            for act in routing.primary_acts[:3]:
                base.append(f"{legal_base} {act}")

        # Gap 1: pass legal_base (translated query) as query so LLM paraphrases
        # are driven by legal terminology, not the original layperson sentence.
        expanded = self.expander.expand(
            query=legal_base, intent=intent,
            rewritten_query=user_query,    # original query kept as hint
            extra_charges=translation.predicted_sections or None,
        )
        all_queries = list(dict.fromkeys(base + expanded))[:10]
        self._log(f"  Total query variants: {len(all_queries)}")

        # ── Stage 3: Hybrid retrieval ─────────────────────────────────────────
        # Gap 9 fix: dense retrieval must NOT have an act filter — cross-act
        # candidates (e.g. IPC_420 for cybercrime) would be cut before RRF.
        # Only the sparse (BM25) path gets the filter, where keyword false
        # positives are more common and the filter reduces noise meaningfully.
        self._log("Stage 3: Hybrid retrieval (BM25 + BGE + RRF)...")
        bm25_chunks = self.retriever.retrieve(
            queries=all_queries, top_k=HYBRID_TOP_K,
            act_filter=effective_act, status_filter=None,
            dense_act_filter=None,   # Gap 9: dense never gets the act filter
        )
        acts_found = set(c.act_code for c in bm25_chunks)
        self._log(f"  BM25/dense: {len(bm25_chunks)} chunks from acts: {acts_found}")

        # Merge: pinned first (guaranteed), then BM25/dense (deduped)
        pinned_ids  = {c.section_id for c in pinned_chunks}
        extra_bm25  = [c for c in bm25_chunks if c.section_id not in pinned_ids]
        raw_chunks  = pinned_chunks + extra_bm25
        all_act_ids = set(c.act_code for c in raw_chunks)
        self._log(f"  Total pool: {len(raw_chunks)} chunks from {all_act_ids}")
        if debug_trace is not None:
            debug_trace["raw_pool"] = [c.section_id for c in raw_chunks]

        # ── Stage 4: Temporal filter ──────────────────────────────────────────
        self._log("Stage 4: Temporal filter...")
        if self.use_temporal:
            validated = (self.t_filter.filter_active_only(raw_chunks, intent)
                         if strict_temporal else self.t_filter.filter(raw_chunks, intent))
        else:
            # Ablation: skip temporal filter — wrap raw chunks in ValidatedChunk stubs
            from pipeline.temporal_filter import ValidatedChunk
            validated = [ValidatedChunk(chunk=c, is_valid=True, validity_label="active",
                                        warning="", penalized_score=c.score)
                         for c in raw_chunks]
        self._log(f"  Valid: {len(validated)}")

        # ── Stage 5: Chunk structuring (minimal — Gap 16) ────────────────────
        self._log("Stage 5: Structuring chunks (minimal, pre-rerank)...")
        if self.use_hierarchy:
            # Gap 16: structure_minimal gives us StructuredChunk without
            # parent/child context — cheap, so we run on all candidates.
            # Enrichment happens AFTER reranking on the final top-K only.
            structured = self.structurer.structure_minimal(validated)
        else:
            from pipeline.chunk_structurer import StructuredChunk
            structured = [
                StructuredChunk(
                    section_id=vc.chunk.section_id, content=vc.chunk.content,
                    act_name=vc.chunk.payload.get("act_name", ""),
                    chapter=vc.chunk.chapter, category=vc.chunk.category,
                    validity_label=vc.validity_label, warning=vc.warning,
                    penalized_score=vc.penalized_score,
                    rule_summary=vc.chunk.rule_summary,
                    issue_tags=vc.chunk.issue_tags,
                    conclusion_type=vc.chunk.conclusion_type,
                    enriched_context=vc.chunk.content,
                ) for vc in validated
            ]

        # ── Stage 6: IRAC reranking ───────────────────────────────────────────
        self._log("Stage 6: IRAC reranking...")
        # BUGFIX (rev 2): the previous fix here scored every candidate
        # against legal_base ALONE (translation.primary_query). That fixed
        # the original problem (scoring against raw user_query missed
        # legal vocabulary) but introduced a new one: legal_base is the
        # LLM's single "most important query" — a narrow, one-concept
        # phrase. For queries with 2-3 gold sections spanning DIFFERENT
        # legal concepts (e.g. false FIR → IPC_211 false charge AND
        # IPC_166 public servant misconduct; a will made by someone of
        # unsound mind → ICA_011 competency AND ICA_012 void agreements),
        # scoring against only the narrow legal_base rewarded whichever
        # one concept the LLM foregrounded and starved the other of
        # lexical overlap — diagnose_recall.py's truncation rate went UP
        # (10.5% → 18.4%) after that change, concentrated exactly on these
        # multi-concept queries. Combining user_query (full narrative,
        # everyday vocabulary — helps content/enriched_context overlap)
        # with legal_base (legal terminology — helps rule_summary/
        # issue_tags overlap) keeps both signal sources instead of
        # replacing one with the other.
        rerank_query = f"{legal_base} {user_query}" if legal_base != user_query else user_query
        if self.use_irac and self.reranker:
            ranked = self.reranker.rerank(
                query=rerank_query, intent=intent, chunks=structured, top_k=RERANK_TOP_K)
        else:
            # Ablation: skip IRAC reranker — use retrieval score as final score
            from pipeline.irac_reranker import RankedChunk
            ranked = [
                RankedChunk(chunk=c, final_score=c.penalized_score,
                            irac_score=c.penalized_score)
                for c in structured
            ]
            ranked.sort(key=lambda r: r.final_score, reverse=True)
            ranked = ranked[:RERANK_TOP_K]
        if ranked:
            self._log(f"  Top: {ranked[0].chunk.section_id} score={ranked[0].final_score:.3f}")
        if debug_trace is not None:
            debug_trace["post_rerank"] = [r.chunk.section_id for r in ranked]

        # ── Gap 16: Enrich top-K chunks AFTER reranking ────────────────────────
        # structure_minimal gave us raw content; now enrich ONLY the RERANK_TOP_K
        # that survived the reranker (not all 35+ candidates). Average saving:
        # ~20 Qdrant scroll() calls avoided per query (35 - 15 chunks).
        if self.use_hierarchy and ranked:
            top_chunks = [r.chunk for r in ranked[:RERANK_TOP_K]]
            self.structurer.enrich_chunks(top_chunks)
            # Sync enriched_context back into ranked items (in-place update above)
            self._log(f"  Enriched {len(top_chunks)} post-ranked chunks with parent/child context")

        # ── Gap 13: Rocchio pseudo-relevance feedback ─────────────────────────
        # After reranking, extract issue_tags from the top-2 sections and use
        # them as additional query terms to re-query BM25. This closes the
        # retrieval loop: good IRAC-scored sections feed back query signal so
        # related sections the original queries missed get a second chance.
        #
        # The newly retrieved chunks are scored at 0.35 (below a genuine IRAC
        # match, above zero) and only novel sections (not already in ranked) are
        # merged in, capped at 3 additional chunks to avoid noise inflation.
        if self.use_irac and ranked and len(ranked) >= 2:
            from pipeline.irac_reranker import RankedChunk as _RC
            feedback_tags: list[str] = []
            for r in ranked[:2]:
                feedback_tags.extend(r.chunk.issue_tags or [])
            feedback_tags = list(dict.fromkeys(feedback_tags))[:8]   # unique, top 8

            if feedback_tags:
                rocchio_query = " ".join(feedback_tags)
                self._log(f"  Gap 13 Rocchio: feedback query = {rocchio_query[:60]!r}")
                try:
                    fb_raw = self.retriever.retrieve(
                        queries=[rocchio_query], top_k=6,
                        act_filter=effective_act, status_filter=None,
                        dense_act_filter=None,
                    )
                    from pipeline.temporal_filter import ValidatedChunk as _VC
                    fb_valid = [
                        _VC(chunk=c, validity_label="active",
                            warning="", penalized_score=c.score)
                        for c in fb_raw
                    ]
                    fb_structured = self.structurer.structure_minimal(fb_valid)

                    ranked_ids_now = {r.chunk.section_id for r in ranked}
                    added = 0
                    for fb_c in fb_structured:
                        if added >= 3:
                            break
                        if fb_c.section_id in ranked_ids_now:
                            continue
                        ranked.append(_RC(
                            chunk=fb_c,
                            final_score=0.35, irac_score=0.35,
                            explanation="Rocchio pseudo-relevance feedback",
                        ))
                        ranked_ids_now.add(fb_c.section_id)
                        added += 1
                    if added:
                        self._log(f"  Rocchio: added {added} novel sections via feedback")
                except Exception as e:
                    self._log(f"  Rocchio feedback failed (non-fatal): {e}")
        if debug_trace is not None:
            debug_trace["post_rocchio"] = [r.chunk.section_id for r in ranked]

        # ── Stage 6.75: KG augmentation ───────────────────────────────────────
        # Expand the ranked list via typed KG edges (READ_WITH, SUPERSEDES,
        # RELATES_TO). KG surfaces sections that BM25+dense+Rocchio all missed
        # because they use different vocabulary but are legally connected:
        #   IPC_302 retrieval → KG adds IPC_34 (common intention) via READ_WITH
        #   IPC_302 retrieval → KG adds BNS_103 via SUPERSEDES (BNS transition)
        #   IPC_498A retrieval → KG adds IPC_304B (dowry death) via READ_WITH
        if self.use_kg and self.kg and ranked:
            try:
                n_before = len(ranked)
                ranked = kg_augment_ranked(
                    ranked_chunks    = ranked,
                    kg               = self.kg,
                    structurer       = self.structurer,
                    max_kg_additions = 4,
                    hops             = 1,
                )
                added_kg = len(ranked) - n_before
                if added_kg:
                    self._log(f"  KG: added {added_kg} sections via graph expansion")
            except Exception as e:
                self._log(f"  KG augmentation failed (non-fatal): {e}")
        if debug_trace is not None:
            debug_trace["post_kg"] = [r.chunk.section_id for r in ranked]

        # ── Pinned section rescue ─────────────────────────────────────────────
        # The IRAC reranker scores sections by token overlap with the query.
        # Pinned sections are guaranteed relevant by pattern match, but may score
        # low if the query uses everyday language (e.g. "car accident" vs "rash act").
        # Re-inject any pinned section that fell out of the top RERANK_TOP_K.
        if pin_result.section_ids and structured:
            ranked_ids   = {r.chunk.section_id for r in ranked}
            pinned_set   = set(pin_result.section_ids)
            missing_pins = pinned_set - ranked_ids

            if missing_pins:
                # Find the structured chunks for missing pins
                structured_map = {c.section_id: c for c in structured}
                from pipeline.irac_reranker import RankedChunk as RC

                # Gap 14 fix: build a sim-score lookup from pin_result so each
                # pinned section gets its actual dense cosine similarity as its
                # final_score rather than a flat 0.55 for every pin.
                pin_sim: dict[str, float] = {}
                for sid, rule in zip(pin_result.section_ids, pin_result.matched_rules):
                    try:
                        # matched_rules stores "dense sim=0.73" — extract the float
                        sim = float(rule.split("=")[-1])
                    except (ValueError, IndexError):
                        sim = 0.55
                    pin_sim[sid] = sim

                for sid in pin_result.section_ids:   # preserve priority order
                    if sid in missing_pins and sid in structured_map:
                        chunk  = structured_map[sid]
                        fscore = pin_sim.get(sid, 0.55)
                        ranked.append(RC(
                            chunk=chunk,
                            issue_score=0.5, rule_score=0.5,
                            application_score=0.5, conclusion_score=0.5,
                            irac_score=0.5, cross_enc_score=0.5,
                            final_score=fscore,
                            explanation=PIN_EXPLANATION,
                        ))
                self._log(f"  Re-injected {len(missing_pins & set(structured_map))} pinned sections")

        # ── Stage 7: Answer generation ────────────────────────────────────────
        self._log("Stage 7: Generating answer...")
        answer = self.generator.generate(
            query=user_query, intent=intent, ranked=ranked, top_k=FINAL_TOP_K)

        if debug_trace is not None:
            debug_trace["final"] = list(answer.retrieved_section_ids)

        if all_gaps:
            answer.warnings = list(set((answer.warnings or []) + all_gaps))

        self._log(f"Done in {time.time()-t0:.2f}s | conf={answer.confidence} | "
                  f"citations={len(answer.citations)}")
        return answer

    def format_output(self, answer):
        lines = [
            f"\n{'='*70}",
            f"QUERY      : {answer.query}",
            f"INTENT     : {answer.intent}",
            f"CONFIDENCE : {answer.confidence.upper()}",
            f"{'='*70}",
            f"\nANSWER:\n{answer.answer}",
        ]
        if answer.citations:
            lines += [f"\n{'─'*70}", "CITATIONS:"]
            for c in answer.citations:
                status = f" [{c.validity.upper()}]" if c.validity != "active" else ""
                lines.append(f"  [{c.section_id}]{status} — {c.act_name} | {c.category}")
                if c.warning:
                    lines.append(f"    ⚠ {c.warning}")
        if answer.warnings:
            lines += [f"\n{'─'*70}", "WARNINGS:"]
            for w in set(answer.warnings):
                lines.append(f"  ⚠ {w}")
        if answer.irac_summary:
            lines += [f"\n{'─'*70}", "IRAC COVERAGE:"]
            for k, v in answer.irac_summary.items():
                bar = "█" * int(v*10) + "░" * (10-int(v*10))
                lines.append(f"  {k:<20} {bar} {v:.2f}")
        lines.append("=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    pipeline = LegalRAGPipeline(verbose=True)

    # Backward-compatible one-shot mode: `python3 main.py "some question"`
    if len(sys.argv) > 1:
        q = sys.argv[1]
        print(pipeline.format_output(pipeline.query(q)))
        sys.exit(0)

    # Interactive mode: models/Qdrant client stay loaded across queries.
    print("\n[LegalRAG] Ready. Type a legal question, or 'quit' / 'exit' to stop.\n")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[LegalRAG] Exiting.")
            break

        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            print("[LegalRAG] Exiting.")
            break

        try:
            answer = pipeline.query(q)
            print(pipeline.format_output(answer))
        except Exception as e:
            # One bad query shouldn't kill the session — print the error
            # and drop back to the prompt instead of exiting.
            print(f"\n[LegalRAG] Error while answering that query: {e}\n")