"""
pipeline/hybrid_retriever.py
Hybrid retrieval: BM25 sparse + BGE dense via Qdrant.
Uses Reciprocal Rank Fusion (RRF) to merge ranked lists.
Compatible with qdrant-client >= 1.9
"""
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import sys
import re

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, SparseVector,
)
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    QDRANT_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K,
)
from data.bm25_tokenizer import tokenize as bm25_tokenize

# bge-large-en-v1.5 is trained asymmetrically: the query side needs this
# instruction prefix, the passage side doesn't. data/indexer.py encodes
# embedding_text with no prefix (it's the passage side of the index).
# This is the query-side half of that pair. Shared as a module constant
# (not just a local string in _cached_encode) because SectionPinner also
# does dense search against this SAME passage-embedded "legal_sections"
# collection and needs the identical prefix — see the wiring note in
# main.py's SectionPinner construction and pipeline/section_pinner.py.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    section_id:       str
    content:          str
    score:            float
    rrf_score:        float        = 0.0
    act_code:         str          = ""
    chapter:          str          = ""
    category:         str          = ""
    status:           str          = "active"
    intent:           str          = ""
    rule_summary:     str          = ""
    issue_tags:       list         = field(default_factory=list)
    conclusion_type:  str          = ""
    parent_section:   str          = ""
    child_sections:   list         = field(default_factory=list)
    related_sections: list         = field(default_factory=list)
    enacted_year:     int | None   = None
    last_amended:     str          = ""
    amended_by:       list         = field(default_factory=list)
    payload:          dict         = field(default_factory=dict)


# ── RRF merger ────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 20,   # Gap 8: tuned from paper default 60 → 20 for small-corpus lists (25-35 items).
                   # At k=60, score diff between rank-1 and rank-10 is only ~0.002.
                   # At k=20, it's ~0.014 — 8× more signal for the merger.
                   # Grid-search over {10,20,40,60} on NDCG@10 to validate.
) -> list[RetrievedChunk]:
    scores: dict[str, float]          = {}
    chunks: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            sid = chunk.section_id
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)
            if sid not in chunks:
                chunks[sid] = chunk

    merged = []
    for sid, rrf in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        c = chunks[sid]
        c.rrf_score = rrf
        merged.append(c)

    return merged


# ── Retriever ─────────────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(self, vocab_path: str = "./data/bm25_vocab.json", client=None,
                 embed_model=None):
        # Accept an already-open client (e.g. shared with CaseIndexer, which
        # points at the same local ./qdrant_db folder) rather than always
        # opening a new one — Qdrant's embedded/local mode file-locks the
        # storage folder to a single client, so two independent clients on
        # the same path raises RuntimeError: "already accessed by another
        # instance". Standalone use (just running main.py alone) still
        # works unchanged, since client defaults to None here.
        self.client      = client or QdrantClient(path=QDRANT_PATH)
        # BUGFIX: this used to unconditionally load a fresh SentenceTransformer
        # here, even when a caller (e.g. evaluate.py's ablation_study, which
        # builds 7 LegalRAGPipeline instances in a loop) already has one
        # loaded. On an 8GB GPU, loading bge-large a second time on top of
        # the first is what caused every ablation variant to CUDA OOM at a
        # trivial 20MiB allocation — there was simply no VRAM left after two
        # full copies of the model. Accept an already-loaded model the same
        # way `client` is already shared, and only load a fresh one as a
        # fallback for callers that don't have one yet (standalone runs).
        self.embed_model = embed_model or SentenceTransformer(EMBEDDING_MODEL)

        with open(vocab_path, "r") as f:
            self.vocab: dict[str, int] = json.load(f)

        # Gap 5 (revised): IDF weights for every token index in the vocab.
        #
        # Real IDF from actual document frequency, computed once by
        # data/build_bm25_idf.py and cached to data/bm25_idf.json. This
        # replaces the old encounter-order approximation, which assumed
        # "built into the vocab earlier == more common" — a coincidence of
        # indexer iteration order, not a measurement of how many sections
        # actually contain each token. Real df gives BM25 its intended
        # signal: e.g. "section"/"act" (appear in nearly every record) get
        # pushed toward ~0, while genuinely rare terms ("dacoity", "pocso")
        # get correctly boosted — the old approximation could get this
        # backwards for any token whose vocab-insertion position didn't
        # match its true rarity.
        #
        # Falls back to the old approximation if bm25_idf.json hasn't been
        # generated yet, so this is non-breaking for anyone who hasn't run
        # `python3 data/build_bm25_idf.py` — but you should run it once and
        # keep the real weights; the fallback is strictly worse.
        vocab_size = len(self.vocab)
        idf_path = Path(vocab_path).parent / "bm25_idf.json"
        if idf_path.exists():
            with open(idf_path, "r") as f:
                raw_idf: dict[str, float] = json.load(f)
            self.idf: dict[int, float] = {int(k): v for k, v in raw_idf.items()}
        else:
            print(
                f"[HybridRetriever] WARNING: {idf_path} not found — falling back to "
                "the encounter-order IDF approximation, which does not reflect true "
                "document frequency. Run `python3 data/build_bm25_idf.py` once to "
                "generate real weights."
            )
            self.idf = {
                idx: math.log((vocab_size + 1.0) / (idx + 2.0))
                for idx in range(vocab_size)
            }

        # Gap 7: Cache dense embeddings so repeated queries (multi-query
        # expansion can send up to 10 variants, some semantically redundant)
        # don't pay the bge-large inference cost twice.
        embed_model_ref = self.embed_model

        @lru_cache(maxsize=256)
        def _cached_encode(text: str) -> tuple:
            # bge-large-en-v1.5 is trained asymmetrically: queries need this
            # instruction prefix prepended before encoding, passages don't.
            # data/indexer.py correctly encodes embedding_text with no
            # prefix (it's building the passage side of the index) — this
            # was the missing query-side half of that pair. Without it,
            # dense retrieval quality is measurably worse across the board
            # (see BAAI/bge-large-en-v1.5 model card). Only applied here,
            # at the single query-encode call site used for corpus search —
            # not in section_pinner.py / query_router.py, whose shared
            # embed_fn does symmetric query-vs-example-phrase matching for
            # classification, not asymmetric query-vs-passage retrieval.
            prefixed = f"{BGE_QUERY_INSTRUCTION}{text}"
            vec = embed_model_ref.encode(prefixed, normalize_embeddings=True)
            return tuple(vec.tolist())

        self._cached_encode = _cached_encode

    # ── Public: properly-prefixed query embedding ────────────────────────────
    # Exposed so other components that dense-search this same passage
    # collection (currently: SectionPinner) use the identical asymmetric
    # query encoding this class uses internally, instead of each call site
    # growing its own copy of the prefix logic (or, worse, silently doing
    # symmetric encoding against an asymmetrically-trained passage index).
    def embed_query_vector(self, text: str) -> list[float]:
        return list(self._cached_encode(text))

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def _dense_retrieve(
        self,
        query:   str,
        top_k:   int           = DENSE_TOP_K,
        filters: Filter | None = None,
    ) -> list[RetrievedChunk]:

        # Gap 7: use cached encode to avoid redundant bge-large inference
        vec = list(self._cached_encode(query))

        results = self.client.query_points(
            collection_name = COLLECTION_NAME,
            query           = vec,
            using           = "dense",
            query_filter    = filters,
            limit           = top_k,
            with_payload    = True,
        )
        return [self._hit_to_chunk(h) for h in results.points]

    # ── Sparse (BM25) retrieval ───────────────────────────────────────────────

    def _sparse_retrieve(
        self,
        query:   str,
        top_k:   int           = BM25_TOP_K,
        filters: Filter | None = None,
    ) -> list[RetrievedChunk]:

        tokens  = bm25_tokenize(query)
        # Gap 5 fix: apply IDF weighting to term frequencies so common terms
        # like "act", "section", "the" don't dominate rare legal terms.
        # Raw TF gives "act" appearing 3× the same weight as "dacoity" 1×;
        # TF×IDF makes "dacoity" correctly outweigh the noise.
        freq: dict[int, float] = {}
        for t in tokens:
            if t in self.vocab:
                idx = self.vocab[t]
                idf = self.idf.get(idx, 1.0)
                freq[idx] = freq.get(idx, 0.0) + idf   # TF*IDF accumulation
        indices = list(freq.keys())
        values  = list(freq.values())

        if not indices:
            return []

        results = self.client.query_points(
            collection_name = COLLECTION_NAME,
            query           = SparseVector(indices=indices, values=values),
            using           = "sparse",
            query_filter    = filters,
            limit           = top_k,
            with_payload    = True,
        )
        return [self._hit_to_chunk(h) for h in results.points]

    # ── Direct section lookup ─────────────────────────────────────────────────

    def _direct_section_lookup(self, query: str) -> list[RetrievedChunk]:
        # Gap 10 fix: expanded from 6 acts to all 18 acts in the dataset.
        # Queries like "BNS 103" or "POCSO Section 4" now trigger direct lookup.
        ACT_CODE_MAP = {
            "ipc":   "IPC",   "cpc":   "CPC",   "crpc":  "CRPC",
            "ita":   "ITA",   "iea":   "IEA",   "coi":   "COI",
            "bns":   "BNS",   "bnss":  "BNSS",  "bsa":   "BSA",
            "sra":   "SRA",   "tpa":   "TPA",   "ica":   "ICA",
            "ndps":  "NDPS",  "pca":   "PCA",   "pocso": "POCSO",
            "scst":  "SCST",  "uapa":  "UAPA",  "la":    "LA",
        }
        # Extended pattern: captures all 18 act codes
        pattern = re.findall(
            r'\b(ipc|cpc|crpc|ita|iea|coi|bns|bnss|bsa|sra|tpa|ica|ndps|pca|pocso|scst|uapa|la)'
            r'\s+(?:section\s+)?(\d+[a-zA-Z]*)\b',
            query.lower()
        )
        chunks = []
        seen: set[str] = set()
        for act_abbr, sec_num in pattern:
            act_code = ACT_CODE_MAP.get(act_abbr)
            if not act_code:
                continue

            # Strategy 1: look up by section_number field (stored as string, e.g. "1")
            results1, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="act_code",       match=MatchValue(value=act_code)),
                    FieldCondition(key="section_number", match=MatchValue(value=sec_num)),
                ]),
                limit=3, with_payload=True,
            )

            # Strategy 2: look up by section_id field (e.g. "IPC_001" for sec_num="1")
            # Pad to 3 digits to match the stored ID format
            padded    = sec_num.zfill(3) if sec_num.isdigit() else sec_num
            target_id = f"{act_code}_{padded}"
            results2, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="section_id", match=MatchValue(value=target_id)),
                ]),
                limit=3, with_payload=True,
            )

            for r in list(results1) + list(results2):
                sid = (r.payload or {}).get("section_id", str(r.id))
                if sid not in seen:
                    seen.add(sid)
                    chunks.append(self._hit_to_chunk(r, default_score=1.0))
        return chunks

    # ── Fetch by explicit section IDs (for section pinner) ───────────────────

    def fetch_by_ids(self, section_ids: list[str]) -> list[RetrievedChunk]:
        """
        Fetches sections directly by their section_id field in Qdrant.
        Used by the section pinner to inject guaranteed sections into the pool.
        Returns chunks in the same order as section_ids (skips any not found).
        """
        from qdrant_client.models import FieldCondition, MatchValue, Filter
        chunks = []
        seen:  set[str] = set()
        for sid in section_ids:
            if sid in seen:
                continue
            try:
                results, _ = self.client.scroll(
                    collection_name = COLLECTION_NAME,
                    scroll_filter   = Filter(must=[
                        FieldCondition(key="section_id", match=MatchValue(value=sid))
                    ]),
                    limit        = 1,
                    with_payload = True,
                )
                if results:
                    chunk = self._hit_to_chunk(results[0], default_score=1.0)
                    chunk.rrf_score = 1.0   # treat as top-priority
                    chunks.append(chunk)
                    seen.add(sid)
            except Exception:
                pass   # section not found — skip silently
        return chunks

    # ── Hybrid retrieval ──────────────────────────────────────────────────────

    def retrieve(
        self,
        queries:            list[str],
        top_k:              int           = HYBRID_TOP_K,
        act_filter:         str | None    = None,
        status_filter:      str | None    = None,
        dense_act_filter:   str | None    = None,   # Gap 9: separate filter for dense
    ) -> list[RetrievedChunk]:

        # ── Build sparse (BM25) filter — act filter applies here ──────────────
        sparse_conditions = []
        if status_filter:
            sparse_conditions.append(
                FieldCondition(key="status", match=MatchValue(value=status_filter.strip().title()))
            )
        if act_filter:
            sparse_conditions.append(
                FieldCondition(key="act_code", match=MatchValue(value=act_filter))
            )
        sparse_filter = Filter(must=sparse_conditions) if sparse_conditions else None

        # ── Build dense filter — Gap 9: no act filter on dense retrieval ──────
        dense_conditions = []
        if status_filter:
            dense_conditions.append(
                FieldCondition(key="status", match=MatchValue(value=status_filter.strip().title()))
            )
        if dense_act_filter:   # only applied if caller explicitly passes it
            dense_conditions.append(
                FieldCondition(key="act_code", match=MatchValue(value=dense_act_filter))
            )
        dense_filter = Filter(must=dense_conditions) if dense_conditions else None

        # ── Step 1: direct section hits (always included) ─────────────────────
        direct_chunks = []
        seen_ids: set[str] = set()
        for query in queries:
            for chunk in self._direct_section_lookup(query):
                if chunk.section_id not in seen_ids:
                    direct_chunks.append(chunk)
                    seen_ids.add(chunk.section_id)

        # ── Step 2: BM25 + dense per query, merged via RRF ────────────────────
        all_ranked_lists = []

        for query in queries:
            # Gap 9: dense uses dense_filter (no act restriction); sparse uses sparse_filter
            dense_results  = self._dense_retrieve(query, top_k=DENSE_TOP_K,  filters=dense_filter)
            sparse_results = self._sparse_retrieve(query, top_k=BM25_TOP_K, filters=sparse_filter)
            per_query      = reciprocal_rank_fusion([dense_results, sparse_results])
            all_ranked_lists.append(per_query)

        # ── Step 3: merge semantic results via RRF, then pin direct hits on top ──
        # Direct section hits (e.g. "ipc 1" → IPC_001) are exact matches by
        # section number — they must appear in the final list regardless of how
        # the IRAC reranker scores their content relevance.  Pinning them to the
        # front (deduplicating against RRF results) guarantees they reach Stage 7.
        if direct_chunks:
            all_ranked_lists.insert(0, direct_chunks)

        rrf_results = reciprocal_rank_fusion(all_ranked_lists)

        if direct_chunks:
            direct_ids = {c.section_id for c in direct_chunks}
            # Put direct hits first (in discovery order), then the rest of RRF
            pinned = list(direct_chunks)
            rest   = [c for c in rrf_results if c.section_id not in direct_ids]
            final  = pinned + rest
        else:
            final = rrf_results

        return final[:top_k]

    # ── Helper ────────────────────────────────────────────────────────────────

    def _hit_to_chunk(self, hit, default_score: float = 0.0) -> RetrievedChunk:
        p = hit.payload or {}
        return RetrievedChunk(
            section_id       = p.get("section_id", str(hit.id)),
            content          = p.get("content", ""),
            score            = getattr(hit, "score", default_score),  # Record has no .score
            act_code         = p.get("act_code", ""),
            chapter          = p.get("chapter", ""),
            category         = p.get("category", ""),
            status           = p.get("status", "active"),
            intent           = p.get("intent", ""),
            rule_summary     = p.get("rule_summary", ""),
            issue_tags       = p.get("issue_tags", []),
            conclusion_type  = p.get("conclusion_type", ""),
            parent_section   = p.get("parent_section", ""),
            child_sections   = p.get("child_sections", []),
            related_sections = p.get("related_sections", []),
            enacted_year     = p.get("enacted_year"),
            last_amended     = p.get("last_amended", ""),
            amended_by       = p.get("amended_by", []),
            payload          = p,
        )


if __name__ == "__main__":
    retriever = HybridRetriever()
    results   = retriever.retrieve(
        queries=["punishment for cybercrime under IT Act"],
        top_k=5,
    )
    for r in results:
        print(f"{r.section_id} | rrf={r.rrf_score:.4f} | {r.category}")