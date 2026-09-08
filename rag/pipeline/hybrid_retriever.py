"""
pipeline/hybrid_retriever.py
Hybrid retrieval: Okapi BM25 (sparse) + BGE dense, fused with weighted RRF.
Compatible with qdrant-client >= 1.9
"""
import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, SparseVector,
)

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    QDRANT_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K, BM25_VOCAB_PATH,
)
from data.bm25_tokenizer import tokenize as bm25_tokenize, build_search_text, TOKENIZER_VERSION
from data.index_manifest import read_manifest, vocab_fingerprint, IndexMismatchError, INDEX_FORMAT_VERSION
from data.section_ref import extract_section_refs
from pipeline.section_store import SectionStore

# bge-large-en-v1.5 is trained asymmetrically: the query side needs this
# instruction prefix, the passage side doesn't. data/indexer.py encodes the
# passage side with no prefix. Shared as a module constant because
# SectionPinner and CaseIndexer dense-search the same passage vectors and
# need the identical prefix.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Okapi query-side term-frequency saturation. A query term repeated by
# several expansion variants shouldn't scale linearly; k3 caps its growth
# the same way k1 caps document-side term frequency.
BM25_K3 = 8.0


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
    k: int = 20,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Weighted Reciprocal Rank Fusion.

    k=20 rather than the paper's 60: these lists are 25-35 items over a
    3.4k-section corpus, and at k=60 the score gap between rank 1 and
    rank 10 is only ~0.002, which gives the merger almost nothing to work
    with. At k=20 it is ~0.014.

    SIDE EFFECT: writes the fused score onto each input chunk's
    `rrf_score` and returns those same objects. TemporalFilter depends on
    that (penalized_score is derived from rrf_score), and retrieve()
    depends on the second, cross-variant fusion overwriting the first —
    but it does mean two fusions over the same objects cannot be compared.

    `weights` scales each list's contribution. Plain (unweighted) RRF sums
    every list equally, which is correct when the lists are INDEPENDENT
    evidence — e.g. a dense ranking and a sparse ranking of the same query.
    It is wrong across multi-query expansion, where the variants are
    near-duplicates by construction ("X", "X under IPC", "X <synonyms>"):
    there, an unweighted sum rewards how redundantly a query was expanded
    rather than how well retrievers agree. See retrieve() for the decay
    schedule applied there.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists in length")

    scores: dict[str, float]          = {}
    chunks: dict[str, RetrievedChunk] = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, chunk in enumerate(ranked, start=1):
            sid = chunk.section_id
            scores[sid] = scores.get(sid, 0.0) + weight / (k + rank)
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
    def __init__(self, vocab_path: str = BM25_VOCAB_PATH, client=None,
                 embed_model=None, section_store: SectionStore | None = None,
                 verify_index: bool = True):
        # Qdrant's local mode file-locks the storage folder to a single
        # client, so callers that already have one (CaseIndexer, the
        # evaluation harness) must be able to share it. Same reasoning for
        # the embedding model: loading a second copy of bge-large on top of
        # an existing one is what used to CUDA-OOM the ablation study.
        self.client      = client or QdrantClient(path=QDRANT_PATH)
        if embed_model is None:
            # Imported lazily: every in-process caller (main.py, the
            # evaluation harness, the backend) injects an already-loaded
            # model, and requiring torch at import time made every module
            # that merely needs RetrievedChunk — temporal_filter,
            # chunk_structurer, the tests — unimportable without it.
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.embed_model = embed_model

        # Every artifact below is generated, gitignored, and written by a
        # single indexer run. A fresh clone has none of them, so report
        # that as the actionable thing it is rather than letting a bare
        # FileNotFoundError surface several frames deep.
        vocab_file = Path(vocab_path)
        if not vocab_file.exists():
            raise IndexMismatchError(
                f"{vocab_file} is missing",
                "The BM25 vocabulary is generated; a fresh checkout has no index yet.",
            )
        with open(vocab_file, "r") as f:
            self.vocab: dict[str, int] = json.load(f)

        idf_path = vocab_file.parent / "bm25_idf.json"
        if not idf_path.exists():
            raise IndexMismatchError(
                f"{idf_path} is missing",
                "The IDF table is written by the indexer alongside the vocabulary.",
            )
        with open(idf_path, "r") as f:
            self.idf: dict[int, float] = {int(k): v for k, v in json.load(f).items()}

        # One scroll of the corpus into memory. Every single-section lookup
        # in the pipeline goes through this instead of a filtered scroll —
        # see pipeline/section_store.py for why that matters here.
        self.sections = section_store or SectionStore(
            client=self.client, collection_name=COLLECTION_NAME,
        )

        if verify_index:
            self._verify_index_consistency(Path(vocab_path).parent)

        embed_model_ref = self.embed_model

        @lru_cache(maxsize=256)
        def _cached_encode(text: str) -> tuple:
            prefixed = f"{BGE_QUERY_INSTRUCTION}{text}"
            vec = embed_model_ref.encode(prefixed, normalize_embeddings=True)
            return tuple(vec.tolist())

        self._cached_encode = _cached_encode

    # ── Startup consistency check ────────────────────────────────────────────

    def _verify_index_consistency(self, data_dir: Path):
        """Refuse to serve a BM25 index that cannot have been produced by
        the vocabulary we just loaded.

        Sparse vectors are not self-describing: indices are integers whose
        meaning lives entirely in the vocabulary. If the two disagree, every
        query silently searches for the wrong terms and retrieval quality
        collapses with no error anywhere. This repository shipped in exactly
        that state — stored vectors from one tokenizer, vocabulary file from
        another, measured index agreement ~15%.

        Two independent checks, because either can catch what the other misses:
          1. the manifest, which records what the indexer actually built;
          2. a live probe of a real stored vector, which catches a stale
             Qdrant collection even when the on-disk files agree.
        """
        manifest = read_manifest(data_dir)
        if manifest is None:
            raise IndexMismatchError(
                f"no {data_dir}/bm25_manifest.json",
                "The index predates consistency checking and cannot be trusted.",
            )

        if manifest.get("index_format_version") != INDEX_FORMAT_VERSION:
            raise IndexMismatchError(
                f"index format v{manifest.get('index_format_version')} "
                f"but this code expects v{INDEX_FORMAT_VERSION}"
            )
        if manifest.get("tokenizer_version") != TOKENIZER_VERSION:
            raise IndexMismatchError(
                f"index built with tokenizer v{manifest.get('tokenizer_version')} "
                f"but this code tokenizes as v{TOKENIZER_VERSION}"
            )
        actual = vocab_fingerprint(self.vocab)
        if manifest.get("vocab_fingerprint") != actual:
            raise IndexMismatchError(
                "bm25_vocab.json does not match the vocabulary the corpus "
                "vectors were built with",
                f"manifest={manifest.get('vocab_fingerprint','?')[:16]}… "
                f"loaded={actual[:16]}…",
            )

        self._probe_stored_vector()

    def _probe_stored_vector(self, min_agreement: float = 0.9):
        """Read one real stored sparse vector back and confirm its indices
        decode against the loaded vocabulary. Cheap (one point) and it is
        the only check that actually looks at what Qdrant holds."""
        try:
            points, _ = self.client.scroll(
                collection_name=COLLECTION_NAME, limit=1,
                with_payload=True, with_vectors=True,
            )
        except Exception as e:                      # pragma: no cover
            print(f"[HybridRetriever] WARNING: could not probe stored vectors ({e}).")
            return
        if not points:
            raise IndexMismatchError("the Qdrant collection is empty")

        point   = points[0]
        vectors = point.vector if isinstance(point.vector, dict) else {}
        sparse  = vectors.get("sparse")
        if sparse is None:
            raise IndexMismatchError("stored points carry no 'sparse' vector")

        stored_idx = set(getattr(sparse, "indices", []) or [])
        if not stored_idx:
            return

        sid    = (point.payload or {}).get("section_id", "")
        record = self.sections.get_record(sid)
        if record is None:
            return
        expected = {
            self.vocab[t] for t in set(bm25_tokenize(build_search_text(record)))
            if t in self.vocab
        }
        if not expected:
            return

        agreement = len(stored_idx & expected) / len(stored_idx)
        if agreement < min_agreement:
            raise IndexMismatchError(
                f"stored sparse vectors do not decode against bm25_vocab.json "
                f"(section {sid}: only {agreement:.0%} of its stored term ids "
                f"correspond to its own text)",
                "The Qdrant collection was built from a different vocabulary "
                "than the one on disk.",
            )

    # ── Public: properly-prefixed query embedding ────────────────────────────

    def embed_query_vector(self, text: str) -> list[float]:
        return list(self._cached_encode(text))

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def _dense_retrieve(
        self,
        query:   str,
        top_k:   int           = DENSE_TOP_K,
        filters: Filter | None = None,
    ) -> list[RetrievedChunk]:
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

    def _bm25_query_weights(self, query: str) -> dict[int, float]:
        """Query side of the Okapi BM25 decomposition:

            w_q(t) = idf(t) * (qtf*(k3+1)) / (k3 + qtf)

        The document side (term-frequency saturation and length
        normalization) is baked into the stored vectors by
        data/indexer.py, so the dot product of the two reproduces BM25
        exactly. IDF appears here and ONLY here — the previous code applied
        it on both sides, effectively squaring it.
        """
        qtf: dict[int, int] = {}
        for t in bm25_tokenize(query):
            idx = self.vocab.get(t)
            if idx is not None:
                qtf[idx] = qtf.get(idx, 0) + 1
        return {
            idx: self.idf.get(idx, 0.0) * (f * (BM25_K3 + 1.0)) / (BM25_K3 + f)
            for idx, f in qtf.items()
        }

    def _sparse_retrieve(
        self,
        query:   str,
        top_k:   int           = BM25_TOP_K,
        filters: Filter | None = None,
    ) -> list[RetrievedChunk]:
        weights = self._bm25_query_weights(query)
        if not weights:
            return []
        results = self.client.query_points(
            collection_name = COLLECTION_NAME,
            query           = SparseVector(indices=list(weights.keys()),
                                           values=list(weights.values())),
            using           = "sparse",
            query_filter    = filters,
            limit           = top_k,
            with_payload    = True,
        )
        return [self._hit_to_chunk(h) for h in results.points]

    # ── Direct section lookup ─────────────────────────────────────────────────

    def _direct_section_lookup(self, query: str) -> list[RetrievedChunk]:
        """Exact "the user named this section" fast path.

        Delegates parsing to data/section_ref.py, which handles both word
        orders ("IPC 302" and "section 302 of the IPC"), multi-number runs
        ("IPC 494 and 495" — the old regex silently kept only 494), letter
        suffixes and the CRPC zero-padding exception. Resolution goes
        through the in-memory SectionStore, so this costs no Qdrant calls
        at all; it used to issue two full-collection scans per match per
        query variant.
        """
        chunks, seen = [], set()
        for sid in extract_section_refs(query, self.sections.section_ids):
            if sid in seen:
                continue
            payload = self.sections.get(sid)
            if payload:
                seen.add(sid)
                chunks.append(self._payload_to_chunk(payload, score=1.0))
        return chunks

    # ── Fetch by explicit section IDs (for section pinner) ───────────────────

    def fetch_by_ids(self, section_ids: list[str]) -> list[RetrievedChunk]:
        """Sections by id, in the order given, skipping any that don't exist.
        Served from the in-memory store — no Qdrant round-trip per id.
        Accepts loose references ("IPC 2", "ipc 304a") as well as canonical
        ids, so callers don't each need their own normalization."""
        chunks, seen = [], set()
        for raw in section_ids:
            sid = raw if raw in self.sections else self.sections.resolve(raw)
            if not sid or sid in seen:
                continue
            payload = self.sections.get(sid)
            if payload:
                seen.add(sid)
                chunk = self._payload_to_chunk(payload, score=1.0)
                chunk.rrf_score = 1.0     # treat as top-priority
                chunks.append(chunk)
        return chunks

    # ── Hybrid retrieval ──────────────────────────────────────────────────────

    def retrieve(
        self,
        queries:            list[str],
        top_k:              int           = HYBRID_TOP_K,
        act_filter:         str | None    = None,
        status_filter:      str | None    = None,
        dense_act_filter:   str | None    = None,
        direct_lookup_query: str | None   = None,
    ) -> list[RetrievedChunk]:
        """direct_lookup_query: the text to scan for EXPLICIT section
        references, which are then pinned to the front and guaranteed a
        place in the result.

        This must be the USER'S OWN WORDS, not an expanded variant.
        Stage 0b's offline synonym rules emit literal section numbers as
        recall hints — "stole" expands to "... IPC 378 IPC 379 IPC 390
        IPC 392" — and scanning the expanded text made the retriever treat
        those guesses as though the user had named them: pinned to rank 1,
        weight 2.0 here, and floored at 0.85 by the reranker. Measured
        effect: "a hacker stole my OTP and drained my account" returned
        four theft sections above ITA_066, because the word "stole"
        triggered the theft rule. Those hints are still valuable, but as
        ordinary query text competing on retrieval merit — which is what
        happens when they are left out of this scan.

        Defaults to the first query for backward compatibility.
        """

        # Sparse gets the act filter (keyword false positives are common and
        # the filter genuinely reduces noise); dense does not, so cross-act
        # candidates — IPC_420 for a cybercrime query — survive to fusion.
        def _build(status: str | None, act: str | None) -> Filter | None:
            conditions = []
            if status:
                conditions.append(
                    FieldCondition(key="status", match=MatchValue(value=status.strip().title()))
                )
            if act:
                conditions.append(FieldCondition(key="act_code", match=MatchValue(value=act)))
            return Filter(must=conditions) if conditions else None

        sparse_filter = _build(status_filter, act_filter)
        dense_filter  = _build(status_filter, dense_act_filter)

        # Preserve caller order but drop exact duplicates — a repeated
        # variant is not extra evidence, and with weighted fusion below it
        # would otherwise consume a high-weight slot.
        unique_queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))

        # ── Step 1: direct section hits (always included) ─────────────────────
        # Scanned from the user's own words only — see direct_lookup_query.
        lookup_text = direct_lookup_query
        if lookup_text is None:
            lookup_text = unique_queries[0] if unique_queries else ""
        direct_chunks, seen_ids = [], set()
        for chunk in self._direct_section_lookup(lookup_text):
            if chunk.section_id not in seen_ids:
                direct_chunks.append(chunk)
                seen_ids.add(chunk.section_id)

        # ── Step 2: per-variant dense+sparse fusion ──────────────────────────
        # Within one query, dense and sparse are independent evidence about
        # the same information need, so they fuse with equal weight.
        per_query_rankings = []
        for query in unique_queries:
            dense_results  = self._dense_retrieve(query, top_k=DENSE_TOP_K, filters=dense_filter)
            sparse_results = self._sparse_retrieve(query, top_k=BM25_TOP_K, filters=sparse_filter)
            per_query_rankings.append(
                reciprocal_rank_fusion([dense_results, sparse_results])
            )

        # ── Step 3: cross-variant fusion, with decaying weight ───────────────
        # Query variants are NOT independent evidence: Stage 2 of the
        # pipeline generates them as paraphrases of one another, so summing
        # them equally scores how redundantly a query was expanded. The
        # first variant is the translated primary query and keeps full
        # weight; later variants are recall boosters and contribute less,
        # bottoming out at 0.35 rather than at zero so a section that only
        # a late paraphrase reaches can still surface.
        variant_weights = [max(0.35, 1.0 / (1.0 + 0.5 * i))
                           for i in range(len(per_query_rankings))]

        ranked_lists = list(per_query_rankings)
        weights      = list(variant_weights)
        if direct_chunks:
            # An explicitly named section outranks anything inferred.
            ranked_lists.insert(0, direct_chunks)
            weights.insert(0, 2.0)

        rrf_results = reciprocal_rank_fusion(ranked_lists, weights=weights)

        if direct_chunks:
            direct_ids = {c.section_id for c in direct_chunks}
            final = list(direct_chunks) + [c for c in rrf_results
                                           if c.section_id not in direct_ids]
        else:
            final = rrf_results

        return final[:top_k]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _payload_to_chunk(self, p: dict, score: float = 0.0) -> RetrievedChunk:
        return RetrievedChunk(
            section_id       = p.get("section_id", ""),
            content          = p.get("content", ""),
            score            = score,
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

    def _hit_to_chunk(self, hit, default_score: float = 0.0) -> RetrievedChunk:
        chunk = self._payload_to_chunk(
            hit.payload or {}, score=getattr(hit, "score", default_score),
        )
        if not chunk.section_id:
            chunk.section_id = str(hit.id)
        return chunk


if __name__ == "__main__":
    retriever = HybridRetriever()
    results   = retriever.retrieve(queries=["punishment for cybercrime under IT Act"], top_k=5)
    for r in results:
        print(f"{r.section_id} | rrf={r.rrf_score:.4f} | {r.category}")
