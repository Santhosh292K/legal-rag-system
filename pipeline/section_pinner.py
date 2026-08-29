"""
pipeline/section_pinner.py
Semantic section pinner — NO hand-maintained scenario table.

The old version of this file matched queries against ~70 hand-written
regex patterns, each hardcoded to a list of section IDs ("stabbed and
died" -> IPC_300, IPC_302...). That table only ever covered scenarios
someone thought to anticipate, and every new scenario meant writing a new
regex.

Why a pinning stage exists at all (unchanged): BM25 needs exact vocabulary
overlap between the query and section content, so a lay-language query can
miss a section BM25 alone would never surface, and the RRF-fused pool can
still drop a section that was correct but ranked just outside the cutoff
on both lists. This stage removes that risk by unconditionally including
sections the dense model is highly confident about, regardless of what
BM25/RRF/reranking do with them — same guarantee the old PIN_TABLE made.

The difference: instead of matching the query against a hand-curated
example table, this searches directly against the SAME indexed corpus the
rest of retrieval already uses (the "legal_sections" Qdrant collection,
bge-large embeddings). There's nothing to maintain here — every section
already in the index is automatically a candidate. Improving the
embedding model or the underlying dataset improves this stage for free;
a regex table never could.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

sys.path.append(str(Path(__file__).parent.parent))
from config import COLLECTION_NAME

from qdrant_client import QdrantClient


# Minimum cosine similarity (dense vectors are normalized, so Qdrant's
# returned score IS cosine similarity) to trust a match enough to
# unconditionally pin it, bypassing the rest of retrieval/reranking.
# Deliberately conservative — this stage trades recall for a hard
# guarantee, so it should only fire on genuinely close matches. NOTE: this
# was picked as a reasonable starting point, not empirically tuned against
# bge-large — validate against evaluation/benchmark_scenarios.json and
# adjust before relying on it in production.
MIN_PIN_SIMILARITY = 0.62

# How many top dense hits to consider pinning per query. Small on purpose:
# this isn't meant to replace the main retrieval pass (which already runs
# a much larger dense + BM25 + RRF search via HybridRetriever), just to
# guarantee the highest-confidence matches survive regardless of what else
# happens downstream.
PIN_TOP_K = 6

# Shared with answer_generator.py and main.py, which both need to identify
# a RankedChunk as "came from the pinner" — a single source of truth
# instead of the same literal string duplicated in two files (which is how
# the old version of this desynced silently: nothing would error if one
# side's string went stale, the pinned-section rescue logic would just
# quietly stop matching anything).
PIN_EXPLANATION = "Pinned by section_pinner (semantic match)"


@dataclass
class PinResult:
    section_ids:   list = field(default_factory=list)
    # Kept for logging/debug parity with the old pinner's matched_rules —
    # now holds the similarity score behind each pin instead of which
    # regex fired.
    matched_rules: list = field(default_factory=list)


class SectionPinner:
    """
    Reuses an already-open Qdrant client and embedding model (e.g.
    HybridRetriever's, wired up in main.py) rather than opening a second
    client against the same file-locked ./qdrant_db folder or loading a
    second copy of bge-large.

    embed_fn follows the same convention used elsewhere in this codebase
    (ALEA, QueryRouter): embed_fn(list[str]) -> vectors.
    """

    def __init__(
        self,
        client: QdrantClient,
        embed_fn: Callable[[list], object],
        collection_name: str = COLLECTION_NAME,
        top_k: int = PIN_TOP_K,
        min_similarity: float = MIN_PIN_SIMILARITY,
    ):
        self.client = client
        self.embed_fn = embed_fn
        self.collection_name = collection_name
        self.top_k = top_k
        self.min_similarity = min_similarity

    def pin(self, query: str) -> PinResult:
        vec = self.embed_fn([query])[0]
        # Qdrant client accepts numpy arrays, but be defensive in case a
        # caller's embed_fn returns something without .tolist() already
        # applied (e.g. a raw list).
        if hasattr(vec, "tolist"):
            vec = vec.tolist()

        results = self.client.query_points(
            collection_name = self.collection_name,
            query           = vec,
            using           = "dense",
            limit           = self.top_k,
            with_payload    = True,
        )

        section_ids: list = []
        rules: list = []
        for hit in results.points:
            if hit.score < self.min_similarity:
                continue
            payload = hit.payload or {}
            sid = payload.get("section_id", str(hit.id))
            if sid not in section_ids:
                section_ids.append(sid)
                rules.append(f"dense sim={hit.score:.2f}")

        return PinResult(section_ids=section_ids, matched_rules=rules)


if __name__ == "__main__":
    """
    Standalone smoke test. Opens its own Qdrant client + bge-large model
    (main.py normally shares HybridRetriever's copy of both instead).
    """
    from config import QDRANT_PATH, EMBEDDING_MODEL
    from sentence_transformers import SentenceTransformer
    from pipeline.hybrid_retriever import BGE_QUERY_INSTRUCTION

    client = QdrantClient(path=QDRANT_PATH)
    model  = SentenceTransformer(EMBEDDING_MODEL)
    # Must match main.py's pin_embed_fn: this pinner dense-searches the
    # same passage-embedded corpus HybridRetriever does, so it needs the
    # same asymmetric query prefix — see the BUGFIX note in main.py.
    embed_fn = lambda texts: [
        model.encode(f"{BGE_QUERY_INSTRUCTION}{t}", normalize_embeddings=True).tolist()
        for t in texts
    ]

    pinner = SectionPinner(client=client, embed_fn=embed_fn)

    tests = [
        "If there is a robbery and complaint registered but police failed to take action, what is the offence?",
        "Ramesh got a fake degree and practiced medicine, but a patient died under his medication",
        "A police officer took bribe to drop the FIR",
        "My cheque bounced and the other party is not paying",
        "A husband beat his wife and her in-laws demanded dowry",
        "What is IPC 302?",
        "How do I apply for bail?",
        "A Dalit person was abused because of caste",
        "A hacker stole my OTP and transferred money",
        "A contractor forced tribal people to work without pay and seized their land",
    ]
    for q in tests:
        r = pinner.pin(q)
        print(f"\nQ: {q[:70]}")
        print(f"  Pinned ({len(r.section_ids)}): {list(zip(r.section_ids, r.matched_rules))}")