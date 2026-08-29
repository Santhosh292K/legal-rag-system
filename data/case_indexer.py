"""
data/case_indexer.py
Phase 1, stage 6 — Case Vector Store.

Mirrors data/indexer.py's structure but for the dynamic, incremental
case-document corpus rather than the static statute corpus.

Design note: unlike the statute corpus, a single case's chunk set is
small (a handful of documents), added incrementally as evidence is
uploaded — not reindexed in bulk. Corpus-wide BM25 (which needs a
fixed vocabulary and document frequencies) doesn't fit that pattern
well here, so this store is dense-only. The existing BM25 + RRF hybrid
retrieval is still used, unchanged, on the statute side — case chunks
reuse dense search only, which is what ALEA's element-matching step
(cosine similarity between an element description and an evidence
fact) actually needs anyway.
"""
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import CASE_DOCUMENTS_COLLECTION, CASE_QDRANT_PATH, EMBEDDING_MODEL, EMBEDDING_DIM

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
)
from sentence_transformers import SentenceTransformer

from pipeline.adaptive_chunkers import Chunk
from pipeline.hybrid_retriever import BGE_QUERY_INSTRUCTION


PAYLOAD_INDEX_FIELDS = [
    ("case_id", "keyword"),
    ("document_id", "keyword"),
    ("doc_type", "keyword"),
    ("chunk_role", "keyword"),
]


class CaseIndexer:
    def __init__(self, qdrant_path: str = CASE_QDRANT_PATH, client=None, embed_model=None):
        # Same reasoning as HybridRetriever — CASE_QDRANT_PATH and
        # QDRANT_PATH point at the same folder by design (one Qdrant
        # instance, two collections), so this must reuse a shared client
        # rather than open a second one on the same locked path.
        self.client = client or QdrantClient(path=qdrant_path)
        # BUGFIX: this used to always load a second, independent copy of
        # bge-large even when the caller (evaluate_cases.py) already had
        # one loaded via pipeline.retriever.embed_model — the same
        # double-loading pattern that caused the ablation OOM. Accept an
        # already-loaded model and only load a fresh one as a fallback.
        self.embed_model = embed_model or SentenceTransformer(EMBEDDING_MODEL)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if CASE_DOCUMENTS_COLLECTION in existing:
            return

        self.client.create_collection(
            collection_name=CASE_DOCUMENTS_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        for field_name, schema in PAYLOAD_INDEX_FIELDS:
            self.client.create_payload_index(
                collection_name=CASE_DOCUMENTS_COLLECTION,
                field_name=field_name,
                field_schema=schema,
            )
        print(f"Collection '{CASE_DOCUMENTS_COLLECTION}' created.")

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and upsert a list of Chunk objects. Safe to call
        repeatedly as new documents are uploaded to the same case —
        and safe to call again for the SAME document (same document_id),
        since it replaces that document's existing chunks first rather
        than accumulating duplicates."""
        if not chunks:
            return 0

        document_ids = {c.document_id for c in chunks}
        for doc_id in document_ids:
            self.delete_document(case_id=chunks[0].case_id, document_id=doc_id)

        texts = [c.text for c in chunks]
        vectors = self.embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32,
        ).tolist()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "case_id": chunk.case_id,
                    "document_id": chunk.document_id,
                    "doc_type": chunk.doc_type,
                    "chunk_role": chunk.chunk_role,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        self.client.upsert(CASE_DOCUMENTS_COLLECTION, points=points)
        return len(points)

    def get_all_chunks(self, case_id: str, limit: int = 200) -> list[dict]:
        """Fetches every chunk for a case, unfiltered by query similarity —
        for broad questions ('summarize this case', 'what are the case
        details') where dense top-k search against the vague question text
        wouldn't reliably surface all the relevant facts. Uses Qdrant's
        scroll (not search), so there's no query vector involved at all."""
        points, _ = self.client.scroll(
            collection_name=CASE_DOCUMENTS_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]),
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "score": 1.0,  # not similarity-ranked — every chunk is included
                "text": p.payload["text"],
                "doc_type": p.payload["doc_type"],
                "chunk_role": p.payload["chunk_role"],
                "document_id": p.payload["document_id"],
                "metadata": p.payload.get("metadata", {}),
            }
            for p in points
        ]

    def search(self, query: str, case_id: str, top_k: int = 10,
               chunk_role: str | None = None) -> list[dict]:
        """Dense search scoped to a single case. chunk_role optionally
        narrows to a specific role, e.g. only 'diagnosis' chunks."""
        # BUGFIX: same asymmetric-encoding bug as SectionPinner (see
        # main.py's pin_embed_fn note) — index_chunks() above embeds
        # chunk.text with no prefix (correct: that's the passage side),
        # but this query-side encode was missing the bge-large query
        # instruction prefix that passage-vs-query search needs to score
        # correctly. Without it, case-document search similarity is
        # systematically depressed, same as it was for section pinning.
        vector = self.embed_model.encode(
            [f"{BGE_QUERY_INSTRUCTION}{query}"], normalize_embeddings=True
        )[0].tolist()

        must = [FieldCondition(key="case_id", match=MatchValue(value=case_id))]
        if chunk_role:
            must.append(FieldCondition(key="chunk_role", match=MatchValue(value=chunk_role)))

        results = self.client.query_points(
            collection_name=CASE_DOCUMENTS_COLLECTION,
            query=vector,
            query_filter=Filter(must=must),
            limit=top_k,
        ).points

        return [
            {
                "score": r.score,
                "text": r.payload["text"],
                "doc_type": r.payload["doc_type"],
                "chunk_role": r.payload["chunk_role"],
                "document_id": r.payload["document_id"],
            }
            for r in results
        ]

    def delete_document(self, case_id: str, document_id: str):
        """Remove all chunks for one document — called automatically before
        re-indexing that same document, and usable directly if a document
        needs to be removed from a case without deleting the whole case."""
        self.client.delete(
            collection_name=CASE_DOCUMENTS_COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key="case_id", match=MatchValue(value=case_id)),
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            ]),
        )

    def delete_case(self, case_id: str):
        """Remove all chunks for a case — e.g. on user-initiated deletion."""
        self.client.delete(
            collection_name=CASE_DOCUMENTS_COLLECTION,
            points_selector=Filter(must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]),
        )


if __name__ == "__main__":
    from pipeline.document_pipeline import DocumentIngestionPipeline
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--case_id", default="demo-case")
    args = parser.parse_args()

    ingested = DocumentIngestionPipeline().ingest(args.file_path, case_id=args.case_id)
    indexer = CaseIndexer()
    n = indexer.index_chunks(ingested.chunks)
    print(f"Indexed {n} chunks for case '{args.case_id}'.")