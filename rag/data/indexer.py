"""
data/indexer.py
Index final_dataset.json into Qdrant with:
  - Dense vectors  (BGE-large, passage side — no query instruction prefix)
  - Sparse vectors (Okapi BM25, document side)
  - Rich metadata payload for filtering
  - A manifest (data/bm25_manifest.json) binding all of the above together

Sparse vectors: a real Okapi BM25 decomposition
-----------------------------------------------
BM25 is a sum over query terms:

    score(q,d) = SUM_t  idf(t) * qtf_w(t) * (tf_d(t)*(k1+1))
                                / (tf_d(t) + k1*(1 - b + b*|d|/avgdl))

Split across a sparse dot product, that is:

    document weight  w_d(t) = (tf*(k1+1)) / (tf + k1*(1 - b + b*|d|/avgdl))
    query weight     w_q(t) = idf(t) * (qtf*(k3+1)) / (k3 + qtf)

so <w_d, w_q> == BM25 exactly. This file writes w_d; the idf table it
saves is what pipeline/hybrid_retriever.py uses to build w_q.

The previous implementation stored raw `tf * idf` on BOTH sides. That is
not BM25: it applied IDF twice (squaring it), had no term-frequency
saturation, and — most importantly — no document-length normalization, so
a section's score grew without bound with its length. Corpus-side vector
L2 norm correlated 0.81 with document length before this change.

Consistency is enforced, not assumed
------------------------------------
Sparse vectors are keyed by integer ids from a vocabulary built here.
Rebuilding that vocabulary without rewriting the vectors silently
repoints every query at the wrong terms. This script therefore writes
vocab, idf, Qdrant vectors and the manifest in ONE run, and there is
deliberately no partial-rebuild flag — see data/index_manifest.py.
"""
import json
import math
import sys
from pathlib import Path

from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector,
)
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    QDRANT_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    BM25_K1, BM25_B,
)
from data.bm25_tokenizer import tokenize, build_search_text, TOKENIZER_VERSION
from data.index_manifest import build_manifest, write_manifest
from data.payload import build_payload, resolve_related_refs
from data.bm25 import build_vocab, compute_idf, bm25_document_weights


# ── Helpers ───────────────────────────────────────────────────────────────────

# ── Main indexer ──────────────────────────────────────────────────────────────

class LegalIndexer:
    def __init__(self, json_path: str, qdrant_path: str = QDRANT_PATH):
        self.json_path   = json_path
        self.client      = QdrantClient(path=qdrant_path)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

    def _setup_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME in existing:
            print(f"Collection '{COLLECTION_NAME}' exists — recreating.")
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
            },
        )

        # NOTE: payload indexes are a no-op in Qdrant's embedded/local mode
        # (storage there is a single sqlite points table and every filter is
        # a full scan). They are created anyway so that pointing this at a
        # real Qdrant server gets the right indexes with no code change.
        # The pipeline does NOT rely on them being fast: hot section_id
        # lookups go through pipeline/section_store.py's in-memory index.
        for field, schema in [
            ("section_id",      "keyword"),
            ("act_code",        "keyword"),
            ("section_number",  "keyword"),
            ("status",          "keyword"),
            ("intent",          "keyword"),
            ("conclusion_type", "keyword"),
            ("enacted_year",    "integer"),
            ("jurisdiction",    "keyword"),
        ]:
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )

        print(f"Collection '{COLLECTION_NAME}' created with dense + sparse vectors.")

    def _embed_all(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encode the whole corpus in batches.

        The previous code called encode([one_text]) once per record inside
        the indexing loop — a batch size of ONE, 3394 separate forward
        passes. Measured on CPU that ran at ~2.4 records/s (~24 minutes);
        batching is several times faster and is the only reason
        batch_size was being passed at all.
        """
        return self.embed_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=batch_size,
        ).tolist()

    def run(self, batch_size: int = 64, embed_batch_size: int = 32):
        """batch_size is the Qdrant upsert size; embed_batch_size is the
        encoder's. They are separate knobs — the first is about network/IO
        chunking, the second about GPU/CPU memory — and conflating them
        makes one of the two wrong on any machine."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        print(f"Loaded {len(records)} records.")
        self._setup_collection()

        section_ids = [r["section"] for r in records]
        known_ids   = set(section_ids)

        # The single definition of "the text this record is indexed on",
        # shared with the dense vectors below and with
        # data/build_bm25_idf.py — includes meta.keywords, the curated
        # bridge vocabulary that maps lay phrasing onto statutory wording.
        print("Tokenizing corpus...")
        all_texts        = [build_search_text(r) for r in records]
        tokenized_corpus = [tokenize(t) for t in all_texts]

        print("Building BM25 vocabulary...")
        vocab = build_vocab(tokenized_corpus)
        print(f"Vocabulary size: {len(vocab)}")

        print("Computing document frequency / IDF...")
        idf   = compute_idf(tokenized_corpus, vocab)
        avgdl = sum(len(t) for t in tokenized_corpus) / max(len(tokenized_corpus), 1)
        print(f"Average document length: {avgdl:.1f} tokens")

        print("Encoding dense vectors...")
        dense_vectors = self._embed_all(all_texts, batch_size=embed_batch_size)

        points: list[PointStruct] = []
        failed: list[dict] = []
        dropped_refs = 0

        for i, record in enumerate(tqdm(records, desc="Indexing")):
            try:
                dense_vec = dense_vectors[i]

                weights    = bm25_document_weights(tokenized_corpus[i], vocab, avgdl)
                payload    = build_payload(record)

                # Resolve cross-references to real section_ids, dropping any
                # that name a section this corpus doesn't contain.
                resolved, dropped = resolve_related_refs(
                    payload["related_sections_raw"], known_ids)
                payload["related_sections"] = resolved
                dropped_refs += dropped

                points.append(PointStruct(
                    id=i,
                    vector={
                        "dense":  dense_vec,
                        "sparse": SparseVector(
                            indices=list(weights.keys()),
                            values=list(weights.values()),
                        ),
                    },
                    payload=payload,
                ))

                if len(points) >= batch_size:
                    self.client.upsert(COLLECTION_NAME, points=points)
                    points = []

            except Exception as e:
                failed.append({"index": i, "section": record["section"], "error": str(e)})

        if points:
            self.client.upsert(COLLECTION_NAME, points=points)

        total = self.client.count(COLLECTION_NAME).count
        print(f"\nIndexed : {total} points in '{COLLECTION_NAME}'")
        print(f"Cross-references resolved; {dropped_refs} dangling reference(s) dropped.")
        if failed:
            print(f"Failed  : {len(failed)}")
            for f_ in failed[:3]:
                print(f"  {f_['section']}: {f_['error']}")

        data_dir = Path(self.json_path).parent

        with open(data_dir / "bm25_idf.json", "w") as jf:
            json.dump({str(k): v for k, v in idf.items()}, jf)
        with open(data_dir / "bm25_vocab.json", "w") as vf:
            json.dump(vocab, vf)

        manifest = build_manifest(
            section_ids       = section_ids,
            search_texts      = all_texts,
            vocab             = vocab,
            tokenizer_version = TOKENIZER_VERSION,
            n_docs            = len(records),
            avgdl             = avgdl,
            k1                = BM25_K1,
            b                 = BM25_B,
            collection_name   = COLLECTION_NAME,
        )
        manifest_path = write_manifest(manifest, data_dir)

        print(f"BM25 idf      -> {data_dir / 'bm25_idf.json'}")
        print(f"BM25 vocab    -> {data_dir / 'bm25_vocab.json'}")
        print(f"Index manifest-> {manifest_path}")
        print("\nVocabulary, IDF, Qdrant vectors and manifest were written by the "
              "same run and are therefore consistent by construction.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Index the legal dataset into Qdrant (full rebuild only).",
        epilog=(
            "There is deliberately no partial/vocab-only rebuild option. "
            "Rebuilding the vocabulary without rewriting the stored sparse "
            "vectors repoints every query at the wrong terms and silently "
            "destroys retrieval quality — see data/index_manifest.py."
        ),
    )
    parser.add_argument("json_path", nargs="?", default="./data/final_dataset.json",
                        help="Path to final_dataset.json")
    args = parser.parse_args()

    LegalIndexer(args.json_path).run()
