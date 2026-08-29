"""
data/indexer.py
Index final_dataset.json into Qdrant with:
  - Dense vectors  (BGE-large)
  - Sparse vectors (BM25)
  - Rich metadata payload for filtering
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
# NOTE: BM25Okapi is no longer used for corpus-side sparse vectors — see
# the fix note in run() below. get_scores() returns per-DOCUMENT scores,
# not per-term weights, and was being misused as one. Corpus vectors are
# now plain TF(doc) * IDF(corpus), matching HybridRetriever's query-side
# encoding.

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    QDRANT_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIM,
)
from data.bm25_tokenizer import tokenize


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def build_payload(record: dict) -> dict:
    m = record["meta"]
    return {
        "section_id":       record["section"],
        "act_code":         m["code"],
        "act_name":         m["hierarchy"]["act"],
        "section_number":   m["section"],
        "chapter":          m.get("chapter") or "",
        "category":         m.get("category") or "",
        "keywords":         m.get("keywords") or [],
        "status":           m["temporal"]["status"] or "active",
        "enacted_year":     m["temporal"]["enacted_year"],
        "last_amended":     m["temporal"]["last_amended"] or "",
        "amended_by":       m["temporal"]["amended_by"] or [],
        # Bug fix: these two were never copied into the payload, so
        # TemporalFilter's `chunk.payload.get("superseded_by")` check
        # (pipeline/temporal_filter.py) always returned None regardless of
        # what final_dataset.json actually said — the "superseded" penalty
        # path was dead code in practice. Re-index after this change.
        "superseded_by":    m["temporal"].get("superseded_by") or "",
        "supersedes":       m["temporal"].get("supersedes") or "",
        "intent":           m["legal_type"]["intent"] or "",
        "rule_type":        m["legal_type"]["rule_type"] or "",
        "applies_to":       m["legal_type"]["applies_to"] or [],
        "punishment_type":  m["legal_type"]["punishment_type"] or "",
        "jurisdiction":     m["legal_type"]["jurisdiction"] or "India",
        "issue_tags":       m["irac"]["issue_tags"] or [],
        "rule_summary":     m["irac"]["rule_summary"] or "",
        "conclusion_type":  m["irac"]["conclusion_type"] or "",
        "parent_section":   m["hierarchy"]["parent_section"] or "",
        "child_sections":   m["hierarchy"]["child_sections"] or [],
        "related_sections": m["related_sections"] or [],
        "content":          record["content"],
        "embedding_text":   record["embedding_text"],
    }


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
                "dense": VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )

        # Payload indices using string literals (compatible with all qdrant versions)
        for field, schema in [
            ("act_code",        "keyword"),
            ("status",          "keyword"),
            ("intent",          "keyword"),
            ("conclusion_type", "keyword"),
            ("enacted_year",    "integer"),
            ("jurisdiction",    "keyword"),
        ]:
            self.client.create_payload_index(
                collection_name = COLLECTION_NAME,
                field_name      = field,
                field_schema    = schema,
            )

        print(f"Collection '{COLLECTION_NAME}' created with dense + sparse vectors.")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embed_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        ).tolist()

    def run(self, batch_size: int = 64):
        with open(self.json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        print(f"Loaded {len(records)} records.")
        self._setup_collection()

        # Build vocab (once, not per-record)
        print("Building BM25 vocabulary...")
        all_texts        = [r["embedding_text"] for r in records]
        tokenized_corpus = [tokenize(t) for t in all_texts]
        vocab            = build_vocab(all_texts)
        print(f"Vocabulary size: {len(vocab)}")

        # BUGFIX (was the root cause of degraded hybrid/BM25 retrieval):
        # The old code called `bm25_corpus.get_scores(tokens)`, which
        # returns ONE relevance score PER DOCUMENT IN THE CORPUS (i.e. how
        # well this document's tokens match every other document), not a
        # per-vocabulary-term weight. It then indexed that array with a
        # *vocab* token id, clamped to the corpus size. Since vocab size
        # (len(vocab)) is usually much larger than the corpus size
        # (n_docs), most distinct tokens collapsed onto the same clamped
        # index and got assigned the same near-arbitrary value — stored
        # sparse vectors ended up dominated by stopwords ("of", "to",
        # "section") instead of the terms that actually distinguish a
        # document, and lived in a totally different value-space than the
        # TF*IDF query vectors HybridRetriever._sparse_retrieve builds.
        # Sparse search was effectively querying real vectors against
        # noise.
        #
        # Fix: build each document's sparse vector the same way the query
        # side does — term-frequency-in-this-document × real IDF — so
        # corpus vectors and query vectors live in the same space and their
        # dot product is meaningful.
        print("Computing document frequency / IDF for corpus-side sparse vectors...")
        df = [0] * len(vocab)
        for tokens in tokenized_corpus:
            for tok in set(tokens):
                df[vocab[tok]] += 1
        n_docs = len(records)
        idf = {}
        for tok, idx in vocab.items():
            d = df[idx]
            idf[idx] = math.log(1 + (n_docs - d + 0.5) / (d + 0.5))

        points = []
        failed = []

        for i, record in enumerate(tqdm(records, desc="Indexing")):
            try:
                # Dense vector
                dense_vec = self._embed_batch([record["embedding_text"]])[0]

                # BM25-style sparse vector: TF(this doc) * IDF(corpus-wide)
                tokens = tokenized_corpus[i]
                tf: dict[int, float] = {}
                for token in tokens:
                    if token in vocab:
                        idx = vocab[token]
                        tf[idx] = tf.get(idx, 0.0) + 1.0

                dedup = {idx: count * idf[idx] for idx, count in tf.items()}

                sp_indices = list(dedup.keys())
                sp_values  = list(dedup.values())

                points.append(PointStruct(
                    id     = i,
                    vector = {
                        "dense":  dense_vec,
                        "sparse": SparseVector(
                            indices=sp_indices,
                            values=sp_values,
                        ),
                    },
                    payload=build_payload(record),
                ))

                if len(points) >= batch_size:
                    self.client.upsert(COLLECTION_NAME, points=points)
                    points = []

            except Exception as e:
                failed.append({
                    "index":   i,
                    "section": record["section"],
                    "error":   str(e),
                })

        if points:
            self.client.upsert(COLLECTION_NAME, points=points)

        total = self.client.count(COLLECTION_NAME).count
        print(f"\nIndexed : {total} points in '{COLLECTION_NAME}'")
        if failed:
            print(f"Failed  : {len(failed)}")
            for f_ in failed[:3]:
                print(f"  {f_['section']}: {f_['error']}")

        # Save idf alongside vocab so HybridRetriever's query-side encoding
        # (which loads data/bm25_idf.json) is guaranteed to match the same
        # df/idf this indexing run just used for the corpus vectors —
        # rather than relying on data/build_bm25_idf.py having been run
        # separately and possibly drifting out of sync (e.g. after the
        # dataset changes and a re-index, but before someone remembers to
        # rerun the standalone idf script).
        idf_path = Path(self.json_path).parent / "bm25_idf.json"
        with open(idf_path, "w") as jf:
            json.dump({str(k): v for k, v in idf.items()}, jf)
        print(f"BM25 idf saved   → {idf_path}")

        # Save vocab for retrieval use
        vocab_path = Path(self.json_path).parent / "bm25_vocab.json"
        with open(vocab_path, "w") as vf:
            json.dump(vocab, vf)
        print(f"BM25 vocab saved → {vocab_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index legal dataset into Qdrant.")
    parser.add_argument("json_path", nargs="?", default="./data/final_dataset.json",
                        help="Path to final_dataset.json")
    parser.add_argument("--rebuild-vocab", action="store_true",
                        help=(
                            "Gap 27 fix: rebuild ONLY the BM25 vocabulary (bm25_vocab.json) "
                            "without re-indexing all vectors. Run this after adding new acts "
                            "or documents to the dataset — failing to do so means new terms "
                            "will never match BM25 queries since the vocab index is stale."
                        ))
    args = parser.parse_args()

    if args.rebuild_vocab:
        print(f"[indexer] Rebuilding BM25 vocabulary from {args.json_path} ...")
        with open(args.json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        all_texts = [r["embedding_text"] for r in records]
        vocab = build_vocab(all_texts)
        vocab_path = Path(args.json_path).parent / "bm25_vocab.json"
        with open(vocab_path, "w") as vf:
            json.dump(vocab, vf)
        print(f"[indexer] BM25 vocab rebuilt: {len(vocab)} tokens → {vocab_path}")
        print("[indexer] NOTE: You must also re-run the full indexer to update Qdrant "
              "sparse vectors with the new vocabulary IDs.")
    else:
        LegalIndexer(args.json_path).run()