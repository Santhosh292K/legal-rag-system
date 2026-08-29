"""
evaluation/evaluate_cases.py

evaluation/benchmark_cases.json (8 multi-document FIR + medical-report
reasoning scenarios) was never wired into evaluate.py — only the
single-query benchmark_scenarios.json is evaluated there. This script
evaluates the case-file scenarios through the actual path a real case
would take: Track B (document ingestion -> chunking -> entity
extraction) fused with Track A (statute retrieval) via
pipeline/fusion.py's CaseStatuteFusion, exactly as legal_rag_system.py
wires them for a live case.

Metrics reported, per case and averaged:
  - Recall/Precision@K of evidence_sections ∪ statute citations vs gold_sections
  - Generation: ROUGE-L, Token-F1, Semantic Similarity vs gold_answer
  - Answer Coverage Score vs gold_answer's implied gold_sections
  - ALEA band accuracy: does ALEA's Strong/Partial/Weak/Missing band per
    section match expected_alea_bands in the benchmark file?

Usage:
    python3 evaluation/evaluate_cases.py
    python3 evaluation/evaluate_cases.py --save results_cases.json

Caveat: this exercises the full stack (SentenceTransformer embedding
model + Qdrant case store), so it needs the same environment evaluate.py
needs. It has not been run end-to-end in this review — verify the
CaseIndexer / DocumentIngestionPipeline call signatures still match your
checked-out pipeline/fusion.py before trusting the numbers.
"""
import json
import sys
import time
import statistics
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from main import LegalRAGPipeline
from pipeline.fusion import CaseStatuteFusion
from pipeline.adaptive_chunkers import get_chunker
from pipeline.alea import ALEA
from data.case_indexer import CaseIndexer
from config import QDRANT_PATH

from evaluation.evaluate import (
    recall_at_k, precision_at_k, rouge_l, token_f1,
)
from evaluation.legal_metrics import answer_coverage_score, semantic_similarity


def _ingest_case(indexer: CaseIndexer, case: dict) -> None:
    """Chunk + entity-extract + index a case's documents directly from the
    benchmark's inline text (no OCR needed — benchmark_cases.json already
    stores plain text, unlike a real uploaded PDF)."""
    from pipeline.entity_timeline_extractor import extract_entities
    import dataclasses

    case_id = case["case_id"]
    all_chunks = []
    for doc in case["case_documents"]:
        entities = extract_entities(doc["text"])
        chunker = get_chunker(doc["doc_type"])
        chunks = chunker.chunk(
            doc["text"], document_id=doc["document_id"], case_id=case_id,
        )
        # Attach entities to every chunk's metadata — mirrors
        # document_pipeline.py's behaviour (fusion.py's
        # _facts_from_case_chunks expects metadata.entities on each chunk).
        entities_dict = dataclasses.asdict(entities)
        for c in chunks:
            c.metadata = {**c.metadata, "entities": entities_dict}
        all_chunks.extend(chunks)

    indexer.index_chunks(all_chunks)


def _alea_band_accuracy(alea_scores, expected_bands: dict) -> float:
    if not expected_bands:
        return 0.0
    got = {s.section_id: s.band for s in alea_scores}
    correct = sum(1 for sid, band in expected_bands.items() if got.get(sid) == band)
    return correct / len(expected_bands)


def evaluate_cases(benchmark_path: str, save_path: str | None = None):
    with open(benchmark_path) as f:
        cases = json.load(f)

    # BUGFIX: this used to call LegalRAGPipeline() and CaseIndexer()
    # back to back with no shared client. Each one opens its own
    # QdrantClient(path=QDRANT_PATH) by default, and Qdrant's embedded/
    # local mode file-locks that folder to a single open client — so the
    # second call always crashed with "already accessed by another
    # instance", every time this script ran. Both classes already accept
    # a `client` param for exactly this (see their docstrings) — just
    # wasn't being used here. Build one client, share it.
    pipeline = LegalRAGPipeline(verbose=False)
    # BUGFIX: CaseIndexer used to load its own second copy of bge-large
    # even though pipeline.retriever already had one loaded — the same
    # double-loading pattern that caused the ablation study's CUDA OOM.
    # Reuse the pipeline's model instead of loading a fresh one.
    case_indexer = CaseIndexer(qdrant_path=QDRANT_PATH, client=pipeline.retriever.client,
                                embed_model=pipeline.retriever.embed_model)
    embed_fn = lambda texts: case_indexer.embed_model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False)
    alea = ALEA(embed_fn=embed_fn)
    fusion = CaseStatuteFusion(pipeline, case_indexer, alea=alea)

    r5_l, p5_l = [], []
    rl_l, tf1_l, sem_l, cov_l = [], [], [], []
    band_acc_l = []
    latency_l = []
    per_case_rows = []

    for case in cases:
        case_id = case["case_id"]
        gold_secs = case.get("gold_sections", [])
        gold_ans = case.get("gold_answer", "")
        expected_bands = case.get("expected_alea_bands", {})

        t0 = time.time()
        try:
            _ingest_case(case_indexer, case)
            fused = fusion.answer(case["query"], case_id=case_id)
        except Exception as e:
            print(f"  ✗ {case_id} failed: {e}")
            continue
        elapsed = time.time() - t0
        latency_l.append(elapsed)

        statute_ids = [c.section_id for c in fused.statute_answer.citations] if fused.statute_answer.citations else []
        retrieved_ids = list(dict.fromkeys(fused.evidence_sections + statute_ids))
        answer_text = fused.statute_answer.answer

        r5 = recall_at_k(retrieved_ids, gold_secs, 5)
        p5 = precision_at_k(retrieved_ids, gold_secs, 5)
        rl = rouge_l(answer_text, gold_ans) if gold_ans else 0.0
        tf1 = token_f1(answer_text, gold_ans) if gold_ans else 0.0
        sem = semantic_similarity(answer_text, gold_ans, embed_fn) if gold_ans else 0.0
        cov = answer_coverage_score(answer_text, gold_secs) if gold_secs else 0.0
        band_acc = _alea_band_accuracy(fused.alea_scores, expected_bands)

        r5_l.append(r5); p5_l.append(p5)
        rl_l.append(rl); tf1_l.append(tf1); sem_l.append(sem); cov_l.append(cov)
        band_acc_l.append(band_acc)

        per_case_rows.append(dict(
            case_id=case_id, recall_at_5=r5, precision_at_5=p5,
            rouge_l=rl, token_f1=tf1, semantic_similarity=sem,
            answer_coverage=cov, alea_band_accuracy=band_acc,
            latency=elapsed,
        ))

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    summary = dict(
        n_cases=len(cases),
        n_evaluated=len(per_case_rows),
        recall_at_5=avg(r5_l), precision_at_5=avg(p5_l),
        rouge_l=avg(rl_l), token_f1=avg(tf1_l),
        semantic_similarity=avg(sem_l), answer_coverage=avg(cov_l),
        alea_band_accuracy=avg(band_acc_l),
        avg_latency=avg(latency_l),
    )

    print("\n" + "═"*70)
    print("CASE-LEVEL EVALUATION (benchmark_cases.json)")
    print("─"*70)
    for k, v in summary.items():
        print(f"  {k:<22} {v}")
    print("═"*70)

    if save_path:
        with open(save_path, "w") as f:
            json.dump({"summary": summary, "per_case": per_case_rows}, f, indent=2)
        print(f"[evaluate_cases] Results saved -> {save_path}")

    return summary, per_case_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", nargs="?", default="./evaluation/benchmark_cases.json")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    evaluate_cases(args.benchmark, args.save)