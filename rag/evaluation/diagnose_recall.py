"""
evaluation/diagnose_recall.py

Answers the question "is recall low because retrieval never finds the gold
section, or because it finds it and then loses it during reranking?" — the
printed evaluation tables can't distinguish these, but they need very
different fixes (better retrieval/query rewriting vs. reranker tuning).

For every benchmark query, this runs the full pipeline once with
debug_trace=True and records, at each stage boundary:
  raw_pool     — Stage 3 hybrid retrieval + pinned sections (~35-50 chunks)
  post_rerank  — Stage 6 IRAC reranking (~15 chunks)
  post_rocchio — Stage 6.5 pseudo-relevance feedback
  post_kg      — Stage 6.75 knowledge-graph augmentation
  final        — what evaluate.py actually measures recall against

whether each gold section is present at that stage. This tells you exactly
where a miss happens:
  - Missing from raw_pool           -> retrieval problem (BM25/dense/query
                                        rewriting aren't surfacing it at all)
  - In raw_pool, missing by final   -> reranking/truncation problem (it was
                                        found, then thrown away)
  - In raw_pool, recovered by       -> that stage is doing its job — good
    post_kg/post_rocchio              signal that the fix should lean harder
                                       on that mechanism, not the retriever

Usage:
    python3 evaluation/diagnose_recall.py
    python3 evaluation/diagnose_recall.py evaluation/benchmark_scenarios_test.json
    python3 evaluation/diagnose_recall.py --max 10
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from main import LegalRAGPipeline

STAGES = ["raw_pool", "post_rerank", "post_rocchio", "post_kg", "final"]


def stage_hits(trace: dict, gold: set[str]) -> dict[str, set[str]]:
    """For each stage, which gold sections are present in that stage's list."""
    return {stage: gold & set(trace.get(stage, [])) for stage in STAGES}


def first_stage_found(hits: dict[str, set[str]], sid: str) -> str | None:
    for stage in STAGES:
        if sid in hits[stage]:
            return stage
    return None


def dropped_stage(trace: dict, hits: dict[str, set[str]], sid: str) -> str | None:
    """
    If sid appears in some stage but is NOT present in 'final', find the last
    stage it was seen at before disappearing — i.e. where it got truncated /
    filtered out. Returns None if it survives to final, or if it was never
    found at all (that's a retrieval miss, not a truncation drop).
    """
    if sid in hits["final"]:
        return None
    last_seen = None
    for stage in STAGES[:-1]:   # everything before 'final'
        if sid in hits[stage]:
            last_seen = stage
    return last_seen   # None if it was never found anywhere


def main():
    parser = argparse.ArgumentParser(description="Stage-wise recall diagnostic")
    parser.add_argument("benchmark", nargs="?",
                         default="./evaluation/benchmark_scenarios_test.json")
    parser.add_argument("--max", type=int, default=None,
                         help="only run the first N queries (for a quick check)")
    args = parser.parse_args()

    with open(args.benchmark, "r") as f:
        benchmark = json.load(f)
    if args.max:
        benchmark = benchmark[: args.max]

    print(f"[diagnose_recall] loading pipeline...", flush=True)
    pipeline = LegalRAGPipeline(verbose=False)
    print(f"[diagnose_recall] pipeline ready — running {len(benchmark)} queries...", flush=True)

    # never_found[section_id] -> how many queries needed it but it never
    # appeared at any stage, across the whole run
    never_found_counter: Counter = Counter()
    truncated_counter: Counter = Counter()   # found somewhere, but not in 'final'
    # first_found_at[stage] -> count of (query, gold_section) pairs whose
    # first appearance was at this stage
    first_found_at: Counter = Counter()
    n_never_found = 0
    n_truncated = 0
    n_survives = 0
    n_gold_total = 0
    rows = []

    for i, item in enumerate(benchmark, start=1):
        query = item["query"]
        gold = set(item.get("gold_sections", []))
        if not gold:
            continue

        t0 = time.time()
        trace: dict = {}
        try:
            pipeline.query(query, debug_trace=trace)
        except Exception as e:
            print(f"[{i}/{len(benchmark)}] FAILED ({time.time()-t0:.0f}s): {query[:60]!r} ({e})", flush=True)
            continue

        hits = stage_hits(trace, gold)
        row = {"query": query, "category": item.get("category", ""), "gold": sorted(gold), "per_section": {}}

        for sid in sorted(gold):
            n_gold_total += 1
            stage = first_stage_found(hits, sid)
            drop = dropped_stage(trace, hits, sid)
            if stage is None:
                status = "never_found"
                never_found_counter[sid] += 1
                n_never_found += 1
            elif drop is not None:
                status = f"truncated_after_{drop}"
                truncated_counter[sid] += 1
                n_truncated += 1
                first_found_at[stage] += 1
            else:
                status = "survives"
                n_survives += 1
                first_found_at[stage] += 1
            row["per_section"][sid] = status

        rows.append(row)
        n_survived_here = sum(1 for s in row["per_section"].values() if s == "survives")
        print(
            f"[{i}/{len(benchmark)}] ({time.time()-t0:.0f}s) "
            f"{n_survived_here}/{len(gold)} gold sections survived | {query[:70]}",
            flush=True,
        )

    # ── Per-query detail: only show queries with at least one miss ──────────
    print("\n" + "=" * 100)
    print("QUERIES WITH AT LEAST ONE GOLD SECTION MISSING FROM FINAL OUTPUT")
    print("=" * 100)
    for row in rows:
        misses = {sid: s for sid, s in row["per_section"].items() if s != "survives"}
        if not misses:
            continue
        print(f"\n[{row['category']}] {row['query'][:90]}")
        for sid, status in row["per_section"].items():
            if status == "never_found":
                tag = "MISS — never surfaced at any stage (retrieval problem)"
            elif status.startswith("truncated_after_"):
                stage = status.replace("truncated_after_", "")
                tag = f"MISS — found at '{stage}' but dropped before final (truncation problem)"
            else:
                tag = "OK — present in final"
            print(f"    {sid:<14} {tag}")

    # ── Aggregate: where gold sections get their first hit, and outcome ─────
    print("\n" + "=" * 100)
    print("AGGREGATE")
    print("=" * 100)
    print(f"{'Outcome':<40} {'Count':>8} {'% of all gold refs':>20}")
    print("-" * 70)
    pct = lambda c: 100.0 * c / n_gold_total if n_gold_total else 0.0
    print(f"{'survives to final':<40} {n_survives:>8} {pct(n_survives):>19.1f}%")
    print(f"{'found, but truncated before final':<40} {n_truncated:>8} {pct(n_truncated):>19.1f}%")
    print(f"{'never found at any stage':<40} {n_never_found:>8} {pct(n_never_found):>19.1f}%")
    print("-" * 70)
    print(f"{'total gold refs':<40} {n_gold_total:>8}")

    print(f"\nFirst-found breakdown (among the {n_survives + n_truncated} that were found at all):")
    for stage in STAGES[:-1]:
        c = first_found_at[stage]
        print(f"  {stage:<15} {c:>6}")

    # ── Interpretation hint ───────────────────────────────────────────────
    print("\n" + "-" * 100)
    if pct(n_never_found) > 30:
        print(f"-> {pct(n_never_found):.0f}% of gold sections are NEVER surfaced at any stage — this is a")
        print(f"   RETRIEVAL problem (BM25/dense/query-rewriting aren't finding them at all).")
        print(f"   Reranker/KG/Rocchio tuning won't fix these; look at query_expander.py,")
        print(f"   universal_translator.py's rewrite quality, and consider fine-tuning the")
        print(f"   embedding model on (scenario, gold_section) pairs.")
    elif pct(n_truncated) > 15:
        print(f"-> {pct(n_truncated):.0f}% of gold sections ARE being retrieved, but get truncated")
        print(f"   before final — this is a RERANKING/TOP-K problem, not a retrieval problem.")
        print(f"   Check RERANK_TOP_K / FINAL_TOP_K, and whether the IRAC reranker is scoring")
        print(f"   these correctly-retrieved-but-dropped sections too low.")
    else:
        print(f"-> Most gold sections that are found do survive to final. If recall is still")
        print(f"   low, the remaining gap is mostly retrieval coverage (never_found), not")
        print(f"   reranking — see the never_found breakdown below.")

    # ── Which sections are hardest to find/keep ────────────────────────────
    if never_found_counter:
        print("\n" + "=" * 100)
        print("SECTIONS NEVER SURFACED AT ALL (retrieval misses — check embedding_text / keywords)")
        print("=" * 100)
        for sid, count in never_found_counter.most_common(15):
            print(f"  {sid:<14} missed in {count} quer{'y' if count == 1 else 'ies'}")

    if truncated_counter:
        print("\n" + "=" * 100)
        print("SECTIONS FOUND BUT TRUNCATED BEFORE FINAL (reranking/top-k losses)")
        print("=" * 100)
        for sid, count in truncated_counter.most_common(15):
            print(f"  {sid:<14} truncated in {count} quer{'y' if count == 1 else 'ies'}")

    print()


if __name__ == "__main__":
    main()