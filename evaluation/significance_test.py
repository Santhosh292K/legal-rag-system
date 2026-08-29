"""
evaluation/significance_test.py

evaluate.py's ablation_study() reports point-estimate averages only
(e.g. "Full Pipeline: NDCG@10 = 0.71" vs "No IRAC Reranker: 0.68") with
no indication of whether a 0.03 gap is a real effect or noise on a
51-query benchmark. This adds paired bootstrap significance testing
between any two saved result files.

Requires evaluate.py's `raw_scores` field on EvalResult (added alongside
this script) — per-query scores, not just the averages, are needed to
resample pairs correctly. Re-run evaluate.py with --save after picking
up that change if your existing saved JSONs predate it.

Usage:
    python3 evaluation/evaluate.py evaluation/benchmark_scenarios.json \
        --save results_full.json
    # ... run again with a different pipeline config, save as results_ablation.json ...
    python3 evaluation/significance_test.py results_full.json results_ablation.json \
        --metric ndcg_at_10

    # Or compare two systems within a single --ablation --save output:
    python3 evaluation/significance_test.py ablation_results.json ablation_results.json \
        --metric mrr --system-a "A: Full Pipeline" --system-b "B: No IRAC Reranker"
"""
import argparse
import json
import random
import statistics


def _load_system(path: str, system_name: str | None):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if system_name is None:
        if len(data) != 1:
            names = [d.get("system_name", "?") for d in data]
            raise ValueError(
                f"{path} contains multiple systems ({names}); pass "
                f"--system-a/--system-b to pick one."
            )
        return data[0]
    for d in data:
        if d.get("system_name") == system_name:
            return d
    raise ValueError(f"System '{system_name}' not found in {path}")


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10000, seed: int = 42):
    """Paired bootstrap test for whether mean(a) != mean(b).
    Returns (mean_diff, p_two_sided, ci_lo, ci_hi) for a - b."""
    if len(a) != len(b):
        raise ValueError(
            f"Paired test needs equal-length, aligned per-query score lists "
            f"(got {len(a)} vs {len(b)}) — the two systems must have been "
            f"evaluated on the same benchmark file in the same order."
        )
    n = len(a)
    if n < 2:
        raise ValueError("Need at least 2 queries to bootstrap.")

    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    observed = statistics.mean(diffs)

    boot_means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.mean(sample))
    boot_means.sort()

    # Two-sided p-value: fraction of bootstrap means on the opposite side
    # of zero from the observed effect, doubled.
    if observed >= 0:
        p = 2 * (sum(1 for m in boot_means if m <= 0) / n_boot)
    else:
        p = 2 * (sum(1 for m in boot_means if m >= 0) / n_boot)
    p = min(p, 1.0)

    ci_lo = boot_means[int(0.025 * n_boot)]
    ci_hi = boot_means[int(0.975 * n_boot)]
    return observed, p, ci_lo, ci_hi


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap significance test between two saved eval results")
    parser.add_argument("file_a")
    parser.add_argument("file_b")
    parser.add_argument("--metric", required=True,
                         help="Key into raw_scores, e.g. recall_at_5, ndcg_at_10, mrr, rouge_l, "
                              "token_f1, semantic_similarity, answer_coverage, citation_f1")
    parser.add_argument("--system-a", default=None, help="system_name to pick from file_a if it holds multiple")
    parser.add_argument("--system-b", default=None, help="system_name to pick from file_b if it holds multiple")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    sys_a = _load_system(args.file_a, args.system_a)
    sys_b = _load_system(args.file_b, args.system_b)

    raw_a = sys_a.get("raw_scores", {}).get(args.metric)
    raw_b = sys_b.get("raw_scores", {}).get(args.metric)
    if raw_a is None or raw_b is None:
        raise ValueError(
            f"Metric '{args.metric}' not found in raw_scores for one or both "
            f"systems. Available: {list(sys_a.get('raw_scores', {}).keys())}"
        )

    observed, p, ci_lo, ci_hi = paired_bootstrap(raw_a, raw_b, n_boot=args.n_boot)

    name_a = sys_a.get("system_name", args.file_a)
    name_b = sys_b.get("system_name", args.file_b)
    print(f"\nPaired bootstrap test — metric: {args.metric}")
    print(f"  {name_a}: mean = {statistics.mean(raw_a):.4f} (n={len(raw_a)})")
    print(f"  {name_b}: mean = {statistics.mean(raw_b):.4f} (n={len(raw_b)})")
    print(f"  Observed difference (A - B): {observed:.4f}")
    print(f"  95% CI of difference: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Two-sided p-value: {p:.4f}")
    if p < 0.05:
        print("  -> Significant at alpha=0.05")
    else:
        print("  -> NOT significant at alpha=0.05 — treat the point-estimate "
              "gap in evaluate.py's tables with caution.")


if __name__ == "__main__":
    main()