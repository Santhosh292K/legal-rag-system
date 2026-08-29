"""
evaluation/split_benchmark.py

Fixes a leakage problem: tune_pin_threshold.py sweeps MIN_PIN_SIMILARITY
against evaluation/benchmark_scenarios.json, and evaluate.py reports
final numbers against the SAME file. Any threshold picked by the sweep
is fit to the exact queries you then report Recall/Precision on, so
those numbers are optimistic — not a fair estimate of performance on
unseen queries.

This script creates a stratified (by `category`) dev/test split with a
fixed seed, so:
  - tune_pin_threshold.py should be pointed at benchmark_scenarios_dev.json
  - evaluate.py should report final numbers on benchmark_scenarios_test.json

Usage:
    python3 evaluation/split_benchmark.py
    python3 evaluation/split_benchmark.py --test-frac 0.4 --seed 7

Note: with only 51 queries across 13 categories (several with n=1-2), a
stratified split will leave some categories entirely in one file — that's
an unavoidable consequence of the benchmark's current size, not a bug in
this script. Consider growing the benchmark (more items per category)
before treating per-category test numbers as reliable.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def stratified_split(items: list[dict], test_frac: float, seed: int):
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it.get("category", "unknown")].append(it)

    rng = random.Random(seed)
    dev, test = [], []
    for cat, cat_items in by_cat.items():
        cat_items = cat_items[:]
        rng.shuffle(cat_items)
        n_test = round(len(cat_items) * test_frac)
        # Guarantee at least one item stays in dev when there's more than
        # one item, so the pinner-threshold sweep always has something to
        # tune against for categories that appear in the sweep's queries.
        if len(cat_items) > 1:
            n_test = min(n_test, len(cat_items) - 1)
        test.extend(cat_items[:n_test])
        dev.extend(cat_items[n_test:])
    return dev, test


def main():
    parser = argparse.ArgumentParser(description="Split benchmark into dev/test to avoid tune/eval leakage")
    parser.add_argument("input", nargs="?", default="./evaluation/benchmark_scenarios.json")
    parser.add_argument("--test-frac", type=float, default=0.35,
                         help="Fraction of each category held out for test (default 0.35)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    with open(in_path) as f:
        items = json.load(f)

    dev, test = stratified_split(items, args.test_frac, args.seed)

    dev_path = in_path.with_name(in_path.stem + "_dev.json")
    test_path = in_path.with_name(in_path.stem + "_test.json")

    with open(dev_path, "w") as f:
        json.dump(dev, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test, f, indent=2)

    print(f"[split_benchmark] {len(items)} total items")
    print(f"  dev  ({len(dev)} items)  -> {dev_path}   (use for tune_pin_threshold.py)")
    print(f"  test ({len(test)} items) -> {test_path}  (use for evaluate.py final numbers)")

    by_cat_dev = defaultdict(int)
    by_cat_test = defaultdict(int)
    for it in dev:
        by_cat_dev[it.get("category", "unknown")] += 1
    for it in test:
        by_cat_test[it.get("category", "unknown")] += 1

    print(f"\n  {'category':<20} {'dev':>5} {'test':>5}")
    for cat in sorted(set(by_cat_dev) | set(by_cat_test)):
        print(f"  {cat:<20} {by_cat_dev.get(cat,0):>5} {by_cat_test.get(cat,0):>5}")


if __name__ == "__main__":
    main()