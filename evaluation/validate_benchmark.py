"""
evaluation/validate_benchmark.py

Doesn't fabricate new benchmark items — I'm not going to invent Indian
legal scenarios and gold sections/punishments myself; getting a
section number or punishment range wrong in a "gold" label is worse
than having fewer labeled items, since it silently corrupts every
metric computed against it. What this script does instead:

  1. Validates that new items you (or a subject-matter reviewer) add to
     benchmark_scenarios.json or benchmark_cases.json are well-formed
     BEFORE they can corrupt the eval: required fields present,
     gold_sections actually exist in data/final_dataset.json (catches
     typos like "IPC_302A" that would silently score as an
     unretrievable gold section forever), category is non-empty,
     no duplicate case_id/query.

  2. Reports current category balance against a minimum-count target,
     so you know exactly which categories need more real-world-reviewed
     items before their domain_breakdown numbers in evaluate.py's
     Table 6 are trustworthy (categories under the target print with a
     count of how many more are needed).

Usage:
    python3 evaluation/validate_benchmark.py
    python3 evaluation/validate_benchmark.py --min-per-category 5
    python3 evaluation/validate_benchmark.py evaluation/benchmark_cases.json --cases
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

REQUIRED_SCENARIO_FIELDS = {"query", "gold_sections", "gold_answer", "category"}
REQUIRED_CASE_FIELDS = {"case_id", "description", "query", "case_documents",
                         "gold_sections", "gold_answer", "category"}


def load_valid_section_ids(dataset_path: str = "./data/final_dataset.json") -> set[str]:
    with open(dataset_path) as f:
        dataset = json.load(f)
    return {item["section"] for item in dataset if "section" in item}


def validate_scenarios(items: list[dict], valid_sections: set[str]) -> list[str]:
    errors = []
    seen_queries = set()

    for i, item in enumerate(items):
        tag = f"item[{i}] ({item.get('query', '?')[:40]!r})"

        missing = REQUIRED_SCENARIO_FIELDS - item.keys()
        if missing:
            errors.append(f"{tag}: missing required fields {missing}")
            continue

        if not item["query"].strip():
            errors.append(f"{tag}: empty query")
        if item["query"].strip().lower() in seen_queries:
            errors.append(f"{tag}: duplicate query")
        seen_queries.add(item["query"].strip().lower())

        if not item["gold_sections"]:
            is_intentional_gap = item.get("category", "").endswith("_gap")
            if not is_intentional_gap:
                errors.append(f"{tag}: empty gold_sections — item will contribute 0 to every retrieval metric "
                               f"(if this is intentional, e.g. testing a not-yet-indexed act, name the "
                               f"category '..._gap' so this check skips it)")
        else:
            unknown = [s for s in item["gold_sections"] if s not in valid_sections]
            if unknown:
                errors.append(f"{tag}: gold_sections not found in dataset (typo?): {unknown}")

        if not item["gold_answer"].strip():
            errors.append(f"{tag}: empty gold_answer — item will contribute 0 to every generation metric")

        if not item["category"].strip():
            errors.append(f"{tag}: empty category")

    return errors


def validate_cases(items: list[dict], valid_sections: set[str]) -> list[str]:
    errors = []
    seen_ids = set()

    for i, item in enumerate(items):
        tag = f"item[{i}] ({item.get('case_id', '?')})"

        missing = REQUIRED_CASE_FIELDS - item.keys()
        if missing:
            errors.append(f"{tag}: missing required fields {missing}")
            continue

        if item["case_id"] in seen_ids:
            errors.append(f"{tag}: duplicate case_id")
        seen_ids.add(item["case_id"])

        if not item["case_documents"]:
            errors.append(f"{tag}: no case_documents")
        else:
            for doc in item["case_documents"]:
                if not doc.get("text", "").strip():
                    errors.append(f"{tag}: document {doc.get('document_id','?')} has empty text")

        unknown = [s for s in item["gold_sections"] if s not in valid_sections]
        if unknown:
            errors.append(f"{tag}: gold_sections not found in dataset (typo?): {unknown}")

        for sid, band in item.get("expected_alea_bands", {}).items():
            if band not in ("Strong", "Partial", "Weak", "Missing"):
                errors.append(f"{tag}: expected_alea_bands[{sid}]='{band}' not a valid ALEA band")

    return errors


def report_category_balance(items: list[dict], min_per_category: int):
    counts = Counter(item.get("category", "unknown") for item in items)
    print(f"\n{'Category':<22} {'Count':>6}  {'Status'}")
    print("-" * 50)
    under_target = []
    for cat, n in sorted(counts.items(), key=lambda kv: kv[1]):
        status = "OK" if n >= min_per_category else f"needs {min_per_category - n} more"
        if n < min_per_category:
            under_target.append(cat)
        print(f"{cat:<22} {n:>6}  {status}")
    print("-" * 50)
    if under_target:
        print(f"{len(under_target)} / {len(counts)} categories below target "
              f"(min {min_per_category}/category) — domain_breakdown numbers "
              f"for these are high-variance until they have more items.")
    else:
        print("All categories meet the minimum count target.")


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark items and report category balance")
    parser.add_argument("benchmark", nargs="?", default="./evaluation/benchmark_scenarios.json")
    parser.add_argument("--cases", action="store_true", help="Validate as benchmark_cases.json format instead")
    parser.add_argument("--dataset", default="./data/final_dataset.json")
    parser.add_argument("--min-per-category", type=int, default=5)
    args = parser.parse_args()

    valid_sections = load_valid_section_ids(args.dataset)
    with open(args.benchmark) as f:
        items = json.load(f)

    validate_fn = validate_cases if args.cases else validate_scenarios
    errors = validate_fn(items, valid_sections)

    print(f"[validate_benchmark] {args.benchmark}: {len(items)} items, "
          f"{len(valid_sections)} known section ids in {args.dataset}\n")

    if errors:
        print(f"{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("No errors — all items well-formed and gold_sections resolve to real dataset entries.")

    report_category_balance(items, args.min_per_category)

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()