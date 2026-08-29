"""
data/csv_to_json.py
Convert enriched CSV dataset → structured JSON
"""
import csv
import json
import sys
from pathlib import Path


def _null(val: str):
    if not val or val.strip().upper() in ("NULL", "NONE", "N/A", ""):
        return None
    return val.strip()


def _split(val: str):
    if not val or val.strip().upper() in ("NULL", "NONE", "N/A", ""):
        return []
    return [v.strip() for v in val.split(",") if v.strip()]


def _int(val: str):
    try:
        return int(val.strip())
    except Exception:
        return None


def build_embedding_text(row: dict) -> str:
    """
    Concatenate key fields into a single rich string for BGE embedding.
    Ordering matters: most important fields first.
    """
    parts = [
        row.get("act_name", ""),
        f"Section {row.get('section_number', '')}",
        row.get("chapter", ""),
        row.get("category", ""),
        row.get("keywords", ""),
        row.get("rule_summary", ""),
        row.get("issue_tags", ""),
        row.get("content", ""),
    ]
    return " ".join(p for p in parts if p and p.upper() not in ("NULL", "NONE", "N/A"))


def row_to_record(row: dict) -> dict:
    embedding_text = row.get("embedding_text", "").strip()
    if not embedding_text or embedding_text.upper() in ("NULL", "NONE"):
        embedding_text = build_embedding_text(row)

    return {
        "section":        row["section_id"],
        "content":        row["content"],
        "embedding_text": embedding_text,
        "meta": {
            "code":     row["act_code"],
            "section":  row["section_number"],
            "chapter":  _null(row.get("chapter", "")),
            "category": row.get("category", ""),
            "keywords": _split(row.get("keywords", "")),
            "hierarchy": {
                "act":            row.get("act_name", ""),
                "part":           _null(row.get("part", "")),
                "chapter":        _null(row.get("chapter", "")),
                "section":        row["section_number"],
                "sub_section":    _null(row.get("sub_section", "")),
                "proviso":        _null(row.get("proviso", "")),
                "parent_section": _null(row.get("parent_section", "")),
                "child_sections": _split(row.get("child_sections", "")),
            },
            "temporal": {
                "enacted_year":  _int(row.get("enacted_year", "")),
                "effective_date": _null(row.get("effective_date", "")),
                "last_amended":   _null(row.get("last_amended", "")),
                "amended_by":     _split(row.get("amended_by", "")),
                "status":         _null(row.get("status", "")) or "active",
                "supersedes":     _null(row.get("supersedes", "")),
                "superseded_by":  _null(row.get("superseded_by", "")),
            },
            "legal_type": {
                "intent":             _null(row.get("intent", "")),
                "rule_type":          _null(row.get("rule_type", "")),
                "applies_to":         _split(row.get("applies_to", "")),
                "offense_type":       _null(row.get("offense_type", "")),
                "punishment_type":    _null(row.get("punishment_type", "")),
                "punishment_severity":_null(row.get("punishment_severity", "")),
                "jurisdiction":       _null(row.get("jurisdiction", "")) or "India",
                "exceptions":         _split(row.get("exceptions", "")),
            },
            "irac": {
                "issue_tags":         _split(row.get("issue_tags", "")),
                "rule_summary":       _null(row.get("rule_summary", "")),
                "application_context":_split(row.get("application_context", "")),
                "conclusion_type":    _null(row.get("conclusion_type", "")),
            },
            "related_sections": _split(row.get("related_sections", "")),
        },
    }


def convert(csv_path: str, json_path: str):
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    records = []
    errors  = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                records.append(row_to_record(row))
            except Exception as e:
                errors.append({"row": i, "error": str(e)})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Converted : {len(records)} records → {json_path}")
    if errors:
        print(f"Errors    : {len(errors)}")
        for e in errors[:5]:
            print(f"  Row {e['row']}: {e['error']}")


if __name__ == "__main__":
    csv_in  = sys.argv[1] if len(sys.argv) > 1 else "./data/dataset.csv"
    json_out = sys.argv[2] if len(sys.argv) > 2 else "./data/final_dataset.json"
    convert(csv_in, json_out)
