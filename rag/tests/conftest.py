"""
tests/conftest.py
Shared fixtures. Everything here runs against the REAL dataset — these
tests exist to catch the class of bug where code and data silently
disagree, so testing against a synthetic fixture would defeat the point.
"""
import json
import sys
from pathlib import Path

import pytest

RAG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_ROOT))

DATASET_PATH = RAG_ROOT / "data" / "final_dataset.json"


@pytest.fixture(scope="session")
def records() -> list[dict]:
    if not DATASET_PATH.exists():
        pytest.skip(f"{DATASET_PATH} not present")
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def known_ids(records) -> set[str]:
    return {r["section"] for r in records}


@pytest.fixture(scope="session")
def store(records):
    from pipeline.section_store import SectionStore
    return SectionStore(json_path=str(DATASET_PATH), verbose=False)
