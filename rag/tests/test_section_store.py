"""The in-memory corpus index that replaced per-lookup Qdrant scrolls."""
import pytest


def test_loads_every_section(store, records):
    assert len(store) == len(records)


def test_lookup_by_id(store):
    payload = store.get("IPC_302")
    assert payload is not None
    assert payload["act_code"] == "IPC"
    assert payload["section_number"] == "302"


def test_lookup_by_act_and_number(store):
    assert store.lookup_act_number("IPC", "302") == "IPC_302"
    assert store.lookup_act_number("IPC", "29a") == "IPC_029A"
    assert store.lookup_act_number("CRPC", "41") == "CRPC_41"     # unpadded act
    assert store.lookup_act_number("IPC", "99999") is None


def test_resolve_loose_reference(store):
    assert store.resolve("IPC 2") == "IPC_002"
    assert store.resolve("ipc 304a") == "IPC_304A"
    assert store.resolve("IPC 9999") is None


def test_related_sections_are_canonical_ids(store):
    """The dataset writes these as 'IPC 2'. They are normalized at index
    time so downstream exact-id lookups actually hit."""
    dangling = [
        rel
        for sid in store.section_ids
        for rel in store.get(sid)["related_sections"]
        if rel not in store
    ]
    assert dangling == []


def test_conclusion_type_is_normalised(store):
    """Lowercased at index time so the reranker's taxonomy can match it."""
    for sid in list(store.section_ids)[:500]:
        value = store.get(sid)["conclusion_type"]
        assert value == value.lower()


def test_record_view_shape(store):
    record = store.get_record("IPC_302")
    assert record["section"] == "IPC_302"
    assert record["meta"]["code"] == "IPC"
    assert "temporal" in record["meta"]
    assert record["meta"]["temporal"]["enacted_year"]
    assert "rule_summary" in record["meta"]["irac"]


def test_missing_section_returns_none(store):
    assert store.get("NOPE_001") is None
    assert store.get_record("NOPE_001") is None
