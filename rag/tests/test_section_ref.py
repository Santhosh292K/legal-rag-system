"""Canonical section-reference parsing — the logic five modules used to
each implement slightly differently."""
import pytest

from data.section_ref import (
    canonical_act, candidates, normalize_section_ref,
    extract_section_refs, find_section_spans,
)


@pytest.mark.parametrize("written,expected", [
    ("ipc", "IPC"), ("IPC", "IPC"), ("Indian Penal Code", "IPC"),
    ("it act", "ITA"), ("IT Act", "ITA"), ("evidence act", "IEA"),
    ("cr.p.c", "CRPC"), ("constitution", "COI"), ("bnss", "BNSS"),
    ("not an act", None),
])
def test_canonical_act(written, expected):
    assert canonical_act(written) == expected


@pytest.mark.parametrize("act,num,expected", [
    ("IPC", "2",    "IPC_002"),
    ("IPC", "29a",  "IPC_029A"),   # zfill must pad the DIGITS, not the string
    ("IPC", "120b", "IPC_120B"),
    ("BNS", "74",   "BNS_074"),
    ("CRPC", "41",  "CRPC_41"),    # CRPC is the unpadded exception
    ("ITA", "66c",  "ITA_066C"),
])
def test_padding_matches_the_dataset(act, num, expected, known_ids):
    resolved = next((c for c in candidates(act, num) if c in known_ids), None)
    assert resolved == expected
    assert expected in known_ids


def test_cross_references_resolve(records, known_ids):
    """The dataset writes these as 'IPC 2' while ids are 'IPC_002'. Only
    228 of 5339 matched verbatim, which is what left legal_kg with 2414
    phantom nodes and enrich_chunks with empty related_contents."""
    refs = [ref for r in records for ref in r["meta"]["related_sections"]]
    resolved = [normalize_section_ref(ref, known_ids) for ref in refs]
    hit_rate = sum(1 for r in resolved if r) / len(refs)
    assert hit_rate > 0.85, f"only {hit_rate:.1%} of cross-references resolve"


def test_unknown_reference_returns_none(known_ids):
    assert normalize_section_ref("IPC 9999", known_ids) is None
    assert normalize_section_ref("XYZ 1", known_ids) is None


@pytest.mark.parametrize("text,expected", [
    ("What does IPC 302 say?",                    ["IPC_302"]),
    ("punishment under section 302 of the IPC",   ["IPC_302"]),
    ("u/s 420 IPC",                               ["IPC_420"]),
    ("IPC 494 and 495",                           ["IPC_494", "IPC_495"]),
    ("Sections invoked: 103, 118 BNS",            ["BNS_103", "BNS_118"]),
    ("charged under sections 302, 34 IPC",        ["IPC_302", "IPC_034"]),
    ("Section 66C of the IT Act",                 ["ITA_066C"]),
    ("bnss 187 and crpc 167",                     ["BNSS_187", "CRPC_167"]),
    ("Article 21 of the Constitution",            ["COI_021"]),
    ("difference between IPC 302 and BNS 103",    ["IPC_302", "BNS_103"]),
])
def test_extraction_both_word_orders(text, expected, known_ids):
    assert extract_section_refs(text, known_ids) == expected


@pytest.mark.parametrize("text", [
    "what is the punishment for murder",
    "he was 21 years old and 30 feet away",
    "my landlord evicted me without notice",
])
def test_extraction_has_no_false_positives(text, known_ids):
    assert extract_section_refs(text, known_ids) == []


def _rewrite(text, known_ids):
    out, cursor = [], 0
    for start, end, ids in find_section_spans(text, known_ids):
        out.append(text[cursor:start])
        out.append("".join(f"[{sid}]" for sid in ids))
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def test_prose_is_not_mangled_into_citations(known_ids):
    """The old normalizer accepted any lowercase word as an act name, so
    'Section 302 defines murder' became '[DEFINES_302]'."""
    text = "Section 302 defines murder as an offence."
    assert _rewrite(text, known_ids) == text


def test_multi_word_act_names_are_recognised(known_ids):
    out = _rewrite("See Section 302 of the Indian Penal Code for details.", known_ids)
    assert "[IPC_302]" in out
