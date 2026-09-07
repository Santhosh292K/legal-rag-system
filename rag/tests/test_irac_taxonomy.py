"""Conclusion-type families must match the dataset's real vocabulary.

Every 1.0-scoring branch in the old reranker was unreachable: it compared
lowercase literals ("punitive", "definitional", "procedural") against a
field the dataset writes as "Penal", "Definitional", "Procedural Order".
These tests fail if that ever becomes true again.
"""
import pytest

from pipeline.irac_taxonomy import (
    conclusion_score, families_are_populated, normalize_conclusion_type,
    INTENT_CONCLUSION_MARKERS,
    CONCLUSION_MATCH, CONCLUSION_HAS_TYPE, CONCLUSION_UNTYPED,
)


def test_every_family_matches_real_corpus_values(records):
    """A family matching zero sections is a dead branch."""
    types = [r["meta"]["irac"]["conclusion_type"] or "" for r in records]
    counts = families_are_populated(types)
    for label, n in counts.items():
        assert n > 0, f"intent family {label!r} matches no section in the corpus"
    assert counts["punitive"] > 100
    assert counts["procedural"] > 100
    assert counts["definition"] > 50


def test_dataset_capitalisation_still_scores(records):
    """The exact spellings that used to break it."""
    assert conclusion_score(["punitive"], "Penal") == CONCLUSION_MATCH
    assert conclusion_score(["procedural"], "Procedural Order") == CONCLUSION_MATCH
    assert conclusion_score(["definition"], "Definitional") == CONCLUSION_MATCH


def test_partial_and_missing_tiers():
    assert conclusion_score(["punitive"], "Procedural") == CONCLUSION_HAS_TYPE
    assert conclusion_score(["punitive"], "") == CONCLUSION_UNTYPED
    assert conclusion_score([], "Penal") == CONCLUSION_HAS_TYPE


def test_secondary_labels_are_considered():
    """A chunk matching a secondary intent must not be scored as a miss."""
    assert conclusion_score(["statute", "punitive"], "Penal") == CONCLUSION_MATCH


def test_normalisation_is_case_and_space_insensitive():
    assert normalize_conclusion_type("  Procedural Order  ") == "procedural order"
    assert normalize_conclusion_type(None) == ""


def test_markers_are_lowercase():
    """Markers are matched against a lowercased value; an uppercase marker
    would silently never fire — the original bug, one level down."""
    for label, markers in INTENT_CONCLUSION_MARKERS.items():
        for m in markers:
            assert m == m.lower(), f"{label}: marker {m!r} is not lowercase"
