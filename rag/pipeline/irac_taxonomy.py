"""
pipeline/irac_taxonomy.py
Mapping between query intent labels and the dataset's own conclusion_type
vocabulary.

The bug this replaces
---------------------
pipeline/irac_reranker.py scored the IRAC "conclusion" component with:

    if lbl == "punitive"    and chunk.conclusion_type == "punitive":     c = 1.0
    elif lbl == "definition" and chunk.conclusion_type == "definitional": c = 1.0
    elif lbl == "procedural" and chunk.conclusion_type == "procedural":   c = 1.0

Every one of those branches was unreachable. The dataset writes
conclusion_type capitalized and free-form — "Penal" (685 sections),
"Procedural Order" (418), "Procedural" (274), "Definitional" (128) — and
the string "punitive" as a conclusion_type barely occurs at all (23 of
3394). So a case-sensitive equality test against lowercase literals never
matched, `conc_score` was a constant, and main.py's "override intent to
punitive so the +1.0 conclusion bonus fires" logic was steering toward a
bonus that did not exist.

Why this is a module
--------------------
Two reasons the original failed silently and this shouldn't:

  * conclusion_type is a long-tailed free-text field (120+ distinct
    values, most occurring once). Exact equality against a hand-picked
    literal is the wrong tool; marker-substring families degrade
    gracefully as the dataset grows new variants.
  * families_are_populated() lets a test assert that every family still
    matches real corpus values, so this can never quietly die again.
    See tests/test_irac_taxonomy.py.
"""
from __future__ import annotations

# Marker substrings, matched against a lowercased conclusion_type.
# Deliberately conservative: a family should capture the clear cases and
# let everything else fall through to partial credit, rather than claim
# territory it isn't sure about.
INTENT_CONCLUSION_MARKERS: dict[str, tuple[str, ...]] = {
    # Sections whose conclusion IS a punishment or criminal outcome.
    "punitive":   ("penal", "punitive", "conviction", "sentenc", "acquittal"),
    # Sections that define, interpret, or delimit scope.
    "definition": ("definition", "definitional", "interpretive", "applicab"),
    # Sections whose conclusion is a step, order, or process.
    "procedural": ("procedural", "order", "appellate", "revisional",
                   "discharge", "pre-trial", "formality"),
    # "statute" and "case_law" have no distinguishing conclusion_type
    # signal in this corpus, so they intentionally get no markers and rely
    # on the partial-credit tier below.
}

# Score tiers for the IRAC conclusion component.
CONCLUSION_MATCH        = 1.0   # conclusion_type is in the intent's family
CONCLUSION_HAS_TYPE     = 0.5   # section is typed, just not this family
CONCLUSION_UNTYPED      = 0.3   # section carries no conclusion_type at all


def normalize_conclusion_type(value: str | None) -> str:
    """One canonical spelling. data/indexer.py already lowercases this into
    the payload; doing it again here keeps the function correct for records
    that came from the raw JSON or an older index."""
    return (value or "").strip().lower()


def conclusion_score(intent_labels: list[str], conclusion_type: str | None) -> float:
    """Best conclusion score across every active intent label.

    Uses all labels rather than only the primary one so a chunk that
    perfectly matches a secondary intent isn't scored as if it matched
    nothing.
    """
    ctype = normalize_conclusion_type(conclusion_type)
    if not ctype:
        return CONCLUSION_UNTYPED

    for label in intent_labels or []:
        for marker in INTENT_CONCLUSION_MARKERS.get(label, ()):
            if marker in ctype:
                return CONCLUSION_MATCH
    return CONCLUSION_HAS_TYPE


def families_are_populated(corpus_conclusion_types: list[str]) -> dict[str, int]:
    """How many corpus sections each family actually matches.

    A family that matches zero sections is a dead branch — exactly the
    failure this module exists to prevent — so tests assert every count is
    non-zero rather than trusting the literals to stay correct.
    """
    normalized = [normalize_conclusion_type(c) for c in corpus_conclusion_types]
    counts: dict[str, int] = {}
    for label, markers in INTENT_CONCLUSION_MARKERS.items():
        counts[label] = sum(
            1 for c in normalized if c and any(m in c for m in markers)
        )
    return counts
