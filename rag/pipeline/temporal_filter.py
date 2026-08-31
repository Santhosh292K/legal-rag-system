"""
pipeline/temporal_filter.py
Novel component #2 — Temporal Validity Filter

Filters and flags retrieved chunks based on:
  1. Amendment status (active / amended / repealed)
  2. Supersession chain (is there a newer version?)
  3. Query temporal intent (current law vs. historical)

Returns chunks with validity flags attached.
"""
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.hybrid_retriever import RetrievedChunk
from pipeline.intent_classifier import QueryIntent


# ── Validity result ───────────────────────────────────────────────────────────

@dataclass
class ValidatedChunk:
    chunk:           RetrievedChunk
    is_valid:        bool
    validity_label:  str          # active | amended | repealed | superseded
    warning:         str          # human-readable warning if not fully valid
    penalized_score: float        # rrf_score after validity penalty applied


# ── Penalty constants ─────────────────────────────────────────────────────────

PENALTY = {
    "active":     1.00,   # no penalty
    "amended":    0.70,   # penalise — may not reflect current law
    "repealed":   0.10,   # severe — almost certainly wrong
    "superseded": 0.50,   # replaced by another section
    "unknown":    0.80,   # metadata gap — slight penalty
    "future":     0.05,   # didn't exist yet at the query's own date — near-hard exclusion
}

WARNING_MSG = {
    "amended":    "This section has been amended. Verify current wording.",
    "repealed":   "This section has been repealed and is no longer in force.",
    "superseded": "This section has been superseded by a later provision.",
    "unknown":    "Amendment status is unknown. Verify independently.",
    "future":     "This provision had not been enacted yet at the time relevant "
                  "to your query — it could not have applied.",
}

# 2024-07-01: IPC, CrPC and the Indian Evidence Act were repealed and
# replaced by BNS, BNSS and BSA respectively. The dataset's own
# superseded_by field is populated on only a single record (the Joseph
# Shine/IPC_497 case) — the old acts were never reciprocally linked to
# their successors (see data/indexer.py's build_payload note on
# superseded_by) — so this chain is hardcoded from the (undisputed, dated)
# legislative fact rather than relying on per-record metadata that doesn't
# carry it. Used only to demote a predecessor act for an EXPLICIT
# "current law" query — see is_stale_law below.
ACT_SUCCESSOR = {
    "IPC":  "BNS",
    "CRPC": "BNSS",
    "IEA":  "BSA",
}


# ── Date-precise chronology ─────────────────────────────────────────────────
# BUGFIX: everything above (and the first pass at this file) compared only
# YEARS (chunk.enacted_year vs. cutoff_year). That's coincidentally correct
# for this dataset's IPC(1860)/CRPC(1973)/IEA(1872) vs. BNS/BNSS/BSA(all
# enacted 2023) split for any query anchored OUTSIDE 2024 — but BNS, BNSS
# and BSA didn't take effect until 2024-07-01 (final_dataset.json's own
# effective_date field, matching the actual notified date), a MID-YEAR
# changeover. A query naming a month in 2024 ("a theft in January 2024",
# "before 1 June 2024") carries exactly the information needed to resolve
# that correctly, but year-only comparison threw it away and treated all of
# 2024 as one ambiguous bucket — e.g. "January 2024" (unambiguously pre-BNS)
# would incorrectly let BNS back in. Compare actual dates when the query
# gives one; fall back to the year-only approximation only when it doesn't
# (a bare "2024" with no month genuinely IS ambiguous — see
# intent_classifier.py's _extract_cutoff).

def _parse_effective_date(s: "str | None") -> "date | None":
    """final_dataset.json's effective_date is written in two different
    formats depending on which import batch wrote it — ISO 'YYYY-MM-DD'
    for IPC/BNS/IEA/BSA, day-first 'DD-MM-YYYY' for CRPC/BNSS (verified
    against the raw dataset directly). Handles both; also accepts an
    already-ISO cutoff_date string built by intent_classifier.py. Returns
    None (never raises) on anything unparseable."""
    if not s:
        return None
    parts = re.split(r"[-/]", s.strip())
    if len(parts) != 3:
        return None
    try:
        if len(parts[0]) == 4:
            y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            d, mo, y = int(parts[0]), int(parts[1]), int(parts[2])
        return date(y, mo, d)
    except (ValueError, IndexError):
        return None


# BUGFIX: is_chronologically_future's fallback — "no effective_date on this
# chunk (real Qdrant payload, before a re-index picks up data/indexer.py's
# new field) -> proxy it as Jan 1 of enacted_year" — is a fine approximation
# for acts where enactment and taking effect are close together (IPC: 1860
# enacted, 1862 effective; CRPC: 1973/1974; IEA: 1872/1872). It is actively
# WRONG for BNS/BNSS/BSA specifically: Parliament passed them in Dec 2023
# (enacted_year=2023 in the dataset) but they weren't notified into force
# until 2024-07-01 — a ~7 MONTH gap. Jan-1-of-2023 is chronologically
# BEFORE any 2024 cutoff, so the fallback proxy made BNS/BNSS/BSA look
# already in force starting 2023 — silently defeating the exact exclusion
# this whole file exists for, for exactly the three acts it's built around,
# on every query asked before a re-index. ("a theft in January 2024" —
# cutoff_date correctly extracted as 2024-01-01, but BNS's Jan-2023 proxy
# effective date wasn't > that, so BNS_146 was not excluded.) Known and
# fixed here directly, independent of whether a re-index has happened yet.
KNOWN_EFFECTIVE_DATE_OVERRIDE: dict[str, date] = {
    "BNS":  date(2024, 7, 1),
    "BNSS": date(2024, 7, 1),
    "BSA":  date(2024, 7, 1),
}


def is_chronologically_future(
    act_code:           "str | None",
    effective_date_str: "str | None",
    enacted_year:       "int | None",
    cutoff_year:        "int | None",
    cutoff_date:        "str | date | None" = None,
) -> bool:
    """True if a section could not possibly have applied to the query's own
    date. Shared by TemporalFilter.filter (Stage 4), main.py's Rocchio
    feedback merge, and legal_kg.py's KG augmentation — three separate
    retrieval paths that can each inject a chunk into the final answer, so
    all three need to agree on the exact same chronology check rather than
    each keeping (or dropping) its own slice of it.

    Prefers an exact date comparison (effective_date_str vs. cutoff_date)
    when both sides carry day/month precision; falls back to enacted_year
    vs. cutoff_year (whole calendar years) when either side only has a
    bare year — see the module note above for why that matters for 2024.
    """
    if cutoff_year is None:
        return False
    if isinstance(cutoff_date, str):
        cutoff_date = _parse_effective_date(cutoff_date)

    eff = _parse_effective_date(effective_date_str)
    if eff is None:
        eff = KNOWN_EFFECTIVE_DATE_OVERRIDE.get(act_code)
    if eff is None and enacted_year:
        eff = date(enacted_year, 1, 1)   # day precision unknown — Jan 1 proxy
    if eff is None:
        return False

    if cutoff_date is not None:
        return eff > cutoff_date
    return eff.year > cutoff_year


class TemporalFilter:
    """
    Applies temporal validity logic to each retrieved chunk.

    If query intent is 'historical', repealed/amended sections
    are NOT penalized (historical query expects old law).
    """

    def filter(
        self,
        chunks:  list[RetrievedChunk],
        intent:  QueryIntent,
        cutoff_year: int | None = None,
        cutoff_date: "str | date | None" = None,
    ) -> list[ValidatedChunk]:

        # `intent.cutoff_year`/`intent.cutoff_date` (see intent_classifier.py)
        # are the query's own extracted date, e.g. "before 2023" -> year
        # 2022, or "before 1 June 2024" -> the exact date 2024-05-31.
        # Callers may also pass either explicitly; either source anchors the
        # query to a past date, which is a stronger and more specific signal
        # than the coarse historical/current label — so it wins if present.
        if cutoff_year is None:
            cutoff_year = getattr(intent, "cutoff_year", None)
        if cutoff_date is None:
            cutoff_date = getattr(intent, "cutoff_date", None)
        if isinstance(cutoff_date, str):
            cutoff_date = _parse_effective_date(cutoff_date)

        historical_query = (intent.temporal == "historical")
        current_query    = (intent.temporal == "current")
        # BUGFIX: an explicit cutoff means the query names a real past
        # date ("an FIR filed in 2019") even when the phrasing never
        # tripped TEMPORAL_RULES's historical regex — without this, such
        # queries fell through to the `else: penalty = PENALTY.get(...)`
        # branch below, which does nothing period-specific, and BNS/BNSS/
        # BSA sections (enacted_year=2023) sailed through with no penalty
        # at all alongside the correct IPC/CRPC/IEA ones.
        if cutoff_year is not None:
            historical_query = True
            current_query    = False
        results = []

        for chunk in chunks:
            status       = (chunk.status or "unknown").lower()
            superseded   = bool(chunk.payload.get("superseded_by"))
            enacted_year = chunk.enacted_year
            act_code     = chunk.act_code

            # Intrinsic validity label, from the dataset's own status/
            # supersession — independent of the query's own temporal anchor.
            if superseded:
                label = "superseded"
            elif status in ("active", "amended", "repealed"):
                label = status
            else:
                label = "unknown"

            # Chronological impossibility: this section took effect AFTER
            # the date the query is anchored to — it could not possibly
            # have applied to that incident. This is a hard exclusion.
            # Unlike the old code, it is NOT overridden by historical_query
            # — historical intent is exactly what usually produces the
            # cutoff in the first place (e.g. "before 2023"), so
            # "historical" must never mean "let future law back in".
            # Date-precise when the query and the record both support it
            # (see is_chronologically_future's module note) — falls back to
            # year-only for the cases that need it (a bare "2024" with no
            # month genuinely can't be resolved more precisely).
            is_future_law = is_chronologically_future(
                act_code, chunk.payload.get("effective_date"), enacted_year,
                cutoff_year, cutoff_date,
            )

            # Predecessor-act demotion: only for an EXPLICIT "current law"
            # query with no date anchor (e.g. "what is the CURRENT
            # punishment for theft") — a query naming IPC's successor
            # act exists and this section is otherwise "active" in the
            # dataset (which doesn't reciprocally mark IPC/CRPC/IEA as
            # superseded — see ACT_SUCCESSOR's note). Deliberately NOT
            # applied to "unspecified" queries: this dataset's own
            # benchmark treats IPC as the default answer for ordinary
            # scenario questions, so demoting it by default would trade a
            # real bug fix for a large recall regression on queries that
            # never asked about "current" law in the first place.
            is_stale_law = bool(
                current_query and not is_future_law
                and label == "active" and ACT_SUCCESSOR.get(act_code)
            )

            if is_future_law:
                label   = "future"
                penalty = PENALTY["future"]
            elif historical_query:
                # Historical queries want the law that applied AT THE TIME —
                # the dataset's own amended/repealed/superseded chunks are
                # exactly what should surface, so don't penalize them.
                penalty = 1.0
            elif is_stale_law:
                label   = "superseded"
                penalty = PENALTY["superseded"]
            else:
                penalty = PENALTY.get(label, 0.80)

            penalized = chunk.rrf_score * penalty

            # A chunk is "valid" (eligible for the strict active-only list)
            # when it's the law that actually applies to this query:
            #   - chronologically impossible ("future") sections are NEVER
            #     valid, regardless of intent — this is the fix for the
            #     "before 2023" query returning BNS bug: the old code's
            #     `label in ("active",) or historical_query` made
            #     is_valid=True for EVERY chunk once historical_query was
            #     True, silently defeating the cutoff exclusion above.
            #   - otherwise, a historical query accepts any intrinsic label
            #     (it deliberately wants amended/repealed/superseded law).
            #   - otherwise, only the dataset's own "active" label counts.
            if is_future_law:
                is_valid = False
            elif historical_query:
                is_valid = True
            else:
                is_valid = (label == "active")

            results.append(ValidatedChunk(
                chunk           = chunk,
                is_valid        = is_valid,
                validity_label  = label,
                warning         = WARNING_MSG.get(label, ""),
                penalized_score = penalized,
            ))

        # Sort by penalized score descending
        results.sort(key=lambda x: x.penalized_score, reverse=True)
        return results

    def filter_active_only(
        self,
        chunks: list[RetrievedChunk],
        intent: QueryIntent,
        cutoff_year: int | None = None,
        cutoff_date: "str | date | None" = None,
    ) -> list[ValidatedChunk]:
        """Strict filter: return only active sections, with warnings attached to rest."""
        validated = self.filter(chunks, intent, cutoff_year=cutoff_year, cutoff_date=cutoff_date)
        active    = [v for v in validated if v.is_valid]
        # If too few active results, include amended ones with warnings
        if len(active) < 3:
            amended = [v for v in validated if v.validity_label == "amended"]
            active.extend(amended)
            # Re-sort so amended chunks slot in by score, not appended at the end
            active.sort(key=lambda x: x.penalized_score, reverse=True)
        return active


if __name__ == "__main__":
    # Quick test
    from pipeline.intent_classifier import IntentClassifier, QueryIntent
    from pipeline.hybrid_retriever  import RetrievedChunk

    dummy = [
        RetrievedChunk("IPC_001", "content A", 0.9, rrf_score=0.05, status="active"),
        RetrievedChunk("IPC_002", "content B", 0.8, rrf_score=0.04, status="amended"),
        RetrievedChunk("IPC_003", "content C", 0.7, rrf_score=0.03, status="repealed"),
    ]

    intent = QueryIntent(label="statute", confidence=0.9,
                         act_hint="IPC", temporal="current")
    tf     = TemporalFilter()
    result = tf.filter(dummy, intent)

    for v in result:
        print(f"{v.chunk.section_id} | {v.validity_label} | "
              f"score={v.penalized_score:.4f} | valid={v.is_valid} | {v.warning}")
