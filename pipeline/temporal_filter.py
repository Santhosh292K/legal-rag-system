"""
pipeline/temporal_filter.py
Novel component #2 — Temporal Validity Filter

Filters and flags retrieved chunks based on:
  1. Amendment status (active / amended / repealed)
  2. Supersession chain (is there a newer version?)
  3. Query temporal intent (current law vs. historical)

Returns chunks with validity flags attached.
"""
from dataclasses import dataclass
from datetime import datetime
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
}

WARNING_MSG = {
    "amended":    "This section has been amended. Verify current wording.",
    "repealed":   "This section has been repealed and is no longer in force.",
    "superseded": "This section has been superseded by a later provision.",
    "unknown":    "Amendment status is unknown. Verify independently.",
}


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
    ) -> list[ValidatedChunk]:

        historical_query = (intent.temporal == "historical")
        current_query    = (intent.temporal == "current")
        results          = []

        for chunk in chunks:
            status       = (chunk.status or "unknown").lower()
            superseded   = bool(chunk.payload.get("superseded_by"))
            enacted_year = chunk.enacted_year

            # Determine validity label
            if superseded:
                label = "superseded"
            elif status in ("active", "amended", "repealed"):
                label = status
            else:
                label = "unknown"

            # Apply penalty
            if historical_query:
                # For historical queries: amended/repealed are VALID
                penalty = 1.0 if label in ("amended", "repealed", "superseded") else PENALTY[label]
            elif current_query:
                penalty = PENALTY.get(label, 0.80)
            else:
                penalty = PENALTY.get(label, 0.80)

            # Year-based filter: if cutoff_year set, exclude sections enacted after
            if cutoff_year and enacted_year and enacted_year > cutoff_year:
                penalty = min(penalty, 0.3)
                label   = "superseded"

            penalized = chunk.rrf_score * penalty
            is_valid  = label in ("active",) or historical_query

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
    ) -> list[ValidatedChunk]:
        """Strict filter: return only active sections, with warnings attached to rest."""
        validated = self.filter(chunks, intent)
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
