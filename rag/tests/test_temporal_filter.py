"""Chronology: a section cannot govern conduct before it existed, or after
it was repealed. IPC/CrPC/IEA were replaced by BNS/BNSS/BSA on 2024-07-01
— a mid-year changeover, so year-only comparison is not enough."""
from datetime import date

import pytest

from pipeline.hybrid_retriever import RetrievedChunk
from pipeline.intent_classifier import QueryIntent
from pipeline.temporal_filter import (
    TemporalFilter, PENALTY,
    is_chronologically_future, is_superseded_by_cutoff, _parse_effective_date,
)


@pytest.mark.parametrize("raw,expected", [
    ("2024-07-01", date(2024, 7, 1)),      # ISO, as IPC/BNS/IEA/BSA are written
    ("01-07-2024", date(2024, 7, 1)),      # day-first, as CRPC/BNSS are written
    ("1862-01-01", date(1862, 1, 1)),
    ("garbage", None), ("", None), (None, None), ("32-13-2024", None),
])
def test_effective_date_parsing_handles_both_dataset_formats(raw, expected):
    assert _parse_effective_date(raw) == expected


class TestFutureLaw:
    def test_bns_excluded_for_a_2020_incident(self):
        assert is_chronologically_future("BNS", "2024-07-01", 2023, 2020, None)

    def test_bns_excluded_for_january_2024(self):
        """Unambiguously pre-changeover. Year-only comparison could not
        tell this apart from December 2024."""
        assert is_chronologically_future("BNS", "2024-07-01", 2023, 2024, "2024-01-01")

    def test_bns_allowed_for_december_2024(self):
        assert not is_chronologically_future("BNS", "2024-07-01", 2023, 2024, "2024-12-01")

    def test_missing_effective_date_falls_back_to_the_known_override(self):
        """enacted_year=2023 with a Jan-1 proxy would place BNS in force
        from Jan 2023 — seven months early — and defeat the exclusion."""
        assert is_chronologically_future("BNS", None, 2023, 2024, "2024-01-01")

    def test_ipc_is_never_future(self):
        assert not is_chronologically_future("IPC", "1862-01-01", 1860, 2020, None)

    def test_no_cutoff_means_no_exclusion(self):
        assert not is_chronologically_future("BNS", "2024-07-01", 2023, None, None)


class TestExpiredLaw:
    def test_ipc_excluded_after_the_changeover(self):
        assert is_superseded_by_cutoff("IPC", 2026, "2026-03-01")

    def test_ipc_allowed_before_the_changeover(self):
        assert not is_superseded_by_cutoff("IPC", 2020, "2020-03-01")

    def test_changeover_year_is_left_ambiguous_on_a_bare_year(self):
        assert not is_superseded_by_cutoff("IPC", 2024, None)

    def test_acts_with_no_successor_are_unaffected(self):
        assert not is_superseded_by_cutoff("ITA", 2026, "2026-03-01")


def _chunk(sid, act, year, eff, status="Active"):
    return RetrievedChunk(
        section_id=sid, content="c", score=1.0, rrf_score=0.5,
        act_code=act, status=status, enacted_year=year,
        payload={"effective_date": eff, "superseded_by": ""},
    )


class TestFilter:
    def test_future_sections_are_invalid_even_for_historical_queries(self):
        """`is_valid = label == "active" or historical_query` made every
        chunk valid the moment a cutoff was present, defeating both
        exclusions."""
        intent = QueryIntent(label="statute", confidence=0.9, act_hint=None,
                             temporal="historical", cutoff_year=2020)
        out = TemporalFilter().filter([_chunk("BNS_103", "BNS", 2023, "2024-07-01")], intent)
        assert out[0].validity_label == "future"
        assert out[0].is_valid is False
        assert out[0].penalized_score == pytest.approx(0.5 * PENALTY["future"])

    def test_expired_sections_are_invalid(self):
        intent = QueryIntent(label="statute", confidence=0.9, act_hint=None,
                             temporal="unspecified", cutoff_year=2026,
                             cutoff_date="2026-03-01")
        out = TemporalFilter().filter([_chunk("IPC_302", "IPC", 1860, "1862-01-01")], intent)
        assert out[0].validity_label == "expired"
        assert out[0].is_valid is False

    def test_active_section_passes_cleanly(self):
        intent = QueryIntent(label="statute", confidence=0.9, act_hint=None,
                             temporal="unspecified")
        out = TemporalFilter().filter([_chunk("IPC_302", "IPC", 1860, "1862-01-01")], intent)
        assert out[0].validity_label == "active"
        assert out[0].is_valid is True
        assert out[0].penalized_score == pytest.approx(0.5)

    def test_results_are_sorted_by_penalised_score(self):
        intent = QueryIntent(label="statute", confidence=0.9, act_hint=None,
                             temporal="unspecified")
        chunks = [
            _chunk("A", "IPC", 1860, "1862-01-01", status="Repealed"),
            _chunk("B", "IPC", 1860, "1862-01-01", status="Active"),
        ]
        out = TemporalFilter().filter(chunks, intent)
        assert [v.chunk.section_id for v in out] == ["B", "A"]
