"""Act disambiguation must rank its evidence tiers consistently whether or
not an embedding function is wired up."""
import pytest

from pipeline.domain_router import DomainRouter, DOMAINS, ALL_ACTS


@pytest.fixture(scope="module")
def regex_router():
    """No embed_fn — domain scores are integer regex hit counts."""
    return DomainRouter(embed_fn=None)


def test_literal_act_mention_wins(regex_router):
    r = regex_router.route("What does IPC 302 say?")
    assert r.primary_acts[0] == "IPC"
    assert r.primary_act_has_direct_evidence is True


def test_tier2_cannot_outrank_a_literal_mention(regex_router):
    """A query with many criminal-domain regex hits AND an explicit
    mention of a less-common act must still resolve to that act. Before
    normalization the Tier-2 fallback scaled with raw hit counts and could
    swamp the per-act evidence."""
    r = regex_router.route(
        "murder theft robbery assault kidnapping fraud forgery bribery "
        "under the NDPS act"
    )
    assert r.primary_acts[0] == "NDPS", r.primary_acts


def test_fallback_only_query_reports_no_direct_evidence(regex_router):
    r = regex_router.route("someone was murdered in the street")
    assert r.primary_acts, "a criminal scenario must still activate acts"
    assert r.primary_act_has_direct_evidence is False, (
        "locking retrieval to one act is only safe with direct evidence"
    )


def test_activated_acts_are_real(regex_router):
    r = regex_router.route("my landlord evicted me without notice")
    assert set(r.acts) <= ALL_ACTS


def test_missing_acts_are_surfaced(regex_router):
    r = regex_router.route("my cheque bounced, what can I do?")
    assert any("Negotiable Instruments" in m for m in r.missing_acts)
    assert r.plausibly_legal is True


def test_non_legal_query_is_rejected(regex_router):
    assert regex_router.route("what's my name").plausibly_legal is False


def test_ambiguous_query_activates_everything(regex_router):
    r = regex_router.route("hello there")
    assert len(r.acts) == len(ALL_ACTS) or not r.plausibly_legal


def test_domain_priorities_stay_within_the_semantic_tier_gap():
    """Tier 2 is bounded by max(priority) * n_domains; it must not reach
    the 100 the semantic tier starts at, or the ordering inverts."""
    assert max(d.priority for d in DOMAINS) * len(DOMAINS) < 100
