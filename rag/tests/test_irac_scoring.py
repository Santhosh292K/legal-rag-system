"""The IRAC reranker's cheap metadata stage.

Two of its three signals were dead before this: the conclusion-type match
(compared lowercase literals against a capitalized free-text field) and
the act bonus (compared an act CODE against a full act TITLE, which
matched the wrong acts for ITA and LA).
"""
import pytest

from pipeline.chunk_structurer import StructuredChunk
from pipeline.intent_classifier import QueryIntent
from pipeline.irac_reranker import metadata_irac_score, _token_overlap


def _chunk(**kw) -> StructuredChunk:
    base = dict(
        section_id="IPC_302", content="Punishment for murder is death or life imprisonment.",
        act_name="Indian Penal Code, 1860", act_code="IPC", chapter="", category="",
        validity_label="active", warning="", penalized_score=0.0,
        rule_summary="Punishment for murder is death or imprisonment for life and fine.",
        issue_tags=["murder", "punishment"], conclusion_type="penal",
    )
    base.update(kw)
    base.setdefault("enriched_context", base["content"])
    return StructuredChunk(**base)


def _intent(**kw) -> QueryIntent:
    base = dict(label="punitive", confidence=0.9, act_hint="IPC",
                temporal="unspecified", labels=["punitive"])
    base.update(kw)
    return QueryIntent(**base)


def test_conclusion_match_fires_on_real_dataset_values():
    *_, conc = metadata_irac_score("punishment for murder", _intent(), _chunk())
    assert conc == 1.0, "the punitive/Penal match must actually fire"


def test_conclusion_partial_for_wrong_family():
    *_, conc = metadata_irac_score(
        "punishment for murder", _intent(), _chunk(conclusion_type="procedural order"))
    assert conc == 0.5


def test_conclusion_untyped():
    *_, conc = metadata_irac_score(
        "punishment for murder", _intent(), _chunk(conclusion_type=""))
    assert conc == 0.3


def test_act_bonus_uses_act_code_not_title():
    """act_hint is 'IPC'; act_name is 'Indian Penal Code, 1860'. A
    substring test against the title never matched for 16 of 18 acts."""
    query = "punishment for murder"
    _, with_hint, *_ = metadata_irac_score(query, _intent(act_hint="IPC"), _chunk())
    _, no_hint, *_   = metadata_irac_score(query, _intent(act_hint=None), _chunk())
    assert with_hint > no_hint


def test_act_bonus_does_not_fire_on_a_different_act():
    """"ITA" is a substring of "Bharatiya Nyaya SanhITA" and
    "LimITAtion Act"; "LA" of "UnLAwful Activities". An IT Act query used
    to boost all 358 BNS sections and none of the 113 ITA ones."""
    query = "hacking under the IT Act"
    bns = _chunk(section_id="BNS_103", act_code="BNS",
                 act_name="Bharatiya Nyaya Sanhita 2023")
    _, bns_rule, *_ = metadata_irac_score(query, _intent(act_hint="ITA"), bns)
    _, bns_base, *_ = metadata_irac_score(query, _intent(act_hint=None), bns)
    assert bns_rule == bns_base, "wrong act received the act bonus"


def test_secondary_intent_labels_count():
    intent = _intent(label="statute", labels=["statute", "punitive"])
    *_, conc = metadata_irac_score("punishment for murder", intent, _chunk())
    assert conc == 1.0


def test_scores_are_bounded():
    for chunk in (_chunk(), _chunk(issue_tags=[], rule_summary=""),
                  _chunk(content="")):
        for score in metadata_irac_score("murder", _intent(), chunk):
            assert 0.0 <= score <= 1.0


def test_short_metadata_fields_are_not_structurally_penalised():
    """Raw Jaccard scored a 3-word issue_tags against a 60-token query at
    ~0. The measured median across a realistic candidate set was 0.000,
    which is what made this stage's ordering effectively arbitrary."""
    long_query = ("a man stabbed another man during an argument and the "
                  "victim died at the hospital, what offence is this and "
                  "what punishment applies under Indian criminal law")
    issue, *_ = metadata_irac_score(long_query, _intent(), _chunk())
    assert issue > 0.0


def test_overlap_is_bounded_and_symmetric():
    a, b = "criminal breach of trust", "breach of trust by a servant"
    assert _token_overlap(a, b) == pytest.approx(_token_overlap(b, a))
    assert 0.0 <= _token_overlap(a, b) <= 1.0
    assert _token_overlap("", "anything") == 0.0
