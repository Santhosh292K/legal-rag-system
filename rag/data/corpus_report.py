"""
data/corpus_report.py
Corpus health metrics, printed by data/indexer.py after every build.

The pipeline's ceiling is set by the dataset, not the code, and the single
most consequential property of this dataset is invisible unless you go
looking: `content` holds a PARAPHRASE of each section, not its statutory
text. IPC_302's content is 65 characters —

    "Punishment for murder is death or imprisonment for life and fine."

— against roughly 110 for the provision as enacted, and most sections lose
their provisos, exceptions, explanations and ingredient lists entirely.

That bounds everything downstream: answers cannot quote statutory
language, "citation-grounded" can only mean "grounded in a summary", and
ALEA scores case evidence against paraphrases. It also explains
evaluation results that otherwise look like retrieval failures —
answer_coverage 0.0 and ROUGE-L 0.12 are largely a corpus property.

No code change reaches this. Surfacing it on every index build at least
makes it a known, measured limitation rather than a silent one.
"""
from __future__ import annotations

import statistics


# Below this median content length, the corpus is summaries rather than
# statutory text. Real statutory sections run to several hundred
# characters even when short.
SUMMARY_LENGTH_THRESHOLD = 400


def corpus_report(records: list[dict]) -> dict:
    lengths = [len(r.get("content", "")) for r in records]
    meta    = [r.get("meta") or {} for r in records]

    def _share(predicate) -> float:
        return sum(1 for m in meta if predicate(m)) / max(len(meta), 1)

    return {
        "n_records":        len(records),
        "content_min":      min(lengths, default=0),
        "content_median":   int(statistics.median(lengths)) if lengths else 0,
        "content_mean":     int(statistics.mean(lengths)) if lengths else 0,
        "content_max":      max(lengths, default=0),
        "with_keywords":    _share(lambda m: bool(m.get("keywords"))),
        "with_rule_summary": _share(lambda m: bool((m.get("irac") or {}).get("rule_summary"))),
        "with_conclusion":  _share(lambda m: bool((m.get("irac") or {}).get("conclusion_type"))),
        "with_effective_date": _share(
            lambda m: bool((m.get("temporal") or {}).get("effective_date"))),
    }


def format_corpus_report(report: dict, dropped_refs: int = 0) -> str:
    lines = [
        "Corpus report",
        f"  records            : {report['n_records']}",
        f"  content length     : min {report['content_min']}  "
        f"median {report['content_median']}  mean {report['content_mean']}  "
        f"max {report['content_max']} chars",
        f"  keywords           : {report['with_keywords']:.0%} of sections",
        f"  rule_summary       : {report['with_rule_summary']:.0%}",
        f"  conclusion_type    : {report['with_conclusion']:.0%}",
        f"  effective_date     : {report['with_effective_date']:.0%}",
    ]
    if dropped_refs:
        lines.append(f"  dangling refs      : {dropped_refs} dropped "
                     f"(cross-references naming no indexed section)")

    if report["content_median"] < SUMMARY_LENGTH_THRESHOLD:
        lines += [
            "",
            f"  ⚠ content is SUMMARISED, not statutory text "
            f"(median {report['content_median']} chars).",
            "    Answers can only be as faithful as these paraphrases: provisos,",
            "    exceptions and ingredient lists are not in the index, so a",
            "    grounded answer cannot quote or reason over them. This is a",
            "    dataset limitation, not a retrieval one — see",
            "    data/corpus_report.py. Expect low answer_coverage/ROUGE in",
            "    evaluation regardless of how well retrieval performs.",
        ]
    return "\n".join(lines)
