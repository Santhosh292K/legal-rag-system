"""
data/section_ref.py

One canonical implementation of "how do I turn something a human (or the
dataset, or an LLM) wrote into a real section_id".

This logic used to exist in five slightly-different copies —
hybrid_retriever._direct_section_lookup, irac_reranker.rerank,
answer_generator._fmt_sec, chunk_structurer._normalize_ref_candidates and
fusion._sections_from_case_chunks — each with its own subset of the
padding/case/act-alias rules and its own bugs. Divergence between them was
a recurring source of silently-dropped sections.

The conventions this encodes, all verified against final_dataset.json:

  * section_id is ``ACT_<digits><LETTERS>``, e.g. IPC_302, IPC_029A.
  * The numeric part is zero-padded to 3 digits for every act EXCEPT
    CRPC, whose ids are a genuine unpadded mix of 1/2/3-digit widths.
    Rather than special-casing CRPC, both candidates are produced and the
    one that actually exists wins.
  * Letter suffixes are uppercase.
  * The dataset writes cross-references as "IPC 2" (space, unpadded).
"""
from __future__ import annotations

import re

# Every act code present in the corpus, plus the aliases people actually
# type. Longest-first alternation matters: "bnss" must win over "bns".
ACT_ALIASES: dict[str, str] = {
    "ipc": "IPC", "indian penal code": "IPC", "penal code": "IPC",
    "crpc": "CRPC", "cr.p.c": "CRPC", "cr p c": "CRPC",
    "code of criminal procedure": "CRPC", "criminal procedure code": "CRPC",
    "cpc": "CPC", "code of civil procedure": "CPC", "civil procedure code": "CPC",
    "ita": "ITA", "it act": "ITA", "information technology act": "ITA",
    "iea": "IEA", "indian evidence act": "IEA", "evidence act": "IEA",
    "coi": "COI", "constitution": "COI", "constitution of india": "COI",
    "bns": "BNS", "bharatiya nyaya sanhita": "BNS",
    "bnss": "BNSS", "bharatiya nagarik suraksha sanhita": "BNSS",
    "bsa": "BSA", "bharatiya sakshya adhiniyam": "BSA",
    "sra": "SRA", "specific relief act": "SRA",
    "tpa": "TPA", "transfer of property act": "TPA",
    "ica": "ICA", "indian contract act": "ICA", "contract act": "ICA",
    "ndps": "NDPS", "pca": "PCA", "prevention of corruption act": "PCA",
    "pocso": "POCSO", "scst": "SCST", "uapa": "UAPA",
    "la": "LA", "limitation act": "LA",
}

ACT_CODES = sorted(set(ACT_ALIASES.values()))

_NUM_RE = re.compile(r"^(\d+)\s*([A-Za-z]{0,2})$")


def canonical_act(name: str) -> str | None:
    """'it act' / 'IPC' / 'Indian Penal Code' -> 'ITA' / 'IPC' / 'IPC'."""
    if not name:
        return None
    key = re.sub(r"[.\s]+", " ", name.strip().lower()).strip()
    if key in ACT_ALIASES:
        return ACT_ALIASES[key]
    upper = name.strip().upper()
    return upper if upper in ACT_CODES else None


def candidates(act_code: str, number: str) -> list[str]:
    """Every plausible section_id for an act + number, most likely first.

    Zero-pads the DIGITS only — ``"29A".zfill(3)`` is a no-op because the
    string is already 3 characters, which is why the old copies of this
    logic produced "IPC_29A" instead of the indexed "IPC_029A".
    """
    num = (number or "").strip().upper()
    m   = _NUM_RE.match(num)
    if not m:
        return [f"{act_code}_{num}"] if num else []
    digits, letters = m.group(1), m.group(2)
    padded   = f"{act_code}_{digits.zfill(3)}{letters}"
    unpadded = f"{act_code}_{digits}{letters}"
    return list(dict.fromkeys([padded, unpadded]))


def normalize_section_ref(raw: str, known_ids: set[str] | None = None) -> str | None:
    """Turn one written reference into a section_id.

    Accepts "IPC 2", "IPC_2", "ipc 304a", "IPC_029A". When ``known_ids``
    is supplied, only a reference that resolves to a real section is
    returned (None otherwise) — which is what lets the indexer drop
    dangling cross-references instead of writing phantom graph nodes.
    Without it, the zero-padded form is returned as a best guess.
    """
    if not raw:
        return None
    text = raw.strip()
    if known_ids and text in known_ids:
        return text

    m = re.match(r"^([A-Za-z][A-Za-z.\s]*?)[\s_]+(\d+\s*[A-Za-z]{0,2})$", text)
    if not m:
        return None
    act = canonical_act(m.group(1))
    if not act:
        return None

    for cand in candidates(act, m.group(2)):
        if known_ids is None or cand in known_ids:
            return cand
    return None


# ── Free-text extraction ────────────────────────────────────────────────────
# Matches BOTH orders, which the old regex did not:
#     "IPC 302", "IPC section 302"          (act first)
#     "section 302 of the IPC", "u/s 302 IPC" (number first)
# and continuation runs, so "IPC 494 and 495" yields BOTH numbers rather
# than silently dropping every number after the first.

_ACT_ALT = "|".join(
    re.escape(a) for a in sorted(ACT_ALIASES, key=len, reverse=True)
)
# "u/s" (under section) is listed FIRST so the alternation consumes the
# whole abbreviation. Matching only its trailing "s" left the "u/" 
# outside the span, so a rewrite produced "u/[IPC_420]".
_SEC_WORD = r"(?:u\s*/\s*s|ss|s|secs|sec|sections|section|articles|article|arts|art)\.?"
# Filler that sits between the word "section" and the numbers in real
# documents: "Sections invoked: 103, 118 BNS", "Section no. 420 IPC",
# "charged under sections 302, 34 IPC".
_SEC_FILLER = r"(?:\s*(?:invoked|applied|framed|charged|registered|under|no)\b\.?)*[\s:,\-]*"
# Letter suffixes are always adjacent to the digits ("304A", "66C", never
# "304 A"), so no whitespace is allowed between them. Permitting it made
# the number group swallow the following English word — "Article 21 of"
# parsed its number as "21 of" — which broke every reference it touched.
_NUM      = r"\d+[A-Za-z]{0,2}"

_ACT_FIRST = re.compile(
    rf"\b({_ACT_ALT})\b[\s,]*(?:{_SEC_WORD}{_SEC_FILLER})?[\s,]*({_NUM}(?:\s*(?:,|and|&|/)\s*{_NUM})*)",
    re.IGNORECASE,
)
_NUM_FIRST = re.compile(
    rf"(?:under\s+)?\b{_SEC_WORD}{_SEC_FILLER}({_NUM}(?:\s*(?:,|and|&|/)\s*{_NUM})*)"
    rf"\s*(?:of\s+)?(?:the\s+)?\b({_ACT_ALT})\b",
    re.IGNORECASE,
)


# A bare "<number> <ACT>" with nothing between them — "468 IPC", the
# second half of "u/s 420 IPC and 468 IPC". Deliberately allows no
# punctuation between the number and the act code, and every candidate
# still has to resolve against known_ids, so a stray "1860 IPC" resolves
# to nothing and is dropped rather than becoming a fake citation.
_BARE_NUM_ACT = re.compile(
    rf"\b({_NUM}(?:\s*(?:,|and|&|/)\s*{_NUM})*)\s+({_ACT_ALT})\b",
    re.IGNORECASE,
)


def _split_numbers(blob: str) -> list[str]:
    return [n.strip() for n in re.split(r"\s*(?:,|and|&|/)\s*", blob) if n.strip()]


def extract_section_refs(text: str, known_ids: set[str] | None = None) -> list[str]:
    """Every explicit section reference in a piece of free text, as
    section_ids, in order of appearance and de-duplicated."""
    if not text:
        return []

    found: list[tuple[int, str, str]] = []
    for rx, act_group, num_group in (
        (_ACT_FIRST, 1, 2), (_NUM_FIRST, 2, 1), (_BARE_NUM_ACT, 2, 1),
    ):
        for m in rx.finditer(text):
            act = canonical_act(m.group(act_group))
            if act:
                for num in _split_numbers(m.group(num_group)):
                    found.append((m.start(), act, num))

    found.sort(key=lambda t: t[0])

    out: list[str] = []
    for _, act, num in found:
        for cand in candidates(act, num):
            if known_ids is None or cand in known_ids:
                if cand not in out:
                    out.append(cand)
                break
    return out


def find_section_spans(
    text: str, known_ids: set[str] | frozenset[str] | None = None,
) -> list[tuple[int, int, list[str]]]:
    """Locate every explicit section reference and report where it sits.

    Returns ``(start, end, [section_id, ...])`` tuples in document order,
    non-overlapping, for callers that need to REWRITE the text rather than
    just collect ids — currently answer_generator's citation normalizer.

    A span is only reported when at least one of its numbers resolves to a
    real section, so ordinary prose ("Section 302 defines murder") is left
    untouched instead of being mangled into a fake citation.
    """
    if not text:
        return []

    spans: list[tuple[int, int, list[str]]] = []
    for rx, act_group, num_group in (
        (_ACT_FIRST, 1, 2), (_NUM_FIRST, 2, 1), (_BARE_NUM_ACT, 2, 1),
    ):
        for m in rx.finditer(text):
            act = canonical_act(m.group(act_group))
            if not act:
                continue
            ids: list[str] = []
            for num in _split_numbers(m.group(num_group)):
                for cand in candidates(act, num):
                    if known_ids is None or cand in known_ids:
                        if cand not in ids:
                            ids.append(cand)
                        break
            if ids:
                spans.append((m.start(), m.end(), ids))

    spans.sort(key=lambda sp: (sp[0], -sp[1]))
    merged: list[tuple[int, int, list[str]]] = []
    last_end = -1
    for start, end, ids in spans:
        if start >= last_end:
            merged.append((start, end, ids))
            last_end = end
    return merged
