"""
pipeline/entity_timeline_extractor.py
Phase 1, stages 3-4 — Entity Extraction + Timeline Extraction.

Pulls out the facts ALEA (Phase 3) will later match against legal
elements: parties, cited/implied statute sections, weapons, injuries,
locations, amounts — plus a normalized, sorted timeline of dated
events across the document.

Deliberately regex/rule-based (no LLM call) — these are the kind of
facts where precision matters more than fluency, and it keeps this
stage fast and deterministic for every uploaded document.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime


# ── Entities ─────────────────────────────────────────────────────────────────

@dataclass
class ExtractedEntities:
    complainants:        list[str] = field(default_factory=list)
    accused:              list[str] = field(default_factory=list)
    witnesses:            list[str] = field(default_factory=list)
    investigating_officer: str | None = None
    sections_cited:        list[str] = field(default_factory=list)
    weapons:                list[str] = field(default_factory=list)
    injuries:                list[str] = field(default_factory=list)
    locations:                list[str] = field(default_factory=list)
    amounts:                  list[str] = field(default_factory=list)


PARTY_PATTERNS = {
    "complainants": r"complainant\s*(?:name)?\s*[:\-]\s*([A-Z][A-Za-z\. \-]{2,50})",
    "accused":       r"accused\s*(?:name)?\s*[:\-]\s*([A-Z][A-Za-z\. \-]{2,50})",
    "witnesses":     r"witness\s*(?:name)?\s*[:\-]\s*([A-Z][A-Za-z\. \-]{2,50})",
}

IO_PATTERN = r"invest(?:igat)?ing\s+officer\s*[:\-]\s*([A-Z][A-Za-z\. \-]{2,50})"

# Section citation: "Section 302 IPC", "Sections invoked: 103, 118 BNS", "S. 420 IPC"
SECTION_PATTERN = re.compile(
    r"\b(?:sections?\s*(?:no\.?|invoked|applied|framed|under)?\s*:?\s*|s\.\s*)"
    r"((?:\d+[A-Za-z]?\s*(?:,|and)?\s*)+)"
    r"(?:of\s+(?:the\s+)?)?(IPC|BNS|CrPC|BNSS|IT\s*Act|Evidence\s*Act|IEA|POCSO|NDPS)?\b",
    re.IGNORECASE,
)

WEAPON_TERMS   = ["knife", "gun", "pistol", "revolver", "rifle", "sword", "axe",
                   "iron rod", "stick", "acid", "poison", "explosive", "blade"]
INJURY_TERMS   = ["fracture", "laceration", "stab wound", "gunshot wound",
                   "contusion", "bruise", "internal bleeding", "burn injury",
                   "grievous", "simple hurt", "abrasion"]
AMOUNT_PATTERN = re.compile(r"(?:Rs\.?|₹|INR)\s*[\d,]+(?:\.\d+)?", re.IGNORECASE)
LOCATION_HINT  = re.compile(r"(?:at|near|in front of)\s+([A-Z][A-Za-z\s]{3,40}?)(?:,|\.|$)")


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        k = i.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(i.strip())
    return out


def extract_entities(text: str) -> ExtractedEntities:
    entities = ExtractedEntities()

    for field_name, pattern in PARTY_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        setattr(entities, field_name, _dedupe(matches))

    io_match = re.search(IO_PATTERN, text, re.IGNORECASE)
    if io_match:
        entities.investigating_officer = io_match.group(1).strip()

    sections = []
    for match in SECTION_PATTERN.finditer(text):
        numbers = re.split(r"[,\s]+and\s+|,\s*", match.group(1).strip())
        act = (match.group(2) or "").replace(" ", "").upper() or None
        for n in numbers:
            n = n.strip().rstrip(",")
            if n:
                sections.append(f"{n} {act}".strip() if act else n)
    entities.sections_cited = _dedupe(sections)

    lower_text = text.lower()
    entities.weapons  = _dedupe([w for w in WEAPON_TERMS if w in lower_text])
    entities.injuries = _dedupe([i for i in INJURY_TERMS if i in lower_text])
    entities.amounts  = _dedupe(AMOUNT_PATTERN.findall(text))
    entities.locations = _dedupe(LOCATION_HINT.findall(text))[:10]

    return entities


# ── Timeline ─────────────────────────────────────────────────────────────────

@dataclass
class TimelineEvent:
    raw_date:        str
    normalized_date: str | None   # ISO format if parseable, else None
    context:         str          # surrounding text snippet


DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",               # 12/05/2024, 12-05-24
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b",  # 12th May 2024
    r"\b[A-Za-z]+\s+\d{1,2},?\s+\d{4}\b",                # May 12, 2024
]
DATE_FORMATS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%B %d %Y", "%b %d, %Y",
]


def _normalize_date(raw: str) -> str | None:
    cleaned = re.sub(r"(st|nd|rd|th)", "", raw).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def extract_timeline(text: str, context_window: int = 60) -> list[TimelineEvent]:
    events, seen_spans = [], set()

    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text):
            span = match.span()
            if span in seen_spans:
                continue
            seen_spans.add(span)

            raw = match.group(0)
            start = max(0, span[0] - context_window)
            end   = min(len(text), span[1] + context_window)
            context = re.sub(r"\s+", " ", text[start:end]).strip()

            events.append(TimelineEvent(
                raw_date=raw,
                normalized_date=_normalize_date(raw),
                context=context,
            ))

    # Dated events first (chronological), undated events kept at the end
    # in their original document order.
    dated   = [e for e in events if e.normalized_date]
    undated = [e for e in events if not e.normalized_date]
    dated.sort(key=lambda e: e.normalized_date)
    return dated + undated


if __name__ == "__main__":
    sample = """
    Complainant: Ramesh Kumar
    Accused: Suresh Yadav
    On 12/05/2024 at Andheri Station Road, the accused attacked the complainant
    with a knife causing a stab wound. FIR registered under Section 103, 118 BNS.
    Investigating Officer: Inspector P. Sharma. A compensation of Rs. 50,000 was sought.
    """
    entities = extract_entities(sample)
    print(entities)
    print()
    for ev in extract_timeline(sample):
        print(ev.raw_date, "->", ev.normalized_date, "|", ev.context)
