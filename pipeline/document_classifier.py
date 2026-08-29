"""
pipeline/document_classifier.py
Phase 1, stage 2 — Document Classifier.

Identifies which of the DOC_TYPES an uploaded document is, so
downstream stages (entity extraction, adaptive chunking) can apply
the right template. Same rule-based-first, LLM-fallback pattern as
pipeline/intent_classifier.py.
"""
import re
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import OLLAMA_FAST_MODEL, DOC_TYPES
import ollama


@dataclass
class DocumentClassification:
    doc_type:    str
    confidence:  float
    reasoning:   str = ""


# Header/phrase patterns that strongly indicate a document type.
# Matched against the first ~1500 chars, where headers/letterheads live.
TYPE_PATTERNS: dict[str, list[str]] = {
    "FIR": [
        r"\bfirst\s+information\s+report\b", r"\bf\.?i\.?r\.?\s*no\b",
        r"\bunder\s+section\s+\d+.{0,20}(cr\.?p\.?c|bnss)\b",
        r"\bcomplainant\b", r"\bstation\s+house\s+officer\b", r"\bp\.?s\.?\s*:",
    ],
    "Charge Sheet": [
        r"\bcharge\s*sheet\b", r"\bfinal\s+report\b", r"\bunder\s+section\s+173\b",
        r"\binvestigating\s+officer\b", r"\bcharges?\s+framed\b", r"\bcognizance\b",
    ],
    "Medical Report": [
        r"\bmedico[\s-]?legal\b", r"\bpost[\s-]?mortem\b", r"\bautopsy\b",
        r"\bpatient\s+(name|history)\b", r"\bdiagnosis\b", r"\battending\s+physician\b",
        r"\bwound\s+certificate\b",
    ],
    "Witness Statement": [
        r"\bstatement\s+of\s+witness\b", r"\bunder\s+section\s+161\b",
        r"\bdeposition\b", r"\bi\s+state\s+(on\s+oath\s+)?as\s+follows\b",
    ],
    "Forensic Report": [
        r"\bforensic\s+(science\s+)?lab(oratory)?\b", r"\bfsl\b", r"\bballistic\b",
        r"\bdna\s+(profil|analysis)\b", r"\bchemical\s+examiner\b",
        r"\bfingerprint\s+analysis\b",
    ],
    "Contract": [
        r"\bthis\s+agreement\b", r"\bwitnesseth\b", r"\bparty\s+of\s+the\s+(first|second)\s+part\b",
        r"\bin\s+consideration\s+of\b", r"\bterms\s+and\s+conditions\b",
    ],
    "Email": [
        r"^from\s*:", r"^to\s*:", r"^subject\s*:", r"^sent\s*:", r"\bforwarded\s+message\b",
    ],
    "Court Order": [
        r"\bin\s+the\s+court\s+of\b", r"\bhon(')?ble\b", r"\border\s+dated\b",
        r"\bjudgment\s+and\s+order\b", r"\bit\s+is\s+hereby\s+ordered\b",
    ],
    "Affidavit": [
        r"\baffidavit\b", r"\bdeponent\b", r"\bsolemnly\s+affirm\b", r"\bverified\s+at\b",
    ],
}


def rule_based_classify(text: str) -> DocumentClassification:
    header = text[:1500].lower()

    scores: dict[str, int] = {}
    for doc_type, patterns in TYPE_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, header, re.MULTILINE))
        if score > 0:
            scores[doc_type] = score

    if not scores:
        return DocumentClassification(doc_type="Other", confidence=0.0,
                                       reasoning="No known header pattern matched.")

    best = max(scores, key=scores.get)
    confidence = min(scores[best] / 3.0, 0.9)
    return DocumentClassification(
        doc_type=best,
        confidence=confidence,
        reasoning=f"Matched {scores[best]} header pattern(s) for {best}.",
    )


CLASSIFY_PROMPT = """You are classifying an Indian legal/evidence document.
Choose exactly one label from this list: {labels}

Return ONLY a JSON object:
{{
  "doc_type": "<one of the labels above>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}

Document excerpt (first portion):
{excerpt}

Return ONLY the JSON. No markdown."""


def llm_classify(text: str) -> DocumentClassification:
    response = ollama.chat(
        model=OLLAMA_FAST_MODEL,
        format="json",
        messages=[{
            "role": "user",
            "content": CLASSIFY_PROMPT.format(labels=", ".join(DOC_TYPES), excerpt=text[:2000]),
        }],
    )
    data = json.loads(response["message"]["content"])
    doc_type = data.get("doc_type", "Other")
    if doc_type not in DOC_TYPES:
        doc_type = "Other"
    return DocumentClassification(
        doc_type=doc_type,
        confidence=float(data.get("confidence", 0.5)),
        reasoning=data.get("reasoning", ""),
    )


class DocumentClassifier:
    def classify(self, text: str) -> DocumentClassification:
        fast = rule_based_classify(text)
        if fast.confidence >= 0.6:
            return fast
        try:
            return llm_classify(text)
        except Exception:
            return fast


if __name__ == "__main__":
    sample = """FIRST INFORMATION REPORT
    F.I.R No: 0142/2025
    P.S: Andheri
    Complainant: Ramesh Kumar
    Under Section 103, 118 BNS
    """
    clf = DocumentClassifier()
    result = clf.classify(sample)
    print(f"doc_type={result.doc_type} confidence={result.confidence:.2f} — {result.reasoning}")
