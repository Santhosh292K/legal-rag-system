"""
pipeline/alea.py
Phase 3 — ALEA (Adaptive Legal Evidence Alignment).

Implements the formal scoring mechanism defined for this system:

    c(ri) = max over e in E of [ w(e) x sim(ri, e) ]        (per-element match confidence)
    C(s)  = (1/m) x sum_i c(ri)                                (section coverage score)
    A(s)  = C(s) x retrieval_prior(s)                            (applicability score)

    bands:  Strong >= 0.75, Partial >= 0.40, Weak > 0, Missing == 0

This module does NOT do retrieval or embedding itself — it's handed
evidence facts (already extracted, Phase 1) and an embed_fn (reused
from whichever model the caller already has loaded, e.g. CaseIndexer's
SentenceTransformer) so no separate model gets loaded for this step.
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from config import ONTOLOGY_PATH

SIM_THRESHOLD = 0.40   # below this, sim(ri, e) is treated as no match (c=0 contribution)

# Source-reliability weights, w(e) — matches the Phase 3 design: medical/
# forensic evidence is the most reliable, witness statements next, everything
# else (FIR/chargesheet narrative, generic) counted as low-to-medium.
SOURCE_WEIGHTS = {
    "Medical Report":    1.0,
    "Forensic Report":   1.0,
    "Witness Statement": 0.6,
    "FIR":               0.6,
    "Charge Sheet":      0.6,
    "Court Order":       0.6,
    "Affidavit":         0.5,
}
DEFAULT_WEIGHT = 0.3   # unknown/other doc types — circumstantial by default


@dataclass
class EvidenceFact:
    text:              str
    fact_type:         str     # "weapon" | "injury" | "party" | "narrative" | ...
    source_doc_type:   str     # e.g. "FIR", "Medical Report" — drives w(e)
    source_document_id: str = ""

    @property
    def weight(self) -> float:
        return SOURCE_WEIGHTS.get(self.source_doc_type, DEFAULT_WEIGHT)


@dataclass
class ElementMatch:
    element_id:          str
    element_description: str
    confidence:          float           # c(ri)
    best_evidence:        EvidenceFact | None = None
    best_sim:              float = 0.0

    @property
    def label(self) -> str:
        if self.confidence >= 0.6:
            return "High"
        if self.confidence >= 0.3:
            return "Medium"
        if self.confidence > 0:
            return "Low"
        return "Missing"

    @property
    def mark(self) -> str:
        return {"High": "\u2714", "Medium": "\u25b3", "Low": "\u25b3", "Missing": "\u2716"}[self.label]


@dataclass
class SectionScore:
    section_id:        str
    title:              str
    coverage:            float                  # C(s)
    applicability:         float                # A(s)
    band:                   str                  # Strong | Partial | Weak | Missing
    element_matches:        list[ElementMatch] = field(default_factory=list)


def load_ontology(path: str = ONTOLOGY_PATH) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def entities_to_facts(entities: dict, doc_type: str, document_id: str = "") -> list[EvidenceFact]:
    """Converts Phase 1's ExtractedEntities.__dict__ (as already stored in
    case chunk metadata) into EvidenceFact objects ALEA can score against.
    Called once per case chunk's metadata, results aggregated by the caller."""
    facts = []

    for w in entities.get("weapons", []):
        facts.append(EvidenceFact(text=f"weapon used: {w}", fact_type="weapon",
                                   source_doc_type=doc_type, source_document_id=document_id))
    for i in entities.get("injuries", []):
        facts.append(EvidenceFact(text=f"injury: {i}", fact_type="injury",
                                   source_doc_type=doc_type, source_document_id=document_id))
    for a in entities.get("amounts", []):
        facts.append(EvidenceFact(text=f"amount/property involved: {a}", fact_type="amount",
                                   source_doc_type=doc_type, source_document_id=document_id))
    if entities.get("complainants"):
        facts.append(EvidenceFact(
            text=f"complainant: {', '.join(entities['complainants'])}", fact_type="party",
            source_doc_type=doc_type, source_document_id=document_id))
    if entities.get("accused"):
        facts.append(EvidenceFact(
            text=f"accused: {', '.join(entities['accused'])}", fact_type="party",
            source_doc_type=doc_type, source_document_id=document_id))

    return facts


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def _band(coverage: float) -> str:
    if coverage <= 0:
        return "Missing"
    if coverage >= 0.75:
        return "Strong"
    if coverage >= 0.40:
        return "Partial"
    return "Weak"


class ALEA:
    """embed_fn: Callable[[list[str]], np.ndarray] — pass an already-loaded
    model's .encode method (e.g. CaseIndexer.embed_model.encode). ALEA never
    loads its own model."""

    def __init__(self, embed_fn, ontology_path: str = ONTOLOGY_PATH):
        self.embed_fn = embed_fn
        self.ontology = load_ontology(ontology_path)

    def score_sections(
        self,
        evidence_facts:  list[EvidenceFact],
        candidate_section_ids: list[str] | None = None,
        retrieval_priors: dict[str, float] | None = None,
    ) -> list[SectionScore]:
        """Scores each candidate section (default: every section in the
        ontology) against the given evidence facts. retrieval_priors is an
        optional {section_id: 0-1 float} map — omit to treat every section's
        prior as neutral (1.0), which reduces A(s) to C(s) alone."""
        retrieval_priors = retrieval_priors or {}
        candidate_ids = candidate_section_ids or list(self.ontology.keys())
        candidate_ids = [sid for sid in candidate_ids if sid in self.ontology]

        if not evidence_facts or not candidate_ids:
            return []

        fact_texts  = [f.text for f in evidence_facts]
        fact_vecs   = np.array(self.embed_fn(fact_texts))
        # BUGFIX: c(ri) is documented (module docstring) as
        # max_e[w(e) * sim(ri,e)] — the fact that maximizes the WEIGHTED
        # product, not the fact with the highest raw similarity. The
        # argmax below used to run on `sims` alone, picking the most
        # similar fact FIRST and only multiplying by ITS weight afterward
        # — a different, lower-quality computation whenever a slightly
        # less-similar but more reliable fact (e.g. a forensic report,
        # weight 0.9) should have outscored a highly-similar but
        # low-reliability one (e.g. a witness statement, weight 0.3): the
        # old code always picked the witness statement in that case,
        # understating confidence by ~3x in a concrete worked example.
        # Weighting fact_vecs' similarity row by fact weight before the
        # argmax fixes this directly.
        fact_weights = np.array([f.weight for f in evidence_facts])

        results = []
        for section_id in candidate_ids:
            section = self.ontology[section_id]
            elements = section["elements"]

            elem_texts = [e["description"] for e in elements]
            elem_vecs  = np.array(self.embed_fn(elem_texts))

            sim_matrix = _cosine_sim_matrix(elem_vecs, fact_vecs)   # [n_elements x n_facts]
            weighted_matrix = sim_matrix * fact_weights[np.newaxis, :]  # w(e) * sim(ri,e)

            matches = []
            for i, elem in enumerate(elements):
                sims = sim_matrix[i]
                # Restrict the weighted argmax to facts that clear
                # SIM_THRESHOLD on their own similarity first. Without this,
                # a low-similarity-but-high-weight fact could still win the
                # weighted argmax (weight can outweigh a sim gap) and then
                # fail ITS OWN threshold check below — masking a different,
                # lower-weight fact that genuinely was similar enough and
                # would otherwise have produced a valid (if smaller)
                # confidence instead of a hard zero.
                above_threshold = np.where(sims >= SIM_THRESHOLD)[0]
                if len(above_threshold) == 0:
                    best_idx = int(np.argmax(sims))   # for reporting best_sim only
                    matches.append(ElementMatch(
                        element_id=elem["id"], element_description=elem["description"],
                        confidence=0.0, best_evidence=None, best_sim=float(sims[best_idx]),
                    ))
                    continue

                best_idx = int(above_threshold[np.argmax(weighted_matrix[i][above_threshold])])
                best_sim = float(sims[best_idx])

                best_fact = evidence_facts[best_idx]
                confidence = best_fact.weight * best_sim   # c(ri) = max[w(e) x sim(ri,e)]
                matches.append(ElementMatch(
                    element_id=elem["id"], element_description=elem["description"],
                    confidence=confidence, best_evidence=best_fact, best_sim=best_sim,
                ))

            coverage = sum(m.confidence for m in matches) / len(matches)   # C(s)
            prior = retrieval_priors.get(section_id, 1.0)
            applicability = coverage * prior                                # A(s)

            results.append(SectionScore(
                section_id=section_id, title=section["title"],
                coverage=coverage, applicability=applicability,
                band=_band(coverage), element_matches=matches,
            ))

        results.sort(key=lambda r: r.applicability, reverse=True)
        return results


def summarize_coverage(score: SectionScore) -> str:
    """Phase 4 — the plain-language verdict, generated deterministically
    from the element match labels (not free LLM generation), so the
    conclusion is always traceable back to the actual scores rather than
    the model's own independent judgment."""
    total = len(score.element_matches)
    if total == 0:
        return f"No legal elements defined for {score.section_id} to evaluate against."

    strong = [m for m in score.element_matches if m.label == "High"]
    partial = [m for m in score.element_matches if m.label in ("Medium", "Low")]
    missing = [m for m in score.element_matches if m.label == "Missing"]

    parts = [f"Available evidence strongly supports {len(strong)} of {total} "
             f"legal requirements for {score.title} ({score.section_id})."]

    if partial:
        names = ", ".join(m.element_id.replace("_", " ") for m in partial)
        parts.append(f"{names.capitalize()} evidence is limited.")
    if missing:
        names = ", ".join(m.element_id.replace("_", " ") for m in missing)
        parts.append(f"{names.capitalize()} evidence is missing.")
    if not partial and not missing:
        parts.append("All required elements are well supported.")

    return " ".join(parts)


def evidence_coverage_report(scores: list[SectionScore], top_n_alternatives: int = 3) -> str:
    """Phase 4's actual user-facing output: leads with the single
    best-matching section's full table + verdict, then a compact
    comparison line for the next few alternatives — rather than printing
    N full tables indiscriminately, which buries the answer to 'is there
    enough evidence for X' in noise."""
    if not scores:
        return "(no candidate sections scored — no evidence facts or ontology entries available)"

    top = scores[0]
    lines = [format_coverage_table([top], top_n=1), "", summarize_coverage(top)]

    alternatives = scores[1:1 + top_n_alternatives]
    if alternatives:
        lines.append("\nOther candidate sections considered:")
        for alt in alternatives:
            lines.append(f"  {alt.section_id} — {alt.title}  [{alt.band}, coverage={alt.coverage:.2f}]")

    return "\n".join(lines)


def format_coverage_table(scores: list[SectionScore], top_n: int = 5) -> str:
    """Phase 4 — Evidence Coverage Analysis (table view). This is a direct,
    deterministic byproduct of ALEA's own match trace, not a separate
    generation step."""
    if not scores:
        return "(no candidate sections scored — no evidence facts or ontology entries available)"

    lines = []
    for score in scores[:top_n]:
        lines.append(f"\n{score.section_id} — {score.title}  [{score.band}, coverage={score.coverage:.2f}]")
        lines.append(f"{'Legal requirement':<28}{'Supporting evidence':<45}{'Confidence'}")
        lines.append("-" * 90)
        for m in score.element_matches:
            evidence_str = m.best_evidence.text if m.best_evidence else "\u2014"
            lines.append(f"{m.mark} {m.element_description[:24]:<26}{evidence_str[:43]:<45}{m.label}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test with a fake embed_fn (word-overlap over a SHARED, fixed
    # vocabulary — matching how a real embedding model behaves: fixed
    # output size no matter when/what it's called with) — just exercises
    # the scoring math end to end without needing a real model.
    facts = [
        EvidenceFact(text="injury: stab wound", fact_type="injury", source_doc_type="Medical Report"),
        EvidenceFact(text="weapon used: knife", fact_type="weapon", source_doc_type="FIR"),
        EvidenceFact(text="accused: Vikram Malhotra", fact_type="party", source_doc_type="FIR"),
    ]

    alea = ALEA(embed_fn=None)  # embed_fn set below once vocab is known
    candidate_ids = ["IPC_302", "BNS_115", "IPC_420"]
    all_texts = [f.text for f in facts] + [
        e["description"] for sid in candidate_ids for e in alea.ontology[sid]["elements"]
    ]
    vocab = sorted(set(w for t in all_texts for w in t.lower().split()))

    def fake_embed(texts):
        return [[1.0 if w in set(t.lower().split()) else 0.0 for w in vocab] for t in texts]

    alea.embed_fn = fake_embed
    scores = alea.score_sections(facts, candidate_section_ids=candidate_ids)
    print(format_coverage_table(scores))
