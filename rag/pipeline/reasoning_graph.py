"""
pipeline/reasoning_graph.py
Phase 5 — Legal Reasoning Graph

Replaces the mental model of "here's a pile of retrieved chunks" with an
explicit, explainable chain:

    Question -> Fact -> Evidence -> Legal Element -> Statute -> Judgment -> Conclusion

This is NOT a new retrieval or scoring mechanism — it's a graph view built
on top of what Phase 1 (entity extraction), Phase 3 (ALEA element scoring)
and Track A (statute retrieval / answer generation) already computed. Every
node in the graph traces back to a concrete object already produced
elsewhere in the pipeline, so the graph is a rendering layer, not a new
source of truth.

Two modes, chosen automatically by build_reasoning_graph():

  * Case mode  (alea_scores available, i.e. an uploaded case + evidence):
        Question -> Fact (EvidenceFact.text)
                 -> Evidence (EvidenceFact.source_doc_type/document_id)
                 -> Legal Element (ElementMatch.element_description)
                 -> Statute (SectionScore.section_id/title)
                 -> Judgment (SectionScore.band + coverage)
                 -> Conclusion (summarize_coverage of the best section)

  * Statute-only mode (plain legal question, no case evidence):
        Question -> Fact (the question's own legal requirement)
                 -> Evidence (the retrieved statutory text itself)
                 -> Legal Element (the section's category/rule)
                 -> Statute (Citation.section_id/act_name)
                 -> Judgment (validity: active / repealed / amended)
                 -> Conclusion (the generated answer)

Usage:
    from pipeline.reasoning_graph import build_reasoning_graph
    graph = build_reasoning_graph(query, statute_answer=answer, alea_scores=fused.alea_scores)
    print(graph.to_ascii())
    print(graph.to_mermaid())
    graph.save_html("reasoning_graph.html")
"""
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.alea import SectionScore, EvidenceFact, summarize_coverage

QUESTION_ID   = "Q"
CONCLUSION_ID = "C"

NODE_ORDER = ["question", "fact", "evidence", "legal_element", "statute", "judgment", "conclusion"]

NODE_STYLE = {
    "question":      "#1f2937",  # slate
    "fact":          "#2563eb",  # blue
    "evidence":      "#0891b2",  # cyan
    "legal_element": "#7c3aed",  # violet
    "statute":       "#b45309",  # amber
    "judgment":      "#be123c",  # rose
    "conclusion":    "#15803d",  # green
}


@dataclass
class GraphNode:
    id:    str
    type:  str           # one of NODE_ORDER
    label: str
    data:  dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    label:  str = ""


@dataclass
class ReasoningGraph:
    query: str
    nodes: list = field(default_factory=list)   # list[GraphNode]
    edges: list = field(default_factory=list)   # list[GraphEdge]

    def add_node(self, node_id: str, node_type: str, label: str, **data) -> str:
        if not any(n.id == node_id for n in self.nodes):
            self.nodes.append(GraphNode(id=node_id, type=node_type, label=label, data=data))
        return node_id

    def add_edge(self, source: str, target: str, label: str = ""):
        # Avoid duplicate edges (same fact can legitimately feed two elements,
        # but the exact same source->target pair only needs to be drawn once).
        if not any(e.source == source and e.target == target for e in self.edges):
            self.edges.append(GraphEdge(source=source, target=target, label=label))

    # ── Rendering ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "nodes": [{"id": n.id, "type": n.type, "label": n.label, "data": n.data} for n in self.nodes],
            "edges": [{"source": e.source, "target": e.target, "label": e.label} for e in self.edges],
        }

    def to_ascii(self) -> str:
        """Layer-by-layer text view — one section per node type, in graph
        order, with outgoing edges listed under each node. Good enough for
        a terminal REPL where a real graph render isn't available."""
        by_type = {t: [n for n in self.nodes if n.type == t] for t in NODE_ORDER}
        out_edges = {}
        for e in self.edges:
            out_edges.setdefault(e.source, []).append(e.target)
        label_of = {n.id: n.label for n in self.nodes}

        lines = [f"LEGAL REASONING GRAPH — {self.query}", "=" * 78]
        for t in NODE_ORDER:
            nodes = by_type[t]
            if not nodes:
                continue
            lines.append(f"\n[{t.upper().replace('_', ' ')}]")
            for n in nodes:
                first_line = n.label.splitlines()[0]
                lines.append(f"  ({n.id}) {first_line}")
                for extra in n.label.splitlines()[1:]:
                    lines.append(f"        {extra}")
                targets = out_edges.get(n.id, [])
                for tgt in targets:
                    arrow = f"    -> {label_of.get(tgt, tgt)} ({tgt})"
                    lines.append(arrow)
        lines.append("\n" + "=" * 78)
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Flowchart TD, one classDef per node type for color coding.
        Node ids are sanitized since raw ids may contain characters mermaid
        dislikes; labels are escaped and newlines become <br/>."""
        def safe_id(raw_id: str) -> str:
            return "n_" + re.sub(r"[^A-Za-z0-9_]", "_", raw_id)

        def safe_label(label: str) -> str:
            escaped = label.replace('"', "'").replace("\n", "<br/>")
            return f'"{escaped}"'

        lines = ["flowchart TD"]
        for n in self.nodes:
            lines.append(f'    {safe_id(n.id)}[{safe_label(n.label)}]:::{n.type}')
        for e in self.edges:
            if e.label:
                lines.append(f'    {safe_id(e.source)} -->|{e.label}| {safe_id(e.target)}')
            else:
                lines.append(f'    {safe_id(e.source)} --> {safe_id(e.target)}')
        for node_type, color in NODE_STYLE.items():
            lines.append(f'    classDef {node_type} fill:{color},color:#ffffff,stroke:#00000033;')
        return "\n".join(lines)

    def save_html(self, path: str, title: str = "Legal Reasoning Graph"):
        """Standalone HTML file (mermaid.js via CDN) so the graph can be
        opened directly in a browser instead of squinting at ASCII."""
        mermaid_src = self.to_mermaid()
        doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f8fafc; }}
  h1 {{ font-size: 1.1rem; color: #1f2937; }}
  .mermaid {{ background: white; padding: 1rem; border-radius: 8px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p style="color:#475569">{html.escape(self.query)}</p>
<pre class="mermaid">
{mermaid_src}
</pre>
<script>mermaid.initialize({{ startOnLoad: true }});</script>
</body>
</html>"""
        Path(path).write_text(doc, encoding="utf-8")
        return path


# ── Builders ──────────────────────────────────────────────────────────────

def _fact_key(fact: EvidenceFact) -> tuple:
    return (fact.text, fact.source_document_id)


def _evidence_key(fact: EvidenceFact) -> tuple:
    return (fact.source_doc_type, fact.source_document_id)


def _evidence_label(fact: EvidenceFact) -> str:
    if fact.source_document_id:
        return f"{fact.source_doc_type}\n({fact.source_document_id})"
    return fact.source_doc_type


def _build_case_graph(
    query: str,
    statute_answer,
    alea_scores: list[SectionScore],
    max_statutes: int,
    max_elements_per_statute: int,
) -> ReasoningGraph:
    graph = ReasoningGraph(query=query)
    graph.add_node(QUESTION_ID, "question", query)

    fact_ids     = {}   # _fact_key -> node id
    evidence_ids = {}   # _evidence_key -> node id

    # Prefer non-Missing bands, but if every candidate is Missing, still show
    # the best one or two — "no evidence supports this" is itself explainable.
    ranked = sorted(alea_scores, key=lambda s: s.applicability, reverse=True)
    non_missing = [s for s in ranked if s.band != "Missing"]
    shown = (non_missing or ranked)[:max_statutes]

    best_for_conclusion = shown[0] if shown else None

    for score in shown:
        statute_id = f"ST::{score.section_id}"
        graph.add_node(
            statute_id, "statute", f"{score.section_id}\n{score.title}",
            section_id=score.section_id, title=score.title,
        )

        judgment_id = f"J::{score.section_id}"
        graph.add_node(
            judgment_id, "judgment",
            f"{score.band} match\ncoverage={score.coverage:.2f}",
            band=score.band, coverage=score.coverage, applicability=score.applicability,
        )
        graph.add_edge(statute_id, judgment_id, "assessed as")
        graph.add_edge(judgment_id, CONCLUSION_ID, "feeds into")

        matched = [m for m in score.element_matches if m.confidence > 0][:max_elements_per_statute]
        if not matched:
            # No supporting element found for this statute at all — connect
            # it straight to the question so it isn't left floating.
            graph.add_edge(QUESTION_ID, statute_id, "candidate section")
            continue

        for m in matched:
            elem_id = f"LE::{score.section_id}::{m.element_id}"
            graph.add_node(
                elem_id, "legal_element",
                f"{m.element_description}\n(confidence={m.confidence:.2f})",
                element_id=m.element_id, confidence=m.confidence, strength_label=m.label,
            )
            graph.add_edge(elem_id, statute_id, "element of")

            if m.best_evidence is not None:
                fact = m.best_evidence
                fkey = _fact_key(fact)
                if fkey not in fact_ids:
                    fid = f"F::{len(fact_ids)}"
                    fact_ids[fkey] = fid
                    graph.add_node(fid, "fact", fact.text, fact_type=fact.fact_type)
                    graph.add_edge(QUESTION_ID, fid, "raises")
                fid = fact_ids[fkey]

                ekey = _evidence_key(fact)
                if ekey not in evidence_ids:
                    eid = f"E::{len(evidence_ids)}"
                    evidence_ids[ekey] = eid
                    graph.add_node(eid, "evidence", _evidence_label(fact), weight=fact.weight)
                fid_evidence = evidence_ids[ekey]

                graph.add_edge(fid, fid_evidence, "documented in")
                graph.add_edge(fid_evidence, elem_id, "supports")
            else:
                # Element with no matching evidence — still worth showing as
                # a gap, connected straight from the question.
                graph.add_edge(QUESTION_ID, elem_id, "unsupported element")

    conclusion_label = "No evidence-backed section could be confidently identified."
    if best_for_conclusion is not None:
        conclusion_label = summarize_coverage(best_for_conclusion)
    if statute_answer is not None and getattr(statute_answer, "confidence", None):
        conclusion_label += f"\n(overall answer confidence: {statute_answer.confidence})"
    graph.add_node(CONCLUSION_ID, "conclusion", conclusion_label)

    return graph


def _build_statute_only_graph(
    query: str,
    statute_answer,
    max_statutes: int,
) -> ReasoningGraph:
    graph = ReasoningGraph(query=query)
    graph.add_node(QUESTION_ID, "question", query)

    citations = list(getattr(statute_answer, "citations", None) or [])[:max_statutes]

    if not citations:
        graph.add_node(CONCLUSION_ID, "conclusion",
                        (getattr(statute_answer, "answer", None) or "No relevant sections found.")[:300])
        graph.add_edge(QUESTION_ID, CONCLUSION_ID, "no applicable law found")
        return graph

    fact_id = "F::0"
    graph.add_node(fact_id, "fact", f"Question requires determining the applicable law for:\n{query[:100]}")
    graph.add_edge(QUESTION_ID, fact_id, "raises")

    evidence_id = "E::0"
    graph.add_node(evidence_id, "evidence", "Statutory corpus\n(retrieved sections)")
    graph.add_edge(fact_id, evidence_id, "documented in")

    for c in citations:
        elem_id = f"LE::{c.section_id}"
        graph.add_node(elem_id, "legal_element", c.category or "Applicable rule")
        graph.add_edge(evidence_id, elem_id, "supports")

        statute_id = f"ST::{c.section_id}"
        graph.add_node(statute_id, "statute", f"{c.section_id}\n{c.act_name}")
        graph.add_edge(elem_id, statute_id, "element of")

        judgment_id = f"J::{c.section_id}"
        judgment_label = c.validity if c.validity == "active" else f"{c.validity.upper()}"
        if c.warning:
            judgment_label += f"\n{c.warning}"
        graph.add_node(judgment_id, "judgment", judgment_label, validity=c.validity, warning=c.warning)
        graph.add_edge(statute_id, judgment_id, "assessed as")
        graph.add_edge(judgment_id, CONCLUSION_ID, "feeds into")

    conclusion_label = (getattr(statute_answer, "answer", None) or "")[:300]
    conclusion_label += f"\n(confidence: {getattr(statute_answer, 'confidence', 'n/a')})"
    graph.add_node(CONCLUSION_ID, "conclusion", conclusion_label)

    return graph


def build_reasoning_graph(
    query: str,
    statute_answer=None,
    alea_scores: list[SectionScore] | None = None,
    max_statutes: int = 4,
    max_elements_per_statute: int = 6,
) -> ReasoningGraph:
    """Single entry point. Picks case-mode (fact/evidence/element chain via
    ALEA) when alea_scores are available and non-empty, otherwise falls back
    to statute-only mode built from statute_answer.citations."""
    if alea_scores:
        return _build_case_graph(
            query, statute_answer, alea_scores,
            max_statutes=max_statutes, max_elements_per_statute=max_elements_per_statute,
        )
    return _build_statute_only_graph(query, statute_answer, max_statutes=max_statutes)
