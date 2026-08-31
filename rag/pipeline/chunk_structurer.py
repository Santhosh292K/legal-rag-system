"""
pipeline/chunk_structurer.py
Novel component #3 — Hierarchy-Aware Chunk Structurer

Gap 15 fix: optionally uses Qdrant client for parent/child lookups instead of
loading the 7.8 MB final_dataset.json into memory on every startup. When
`client` is provided, the in-memory JSON index is never loaded — Qdrant
scroll() calls replace the dict lookups. This removes the duplicate data copy.

Gap 16 fix: added `structure_minimal()` and `enrich_chunks()` methods so the
pipeline can run Stage 5 without building the expensive enriched context, and
then only enrich the final top-K chunks AFTER reranking (not all 35+ candidates).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.temporal_filter import ValidatedChunk
from config import JSON_PATH


# ── Structured chunk ──────────────────────────────────────────────────────────

@dataclass
class StructuredChunk:
    section_id:       str
    content:          str
    act_name:         str
    chapter:          str
    category:         str
    validity_label:   str
    warning:          str
    penalized_score:  float
    rule_summary:     str
    issue_tags:       list[str]
    conclusion_type:  str

    # Hierarchy-enriched fields (your novel contribution)
    parent_content:   str        = ""
    parent_id:        str        = ""
    child_summaries:  list[dict] = field(default_factory=list)
    related_contents: list[dict] = field(default_factory=list)
    structural_path:  str        = ""   # e.g. "IPC → Chapter XVII → Section 378"
    hierarchy_depth:  int        = 0

    # Merged context for LLM (parent + self + children)
    enriched_context: str        = ""


# ── Structurer ────────────────────────────────────────────────────────────────

class ChunkStructurer:
    """
    Enriches each retrieved chunk with parent/child/related section context.

    Gap 15: Pass `client` (the shared Qdrant client) to avoid loading the
    7.8 MB final_dataset.json duplicate. When client is provided, all lookups
    are done via Qdrant scroll() using the section_id payload field.

    Gap 16: Use structure_minimal() + enrich_chunks() to defer the expensive
    parent/child context assembly until AFTER reranking, so only the final
    FINAL_TOP_K chunks pay the enrichment cost.
    """

    def __init__(
        self,
        json_path: str = JSON_PATH,
        client=None,
        collection_name: str = "legal_sections",
    ):
        self._client = client
        self._collection = collection_name
        self._index: dict[str, dict] = {}

        if client is None:
            # Fallback: load JSON when no Qdrant client is available
            self._load(json_path)
        else:
            print("ChunkStructurer: using Qdrant client for parent/child lookups "
                  "(Gap 15 — no JSON loaded into memory).")

    def _load(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            self._index[r["section"]] = r
        size_mb = sum(len(str(r)) for r in records) / 1_000_000
        print(f"ChunkStructurer: loaded {len(self._index)} sections "
              f"({size_mb:.1f} MB) into lookup index.")

    def _get(self, section_id: str) -> dict | None:
        """Fetch section record — from Qdrant if client available, else in-memory."""
        if self._client is not None:
            return self._get_from_qdrant(section_id)
        record = self._index.get(section_id)
        if record is None:
            record = self._index.get(section_id.strip())
        return record

    def _get_from_qdrant(self, section_id: str) -> dict | None:
        """Gap 15: fetch section record from Qdrant payload by section_id."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            results, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(must=[
                    FieldCondition(key="section_id", match=MatchValue(value=section_id))
                ]),
                limit=1,
                with_payload=True,
            )
            if not results:
                return None
            p = results[0].payload or {}
            # Reconstruct a record compatible with the JSON format that the rest
            # of this file expects — only the fields actually used here.
            return {
                "section": p.get("section_id", section_id),
                "content": p.get("content", ""),
                "meta": {
                    "hierarchy": {
                        "act":            p.get("act_name", ""),
                        "part":           None,
                        "chapter":        p.get("chapter", ""),
                        "section":        p.get("section_number", ""),
                        "sub_section":    None,
                        "proviso":        None,
                        "parent_section": p.get("parent_section", ""),
                        "child_sections": p.get("child_sections", []),
                    },
                    "chapter":          p.get("chapter", ""),
                    "category":         p.get("category", ""),
                    "related_sections": p.get("related_sections", []),
                    "irac": {
                        "rule_summary":   p.get("rule_summary", ""),
                        "issue_tags":     p.get("issue_tags", []),
                        "conclusion_type":p.get("conclusion_type", ""),
                    },
                    # BUGFIX: this reconstructed record was missing
                    # meta.temporal entirely — the in-memory (final_dataset.
                    # json) fallback path below has it nested exactly like
                    # this, but this Qdrant path (the one actually used at
                    # runtime, since main.py always passes a shared client)
                    # silently dropped it. legal_kg.py's kg_augment_ranked
                    # reads meta["temporal"]["enacted_year"] off records
                    # fetched through this exact method to decide whether a
                    # KG-expanded section is chronologically eligible for a
                    # cutoff_year query — without this it always evaluated
                    # to None and the check quietly did nothing.
                    "temporal": {
                        "enacted_year":   p.get("enacted_year"),
                        "effective_date": p.get("effective_date") or None,
                        "status":         p.get("status", "active"),
                        "superseded_by":  p.get("superseded_by") or None,
                    },
                },
            }
        except Exception:
            return None

    def _build_path(self, record: dict) -> str:
        h    = record["meta"]["hierarchy"]
        parts = [
            h.get("act", ""),
            h.get("part") or "",
            h.get("chapter") or "",
            f"Section {h.get('section', '')}",
            f"Sub-section {h.get('sub_section')}" if h.get("sub_section") else "",
        ]
        return " → ".join(p for p in parts if p)

    def _build_enriched_context(
        self,
        section_id:     str,
        content:        str,
        parent_content: str,
        child_summaries: list[dict],
    ) -> str:
        parts = []
        if parent_content:
            parts.append(f"[Parent Section]\n{parent_content}")
        parts.append(f"[Current Section: {section_id}]\n{content}")
        if child_summaries:
            child_text = "\n".join(
                f"  - {c['section_id']}: {c['summary']}"
                for c in child_summaries[:3]
            )
            parts.append(f"[Related Sub-sections]\n{child_text}")
        return "\n\n".join(parts)

    # ── Gap 16: minimal structure (no enrichment) ─────────────────────────────

    def structure_minimal(
        self,
        validated_chunks: list[ValidatedChunk],
    ) -> list[StructuredChunk]:
        """Gap 16: build StructuredChunks WITHOUT parent/child enriched_context.
        Fast pass used before reranking so we don't pay enrichment cost for
        chunks that will be discarded by the reranker. Call enrich_chunks() on
        the post-reranked top-K list to add the full hierarchy context."""
        structured = []
        for vc in validated_chunks:
            chunk  = vc.chunk
            record = self._get(chunk.section_id)
            if not record:
                sc = StructuredChunk(
                    section_id      = chunk.section_id,
                    content         = chunk.content,
                    act_name        = chunk.payload.get("act_name", ""),
                    chapter         = chunk.chapter,
                    category        = chunk.category,
                    validity_label  = vc.validity_label,
                    warning         = vc.warning,
                    penalized_score = vc.penalized_score,
                    rule_summary    = chunk.rule_summary,
                    issue_tags      = chunk.issue_tags,
                    conclusion_type = chunk.conclusion_type,
                    enriched_context= chunk.content,
                )
                structured.append(sc)
                continue

            meta = record["meta"]
            hier = meta["hierarchy"]
            path  = self._build_path(record)
            depth = sum(1 for x in [
                hier.get("part"), hier.get("chapter"),
                hier.get("section"), hier.get("sub_section"), hier.get("proviso")
            ] if x)

            sc = StructuredChunk(
                section_id       = chunk.section_id,
                content          = chunk.content,
                act_name         = hier.get("act", ""),
                chapter          = meta.get("chapter") or "",
                category         = meta.get("category", ""),
                validity_label   = vc.validity_label,
                warning          = vc.warning,
                penalized_score  = vc.penalized_score,
                rule_summary     = meta["irac"].get("rule_summary") or "",
                issue_tags       = meta["irac"].get("issue_tags") or [],
                conclusion_type  = meta["irac"].get("conclusion_type") or "",
                parent_id        = hier.get("parent_section") or "",
                structural_path  = path,
                hierarchy_depth  = depth,
                enriched_context = chunk.content,   # minimal — no parent/child yet
            )
            structured.append(sc)
        return structured

    def enrich_chunks(
        self,
        chunks:          list[StructuredChunk],
        include_parent:  bool = True,
        include_children:bool = True,
        include_related: bool = True,
        max_related:     int  = 2,
    ) -> list[StructuredChunk]:
        """Gap 16: add parent/child/related context to an already-structured chunk
        list. Call this AFTER reranking on the final top-K list only."""
        for sc in chunks:
            record = self._get(sc.section_id)
            if not record:
                continue
            meta = record["meta"]
            hier = meta["hierarchy"]

            parent_content = ""
            parent_id      = hier.get("parent_section") or ""
            if include_parent and parent_id:
                parent_rec = self._get(parent_id)
                if parent_rec:
                    parent_content = parent_rec["content"]
            sc.parent_content = parent_content
            sc.parent_id      = parent_id

            child_summaries = []
            if include_children:
                for child_id in (hier.get("child_sections") or [])[:4]:
                    child_rec = self._get(child_id)
                    if child_rec:
                        rule_sum = child_rec["meta"]["irac"].get("rule_summary") or \
                                   child_rec["content"][:100]
                        child_summaries.append({"section_id": child_id, "summary": rule_sum})
            sc.child_summaries = child_summaries

            related_contents = []
            if include_related:
                for rel_id in (meta.get("related_sections") or [])[:max_related]:
                    rel_rec = self._get(rel_id)
                    if rel_rec:
                        related_contents.append({
                            "section_id": rel_id,
                            "content":    rel_rec["content"][:450],
                            "category":   rel_rec["meta"].get("category", ""),
                        })
            sc.related_contents = related_contents

            # Rebuild enriched context with full hierarchy
            sc.enriched_context = self._build_enriched_context(
                sc.section_id, sc.content, parent_content, child_summaries
            )
        return chunks

    # ── Full structure (legacy path — structure + enrich in one call) ─────────

    def structure(
        self,
        validated_chunks: list[ValidatedChunk],
        include_parent:   bool = True,
        include_children: bool = True,
        include_related:  bool = True,
        max_related:      int  = 2,
    ) -> list[StructuredChunk]:
        """Original combined method: structure + immediately enrich all chunks.
        Kept for backward compatibility (ablation variants, standalone use).
        For the live pipeline use structure_minimal() + enrich_chunks() instead
        to implement Gap 16 (context assembled post-reranking)."""
        chunks = self.structure_minimal(validated_chunks)
        return self.enrich_chunks(chunks, include_parent, include_children,
                                  include_related, max_related)
