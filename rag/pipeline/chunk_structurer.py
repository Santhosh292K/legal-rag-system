"""
pipeline/chunk_structurer.py
Hierarchy-aware chunk structurer — enriches each retrieved section with its
parent, children and cross-referenced sections.

Lookups go through pipeline/section_store.py (one in-memory scroll of the
corpus), NOT through per-section Qdrant scrolls. In embedded/local mode
Qdrant has no payload indexes, so every filtered scroll is a full scan of
all 3394 points; this class alone used to issue ~35 of them in
structure_minimal and up to ~240 more in enrich_chunks, per query.

Two-phase by design:
  * structure_minimal() — cheap, run on every candidate before reranking.
  * enrich_chunks()     — parent/child/related assembly, run only on the
                          chunks that survived reranking.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.temporal_filter import ValidatedChunk
from pipeline.section_store import SectionStore
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

    # The act CODE ("IPC"), as distinct from act_name ("Indian Penal Code,
    # 1860"). The IRAC reranker compares this against intent.act_hint;
    # matching against act_name instead meant the act bonus never fired
    # for 16 of 18 acts and fired on the wrong act for the other two.
    act_code:         str        = ""

    # Hierarchy-enriched fields
    parent_content:   str        = ""
    parent_id:        str        = ""
    child_summaries:  list[dict] = field(default_factory=list)
    related_contents: list[dict] = field(default_factory=list)
    structural_path:  str        = ""   # "IPC → Chapter XVII → Section 378"
    hierarchy_depth:  int        = 0

    # Merged context handed to the answer generator (parent + self + children)
    enriched_context: str        = ""


# ── Structurer ────────────────────────────────────────────────────────────────

class ChunkStructurer:
    def __init__(
        self,
        json_path: str = JSON_PATH,
        client=None,
        collection_name: str = "legal_sections",
        section_store: SectionStore | None = None,
    ):
        if section_store is not None:
            self.sections = section_store
        elif client is not None:
            self.sections = SectionStore(client=client, collection_name=collection_name)
        else:
            # Offline/test path — same payload shape, read straight from JSON.
            self.sections = SectionStore(json_path=json_path)

    def _get(self, section_id: str) -> dict | None:
        return self.sections.get_record(section_id)

    # ── Context assembly ──────────────────────────────────────────────────

    @staticmethod
    def _build_path(record: dict) -> str:
        h = record["meta"]["hierarchy"]
        parts = [
            h.get("act", ""),
            h.get("part") or "",
            h.get("chapter") or "",
            f"Section {h.get('section', '')}" if h.get("section") else "",
            f"Sub-section {h.get('sub_section')}" if h.get("sub_section") else "",
        ]
        return " → ".join(p for p in parts if p)

    @staticmethod
    def _build_enriched_context(
        section_id:       str,
        content:          str,
        parent_content:   str,
        child_summaries:  list[dict],
        related_contents: list[dict] | None = None,
    ) -> str:
        parts = []
        if parent_content:
            parts.append(f"[Parent Section]\n{parent_content}")
        parts.append(f"[Current Section: {section_id}]\n{content}")
        if child_summaries:
            child_text = "\n".join(
                f"  - {c['section_id']}: {c['summary']}" for c in child_summaries[:3]
            )
            parts.append(f"[Related Sub-sections]\n{child_text}")
        if related_contents:
            related_text = "\n".join(
                f"  - {r['section_id']}: {r['content'][:200]}" for r in related_contents
            )
            parts.append(f"[Related Sections]\n{related_text}")
        return "\n\n".join(parts)

    # ── Minimal structure (pre-rerank) ────────────────────────────────────

    def structure_minimal(
        self,
        validated_chunks: list[ValidatedChunk],
    ) -> list[StructuredChunk]:
        """StructuredChunks WITHOUT parent/child context. Cheap enough to
        run on every candidate; call enrich_chunks() on the survivors."""
        structured = []
        for vc in validated_chunks:
            chunk  = vc.chunk
            record = self._get(chunk.section_id)

            if not record:
                structured.append(StructuredChunk(
                    section_id      = chunk.section_id,
                    content         = chunk.content,
                    act_name        = chunk.payload.get("act_name", ""),
                    act_code        = chunk.act_code or chunk.payload.get("act_code", ""),
                    chapter         = chunk.chapter,
                    category        = chunk.category,
                    validity_label  = vc.validity_label,
                    warning         = vc.warning,
                    penalized_score = vc.penalized_score,
                    rule_summary    = chunk.rule_summary,
                    issue_tags      = chunk.issue_tags,
                    conclusion_type = chunk.conclusion_type,
                    enriched_context= chunk.content,
                ))
                continue

            meta = record["meta"]
            hier = meta["hierarchy"]
            depth = sum(1 for x in (
                hier.get("part"), hier.get("chapter"),
                hier.get("section"), hier.get("sub_section"), hier.get("proviso"),
            ) if x)

            structured.append(StructuredChunk(
                section_id       = chunk.section_id,
                content          = chunk.content or record.get("content", ""),
                act_name         = hier.get("act", ""),
                act_code         = meta.get("code", "") or chunk.act_code,
                chapter          = meta.get("chapter") or "",
                category         = meta.get("category", ""),
                validity_label   = vc.validity_label,
                warning          = vc.warning,
                penalized_score  = vc.penalized_score,
                rule_summary     = meta["irac"].get("rule_summary") or "",
                issue_tags       = meta["irac"].get("issue_tags") or [],
                conclusion_type  = meta["irac"].get("conclusion_type") or "",
                parent_id        = hier.get("parent_section") or "",
                structural_path  = self._build_path(record),
                hierarchy_depth  = depth,
                enriched_context = chunk.content or record.get("content", ""),
            ))
        return structured

    # ── Full enrichment (post-rerank) ─────────────────────────────────────

    def enrich_chunks(
        self,
        chunks:           list[StructuredChunk],
        include_parent:   bool = True,
        include_children: bool = True,
        include_related:  bool = True,
        max_related:      int  = 2,
    ) -> list[StructuredChunk]:
        for sc in chunks:
            record = self._get(sc.section_id)
            if not record:
                continue
            meta = record["meta"]
            hier = meta["hierarchy"]

            parent_content = ""
            parent_id      = hier.get("parent_section") or ""
            if include_parent and parent_id:
                parent_rec = self._get(parent_id) or self.sections.get_record(
                    self.sections.resolve(parent_id) or ""
                )
                if parent_rec:
                    parent_content = parent_rec["content"]
            sc.parent_content = parent_content
            sc.parent_id      = parent_id

            child_summaries = []
            if include_children:
                for child_id in (hier.get("child_sections") or [])[:4]:
                    child_rec = self._get(child_id)
                    if child_rec is None:
                        resolved = self.sections.resolve(child_id)
                        child_rec = self._get(resolved) if resolved else None
                    if child_rec:
                        child_summaries.append({
                            "section_id": child_rec.get("section", child_id),
                            "summary": (child_rec["meta"]["irac"].get("rule_summary")
                                        or child_rec["content"][:100]),
                        })
            sc.child_summaries = child_summaries

            related_contents = []
            if include_related:
                # related_sections are normalized to real section_ids at
                # index time (data/indexer.py), so these resolve directly.
                # They used to arrive as "IPC 2" against an exact-match
                # lookup for "IPC_002" and silently returned nothing, on
                # every chunk, on every query.
                for rel_id in (meta.get("related_sections") or [])[:max_related]:
                    rel_rec = self._get(rel_id)
                    if rel_rec is None:
                        resolved = self.sections.resolve(rel_id)
                        rel_rec = self._get(resolved) if resolved else None
                    if rel_rec:
                        related_contents.append({
                            "section_id": rel_rec.get("section", rel_id),
                            "content":    rel_rec["content"][:450],
                            "category":   rel_rec["meta"].get("category", ""),
                        })
            sc.related_contents = related_contents

            sc.enriched_context = self._build_enriched_context(
                sc.section_id, sc.content, parent_content,
                child_summaries, related_contents,
            )
        return chunks

    # ── Combined (tests / ablations) ──────────────────────────────────────

    def structure(
        self,
        validated_chunks: list[ValidatedChunk],
        include_parent:   bool = True,
        include_children: bool = True,
        include_related:  bool = True,
        max_related:      int  = 2,
    ) -> list[StructuredChunk]:
        chunks = self.structure_minimal(validated_chunks)
        return self.enrich_chunks(chunks, include_parent, include_children,
                                  include_related, max_related)
