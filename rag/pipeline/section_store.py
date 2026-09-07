"""
pipeline/section_store.py

An in-memory, read-only index of every statute section, loaded with ONE
Qdrant scroll at startup.

Why
---
Qdrant's embedded/local mode stores everything in a single sqlite table
(``points(id TEXT PRIMARY KEY, point BLOB)``) and has no payload indexes
at all — ``create_payload_index`` is accepted and ignored. Every filtered
``scroll()`` therefore deserializes and scans all 3394 points in Python.

The pipeline used to do exactly that for single-section lookups, on the
hot path, constantly:

  * ChunkStructurer.structure_minimal   — 1 scan per candidate (~35-45)
  * ChunkStructurer.enrich_chunks       — self + parent + up to 4 children
                                          + up to 6 related candidates,
                                          times 20 chunks (~240 scans)
  * HybridRetriever.fetch_by_ids        — 1 scan per pinned section
  * HybridRetriever direct lookup       — 2 scans per act/number match,
                                          per query variant (up to 10)
  * legal_kg / Rocchio                  — more of the same

That is 300+ full-collection scans per query, and it was a large part of
the 178 s p50 latency. The whole corpus is ~4 MB of payload; holding it
in a dict makes every one of those lookups O(1) and costs one scan total.

This also becomes the single place that knows how to resolve a written
reference ("IPC 2") to a real section, so callers stop each carrying
their own copy of the padding rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from data.section_ref import normalize_section_ref, candidates


class SectionStore:
    """Read-only ``section_id -> payload`` index over the statute corpus.

    Build once per process and share it (the pipeline already shares one
    Qdrant client and one embedding model for the same reason).
    """

    def __init__(self, client=None, collection_name: str = "legal_sections",
                 json_path: str | None = None, batch_size: int = 512,
                 verbose: bool = True):
        self._payloads: dict[str, dict] = {}
        # (act_code, section_number_upper) -> section_id, for direct lookups
        # like "IPC 302" without touching Qdrant at all.
        self._by_act_number: dict[tuple[str, str], str] = {}

        if client is not None:
            self._load_from_qdrant(client, collection_name, batch_size)
        elif json_path is not None:
            self._load_from_json(json_path)
        else:
            raise ValueError("SectionStore needs either a Qdrant client or a json_path.")

        self._build_lookup()
        if verbose:
            print(f"SectionStore: {len(self._payloads)} sections indexed in memory "
                  f"({len(self._by_act_number)} act/number keys).")

    # ── Loading ──────────────────────────────────────────────────────────

    def _load_from_qdrant(self, client, collection_name: str, batch_size: int):
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name = collection_name,
                limit           = batch_size,
                offset          = offset,
                with_payload    = True,
                with_vectors    = False,
            )
            for p in points:
                payload = p.payload or {}
                sid = payload.get("section_id")
                if sid:
                    self._payloads[sid] = payload
            if offset is None:
                break

    def _load_from_json(self, json_path: str):
        """Fallback for tests and offline tooling. Produces the same payload
        shape data/indexer.py writes, so callers can't tell the difference."""
        import json
        from data.payload import build_payload, resolve_related_refs
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        known = {r["section"] for r in records}
        for r in records:
            payload = build_payload(r)
            payload["related_sections"], _ = resolve_related_refs(
                payload.get("related_sections_raw", []), known,
            )
            self._payloads[r["section"]] = payload

    def _build_lookup(self):
        for sid, payload in self._payloads.items():
            act = (payload.get("act_code") or "").upper()
            num = str(payload.get("section_number") or "").strip().upper()
            if act and num:
                self._by_act_number.setdefault((act, num), sid)

    # ── Lookup ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._payloads)

    def __contains__(self, section_id: str) -> bool:
        return section_id in self._payloads

    @property
    def section_ids(self) -> set[str]:
        return set(self._payloads)

    def get(self, section_id: str) -> dict | None:
        """Payload for an exact section_id."""
        if not section_id:
            return None
        return self._payloads.get(section_id) or self._payloads.get(section_id.strip())

    def resolve(self, raw_ref: str) -> str | None:
        """Any written reference -> a section_id that actually exists.
        Returns None rather than a plausible-looking guess, so callers can
        drop dangling references instead of propagating phantom ids."""
        return normalize_section_ref(raw_ref, self.section_ids)

    def get_by_ref(self, raw_ref: str) -> dict | None:
        sid = self.resolve(raw_ref)
        return self.get(sid) if sid else None

    def lookup_act_number(self, act_code: str, number: str) -> str | None:
        """'IPC', '302' -> 'IPC_302'. Tries the act/section_number index
        first (authoritative, straight from the payload) and falls back to
        the id-shape candidates, which covers acts whose section_number
        field is written differently from its id."""
        key = (act_code.upper(), str(number).strip().upper())
        if key in self._by_act_number:
            return self._by_act_number[key]
        for cand in candidates(act_code.upper(), number):
            if cand in self._payloads:
                return cand
        return None

    # ── Record view (JSON-dataset shape) ─────────────────────────────────

    def get_record(self, section_id: str) -> dict | None:
        """The nested ``{"section", "content", "meta": {...}}`` shape that
        ChunkStructurer and legal_kg expect, rebuilt from the flat payload.
        Kept here so there is exactly one translation between the two
        shapes rather than one per consumer."""
        p = self.get(section_id)
        if p is None:
            return None
        return {
            "section": p.get("section_id", section_id),
            "content": p.get("content", ""),
            "meta": {
                "code":     p.get("act_code", ""),
                "section":  p.get("section_number", ""),
                "chapter":  p.get("chapter", ""),
                "category": p.get("category", ""),
                "keywords": p.get("keywords", []),
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
                "related_sections": p.get("related_sections", []),
                "irac": {
                    "rule_summary":    p.get("rule_summary", ""),
                    "issue_tags":      p.get("issue_tags", []),
                    "conclusion_type": p.get("conclusion_type", ""),
                },
                "temporal": {
                    "enacted_year":   p.get("enacted_year"),
                    "effective_date": p.get("effective_date") or None,
                    "status":         p.get("status", "active"),
                    "superseded_by":  p.get("superseded_by") or None,
                },
            },
        }
