"""
pipeline/legal_kg.py
Legal Knowledge Graph — novel component for the research paper.

Builds a typed directed graph over all sections in the Qdrant corpus.
Edge types:
  PARENT_OF    — statutory hierarchy (Part → Chapter → Section)
  CHILD_OF     — inverse of PARENT_OF
  RELATES_TO   — cross-reference between sections (bidirectional)
  READ_WITH    — sections frequently cited together in legal practice
  SUPERSEDES   — BNS/BNSS/BSA sections that replace IPC/CrPC/IEA equivalents
  SAME_ACT     — sections in the same act (weak structural link)

Uses NetworkX for the graph. No extra infrastructure needed.
The graph is built lazily from the Qdrant payload — the same data pipeline
already stores parent_section, child_sections, related_sections per section.

Research paper contribution:
  Standard RAG retrieves sections independently. The KG enables:
  1. Seed expansion — retrieved sections pull in their typed neighbors
  2. "Read with" chains — IPC 302 → IPC 34 (common intention) automatically
  3. Supersession awareness — querying IPC 302 also surfaces BNS 103
  4. Multi-hop traversal — surface sections 2 hops away that BM25 + dense missed

Usage:
    kg = LegalKnowledgeGraph()
    kg.build_from_qdrant(client, collection_name)
    expanded = kg.expand(seed_ids=["IPC_302"], hops=1,
                         edge_types={"SUPERSEDES", "READ_WITH", "RELATES_TO"})
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

sys.path.append(str(Path(__file__).parent.parent))
from config import JSON_PATH
from data.section_ref import normalize_section_ref

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


# ── Edge types ────────────────────────────────────────────────────────────────

EDGE_PARENT_OF  = "PARENT_OF"
EDGE_CHILD_OF   = "CHILD_OF"
EDGE_RELATES_TO = "RELATES_TO"
EDGE_READ_WITH  = "READ_WITH"
EDGE_SUPERSEDES = "SUPERSEDES"
EDGE_SAME_ACT   = "SAME_ACT"

# IPC → BNS supersession map (manually curated domain knowledge).
# Key  = old IPC section_id
# Value = new BNS section_id that replaced it
IPC_TO_BNS: dict[str, str] = {
    "IPC_299": "BNS_100",
    "IPC_300": "BNS_101",
    "IPC_302": "BNS_103",
    "IPC_304": "BNS_105",
    "IPC_304A": "BNS_106",
    "IPC_304B": "BNS_107",
    "IPC_306": "BNS_109",
    "IPC_307": "BNS_109",
    "IPC_308": "BNS_110",
    "IPC_323": "BNS_115",
    "IPC_324": "BNS_116",
    "IPC_325": "BNS_117",
    "IPC_326": "BNS_118",
    "IPC_326A": "BNS_124",
    "IPC_326B": "BNS_125",
    "IPC_341": "BNS_126",
    "IPC_342": "BNS_127",
    "IPC_343": "BNS_128",
    "IPC_344": "BNS_129",
    "IPC_354": "BNS_74",
    "IPC_354A": "BNS_75",
    "IPC_354B": "BNS_76",
    "IPC_363": "BNS_137",
    "IPC_364": "BNS_138",
    "IPC_364A": "BNS_140",
    "IPC_376": "BNS_63",
    "IPC_376A": "BNS_65",
    "IPC_376C": "BNS_68",
    "IPC_378": "BNS_303",
    "IPC_379": "BNS_304",
    "IPC_380": "BNS_305",
    "IPC_382": "BNS_307",
    "IPC_383": "BNS_308",
    "IPC_384": "BNS_309",
    "IPC_386": "BNS_311",
    "IPC_390": "BNS_313",
    "IPC_391": "BNS_310",
    "IPC_392": "BNS_309",
    "IPC_395": "BNS_310",
    "IPC_397": "BNS_310",
    "IPC_399": "BNS_311",
    "IPC_405": "BNS_316",
    "IPC_406": "BNS_316",
    "IPC_407": "BNS_317",
    "IPC_408": "BNS_318",
    "IPC_415": "BNS_318",
    "IPC_416": "BNS_319",
    "IPC_419": "BNS_319",
    "IPC_420": "BNS_318",
    "IPC_441": "BNS_329",
    "IPC_447": "BNS_329",
    "IPC_463": "BNS_334",
    "IPC_467": "BNS_336",
    "IPC_468": "BNS_337",
    "IPC_471": "BNS_340",
    "IPC_498A": "BNS_85",
    "IPC_499": "BNS_356",
    "IPC_500": "BNS_356",
    "IPC_505": "BNS_353",
    "IPC_509": "BNS_79",
}

# "Read with" pairs — sections frequently co-cited in Indian legal practice.
# (src, dst, note) — constructed from legal domain knowledge.
READ_WITH_PAIRS: list[tuple[str, str, str]] = [
    ("IPC_302", "IPC_300",  "murder requires proof of culpable homicide"),
    ("IPC_302", "IPC_34",   "common intention — co-accused liability"),
    ("IPC_302", "IPC_120B", "criminal conspiracy to murder"),
    ("IPC_304", "IPC_299",  "culpable homicide definition"),
    ("IPC_304A", "IPC_279", "rash driving causing death"),
    ("IPC_376", "POCSO_004","rape + POCSO when victim is minor"),
    ("IPC_498A", "IPC_304B","dowry cruelty + dowry death"),
    ("IPC_420", "IPC_468",  "cheating + forgery for cheating"),
    ("IPC_420", "IPC_467",  "cheating + forgery of valuable security"),
    ("IPC_307", "IPC_324",  "attempt to murder + hurt by dangerous means"),
    ("IPC_395", "IPC_397",  "dacoity + dacoity with violence"),
    ("IPC_441", "TPA_108",  "criminal trespass + landlord-tenant rights"),
    ("IPC_166", "PCA_007",  "public servant disobeying law + bribery"),
    ("PCA_007", "PCA_013",  "taking gratification + criminal misconduct"),
    ("ITA_066", "IPC_420",  "hacking + cheating"),
    ("ITA_066C","ITA_066D", "identity theft + cheating by personation"),
    ("POCSO_004","POCSO_005","penetrative assault + aggravated assault"),
    ("SCST_003", "IPC_447", "atrocity + criminal trespass on tribal land"),
    ("ICA_015", "ICA_019",  "coercion definition + voidable contract"),
    ("SRA_010", "SRA_014",  "specific performance + who can obtain it"),
    ("IPC_363", "IPC_364A", "kidnapping + kidnapping for ransom"),
    ("CRPC_438","CRPC_439", "anticipatory bail + bail by Sessions Court"),
    ("IEA_062", "IEA_064",  "primary evidence + secondary evidence"),
    ("IPC_153A","IPC_505",  "promoting enmity + statements causing disharmony"),
    ("IPC_326A","IPC_307",  "acid attack + attempt to murder"),
]


# ── KG data classes ──────────────────────────────────────────────────────────

@dataclass
class KGNode:
    section_id:    str
    act_name:      str  = ""
    chapter:       str  = ""
    title:         str  = ""
    status:        str  = "active"   # active / repealed / amended


@dataclass
class KGEdge:
    src:       str
    dst:       str
    edge_type: str
    note:      str  = ""


@dataclass
class KGExpansionResult:
    seed_ids:        list[str]
    expanded_ids:    list[str]          # new sections (not in seed)
    edges_traversed: list[KGEdge]
    hop_depth:       int


# ── Knowledge Graph ───────────────────────────────────────────────────────────

class LegalKnowledgeGraph:
    """
    Lightweight typed directed graph over the legal corpus.

    Build once at pipeline init time and reuse. Thread-safe for reads.

    Build workflow:
        kg = LegalKnowledgeGraph()
        kg.build_from_qdrant(client, collection_name="legal_sections")
        # − or −
        kg.build_from_json("./data/final_dataset.json")
    """

    def __init__(self):
        if not _NX_AVAILABLE:
            raise ImportError(
                "networkx is required for LegalKnowledgeGraph. "
                "Install it with: pip install networkx"
            )
        self.graph: nx.DiGraph = nx.DiGraph()
        self._built = False
        self._known_ids: set[str] = set()

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_from_qdrant(
        self,
        client,
        collection_name: str = "legal_sections",
        batch_size:      int = 100,
    ) -> "LegalKnowledgeGraph":
        """
        Populate the graph from Qdrant payloads.
        Adds structural edges (PARENT_OF, CHILD_OF, RELATES_TO, SAME_ACT)
        then overlays domain-knowledge edges (SUPERSEDES, READ_WITH).
        """
        print("[LegalKG] Building graph from Qdrant ...")
        offset = None
        all_records = []

        while True:
            results, next_offset = client.scroll(
                collection_name = collection_name,
                limit           = batch_size,
                offset          = offset,
                with_payload    = True,
                with_vectors    = False,
            )
            all_records.extend(results)
            if next_offset is None:
                break
            offset = next_offset

        self._add_records(all_records)
        print(f"[LegalKG] Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        self._built = True
        return self

    def build_from_section_store(self, store) -> "LegalKnowledgeGraph":
        """Build from an already-loaded pipeline.section_store.SectionStore.

        Preferred over build_from_qdrant() inside the running pipeline: the
        store has already scrolled the whole collection once, so this
        avoids a second full pass over every point at startup.
        """
        class _Stub:
            __slots__ = ("payload",)
            def __init__(self, payload): self.payload = payload

        print("[LegalKG] Building graph from the shared SectionStore ...")
        self._add_records([_Stub(store.get(sid)) for sid in store.section_ids])
        print(f"[LegalKG] Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        self._built = True
        return self

    def build_from_json(
        self,
        json_path: str = JSON_PATH,
    ) -> "LegalKnowledgeGraph":
        """Fallback: build from the dataset JSON when no Qdrant client is
        available (tests, offline tooling). Goes through the same payload
        builder data/indexer.py uses, so the graph is identical either way
        — the previous version hand-rolled a second, subtly different
        payload shape whose keys (e.g. "temporal_status") did not match
        what _add_records actually reads."""
        import json
        from data.payload import build_payload, resolve_related_refs

        print(f"[LegalKG] Building graph from {json_path} ...")
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
        known = {r["section"] for r in records}

        class _Stub:
            __slots__ = ("payload",)
            def __init__(self, payload): self.payload = payload

        stubs = []
        for r in records:
            payload = build_payload(r)
            payload["related_sections"], _ = resolve_related_refs(
                payload.get("related_sections_raw", []), known,
            )
            stubs.append(_Stub(payload))

        self._add_records(stubs)
        print(f"[LegalKG] Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        self._built = True
        return self

    def _add_records(self, records):
        """Populate the graph from Qdrant points (or JSON stubs).

        Every edge endpoint is checked against the set of sections that
        actually exist before the edge is added. networkx's add_edge()
        creates missing nodes silently, and the previous version relied on
        that while feeding it raw cross-references — the dataset writes
        those as "IPC 2" while ids are "IPC_002", so 2414 of 5808 nodes
        (41%) were phantoms that no Qdrant lookup could ever resolve.
        Expansion then filled its result cap with those unresolvable ids
        and the READ_WITH/SUPERSEDES edges the graph exists for never made
        it into the output at all.
        """
        payload_by_id: dict[str, dict] = {}
        for point in records:
            p = point.payload or {}
            sid = p.get("section_id", "")
            if sid:
                payload_by_id[sid] = p

        known = set(payload_by_id)
        self._known_ids = known

        def _resolve(ref: str) -> str | None:
            if not ref:
                return None
            if ref in known:
                return ref
            return normalize_section_ref(ref, known)

        for sid, p in payload_by_id.items():
            self.graph.add_node(
                sid,
                act_name = p.get("act_name", ""),
                chapter  = p.get("chapter", ""),
                # BUGFIX: the payload key is "status"; this read
                # "temporal_status", which no payload has, so every node's
                # status silently defaulted to "active" — including the
                # repealed and struck-down ones.
                status   = p.get("status", "active"),
                title    = (p.get("rule_summary", "") or "")[:80],
                act_code = p.get("act_code", ""),
            )

        for sid, p in payload_by_id.items():
            parent = _resolve(p.get("parent_section", ""))
            if parent and parent != sid:
                self.graph.add_edge(parent, sid, type=EDGE_CHILD_OF)
                self.graph.add_edge(sid, parent, type=EDGE_PARENT_OF)

            for child in (p.get("child_sections") or []):
                child_id = _resolve(child)
                if child_id and child_id != sid:
                    self.graph.add_edge(sid, child_id, type=EDGE_CHILD_OF)
                    self.graph.add_edge(child_id, sid, type=EDGE_PARENT_OF)

            for rel in (p.get("related_sections") or []):
                rel_id = _resolve(rel)
                if rel_id and rel_id != sid:
                    self.graph.add_edge(sid, rel_id, type=EDGE_RELATES_TO)
                    self.graph.add_edge(rel_id, sid, type=EDGE_RELATES_TO)

        # Overlay SUPERSEDES (IPC -> BNS domain knowledge). Both endpoints
        # are resolved because the hardcoded map was written with unpadded
        # ids for 8 of its 60 entries (BNS_74, BNS_85, ...) while the
        # dataset stores BNS_074/BNS_085 — those supersession links, which
        # cover the sexual-offence and dowry-cruelty group, were dropped.
        for ipc_ref, bns_ref in IPC_TO_BNS.items():
            ipc_id, bns_id = _resolve(ipc_ref), _resolve(bns_ref)
            if ipc_id and bns_id:
                self.graph.add_edge(bns_id, ipc_id, type=EDGE_SUPERSEDES,
                                    note="BNS replaces IPC")
                self.graph.add_edge(ipc_id, bns_id, type=EDGE_SUPERSEDES,
                                    note="IPC superseded by BNS")

        # Overlay READ_WITH (domain knowledge pairs), same resolution —
        # this list referenced "IPC_34", which is stored as "IPC_034".
        for src_ref, dst_ref, note in READ_WITH_PAIRS:
            src_id, dst_id = _resolve(src_ref), _resolve(dst_ref)
            if src_id and dst_id:
                self.graph.add_edge(src_id, dst_id, type=EDGE_READ_WITH, note=note)
                self.graph.add_edge(dst_id, src_id, type=EDGE_READ_WITH, note=note)

    def unresolved_nodes(self) -> list[str]:
        """Nodes with no backing section. Should always be empty — a
        non-empty result means an edge escaped resolution. Asserted in
        tests/test_legal_kg.py."""
        known = getattr(self, "_known_ids", set())
        return [n for n in self.graph.nodes if n not in known]

    # ── Query interface ────────────────────────────────────────────────────────

    def neighbors(
        self,
        section_id: str,
        edge_types: set[str] | None = None,
    ) -> list[str]:
        """Return direct neighbors reachable via edges of the given types."""
        if not self.graph.has_node(section_id):
            return []
        result = []
        for _, dst, data in self.graph.out_edges(section_id, data=True):
            if edge_types is None or data.get("type") in edge_types:
                result.append(dst)
        return result

    # Edge types in descending order of how much they tell you that
    # retrieval didn't already know. Structural edges (PARENT_OF/CHILD_OF)
    # are excluded by default because ChunkStructurer already supplies
    # hierarchy; SAME_ACT is far too weak to expand on.
    EXPANSION_PRIORITY = (EDGE_SUPERSEDES, EDGE_READ_WITH, EDGE_RELATES_TO)

    def expand(
        self,
        seed_ids:   list[str],
        hops:       int       = 1,
        edge_types: set[str]  | None = None,
        max_expand: int       = 8,
    ) -> KGExpansionResult:
        """Multi-hop expansion from seed sections, novel ids only.

        Neighbours are visited in EXPANSION_PRIORITY order rather than
        graph-insertion order. That ordering is the whole value of the
        stage: RELATES_TO outnumbers READ_WITH and SUPERSEDES by roughly
        100:1, so an insertion-ordered walk filled the (small) result cap
        with generic cross-references and the curated legal-practice edges
        — IPC_302 -> IPC_034 common intention, IPC_302 -> BNS_103
        supersession — never survived truncation.
        """
        if edge_types is None:
            edge_types = set(self.EXPANSION_PRIORITY)

        priority = {t: i for i, t in enumerate(self.EXPANSION_PRIORITY)}

        visited   = set(seed_ids)
        frontier  = list(seed_ids)
        new_ids:   list[str]   = []
        traversed: list[KGEdge] = []

        for _hop in range(hops):
            if len(new_ids) >= max_expand:
                break
            candidates: list[tuple[int, str, str, str, str]] = []
            for src in frontier:
                if not self.graph.has_node(src):
                    continue
                for _, dst, data in self.graph.out_edges(src, data=True):
                    etype = data.get("type", "")
                    if etype not in edge_types or dst in visited:
                        continue
                    candidates.append(
                        (priority.get(etype, len(priority)), src, dst, etype,
                         data.get("note", ""))
                    )

            # Stable sort by edge-type priority; ties keep discovery order.
            candidates.sort(key=lambda c: c[0])

            next_frontier: list[str] = []
            for _prio, src, dst, etype, note in candidates:
                if len(new_ids) >= max_expand:
                    break
                if dst in visited:
                    continue
                visited.add(dst)
                new_ids.append(dst)
                next_frontier.append(dst)
                traversed.append(KGEdge(src=src, dst=dst, edge_type=etype, note=note))

            frontier = next_frontier
            if not frontier:
                break

        return KGExpansionResult(
            seed_ids        = list(seed_ids),
            expanded_ids    = new_ids,
            edges_traversed = traversed,
            hop_depth       = hops,
        )

    # Acts that replaced IPC / CrPC / IEA on 2024-07-01.
    SUCCESSOR_ACT_CODES = frozenset({"BNS", "BNSS", "BSA"})

    def get_superseded_by(self, section_id: str) -> list[str]:
        """Sections that supersede this one (e.g. BNS_103 for IPC_302).

        BUGFIX: this filtered on ``act_name in ("BNS", "BNSS", "BSA")``,
        but act_name holds the full title — "Bharatiya Nyaya Sanhita 2023"
        — so the test never passed and this always returned []. Same
        act_code-vs-act_name confusion that made the IRAC reranker's act
        bonus fire on the wrong acts.
        """
        if not self.graph.has_node(section_id):
            return []
        return [
            dst for _, dst, d in self.graph.out_edges(section_id, data=True)
            if d.get("type") == EDGE_SUPERSEDES
            and self.graph.nodes.get(dst, {}).get("act_code", "") in self.SUCCESSOR_ACT_CODES
        ]

    def get_read_with(self, section_id: str) -> list[str]:
        """Return sections commonly cited together with this section."""
        if not self.graph.has_node(section_id):
            return []
        return [
            dst for _, dst, d in self.graph.out_edges(section_id, data=True)
            if d.get("type") == EDGE_READ_WITH
        ]

    def shortest_path(self, src: str, dst: str) -> list[str]:
        """Shortest legal connection between two sections."""
        try:
            return nx.shortest_path(self.graph, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def stats(self) -> dict:
        edge_type_counts: dict = {}
        for _, _, d in self.graph.edges(data=True):
            t = d.get("type", "UNKNOWN")
            edge_type_counts[t] = edge_type_counts.get(t, 0) + 1
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "edge_types": edge_type_counts,
        }


# ── KG-augmented retrieval helper ─────────────────────────────────────────────

def kg_augment_ranked(
    ranked_chunks:    list,
    kg:               LegalKnowledgeGraph,
    structurer,
    max_kg_additions: int  = 4,
    edge_types:       set  | None = None,
    hops:             int  = 1,
    cutoff_year:      int  | None = None,
    cutoff_date:      str  | None = None,
) -> list:
    """
    Stage 6.75 — KG-augmented ranked list expansion.

    Takes the top-ranked sections, expands them via the KG, fetches the
    novel sections from Qdrant (via structurer), and appends them to the
    ranked list at a discounted score of 0.30.

    cutoff_year: same meaning as TemporalFilter's — the latest year a
    provision could have been enacted and still apply to the query (e.g.
    "before 2023" -> 2022). BUGFIX: this used to fetch KG-expanded sections
    straight from Qdrant and stamp them validity_label="active"
    unconditionally, bypassing Stage 4's temporal filter entirely — the
    KG's own SUPERSEDES edge (IPC -> BNS, by design) could silently
    re-inject a section a "before 2023" query had already correctly
    excluded. When cutoff_year is set, any candidate enacted after it is
    skipped instead of added.

    Returns the augmented ranked list.
    """
    from pipeline.irac_reranker import RankedChunk
    from pipeline.temporal_filter import ValidatedChunk

    if not ranked_chunks or not kg._built:
        return ranked_chunks

    # Use only the top-5 as seeds to keep expansion focused
    seed_ids = [r.chunk.section_id for r in ranked_chunks[:5]]
    expansion = kg.expand(seed_ids, hops=hops, edge_types=edge_types,
                          max_expand=max_kg_additions * 2)

    if not expansion.expanded_ids:
        return ranked_chunks

    existing_ids = {r.chunk.section_id for r in ranked_chunks}
    novel_ids    = [sid for sid in expansion.expanded_ids
                    if sid not in existing_ids][:max_kg_additions]

    if not novel_ids:
        return ranked_chunks

    # Fetch novel sections from Qdrant via ChunkStructurer
    added = 0
    augmented = list(ranked_chunks)
    for sid in novel_ids:
        record = structurer._get(sid)
        if not record:
            continue
        meta  = record.get("meta", {})
        irac  = meta.get("irac", {})
        hier  = meta.get("hierarchy", {})

        # Chronological check (see cutoff_year docstring above): skip a KG
        # addition that could not possibly have applied to the query's own
        # date instead of blindly stamping it "active". Checks both
        # directions — is_chronologically_future (didn't exist yet, e.g.
        # BNS for a pre-2024 query) and is_superseded_by_cutoff (already
        # repealed by then, e.g. IPC for a query dated after 2024-07-01;
        # this is exactly how a "before 2023" query getting BNS pulled back
        # in via the KG's own IPC->BNS SUPERSEDES edge was found and fixed,
        # and the mirror direction — IPC surfacing via KG for a query dated
        # well after the changeover — is the same class of bug) — so a
        # mid-2024-changeover query resolves consistently across all three
        # places a chunk can enter the ranked list.
        temporal_meta = meta.get("temporal") or {}
        # section_id is always "ACT_NUM" (e.g. "BNS_146") — no separate
        # act_code field is reconstructed onto this record, so derive it
        # the same reliable way the rest of this file already treats
        # section_ids.
        act_code = sid.split("_", 1)[0]
        from pipeline.temporal_filter import (
            is_chronologically_future, is_superseded_by_cutoff,
        )
        if is_chronologically_future(
            act_code, temporal_meta.get("effective_date"), temporal_meta.get("enacted_year"),
            cutoff_year, cutoff_date,
        ) or is_superseded_by_cutoff(act_code, cutoff_year, cutoff_date):
            continue

        from pipeline.chunk_structurer import StructuredChunk
        sc = StructuredChunk(
            section_id       = sid,
            content          = record.get("content", ""),
            act_name         = hier.get("act", ""),
            chapter          = meta.get("chapter", ""),
            category         = meta.get("category", ""),
            validity_label   = "active",
            warning          = "",
            penalized_score  = 0.30,
            rule_summary     = irac.get("rule_summary", ""),
            issue_tags       = irac.get("issue_tags", []),
            conclusion_type  = irac.get("conclusion_type", ""),
            enriched_context = record.get("content", ""),
        )
        # Find the KG edge that brought this section in
        edge_notes = [e.note for e in expansion.edges_traversed if e.dst == sid]
        note_str   = edge_notes[0] if edge_notes else "KG expansion"
        from pipeline.section_pinner import KG_EXPLANATION_PREFIX

        augmented.append(RankedChunk(
            chunk       = sc,
            final_score = 0.30,
            irac_score  = 0.30,
            explanation = f"{KG_EXPLANATION_PREFIX} ({note_str})",
        ))
        added += 1

    return augmented


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import QDRANT_PATH, COLLECTION_NAME
    from qdrant_client import QdrantClient

    client = QdrantClient(path=QDRANT_PATH)
    kg = LegalKnowledgeGraph()
    kg.build_from_qdrant(client, COLLECTION_NAME)

    print("\nKG stats:", kg.stats())
    print()

    # Test expansion
    tests = [
        (["IPC_302"],          "Murder (IPC) → superseded by BNS + common intention"),
        (["IPC_498A"],         "Dowry cruelty → read with dowry death sections"),
        (["PCA_007"],          "Bribery → related corruption provisions"),
        (["ITA_066"],          "Hacking → cheating + identity theft"),
        (["POCSO_004"],        "Penetrative assault → aggravated + IPC provisions"),
    ]
    for seeds, desc in tests:
        exp = kg.expand(seeds, hops=1)
        rw  = kg.get_read_with(seeds[0])
        sup = kg.get_superseded_by(seeds[0])
        print(f"Seed: {seeds[0]}  [{desc}]")
        print(f"  Expanded ({len(exp.expanded_ids)}): {exp.expanded_ids}")
        print(f"  Read-with: {rw}")
        print(f"  Superseded-by: {sup}")
        if exp.edges_traversed:
            for e in exp.edges_traversed[:3]:
                print(f"    {e.src} --[{e.edge_type}]--> {e.dst}  {e.note}")
        print()
