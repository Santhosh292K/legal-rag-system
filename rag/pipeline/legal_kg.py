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

    def build_from_json(
        self,
        json_path: str = JSON_PATH,
    ) -> "LegalKnowledgeGraph":
        """Fallback: build from dataset JSON when Qdrant client unavailable."""
        import json
        print(f"[LegalKG] Building graph from {json_path} ...")
        with open(json_path) as f:
            records = json.load(f)

        class _Stub:
            def __init__(self, payload): self.payload = payload

        stubs = [_Stub(r) for r in records]
        # Remap flat JSON keys to payload shape
        for r in records:
            r.setdefault("section_id",        r.get("section", ""))
            r.setdefault("parent_section",    r.get("meta", {}).get("hierarchy", {}).get("parent_section", ""))
            r.setdefault("child_sections",    r.get("meta", {}).get("hierarchy", {}).get("child_sections", []))
            r.setdefault("related_sections",  r.get("meta", {}).get("related_sections", []))
            r.setdefault("act_name",          r.get("meta", {}).get("hierarchy", {}).get("act", ""))
            r.setdefault("chapter",           r.get("meta", {}).get("chapter", ""))
            r.setdefault("temporal_status",   r.get("meta", {}).get("temporal_status", "active"))

        self._add_records(stubs)
        print(f"[LegalKG] Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        self._built = True
        return self

    def _add_records(self, records):
        """Core graph population from a list of Qdrant points (or stubs)."""
        act_buckets: dict[str, list[str]] = {}

        for point in records:
            p = point.payload or {}
            sid    = p.get("section_id", "")
            act    = p.get("act_name", "")
            chapter= p.get("chapter", "")
            status = p.get("temporal_status", "active")

            if not sid:
                continue

            self.graph.add_node(sid,
                act_name=act, chapter=chapter, status=status,
                title=p.get("rule_summary", "")[:80],
            )
            act_buckets.setdefault(act, []).append(sid)

            # PARENT_OF / CHILD_OF
            parent = p.get("parent_section", "")
            if parent and parent != sid:
                self.graph.add_edge(parent, sid, type=EDGE_CHILD_OF)
                self.graph.add_edge(sid, parent, type=EDGE_PARENT_OF)

            for child in (p.get("child_sections") or []):
                if child and child != sid:
                    self.graph.add_edge(sid, child, type=EDGE_CHILD_OF)
                    self.graph.add_edge(child, sid, type=EDGE_PARENT_OF)

            # RELATES_TO (bidirectional)
            for rel in (p.get("related_sections") or []):
                if rel and rel != sid:
                    self.graph.add_edge(sid, rel, type=EDGE_RELATES_TO)
                    self.graph.add_edge(rel, sid, type=EDGE_RELATES_TO)

        # SAME_ACT edges (connect consecutive sections in the same act)
        for act, sids in act_buckets.items():
            for s in sids:
                self.graph.nodes[s]["act_name"] = act

        # Overlay SUPERSEDES (IPC → BNS domain knowledge)
        for ipc_id, bns_id in IPC_TO_BNS.items():
            if self.graph.has_node(ipc_id) and self.graph.has_node(bns_id):
                self.graph.add_edge(bns_id, ipc_id, type=EDGE_SUPERSEDES,
                                    note="BNS replaces IPC")
                self.graph.add_edge(ipc_id, bns_id, type=EDGE_SUPERSEDES,
                                    note="IPC superseded by BNS")

        # Overlay READ_WITH (domain knowledge pairs)
        for src, dst, note in READ_WITH_PAIRS:
            self.graph.add_edge(src, dst, type=EDGE_READ_WITH, note=note)
            self.graph.add_edge(dst, src, type=EDGE_READ_WITH, note=note)

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

    def expand(
        self,
        seed_ids:   list[str],
        hops:       int       = 1,
        edge_types: set[str]  | None = None,
        max_expand: int       = 8,
    ) -> KGExpansionResult:
        """
        Multi-hop expansion from seed sections.

        Returns only NOVEL sections (not already in seed_ids),
        ordered by hop distance (closer = earlier in list).
        Capped at max_expand to avoid flooding the context.

        Default edge_types = {READ_WITH, SUPERSEDES, RELATES_TO}
        (PARENT_OF / CHILD_OF / SAME_ACT omitted by default — structural
        hierarchy is already handled by ChunkStructurer; KG expansion is
        most valuable for cross-section legal relationships).
        """
        if edge_types is None:
            edge_types = {EDGE_READ_WITH, EDGE_SUPERSEDES, EDGE_RELATES_TO}

        visited     = set(seed_ids)
        frontier    = set(seed_ids)
        new_ids:  list[str]   = []
        traversed: list[KGEdge] = []

        for hop in range(hops):
            next_frontier: set[str] = set()
            for src in frontier:
                for _, dst, data in self.graph.out_edges(src, data=True):
                    etype = data.get("type", "")
                    if etype not in edge_types:
                        continue
                    if dst in visited:
                        continue
                    if len(new_ids) >= max_expand:
                        break
                    visited.add(dst)
                    next_frontier.add(dst)
                    new_ids.append(dst)
                    traversed.append(KGEdge(src=src, dst=dst, edge_type=etype,
                                             note=data.get("note", "")))
                if len(new_ids) >= max_expand:
                    break
            frontier = next_frontier
            if not frontier:
                break

        return KGExpansionResult(
            seed_ids        = list(seed_ids),
            expanded_ids    = new_ids,
            edges_traversed = traversed,
            hop_depth       = hops,
        )

    def get_superseded_by(self, section_id: str) -> list[str]:
        """Return sections that supersede the given section (e.g. BNS for IPC)."""
        return [
            dst for _, dst, d in self.graph.out_edges(section_id, data=True)
            if d.get("type") == EDGE_SUPERSEDES and
            self.graph.nodes.get(dst, {}).get("act_name", "") in ("BNS", "BNSS", "BSA")
        ]

    def get_read_with(self, section_id: str) -> list[str]:
        """Return sections commonly cited together with this section."""
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
) -> list:
    """
    Stage 6.75 — KG-augmented ranked list expansion.

    Takes the top-ranked sections, expands them via the KG, fetches the
    novel sections from Qdrant (via structurer), and appends them to the
    ranked list at a discounted score of 0.30.

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

        augmented.append(RankedChunk(
            chunk       = sc,
            final_score = 0.30,
            irac_score  = 0.30,
            explanation = f"KG-augmented ({note_str})",
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
