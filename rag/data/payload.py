"""
data/payload.py
The Qdrant payload shape for a statute section, and the normalization
applied when building it.

Split out of data/indexer.py so that anything needing the payload
definition — notably pipeline/section_store.py's offline JSON path, and
the tests — can import it without dragging in sentence-transformers,
torch and qdrant-client.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from data.section_ref import normalize_section_ref


def build_payload(record: dict) -> dict:
    m = record["meta"]
    return {
        "section_id":       record["section"],
        "act_code":         m["code"],
        "act_name":         m["hierarchy"]["act"],
        "section_number":   m["section"],
        "chapter":          m.get("chapter") or "",
        "category":         m.get("category") or "",
        "keywords":         m.get("keywords") or [],
        "status":           m["temporal"]["status"] or "active",
        "enacted_year":     m["temporal"]["enacted_year"],
        "effective_date":   m["temporal"].get("effective_date") or "",
        "last_amended":     m["temporal"]["last_amended"] or "",
        "amended_by":       m["temporal"]["amended_by"] or [],
        "superseded_by":    m["temporal"].get("superseded_by") or "",
        "supersedes":       m["temporal"].get("supersedes") or "",
        "intent":           m["legal_type"]["intent"] or "",
        "rule_type":        m["legal_type"]["rule_type"] or "",
        "applies_to":       m["legal_type"]["applies_to"] or [],
        "punishment_type":  m["legal_type"]["punishment_type"] or "",
        "jurisdiction":     m["legal_type"]["jurisdiction"] or "India",
        "issue_tags":       m["irac"]["issue_tags"] or [],
        "rule_summary":     m["irac"]["rule_summary"] or "",
        # Normalized at index time so every consumer sees one canonical
        # spelling. The dataset writes these capitalized and inconsistently
        # ("Penal", "Procedural Order", "Definitional"); the IRAC reranker
        # used to compare them against lowercase literals and every match
        # branch was consequently dead. See pipeline/irac_reranker.py.
        "conclusion_type":  (m["irac"]["conclusion_type"] or "").strip().lower(),
        "parent_section":   m["hierarchy"]["parent_section"] or "",
        "child_sections":   m["hierarchy"]["child_sections"] or [],
        # Stored in the dataset as "IPC 2" (space-separated, unpadded) while
        # every section_id is "IPC_002". Normalizing here means the graph
        # builder and the chunk structurer both get real, resolvable ids
        # instead of dangling references — 5111 of 5339 cross-references
        # pointed at nothing before this.
        "related_sections": [],   # filled by run() once every id is known
        "related_sections_raw": m["related_sections"] or [],
        "content":          record["content"],
        "embedding_text":   record["embedding_text"],
    }


def resolve_related_refs(raw_refs: list[str], known_ids: set[str]) -> tuple[list[str], int]:
    """'IPC 2' -> 'IPC_002', dropping references to sections this corpus
    does not contain.

    The dataset writes cross-references space-separated and unpadded while
    every section_id is underscored and zero-padded, so only 228 of 5339
    references matched verbatim. Consumers that looked them up by exact id
    (the chunk structurer) silently got nothing, and the one that created
    graph nodes on demand (legal_kg) manufactured 2414 phantom nodes —
    41% of the graph — that no lookup could ever resolve.

    Returns (resolved_ids, n_dropped).
    """
    resolved: list[str] = []
    dropped = 0
    for raw in raw_refs:
        sid = normalize_section_ref(raw, known_ids)
        if sid is None:
            dropped += 1
        elif sid not in resolved:
            resolved.append(sid)
    return resolved, dropped
