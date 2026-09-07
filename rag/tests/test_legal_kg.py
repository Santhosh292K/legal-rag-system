"""The knowledge graph must contain only sections that actually exist.

Before the fix it had 5808 nodes for a 3394-section corpus: 2414 phantoms
created by networkx's add_edge() from unresolvable cross-references like
"IPC 2". Expansion then filled its result cap with those, so the curated
READ_WITH/SUPERSEDES edges the stage exists for never reached the output.
"""
import pytest

networkx = pytest.importorskip("networkx")

from pipeline.legal_kg import (
    LegalKnowledgeGraph, IPC_TO_BNS, READ_WITH_PAIRS,
    EDGE_READ_WITH, EDGE_SUPERSEDES, EDGE_RELATES_TO,
)
from tests.conftest import DATASET_PATH


@pytest.fixture(scope="module")
def kg():
    return LegalKnowledgeGraph().build_from_json(str(DATASET_PATH))


def test_no_phantom_nodes(kg, known_ids):
    assert kg.unresolved_nodes() == []
    assert kg.graph.number_of_nodes() <= len(known_ids)


def test_curated_edges_survive_resolution(kg):
    """IPC_TO_BNS wrote 8 of its 60 targets unpadded (BNS_74 for the
    indexed BNS_074) and READ_WITH_PAIRS referenced IPC_34 for IPC_034.
    Those edges were silently dropped."""
    stats = kg.stats()["edge_types"]
    assert stats[EDGE_SUPERSEDES] >= 2 * (len(IPC_TO_BNS) - 5)
    assert stats[EDGE_READ_WITH] >= 2 * (len(READ_WITH_PAIRS) - 2)


def test_ipc_302_supersession_and_read_with(kg):
    assert "BNS_103" in kg.get_superseded_by("IPC_302")
    read_with = set(kg.get_read_with("IPC_302"))
    assert {"IPC_034", "IPC_120B"} <= read_with, (
        "common-intention and conspiracy links must resolve to padded ids"
    )


def test_expansion_prefers_curated_edges(kg):
    """RELATES_TO outnumbers the curated edges ~100:1, so an
    insertion-ordered walk buried them below the result cap."""
    result = kg.expand(["IPC_302"], hops=1, max_expand=4)
    assert "BNS_103" in result.expanded_ids
    assert "IPC_034" in result.expanded_ids
    first_types = [e.edge_type for e in result.edges_traversed[:2]]
    assert EDGE_RELATES_TO not in first_types


def test_expansion_respects_cap(kg):
    for cap in (1, 3, 8):
        assert len(kg.expand(["IPC_302"], hops=1, max_expand=cap).expanded_ids) <= cap


def test_expansion_never_returns_seeds(kg):
    result = kg.expand(["IPC_302", "IPC_300"], hops=2, max_expand=10)
    assert not ({"IPC_302", "IPC_300"} & set(result.expanded_ids))


def test_node_status_comes_from_the_payload(kg):
    """The builder read a non-existent "temporal_status" key, so every node
    defaulted to active — including repealed and struck-down ones."""
    statuses = {kg.graph.nodes[n].get("status", "") for n in kg.graph.nodes}
    assert len(statuses) > 1, "every node has the same status — key mismatch?"
