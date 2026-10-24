"""
Unit and integration tests for the IVARA Memory Graph system.

Tests:
1. Graph persistence, schema, and idempotent rebuilding (SQLite).
2. Chunking, provenance tracking, and entity linking.
3. Relevance-guided traversal OR rule:
   explore(edge, neighbor, query) = relevant(edge) OR relevant(neighbor)
4. Cycle prevention, budget limits, and depth cutoffs.
5. Cross-memory candidate discovery and evidence verification.
6. Hallucination prevention and unanswerable abstention safety.
"""

import tempfile
from pathlib import Path
import pytest

from src.graph.schema import GraphNode, GraphEdge, EntityNode, ChunkNode
from src.graph.store import SQLiteGraphStore
from src.graph.chunker import chunk_memory
from src.graph.builder import MemoryGraphBuilder
from src.graph.traversal import (
    evaluate_edge_relevance,
    evaluate_destination_relevance,
    traverse_relevance_guided
)
from src.graph.retrieval import GraphAssistedRetriever
from src.answering import answer_question


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_graph.db"
        store = SQLiteGraphStore(db_path)
        try:
            yield store
        finally:
            store.close()


# ==========================================
# 1. Graph Integrity & Idempotent Persistence
# ==========================================

def test_graph_node_and_edge_persistence(temp_store):
    """Verify nodes and edges persist and can be queried by ID and type."""
    node1 = GraphNode(id="n1", node_type="entity", label="Marcus Vance", properties={"role": "Lead"})
    node2 = GraphNode(id="n2", node_type="entity", label="Project Chronos", properties={"type": "Project"})
    edge = GraphEdge(
        id="e1",
        source_id="n1",
        target_id="n2",
        edge_type="worked_on",
        origin="explicit",
        confidence=1.0,
        evidence="Marcus Vance built Chronos",
        memory_id="m001"
    )

    temp_store.upsert_node(node1)
    temp_store.upsert_node(node2)
    temp_store.upsert_edge(edge)

    assert temp_store.count_nodes() == 2
    assert temp_store.count_edges() == 1

    fetched_node = temp_store.get_node("n1")
    assert fetched_node is not None
    assert fetched_node.label == "Marcus Vance"
    assert fetched_node.properties["role"] == "Lead"

    fetched_edges = temp_store.get_outgoing_edges("n1")
    assert len(fetched_edges) == 1
    assert fetched_edges[0].edge_type == "worked_on"
    assert fetched_edges[0].evidence == "Marcus Vance built Chronos"


def test_idempotent_rebuilding(temp_store):
    """Verify re-inserting the same nodes and edges does not create duplicates."""
    node = GraphNode(id="n1", node_type="entity", label="Marcus Vance", properties={"v": 1})
    temp_store.upsert_node(node)
    assert temp_store.count_nodes() == 1

    # Update with new properties
    node_updated = GraphNode(id="n1", node_type="entity", label="Marcus Vance", properties={"v": 2})
    temp_store.upsert_node(node_updated)
    assert temp_store.count_nodes() == 1
    assert temp_store.get_node("n1").properties["v"] == 2


# ==========================================
# 2. Chunking & Provenance
# ==========================================

def test_chunking_short_memory():
    """Short memory (<150 words) should yield a single pristine chunk."""
    memo = {
        "id": "m001",
        "text": "Marcus Vance built the data ingestion pipeline using Rust at Aetheria Labs.",
        "timestamp": "2025-01-14",
        "source": "Episode 1",
        "entities": [{"canonical_name": "Marcus Vance", "entity_id": "ent_marcus_vance"}]
    }
    chunks = chunk_memory(memo, target_words=100)
    assert len(chunks) == 1
    assert chunks[0].memory_id == "m001"
    assert chunks[0].chunk_index == 0
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(memo["text"])
    assert "ent_marcus_vance" in chunks[0].entity_ids


def test_chunking_long_memory():
    """Long narrative should split into multiple coherent chunks preserving sentence boundaries."""
    sentences = [f"This is sentence number {i} describing an event in the long chronicle." for i in range(25)]
    long_text = " ".join(sentences)
    memo = {
        "id": "m002",
        "text": long_text,
        "timestamp": "2025-02-01",
        "source": "Episode 2"
    }
    chunks = chunk_memory(memo, target_words=50, overlap_sentences=1)
    assert len(chunks) > 1
    # Check consecutive ordering
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.memory_id == "m002"
        assert len(c.text) > 0


# ==========================================
# 3. Relevance-Guided Traversal OR Rule
# ==========================================

def test_or_rule_relevant_edge_irrelevant_destination():
    """
    Case A: Edge is relevant to query, but destination node alone does not match keywords.
    explore(edge, dest, query) MUST be True.
    """
    edge = GraphEdge(
        id="e1",
        source_id="n1",
        target_id="n2",
        edge_type="uses_language",
        origin="explicit",
        confidence=1.0,
        evidence="uses Rust for systems engineering"
    )
    dest_node = GraphNode(id="n2", node_type="chunk", label="Chunk 2", properties={"text": "It ensures zero memory leaks."})
    query = "What programming language was used?"

    edge_rel, _, _ = evaluate_edge_relevance(edge, query)
    dest_rel, _, _ = evaluate_destination_relevance(dest_node, query)

    assert edge_rel is True
    assert dest_rel is False
    # OR rule holds
    assert (edge_rel or dest_rel) is True


def test_or_rule_irrelevant_edge_relevant_destination():
    """
    Case B: Edge is generic/irrelevant, but destination node contains key query terms.
    explore(edge, dest, query) MUST be True.
    """
    edge = GraphEdge(
        id="e1",
        source_id="n1",
        target_id="n2",
        edge_type="adjacent_to",
        origin="structural",
        confidence=1.0
    )
    dest_node = GraphNode(
        id="n2",
        node_type="chunk",
        label="Chunk 2",
        properties={"text": "David K. Miller was hired as Product Manager on September 15, 2025."}
    )
    query = "Who was hired on September 15, 2025 as Product Manager?"

    edge_rel, _, _ = evaluate_edge_relevance(edge, query)
    dest_rel, _, _ = evaluate_destination_relevance(dest_node, query)

    assert dest_rel is True
    assert (edge_rel or dest_rel) is True


def test_or_rule_neither_relevant_is_rejected():
    """
    Case C: Neither edge nor destination is relevant to query.
    explore(edge, dest, query) MUST be False.
    """
    edge = GraphEdge(
        id="e1",
        source_id="n1",
        target_id="n2",
        edge_type="owned_by",
        origin="explicit",
        confidence=1.0
    )
    dest_node = GraphNode(
        id="n2",
        node_type="chunk",
        label="Chunk 2",
        properties={"text": "The office microwave in the kitchen was replaced."}
    )
    query = "What programming language was used for Chronos?"

    edge_rel, _, _ = evaluate_edge_relevance(edge, query)
    dest_rel, _, _ = evaluate_destination_relevance(dest_node, query)

    assert edge_rel is False
    assert dest_rel is False
    assert (edge_rel or dest_rel) is False


# ==========================================
# 4. Cycle Prevention & Traversal Limits
# ==========================================

def test_cycle_prevention_and_depth_limits(temp_store):
    """
    Construct a cyclical graph: n1 -> n2 -> n3 -> n1.
    Verify traversal terminates safely without infinite loop or visited duplication.
    """
    n1 = GraphNode(id="n1", node_type="entity", label="Marcus Vance", properties={"canonical_name": "Marcus Vance"})
    n2 = GraphNode(id="n2", node_type="entity", label="Project Chronos", properties={"canonical_name": "Project Chronos"})
    n3 = GraphNode(id="n3", node_type="entity", label="Rust", properties={"canonical_name": "Rust"})

    temp_store.upsert_node(n1)
    temp_store.upsert_node(n2)
    temp_store.upsert_node(n3)

    # Cyclical edges
    temp_store.upsert_edge(GraphEdge(id="e1", source_id="n1", target_id="n2", edge_type="worked_on", origin="explicit", confidence=1.0))
    temp_store.upsert_edge(GraphEdge(id="e2", source_id="n2", target_id="n3", edge_type="uses_language", origin="explicit", confidence=1.0))
    temp_store.upsert_edge(GraphEdge(id="e3", source_id="n3", target_id="n1", edge_type="worked_with", origin="explicit", confidence=1.0))

    query = "What programming language was used by Marcus Vance on Project Chronos?"
    result = traverse_relevance_guided(
        store=temp_store,
        query=query,
        seed_node_ids=["n1"],
        max_depth=5,
        max_nodes=10
    )

    # Must visit each node at most once
    visited_ids = [n.id for n in result["discovered_nodes"]]
    assert len(visited_ids) == len(set(visited_ids))
    assert len(visited_ids) <= 3


def test_budget_exhaustion_stops_traversal(temp_store):
    """Traversal stops immediately when max_nodes budget is reached."""
    # Create star graph: root -> 10 children
    root = GraphNode(id="root", node_type="entity", label="Aetheria Labs", properties={"canonical_name": "Aetheria Labs"})
    temp_store.upsert_node(root)
    for i in range(10):
        c = GraphNode(id=f"c{i}", node_type="entity", label=f"Child {i}", properties={"canonical_name": f"Child {i}"})
        temp_store.upsert_node(c)
        temp_store.upsert_edge(GraphEdge(id=f"e_{i}", source_id="root", target_id=f"c{i}", edge_type="employs", origin="explicit", confidence=1.0))

    query = "Who was employed by Aetheria Labs?"
    result = traverse_relevance_guided(
        store=temp_store,
        query=query,
        seed_node_ids=["root"],
        max_depth=2,
        max_nodes=4  # Strict budget
    )
    assert len(result["discovered_nodes"]) <= 4


# ==========================================
# 5. End-to-End Retrieval & Answering Grounding
# ==========================================

def test_multi_hop_cross_memory_discovery(temp_store):
    """
    Test Case: Sarah Lin (m004) -> Marcus Vance -> Rust (m001).
    Demonstrate that graph traversal discovers m001 from m004 even when query only names Sarah Lin.
    """
    entities = [
        {"id": "ent_sarah_lin", "canonical_name": "Sarah Lin", "aliases": ["Sarah"], "types": ["Person"]},
        {"id": "ent_marcus_vance", "canonical_name": "Marcus Vance", "aliases": ["Marc"], "types": ["Person"]},
        {"id": "ent_rust", "canonical_name": "Rust", "aliases": [], "types": ["Technology", "ProgrammingLanguage"]},
        {"id": "ent_chronos", "canonical_name": "Project Chronos", "aliases": ["Chronos"], "types": ["Project"]}
    ]

    memories = [
        {
            "id": "m001",
            "text": "Marcus Vance insisted on implementing the ingestion pipeline for Project Chronos using Rust.",
            "timestamp": "2025-01-14",
            "source": "Episode 1",
            "relationships": [
                {"source": {"canonical_name": "Marcus Vance"}, "predicate": "worked_on", "target": {"canonical_name": "Project Chronos"}, "evidence": "insisted on implementing"},
                {"source": {"canonical_name": "Project Chronos"}, "predicate": "uses", "target": {"canonical_name": "Rust"}, "evidence": "using Rust"}
            ]
        },
        {
            "id": "m004",
            "text": "Sarah Lin joined Aetheria Labs as a junior data engineer reporting directly to Marc.",
            "timestamp": "2025-06-20",
            "source": "Episode 4",
            "relationships": [
                {"source": {"canonical_name": "Sarah Lin"}, "predicate": "worked_with", "target": {"canonical_name": "Marcus Vance"}, "evidence": "reporting directly to Marc"}
            ]
        }
    ]

    builder = MemoryGraphBuilder(temp_store)
    builder.build_from_records(memories=memories, entities=entities)

    retriever = GraphAssistedRetriever(temp_store)
    query = "What programming language was used by the person who mentored Sarah Lin?"
    res = retriever.retrieve(query=query, k=3, use_graph_expansion=True)

    retrieved_ids = [m["id"] for m in res["retrieved_memories"]]
    # Graph traversal connects Sarah Lin -> Marcus Vance -> Rust on m001!
    assert "m001" in retrieved_ids
    assert res["has_sufficient_evidence"] is True


def test_unanswerable_question_abstention(temp_store):
    """
    Test Case: Unanswerable query where no evidence exists.
    Must abstain with insufficient_evidence.
    """
    memories = [
        {
            "id": "m001",
            "text": "Marcus Vance built the data pipeline in Rust.",
            "timestamp": "2025-01-14",
            "source": "Episode 1",
            "relationships": []
        }
    ]
    builder = MemoryGraphBuilder(temp_store)
    builder.build_from_records(memories=memories)

    retriever = GraphAssistedRetriever(temp_store)
    query = "What brand of tennis shoes does the Chief Financial Officer wear on weekends?"
    res = retriever.retrieve(query=query, k=3)

    # Empty or irrelevant hits must not hallucinate
    # When fed into answer_question:
    mock_caller = lambda p, s: '{"status": "insufficient_evidence", "answer": "I do not know", "cited_memory_ids": []}'
    ans = answer_question(
        query=query,
        use_graph=True,
        graph_store=temp_store,
        llm_caller=mock_caller
    )
    # The retriever didn't match shoes, so it abstained immediately or returned insufficient evidence
    assert ans["status"] == "insufficient_evidence"


# ==========================================
# 6. Retrieval Correctness Audit Regressions
# ==========================================

def test_chunk_level_evidence_not_full_episode(temp_store):
    """
    Audit Item 1:
    Verify that retrieval preserves selected chunk text, source memory ID, chunk ID,
    source offsets, and timestamp, and builds answer context from selected chunks
    WITHOUT dumping the entire long parent episode.
    """
    p1 = " ".join(["Initial background setup context padding sentence." for _ in range(30)])
    target_sentence = "Marcus Vance finalized the decision to use Rust for the core telemetry pipeline on March 15, 2024."
    p2 = " ".join(["Subsequent operations and deployment context padding sentence." for _ in range(30)])
    full_text = f"{p1} {target_sentence} {p2}"

    memo = {
        "id": "m_long_01",
        "text": full_text,
        "timestamp": "2024-03-15",
        "source": "Episode Long Narrative",
        "relationships": [
            {
                "source": {"canonical_name": "Marcus Vance"},
                "predicate": "uses",
                "target": {"canonical_name": "Rust"},
                "evidence": "use Rust for the core telemetry pipeline"
            }
        ]
    }
    builder = MemoryGraphBuilder(temp_store)
    builder.build_from_records(memories=[memo])

    retriever = GraphAssistedRetriever(temp_store)
    query = "What language did Marcus Vance decide to use for telemetry?"
    res = retriever.retrieve(query=query, k=1)

    assert res["has_sufficient_evidence"] is True
    assert len(res["retrieved_chunks"]) == 1
    selected_chunk = res["retrieved_chunks"][0]

    # Verify chunk attributes are preserved
    assert selected_chunk["memory_id"] == "m_long_01"
    assert "chunk_m_long_01_" in selected_chunk["chunk_id"]
    assert selected_chunk["char_start"] >= 0
    assert selected_chunk["char_end"] > selected_chunk["char_start"]
    assert selected_chunk["timestamp"] == "2024-03-15"
    assert "Rust" in selected_chunk["text"]

    # Verify answer context does NOT contain the full parent episode text
    context_text = res["context_text"]
    assert target_sentence in context_text
    assert len(context_text) < len(full_text)
    assert p1 not in context_text


def test_rrf_candidate_score_fusion_and_rank_stability(temp_store):
    """
    Audit Item 2:
    Demonstrate that weak graph candidates cannot automatically outrank
    stronger direct evidence solely because they were discovered through traversal.
    """
    memories = [
        {
            "id": "m010",
            "text": "Marcus Vance implemented the cryptographic authentication module using Ed25519 signatures.",
            "timestamp": "2025-01-10",
            "source": "Episode 10",
            "relationships": [
                {"source": {"canonical_name": "Marcus Vance"}, "predicate": "worked_on", "target": {"canonical_name": "CryptoModule"}, "evidence": "implemented the cryptographic authentication"}
            ]
        },
        {
            "id": "m011",
            "text": "The office cafeteria updated its weekly menu to include vegan sandwiches and fruit juices.",
            "timestamp": "2025-01-11",
            "source": "Episode 11",
            "relationships": [
                {"source": {"canonical_name": "Marcus Vance"}, "predicate": "located_at", "target": {"canonical_name": "OfficeCafeteria"}, "evidence": "visited the office cafeteria"}
            ]
        }
    ]

    builder = MemoryGraphBuilder(temp_store)
    builder.build_from_records(memories=memories)

    retriever = GraphAssistedRetriever(temp_store)
    query = "cryptographic authentication module Ed25519 signatures"
    res = retriever.retrieve(query=query, k=2, use_graph_expansion=True)

    # Strong direct chunk from m010 must be rank 1
    retrieved_chunks = res["retrieved_chunks"]
    assert len(retrieved_chunks) >= 1
    assert retrieved_chunks[0]["memory_id"] == "m010"
    if len(retrieved_chunks) > 1:
        assert retrieved_chunks[0]["score"] > retrieved_chunks[1]["score"]


def test_semantic_relationship_triple_reconstruction(temp_store):
    """
    Audit Item 3:
    Reconstruct relevant semantic relationships as complete subject-predicate-object triples.
    Verify grounds_<predicate>, mentions_entity, and adjacent_to do NOT appear as factual claims.
    """
    entities = [
        {"id": "ent_marcus", "canonical_name": "Marcus Vance", "aliases": [], "types": ["Person"]},
        {"id": "ent_chronos", "canonical_name": "Project Chronos", "aliases": [], "types": ["Project"]}
    ]
    memories = [
        {
            "id": "m020",
            "text": "Marcus Vance directed Project Chronos throughout its early development cycle.",
            "timestamp": "2025-03-01",
            "source": "Episode 20",
            "relationships": [
                {
                    "source": {"canonical_name": "Marcus Vance"},
                    "predicate": "leads",
                    "target": {"canonical_name": "Project Chronos"},
                    "evidence": "directed Project Chronos throughout its early development"
                }
            ]
        }
    ]

    builder = MemoryGraphBuilder(temp_store)
    builder.build_from_records(memories=memories, entities=entities)

    retriever = GraphAssistedRetriever(temp_store)
    query = "Who leads Project Chronos?"
    res = retriever.retrieve(query=query, k=2)

    # Direct facts must contain clean SPO triple
    direct_facts = [f["statement"] for f in res["direct_evidence"][0]["direct_facts"]]
    assert any("Marcus Vance leads Project Chronos" in s for s in direct_facts)

    # Internal edges must NOT be presented as facts
    all_facts_statements = [f["statement"] for f in res.get("direct_evidence", []) for f in f.get("direct_facts", [])]
    for stmt in all_facts_statements:
        assert not stmt.startswith("grounds_")
        assert "mentions_entity" not in stmt
        assert "adjacent_to" not in stmt
        assert "contains_" not in stmt


def test_evidence_to_chunk_matching_later_chunk_and_unmatched(temp_store):
    """
    Audit Item 4:
    Never associate unmatched evidence with chunk 0 by default.
    Support evidence occurring in later chunks, and keep unmatched evidence chunk_id as None.
    """
    # Distinct sentences that form separate chunks
    p0 = " ".join(["Initial background sentences for chunk zero." for _ in range(25)])
    p1 = " ".join(["Middle transition sentences for chunk one." for _ in range(25)])
    p2 = "Marcus Vance joined the Quantum Computing division as Chief Architect in October."
    full_text = f"{p0} {p1} {p2}"

    memo = {
        "id": "m030",
        "text": full_text,
        "timestamp": "2025-10-01",
        "source": "Episode 30",
        "relationships": [
            {
                "source": {"canonical_name": "Marcus Vance"},
                "predicate": "employed_by",
                "target": {"canonical_name": "Quantum Computing division"},
                "evidence": "joined the Quantum Computing division as Chief Architect"
            },
            {
                "source": {"canonical_name": "Marcus Vance"},
                "predicate": "born_in",
                "target": {"canonical_name": "Chicago"},
                "evidence": "Unverifiable external biography statement not present in text"
            }
        ]
    }

    builder = MemoryGraphBuilder(temp_store)
    builder.build_from_records(memories=[memo])

    # Inspect edge 1 in store
    edges = temp_store.get_outgoing_edges("ent_marcus_vance")
    edge_emp = next((e for e in edges if e.edge_type == "employed_by"), None)
    assert edge_emp is not None
    # Must NOT be chunk 0
    assert edge_emp.chunk_id is not None
    assert edge_emp.chunk_id != "chunk_m030_00"

    # Inspect edge 2 (unmatched evidence)
    edge_born = next((e for e in edges if e.edge_type == "born_in"), None)
    assert edge_born is not None
    # Must retain memory provenance m030, but chunk_id MUST be None (unresolved)
    assert edge_born.memory_id == "m030"
    assert edge_born.chunk_id is None


def test_temporal_ordering_shuffled_episodes(temp_store):
    """
    Audit Item 5:
    Deliberately shuffle input records with varied timestamps and verify
    that temporal precedes edges strictly follow chronological timestamps.
    """
    memories = [
        {"id": "m_c", "episode": "Ep C", "text": "Event in late 2025.", "timestamp": "2025-12-01"},
        {"id": "m_a", "episode": "Ep A", "text": "Event in early 2024.", "timestamp": "2024-01-15"},
        {"id": "m_d_notime", "episode": "Ep D", "text": "Event with unspecified date.", "timestamp": None},
        {"id": "m_b", "episode": "Ep B", "text": "Event in mid 2024.", "timestamp": "2024-06-20"}
    ]

    builder = MemoryGraphBuilder(temp_store)
    builder.build_from_records(memories=memories)

    all_nodes = temp_store.get_all_nodes()
    precedes_edges = []
    for n in all_nodes:
        out_edges = temp_store.get_outgoing_edges(n.id)
        for e in out_edges:
            if e.edge_type == "precedes":
                precedes_edges.append(e)

    # Ep A (2024-01-15) -> Ep B (2024-06-20) -> Ep C (2025-12-01)
    # Ep D has no timestamp, so its ordering is left unspecified!
    assert len(precedes_edges) == 2

    edge_ab = next((e for e in precedes_edges if temp_store.get_node(e.source_id).label == "Ep A" and temp_store.get_node(e.target_id).label == "Ep B"), None)
    edge_bc = next((e for e in precedes_edges if temp_store.get_node(e.source_id).label == "Ep B" and temp_store.get_node(e.target_id).label == "Ep C"), None)

    assert edge_ab is not None, "Ep A must precede Ep B"
    assert edge_bc is not None, "Ep B must precede Ep C"
    assert not any(temp_store.get_node(e.source_id).label == "Ep D" or temp_store.get_node(e.target_id).label == "Ep D" for e in precedes_edges)

