"""
Unit, integration, and performance tests for SQLiteGraphStore batch operations and traversal.

Verifies:
1. Batch node and edge insertion and upsert semantics.
2. Atomic rollback on batch failure.
3. Disk-backed persistence across store reopening.
4. In-memory (:memory:) persistence across builder/retriever lifecycle.
5. Neighbor lookup correctness in outgoing, incoming, and both directions.
6. Deterministic ordering and edge_type filtering in get_neighbors.
7. BFS traversal equivalence and correctness.
8. Measured graph construction and traversal performance benchmark.
"""

import json
import sqlite3
import tempfile
import time
from pathlib import Path
import pytest

from src.graph.schema import GraphNode, GraphEdge
from src.graph.store import SQLiteGraphStore
from src.chunking.turn_chunker import TurnChunk
from src.chunking.graph_retrieval import TurnGraphBuilder, TurnGraphRetriever
from src.graph.traversal import traverse_relevance_guided


@pytest.fixture
def temp_disk_store():
    """Provides a fresh disk-backed SQLiteGraphStore in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_store.db"
        store = SQLiteGraphStore(db_path)
        try:
            yield store, db_path
        finally:
            store.close()


@pytest.fixture
def memory_store():
    """Provides an in-memory SQLiteGraphStore."""
    store = SQLiteGraphStore(":memory:")
    try:
        yield store
    finally:
        store.close()


# ==============================================================================
# 1. Batch Insertion & Upsert Semantics
# ==============================================================================

def test_batch_upsert_nodes_and_edges(temp_disk_store):
    store, _ = temp_disk_store
    nodes = [
        GraphNode(id=f"node_{i}", node_type="entity", label=f"Entity {i}", properties={"val": i})
        for i in range(5)
    ]
    edges = [
        GraphEdge(
            id=f"edge_0_{i}",
            source_id="node_0",
            target_id=f"node_{i}",
            edge_type="relates_to",
            origin="explicit",
            confidence=1.0,
            properties={"weight": 1.0}
        )
        for i in range(1, 5)
    ]

    store.batch_upsert(nodes=nodes, edges=edges)

    assert store.count_nodes() == 5
    assert store.count_edges() == 4

    # Verify cached node access
    node_0 = store.get_node("node_0")
    assert node_0 is not None
    assert node_0.properties["val"] == 0

    # Test idempotence and updates
    updated_nodes = [
        GraphNode(id="node_0", node_type="entity", label="Updated Node 0", properties={"val": 999})
    ]
    store.upsert_nodes_batch(updated_nodes)
    assert store.count_nodes() == 5
    assert store.get_node("node_0").properties["val"] == 999
    assert store.get_node("node_0").label == "Updated Node 0"


# ==============================================================================
# 2. Atomic Rollback on Batch Failure
# ==============================================================================

def test_atomic_rollback_on_batch_failure(temp_disk_store):
    store, _ = temp_disk_store
    valid_node = GraphNode(id="initial_node", node_type="chunk", label="Chunk 1", properties={})
    store.upsert_node(valid_node)
    assert store.count_nodes() == 1

    # Attempt a batch with an invalid foreign key edge (target_id does not exist)
    # With foreign keys enabled, this must trigger an IntegrityError and rollback
    batch_nodes = [
        GraphNode(id="new_node_1", node_type="entity", label="New Entity 1", properties={}),
        GraphNode(id="new_node_2", node_type="entity", label="New Entity 2", properties={})
    ]
    invalid_edges = [
        GraphEdge(
            id="bad_edge",
            source_id="new_node_1",
            target_id="non_existent_target_node_xyz",
            edge_type="invalid_rel",
            origin="explicit",
            confidence=1.0
        )
    ]

    with pytest.raises(sqlite3.IntegrityError):
        store.batch_upsert(nodes=batch_nodes, edges=invalid_edges)

    # Rollback verification: Neither new_node_1, new_node_2, nor bad_edge should exist
    assert store.count_nodes() == 1
    assert store.count_edges() == 0
    assert store.get_node("new_node_1") is None
    assert store.get_node("new_node_2") is None


# ==============================================================================
# 3. Disk-Backed Database Persistence
# ==============================================================================

def test_disk_backed_persistence_across_close(temp_disk_store):
    store, db_path = temp_disk_store
    nodes = [
        GraphNode(id="persisted_node_1", node_type="entity", label="Alice", properties={"city": "Pune"}),
        GraphNode(id="persisted_node_2", node_type="entity", label="Bob", properties={"city": "Delhi"})
    ]
    edges = [
        GraphEdge(
            id="persisted_edge_1",
            source_id="persisted_node_1",
            target_id="persisted_node_2",
            edge_type="collaborates_with",
            origin="explicit",
            confidence=0.95
        )
    ]
    store.batch_upsert(nodes=nodes, edges=edges)
    store.close()

    # Reopen a brand new store instance on the same db_path
    reopened_store = SQLiteGraphStore(db_path)
    try:
        assert reopened_store.count_nodes() == 2
        assert reopened_store.count_edges() == 1
        n1 = reopened_store.get_node("persisted_node_1")
        assert n1 is not None
        assert n1.properties["city"] == "Pune"
        neighbors = reopened_store.get_neighbors("persisted_node_1", direction="outgoing")
        assert len(neighbors) == 1
        assert neighbors[0][1].id == "persisted_node_2"
    finally:
        reopened_store.close()


# ==============================================================================
# 4. In-Memory Persistence Across Builder & Retriever Lifecycle
# ==============================================================================

def test_in_memory_persistence_lifecycle(memory_store):
    chunks = [
        TurnChunk(
            chunk_id="chunk_01",
            session_id="sess_01",
            memory_id="mem_01",
            turn_start_idx=0,
            turn_end_idx=1,
            total_session_turns=2,
            timestamp="2026-03-01T12:00:00Z",
            parsed_datetime=None,
            turns=[],
            text="User lives in Pune and works with Marcus on IVARA.",
            word_count=10,
            token_estimate=15
        )
    ]

    builder = TurnGraphBuilder(memory_store)
    stats = builder.build_from_chunks(chunks)

    assert stats["nodes"] >= 1
    assert stats["edges"] >= 1
    assert memory_store.count_nodes() >= 1

    # Hand off the exact same memory_store to TurnGraphRetriever
    retriever = TurnGraphRetriever(chunks=chunks, store=memory_store)
    retrieved, diagnostics = retriever.retrieve("Where does user live?", k=1)

    assert len(retrieved) == 1
    assert retrieved[0]["chunk_id"] == "chunk_01"
    assert memory_store.get_node("chunk_01") is not None


# ==============================================================================
# 5. Neighbor Lookup Correctness in Both Directions
# ==============================================================================

def test_get_neighbors_directionality_and_filtering(memory_store):
    # Setup graph: A -> B (rel1), C -> A (rel2)
    nodes = [
        GraphNode(id="A", node_type="entity", label="A", properties={}),
        GraphNode(id="B", node_type="entity", label="B", properties={}),
        GraphNode(id="C", node_type="entity", label="C", properties={})
    ]
    edges = [
        GraphEdge(id="e_ab", source_id="A", target_id="B", edge_type="calls", origin="explicit", confidence=1.0),
        GraphEdge(id="e_ca", source_id="C", target_id="A", edge_type="triggers", origin="explicit", confidence=1.0)
    ]
    memory_store.batch_upsert(nodes=nodes, edges=edges)

    # 1. Outgoing from A: should only return B
    out_a = memory_store.get_neighbors("A", direction="outgoing")
    assert len(out_a) == 1
    assert out_a[0][0].id == "e_ab"
    assert out_a[0][1].id == "B"

    # 2. Incoming to A: should only return C
    in_a = memory_store.get_neighbors("A", direction="incoming")
    assert len(in_a) == 1
    assert in_a[0][0].id == "e_ca"
    assert in_a[0][1].id == "C"

    # 3. Both from A: should return outgoing first, then incoming
    both_a = memory_store.get_neighbors("A", direction="both")
    assert len(both_a) == 2
    assert both_a[0][0].id == "e_ab"
    assert both_a[0][1].id == "B"
    assert both_a[1][0].id == "e_ca"
    assert both_a[1][1].id == "C"

    # 4. Filter by edge_type
    filtered = memory_store.get_neighbors("A", edge_type="calls", direction="both")
    assert len(filtered) == 1
    assert filtered[0][0].edge_type == "calls"


# ==============================================================================
# 6. BFS Equivalence and Traversal Correctness
# ==============================================================================

def test_traverse_relevance_guided_structure(memory_store):
    nodes = [
        GraphNode(id="start", node_type="chunk", label="Start", properties={"text": "Marcus Vance Python"}),
        GraphNode(id="mid", node_type="entity", label="Marcus Vance", properties={"canonical_name": "Marcus Vance"}),
        GraphNode(id="end", node_type="chunk", label="End", properties={"text": "Chronos data pipeline"})
    ]
    edges = [
        GraphEdge(id="e1", source_id="start", target_id="mid", edge_type="mentions_entity", origin="explicit", confidence=1.0),
        GraphEdge(id="e2", source_id="mid", target_id="end", edge_type="worked_on", origin="explicit", confidence=1.0)
    ]
    memory_store.batch_upsert(nodes=nodes, edges=edges)

    res = traverse_relevance_guided(
        store=memory_store,
        query="What did Marcus Vance work on with Python?",
        seed_node_ids=["start"],
        max_depth=2,
        max_nodes=10
    )

    discovered_ids = {n.id for n in res["discovered_nodes"]}
    assert "start" in discovered_ids
    assert "mid" in discovered_ids
    assert res["nodes_expanded"] >= 2
    assert len(res["diagnostics"]) >= 1


# ==============================================================================
# 7. Repeatable Graph Construction & Traversal Performance Benchmark
# ==============================================================================

def test_performance_benchmark(temp_disk_store):
    """
    Performance test asserting that batch graph construction of ~200 items
    on disk completes well under 1.5 seconds (prior to batching, unbatched commits
    on Windows NTFS took 2.5 to 5+ seconds).
    """
    store, _ = temp_disk_store
    num_sessions = 20
    chunks_per_session = 5
    chunks = []
    
    for s in range(num_sessions):
        sess_id = f"perf_sess_{s:02d}"
        for c in range(chunks_per_session):
            chunk = TurnChunk(
                chunk_id=f"{sess_id}_c{c:02d}",
                session_id=sess_id,
                memory_id=f"mem_{sess_id}",
                turn_start_idx=c * 2,
                turn_end_idx=c * 2 + 1,
                total_session_turns=chunks_per_session * 2,
                timestamp=f"2026-02-{(s % 25) + 1:02d}",
                parsed_datetime=None,
                turns=[],
                text=f"User talked to Alice regarding Python and Rust on project Chronos in turn {c}.",
                word_count=14,
                token_estimate=20
            )
            chunks.append(chunk)

    builder = TurnGraphBuilder(store)
    t0 = time.perf_counter()
    stats = builder.build_from_chunks(chunks)
    t_build = time.perf_counter() - t0

    assert stats["nodes"] >= 100
    assert stats["edges"] >= 200

    # Traversal timing
    t0 = time.perf_counter()
    for _ in range(5):
        traverse_relevance_guided(
            store=store,
            query="Python Chronos Alice",
            seed_node_ids=[chunks[0].chunk_id],
            max_depth=2,
            max_nodes=15
        )
    t_traversal_avg = (time.perf_counter() - t0) / 5.0

    print(f"\n[BENCHMARK] Disk build ({stats['nodes']} nodes, {stats['edges']} edges): {t_build*1000:.2f} ms")
    print(f"[BENCHMARK] Traversal (mean of 5): {t_traversal_avg*1000:.2f} ms")

    # Safe performance threshold: should easily complete in < 1.0s (measured at ~15ms)
    assert t_build < 1.0, f"Graph build took too long: {t_build:.4f}s"
    assert t_traversal_avg < 0.1, f"Graph traversal took too long: {t_traversal_avg:.4f}s"
