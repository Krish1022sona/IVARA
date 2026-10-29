"""
Integration and validation tests for Step 6: Real Pipeline Temporal Integration.

Verifies end-to-end integration of temporal claims and conservative supersession:
1. Production ingestion creates or updates temporal claims.
2. LongMemEval chunk/graph ingestion automatically creates temporal claims.
3. Historical and current queries resolve correctly through the public answering interface
   without callers manually constructing temporal_claims.
4. A genuine update creates a directed supersession edge and preserves the older claim.
5. Ambiguous contradictions remain unresolved without fabricating links.
6. Unrelated subjects and sessions do not supersede one another.
7. Unknown dates are not fabricated.
8. Repeated ingestion is idempotent: rebuilding the same input creates 0 duplicate claims or edges.
9. Citation validation still works with temporal claims and source chunks.
10. Existing baseline behavior remains unchanged when temporal reasoning is disabled.
"""

import json
import tempfile
from pathlib import Path
import pytest

from src.graph.schema import GraphNode, GraphEdge, TemporalClaim
from src.graph.store import SQLiteGraphStore
from src.graph.builder import MemoryGraphBuilder
from src.chunking.turn_chunker import TurnChunk
from src.chunking.graph_retrieval import TurnGraphBuilder
from src.extraction.pipeline import process_and_store_memory
from src.extraction.extractor import Extraction
from src.memory import add_memory
from src.answering import answer_question, validate_citations


def make_chunk(chunk_id: str, session_id: str, text: str, timestamp: str = "2026-01-01T00:00:00Z") -> TurnChunk:
    """Helper to construct TurnChunk instances with all dataclass fields."""
    return TurnChunk(
        chunk_id=chunk_id,
        session_id=session_id,
        memory_id=session_id,
        turn_start_idx=0,
        turn_end_idx=1,
        total_session_turns=2,
        timestamp=timestamp,
        parsed_datetime=None,
        turns=[{"role": "user", "content": text}],
        text=text,
        word_count=len(text.split()),
        token_estimate=len(text.split())
    )


@pytest.fixture
def temp_graph_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_pipe_temporal.db"
        store = SQLiteGraphStore(db_path)
        try:
            yield store
        finally:
            store.close()


# ==============================================================================
# Test 1: Production Ingestion Pipeline Creates & Stores Temporal Claims
# ==============================================================================

def test_production_ingestion_creates_temporal_claims(temp_graph_store):
    with tempfile.TemporaryDirectory() as tmpdir:
        mem_path = Path(tmpdir) / "memories.json"
        ent_path = Path(tmpdir) / "entities.json"

        mock_extraction = Extraction(entities=[], topics=[], relationships=[])

        # Ingest memory stating residence update
        memo = process_and_store_memory(
            text="I moved to Pune in 2026.",
            timestamp="2026-03-01T12:00:00Z",
            source="session_production_01",
            memory_type="lifestyle",
            importance=0.9,
            memory_store_path=mem_path,
            entity_store_path=ent_path,
            extraction=mock_extraction,
            graph_store=temp_graph_store
        )

        assert "temporal_claims" in memo
        assert len(memo["temporal_claims"]) == 1
        claim_data = memo["temporal_claims"][0]
        assert claim_data["predicate"] == "lives_in"
        assert claim_data["object"] == "Pune"
        assert claim_data["valid_from"] == "2026"

        # Verify claim persisted in SQLiteGraphStore
        claim_nodes = temp_graph_store.get_nodes_by_type("claim")
        assert len(claim_nodes) == 1
        assert claim_nodes[0].properties["object"] == "Pune"

        # Verify has_claim edge exists
        claim_edges = temp_graph_store.get_edges_by_type("has_claim")
        assert len(claim_edges) == 1
        assert claim_edges[0].source_id == memo["id"]
        assert claim_edges[0].target_id == claim_nodes[0].id


# ==============================================================================
# Test 2: LongMemEval Chunk Graph Construction Automatically Populates Claims
# ==============================================================================

def test_longmemeval_chunk_graph_creates_temporal_claims(temp_graph_store):
    chunks = [
        make_chunk("chunk_delhi", "sess_01", "User: I lived in Delhi in 2025.\nAssistant: Understood.", "2025-06-01T10:00:00Z"),
        make_chunk("chunk_pune", "sess_02", "User: I moved to Pune in 2026.\nAssistant: Congratulations on the move!", "2026-02-15T10:00:00Z")
    ]

    builder = TurnGraphBuilder(temp_graph_store)
    stats = builder.build_from_chunks(chunks)

    assert stats["claim_nodes"] >= 2
    claim_nodes = temp_graph_store.get_nodes_by_type("claim")
    assert len(claim_nodes) == 2

    # Check that Pune claim supersedes Delhi claim in SQLite
    pune_nodes = [n for n in claim_nodes if n.properties.get("object") == "Pune"]
    delhi_nodes = [n for n in claim_nodes if n.properties.get("object") == "Delhi"]
    assert len(pune_nodes) == 1
    assert len(delhi_nodes) == 1

    pune_node = pune_nodes[0]
    delhi_node = delhi_nodes[0]

    assert pune_node.properties.get("supersedes_claim_id") == delhi_node.id
    assert delhi_node.properties.get("is_current") is False

    # Check directed 'supersedes' edge
    sup_edges = temp_graph_store.get_edges_by_type("supersedes")
    assert len(sup_edges) == 1
    assert sup_edges[0].source_id == pune_node.id
    assert sup_edges[0].target_id == delhi_node.id

    # Check chunk -> claim provenance edge
    has_claim_edges = temp_graph_store.get_edges_by_type("has_claim")
    assert len(has_claim_edges) == 2


# ==============================================================================
# Test 3: Public Answering Resolves Current vs Historical Queries Automatically
# ==============================================================================

def test_public_answering_resolves_temporal_queries_automatically(temp_graph_store):
    chunks = [
        make_chunk("chk_hist", "sess_01", "User: I lived in Delhi in 2025.\nAssistant: Got it.", "2025-06-01T10:00:00Z"),
        make_chunk("chk_curr", "sess_02", "User: I moved to Pune in 2026.\nAssistant: Welcome to Pune!", "2026-02-15T10:00:00Z")
    ]

    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(chunks)

    # 1. Current state query without passing temporal_claims
    mock_llm_curr = lambda prompt, sys: {
        "answer": "You currently live in Pune.",
        "cited_memory_ids": ["chk_curr"],
        "status": "answered",
        "inferred_claims": [],
        "reasoning": "Temporal claims show move to Pune in 2026 superseding Delhi."
    }

    resp_curr = answer_question(
        query="Where do I live now?",
        chunks=chunks,
        graph_store=temp_graph_store,
        use_chunking=True,
        llm_caller=mock_llm_curr,
        min_score=None
    )

    assert resp_curr["status"] == "answered"
    assert "Pune" in resp_curr["answer"]
    ctx_curr = resp_curr["retrieved_context"]
    assert "temporal_resolution" in ctx_curr
    assert ctx_curr["temporal_resolution"]["intent"] == "current"
    assert any("Pune" in c.object for c in ctx_curr["temporal_resolution"]["selected_claims"])
    assert any("Delhi" in c.object for c in ctx_curr["temporal_resolution"]["historical_claims"])

    # 2. Historical state query without passing temporal_claims
    mock_llm_hist = lambda prompt, sys: {
        "answer": "In 2025, you lived in Delhi.",
        "cited_memory_ids": ["chk_hist"],
        "status": "answered",
        "inferred_claims": [],
        "reasoning": "Historical query for 2025 matches Delhi."
    }

    resp_hist = answer_question(
        query="Where did I live in 2025?",
        chunks=chunks,
        graph_store=temp_graph_store,
        use_chunking=True,
        llm_caller=mock_llm_hist,
        min_score=None
    )

    assert resp_hist["status"] == "answered"
    assert "Delhi" in resp_hist["answer"]
    ctx_hist = resp_hist["retrieved_context"]
    assert ctx_hist["temporal_resolution"]["intent"] == "historical"
    assert any("Delhi" in c.object for c in ctx_hist["temporal_resolution"]["selected_claims"])


# ==============================================================================
# Test 4: Genuine Update Creates Supersession Edge & Preserves Older Claim
# ==============================================================================

def test_genuine_update_creates_edge_and_preserves_older_claim(temp_graph_store):
    memories = [
        {"id": "m1", "text": "I worked at Acme Corp in 2024.", "timestamp": "2024-05-01"},
        {"id": "m2", "text": "I worked at Beta Tech in 2026.", "timestamp": "2026-06-01"}
    ]

    mem_builder = MemoryGraphBuilder(temp_graph_store)
    mem_builder.build_from_records(memories)

    sup_edges = temp_graph_store.get_edges_by_type("supersedes")
    assert len(sup_edges) == 1

    claim_nodes = temp_graph_store.get_nodes_by_type("claim")
    assert len(claim_nodes) == 2

    acme_node = next(n for n in claim_nodes if "Acme" in n.label)
    beta_node = next(n for n in claim_nodes if "Beta" in n.label)

    # Older claim is preserved with is_current=False
    assert acme_node.properties["is_current"] is False
    assert beta_node.properties["is_current"] is True
    assert sup_edges[0].source_id == beta_node.id
    assert sup_edges[0].target_id == acme_node.id


# ==============================================================================
# Test 5: Ambiguous Contradictions Remain Unresolved
# ==============================================================================

def test_ambiguous_contradictions_remain_unresolved(temp_graph_store):
    # Two conflicting functional claims without explicit event dates or transition cues
    chunks = [
        make_chunk("chk_amb_1", "sess_01", "I live in Delhi.", "2026-01-01T00:00:00Z"),
        make_chunk("chk_amb_2", "sess_02", "I live in Pune.", "2026-01-01T00:00:00Z")
    ]

    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(chunks)

    # Must NOT create a supersession edge
    sup_edges = temp_graph_store.get_edges_by_type("supersedes")
    assert len(sup_edges) == 0

    claim_nodes = temp_graph_store.get_nodes_by_type("claim")
    assert len(claim_nodes) == 2
    for cn in claim_nodes:
        assert cn.properties.get("conflict_status") == "unresolved"


# ==============================================================================
# Test 6: Unrelated Subjects and Sessions Do Not Supersede One Another
# ==============================================================================

def test_unrelated_subjects_and_predicates_do_not_supersede(temp_graph_store):
    chunks = [
        make_chunk("chk_alice", "sess_01", "Alice lived in Delhi in 2025.", "2025-01-01T00:00:00Z"),
        make_chunk("chk_bob", "sess_02", "Bob moved to Pune in 2026.", "2026-01-01T00:00:00Z")
    ]

    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(chunks)

    # Alice and Bob must NOT supersede one another
    sup_edges = temp_graph_store.get_edges_by_type("supersedes")
    assert len(sup_edges) == 0

    claim_nodes = temp_graph_store.get_nodes_by_type("claim")
    assert len(claim_nodes) == 2
    for cn in claim_nodes:
        assert cn.properties.get("is_current") is True


# ==============================================================================
# Test 7: Unknown Dates Are Not Fabricated
# ==============================================================================

def test_unknown_dates_are_not_fabricated(temp_graph_store):
    chunks = [
        make_chunk("chk_undated", "sess_01", "I live in Delhi.", "2026-05-10T14:30:00Z")
    ]

    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(chunks)

    claim_nodes = temp_graph_store.get_nodes_by_type("claim")
    assert len(claim_nodes) == 1
    p = claim_nodes[0].properties
    assert p.get("valid_from") is None
    assert p.get("valid_to") is None
    # recorded_at retains session timestamp without fabricating event validity
    assert p.get("recorded_at") == "2026-05-10T14:30:00Z"


# ==============================================================================
# Test 8: Repeated Ingestion Is Strictly Idempotent
# ==============================================================================

def test_repeated_ingestion_is_strictly_idempotent(temp_graph_store):
    chunks = [
        make_chunk("chk_idemp_1", "sess_01", "I lived in Delhi in 2025.", "2025-06-01"),
        make_chunk("chk_idemp_2", "sess_02", "I moved to Pune in 2026.", "2026-02-15")
    ]

    builder = TurnGraphBuilder(temp_graph_store)

    # Run 1
    builder.build_from_chunks(chunks)
    nodes_run1 = temp_graph_store.count_nodes()
    edges_run1 = temp_graph_store.count_edges()

    # Run 2 (exact same chunks)
    builder.build_from_chunks(chunks)
    nodes_run2 = temp_graph_store.count_nodes()
    edges_run2 = temp_graph_store.count_edges()

    # Run 3
    builder.build_from_chunks(chunks)
    nodes_run3 = temp_graph_store.count_nodes()
    edges_run3 = temp_graph_store.count_edges()

    assert nodes_run1 == nodes_run2 == nodes_run3
    assert edges_run1 == edges_run2 == edges_run3
    assert len(temp_graph_store.get_nodes_by_type("claim")) == 2
    assert len(temp_graph_store.get_edges_by_type("supersedes")) == 1


# ==============================================================================
# Test 9: Citation Validation Works with Temporal Claims and Provenance Chunks
# ==============================================================================

def test_citation_validation_with_temporal_claims():
    retrieved_context = {
        "direct_evidence": [
            {"chunk_id": "chk_001", "memory_id": "sess_001", "text": "I moved to Pune in 2026.", "score": 0.95}
        ],
        "temporal_resolution": {
            "selected_claims": [
                TemporalClaim(
                    claim_id="claim_pune_2026",
                    subject="user",
                    predicate="lives_in",
                    object="Pune",
                    valid_from="2026",
                    source_chunk_id="chk_001"
                )
            ],
            "historical_claims": []
        }
    }

    # Citations citing chunk_id, memory_id, or claim_id are validated; fake IDs rejected
    cited = ["chk_001", "claim_pune_2026", "fake_chunk_999"]
    valid, rejected = validate_citations(cited, retrieved_context)

    valid_ids = [c["chunk_id"] for c in valid]
    assert "chk_001" in valid_ids
    assert "fake_chunk_999" in rejected


# ==============================================================================
# Test 10: Baseline Behavior Remains Unchanged When Temporal Reasoning is Disabled
# ==============================================================================

def test_baseline_behavior_unchanged_when_temporal_disabled(temp_graph_store):
    chunks = [
        make_chunk("chk_01", "sess_01", "I lived in Delhi in 2025. I moved to Pune in 2026.", "2026-02-15")
    ]

    mock_llm = lambda prompt, sys: {
        "answer": "Test answer without temporal modification.",
        "cited_memory_ids": ["chk_01"],
        "status": "answered",
        "inferred_claims": []
    }

    # Run with enable_temporal=False
    resp = answer_question(
        query="Where do I live?",
        chunks=chunks,
        graph_store=temp_graph_store,
        use_chunking=True,
        llm_caller=mock_llm,
        min_score=None,
        enable_temporal=False
    )

    ctx = resp["retrieved_context"]
    assert "temporal_resolution" not in ctx
    assert "--- CURRENT VALID CLAIMS ---" not in ctx.get("context_text", "")
