"""
Unit and integration tests for the IVARA Graph Ablation configurations:
1. B-Chunked (baseline)
2. B-Chunked + Ontology (OntologyGuidedChunkRetriever)
3. B-Chunked + Graph (TurnGraphBuilder and TurnGraphRetriever)
"""

import tempfile
from pathlib import Path
import pytest

from src.chunking.turn_chunker import TurnChunk
from src.chunking.ontology_retrieval import OntologyGuidedChunkRetriever
from src.chunking.graph_retrieval import TurnGraphBuilder, TurnGraphRetriever
from src.graph.store import SQLiteGraphStore
from src.answering import answer_question


@pytest.fixture
def sample_turn_chunks():
    """Create a controlled set of conversational TurnChunks for testing."""
    return [
        TurnChunk(
            chunk_id="sess_01_c000",
            session_id="01",
            memory_id="sess_01",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=5,
            timestamp="2024-03-01",
            parsed_datetime=None,
            turns=[],
            text="User: I went to Target yesterday to buy some groceries.\nAssistant: Did you find everything you needed?\nUser: Yes, I bought some milk and vegetables.",
            word_count=23,
            token_estimate=30
        ),
        TurnChunk(
            chunk_id="sess_01_c001",
            session_id="01",
            memory_id="sess_01",
            turn_start_idx=2,
            turn_end_idx=4,
            total_session_turns=5,
            timestamp="2024-03-01",
            parsed_datetime=None,
            turns=[],
            text="User: Yes, I bought some milk and vegetables.\nAssistant: Was there any discount or coupon applied?\nUser: I used a coupon that gave me 20% off my total purchase.",
            word_count=26,
            token_estimate=35
        ),
        TurnChunk(
            chunk_id="sess_02_c000",
            session_id="02",
            memory_id="sess_02",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2024-04-10",
            parsed_datetime=None,
            turns=[],
            text="User: I graduated from college last year.\nAssistant: That is great, what was your field of study?\nUser: I graduated with a degree in Business Administration.",
            word_count=24,
            token_estimate=32
        ),
        TurnChunk(
            chunk_id="sess_03_c000",
            session_id="03",
            memory_id="sess_03",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2024-05-15",
            parsed_datetime=None,
            turns=[],
            text="User: I am building a 1/72 scale B-29 bomber aircraft model kit.\nAssistant: That sounds like a detailed project.\nUser: It has over 200 plastic pieces.",
            word_count=26,
            token_estimate=35
        )
    ]


@pytest.fixture
def temp_graph_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_turn_graph.db"
        store = SQLiteGraphStore(db_path)
        try:
            yield store
        finally:
            store.close()


def test_ontology_concept_extraction_and_alias_expansion(sample_turn_chunks):
    """Verify ontology retriever extracts entities and expands domain aliases without graph."""
    retriever = OntologyGuidedChunkRetriever(sample_turn_chunks)
    concepts = retriever.extract_query_concepts("What percentage discount was the coupon I used at Target?")

    assert "Target" in concepts["entities"] or "target" in [e.lower() for e in concepts["entities"]]
    assert any(a in concepts["aliases"] for a in ["discount", "percentage", "off", "store"])


def test_ontology_guided_retrieval_re_ranking(sample_turn_chunks):
    """Verify ontology retriever boosts chunk matching query entities."""
    retriever = OntologyGuidedChunkRetriever(sample_turn_chunks)
    res = retriever.retrieve("What degree did I graduate with in college?", k=2)

    assert len(res) > 0
    top_hit = res[0]
    assert top_hit["chunk_id"] == "sess_02_c000"
    assert "Business Administration" in top_hit["text"]
    assert "ontology_bonus" in top_hit
    assert top_hit["ontology_bonus"] > 0.0


def test_turn_graph_builder_construction(temp_graph_store, sample_turn_chunks):
    """Verify TurnGraphBuilder populates ChunkNodes, EpisodeNodes, and adjacent_to edges."""
    builder = TurnGraphBuilder(temp_graph_store)
    stats = builder.build_from_chunks(sample_turn_chunks)

    assert stats["nodes"] >= len(sample_turn_chunks)
    assert stats["edges"] >= len(sample_turn_chunks)

    # Check ChunkNode
    c0 = temp_graph_store.get_node("sess_01_c000")
    assert c0 is not None
    assert c0.node_type == "chunk"
    assert "Target" in c0.properties["text"]

    # Check EpisodeNode
    ep1 = temp_graph_store.get_node("sess_01")
    assert ep1 is not None

    # Check adjacent_to edge between sess_01_c000 and sess_01_c001
    out_edges = temp_graph_store.get_outgoing_edges("sess_01_c000")
    edge_types = [e.edge_type for e in out_edges]
    assert "adjacent_to" in edge_types
    adj_edge = [e for e in out_edges if e.edge_type == "adjacent_to"][0]
    assert adj_edge.target_id == "sess_01_c001"


def test_turn_graph_retrieval_adjacent_discovery(temp_graph_store, sample_turn_chunks):
    """
    Key test: Query mentions 'Target' and 'coupon'.
    sess_01_c000 has 'Target' (Turn 0-2).
    sess_01_c001 has 'coupon 20% off' (Turn 2-4).
    Graph traversal along 'adjacent_to' should connect Target to the coupon chunk!
    """
    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(sample_turn_chunks)

    retriever = TurnGraphRetriever(sample_turn_chunks, temp_graph_store)
    results, diag = retriever.retrieve(
        query="What percentage discount was the coupon I used at Target?",
        k=2
    )

    assert len(results) >= 2
    res_cids = [r["chunk_id"] for r in results]
    # Both Target chunk and coupon chunk must be retrieved
    assert "sess_01_c000" in res_cids
    assert "sess_01_c001" in res_cids
    assert diag["nodes_expanded"] > 0


def test_answer_question_routing_ablation(temp_graph_store, sample_turn_chunks):
    """Verify answer_question handles use_ontology and use_chunk_graph routing cleanly."""
    mock_caller = lambda p, s: '{"status": "answered", "answer": "You graduated with Business Administration.", "cited_memory_ids": ["sess_02"]}'

    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(sample_turn_chunks)

    # 1. Config A: B-Chunked Baseline
    ans_a = answer_question(
        query="What degree did I graduate with in college?",
        use_chunking=True,
        chunks=sample_turn_chunks,
        llm_caller=mock_caller
    )
    assert ans_a["status"] == "answered"

    # 2. Config B: B-Chunked + Ontology
    ans_b = answer_question(
        query="What degree did I graduate with in college?",
        use_chunking=True,
        chunks=sample_turn_chunks,
        use_ontology=True,
        llm_caller=mock_caller
    )
    assert ans_b["status"] == "answered"

    # 3. Config C: B-Chunked + Graph
    ans_c = answer_question(
        query="What degree did I graduate with in college?",
        use_chunking=True,
        chunks=sample_turn_chunks,
        use_chunk_graph=True,
        graph_store=temp_graph_store,
        llm_caller=mock_caller
    )
    assert ans_c["status"] == "answered"
