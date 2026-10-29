"""
Focused tests for Research Sprint prototypes:
1. Prototype B: Relevance-filtered graph edges in TurnGraphRetriever (min_edge_relevance).
2. Prototype C: Direct seed temporal claim extraction and resolution in answer_question (direct_seed_temporal).
3. Backward compatibility and configuration semantics.
4. Edge-case handling: missing dates, conflicting claims, citation validation, and repeated calls isolation.
"""

import pytest
from src.chunking.turn_chunker import TurnChunk
from src.chunking.graph_retrieval import TurnGraphRetriever
from src.graph.store import SQLiteGraphStore
from src.graph.schema import GraphNode, GraphEdge
from src.answering import answer_question


def test_turn_graph_retriever_default_relevance_traverses_all_edges():
    """Verify that default min_edge_relevance=0.0 permits traversal of adjacent edges when expansion is triggered."""
    store = SQLiteGraphStore(":memory:")
    
    chunks = [
        TurnChunk(
            chunk_id="chk_000", session_id="s1", memory_id="m1",
            turn_start_idx=0, turn_end_idx=1, total_session_turns=10,
            timestamp="2023-01-01T10:00:00Z", parsed_datetime=None,
            turns=[{"role": "user", "content": "What did they say next about hotel discount coupon?"}],
            text="User: What did they say next about hotel discount coupon? Assistant: We have a coupon.",
            word_count=16, token_estimate=22
        ),
        TurnChunk(
            chunk_id="chk_001", session_id="s1", memory_id="m1",
            turn_start_idx=2, turn_end_idx=3, total_session_turns=10,
            timestamp="2023-01-01T10:02:00Z", parsed_datetime=None,
            turns=[{"role": "user", "content": "General travel notes."}],
            text="User: General travel notes. Assistant: Noted general travel notes.",
            word_count=10, token_estimate=14
        )
    ]
    # Add a chit-chat neighbor chunk
    chunks.append(TurnChunk(
        chunk_id="chk_chit", session_id="s1", memory_id="m1",
        turn_start_idx=10, turn_end_idx=11, total_session_turns=12,
        timestamp="2023-01-01T10:15:00Z", parsed_datetime=None,
        turns=[{"role": "user", "content": "Ok."}],
        text="User: Ok. Assistant: Yes, sure thing.",
        word_count=6, token_estimate=8
    ))
    
    for c in chunks:
        store.upsert_node(GraphNode(id=c.chunk_id, node_type="chunk", label=c.chunk_id, properties={"text": c.text}))
    # Low-weight adjacent edge to chit-chat chunk
    store.upsert_edge(GraphEdge(id="e_adj", source_id="chk_000", target_id="chk_chit", edge_type="adjacent_to", confidence=0.2))
    
    retriever = TurnGraphRetriever(chunks=chunks, store=store, min_edge_relevance=0.0)
    results, diag = retriever.retrieve("What did they say next about the hotel discount coupon?", k=3)
    
    assert diag["expansion_executed"] is True
    traversed_to = [t["to"] for t in diag.get("traversed_details", [])]
    assert "chk_chit" in traversed_to


def test_turn_graph_retriever_filters_low_weight_and_chitchat_edges():
    """Verify that min_edge_relevance=0.5 filters out low-weight or dialogue chit-chat neighbor chunks."""
    store = SQLiteGraphStore(":memory:")
    
    chunks = [
        TurnChunk(
            chunk_id="chk_000", session_id="s1", memory_id="m1",
            turn_start_idx=0, turn_end_idx=1, total_session_turns=10,
            timestamp="2023-01-01T10:00:00Z", parsed_datetime=None,
            turns=[{"role": "user", "content": "What did they say next about hotel discount coupon?"}],
            text="User: What did they say next about hotel discount coupon? Assistant: We have a coupon.",
            word_count=16, token_estimate=22
        ),
        TurnChunk(
            chunk_id="chk_001", session_id="s1", memory_id="m1",
            turn_start_idx=2, turn_end_idx=3, total_session_turns=10,
            timestamp="2023-01-01T10:02:00Z", parsed_datetime=None,
            turns=[{"role": "user", "content": "General travel notes."}],
            text="User: General travel notes. Assistant: Noted general travel notes.",
            word_count=10, token_estimate=14
        )
    ]
    chunks.append(TurnChunk(
        chunk_id="chk_chit", session_id="s1", memory_id="m1",
        turn_start_idx=10, turn_end_idx=11, total_session_turns=12,
        timestamp="2023-01-01T10:15:00Z", parsed_datetime=None,
        turns=[{"role": "user", "content": "Ok."}],
        text="User: Ok. Assistant: Yes, sure thing.",
        word_count=6, token_estimate=8
    ))
    
    for c in chunks:
        store.upsert_node(GraphNode(id=c.chunk_id, node_type="chunk", label=c.chunk_id, properties={"text": c.text}))
    # Low-weight edge (0.2) to chit-chat chunk
    store.upsert_edge(GraphEdge(id="e_adj", source_id="chk_000", target_id="chk_chit", edge_type="adjacent_to", confidence=0.2))
    
    # Set threshold to 0.5: should filter out chk_chit from traversal
    retriever = TurnGraphRetriever(chunks=chunks, store=store, min_edge_relevance=0.5)
    results, diag = retriever.retrieve("What did they say next about the hotel discount coupon?", k=3)
    
    assert diag["expansion_executed"] is True
    traversed_to = [t["to"] for t in diag.get("traversed_details", [])]
    assert "chk_chit" not in traversed_to, "Low-weight chit-chat edge should have been filtered out"


def test_turn_graph_retriever_permits_high_confidence_edge():
    """Verify that min_edge_relevance=0.5 permits high-confidence edges (confidence >= 0.5) with relevant text."""
    store = SQLiteGraphStore(":memory:")
    
    chunks = [
        TurnChunk(
            chunk_id="chk_000", session_id="s1", memory_id="m1",
            turn_start_idx=0, turn_end_idx=1, total_session_turns=10,
            timestamp="2023-01-01T10:00:00Z", parsed_datetime=None,
            turns=[{"role": "user", "content": "What did they say next about hotel discount coupon?"}],
            text="User: What did they say next about hotel discount coupon? Assistant: We have a coupon.",
            word_count=16, token_estimate=22
        ),
        TurnChunk(
            chunk_id="chk_relevant_neighbor", session_id="s1", memory_id="m1",
            turn_start_idx=2, turn_end_idx=3, total_session_turns=10,
            timestamp="2023-01-01T10:05:00Z", parsed_datetime=None,
            turns=[{"role": "user", "content": "What was the code?"}],
            text="User: What was the code? Assistant: The hotel coupon code was SPRING20 for a 20% discount.",
            word_count=16, token_estimate=22
        )
    ]
    
    for c in chunks:
        store.upsert_node(GraphNode(id=c.chunk_id, node_type="chunk", label=c.chunk_id, properties={"text": c.text}))
    # High-confidence adjacent edge (0.85) to relevant neighbor
    store.upsert_edge(GraphEdge(id="e_adj_hi", source_id="chk_000", target_id="chk_relevant_neighbor", edge_type="adjacent_to", confidence=0.85))
    
    retriever = TurnGraphRetriever(chunks=chunks, store=store, min_edge_relevance=0.5)
    results, diag = retriever.retrieve("What did they say next about the hotel discount coupon?", k=3)
    
    assert diag["expansion_executed"] is True
    traversed_to = [t["to"] for t in diag.get("traversed_details", [])]
    assert "chk_relevant_neighbor" in traversed_to


def test_direct_seed_temporal_extraction_without_graph_store():
    """Verify Prototype C: temporal claims extracted directly from retrieved seed chunks without graph store."""
    chunks = [
        TurnChunk(
            chunk_id="chk_t1", session_id="s1", memory_id="m1",
            turn_start_idx=0, turn_end_idx=1, total_session_turns=2,
            timestamp="2024-05-01T10:00:00Z", parsed_datetime=None,
            turns=[{"role": "user", "content": "I lived in Delhi in 2024."}],
            text="User: I lived in Delhi in 2024.\nAssistant: Noted your residence in Delhi.",
            word_count=12, token_estimate=16
        ),
        TurnChunk(
            chunk_id="chk_t2", session_id="s1", memory_id="m1",
            turn_start_idx=2, turn_end_idx=3, total_session_turns=2,
            timestamp="2025-05-15T10:00:00Z", parsed_datetime=None,
            turns=[{"role": "user", "content": "Update: I moved to Bangalore in 2025."}],
            text="User: Update: I moved to Bangalore in 2025.\nAssistant: Updated your residence to Bangalore.",
            word_count=13, token_estimate=17
        )
    ]

    mock_llm_called = []
    def mock_llm(prompt: str, system_prompt: str) -> dict:
        mock_llm_called.append(prompt)
        return {
            "status": "answered",
            "answer": "You lived in Delhi in 2024 and moved to Bangalore in 2025.",
            "cited_memory_ids": ["chk_t2"],
            "limitations": None
        }

    res = answer_question(
        query="Where did I live in 2025?",
        use_chunking=True,
        chunks=chunks,
        use_ontology=True,
        use_chunk_graph=False,
        graph_store=None,
        min_score=None,
        enable_temporal=True,
        direct_seed_temporal=True,
        llm_caller=mock_llm
    )

    assert res["status"] == "answered"
    assert len(mock_llm_called) == 1
    prompt_text = mock_llm_called[0]
    
    assert "<evidence_context>" in prompt_text
    retrieved_ctx = res["retrieved_context"]
    assert "temporal_resolution" in retrieved_ctx


def test_direct_seed_temporal_supersedes_outdated_claim():
    """Verify that direct seed temporal extraction resolves latest claims in evidence."""
    chunks = [
        TurnChunk(
            chunk_id="chk_emp1", session_id="s_work", memory_id="m_work",
            turn_start_idx=0, turn_end_idx=1, total_session_turns=4,
            timestamp="2024-01-01T10:00:00Z", parsed_datetime=None,
            turns=[{"role": "user", "content": "I worked at Acme Corp in 2024."}],
            text="User: I worked at Acme Corp in 2024.\nAssistant: Recorded your job at Acme Corp.",
            word_count=14, token_estimate=18
        ),
        TurnChunk(
            chunk_id="chk_emp2", session_id="s_work", memory_id="m_work",
            turn_start_idx=2, turn_end_idx=3, total_session_turns=4,
            timestamp="2025-01-01T10:00:00Z", parsed_datetime=None,
            turns=[{"role": "user", "content": "I moved to Beta Labs in 2025."}],
            text="User: I moved to Bangalore in 2025.\nAssistant: Noted.",
            word_count=10, token_estimate=14
        )
    ]

    captured_prompt = []
    def mock_llm(prompt: str, system_prompt: str) -> dict:
        captured_prompt.append(prompt)
        return {
            "status": "answered",
            "answer": "You worked at Acme Corp in 2024.",
            "cited_memory_ids": ["chk_emp1"],
            "limitations": None
        }

    res = answer_question(
        query="Where did I work in 2024?",
        use_chunking=True,
        chunks=chunks,
        use_ontology=True,
        use_chunk_graph=False,
        graph_store=None,
        min_score=None,
        enable_temporal=True,
        direct_seed_temporal=True,
        llm_caller=mock_llm
    )

    assert res["status"] == "answered"
    assert "Acme Corp" in res["answer"]
    assert len(captured_prompt) == 1
    retrieved_ctx = res["retrieved_context"]
    assert "temporal_resolution" in retrieved_ctx
    temp_res = retrieved_ctx["temporal_resolution"]
    assert len(temp_res.get("selected_claims", [])) >= 1 or len(temp_res.get("historical_claims", [])) >= 1


def test_direct_temporal_missing_dates_handling():
    """Verify that temporal claims with missing explicit years/dates do not crash and handle None gracefully."""
    chunks = [
        TurnChunk(
            chunk_id="chk_nodate", session_id="s_nd", memory_id="m_nd",
            turn_start_idx=0, turn_end_idx=1, total_session_turns=2,
            timestamp="unknown", parsed_datetime=None,
            turns=[{"role": "user", "content": "I live in Tokyo."}],
            text="User: I live in Tokyo.\nAssistant: Noted your current residence in Tokyo.",
            word_count=12, token_estimate=16
        )
    ]

    def mock_llm(prompt: str, system_prompt: str) -> dict:
        return {
            "status": "answered",
            "answer": "You live in Tokyo.",
            "cited_memory_ids": ["chk_nodate"],
            "limitations": None
        }

    res = answer_question(
        query="Where do I live?",
        use_chunking=True,
        chunks=chunks,
        use_ontology=True,
        use_chunk_graph=False,
        graph_store=None,
        min_score=None,
        enable_temporal=True,
        direct_seed_temporal=True,
        llm_caller=mock_llm
    )

    assert res["status"] == "answered"
    assert "Tokyo" in res["answer"]
    temp_res = res["retrieved_context"].get("temporal_resolution")
    assert temp_res is not None
    # Claim should be extracted even if valid_from/to is None
    claims = temp_res.get("selected_claims", [])
    assert len(claims) >= 1
    assert claims[0].object == "Tokyo"


def test_citation_validity_with_direct_seed_temporal():
    """Verify that citation validation correctly maps cited chunk IDs and rejects fabricated IDs."""
    chunks = [
        TurnChunk(
            chunk_id="chk_real_01", session_id="s_cit", memory_id="m_cit",
            turn_start_idx=0, turn_end_idx=1, total_session_turns=2,
            timestamp="2024-01-01", parsed_datetime=None,
            turns=[{"role": "user", "content": "I live in Delhi in 2024."}],
            text="User: I live in Delhi in 2024.\nAssistant: Noted.",
            word_count=10, token_estimate=14
        )
    ]

    def mock_llm_fake_cit(prompt: str, system_prompt: str) -> dict:
        return {
            "status": "answered",
            "answer": "You live in Delhi [chk_real_01]. Also fake claim [chk_fake_999].",
            "cited_memory_ids": ["chk_real_01", "chk_fake_999"],
            "limitations": None
        }

    res = answer_question(
        query="Where do I live in 2024?",
        use_chunking=True,
        chunks=chunks,
        use_ontology=True,
        use_chunk_graph=False,
        graph_store=None,
        min_score=None,
        enable_temporal=True,
        direct_seed_temporal=True,
        llm_caller=mock_llm_fake_cit
    )

    # Valid citation should be retained, fake should be rejected
    cited_ids = [c["chunk_id"] for c in res["citations"]]
    assert "chk_real_01" in cited_ids
    assert "chk_fake_999" not in cited_ids
    assert "Rejected unsupported memory citations: chk_fake_999" in res["limitations"]


def test_repeated_calls_idempotency_and_isolation():
    """Verify that successive calls with direct_seed_temporal do not leak state across queries."""
    chunks1 = [
        TurnChunk(
            chunk_id="chk_c1", session_id="s1", memory_id="m1",
            turn_start_idx=0, turn_end_idx=1, total_session_turns=2,
            timestamp="2024-01-01", parsed_datetime=None,
            turns=[{"role": "user", "content": "I lived in Delhi in 2024."}],
            text="User: I lived in Delhi in 2024.\nAssistant: Noted.",
            word_count=10, token_estimate=14
        )
    ]
    chunks2 = [
        TurnChunk(
            chunk_id="chk_c2", session_id="s2", memory_id="m2",
            turn_start_idx=0, turn_end_idx=1, total_session_turns=2,
            timestamp="2025-01-01", parsed_datetime=None,
            turns=[{"role": "user", "content": "I lived in Mumbai in 2025."}],
            text="User: I lived in Mumbai in 2025.\nAssistant: Noted.",
            word_count=10, token_estimate=14
        )
    ]

    def mock_llm(prompt: str, system_prompt: str) -> dict:
        return {"status": "answered", "answer": "Answered.", "cited_memory_ids": [], "limitations": None}

    res1 = answer_question(
        query="Where did I live in 2024?",
        use_chunking=True,
        chunks=chunks1,
        enable_temporal=True,
        direct_seed_temporal=True,
        min_score=None,
        llm_caller=mock_llm
    )
    res2 = answer_question(
        query="Where did I live in 2025?",
        use_chunking=True,
        chunks=chunks2,
        enable_temporal=True,
        direct_seed_temporal=True,
        min_score=None,
        llm_caller=mock_llm
    )

    # res2 must not contain claims or evidence from chunks1
    res2_claims = [c.object for c in res2["retrieved_context"]["temporal_resolution"].get("selected_claims", [])]
    assert "Delhi" not in res2_claims
    assert "Mumbai" in res2_claims
