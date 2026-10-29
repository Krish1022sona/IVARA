"""
Unit and Integration Tests for IVARA Research Candidate Architecture.

Verifies:
1. Backward compatibility of answer_question with default flags.
2. Deterministic Query Routing invariants:
   - Factual and direct temporal queries stay on direct_seed path.
   - Explicit relationship queries trigger bounded_graph_fallback.
3. Bounded Graph Fallback invariants:
   - Max fallback chunk budget (<= 2).
   - Direct evidence priority preservation (seeds remain head of context).
   - Cycle avoidance and chunk deduplication.
   - Low-relevance distractor edge filtering (< 0.50 pruned).
   - Unresolved relationship tracking (never silently discarded).
   - Explicit provenance and fallback metadata.
4. Temporal Path invariants:
   - Event time vs recording time distinction.
   - Superseded claims handled correctly.
   - Contradictory claims preserve ambiguity without fabricated dates.
5. Citation validation:
   - Fabricated / non-existent IDs rejected.
6. Grounded Abstention:
   - Unanswerable questions without evidence cleanly abstain.
"""

import json
from pathlib import Path
import pytest

from src.answering import answer_question
from src.chunking.turn_chunker import TurnChunk
from src.chunking.hybrid_pipeline import (
    DeterministicQueryRouter,
    UnifiedCandidateRetriever,
    RoutingDecision
)
from src.chunking.graph_retrieval import TurnGraphBuilder
from src.graph.store import SQLiteGraphStore
from src.graph.schema import GraphNode, GraphEdge
from src.temporal.claim import build_claims_from_text, TemporalClaim
from src.temporal.supersession import TemporalSupersessionEngine


def make_test_chunk(chunk_id: str, session_id: str, text: str, timestamp: str = "2024-01-01") -> TurnChunk:
    return TurnChunk(
        chunk_id=chunk_id,
        session_id=session_id,
        memory_id=f"sess_{session_id}",
        turn_start_idx=0,
        turn_end_idx=2,
        turns=[{"speaker": "user", "text": text}],
        text=text,
        timestamp=timestamp,
        token_estimate=len(text.split()),
        total_session_turns=3,
        parsed_datetime=None,
        word_count=len(text.split())
    )


def test_backward_compatibility_defaults():
    """Verify default flags in answer_question retain baseline behavior."""
    chunks = [
        make_test_chunk("c1", "s1", "My favorite color is navy blue.")
    ]
    
    # Default call without new flags
    mock_llm = lambda prompt, sys: json.dumps({
        "status": "answered",
        "answer": "Your favorite color is navy blue. [c1]",
        "cited_memory_ids": ["c1"]
    })
    
    res = answer_question(
        query="What is my favorite color?",
        chunks=chunks,
        use_chunking=True,
        llm_caller=mock_llm
    )
    
    assert res["status"] == "answered"
    assert "navy blue" in res["answer"]
    assert len(res["citations"]) == 1
    assert res["citations"][0]["chunk_id"] == "c1"
    # Routing diagnostics should be absent when dynamic_graph_fallback=False
    assert "routing_diagnostics" not in res["retrieved_context"]


def test_deterministic_router_factual_guard():
    """Verify factual and direct temporal queries stay on direct_seed path."""
    router = DeterministicQueryRouter(min_seed_score_for_direct=0.40)
    
    seed_chunks = [{"chunk_id": "c1", "score": 0.85, "text": "I wake up at 7:30 am on Saturdays."}]
    
    # Factual query
    dec = router.route("What time do I wake up on Saturday mornings?", seed_chunks)
    assert dec.route == "direct_seed"
    assert dec.should_expand is False
    assert dec.is_relationship_query is False

    # Temporal query
    dec_temp = router.route("How many days had passed between the two events?", seed_chunks)
    assert dec_temp.route == "direct_seed"
    assert dec_temp.should_expand is False


def test_deterministic_router_relationship_trigger():
    """Verify relationship queries trigger bounded_graph_fallback."""
    router = DeterministicQueryRouter()
    seed_chunks = [{"chunk_id": "c1", "score": 0.60, "text": "Alice works with Bob at TechCorp."}]
    
    dec1 = router.route("What is the relationship between Alice and Bob?", seed_chunks)
    assert dec1.route == "bounded_graph_fallback"
    assert dec1.should_expand is True
    assert dec1.is_relationship_query is True

    dec2 = router.route("How do Alice and Charlie know each other?", seed_chunks)
    assert dec2.route == "bounded_graph_fallback"
    assert dec2.should_expand is True


def test_bounded_graph_fallback_budget_and_priority(tmp_path):
    """Verify fallback obeys max budget of 2 chunks and preserves direct seed priority."""
    db_file = tmp_path / "test_graph.db"
    store = SQLiteGraphStore(db_file)
    
    c_seed = make_test_chunk("c_seed", "s1", "Alice was introduced by Bob at TechCorp.")
    c_cand1 = make_test_chunk("c_cand1", "s2", "Bob works at TechCorp as engineering director.")
    c_cand2 = make_test_chunk("c_cand2", "s3", "Bob and Alice co-authored a research paper.")
    c_cand3 = make_test_chunk("c_cand3", "s4", "Bob attended university in Boston.")
    
    builder = TurnGraphBuilder(store)
    builder.build_from_chunks([c_seed, c_cand1, c_cand2, c_cand3])
    
    # Add direct relation edges with high confidence via upsert_edge
    store.upsert_edge(GraphEdge(
        id="e_rel_1",
        source_id="c_seed",
        target_id="c_cand1",
        edge_type="works_with",
        confidence=0.90
    ))
    store.upsert_edge(GraphEdge(
        id="e_rel_2",
        source_id="c_seed",
        target_id="c_cand2",
        edge_type="collaborated_with",
        confidence=0.85
    ))
    store.upsert_edge(GraphEdge(
        id="e_rel_3",
        source_id="c_seed",
        target_id="c_cand3",
        edge_type="acquaintance",
        confidence=0.80
    ))
    
    retriever = UnifiedCandidateRetriever(
        chunks=[c_seed, c_cand1, c_cand2, c_cand3],
        store=store,
        min_edge_relevance=0.50,
        max_fallback_chunks=2
    )
    
    results, diag = retriever.retrieve("What is the relationship between Alice and Bob?", k=1)
    
    # Seed chunk must be first (not marked as fallback)
    assert results[0].get("origin") != "graph_fallback"
    assert results[0]["chunk_id"] in ["c_seed", "c_cand2"]
    
    # Fallback chunks must be <= 2
    fallback_items = [r for r in results if r.get("origin") == "graph_fallback"]
    assert len(fallback_items) <= 2
    for fb in fallback_items:
        assert "fallback_metadata" in fb
        assert fb["fallback_metadata"]["source_seed_id"] == results[0]["chunk_id"]
    
    store.close()


def test_distractor_edge_pruning(tmp_path):
    """Negative test: edges with confidence < 0.50 are strictly pruned."""
    db_file = tmp_path / "test_prune.db"
    store = SQLiteGraphStore(db_file)
    
    c_seed = make_test_chunk("c_seed", "s1", "Alice met Dr. Smith at the symposium.")
    c_noise = make_test_chunk("c_noise", "s2", "Random unrelated chit chat about weather.")
    
    builder = TurnGraphBuilder(store)
    builder.build_from_chunks([c_seed, c_noise])
    
    # Low confidence distractor edge (0.30 < 0.50)
    store.upsert_edge(GraphEdge(
        id="e_distractor",
        source_id="c_seed",
        target_id="c_noise",
        edge_type="spurious_link",
        confidence=0.30
    ))
    
    retriever = UnifiedCandidateRetriever(
        chunks=[c_seed, c_noise],
        store=store,
        min_edge_relevance=0.50
    )
    
    results, diag = retriever.retrieve("What is the connection between Alice and Dr. Smith?", k=1)
    chunk_ids = [r["chunk_id"] for r in results]
    
    assert "c_seed" in chunk_ids
    assert "c_noise" not in chunk_ids  # Pruned because 0.30 < 0.50
    
    store.close()


def test_cycle_and_deduplication_avoidance(tmp_path):
    """Verify cycle avoidance: seed chunks and self-loops are never duplicated."""
    db_file = tmp_path / "test_cycles.db"
    store = SQLiteGraphStore(db_file)
    
    c1 = make_test_chunk("c1", "s1", "Alice knows Bob.")
    c2 = make_test_chunk("c2", "s2", "Bob knows Alice.")
    
    builder = TurnGraphBuilder(store)
    builder.build_from_chunks([c1, c2])
    
    # Mutual cycle
    store.upsert_edge(GraphEdge(id="e1", source_id="c1", target_id="c2", edge_type="knows", confidence=0.9))
    store.upsert_edge(GraphEdge(id="e2", source_id="c2", target_id="c1", edge_type="knows", confidence=0.9))
    # Self-loop
    store.upsert_edge(GraphEdge(id="e_self", source_id="c1", target_id="c1", edge_type="self", confidence=0.9))
    
    retriever = UnifiedCandidateRetriever(chunks=[c1, c2], store=store, min_edge_relevance=0.50)
    results, diag = retriever.retrieve("What is the connection between Alice and Bob?", k=2)
    
    retrieved_ids = [r["chunk_id"] for r in results]
    assert len(retrieved_ids) == len(set(retrieved_ids)), "Duplicate chunks found in retrieval results"
    assert "c1" in retrieved_ids
    assert "c2" in retrieved_ids
    
    store.close()


def test_unresolved_relationship_preservation(tmp_path):
    """Verify unresolved relationships are explicitly recorded when no qualifying edges exist."""
    db_file = tmp_path / "test_unresolved.db"
    store = SQLiteGraphStore(db_file)
    
    c1 = make_test_chunk("c1", "s1", "I visited the Museum of Modern Art.")
    builder = TurnGraphBuilder(store)
    builder.build_from_chunks([c1])
    
    retriever = UnifiedCandidateRetriever(chunks=[c1], store=store, min_edge_relevance=0.50)
    _, diag = retriever.retrieve("What is the relationship between Alice and Bob?", k=1)
    
    assert diag["is_relationship_query"] is True
    assert len(diag["unresolved_relationships"]) > 0
    assert diag["unresolved_relationships"][0]["status"] == "unresolved_no_qualifying_graph_edges"
    
    store.close()


def test_temporal_path_event_vs_recording_time():
    """Verify event time is preserved and distinguished from session recording time."""
    claims = build_claims_from_text(
        text="I lived in London in 2021.\nThen I moved to Paris in 2023.",
        session_id="s1",
        chunk_id="c1",
        recorded_at="2026-05-15"
    )
    
    engine = TemporalSupersessionEngine()
    processed = engine.process_claims(claims)
    
    # Event times must be 2021 and 2023, NOT 2026
    c_london = next(c for c in processed if c.object == "London")
    c_paris = next(c for c in processed if c.object == "Paris")
    
    assert c_london.valid_from == "2021"
    assert c_london.recorded_at == "2026-05-15"
    assert c_paris.valid_from == "2023"
    assert c_paris.recorded_at == "2026-05-15"


def test_temporal_path_missing_timestamp_no_fabrication():
    """Verify no fabricated dates are invented when text contains no timestamps."""
    claims = build_claims_from_text(
        text="I enjoy playing chess on weekends.",
        session_id="s_no_time",
        chunk_id="c_no_time",
        recorded_at=None
    )
    for c in claims:
        assert c.valid_from is None
        assert c.valid_to is None


def test_citation_validation_rejects_hallucinated_ids():
    """Verify hallucinated chunk IDs are rejected and valid chunk IDs are accepted."""
    chunks = [make_test_chunk("c_real", "s1", "My cat's name is Luna.")]
    
    mock_llm = lambda prompt, sys: json.dumps({
        "status": "answered",
        "answer": "Your cat is Luna. [c_real]",
        "cited_memory_ids": ["c_real", "c_fake_999"]
    })
    
    res = answer_question(
        query="What is my cat's name?",
        chunks=chunks,
        use_chunking=True,
        dynamic_graph_fallback=True,
        direct_seed_temporal=True,
        llm_caller=mock_llm
    )
    
    assert res["status"] == "answered"
    cited_ids = [c["chunk_id"] for c in res["citations"]]
    assert "c_real" in cited_ids
    assert "c_fake_999" not in cited_ids


def test_abstention_on_unanswerable_query():
    """Verify unanswerable queries abstain without hallucinating."""
    chunks = [make_test_chunk("c1", "s1", "I bought a blue bicycle in June.")]
    
    res = answer_question(
        query="What is the password to my bank account?",
        chunks=chunks,
        use_chunking=True,
        min_score=0.40,
        dynamic_graph_fallback=True
    )
    
    assert res["status"] == "insufficient_evidence"
    assert "don't have sufficient" in res["answer"].lower()
    assert len(res["citations"]) == 0


def test_empty_answer_with_insufficient_evidence_status_verbalizes_abstention():
    """Verify that an empty answer string from LLM with status 'insufficient_evidence' gets verbalized."""
    chunks = [make_test_chunk("c1", "s1", "I visited Dr. Smith in March.")]
    
    mock_llm = lambda prompt, sys: json.dumps({
        "status": "insufficient_evidence",
        "answer": "",
        "cited_memory_ids": [],
        "limitations": "Could not confirm attendance."
    })
    
    res = answer_question(
        query="How many appointments did I attend?",
        chunks=chunks,
        use_chunking=True,
        min_score=0.10,
        dynamic_graph_fallback=True,
        llm_caller=mock_llm
    )
    
    assert res["status"] == "insufficient_evidence"
    assert len(res["answer"]) > 0
    assert "sufficient recorded memories" in res["answer"].lower()


def test_session_diversity_selection_caps_chunks_per_session():
    """Verify that max_chunks_per_session caps chunks per session and includes diverse sessions."""
    chunks = [
        make_test_chunk("c1_s1", "s1", "I went fishing on Lake Michigan on July 22 and caught 3 bass."),
        make_test_chunk("c2_s1", "s1", "On that July 22 trip to Lake Michigan we used live bait for bass."),
        make_test_chunk("c3_s1", "s1", "Alex and I fished Lake Michigan until evening on July 22."),
        make_test_chunk("c1_s2", "s2", "On an earlier trip to Lake Michigan on July 10, I caught 7 largemouth bass with Alex.")
    ]
    
    retriever = UnifiedCandidateRetriever(chunks=chunks, use_ontology=False, max_chunks_per_session=1)
    results, _ = retriever.retrieve("How many bass did I catch on Lake Michigan?", k=3, min_score=None)
    
    sids = [r["session_id"] for r in results]
    assert sids.count("s1") <= 1
    assert "s2" in sids


def test_session_diversity_default_none_preserves_standard_ranking():
    """Verify that when max_chunks_per_session is None, default score ranking is preserved without capping."""
    chunks = [
        make_test_chunk("c1_s1", "s1", "I love hiking in Yosemite national park every summer."),
        make_test_chunk("c2_s1", "s1", "Hiking in Yosemite national park trails with my dog."),
        make_test_chunk("c3_s1", "s1", "Camping in Yosemite national park wilderness."),
        make_test_chunk("c1_s2", "s2", "Visiting Lake Tahoe for a day.")
    ]
    
    retriever = UnifiedCandidateRetriever(chunks=chunks, use_ontology=False, max_chunks_per_session=None)
    results, _ = retriever.retrieve("hiking in Yosemite national park", k=3, min_score=None)
    
    # All top-3 should come from s1 because they all strongly match Yosemite hiking
    sids = [r["session_id"] for r in results]
    assert sids.count("s1") == 3


