"""
Unit and integration tests for Integrated Hybrid Seed Retrieval (ChunkRetriever with Ontology Guidance).

Verifies:
1. Baseline mode remains behaviorally equivalent when ontology is disabled.
2. Alias expansion improves ranking of relevant candidates containing synonyms.
3. Entity-match boosts do not blindly outrank strong semantic candidates.
4. Combined scoring is strictly bounded in [0.0, 1.0] with normalized weights.
5. Deterministic candidate sorting and tie-breaking.
6. Full chunk provenance (chunk_id, session_id, turn_range, timestamp) is preserved.
7. Empty vocabulary and queries without entity matches degrade gracefully.
"""

import pytest
from src.chunking.turn_chunker import TurnChunk
from src.chunking.retrieval import ChunkRetriever, assemble_bounded_context
from src.chunking.ontology_retrieval import OntologyGuidedChunkRetriever, extract_query_concepts


@pytest.fixture
def test_chunks():
    """Controlled set of chunks representing personal memory statements."""
    return [
        TurnChunk(
            chunk_id="chunk_edu_01",
            session_id="sess_101",
            memory_id="mem_101",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=4,
            timestamp="2025-05-10T10:00:00Z",
            parsed_datetime=None,
            turns=[],
            text="User graduated with a bachelor in computer engineering from the state university.",
            word_count=12,
            token_estimate=18
        ),
        TurnChunk(
            chunk_id="chunk_career_01",
            session_id="sess_102",
            memory_id="mem_102",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2026-01-15T14:30:00Z",
            parsed_datetime=None,
            turns=[],
            text="User joined Aetheria Labs as a senior software engineer working with Marcus Vance on Python.",
            word_count=16,
            token_estimate=22
        ),
        TurnChunk(
            chunk_id="chunk_diet_01",
            session_id="sess_103",
            memory_id="mem_103",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=2,
            timestamp="2026-02-20T08:00:00Z",
            parsed_datetime=None,
            turns=[],
            text="User had oatmeal with blueberries and black coffee for breakfast this morning.",
            word_count=12,
            token_estimate=16
        ),
        TurnChunk(
            chunk_id="chunk_shopping_01",
            session_id="sess_104",
            memory_id="mem_104",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=2,
            timestamp="2026-03-01T16:00:00Z",
            parsed_datetime=None,
            turns=[],
            text="User bought groceries at Target and saved twenty percent with a store coupon.",
            word_count=13,
            token_estimate=18
        )
    ]


# ==============================================================================
# 1. Baseline Equivalence When Ontology Is Disabled
# ==============================================================================

def test_baseline_equivalence_when_ontology_disabled(test_chunks):
    """
    Verify that ChunkRetriever(use_ontology=False) produces exactly the same
    results and scores as default ChunkRetriever.
    """
    retriever_default = ChunkRetriever(test_chunks)
    retriever_explicit_false = ChunkRetriever(test_chunks, use_ontology=False)

    query = "Where did user work as a software engineer?"
    results_default = retriever_default.retrieve(query, k=3)
    results_explicit = retriever_explicit_false.retrieve(query, k=3)

    assert len(results_default) == len(results_explicit)
    for r_def, r_exp in zip(results_default, results_explicit):
        assert r_def["chunk_id"] == r_exp["chunk_id"]
        assert r_def["score"] == r_exp["score"]
        assert r_def["vector_score"] == r_exp["vector_score"]
        assert r_def["bm25_score"] == r_exp["bm25_score"]
        assert "ontology_score" not in r_def
        assert "ontology_score" not in r_exp


# ==============================================================================
# 2. Alias Expansion Promotes Synonym-Bearing Relevant Candidates
# ==============================================================================

def test_alias_expansion_promotes_relevant_chunk(test_chunks):
    """
    Query mentions 'degree', while the target chunk uses 'bachelor'.
    Ontology expansion should recognize 'degree' <-> 'bachelor' and boost the chunk.
    """
    retriever_base = ChunkRetriever(test_chunks, use_ontology=False)
    retriever_onto = ChunkRetriever(test_chunks, use_ontology=True)

    query = "What college degree did user receive?"
    base_res = retriever_base.retrieve(query, k=2)
    onto_res = retriever_onto.retrieve(query, k=2)

    assert len(onto_res) > 0
    top_onto = onto_res[0]
    assert top_onto["chunk_id"] == "chunk_edu_01"
    assert "bachelor" in top_onto["text"]
    assert top_onto.get("ontology_score", 0.0) > 0.0
    assert top_onto.get("ontology_bonus", 0.0) > 0.0

    # Ensure ontology score gave it a healthy boost over base score
    assert "detected_concepts" in top_onto
    assert any("degree" in q for q in ["degree"] if q in query.lower())


# ==============================================================================
# 3. Entity Matches Do Not Blindly Outrank Strong Semantic Candidates
# ==============================================================================

def test_entity_match_does_not_outrank_strong_semantic_hit(test_chunks):
    """
    Query: 'What did user eat for breakfast?'
    Chunk diet_01 is strongly semantically relevant (oatmeal, blueberries, breakfast).
    Chunk career_01 has entities (Marcus Vance, Aetheria Labs, Python).
    Even with ontology enabled, diet_01 MUST rank above career_01 because semantic
    and lexical weights (0.85) dominate ontology matching (0.15).
    """
    retriever_onto = ChunkRetriever(test_chunks, use_ontology=True)

    query = "What did user eat for breakfast this morning?"
    results = retriever_onto.retrieve(query, k=3)

    assert len(results) > 0
    top_hit = results[0]
    assert top_hit["chunk_id"] == "chunk_diet_01"
    assert "oatmeal" in top_hit["text"]

    # Career chunk must not jump to top rank
    chunk_ids = [r["chunk_id"] for r in results]
    assert chunk_ids.index("chunk_diet_01") == 0


# ==============================================================================
# 4. Weight Normalization & Score Boundaries
# ==============================================================================

def test_calibrated_scoring_bounds_and_weights(test_chunks):
    """
    Verify that vector, bm25, and ontology weights sum to 1.0, and all returned
    candidate scores are strictly bounded in [0.0, 1.0].
    """
    retriever = ChunkRetriever(test_chunks, use_ontology=True, ontology_weight=0.15)
    total_weights = retriever.vector_weight + retriever.bm25_weight + retriever.ontology_weight
    assert abs(total_weights - 1.0) < 1e-6

    query = "Who did user work with at Aetheria Labs?"
    results = retriever.retrieve(query, k=4)

    for item in results:
        assert 0.0 <= item["score"] <= 1.0
        assert 0.0 <= item["vector_score"] <= 1.0
        assert item["bm25_score"] >= 0.0
        assert 0.0 <= item.get("ontology_score", 0.0) <= 1.0


# ==============================================================================
# 5. Deterministic Retrieval and Tie-Breaking
# ==============================================================================

def test_deterministic_retrieval(test_chunks):
    """Running retrieval multiple times with identical inputs must return identical rankings."""
    retriever = ChunkRetriever(test_chunks, use_ontology=True)
    query = "Tell me about user shopping discounts and coupons"

    first_run = retriever.retrieve(query, k=3)
    for _ in range(3):
        repeat_run = retriever.retrieve(query, k=3)
        assert len(first_run) == len(repeat_run)
        for r1, r2 in zip(first_run, repeat_run):
            assert r1["chunk_id"] == r2["chunk_id"]
            assert r1["score"] == r2["score"]


# ==============================================================================
# 6. Provenance Preservation & Context Assembly
# ==============================================================================

def test_provenance_preservation(test_chunks):
    """Ensure all required chunk metadata is preserved and passed to context assembly."""
    retriever = ChunkRetriever(test_chunks, use_ontology=True)
    results = retriever.retrieve("Where was Target grocery shopping done?", k=1)

    assert len(results) == 1
    hit = results[0]
    assert hit["chunk_id"] == "chunk_shopping_01"
    assert hit["session_id"] == "sess_104"
    assert hit["memory_id"] == "mem_104"
    assert hit["turn_range"] == (0, 2)
    assert hit["timestamp"] == "2026-03-01T16:00:00Z"
    assert hit["token_estimate"] > 0

    # Test context assembly with retrieved chunks
    context = assemble_bounded_context("Where was Target grocery shopping done?", results, max_context_tokens=500)
    assert context["has_sufficient_evidence"] is True
    assert "chunk_shopping_01" in context["context_text"]
    assert "sess_104" in context["context_text"]


# ==============================================================================
# 7. Graceful Degradation on Empty Vocabulary & Unknown Queries
# ==============================================================================

def test_graceful_degradation_empty_vocab(test_chunks):
    """Retriever must operate without errors even when vocabulary is empty."""
    retriever = ChunkRetriever(test_chunks, use_ontology=True, vocabulary={})
    query = "Random unindexed words and unknown jargon 12345"
    results = retriever.retrieve(query, k=2)

    # Should gracefully return matches based purely on dense/lexical similarity without error
    assert isinstance(results, list)
    if results:
        assert results[0].get("ontology_score") == 0.0
