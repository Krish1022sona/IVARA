"""
Tests for IVARA Step 3: Query-Conditioned, Bounded Graph Expansion.

Validates:
1. Ordinary factual queries do not trigger unnecessary expansion.
2. Relational or multi-hop queries trigger expansion.
3. Useful adjacent chunks are recovered when the seed alone is insufficient.
4. Common-entity distractors do not outrank strong direct evidence.
5. Expansion strictly respects the maximum additional chunks cap (max 3).
6. Duplicate candidates are removed deterministically.
7. Graph-derived candidates preserve complete provenance metadata.
8. Missing or empty graph stores degrade gracefully to hybrid retrieval.
9. Ontology-guided seed retrieval remains active in graph-assisted mode.
10. Context assembly respects the 1,500-token budget.
11. Answer generation and citation validation work seamlessly with bounded graph context.
"""

import tempfile
from pathlib import Path
import pytest

from src.chunking.turn_chunker import TurnChunk
from src.chunking.graph_retrieval import (
    TurnGraphBuilder,
    TurnGraphRetriever,
    should_expand_graph
)
from src.chunking.retrieval import assemble_bounded_context
from src.graph.store import SQLiteGraphStore
from src.answering import answer_question


@pytest.fixture
def synthetic_conversational_chunks():
    """
    Synthetic corpus with:
    - Session A: Multi-turn interaction where Turn 0-2 (buying bike at Summit) is followed by Turn 2-4 (warranty code).
    - Session B: Unrelated conversation that happens to mention 'Summit' (common-entity distractor).
    - Session C: Multi-person relational conversation (Alice collaborating with Bob on Project Titan).
    - Session D: Ordinary factual conversation (eating oatmeal for breakfast).
    - Session E: Another unrelated conversation mentioning 'Summit' in a completely different context.
    - Session F: Another distractor mentioning 'Summit'.
    - Session G: Another distractor mentioning 'Summit'.
    """
    return [
        # Session A: Target conversation
        TurnChunk(
            chunk_id="sess_A_c000",
            session_id="sess_A",
            memory_id="mem_A",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=5,
            timestamp="2024-01-10T10:00:00",
            parsed_datetime=None,
            turns=[],
            text="User: I purchased a mountain bike at Summit Sports today.\nAssistant: That is a great bike! Did you get any accessories?\nUser: Yes, I bought a helmet and a water bottle cage.",
            word_count=31,
            token_estimate=40
        ),
        TurnChunk(
            chunk_id="sess_A_c001",
            session_id="sess_A",
            memory_id="mem_A",
            turn_start_idx=2,
            turn_end_idx=4,
            total_session_turns=5,
            timestamp="2024-01-10T10:05:00",
            parsed_datetime=None,
            turns=[],
            text="Assistant: Did they provide a warranty registration code?\nUser: The cashier gave me registration code #BIKE-9921 for the five year warranty.\nAssistant: Keep that code safe!",
            word_count=27,
            token_estimate=35
        ),
        # Session B: Common-entity distractor 1 mentioning 'Summit'
        TurnChunk(
            chunk_id="sess_B_c000",
            session_id="sess_B",
            memory_id="mem_B",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2024-02-15T14:00:00",
            parsed_datetime=None,
            turns=[],
            text="User: I attended the Global Tech Summit last week in Seattle.\nAssistant: What was the main topic discussed?\nUser: The summit focused entirely on enterprise cloud infrastructure.",
            word_count=26,
            token_estimate=35
        ),
        # Session C: Relational multi-hop conversation
        TurnChunk(
            chunk_id="sess_C_c000",
            session_id="sess_C",
            memory_id="mem_C",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2024-03-01T09:00:00",
            parsed_datetime=None,
            turns=[],
            text="User: I worked with Alice and Bob on Project Titan.\nAssistant: Who was leading the initiative?\nUser: Alice was the project manager and Bob designed the backend database.",
            word_count=27,
            token_estimate=35
        ),
        # Session D: Ordinary factual conversation
        TurnChunk(
            chunk_id="sess_D_c000",
            session_id="sess_D",
            memory_id="mem_D",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2024-03-20T08:30:00",
            parsed_datetime=None,
            turns=[],
            text="User: I ate oatmeal with berries and honey for breakfast this morning.\nAssistant: Sounds nutritious and delicious!\nUser: It gives me good energy for morning workouts.",
            word_count=25,
            token_estimate=32
        ),
        # Session E: Common-entity distractor 2 mentioning 'Summit'
        TurnChunk(
            chunk_id="sess_E_c000",
            session_id="sess_E",
            memory_id="mem_E",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2024-04-05T16:00:00",
            parsed_datetime=None,
            turns=[],
            text="User: We reached the mountain summit during our hike in Colorado.\nAssistant: How long did the hike take?\nUser: It took four hours to climb to the summit.",
            word_count=26,
            token_estimate=35
        ),
        # Session F: Common-entity distractor 3 mentioning 'Summit'
        TurnChunk(
            chunk_id="sess_F_c000",
            session_id="sess_F",
            memory_id="mem_F",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2024-04-12T11:00:00",
            parsed_datetime=None,
            turns=[],
            text="User: Summit Street has a lot of heavy road construction right now.\nAssistant: Is traffic delayed?\nUser: Yes, traffic is backed up for several blocks.",
            word_count=24,
            token_estimate=32
        ),
        # Session G: Common-entity distractor 4 mentioning 'Summit'
        TurnChunk(
            chunk_id="sess_G_c000",
            session_id="sess_G",
            memory_id="mem_G",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2024-04-18T12:00:00",
            parsed_datetime=None,
            turns=[],
            text="User: Summit Coffee opened a brand new cafe downtown.\nAssistant: Have you tried their roast?\nUser: Yes, their dark roast blend is fantastic.",
            word_count=23,
            token_estimate=30
        )
    ]


@pytest.fixture
def temp_graph_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_bounded_graph.db"
        store = SQLiteGraphStore(db_path)
        try:
            yield store
        finally:
            store.close()


def test_ordinary_factual_queries_bypass_expansion(synthetic_conversational_chunks, temp_graph_store):
    """1. Ordinary factual queries do not trigger unnecessary expansion."""
    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(synthetic_conversational_chunks)

    retriever = TurnGraphRetriever(synthetic_conversational_chunks, temp_graph_store)
    query = "What did I eat for breakfast this morning?"

    results, diag = retriever.retrieve(query=query, k=3)

    assert diag["expansion_executed"] is False
    assert "Ordinary factual query" in diag["decision_reason"]
    assert diag["extra_chunks_introduced"] == 0
    assert len(results) > 0
    assert results[0]["chunk_id"] == "sess_D_c000"


def test_relational_queries_trigger_expansion(synthetic_conversational_chunks, temp_graph_store):
    """2. Relational or multi-hop queries trigger graph expansion."""
    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(synthetic_conversational_chunks)

    retriever = TurnGraphRetriever(synthetic_conversational_chunks, temp_graph_store)
    query = "Who did I work with on Project Titan?"

    results, diag = retriever.retrieve(query=query, k=3)

    assert diag["expansion_executed"] is True
    assert "relational" in diag["decision_reason"].lower() or "entities" in diag["decision_reason"].lower()
    assert diag["nodes_expanded"] > 0


def test_adjacent_chunk_recovered_when_seed_insufficient(synthetic_conversational_chunks, temp_graph_store):
    """
    3. Conversational adjacency:
    Query asks for the warranty registration code for the bike bought at Summit Sports.
    The primary seed matches 'Summit Sports' (sess_A_c000), but the registration code is in sess_A_c001.
    Bounded graph expansion over 'adjacent_to' must recover sess_A_c001!
    """
    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(synthetic_conversational_chunks)

    retriever = TurnGraphRetriever(synthetic_conversational_chunks, temp_graph_store)
    query = "What was the warranty registration code for the bike I purchased at Summit Sports?"

    results, diag = retriever.retrieve(query=query, k=3)

    cids = [r["chunk_id"] for r in results]
    assert "sess_A_c000" in cids
    assert "sess_A_c001" in cids

    # Verify sess_A_c001 is retrieved and has the registration code
    c001_item = next(r for r in results if r["chunk_id"] == "sess_A_c001")
    assert "BIKE-9921" in c001_item["text"]
    assert c001_item["turn_range"] == (2, 4)


def test_common_entity_distractors_do_not_displace_direct_evidence(synthetic_conversational_chunks, temp_graph_store):
    """
    4. Distractor resistance:
    The word 'Summit' appears in Sessions B, E, F, G.
    Direct evidence chunk (sess_A_c000) must retain rank 1 and not be displaced
    by irrelevant Summit chunks.
    """
    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(synthetic_conversational_chunks)

    retriever = TurnGraphRetriever(synthetic_conversational_chunks, temp_graph_store)
    query = "What accessories did I buy when purchasing the mountain bike at Summit Sports?"

    results, diag = retriever.retrieve(query=query, k=3)

    assert len(results) > 0
    # Top result must be the mountain bike purchase chunk
    assert results[0]["chunk_id"] == "sess_A_c000"
    assert results[0]["direct_score"] > 0.0

    # Entity distractors from unrelated sessions (e.g., Summit Coffee or Summit Street) must not displace direct evidence
    for r in results:
        if r["chunk_id"] != results[0]["chunk_id"] and r.get("graph_edge", "").startswith("mentions_entity"):
            assert r["score"] < results[0]["score"]


def test_expansion_strictly_capped_at_three(synthetic_conversational_chunks, temp_graph_store):
    """5. Expansion never introduces more than max_graph_chunks (3)."""
    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(synthetic_conversational_chunks)

    retriever = TurnGraphRetriever(
        synthetic_conversational_chunks,
        temp_graph_store,
        max_graph_chunks=3
    )
    query = "Tell me about my connections and people I worked with at Summit."

    results, diag = retriever.retrieve(query=query, k=10)

    assert diag["extra_chunks_introduced"] <= 3


def test_deterministic_deduplication(synthetic_conversational_chunks, temp_graph_store):
    """6. Results contain no duplicate chunk IDs and ranking is 100% deterministic."""
    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(synthetic_conversational_chunks)

    retriever = TurnGraphRetriever(synthetic_conversational_chunks, temp_graph_store)
    query = "What happened after I bought the bike at Summit Sports?"

    res1, _ = retriever.retrieve(query=query, k=5)
    res2, _ = retriever.retrieve(query=query, k=5)

    cids1 = [r["chunk_id"] for r in res1]
    cids2 = [r["chunk_id"] for r in res2]

    # No duplicates
    assert len(cids1) == len(set(cids1))
    # Exact deterministic equality
    assert cids1 == cids2
    for r1, r2 in zip(res1, res2):
        assert r1["score"] == r2["score"]


def test_provenance_preserved_in_graph_candidates(synthetic_conversational_chunks, temp_graph_store):
    """7. All retrieved candidates preserve chunk_id, session_id, turn_range, timestamp, text, and memory_id."""
    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(synthetic_conversational_chunks)

    retriever = TurnGraphRetriever(synthetic_conversational_chunks, temp_graph_store)
    query = "What discount or code did I get at Summit Sports?"

    results, _ = retriever.retrieve(query=query, k=4)

    for item in results:
        assert "chunk_id" in item
        assert "session_id" in item
        assert "turn_range" in item
        assert isinstance(item["turn_range"], tuple)
        assert len(item["turn_range"]) == 2
        assert "timestamp" in item
        assert "text" in item
        assert "score" in item
        assert len(item["text"]) > 0


def test_graceful_degradation_on_empty_or_missing_graph(synthetic_conversational_chunks):
    """8. Missing or empty graph stores degrade gracefully to hybrid seed retrieval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_store = SQLiteGraphStore(Path(tmpdir) / "empty.db")
        try:
            retriever = TurnGraphRetriever(synthetic_conversational_chunks, empty_store)
            results, diag = retriever.retrieve(query="Who did I work with on Project Titan?", k=3)

            assert len(results) > 0
            assert diag["extra_chunks_introduced"] == 0
            assert results[0]["chunk_id"] == "sess_C_c000"
        finally:
            empty_store.close()


def test_ontology_remains_active_in_graph_assisted_retrieval(synthetic_conversational_chunks, temp_graph_store):
    """9. When use_ontology=True, ontology concept extraction & scoring remain active in seed retriever."""
    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(synthetic_conversational_chunks)

    retriever = TurnGraphRetriever(
        synthetic_conversational_chunks,
        temp_graph_store,
        use_ontology=True
    )
    assert retriever.seed_retriever.use_ontology is True
    assert retriever.seed_retriever.ontology_weight > 0.0

    results, diag = retriever.retrieve(query="What was my role and who collaborated with me on Project Titan?", k=3)
    assert len(results) > 0


def test_context_assembly_token_budget_compliance(synthetic_conversational_chunks, temp_graph_store):
    """10. Context assembly from bounded graph results strictly respects max_context_tokens."""
    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(synthetic_conversational_chunks)

    retriever = TurnGraphRetriever(synthetic_conversational_chunks, temp_graph_store)
    results, _ = retriever.retrieve(
        query="Tell me everything about Summit Sports, Project Titan, and breakfast.",
        k=6
    )

    context = assemble_bounded_context(
        query="Tell me everything about Summit Sports, Project Titan, and breakfast.",
        retrieved_chunks=results,
        max_context_tokens=1500
    )

    assert context["context_token_count"] <= 1500
    assert len(context["direct_evidence"]) > 0
    assert "--- DIRECTLY SUPPORTED EVIDENCE ---" in context["context_text"]


def test_answer_question_with_bounded_graph_and_citation_validation(synthetic_conversational_chunks, temp_graph_store):
    """11. Answering pipeline with use_chunking=True and use_chunk_graph=True validates citations."""
    builder = TurnGraphBuilder(temp_graph_store)
    builder.build_from_chunks(synthetic_conversational_chunks)

    mock_llm = lambda prompt, sys: (
        '{"status": "answered", "answer": "You received warranty registration code #BIKE-9921 for your mountain bike.", "cited_memory_ids": ["mem_A"]}'
    )

    response = answer_question(
        query="What was the warranty code for the bike I purchased at Summit Sports?",
        use_chunking=True,
        use_chunk_graph=True,
        use_ontology=True,
        chunks=synthetic_conversational_chunks,
        graph_store=temp_graph_store,
        llm_caller=mock_llm
    )

    assert response["status"] == "answered"
    assert "BIKE-9921" in response["answer"]
    assert any(c["memory_id"] == "mem_A" for c in response["citations"])
    assert response["retrieved_context"] is not None
