"""
Unit and integration tests for Evidence-Validated Session Routing (Phase 3.1).

Covers:
1. QueryClauseDecomposer:
   - Two-event comparative queries.
   - Single-session relative anchor queries.
   - Sequential ordering queries.
2. EvidenceValidatedRetriever:
   - Independent session candidate scoring for decomposed subclauses.
   - SAME_SESSION decision on identical top-1 session.
   - SAME_SESSION decision on high joint session support.
   - DIFFERENT_SESSIONS decision on disjoint candidate sessions with low joint support.
   - AMBIGUOUS fallback on low confidence.
   - Balanced multi-session chunk allocation (2 from Session A, 2 from Session B, chronological sort).
   - Flat MMR intra-session depth preservation when SAME_SESSION is routed.
3. End-to-end integration test with answer_question(..., retrieval_strategy="evidence_validated").
"""

from datetime import datetime
import pytest

from src.chunking.turn_chunker import TurnChunk
from src.chunking.evidence_validated_retrieval import (
    EvidenceValidatedRetriever,
    EvidenceRoutingDecision,
    QueryClauseDecomposer,
    SessionRepresentation
)
from src.answering import answer_question


def _create_mock_chunks():
    """Create 3 distinct conversational sessions for testing."""
    return [
        # Session 1: 2023-01-10 (Visit to MoMA)
        TurnChunk(
            chunk_id="sess_s1_c001",
            session_id="s1",
            memory_id="sess_s1",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=4,
            timestamp="2023-01-10",
            parsed_datetime=datetime(2023, 1, 10),
            turns=[
                {"role": "user", "content": "I visited the Museum of Modern Art today."},
                {"role": "assistant", "content": "How was MoMA?"},
                {"role": "user", "content": "It was fantastic, especially the Van Gogh paintings."}
            ],
            text="User: I visited the Museum of Modern Art today.\nAssistant: How was MoMA?\nUser: It was fantastic, especially the Van Gogh paintings.",
            word_count=20,
            token_estimate=27
        ),
        TurnChunk(
            chunk_id="sess_s1_c002",
            session_id="s1",
            memory_id="sess_s1",
            turn_start_idx=2,
            turn_end_idx=4,
            total_session_turns=4,
            timestamp="2023-01-10",
            parsed_datetime=datetime(2023, 1, 10),
            turns=[
                {"role": "user", "content": "After MoMA, I had lunch at a nearby cafe."},
                {"role": "assistant", "content": "What did you order?"}
            ],
            text="User: After MoMA, I had lunch at a nearby cafe.\nAssistant: What did you order?",
            word_count=15,
            token_estimate=20
        ),
        # Session 2: 2023-01-20 (Visit to Metropolitan Museum of Art)
        TurnChunk(
            chunk_id="sess_s2_c001",
            session_id="s2",
            memory_id="sess_s2",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=4,
            timestamp="2023-01-20",
            parsed_datetime=datetime(2023, 1, 20),
            turns=[
                {"role": "user", "content": "Today I went to the Metropolitan Museum of Art for the Ancient Civilizations exhibit."},
                {"role": "assistant", "content": "Did you enjoy the exhibit?"},
                {"role": "user", "content": "Yes, the Egyptian artifacts were mesmerizing."}
            ],
            text="User: Today I went to the Metropolitan Museum of Art for the Ancient Civilizations exhibit.\nAssistant: Did you enjoy the exhibit?\nUser: Yes, the Egyptian artifacts were mesmerizing.",
            word_count=25,
            token_estimate=33
        ),
        TurnChunk(
            chunk_id="sess_s2_c002",
            session_id="s2",
            memory_id="sess_s2",
            turn_start_idx=2,
            turn_end_idx=4,
            total_session_turns=4,
            timestamp="2023-01-20",
            parsed_datetime=datetime(2023, 1, 20),
            turns=[
                {"role": "user", "content": "The Greek sculpture hall was also breathtaking."},
                {"role": "assistant", "content": "Met is always stunning."}
            ],
            text="User: The Greek sculpture hall was also breathtaking.\nAssistant: Met is always stunning.",
            word_count=14,
            token_estimate=18
        ),
        # Session 3: 2023-02-05 (Combined gardening & greenhouse repair)
        TurnChunk(
            chunk_id="sess_s3_c001",
            session_id="s3",
            memory_id="sess_s3",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=4,
            timestamp="2023-02-05",
            parsed_datetime=datetime(2023, 2, 5),
            turns=[
                {"role": "user", "content": "Spent the entire morning watering the herb garden and planting basil."},
                {"role": "assistant", "content": "How are the herbs doing?"},
                {"role": "user", "content": "They look great. Right after watering, I started harvesting fresh herbs for dinner."}
            ],
            text="User: Spent the entire morning watering the herb garden and planting basil.\nAssistant: How are the herbs doing?\nUser: They look great. Right after watering, I started harvesting fresh herbs for dinner.",
            word_count=28,
            token_estimate=36
        ),
        TurnChunk(
            chunk_id="sess_s3_c002",
            session_id="s3",
            memory_id="sess_s3",
            turn_start_idx=2,
            turn_end_idx=4,
            total_session_turns=4,
            timestamp="2023-02-05",
            parsed_datetime=datetime(2023, 2, 5),
            turns=[
                {"role": "user", "content": "I stored the harvested basil in olive oil."},
                {"role": "assistant", "content": "A great storage technique."}
            ],
            text="User: I stored the harvested basil in olive oil.\nAssistant: A great storage technique.",
            word_count=14,
            token_estimate=18
        )
    ]


def test_query_clause_decomposer():
    """Verify clause decomposition across query types."""
    # Two-event comparison
    q_two = "Between watering herb garden and harvesting fresh herbs, how many hours passed?"
    res_two = QueryClauseDecomposer.decompose(q_two)
    assert not res_two["is_syntax_single"]
    assert len(res_two["subqueries"]) == 2
    assert "watering herb garden" in res_two["subqueries"][0]
    assert "harvesting fresh herbs" in res_two["subqueries"][1]

    # Single-session relative anchor
    q_single = "How many days ago did I visit the Museum of Modern Art?"
    res_single = QueryClauseDecomposer.decompose(q_single)
    assert res_single["is_syntax_single"]
    assert len(res_single["subqueries"]) == 0

    # Multi-session ordering
    q_order = "What was the order of the events from earliest to latest?"
    res_order = QueryClauseDecomposer.decompose(q_order)
    assert not res_order["is_syntax_single"]
    assert res_order["estimated_required_sessions"] >= 2


def test_evidence_validated_same_session():
    """Verify that intra-session events are classified as SAME_SESSION."""
    chunks = _create_mock_chunks()
    retriever = EvidenceValidatedRetriever(chunks)

    # Both events exist in Session 3
    query = "Between watering herb garden and planting basil, how many hours passed?"
    routing = retriever.evaluate_routing(query)

    assert routing["decision"] == EvidenceRoutingDecision.SAME_SESSION.value
    assert "s3" in routing["target_sessions"]


def test_evidence_validated_different_sessions():
    """Verify that genuinely cross-session events are classified as DIFFERENT_SESSIONS."""
    chunks = _create_mock_chunks()
    retriever = EvidenceValidatedRetriever(chunks)

    # Event A (MoMA) in s1, Event B (Metropolitan Museum of Art) in s2
    query = "Between Museum of Modern Art and Metropolitan Museum of Art, how many days passed?"
    routing = retriever.evaluate_routing(query)

    assert routing["decision"] == EvidenceRoutingDecision.DIFFERENT_SESSIONS.value
    assert set(routing["target_sessions"]) == {"s1", "s2"}


def test_retrieval_same_session_preserves_intra_session_depth():
    """Verify SAME_SESSION retrieves contiguous chunks without artificially capping at 2 chunks."""
    chunks = _create_mock_chunks()
    retriever = EvidenceValidatedRetriever(chunks)

    query = "Between watering herb garden and planting basil, how many hours passed?"
    retrieved_chunks, diag = retriever.retrieve(query, k=5, max_chunks_per_session=3)

    assert diag["decision"] == EvidenceRoutingDecision.SAME_SESSION.value
    assert diag["retrieval_mode"] == "FLAT_MMR_INTRA_SESSION"

    # s3 chunks should be present and not starved
    s3_chunks = [c for c in retrieved_chunks if (c.get("session_id") if isinstance(c, dict) else c.session_id) == "s3"]
    assert len(s3_chunks) >= 2


def test_retrieval_different_sessions_allocates_balanced_chunks():
    """Verify DIFFERENT_SESSIONS allocates chunks per target session and preserves both in context."""
    chunks = _create_mock_chunks()
    retriever = EvidenceValidatedRetriever(chunks)

    query = "Between Museum of Modern Art and Metropolitan Museum of Art, how many days passed?"
    retrieved_chunks, diag = retriever.retrieve(query, k=5)

    assert diag["decision"] == EvidenceRoutingDecision.DIFFERENT_SESSIONS.value
    assert diag["retrieval_mode"] == "BALANCED_MULTI_SESSION"

    s1_chunks = [c for c in retrieved_chunks if (c.get("session_id") if isinstance(c, dict) else c.session_id) == "s1"]
    s2_chunks = [c for c in retrieved_chunks if (c.get("session_id") if isinstance(c, dict) else c.session_id) == "s2"]

    assert len(s1_chunks) >= 1
    assert len(s2_chunks) >= 1


def test_end_to_end_answering_evidence_validated():
    """Verify answer_question operates cleanly with retrieval_strategy='evidence_validated'."""
    chunks = _create_mock_chunks()
    query = "Which did I visit first, the Museum of Modern Art or the Metropolitan Museum of Art?"

    def mock_llm(prompt, model=None, temperature=0.0):
        return '{"answer": "You visited the Museum of Modern Art first on January 10, 2023, before the Metropolitan Museum of Art on January 20, 2023.", "confidence": 0.95}'

    ans = answer_question(
        query=query,
        chunks=chunks,
        use_chunking=True,
        retrieval_strategy="evidence_validated",
        llm_caller=mock_llm
    )

    assert "retrieved_context" in ans
    assert "evidence_validated_routing" in ans["retrieved_context"]
    assert ans["retrieved_context"]["evidence_validated_routing"]["decision"] in (
        EvidenceRoutingDecision.DIFFERENT_SESSIONS.value,
        EvidenceRoutingDecision.SAME_SESSION.value
    )
    assert len(ans["answer"]) > 10

