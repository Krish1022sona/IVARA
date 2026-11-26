"""
Unit and integration tests for Query-Aware Multi-Session Retrieval (Phase 3).

Covers:
1. Router classification: SINGLE_SESSION, MULTI_SESSION, UNCERTAIN.
2. Subquery extraction for two-event comparison and three-event ordering queries.
3. Required session count estimation (S=2, S=3, S>=4).
4. Session representation construction and hybrid Vector+BM25 scoring.
5. Dynamic budget allocation across sessions (proportional allocation, avoiding 1-chunk trap).
6. Chronological ordering across multi-session snippets.
7. Inspectable diagnostics output.
8. End-to-end integration with answer_question(..., retrieval_strategy="query_aware").
"""

from datetime import datetime
import pytest

from src.chunking.turn_chunker import TurnChunk
from src.chunking.query_aware_retrieval import (
    QueryAwareRouter,
    QueryAwareMultiSessionRetriever,
    QueryIntent,
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
            token_estimate=19
        ),
        # Session 3: 2023-02-15 (Guggenheim Museum)
        TurnChunk(
            chunk_id="sess_s3_c001",
            session_id="s3",
            memory_id="sess_s3",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=2,
            timestamp="2023-02-15",
            parsed_datetime=datetime(2023, 2, 15),
            turns=[
                {"role": "user", "content": "I explored the spiral gallery at the Guggenheim Museum today."},
                {"role": "assistant", "content": "The architecture is iconic!"}
            ],
            text="User: I explored the spiral gallery at the Guggenheim Museum today.\nAssistant: The architecture is iconic!",
            word_count=18,
            token_estimate=24
        )
    ]


class TestQueryAwareRouter:
    """Tests for QueryAwareRouter."""

    def test_single_session_routing(self):
        single_queries = [
            "How many days ago did I buy a smoker?",
            "How many weeks ago did I attend the photography workshop?",
            "What charity event did I participate in a month ago?",
            "Which book did I finish a week ago?",
            "I mentioned participating in a sports event two weeks ago. What was the event?",
            "What gardening-related activity did I do two weeks ago?",
            "Which streaming service did I start using most recently?"
        ]
        for q in single_queries:
            res = QueryAwareRouter.route(q)
            assert res["outcome"] == QueryIntent.SINGLE_SESSION.value, f"Failed on: {q}"
            assert res["confidence"] >= 0.90
            assert res["estimated_required_sessions"] == 1
            assert len(res["subqueries"]) == 0

    def test_multi_session_two_event_subquery_extraction(self):
        two_event_queries = [
            (
                "How many days passed between my visit to the Museum of Modern Art (MoMA) and the 'Ancient Civilizations' exhibit at the Metropolitan Museum of Art?",
                "the Museum of Modern Art (MoMA)",
                "the 'Ancient Civilizations' exhibit at the Metropolitan Museum of Art"
            ),
            (
                "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?",
                "Effective Time Management' workshop",
                "Data Analysis using Python' webinar"
            ),
            (
                "Which event happened first, my cousin's wedding or Michael's engagement party?",
                "my cousin's wedding",
                "Michael's engagement party"
            ),
            (
                "How many weeks have I been accepted into the exchange program when I started attending the pre-departure orientation sessions?",
                "the exchange program",
                "attending the pre-departure orientation sessions"
            )
        ]
        for q, expected_sub1, expected_sub2 in two_event_queries:
            res = QueryAwareRouter.route(q)
            assert res["outcome"] == QueryIntent.MULTI_SESSION.value, f"Failed on: {q}"
            assert res["estimated_required_sessions"] == 2
            assert len(res["subqueries"]) == 2, f"Subqueries missing for: {q}"
            assert expected_sub1.lower() in res["subqueries"][0].lower()
            assert expected_sub2.lower() in res["subqueries"][1].lower()

    def test_multi_session_ordering_and_high_cardinality(self):
        ordering_queries = [
            ("What is the order of the six museums I visited from earliest to latest?", 6),
            ("What is the order of the three trips I took in the past three months, from earliest to latest?", 3),
            ("Which three events happened in the order from first to last: A, B, and C?", 3),
            ("What is the order of airlines I flew with from earliest to latest before today?", 3)
        ]
        for q, expected_s in ordering_queries:
            res = QueryAwareRouter.route(q)
            assert res["outcome"] == QueryIntent.MULTI_SESSION.value, f"Failed on: {q}"
            assert res["estimated_required_sessions"] >= expected_s

    def test_uncertain_fallback(self):
        vague_queries = [
            "Tell me about my day.",
            "What is my dog's name?",
            "",
            "   "
        ]
        for q in vague_queries:
            res = QueryAwareRouter.route(q)
            assert res["outcome"] == QueryIntent.UNCERTAIN.value


class TestQueryAwareMultiSessionRetriever:
    """Tests for QueryAwareMultiSessionRetriever execution and budget allocation."""

    def test_session_representation_and_scoring(self):
        chunks = _create_mock_chunks()
        retriever = QueryAwareMultiSessionRetriever(chunks)
        assert len(retriever.sessions) == 3

        scores = retriever._score_sessions("Museum of Modern Art MoMA", top_m=2)
        assert len(scores) >= 1
        top_sess, top_score = scores[0]
        assert top_sess.session_id == "s1"
        assert top_score > 0.3

    def test_dynamic_budget_allocation_two_events(self):
        chunks = _create_mock_chunks()
        retriever = QueryAwareMultiSessionRetriever(chunks)

        query = "How many days passed between my visit to MoMA and the Ancient Civilizations exhibit at the Met?"
        routing = QueryAwareRouter.route(query)
        assert routing["outcome"] == QueryIntent.MULTI_SESSION.value
        assert routing["estimated_required_sessions"] == 2

        retrieved_chunks, diag = retriever.retrieve(query, routing_decision=routing, k=5)
        # Verify both s1 and s2 are present
        retrieved_sids = {c["session_id"] for c in retrieved_chunks}
        assert "s1" in retrieved_sids, "Session s1 (MoMA) was starved!"
        assert "s2" in retrieved_sids, "Session s2 (Met) was starved!"

        # Verify budget map allocates at least 2 chunks to s1 and 2 to s2
        budget = diag["budget_allocation"]
        assert budget.get("s1", 0) >= 2
        assert budget.get("s2", 0) >= 2

        # Verify chronological sorting
        timestamps = [c["timestamp"] for c in retrieved_chunks]
        assert timestamps == sorted(timestamps)

    def test_inspectable_diagnostics_structure(self):
        chunks = _create_mock_chunks()
        retriever = QueryAwareMultiSessionRetriever(chunks)

        query = "Which did I visit first, MoMA or the Guggenheim Museum?"
        routing = QueryAwareRouter.route(query)
        retrieved_chunks, diag = retriever.retrieve(query, routing_decision=routing, k=5)

        assert "routing_decision" in diag
        assert "selected_sessions" in diag
        assert "budget_allocation" in diag
        assert "total_retrieved_chunks" in diag
        assert len(diag["selected_sessions"]) >= 2


class TestEndToEndAnsweringIntegration:
    """Integration test with answer_question using retrieval_strategy='query_aware'."""

    def test_query_aware_answering_flow(self):
        chunks = _create_mock_chunks()
        query = "How many days passed between my visit to MoMA and the Metropolitan Museum of Art?"

        def mock_llm(prompt, system_prompt):
            import json
            return json.dumps({
                "status": "answered",
                "answer": "7 days passed between your visit to MoMA and the Metropolitan Museum of Art.",
                "cited_memory_ids": [chunks[0].chunk_id],
                "limitations": None,
            })

        result = answer_question(
            query=query,
            chunks=chunks,
            use_chunking=True,
            retrieval_strategy="query_aware",
            llm_caller=mock_llm,
        )
        assert "status" in result
        assert "answer" in result
        assert "retrieved_context" in result
        routing_diag = result["retrieved_context"].get("query_aware_routing")
        assert routing_diag is not None
        assert "routing_decision" in routing_diag or "outcome" in routing_diag

