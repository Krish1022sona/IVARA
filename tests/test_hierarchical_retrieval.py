"""
Unit tests for Hierarchical Episodic Retriever (Phase 2A).

Covers:
1. Session-level representation construction.
2. Stage 1 session retrieval (Hybrid Vector + BM25).
3. Session diversity controls (penalizing redundant same-date sessions).
4. Stage 2 within-session chunk retrieval with per-session allocation.
5. Chronological sorting across retrieved sessions.
6. Answering pipeline integration with retrieval_strategy='hierarchical'.
"""

import pytest
from datetime import datetime
from src.chunking.turn_chunker import TurnChunk
from src.chunking.hierarchical_retrieval import (
    HierarchicalEpisodicRetriever,
    SessionRepresentation
)
from src.answering import answer_question


def _create_mock_chunks():
    chunks = [
        # Session 1: 2023-01-10 (Visit to MoMA)
        TurnChunk(
            chunk_id="sess_s1_c001",
            session_id="s1",
            memory_id="sess_s1",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
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
        # Session 2: 2023-01-17 (Visit to the Met)
        TurnChunk(
            chunk_id="sess_s2_c001",
            session_id="s2",
            memory_id="sess_s2",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2023-01-17",
            parsed_datetime=datetime(2023, 1, 17),
            turns=[
                {"role": "user", "content": "Today I went to the Metropolitan Museum of Art for the Ancient Civilizations exhibit."},
                {"role": "assistant", "content": "Did you enjoy the exhibit?"},
                {"role": "user", "content": "Yes, the Egyptian artifacts were mesmerizing."}
            ],
            text="User: Today I went to the Metropolitan Museum of Art for the Ancient Civilizations exhibit.\nAssistant: Did you enjoy the exhibit?\nUser: Yes, the Egyptian artifacts were mesmerizing.",
            word_count=25,
            token_estimate=33
        ),
        # Session 3: 2023-02-05 (Cooking Italian dinner)
        TurnChunk(
            chunk_id="sess_s3_c001",
            session_id="s3",
            memory_id="sess_s3",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2023-02-05",
            parsed_datetime=datetime(2023, 2, 5),
            turns=[
                {"role": "user", "content": "I cooked homemade lasagna for my friends tonight."},
                {"role": "assistant", "content": "Sounds delicious!"},
                {"role": "user", "content": "Everyone loved the béchamel sauce."}
            ],
            text="User: I cooked homemade lasagna for my friends tonight.\nAssistant: Sounds delicious!\nUser: Everyone loved the béchamel sauce.",
            word_count=18,
            token_estimate=24
        )
    ]
    return chunks


def test_session_representation_construction():
    chunks = _create_mock_chunks()
    retriever = HierarchicalEpisodicRetriever(chunks)
    assert len(retriever.sessions) == 3
    s1 = next(s for s in retriever.sessions if s.session_id == "s1")
    assert s1.date_str == "2023-01-10"
    assert "Museum of Modern Art" in s1.text
    assert len(s1.chunks) == 1


def test_stage1_session_retrieval():
    chunks = _create_mock_chunks()
    retriever = HierarchicalEpisodicRetriever(chunks)
    # Query about museum visits
    res = retriever.retrieve_sessions("When did I visit the Museum of Modern Art and the Met?", top_m=2)
    assert len(res) == 2
    matched_sids = {s.session_id for s, score in res}
    assert "s1" in matched_sids
    assert "s2" in matched_sids


def test_stage2_chunk_retrieval_and_chronological_ordering():
    chunks = _create_mock_chunks()
    retriever = HierarchicalEpisodicRetriever(chunks)
    res_chunks = retriever.retrieve("Tell me about my museum visits", k=5, top_m_sessions=2)
    assert len(res_chunks) == 2
    retrieved_sids = {c["session_id"] for c in res_chunks}
    assert "s1" in retrieved_sids
    assert "s2" in retrieved_sids

    # Verify assemble_bounded_context formats them chronologically
    from src.chunking.retrieval import assemble_bounded_context
    ctx = assemble_bounded_context("Tell me about my museum visits", res_chunks)
    assert ctx["direct_evidence"][0]["session_id"] == "s1"
    assert ctx["direct_evidence"][1]["session_id"] == "s2"


def test_session_diversity_controls():
    # Create two sessions on the same date and one on a different date
    chunks = _create_mock_chunks()
    extra_chunk = TurnChunk(
        chunk_id="sess_s1b_c001",
        session_id="s1b",
        memory_id="sess_s1b",
        turn_start_idx=0,
        turn_end_idx=2,
        total_session_turns=3,
        timestamp="2023-01-10",  # Same date as s1
        parsed_datetime=datetime(2023, 1, 10),
        turns=[{"role": "user", "content": "Museum gift shop purchase"}],
        text="User: Museum gift shop purchase",
        word_count=5,
        token_estimate=7
    )
    chunks.append(extra_chunk)
    retriever = HierarchicalEpisodicRetriever(chunks)

    # Without diversity: s1 and s1b might dominate
    res_no_div = retriever.retrieve_sessions("Museum visit", top_m=2, use_diversity=False)
    # With diversity: s1 is selected, s1b is penalized because it has the same date
    res_div = retriever.retrieve_sessions("Museum visit", top_m=2, use_diversity=True, diversity_lambda=0.5)
    assert len(res_div) == 2


def test_answer_question_hierarchical_integration():
    chunks = _create_mock_chunks()

    def mock_llm(prompt, sys_prompt):
        return '{"answer": "You visited MoMA on 2023-01-10 and The Met on 2023-01-17.", "status": "answered", "cited_memory_ids": ["sess_s1", "sess_s2"]}'

    res = answer_question(
        query="When did I visit the museums?",
        use_chunking=True,
        chunks=chunks,
        llm_caller=mock_llm,
        retrieval_strategy="hierarchical",
        top_m_sessions=2
    )
    assert res["status"] == "answered"
    assert "MoMA" in res["answer"]
    assert len(res["citations"]) == 2


def test_adaptive_temporal_router():
    from src.chunking.hierarchical_retrieval import AdaptiveTemporalRouter
    
    # High-confidence multi-session queries -> hierarchical_diversity
    q1 = "Which three events happened in the order from first to last: event A, event B, and event C?"
    r1 = AdaptiveTemporalRouter.route(q1)
    assert r1["strategy"] == "hierarchical_diversity"
    assert r1["confidence"] >= 0.90
    
    q2 = "How many days passed between my visit to MoMA and the Ancient Civilizations exhibit at the Met?"
    r2 = AdaptiveTemporalRouter.route(q2)
    assert r2["strategy"] == "hierarchical_diversity"
    
    q3 = "Which event did I attend first, the workshop or the webinar?"
    r3 = AdaptiveTemporalRouter.route(q3)
    assert r3["strategy"] == "hierarchical_diversity"

    # Single-session relative queries -> flat
    q4 = "How many weeks ago did I start using the cashback app 'Ibotta'?"
    r4 = AdaptiveTemporalRouter.route(q4)
    assert r4["strategy"] == "flat"
    assert r4["confidence"] >= 0.90

    q5 = "How many months have passed since I last visited a museum with a friend?"
    r5 = AdaptiveTemporalRouter.route(q5)
    assert r5["strategy"] == "flat"

    q6 = "What was the life event of one of my relatives that I participated in a week ago?"
    r6 = AdaptiveTemporalRouter.route(q6)
    assert r6["strategy"] == "flat"

    # Standard factual / uncertain query -> flat
    q7 = "What is Arjun's favorite programming language?"
    r7 = AdaptiveTemporalRouter.route(q7)
    assert r7["strategy"] == "flat"


def test_answer_question_adaptive_integration():
    chunks = _create_mock_chunks()

    def mock_llm(prompt, sys_prompt):
        return '{"answer": "First you visited MoMA, then you visited The Met.", "status": "answered", "cited_memory_ids": ["sess_s1", "sess_s2"]}'

    res = answer_question(
        query="Which event happened first, my visit to MoMA or my visit to the Met?",
        use_chunking=True,
        chunks=chunks,
        llm_caller=mock_llm,
        retrieval_strategy="adaptive",
        top_m_sessions=2
    )
    assert res["status"] == "answered"
    assert "MoMA" in res["answer"]
    assert "adaptive_routing" in res["retrieved_context"]
    assert res["retrieved_context"]["adaptive_routing"]["strategy"] == "hierarchical_diversity"


