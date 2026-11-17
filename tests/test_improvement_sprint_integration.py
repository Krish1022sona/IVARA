"""
Targeted tests for Phase 1 Sprint production integration:
1. Context-budget enforcement and prompt bounds
2. Query-scoped contradiction filtering
3. Session-Aware MMR diversity and fallback behavior
4. Empty model response handling
5. Malformed JSON recovery and markdown code fence extraction
"""

import pytest
from unittest.mock import MagicMock
from src.chunking.retrieval import select_session_mmr_chunks
from src.chunking.turn_chunker import TurnChunk
from src.answering import (
    answer_question,
    format_user_prompt,
    filter_query_relevant_contradictions,
    AnswerPayload
)


def test_select_session_mmr_chunks_diversity():
    """Verify that Session-Aware MMR penalizes chunks from already-selected sessions."""
    # 5 candidates: 3 from session_A, 2 from session_B
    candidates = [
        {"chunk_id": "c_A_1", "session_id": "sess_A", "score": 0.95},
        {"chunk_id": "c_A_2", "session_id": "sess_A", "score": 0.90},
        {"chunk_id": "c_A_3", "session_id": "sess_A", "score": 0.85},
        {"chunk_id": "c_B_1", "session_id": "sess_B", "score": 0.80},
        {"chunk_id": "c_B_2", "session_id": "sess_B", "score": 0.75},
    ]

    # Unconstrained top-3 would take c_A_1, c_A_2, c_A_3
    # With MMR (lambda=0.25):
    # Step 1: c_A_1 (0.95) selected -> sess_A penalized
    # Step 2: c_A_2 (0.90 - 0.25 = 0.65) vs c_B_1 (0.80) -> c_B_1 selected!
    selected = select_session_mmr_chunks(candidates, k=3, lambda_penalty=0.25)
    selected_cids = [c["chunk_id"] for c in selected]
    selected_sids = [c["session_id"] for c in selected]

    assert len(selected) == 3
    assert selected_cids[0] == "c_A_1"
    assert selected_cids[1] == "c_B_1"  # Session B was boosted by MMR diversity!
    assert "sess_A" in selected_sids
    assert "sess_B" in selected_sids


def test_select_session_mmr_fallback_behavior():
    """Verify that lambda_penalty=0.0 preserves pure score ranking as an exact fallback."""
    candidates = [
        {"chunk_id": "c_A_1", "session_id": "sess_A", "score": 0.95},
        {"chunk_id": "c_A_2", "session_id": "sess_A", "score": 0.90},
        {"chunk_id": "c_A_3", "session_id": "sess_A", "score": 0.85},
        {"chunk_id": "c_B_1", "session_id": "sess_B", "score": 0.80},
    ]

    fallback = select_session_mmr_chunks(candidates, k=3, lambda_penalty=0.0)
    assert [c["chunk_id"] for c in fallback] == ["c_A_1", "c_A_2", "c_A_3"]


def test_filter_query_relevant_contradictions():
    """Verify that contradiction filtering excludes irrelevant topics and keeps matching ones."""
    conflicts = [
        {
            "predicate": "beverage_preference",
            "subject": "User",
            "values": ["hoppy beer", "Negroni"],
            "explanation": "Contradictory drinks"
        },
        {
            "predicate": "commute_time",
            "subject": "User",
            "values": ["45 minutes", "20 minutes"],
            "explanation": "Different commute lengths"
        }
    ]

    # Query about commute
    q1 = "How long is my daily commute to work?"
    filtered1 = filter_query_relevant_contradictions(q1, conflicts)
    assert len(filtered1) == 1
    assert filtered1[0]["predicate"] == "commute_time"

    # Query about beverages
    q2 = "What kind of beer or cocktail do I like?"
    filtered2 = filter_query_relevant_contradictions(q2, conflicts)
    assert len(filtered2) == 1
    assert filtered2[0]["predicate"] == "beverage_preference"

    # Query about an unrelated topic (e.g. college degree)
    q3 = "What degree did I graduate with from university?"
    filtered3 = filter_query_relevant_contradictions(q3, conflicts)
    assert len(filtered3) == 0  # Neither contradiction pollutes the degree prompt!


def test_format_user_prompt_truncation_guard():
    """Verify that format_user_prompt bounds context length to preserve token budget."""
    long_text = "word " * 5000  # 25,000 characters
    context = {"context_text": long_text}
    prompt = format_user_prompt("Where do I live?", context, max_prompt_tokens=1000)

    # Max chars is 1000 * 3.5 = 3500 chars
    assert len(prompt) < 4500
    assert "[... Context truncated to preserve model token limit ...]" in prompt


def test_malformed_json_recovery():
    """Verify that JSON wrapped in markdown fences or having extra text is safely parsed."""
    raw_markdown = """```json
{
  "answer": "You graduated with a Bachelor of Science in Computer Science.",
  "status": "answered",
  "cited_memory_ids": ["m001"]
}
```"""

    def mock_caller(prompt, system_prompt):
        return raw_markdown

    memories = [
        {"id": "m001", "text": "I graduated with a Bachelor of Science in Computer Science in 2021.", "timestamp": "2021-06-01"}
    ]

    res = answer_question(
        query="What degree did I graduate with?",
        memories=memories,
        llm_caller=mock_caller,
        enable_temporal=False
    )

    assert res["status"] == "answered"
    assert "Computer Science" in res["answer"]
    assert len(res["citations"]) >= 1


def test_raw_json_string_clean_extraction():
    """Verify that raw JSON string outputs are extracted cleanly rather than dumped as code."""
    raw_unparsed = '{"answer": "Your daily commute takes 45 minutes each way.", "status": "answered"}'

    def mock_caller(prompt, system_prompt):
        return raw_unparsed

    memories = [
        {"id": "m001", "text": "My commute takes about 45 minutes each way.", "timestamp": "2023-05-26"}
    ]

    res = answer_question(
        query="How long is my commute?",
        memories=memories,
        llm_caller=mock_caller,
        enable_temporal=False
    )

    # Answer should be clean text, NOT starting with '{' or '\"answer\":'
    assert not res["answer"].strip().startswith('{')
    assert "45 minutes" in res["answer"]


def test_empty_model_response_graceful_handling():
    """Verify that an empty response string does not crash and produces an informative response."""
    def empty_caller(prompt, system_prompt):
        return ""

    memories = [
        {"id": "m001", "text": "I like hiking.", "timestamp": "2023-01-01"}
    ]

    res = answer_question(
        query="What is my hobby?",
        memories=memories,
        llm_caller=empty_caller,
        enable_temporal=False
    )

    assert res["status"] == "insufficient_evidence"
    assert "Could not reliably synthesize an answer" in res["answer"]

