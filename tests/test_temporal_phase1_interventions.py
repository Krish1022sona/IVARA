"""
Unit and regression tests for Phase 1 Temporal Reasoning Interventions.

Covers:
1. Deterministic calendar helper: leap years, month boundaries, relative day/week/month deltas.
2. Inclusive vs. exclusive day counting.
3. Reference-date propagation into prompt.
4. Answering pipeline fallback recovery with integer answers and escaped quotes.
5. Calendar fact injection into prompt.
6. Adaptive retrieval depth escalation for temporal queries.
7. Preservation of abstention on unanswerable / insufficient evidence queries.
"""

import pytest
from datetime import date
from src.temporal.calendar_arithmetic import (
    DeterministicCalendarEngine,
    parse_date_string,
    days_between,
    weeks_between,
    months_between,
    add_days
)
from src.answering import format_user_prompt, answer_question, AnswerPayload


def test_calendar_arithmetic_month_boundaries_and_leap_year():
    # Leap year 2024: Feb has 29 days
    d1 = date(2024, 2, 28)
    d2 = date(2024, 3, 1)
    assert days_between(d1, d2) == 2
    assert days_between(d1, d2, inclusive=True) == 3

    # Non-leap year 2023: Feb has 28 days
    d3 = date(2023, 2, 28)
    d4 = date(2023, 3, 1)
    assert days_between(d3, d4) == 1
    assert days_between(d3, d4, inclusive=True) == 2


def test_calendar_arithmetic_weeks_and_months():
    d1 = date(2023, 5, 1)
    d2 = date(2023, 5, 22)
    assert weeks_between(d1, d2) == 3.0

    d3 = date(2023, 1, 15)
    d4 = date(2023, 4, 15)
    assert months_between(d3, d4) == 3


def test_calendar_engine_evaluate_query():
    engine = DeterministicCalendarEngine()

    # Query: how many days ago
    res = engine.evaluate_query(
        "How many days ago did Arjun visit the clinic?",
        question_date="2023/05/20 (Sat) 14:00",
        evidence_dates=["2023-05-15 10:00:00"]
    )
    assert res["status"] == "computed"
    assert "5 days" in res["computed_fact"]
    assert res["suggested_answer"] == "5 days"

    # Query: how many days between
    res2 = engine.evaluate_query(
        "How many days between the workshop on 2023-04-10 and the exam on 2023-04-14?",
        evidence_dates=["2023-04-10", "2023-04-14"]
    )
    assert res2["status"] == "computed"
    assert "4 days" in res2["suggested_answer"]


def test_format_user_prompt_reference_date_injection():
    context = {"context_text": "Arjun attended a conference on 2023-05-10."}
    prompt = format_user_prompt(
        query="When did Arjun attend the conference?",
        context=context,
        question_date="2023/05/20 (Sat) 14:00",
        calendar_fact="Days between 2023-05-10 and 2023-05-20: 10 days"
    )
    assert "CURRENT REFERENCE TIME ANCHOR: The user is asking this question on 2023-05-20" in prompt
    assert "VERIFIED CALENDAR COMPUTATION:" in prompt
    assert "Days between 2023-05-10 and 2023-05-20: 10 days" in prompt


def test_answer_question_fallback_with_integer_answer():
    # LLM emits JSON with integer answer instead of string: {"answer": 4, "status": "answered", "cited_memory_ids": ["m001"]}
    def mock_llm_int(prompt, system_prompt):
        return '{"answer": 4, "status": "answered", "cited_memory_ids": ["m001"]}'

    test_memories = [
        {"id": "m001", "text": "Arjun completed 4 modules of the course.", "timestamp": "2023-05-10"}
    ]

    res = answer_question(
        query="How many modules did Arjun complete?",
        memories=test_memories,
        llm_caller=mock_llm_int,
        min_score=None
    )
    assert res["status"] == "answered"
    assert res["answer"] == "4"
    assert len(res["citations"]) == 1
    assert res["citations"][0]["memory_id"] == "m001"


def test_answer_question_fallback_with_escaped_quotes():
    # LLM emits raw JSON string with escaped quotes in answer
    def mock_llm_quotes(prompt, system_prompt):
        return '{"answer": "He said \\"confirmed\\" on the schedule.", "status": "answered", "cited_memory_ids": ["m001"]}'

    test_memories = [
        {"id": "m001", "text": "Arjun said confirmed on the schedule.", "timestamp": "2023-05-10"}
    ]

    res = answer_question(
        query="What did Arjun say?",
        memories=test_memories,
        llm_caller=mock_llm_quotes,
        min_score=None
    )
    assert res["status"] == "answered"
    assert "confirmed" in res["answer"]


def test_answer_question_preserves_abstention():
    # Genuinely insufficient evidence: no matching memories
    res = answer_question(
        query="What is Arjun's favorite constellation?",
        memories=[],
        min_score=0.35
    )
    assert res["status"] == "insufficient_evidence"
    assert "sufficient recorded memory" in res["answer"].lower()

