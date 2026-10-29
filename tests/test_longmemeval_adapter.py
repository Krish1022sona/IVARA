"""
Tests for LongMemEval Adapter and Evaluator.

Covers:
- Dataset parsing and schema validation
- Session and timestamp preservation
- Gold-answer isolation from memories/retrieval
- Question-ID uniqueness and malformed record validation
- Production store safety guards
- Official prompt template verification
- Deterministic scoring calculations
"""

import json
from pathlib import Path
import pytest

from evaluation.adapters.longmemeval_adapter import (
    LongMemEvalAdapter,
    LongMemEvalInstance,
    LongMemEvalSession,
    LongMemEvalTurn,
    ProductionStoreIsolationError
)
from evaluation.evaluators.longmemeval_evaluator import (
    LongMemEvalEvaluator,
    get_anscheck_prompt,
    evaluate_response_locally
)


@pytest.fixture
def sample_raw_data():
    return [
        {
            "question_id": "test_q01",
            "question_type": "temporal-reasoning",
            "question": "How many days did my car repair take?",
            "answer": "18 days",
            "question_date": "2023/04/10 (Mon) 23:07",
            "haystack_dates": ["2023/03/15 (Wed) 10:20", "2023/04/02 (Sun) 14:00"],
            "haystack_session_ids": ["sess_01", "sess_02"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I dropped off my car for repair today.", "has_answer": True},
                    {"role": "assistant", "content": "Got it, we will inspect the engine."}
                ],
                [
                    {"role": "user", "content": "Finally picked up the car after 18 days of waiting."},
                    {"role": "assistant", "content": "Glad to hear your car is back in service."}
                ]
            ],
            "answer_session_ids": ["sess_01", "sess_02"]
        },
        {
            "question_id": "test_q02_abs",
            "question_type": "single-session-user",
            "question": "What is my mother's maiden name?",
            "answer": "The user never mentions their mother's maiden name.",
            "question_date": "2023/05/01 (Mon) 12:00",
            "haystack_dates": ["2023/04/20 (Thu) 09:00"],
            "haystack_session_ids": ["sess_03"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I had lunch with my sister today."},
                    {"role": "assistant", "content": "Sounds nice! How is she doing?"}
                ]
            ],
            "answer_session_ids": []
        }
    ]


def test_adapter_parses_valid_instances(tmp_path, sample_raw_data):
    dataset_file = tmp_path / "test_dataset.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(sample_raw_data, f)

    adapter = LongMemEvalAdapter(dataset_file, output_dir=tmp_path / "runs")
    instances, failures = adapter.load_and_validate()

    assert len(failures) == 0
    assert len(instances) == 2

    q1 = instances[0]
    assert q1.question_id == "test_q01"
    assert q1.question_type == "temporal-reasoning"
    assert q1.gold_answer == "18 days"
    assert q1.is_abstention is False
    assert len(q1.sessions) == 2
    assert q1.sessions[0].session_id == "sess_01"
    assert len(q1.sessions[0].turns) == 2

    # Check transcript formatting
    transcript = q1.sessions[0].to_formatted_text()
    assert "User: I dropped off my car" in transcript
    assert "Assistant: Got it" in transcript

    q2 = instances[1]
    assert q2.question_id == "test_q02_abs"
    assert q2.is_abstention is True


def test_gold_answer_isolated_from_memories(tmp_path, sample_raw_data):
    dataset_file = tmp_path / "test_dataset.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(sample_raw_data, f)

    adapter = LongMemEvalAdapter(dataset_file, output_dir=tmp_path / "runs")
    instances, _ = adapter.load_and_validate()

    q1 = instances[0]
    memories = adapter.convert_instance_to_memories(q1)

    assert len(memories) == 2
    for memo in memories:
        # Verify no gold answer, answer_session_ids, or has_answer flags leaked into memories
        assert "gold_answer" not in memo
        assert "answer" not in memo
        assert "answer_session_ids" not in memo
        assert "has_answer" not in memo
        assert "18 days" not in memo["id"]


def test_adapter_rejects_production_store_path():
    with pytest.raises(ProductionStoreIsolationError):
        LongMemEvalAdapter(dataset_path="data/memories.json")

    with pytest.raises(ProductionStoreIsolationError):
        LongMemEvalAdapter(dataset_path="test.json", output_dir="data/memories.json")


def test_adapter_handles_duplicates_and_malformed_records(tmp_path):
    bad_data = [
        {"question_id": "dup1", "question_type": "temporal-reasoning", "question": "Q1", "answer": "A1",
         "haystack_dates": ["2023/01/01"], "haystack_session_ids": ["s1"], "haystack_sessions": [[{"role": "user", "content": "hi"}]]},
        {"question_id": "dup1", "question_type": "temporal-reasoning", "question": "Q1", "answer": "A1",
         "haystack_dates": ["2023/01/01"], "haystack_session_ids": ["s1"], "haystack_sessions": [[{"role": "user", "content": "hi"}]]},
        {"question_id": "bad_type", "question_type": "unknown-task", "question": "Q2", "answer": "A2",
         "haystack_dates": ["2023/01/01"], "haystack_session_ids": ["s1"], "haystack_sessions": [[{"role": "user", "content": "hi"}]]},
        {"question_id": "len_mismatch", "question_type": "multi-session", "question": "Q3", "answer": "A3",
         "haystack_dates": ["2023/01/01"], "haystack_session_ids": ["s1", "s2"], "haystack_sessions": [[{"role": "user", "content": "hi"}]]}
    ]
    dataset_file = tmp_path / "bad_dataset.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(bad_data, f)

    adapter = LongMemEvalAdapter(dataset_file, output_dir=tmp_path / "runs")
    instances, failures = adapter.load_and_validate()

    assert len(instances) == 1
    assert instances[0].question_id == "dup1"
    assert len(failures) == 3
    reasons = [f.error_reason for f in failures]
    assert any("Duplicate question_id" in r for r in reasons)
    assert any("Unsupported question_type" in r for r in reasons)
    assert any("Mismatched haystack array lengths" in r for r in reasons)


def test_evaluator_prompts_match_official_spec():
    # 1. Standard QA prompt
    p_std = get_anscheck_prompt("multi-session", "What is my car model?", "Civic", "You drive a Honda Civic.")
    assert "Is the model response correct? Answer yes or no only." in p_std
    assert "What is my car model?" in p_std

    # 2. Temporal reasoning prompt with off-by-one rule
    p_temp = get_anscheck_prompt("temporal-reasoning", "How many days?", "18 days", "19 days")
    assert "do not penalize off-by-one errors for the number of days" in p_temp

    # 3. Knowledge update prompt
    p_ku = get_anscheck_prompt("knowledge-update", "Where do I work?", "Google", "Used to be Meta, now at Google.")
    assert "as long as the updated answer is the required answer" in p_ku

    # 4. Abstention prompt
    p_abs = get_anscheck_prompt("single-session-user", "What is my dog's name?", "Never mentioned", "I don't know.", abstention=True)
    assert "Does the model correctly identify the question as unanswerable? Answer yes or no only." in p_abs


def test_evaluator_local_scoring_rules():
    # Exact match
    assert evaluate_response_locally("single-session-user", "Q", "Blue", "My favorite color is Blue.") is True
    assert evaluate_response_locally("single-session-user", "Q", "Blue", "My favorite color is Red.") is False

    # Temporal off-by-one tolerance (18 vs 19)
    assert evaluate_response_locally("temporal-reasoning", "Q", "18 days", "It took roughly 19 days.") is True
    assert evaluate_response_locally("temporal-reasoning", "Q", "18 days", "It took 25 days.") is False

    # Knowledge update
    assert evaluate_response_locally("knowledge-update", "Q", "Tokyo", "I moved from New York to Tokyo.") is True

    # Abstention
    assert evaluate_response_locally("single-session-user", "Q", "None", "I don't have sufficient recorded memory to answer.", is_abstention=True) is True
    assert evaluate_response_locally("single-session-user", "Q", "None", "Your dog's name is Buddy.", is_abstention=True) is False


def test_evaluator_aggregates_metrics():
    references = {
        "q1": {"question_type": "temporal-reasoning", "question": "Q1", "answer": "18", "is_abstention": False},
        "q2": {"question_type": "temporal-reasoning", "question": "Q2", "answer": "10", "is_abstention": False},
        "q3": {"question_type": "knowledge-update", "question": "Q3", "answer": "Berlin", "is_abstention": False},
        "q4_abs": {"question_type": "single-session-user", "question": "Q4", "answer": "None", "is_abstention": True}
    }

    hypotheses = [
        {"question_id": "q1", "hypothesis": "18"},
        {"question_id": "q2", "hypothesis": "50"},  # wrong
        {"question_id": "q3", "hypothesis": "Berlin"},
        {"question_id": "q4_abs", "hypothesis": "Insufficient information available"}
    ]

    evaluator = LongMemEvalEvaluator(use_local_heuristic=True)
    summary = evaluator.evaluate(hypotheses, references)

    # 3 out of 4 correct: official overall accuracy = 0.75 (denominator 4)
    assert summary["official_metrics"]["overall_accuracy"] == 0.75
    assert summary["official_metrics"]["denominators"]["overall_accuracy"] == 4
    assert summary["official_metrics"]["abstention_accuracy"] == 1.0
    assert summary["official_metrics"]["denominators"]["abstention_accuracy"] == 1

    # Answerable-only metrics (3 answerable questions: q1, q2, q3; 2 correct)
    assert summary["answerable_only_metrics"]["overall_accuracy"] == round(2 / 3, 4)
    assert summary["answerable_only_metrics"]["denominators"]["overall_accuracy"] == 3

    # Tasks official breakdown
    assert summary["official_metrics"]["tasks"]["temporal-reasoning"]["accuracy"] == 0.5
    assert summary["official_metrics"]["tasks"]["knowledge-update"]["accuracy"] == 1.0


def test_metrics_fixture_where_official_and_all_500_differ():
    """
    Intentionally tests a small fixture where official overall accuracy,
    answerable accuracy, and supplementary all-500 accuracy all differ.
    """
    references = {
        "ans_01": {"question_type": "single-session-user", "question": "Q1", "answer": "Apple", "is_abstention": False},
        "ans_02": {"question_type": "single-session-user", "question": "Q2", "answer": "Banana", "is_abstention": False},
        "abs_01_abs": {"question_type": "single-session-user", "question": "Q3", "answer": "None", "is_abstention": True}
    }

    # Hypothesis: ans_01 correct, ans_02 incorrect, abs_01 correct (2 correct out of 3)
    hypotheses = [
        {"question_id": "ans_01", "hypothesis": "I like Apple."},
        {"question_id": "ans_02", "hypothesis": "I prefer Orange."},
        {"question_id": "abs_01_abs", "hypothesis": "I don't have sufficient recorded memory."}
    ]

    evaluator = LongMemEvalEvaluator(use_local_heuristic=True)
    summary = evaluator.evaluate(hypotheses, references)

    # 1. Official overall accuracy = 2 / 3 = 0.6667
    assert summary["official_metrics"]["overall_accuracy"] == 0.6667
    assert summary["official_metrics"]["denominators"]["overall_accuracy"] == 3

    # 2. Answerable-only accuracy = 1 / 2 = 0.5000 (denominator 2)
    assert summary["answerable_only_metrics"]["overall_accuracy"] == 0.5
    assert summary["answerable_only_metrics"]["denominators"]["overall_accuracy"] == 2

    # 3. Abstention accuracy = 1 / 1 = 1.0000 (denominator 1)
    assert summary["official_metrics"]["abstention_accuracy"] == 1.0
    assert summary["official_metrics"]["denominators"]["abstention_accuracy"] == 1

    # 4. Supplementary all-500 accuracy = 2 / 500 = 0.0040 (fixed denominator 500)
    assert summary["supplementary_all_500_metrics"]["all_500_accuracy"] == 0.0040
    assert summary["supplementary_all_500_metrics"]["denominator"] == 500

    # Ensure official overall (0.6667) != answerable only (0.5000) != all-500 (0.0040)
    assert summary["official_metrics"]["overall_accuracy"] != summary["answerable_only_metrics"]["overall_accuracy"]
    assert summary["official_metrics"]["overall_accuracy"] != summary["supplementary_all_500_metrics"]["all_500_accuracy"]


def test_oracle_sessions_sorted_chronologically(tmp_path):
    """
    Verifies that unsorted Oracle sessions are chronologically re-ordered
    while preserving original indices and session turn contents.
    """
    raw_data = [
        {
            "question_id": "unsorted_oracle_q",
            "question_type": "temporal-reasoning",
            "question": "What happened first?",
            "answer": "First event",
            "question_date": "2023/04/10 (Mon) 23:07",
            # Dates out of order: 17:50, 14:47, 17:15
            "haystack_dates": [
                "2023/04/10 (Mon) 17:50",
                "2023/04/10 (Mon) 14:47",
                "2023/04/10 (Mon) 17:15"
            ],
            "haystack_session_ids": ["sess_late", "sess_early", "sess_mid"],
            "haystack_sessions": [
                [{"role": "user", "content": "Late session message"}],
                [{"role": "user", "content": "Early session message"}],
                [{"role": "user", "content": "Mid session message"}]
            ],
            "answer_session_ids": ["sess_early"]
        }
    ]
    dataset_file = tmp_path / "unsorted_dataset.json"
    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(raw_data, f)

    adapter = LongMemEvalAdapter(dataset_file, output_dir=tmp_path / "runs", sort_sessions_chronologically=True)
    instances, _ = adapter.load_and_validate()
    inst = instances[0]

    # Verify sessions are now ordered: sess_early (14:47) -> sess_mid (17:15) -> sess_late (17:50)
    assert inst.sessions[0].session_id == "sess_early"
    assert inst.sessions[1].session_id == "sess_mid"
    assert inst.sessions[2].session_id == "sess_late"

    # Verify original indices are preserved
    assert inst.sessions[0].original_index == 1
    assert inst.sessions[1].original_index == 2
    assert inst.sessions[2].original_index == 0

    # Verify memory conversion keeps chronological ordering
    memories = adapter.convert_instance_to_memories(inst)
    assert memories[0]["id"] == "sess_sess_early"
    assert memories[0]["chronological_order"] == 0
    assert memories[0]["original_order"] == 1
    assert "Early session message" in memories[0]["text"]

