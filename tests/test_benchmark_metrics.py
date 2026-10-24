"""
Unit tests for IVARA benchmark evaluation metric definitions.

Tests standard IR Precision@k, Recall@k, Set-Precision, Abstention Confusion Matrix,
and Rubric vs Lexical Heuristic fact evaluation on hand-constructed examples with
known mathematical results.
"""

import pytest
from evaluation.long_context.corrected_metrics import (
    calculate_retrieval_metrics,
    calculate_comprehensive_abstention,
    evaluate_rubric_fact,
    evaluate_lexical_heuristic
)


def test_single_memory_index_retrieval():
    """
    Test Case 1: A single-memory index (index contains only 1 memory 'm001').
    When k=3, FAISS returns ['m001'].
    Relevant memory is ['m001'].
    Standard Precision@3 MUST be 1/3 = 0.3333, NOT 1.0 (100%).
    Set-Precision is 1/1 = 1.0.
    Recall@3 is 1/1 = 1.0.
    """
    metrics = calculate_retrieval_metrics(
        retrieved_ids=["m001"],
        supporting_ids=["m001"],
        k=3,
        total_index_size=1
    )
    assert metrics["precision_at_k"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["set_precision"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["retrieved_count"] == 1


def test_index_fewer_than_k_memories():
    """
    Test Case 2: Index containing fewer than k memories (e.g. 2 memories in index).
    When k=3, system retrieves ['m001', 'm002'].
    Only 'm001' is relevant.
    Standard Precision@3 = 1/3 = 0.3333.
    Set-Precision = 1/2 = 0.5000.
    Recall@3 = 1/1 = 1.0.
    """
    metrics = calculate_retrieval_metrics(
        retrieved_ids=["m001", "m002"],
        supporting_ids=["m001"],
        k=3,
        total_index_size=2
    )
    assert metrics["precision_at_k"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["set_precision"] == 0.5
    assert metrics["recall_at_k"] == 1.0


def test_query_with_no_relevant_results():
    """
    Test Case 3: A query with no relevant results in the retrieved set.
    Retrieved: ['m002', 'm003', 'm004'].
    Supporting: ['m001'].
    Precision@3 = 0/3 = 0.0.
    Set-Precision = 0/3 = 0.0.
    Recall@3 = 0/1 = 0.0.
    """
    metrics = calculate_retrieval_metrics(
        retrieved_ids=["m002", "m003", "m004"],
        supporting_ids=["m001"],
        k=3,
        total_index_size=10
    )
    assert metrics["precision_at_k"] == 0.0
    assert metrics["set_precision"] == 0.0
    assert metrics["recall_at_k"] == 0.0


def test_one_of_three_retrieved_is_relevant():
    """
    Test Case 4: A query where one of three retrieved results is relevant.
    Retrieved: ['m001', 'm002', 'm003'].
    Supporting: ['m001'].
    Precision@3 = 1/3 = 0.3333.
    Set-Precision = 1/3 = 0.3333.
    Recall@3 = 1/1 = 1.0.
    """
    metrics = calculate_retrieval_metrics(
        retrieved_ids=["m001", "m002", "m003"],
        supporting_ids=["m001"],
        k=3,
        total_index_size=10
    )
    assert metrics["precision_at_k"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["set_precision"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["recall_at_k"] == 1.0


def test_multi_evidence_partial_retrieval():
    """
    Test Case 5: A multi-evidence question requiring 2 memories ('m001' and 'm004').
    Only one required passage is retrieved ('m001', 'm002', 'm003').
    Precision@3 = 1/3 = 0.3333.
    Recall@3 = 1/2 = 0.5000.
    """
    metrics = calculate_retrieval_metrics(
        retrieved_ids=["m001", "m002", "m003"],
        supporting_ids=["m001", "m004"],
        k=3,
        total_index_size=10
    )
    assert metrics["precision_at_k"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["recall_at_k"] == 0.5


def test_unanswerable_query_retrieval():
    """
    Test Case: Unanswerable query has no supporting IDs.
    In standard IR, Recall is undefined (None) on unanswerable queries.
    Precision@k is 0.0 when items are retrieved, because 0 relevant items were retrieved.
    """
    metrics = calculate_retrieval_metrics(
        retrieved_ids=["m001", "m002"],
        supporting_ids=[],
        k=3,
        total_index_size=10
    )
    assert metrics["precision_at_k"] == 0.0
    assert metrics["set_precision"] == 0.0
    assert metrics["recall_at_k"] is None
    assert metrics["is_unanswerable"] is True


def test_abstention_metrics_evaluation():
    """
    Test Cases 6 & 7:
    - An unanswerable question that is correctly abstained from (TP).
    - An answerable question that is incorrectly abstained from (FP).
    - An answerable question that is correctly answered (TN).
    - An unanswerable question that is incorrectly answered (FN).
    """
    mock_cases = [
        # Correctly abstained unanswerable (TP)
        {"id": "q41", "supporting_memory_ids": [], "actual_status": "insufficient_evidence"},
        # Failed to abstain on unanswerable / hallucination (FN)
        {"id": "q42", "supporting_memory_ids": [], "actual_status": "answered"},
        # Falsely abstained answerable (FP)
        {"id": "q01", "supporting_memory_ids": ["m001"], "actual_status": "insufficient_evidence"},
        # Correctly answered answerable (TN)
        {"id": "q02", "supporting_memory_ids": ["m007"], "actual_status": "answered"},
    ]
    
    abstention = calculate_comprehensive_abstention(mock_cases)
    
    assert abstention["true_positives_abstained"] == 1
    assert abstention["false_positives_abstained"] == 1
    assert abstention["false_negatives_hallucinated"] == 1
    assert abstention["true_negatives_answered"] == 1
    
    # Precision = TP / (TP + FP) = 1 / 2 = 0.5
    assert abstention["abstention_precision"] == 0.5
    # Recall = TP / (TP + FN) = 1 / 2 = 0.5
    assert abstention["abstention_recall"] == 0.5
    # F1 = 0.5
    assert abstention["abstention_f1"] == 0.5
    # False Abstention Rate = FP / len(answerable) = 1 / 2 = 0.5
    assert abstention["false_abstention_rate_on_answerables"] == 0.5
    # Hallucination Rate on Unanswerables = FN / len(unanswerable) = 1 / 2 = 0.5
    assert abstention["hallucination_rate_on_unanswerables"] == 0.5


def test_fact_evaluation_rubric_vs_keyword_flaw():
    """
    Demonstrate that a factually wrong answer containing expected keywords:
    1. Erroneously receives high credit under the legacy lexical heuristic.
    2. Is correctly rejected (score 0.0) under rubric evaluation.
    """
    gold = "Project Chronos was implemented in Rust to guarantee zero-cost abstractions."
    rubric = "Must name Rust as the programming language for the ingestion pipeline."
    
    # False answer claiming COBOL was used, but echoing question/gold keywords
    false_answer = "Project Chronos was implemented in COBOL to guarantee zero-cost abstractions."
    
    # 1. Legacy lexical heuristic: awards high score due to word overlap
    legacy_score = evaluate_lexical_heuristic(
        actual_answer=false_answer,
        actual_status="answered",
        gold_answer=gold,
        expected_status="answered",
        should_abstain=False
    )
    # The false answer matches 5 of 6 long words ('project', 'chronos', 'implemented', 'guarantee', 'zero-cost', 'abstractions')
    assert legacy_score > 0.70  # Undeserved high credit!

    # 2. Corrected Rubric evaluation: detects COBOL negative constraint & missing Rust
    rubric_res = evaluate_rubric_fact(
        qid="q01",
        actual_answer=false_answer,
        actual_status="answered",
        gold_answer=gold,
        rubric=rubric,
        should_abstain=False
    )
    assert rubric_res["score"] == 0.0
    assert rubric_res["passed"] is False


def test_concise_correct_answer_receives_credit_in_rubric():
    """
    Demonstrate that a concise, 100% correct answer ("Neo4j" for q15):
    1. Was heavily penalized by legacy lexical heuristic (0.07 score).
    2. Receives full credit (score 1.0) under rubric evaluation.
    """
    gold = "Project Nexus selected Neo4j as its graph database. Dr. Elena Rostova served as Chief Scientist."
    rubric = "Must link The Oracle -> Dr. Elena Rostova -> Project Nexus -> Neo4j."
    concise_answer = "Neo4j"

    legacy_score = evaluate_lexical_heuristic(
        actual_answer=concise_answer,
        actual_status="partial",
        gold_answer=gold,
        expected_status="answered",
        should_abstain=False
    )
    # Legacy heuristic heavily penalizes conciseness
    assert legacy_score < 0.15

    # Rubric evaluation gives full credit for correctly naming Neo4j
    rubric_res = evaluate_rubric_fact(
        qid="q15",
        actual_answer=concise_answer,
        actual_status="answered",
        gold_answer=gold,
        rubric=rubric,
        should_abstain=False
    )
    assert rubric_res["score"] == 1.0
    assert rubric_res["passed"] is True
