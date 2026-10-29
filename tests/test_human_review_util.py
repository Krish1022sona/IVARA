"""
Unit tests for human_review_util.
"""

import csv
from pathlib import Path
from evaluation.autonomous_sprint.human_review_util import load_human_review_analysis


def test_human_review_util_detects_blank_annotations(tmp_path):
    """Utility must detect blank verdict columns and report awaiting_human_annotation."""
    csv_file = tmp_path / "blank_review.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "question_id", "judge_verdict", "human_verdict", "human_failure_category", "human_notes"])
        writer.writeheader()
        writer.writerow({"sample_id": 1, "question_id": "q1", "judge_verdict": "CORRECT", "human_verdict": "", "human_failure_category": "", "human_notes": ""})

    res = load_human_review_analysis(csv_file)
    assert res["status"] == "awaiting_human_annotation"
    assert res["annotated_instances"] == 0
    assert res["total_instances"] == 1


def test_human_review_util_computes_metrics_on_annotations(tmp_path):
    """Utility must compute agreement and confusion metrics when annotations are present."""
    csv_file = tmp_path / "filled_review.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "question_id", "judge_verdict", "human_verdict", "human_failure_category", "human_notes"])
        writer.writeheader()
        writer.writerow({"sample_id": 1, "question_id": "q1", "judge_verdict": "CORRECT", "human_verdict": "CORRECT", "human_failure_category": "", "human_notes": ""})
        writer.writerow({"sample_id": 2, "question_id": "q2", "judge_verdict": "CORRECT", "human_verdict": "INCORRECT", "human_failure_category": "hallucination", "human_notes": "judge was tricked"})

    res = load_human_review_analysis(csv_file)
    assert res["status"] == "annotated"
    assert res["annotated_instances"] == 2
    assert res["metrics"]["judge_agreement_rate"] == 50.0
    assert res["metrics"]["confusion"]["true_positives"] == 1
    assert res["metrics"]["confusion"]["false_positives"] == 1
