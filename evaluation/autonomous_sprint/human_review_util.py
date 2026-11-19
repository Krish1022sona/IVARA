"""
Human Review Analysis Utility.

Imports completed human verdicts into an analysis without modifying the original file.
If human review cannot be completed autonomously, labels the artifact as awaiting human annotation.
Never fabricates human labels.
"""

import csv
import json
from pathlib import Path
from typing import Any


def load_human_review_analysis(csv_path: str | Path) -> dict[str, Any]:
    """
    Read a human review CSV and compute alignment metrics if labels are present.
    Does NOT modify the CSV file.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Review CSV not found: {p}")

    rows = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total_rows = len(rows)
    annotated_rows = [r for r in rows if r.get("human_verdict", "").strip()]

    if not annotated_rows:
        return {
            "status": "awaiting_human_annotation",
            "message": "Human verdict columns are strictly blank. Awaiting developer/evaluator annotation.",
            "total_instances": total_rows,
            "annotated_instances": 0,
            "file_path": str(p),
            "unannotated_instances": total_rows,
            "metrics": None
        }

    # If completed annotations exist, compute agreement against judge
    judge_agreements = 0
    judge_tp = 0  # Both say CORRECT
    judge_fp = 0  # Judge says CORRECT, Human says INCORRECT
    judge_tn = 0  # Both say INCORRECT
    judge_fn = 0  # Judge says INCORRECT, Human says CORRECT
    failure_categories = {}

    for r in annotated_rows:
        h_verdict = r.get("human_verdict", "").strip().upper()
        j_verdict = r.get("judge_verdict", "").strip().upper()

        is_h_corr = (h_verdict in ("CORRECT", "TRUE", "YES", "1"))
        is_j_corr = (j_verdict in ("CORRECT", "TRUE", "YES", "1"))

        if is_h_corr == is_j_corr:
            judge_agreements += 1

        if is_j_corr and is_h_corr:
            judge_tp += 1
        elif is_j_corr and not is_h_corr:
            judge_fp += 1
        elif not is_j_corr and not is_h_corr:
            judge_tn += 1
        elif not is_j_corr and is_h_corr:
            judge_fn += 1

        cat = r.get("human_failure_category", "").strip()
        if cat:
            failure_categories[cat] = failure_categories.get(cat, 0) + 1

    ann_n = len(annotated_rows)
    agreement_rate = round(judge_agreements / ann_n * 100, 2)
    precision = round(judge_tp / (judge_tp + judge_fp) * 100, 2) if (judge_tp + judge_fp) else 0.0
    recall = round(judge_tp / (judge_tp + judge_fn) * 100, 2) if (judge_tp + judge_fn) else 0.0

    return {
        "status": "annotated",
        "total_instances": total_rows,
        "annotated_instances": ann_n,
        "file_path": str(p),
        "metrics": {
            "judge_agreement_rate": agreement_rate,
            "judge_precision": precision,
            "judge_recall": recall,
            "confusion": {
                "true_positives": judge_tp,
                "false_positives": judge_fp,
                "true_negatives": judge_tn,
                "false_negatives": judge_fn
            },
            "failure_categories": failure_categories
        }
    }
