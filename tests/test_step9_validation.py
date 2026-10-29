"""
Unit tests validating Step 9 evaluation integrity and sampling reproducibility.
"""

import csv
import json
from pathlib import Path
import pytest

STEP9_DIR = Path("evaluation/step9")
INTEGRITY_FILE = STEP9_DIR / "metric_integrity_audit.json"
REVIEW_SAMPLE_FILE = STEP9_DIR / "human_review_sample.csv"
FAILURE_ANALYSIS_FILE = STEP9_DIR / "failure_analysis.json"


class TestStep9Validation:
    """Verifies that metric integrity audits pass and sampling is strictly reproducible."""

    def test_metric_integrity_audit_passes(self):
        assert INTEGRITY_FILE.exists(), "Missing metric_integrity_audit.json"
        with open(INTEGRITY_FILE, "r", encoding="utf-8") as f:
            audit = json.load(f)
        assert audit["status"] == "PASS", f"Audit failed with discrepancies: {audit.get('discrepancies')}"
        assert len(audit["discrepancies"]) == 0

    def test_qid_consistency_across_configurations(self):
        with open(INTEGRITY_FILE, "r", encoding="utf-8") as f:
            audit = json.load(f)
        qid_check = audit["checks"]["qid_consistency"]
        assert qid_check["passed"] is True
        assert qid_check["total_instances_per_config"] == 60
        assert qid_check["no_duplicates"] is True
        assert qid_check["exact_order_match"] is True

    def test_abstention_denominators(self):
        with open(INTEGRITY_FILE, "r", encoding="utf-8") as f:
            audit = json.load(f)
        abs_check = audit["checks"]["abstention_label_alignment"]
        assert abs_check["passed"] is True
        assert abs_check["total_abstentions"] == 8
        assert abs_check["total_answerable"] == 52

    def test_human_review_sample_structure_and_blank_verdicts(self):
        assert REVIEW_SAMPLE_FILE.exists(), "Missing human_review_sample.csv"
        with open(REVIEW_SAMPLE_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 26, f"Expected exactly 26 review questions, got {len(rows)}"

        for idx, row in enumerate(rows, 1):
            assert row["question_id"], f"Row {idx} missing question_id"
            assert row["question"], f"Row {idx} missing question text"
            assert row["gold_answer"], f"Row {idx} missing gold_answer"
            assert row["gemma_score_A"] in ("CORRECT", "INCORRECT")
            assert row["gemma_score_D"] in ("CORRECT", "INCORRECT")

            # Human review fields MUST be blank
            assert row["human_verdict_A"] == "", f"Row {idx} human_verdict_A must be blank"
            assert row["human_verdict_B"] == "", f"Row {idx} human_verdict_B must be blank"
            assert row["human_verdict_C"] == "", f"Row {idx} human_verdict_C must be blank"
            assert row["human_verdict_D"] == "", f"Row {idx} human_verdict_D must be blank"
            assert row["human_failure_category"] == "", f"Row {idx} human_failure_category must be blank"
            assert row["human_notes"] == "", f"Row {idx} human_notes must be blank"

    def test_sampling_reproducibility(self):
        assert FAILURE_ANALYSIS_FILE.exists(), "Missing failure_analysis.json"
        with open(FAILURE_ANALYSIS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data["sampling_metadata"]
        assert meta["seed"] == 42
        assert meta["sample_size"] == 26
        assert len(meta["selected_question_ids"]) == 26
        # Verify deterministic selection order
        assert meta["selected_question_ids"][0] == "099778bb"
