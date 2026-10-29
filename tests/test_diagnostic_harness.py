"""
Unit tests for the hardened 6-arm diagnostic evaluation harness.

Verifies:
- All 6 arm definitions (ARM_A through ARM_F) are present and valid.
- Resumption behavior on partial outputs without duplicate records.
- Incremental JSONL output and deduplication.
- Malformed judge JSON response handling and fallback recovery.
- End-to-end execution with mock LLM and judge callers.
"""

import json
from pathlib import Path
import pytest
from evaluation.autonomous_sprint.diagnostic_runner import DiagnosticRunner, ARM_DEFINITIONS


@pytest.fixture
def temp_sample_file(tmp_path):
    """Create a minimal synthetic sample file with 2 instances."""
    sample_data = [
        {
            "question_id": "diag_q001",
            "question_type": "knowledge-update",
            "question": "What is my current book page?",
            "answer": "Page 220",
            "question_date": "2023/04/01 (Sat) 10:00",
            "is_abstention": False,
            "haystack_dates": ["2023/04/01 (Sat) 10:00"],
            "haystack_session_ids": ["s001"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I am reading my book and I reached page 220."},
                    {"role": "assistant", "content": "Great, noted."}
                ]
            ]
        },
        {
            "question_id": "diag_q002_abs",
            "question_type": "temporal-reasoning",
            "question": "When did I fly to Mars?",
            "answer": "You never flew to Mars.",
            "question_date": "2023/04/01 (Sat) 10:00",
            "is_abstention": True,
            "haystack_dates": ["2023/04/01 (Sat) 10:00"],
            "haystack_session_ids": ["s002"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I like reading space books."},
                    {"role": "assistant", "content": "Space books are wonderful."}
                ]
            ]
        }
    ]
    path = tmp_path / "synthetic_sample.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2)
    return path


def test_arm_definitions_completeness():
    """Verify all 6 required arms are defined with correct parameter signatures."""
    expected_arms = ["ARM_A", "ARM_B", "ARM_C", "ARM_D", "ARM_E", "ARM_F"]
    for arm in expected_arms:
        assert arm in ARM_DEFINITIONS, f"Missing definition for {arm}"
        cfg = ARM_DEFINITIONS[arm]
        assert "use_ontology" in cfg
        assert "use_chunk_graph" in cfg
        assert "min_edge_relevance" in cfg
        assert "direct_seed_temporal" in cfg
        assert "enable_temporal" in cfg


def test_runner_mock_execution_and_incremental_flushing(temp_sample_file, tmp_path):
    """Verify end-to-end execution of an arm with a mock LLM caller and incremental JSONL flush."""
    out_dir = tmp_path / "run_output"

    def mock_llm(prompt: str, sys_prompt: str) -> dict:
        return {
            "status": "answered",
            "answer": "You reached page 220.",
            "cited_memory_ids": [],
            "limitations": None
        }

    runner = DiagnosticRunner(
        sample_path=temp_sample_file,
        output_dir=out_dir,
        llm_caller=mock_llm
    )

    manifest_path = runner.write_manifest(["ARM_A", "ARM_B"])
    assert manifest_path.exists()

    res = runner.run_arm("ARM_B", limit=None, resume=True)
    assert res["total_instances"] == 2
    assert res["newly_processed"] == 2

    # Check results file
    preds_file = out_dir / "predictions_ARM_B.jsonl"
    assert preds_file.exists()
    lines = [json.loads(line) for line in open(preds_file, encoding="utf-8") if line.strip()]
    assert len(lines) == 2
    assert lines[0]["question_id"] == "diag_q001"
    assert lines[1]["question_id"] == "diag_q002_abs"


def test_runner_resume_skips_already_processed_questions(temp_sample_file, tmp_path):
    """Verify that restarting with existing results skips already processed questions without duplicating."""
    out_dir = tmp_path / "run_output_resume"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate one question in predictions_ARM_A.jsonl
    preds_file = out_dir / "predictions_ARM_A.jsonl"
    with open(preds_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "question_id": "diag_q001",
            "hypothesis": "Already answered.",
            "status": "answered"
        }) + "\n")

    call_count = [0]
    def mock_llm(prompt: str, sys_prompt: str) -> dict:
        call_count[0] += 1
        return {"status": "answered", "answer": "Answer.", "cited_memory_ids": []}

    runner = DiagnosticRunner(
        sample_path=temp_sample_file,
        output_dir=out_dir,
        llm_caller=mock_llm
    )

    # Run arm with resume=True: should process only diag_q002_abs
    res = runner.run_arm("ARM_A", resume=True)
    assert res["newly_processed"] == 1
    assert res["total_instances"] == 2

    lines = [json.loads(line) for line in open(preds_file, encoding="utf-8") if line.strip()]
    assert len(lines) == 2
    qids = [l["question_id"] for l in lines]
    assert qids == ["diag_q001", "diag_q002_abs"]


def test_judge_malformed_json_fallback_and_recovery(temp_sample_file, tmp_path):
    """Verify that the diagnostic judge handles malformed text responses via regex fallback without crashing."""
    out_dir = tmp_path / "run_output_judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_file = out_dir / "predictions_ARM_E.jsonl"
    with open(preds_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "question_id": "diag_q001",
            "task_type": "knowledge-update",
            "question": "What page?",
            "gold_answer": "Page 220",
            "hypothesis": "Page 220",
            "is_abstention": False
        }) + "\n")
        f.write(json.dumps({
            "question_id": "diag_q002_abs",
            "task_type": "temporal-reasoning",
            "question": "When Mars?",
            "gold_answer": "Never",
            "hypothesis": "I do not have record.",
            "is_abstention": True
        }) + "\n")

    # Mock judge that returns markdown wrapped text instead of raw JSON
    call_idx = [0]
    def mock_judge(prompt: str) -> str:
        call_idx[0] += 1
        if call_idx[0] == 1:
            # Markdown fence response with true
            return "```json\n{\"is_correct\": true, \"reasoning\": \"Exact match.\"}\n```"
        else:
            # Malformed string with embedded key
            return "The prediction is accurate! \"is_correct\": true according to the evidence."

    runner = DiagnosticRunner(
        sample_path=temp_sample_file,
        output_dir=out_dir,
        judge_caller=mock_judge
    )

    summary = runner.evaluate_with_judge("ARM_E", resume=False)
    assert summary["overall_accuracy"] == 100.0
    assert summary["total_instances"] == 2

    judge_file = out_dir / "judge_results_ARM_E.jsonl"
    assert judge_file.exists()
    lines = [json.loads(line) for line in open(judge_file, encoding="utf-8") if line.strip()]
    assert len(lines) == 2
    assert lines[0]["is_correct"] is True
    assert lines[1]["is_correct"] is True
