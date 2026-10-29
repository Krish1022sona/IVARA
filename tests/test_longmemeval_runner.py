"""
Tests for LongMemEval Baseline Runner.

Covers:
- End-to-end synthetic run with mocked LLM answering
- Resume behavior and skipping already-completed questions
- Safe execution in isolated SQLite database
- Production store preservation
"""

import json
from pathlib import Path
import pytest

from evaluation.runners.run_longmemeval import LongMemEvalRunner


@pytest.fixture
def synthetic_benchmark_dataset(tmp_path):
    data = [
        {
            "question_id": "syn_q01",
            "question_type": "single-session-user",
            "question": "What is my favorite programming language?",
            "answer": "Python",
            "question_date": "2023/04/10 (Mon) 23:07",
            "haystack_dates": ["2023/03/15 (Wed) 10:20"],
            "haystack_session_ids": ["s01"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I love writing code in Python.", "has_answer": True},
                    {"role": "assistant", "content": "Python is very clean."}
                ]
            ],
            "answer_session_ids": ["s01"]
        },
        {
            "question_id": "syn_q02_abs",
            "question_type": "temporal-reasoning",
            "question": "What year did I graduate high school?",
            "answer": "Not mentioned in the history.",
            "question_date": "2023/05/01 (Mon) 12:00",
            "haystack_dates": ["2023/04/20 (Thu) 09:00"],
            "haystack_session_ids": ["s02"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I am working on my college resume."},
                    {"role": "assistant", "content": "Let me know if you need review."}
                ]
            ],
            "answer_session_ids": []
        }
    ]
    file_path = tmp_path / "synthetic_dataset.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return file_path


def mock_answering_llm(user_prompt: str, system_prompt: str) -> dict:
    """Mock answering model to avoid live GPU or network calls in tests."""
    prompt_lower = user_prompt.lower()
    if "programming language" in prompt_lower:
        return {
            "status": "answered",
            "answer": "Your favorite programming language is Python.",
            "cited_memory_ids": ["sess_s01"]
        }
    return {
        "status": "insufficient_evidence",
        "answer": "I don't have sufficient recorded memory to answer.",
        "cited_memory_ids": []
    }


def test_runner_executes_isolated_synthetic_run(tmp_path, synthetic_benchmark_dataset):
    out_dir = tmp_path / "run_output"

    runner = LongMemEvalRunner(
        dataset_path=synthetic_benchmark_dataset,
        output_dir=out_dir,
        config="B",
        model="mock-model",
        k=3,
        llm_caller=mock_answering_llm,
        use_local_judge=True
    )

    summary = runner.run(limit=None, resume=False)

    assert summary["metrics"]["official_metrics"]["overall_accuracy"] == 1.0
    assert summary["metrics"]["official_metrics"]["abstention_accuracy"] == 1.0
    assert summary["run_metadata"]["evaluated_instances"] == 2

    # Verify output files exist
    results_file = out_dir / "results_config_B.jsonl"
    summary_file = out_dir / "summary_config_B.json"
    manifest_file = out_dir / "run_manifest.json"
    assert results_file.exists()
    assert summary_file.exists()
    assert manifest_file.exists()

    with open(results_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert lines[0]["question_id"] == "syn_q01"
    assert lines[1]["question_id"] == "syn_q02_abs"


def test_runner_resume_skips_completed(tmp_path, synthetic_benchmark_dataset):
    out_dir = tmp_path / "resume_output"

    runner = LongMemEvalRunner(
        dataset_path=synthetic_benchmark_dataset,
        output_dir=out_dir,
        config="B",
        model="mock-model",
        k=3,
        llm_caller=mock_answering_llm,
        use_local_judge=True
    )

    # Run only 1 question first
    summary1 = runner.run(limit=1, resume=True)
    assert summary1["run_metadata"]["evaluated_instances"] == 1

    # Run again with all questions, resume=True
    summary2 = runner.run(limit=None, resume=True)
    assert summary2["run_metadata"]["evaluated_instances"] == 2

    results_file = out_dir / "results_config_B.jsonl"
    with open(results_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    # Verify no duplicates
    assert len(lines) == 2
    assert lines[0]["question_id"] == "syn_q01"
    assert lines[1]["question_id"] == "syn_q02_abs"


def test_runner_manifest_mismatch_prevents_silent_corruption(tmp_path, synthetic_benchmark_dataset):
    """
    Verifies that attempting to resume with a different model or config
    raises RunConfigurationMismatchError and aborts rather than corrupting results.
    """
    out_dir = tmp_path / "mismatch_output"

    runner_a = LongMemEvalRunner(
        dataset_path=synthetic_benchmark_dataset,
        output_dir=out_dir,
        config="B",
        model="model-v1",
        k=3,
        llm_caller=mock_answering_llm,
        use_local_judge=True
    )
    runner_a.run(limit=1, resume=False)

    # Attempt to resume with different model name
    from evaluation.runners.run_longmemeval import RunConfigurationMismatchError
    runner_b = LongMemEvalRunner(
        dataset_path=synthetic_benchmark_dataset,
        output_dir=out_dir,
        config="B",
        model="model-v2-conflicting",
        k=3,
        llm_caller=mock_answering_llm,
        use_local_judge=True
    )

    with pytest.raises(RunConfigurationMismatchError):
        runner_b.run(limit=2, resume=True)


def test_production_stores_untouched_during_benchmark(tmp_path, synthetic_benchmark_dataset):
    prod_memories_file = Path("data/memories.json")
    prod_entities_file = Path("data/entities.json")

    mem_before = prod_memories_file.read_bytes() if prod_memories_file.exists() else None
    ent_before = prod_entities_file.read_bytes() if prod_entities_file.exists() else None

    out_dir = tmp_path / "isolated_run"
    runner = LongMemEvalRunner(
        dataset_path=synthetic_benchmark_dataset,
        output_dir=out_dir,
        config="B",
        model="mock-model",
        k=3,
        llm_caller=mock_answering_llm,
        use_local_judge=True
    )
    runner.run(limit=2)

    mem_after = prod_memories_file.read_bytes() if prod_memories_file.exists() else None
    ent_after = prod_entities_file.read_bytes() if prod_entities_file.exists() else None

    assert mem_before == mem_after, "Production memories.json was modified!"
    assert ent_before == ent_after, "Production entities.json was modified!"


def test_four_evaluation_campaign_configurations_support(tmp_path, synthetic_benchmark_dataset):
    """
    Verifies that Config-A, Config-B, Config-C, and Config-D are recognized,
    execute with the intended components, and record observables properly.
    """
    configs = ["Config-A", "Config-B", "Config-C", "Config-D"]
    for cfg in configs:
        out_dir = tmp_path / f"test_{cfg}"
        runner = LongMemEvalRunner(
            dataset_path=synthetic_benchmark_dataset,
            output_dir=out_dir,
            config=cfg,
            model="mock-model",
            k=3,
            llm_caller=mock_answering_llm,
            use_local_judge=True
        )
        summary = runner.run(limit=1, resume=False)
        assert summary["run_metadata"]["config"] == cfg.upper()
        assert summary["run_metadata"]["evaluated_instances"] == 1

        results_file = out_dir / f"results_config_{cfg.upper()}.jsonl"
        assert results_file.exists()
        with open(results_file, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        assert "stage_latencies_ms" in rec
        assert "temporal_activated" in rec
        assert "graph_nodes_count" in rec

