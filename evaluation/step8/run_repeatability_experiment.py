"""
Step 8: Deterministic Repeatability Experiment.
Uses LongMemEvalRunner to execute repeated evaluation passes on a fixed sample of questions
under Config-D with identical parameters (temperature=0, seed=42).
Verifies prediction identity, status agreement, and latency stability.
"""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.runners.run_longmemeval import LongMemEvalRunner

STEP8_DIR = Path("evaluation/step8")
REPEAT_DIR = STEP8_DIR / "repeatability"
REPEAT_DIR.mkdir(parents=True, exist_ok=True)

# 6 deterministic test questions across diverse tasks
SAMPLE_QIDS = [
    "e47becba",      # single-session-user
    "0a995998",      # multi-session
    "8a2466db",      # single-session-preference
    "dcfa8644",      # temporal-reasoning
    "6a1eabeb",      # knowledge-update
    "2698e78f_abs"   # knowledge-update abstention
]

def main():
    print("=== Starting Deterministic Repeatability Experiment ===")
    runs = ["run_1", "run_2"]
    run_records = {}

    for r_name in runs:
        out_dir = REPEAT_DIR / r_name
        # Clean directory if exists to test clean execution
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Running Iteration: {r_name} ---")
        runner = LongMemEvalRunner(
            dataset_path="evaluation/data/longmemeval/longmemeval_s_cleaned.json",
            output_dir=str(out_dir),
            config="CONFIG-D",
            use_local_judge=True
        )
        t0 = time.time()
        runner.run(target_question_ids=SAMPLE_QIDS)
        dur = time.time() - t0
        print(f"Iteration {r_name} completed in {dur:.2f}s")

        # Load predictions
        records_path = out_dir / "results_config_CONFIG-D.jsonl"
        recs = {}
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    h = hashlib.sha256(item["hypothesis"].strip().encode("utf-8")).hexdigest()
                    recs[item["question_id"]] = {
                        "hypothesis": item["hypothesis"],
                        "hypothesis_hash": h,
                        "status": item["status"],
                        "score": item["local_heuristic_score"],
                        "latency_ms": item["latency_ms"]
                    }
        run_records[r_name] = recs

    # Compare Run 1 vs Run 2
    print("\n=== Repeatability Comparison (Run 1 vs Run 2) ===")
    all_identical = True
    question_comparisons = {}

    r1 = run_records["run_1"]
    r2 = run_records["run_2"]

    for qid in SAMPLE_QIDS:
        h1 = r1[qid]["hypothesis_hash"]
        h2 = r2[qid]["hypothesis_hash"]
        is_exact = (h1 == h2)
        if not is_exact:
            all_identical = False
        print(f"[{qid:14s}] Exact Match: {is_exact} | Hash: {h1[:8]} vs {h2[:8]} | Status: {r1[qid]['status']} vs {r2[qid]['status']}")
        question_comparisons[qid] = {
            "exact_text_match": is_exact,
            "status_match": (r1[qid]["status"] == r2[qid]["status"]),
            "score_match": (r1[qid]["score"] == r2[qid]["score"]),
            "hash_run_1": h1,
            "hash_run_2": h2,
            "latency_run_1_ms": r1[qid]["latency_ms"],
            "latency_run_2_ms": r2[qid]["latency_ms"]
        }

    summary = {
        "experiment": "IVARA Step 8 Deterministic Repeatability Evaluation",
        "configuration": "CONFIG-D",
        "sample_size": len(SAMPLE_QIDS),
        "all_predictions_identical": all_identical,
        "sample_question_ids": SAMPLE_QIDS,
        "comparisons": question_comparisons,
        "runs": run_records
    }

    out_json = REPEAT_DIR / "repeatability_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {out_json}")
    print(f"Result: {'100% BIT-FOR-BIT DETERMINISTIC' if all_identical else 'STOCHASTIC DISCREPANCY OBSERVED'}")

if __name__ == "__main__":
    main()
