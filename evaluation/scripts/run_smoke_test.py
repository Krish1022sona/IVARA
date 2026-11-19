"""
Phase 2 Smoke Test for IVARA Evaluation Campaign v1.
Runs 6 deterministic questions across Config-A, Config-B, Config-C, Config-D.
Validates latency tracking, observable recording, resume capability, and clean execution.
"""

import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.runners.run_longmemeval import LongMemEvalRunner

SMOKE_QIDS = [
    "e47becba",      # single-session-user
    "0a995998",      # multi-session
    "8a2466db",      # single-session-preference
    "gpt4_59149c77", # temporal-reasoning
    "6a1eabeb",      # knowledge-update
    "0862e8bf_abs"   # abstention
]

RUN_BASE_DIR = PROJECT_ROOT / "evaluation" / "data" / "longmemeval" / "runs" / "campaign_v1_smoke"
DATASET_PATH = PROJECT_ROOT / "evaluation" / "data" / "longmemeval" / "longmemeval_s_cleaned.json"

CONFIGS = [
    ("Config-A", "config_a"),
    ("Config-B", "config_b"),
    ("Config-C", "config_c"),
    ("Config-D", "config_d"),
]

def run_smoke():
    print("=" * 70)
    print("PHASE 2 — SMOKE TEST (4 CONFIGURATIONS x 6 QUESTIONS)")
    print(f"Target QIDs: {SMOKE_QIDS}")
    print(f"Output directory: {RUN_BASE_DIR}")
    print("=" * 70)

    RUN_BASE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_smoke = RUN_BASE_DIR / "smoke_sample_manifest.json"
    with open(manifest_smoke, "w", encoding="utf-8") as f:
        json.dump({"smoke_qids": SMOKE_QIDS, "count": len(SMOKE_QIDS)}, f, indent=2)

    summaries = {}

    for config_name, sub_dir_name in CONFIGS:
        cfg_dir = RUN_BASE_DIR / sub_dir_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n---> Running {config_name} in {cfg_dir}...")
        runner = LongMemEvalRunner(
            dataset_path=DATASET_PATH,
            output_dir=cfg_dir,
            config=config_name,
            model="gpt-oss",
            k=5,
            use_local_judge=True
        )
        summary = runner.run(target_question_ids=SMOKE_QIDS, resume=True)
        summaries[config_name] = summary

        # Verify output records
        results_file = cfg_dir / f"results_config_{config_name.upper()}.jsonl"
        assert results_file.exists(), f"Results file missing for {config_name}"
        with open(results_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == len(SMOKE_QIDS), f"Expected {len(SMOKE_QIDS)} lines, found {len(lines)}"

        for rec in lines:
            assert rec["hypothesis"] is not None and len(rec["hypothesis"]) > 0, "Empty hypothesis"
            assert "stage_latencies_ms" in rec, "Missing stage latencies"
            assert "temporal_activated" in rec, "Missing temporal_activated"
            assert "graph_nodes_count" in rec, "Missing graph_nodes_count"
            assert "local_heuristic_score" in rec, "Missing local_heuristic_score"

        print(f"[{config_name}] Successfully validated {len(lines)} records.")

    # Test resume behavior on Config-A: rerun and assert no duplicates created
    print("\n---> Validating Resume Behavior on Config-A...")
    cfg_a_dir = RUN_BASE_DIR / "config_a"
    runner_resume = LongMemEvalRunner(
        dataset_path=DATASET_PATH,
        output_dir=cfg_a_dir,
        config="Config-A",
        model="gpt-oss",
        k=5,
        use_local_judge=True
    )
    runner_resume.run(target_question_ids=SMOKE_QIDS, resume=True)
    res_file = cfg_a_dir / "results_config_CONFIG-A.jsonl"
    with open(res_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == len(SMOKE_QIDS), f"Resume test duplicated records: expected {len(SMOKE_QIDS)}, found {len(lines)}"
    print("[Resume Test] Verified: 0 duplicated records. Resume is safe.")

    print("\n" + "=" * 70)
    print("PHASE 2 SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke()
