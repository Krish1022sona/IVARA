"""
Complete 6-Arm Controlled Diagnostic Evaluation Experiment (Full 100 Sample).

Resumes safely from the 30-question pilot:
- Seeded from evaluation/autonomous_sprint/runs/diagnostic_pilot_30/
- Evaluates the remaining 70 questions from evaluation/autonomous_sprint/diagnostic_sample_100.json
- Preserves identical model tags, prompt templates, context limits (1500 tokens), and judge rubric.
- Executes Phase A (generation with gpt-oss:latest across all 6 arms)
- Executes Phase B (judging with VladimirGav/gemma4-26b-16GB-VRAM:latest across all 6 arms)
"""

import sys
import os
import shutil
os.environ["OLLAMA_LLM_LIBRARY"] = "cuda_v12"
sys.path.insert(0, ".")

import json
import time
from pathlib import Path
from evaluation.autonomous_sprint.diagnostic_runner import DiagnosticRunner, ARM_DEFINITIONS

def main():
    sample_path = Path("evaluation/autonomous_sprint/diagnostic_sample_100.json")
    pilot_dir = Path("evaluation/autonomous_sprint/runs/diagnostic_pilot_30")
    output_dir = Path("evaluation/autonomous_sprint/runs/diagnostic_full_100")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    arms = ["ARM_A", "ARM_B", "ARM_C", "ARM_D", "ARM_E", "ARM_F"]
    
    # Step 1: Seed target directory from pilot_dir if files do not exist
    print("--- Step 1: Checking and Seeding Target Directory ---")
    for arm in arms:
        pred_src = pilot_dir / f"predictions_{arm}.jsonl"
        pred_dst = output_dir / f"predictions_{arm}.jsonl"
        judge_src = pilot_dir / f"judge_results_{arm}.jsonl"
        judge_dst = output_dir / f"judge_results_{arm}.jsonl"
        
        if not pred_dst.exists() and pred_src.exists():
            shutil.copyfile(pred_src, pred_dst)
            print(f"  Seeded {pred_dst.name} from pilot (30 predictions)")
        if not judge_dst.exists() and judge_src.exists():
            shutil.copyfile(judge_src, judge_dst)
            print(f"  Seeded {judge_dst.name} from pilot (30 judgments)")

    runner = DiagnosticRunner(
        sample_path=sample_path,
        output_dir=output_dir,
        model="gpt-oss:latest",
        judge_model="VladimirGav/gemma4-26b-16GB-VRAM:latest",
        k=5
    )
    
    manifest_path = runner.write_manifest(arms)
    print(f"Manifest written to {manifest_path}")
    print(f"=== Starting 6-Arm Diagnostic Full Run (Target N=100 questions x {len(arms)} arms = 600 evaluations) ===")
    
    total_start = time.time()
    
    # Phase A: Answer Generation across all 6 arms
    print("\n--- PHASE A: Answer Generation (gpt-oss:latest) ---")
    gen_results = {}
    for arm in arms:
        print(f"\nProcessing Arm {arm}: {ARM_DEFINITIONS[arm]['name']}")
        res = runner.run_arm(arm, limit=100, resume=True)
        gen_results[arm] = res
        print(f"  Finished {arm}: {res['newly_processed']} new / {res['total_instances']} total in {res['runtime_s']}s")
        
    gen_total_time = round(time.time() - total_start, 2)
    print(f"\nPhase A Complete: All {len(arms)} arms generated in {gen_total_time}s ({gen_total_time/60:.1f} min)")
    
    # Phase B: Diagnostic Gemma Judging across all 6 arms
    print("\n--- PHASE B: Diagnostic Gemma Judging (gemma4-26b) ---")
    judge_start = time.time()
    judge_results = {}
    for arm in arms:
        print(f"\nJudging Arm {arm}: {ARM_DEFINITIONS[arm]['name']}")
        res = runner.evaluate_with_judge(arm, resume=True)
        judge_results[arm] = res
        print(f"  Finished judging {arm}: Overall {res['overall_accuracy']}%, Answerable {res['answerable_accuracy']}%, Abstention {res['abstention_accuracy']}%")
        
    judge_total_time = round(time.time() - judge_start, 2)
    print(f"\nPhase B Complete: All {len(arms)} arms judged in {judge_total_time}s ({judge_total_time/60:.1f} min)")
    
    grand_total_time = round(time.time() - total_start, 2)
    print(f"\n=== Full 6-Arm Diagnostic Experiment Complete in {grand_total_time}s ({grand_total_time/60:.1f} min) ===")

    # Write combined experiment summary
    summary_path = output_dir / "experiment_summary.json"
    full_summary = {
        "run_id": "diagnostic_full_100",
        "sample_size": 100,
        "runtime_seconds": {
            "generation": gen_total_time,
            "judging": judge_total_time,
            "total": grand_total_time
        },
        "arms": judge_results
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    print(f"Saved complete summary to {summary_path}")

if __name__ == "__main__":
    main()
