"""
Controlled 6-Arm Diagnostic Evaluation Experiment (Diagnostic Pilot 30).

Evaluates 6 isolated memory architectures over a fixed 30-question stratified held-out sample:
- ARM_A: Dense + BM25 Baseline
- ARM_B: Baseline + Ontology Guidance
- ARM_C: Ontology + Raw 1-Hop Graph (min_edge_relevance=0.0)
- ARM_D: Ontology + Relevance-Filtered Graph (min_edge_relevance=0.50) [Prototype B]
- ARM_E: Ontology + Direct Seed Temporal (direct_seed_temporal=True) [Prototype C]
- ARM_F: Full Existing Architecture (Graph + Temporal)

Sequential Execution Architecture:
1. Phase 3A: Generate all predictions for ARM_A through ARM_F using gpt-oss:latest
   (Keeps gpt-oss continuously in VRAM without hot-swap overhead).
2. Phase 3B: Run diagnostic Gemma judge on all 6 arms using gemma4-26b
   (Loads Gemma once, evaluating all 180 predictions sequentially).
"""

import sys
import os
os.environ["OLLAMA_LLM_LIBRARY"] = "cuda_v12"
sys.path.insert(0, ".")

import json
import time
from pathlib import Path
from evaluation.autonomous_sprint.diagnostic_runner import DiagnosticRunner, ARM_DEFINITIONS

def main():
    sample_path = Path("evaluation/autonomous_sprint/diagnostic_sample_100.json")
    output_dir = Path("evaluation/autonomous_sprint/runs/diagnostic_pilot_30")
    limit = 30  # Bound to 30 questions x 6 arms = 180 evaluations to fit session deadline
    
    runner = DiagnosticRunner(
        sample_path=sample_path,
        output_dir=output_dir,
        model="gpt-oss:latest",
        judge_model="VladimirGav/gemma4-26b-16GB-VRAM:latest",
        k=5
    )
    
    arms = ["ARM_A", "ARM_B", "ARM_C", "ARM_D", "ARM_E", "ARM_F"]
    manifest_path = runner.write_manifest(arms)
    print(f"Manifest written to {manifest_path}")
    print(f"=== Starting 6-Arm Diagnostic Experiment (N={limit} questions, {len(arms)} arms = {limit * len(arms)} predictions) ===")
    
    total_start = time.time()
    
    # Phase 3A: Answer Generation across all 6 arms
    print("\n--- PHASE 3A: Answer Generation (gpt-oss:latest) ---")
    gen_results = {}
    for arm in arms:
        t_arm_start = time.time()
        res = runner.run_arm(arm, limit=limit, resume=True)
        gen_results[arm] = res
        print(f"  Completed {arm}: {res['newly_processed']} new / {res['total_instances']} total in {res['runtime_s']}s")
        
    gen_total_time = round(time.time() - total_start, 2)
    print(f"\nPhase 3A Complete: All {len(arms)} arms generated in {gen_total_time}s ({gen_total_time/60:.1f} min)")
    
    # Phase 3B: Diagnostic Gemma Judging
    print("\n--- PHASE 3B: Diagnostic Gemma Judging (VladimirGav/gemma4-26b-16GB-VRAM:latest) ---")
    judge_start = time.time()
    judge_summaries = {}
    for arm in arms:
        print(f"Judging {arm}...")
        summary = runner.evaluate_with_judge(arm, resume=True)
        judge_summaries[arm] = summary
        
    judge_total_time = round(time.time() - judge_start, 2)
    print(f"\nPhase 3B Complete: All {len(arms)} arms judged in {judge_total_time}s ({judge_total_time/60:.1f} min)")
    
    # Save overall experiment summary
    experiment_summary = {
        "experiment_title": "6-Arm Controlled Diagnostic Evaluation (Pilot 30)",
        "sample_size": limit,
        "total_evaluations": limit * len(arms),
        "total_runtime_s": round(time.time() - total_start, 2),
        "generation_runtime_s": gen_total_time,
        "judge_runtime_s": judge_total_time,
        "arm_summaries": judge_summaries
    }
    
    exp_summary_file = output_dir / "experiment_summary.json"
    with open(exp_summary_file, "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2)
        
    print(f"\n=== Experiment Finished in {experiment_summary['total_runtime_s']}s ===")
    print(f"Summary written to {exp_summary_file}")
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Arm':<8} | {'Architecture Description':<38} | {'Overall %':<10} | {'Ans %':<8} | {'Abs %':<8}")
    print("-" * 80)
    for arm in arms:
        s = judge_summaries[arm]
        print(f"{arm:<8} | {s['arm_name'][:38]:<38} | {s['overall_accuracy']:<10.2f} | {s['answerable_accuracy']:<8.2f} | {s['abstention_accuracy']:<8.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
