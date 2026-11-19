"""
Finalization Sprint: 4-Arm Controlled Experiment on Fresh Held-Out Dataset (N=40).

Evaluates 4 key architectural configurations:
- ARM_1_BASELINE: Baseline (Dense + BM25)
- ARM_2_ONTOLOGY: Ontology-Guided Hybrid
- ARM_3_DIRECT_TEMPORAL: Ontology + Direct-Seed Temporal
- ARM_4_UNIFIED_HYBRID: Unified Research Candidate (IVARA-Hybrid: Ontology + Direct Seed Temporal + Relevance-Filtered Graph Fallback)

Dataset: evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json
(100% fresh held-out: 0 overlap with Step 7 pilot, 0 overlap with diagnostic 100 sample).
Preflight Gate enforced before launch.
"""

import sys
import os
os.environ["OLLAMA_LLM_LIBRARY"] = "cuda_v12"
sys.path.insert(0, ".")

import json
import time
from pathlib import Path
from evaluation.autonomous_sprint.preflight import PreflightGate
from evaluation.autonomous_sprint.diagnostic_runner import DiagnosticRunner, ARM_DEFINITIONS

def main():
    print("=== IVARA FINALIZATION SPRINT: CONTROLLED 4-ARM EXPERIMENT ===")
    
    sample_path = Path("evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json")
    expected_sample_sha256 = "5a42338eb5758b7a73c8357b6a3e223b68d38ef27368ada03617865a04cacf53"
    output_dir = Path("evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator_model = "gpt-oss:latest"
    judge_model = "VladimirGav/gemma4-26b-16GB-VRAM:latest"
    judge_status_label = "diagnostic_local_gemma_not_official"
    
    arms = ["ARM_1_BASELINE", "ARM_2_ONTOLOGY", "ARM_3_DIRECT_TEMPORAL", "ARM_4_UNIFIED_HYBRID"]
    
    # --- 1. PREFLIGHT GATE ---
    print("\n--- 1. Executing Preflight Gate ---")
    gate = PreflightGate(
        output_dir=output_dir,
        sample_path=sample_path,
        expected_sample_sha256=expected_sample_sha256,
        generator_model=generator_model,
        judge_model=judge_model,
        judge_status_label=judge_status_label
    )
    preflight_meta = gate.validate_all(arms)
    print(f"Preflight Gate PASSED: {preflight_meta}")
    
    runner = DiagnosticRunner(
        sample_path=sample_path,
        output_dir=output_dir,
        model=generator_model,
        judge_model=judge_model,
        k=5
    )
    
    manifest_path = runner.write_manifest(arms)
    print(f"Manifest written to {manifest_path}")
    
    # --- 2. PREFLIGHT SMOKE TEST (1 question x 4 arms) ---
    print("\n--- 2. Executing 1-Question Preflight Smoke Test ---")
    for arm in arms:
        smoke_res = runner.run_arm(arm, limit=1, resume=True)
        print(f"  [Smoke Test] {arm}: generated {smoke_res['total_instances']} instance in {smoke_res['runtime_s']}s")
    print("Preflight Smoke Test PASSED cleanly.")
    
    # --- 3. FULL EXPERIMENT GENERATION (40 questions x 4 arms = 160 instances) ---
    print(f"\n--- 3. Full Answer Generation (N=40 questions x {len(arms)} arms = {40 * len(arms)} predictions) ---")
    gen_start = time.time()
    gen_results = {}
    for arm in arms:
        print(f"\nProcessing {arm}: {ARM_DEFINITIONS[arm]['name']}")
        res = runner.run_arm(arm, limit=40, resume=True)
        gen_results[arm] = res
        print(f"  Finished {arm}: {res['newly_processed']} new / {res['total_instances']} total in {res['runtime_s']}s")
    gen_total_s = round(time.time() - gen_start, 2)
    print(f"\nPhase A Complete: All {len(arms)} arms generated in {gen_total_s}s ({gen_total_s/60:.1f} min)")
    
    # --- 4. DIAGNOSTIC GEMMA JUDGING (160 evaluations) ---
    print(f"\n--- 4. Diagnostic Gemma Judging (160 evaluations) ---")
    judge_start = time.time()
    judge_results = {}
    for arm in arms:
        print(f"\nJudging {arm}: {ARM_DEFINITIONS[arm]['name']}")
        res = runner.evaluate_with_judge(arm, resume=True)
        judge_results[arm] = res
        print(f"  Finished {arm}: Overall {res['overall_accuracy']}%, Answerable {res['answerable_accuracy']}%, Abstention {res['abstention_accuracy']}%")
    judge_total_s = round(time.time() - judge_start, 2)
    print(f"\nPhase B Complete: All {len(arms)} arms judged in {judge_total_s}s ({judge_total_s/60:.1f} min)")
    
    grand_total_s = round(time.time() - gen_start, 2)
    print(f"\n=== Experiment Complete in {grand_total_s}s ({grand_total_s/60:.1f} min) ===")
    
    # Write summary JSON
    summary_data = {
        "evaluation_tier": "fresh_heldout_40",
        "benchmark_status": "diagnostic_only_not_official_longmemeval",
        "sample_size": 40,
        "preflight_validation": preflight_meta,
        "runtime_seconds": {
            "generation": gen_total_s,
            "judging": judge_total_s,
            "total": grand_total_s
        },
        "arms": judge_results
    }
    with open(output_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved experiment summary to {output_dir / 'experiment_summary.json'}")

if __name__ == "__main__":
    main()
