"""
Smoke test preflight for the 6-arm diagnostic evaluation.
Runs 2 questions from diagnostic_sample_100.json across all 6 arms using Ollama gpt-oss:latest.
Also verifies 1 judgment with Gemma judge.
"""

import sys
sys.path.insert(0, ".")

import time
from pathlib import Path
from evaluation.autonomous_sprint.diagnostic_runner import DiagnosticRunner, ARM_DEFINITIONS

def main():
    sample_path = Path("evaluation/autonomous_sprint/diagnostic_sample_100.json")
    output_dir = Path("evaluation/autonomous_sprint/runs/smoke_test_run")
    
    runner = DiagnosticRunner(
        sample_path=sample_path,
        output_dir=output_dir,
        model="gpt-oss:latest",
        judge_model="VladimirGav/gemma4-26b-16GB-VRAM:latest",
        k=5
    )
    
    arms = list(ARM_DEFINITIONS.keys())
    runner.write_manifest(arms)
    
    print("=== Starting 6-Arm Smoke Test (limit=2 questions per arm) ===")
    t0 = time.time()
    
    arm_results = {}
    for arm in arms:
        res = runner.run_arm(arm, limit=2, resume=False)
        arm_results[arm] = res
        
    print("\n=== Smoke Test Generations Complete ===")
    for arm, res in arm_results.items():
        print(f"  {arm}: {res['newly_processed']} questions in {res['runtime_s']}s")
        
    # Run judge smoke test on ARM_B
    print("\n=== Verifying Diagnostic Judge Smoke Test (ARM_B) ===")
    judge_summary = runner.evaluate_with_judge("ARM_B", resume=False)
    print(f"  Judge ARM_B result: {judge_summary['overall_accuracy']}% ({judge_summary['total_instances']} instances)")
    
    total_smoke_time = round(time.time() - t0, 2)
    print(f"\n=== Smoke Test Passed in {total_smoke_time}s ===")

if __name__ == "__main__":
    main()
