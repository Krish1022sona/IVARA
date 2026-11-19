"""Autonomous runner for all remaining arms in the benchmark validation sprint."""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.validation_sprint.dataset_expanded import SESSIONS, QUESTIONS
from evaluation.validation_sprint.runner import run_evaluation_arm, compare_and_analyze

test_qs = [q for q in QUESTIONS if q["split"] == "test"]
print(f"Total held-out test questions: {len(test_qs)}")

# Wait for baseline_vector if still running
res_dir = Path("evaluation/validation_sprint/results")
bv_file = res_dir / "predictions_baseline_vector.json"

while True:
    if bv_file.exists():
        try:
            with open(bv_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            print(f"[Wait] baseline_vector has {len(d)} / 250 predictions...")
            if len(d) >= len(test_qs):
                break
        except Exception:
            pass
    time.sleep(5)

print("\n--- baseline_vector verified complete. Proceeding to remaining arms ---")

remaining_arms = [
    "baseline_hybrid",
    "ivara_hybrid_no_graph",
    "ivara_full_no_temporal",
    "ivara_no_ontology_no_graph"
]

for arm in remaining_arms:
    print(f"\n=======================================================")
    print(f"  RUNNING ARM: {arm}")
    print(f"=======================================================")
    t0 = time.time()
    preds = run_evaluation_arm(arm, test_qs, SESSIONS, resume=True)
    elapsed = time.time() - t0
    print(f"Arm {arm} completed ({len(preds)} total predictions, elapsed {elapsed:.1f}s)")

# Final comparison and statistical tests
print("\n=======================================================")
print("  GENERATING COMPREHENSIVE VALIDATION REPORT & STATS   ")
print("=======================================================")
summary = compare_and_analyze(test_qs)
print("All remaining arms and statistical deliverables generated successfully!")
