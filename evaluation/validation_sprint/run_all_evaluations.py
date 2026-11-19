"""Master script to run all benchmark evaluations and ablations on the held-out test split (250 questions)."""

import json
from pathlib import Path
import time

from evaluation.validation_sprint.dataset_expanded import SESSIONS, QUESTIONS
from evaluation.validation_sprint.runner import run_evaluation_arm, compare_and_analyze

test_qs = [q for q in QUESTIONS if q["split"] == "test"]
print(f"Total held-out test questions: {len(test_qs)}")

# Arms to evaluate in sequence
all_arms = [
    "baseline_hybrid_speaker_aware",
    "baseline_vector",
    "baseline_hybrid",
    "ivara_hybrid_no_graph",
    "ivara_full",
    # Ablations
    "ivara_no_ontology_no_graph",
    "ivara_graph_no_ontology",
    "ivara_full_no_temporal"
]

for arm in all_arms:
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
print("Complete.")

