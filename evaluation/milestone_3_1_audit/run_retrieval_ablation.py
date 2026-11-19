"""Phase 4: Retrieval Ablation Study for Milestone 3 Audit.

Analyzes the full 5-way ablation across the 60 stratified questions:
A. Episodic Only
B. Semantic Only
C. Episodic + Semantic
D. Episodic + Semantic + Graph
E. Counterfactual Ablation (Disabling Graph on discordant questions)
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
RESULTS_DIR = WORKSPACE_ROOT / "evaluation" / "milestone_3_results"
AUDIT_DIR = WORKSPACE_ROOT / "evaluation" / "milestone_3_1_audit"

from evaluation.validation_sprint.dataset_expanded import QUESTIONS

def run_ablation():
    test_questions = [q for q in QUESTIONS if q.get("split") == "test"]
    by_cat = defaultdict(list)
    for q in test_questions:
        by_cat[q["category"]].append(q)
    stratified_q = []
    for cat in sorted(by_cat.keys()):
        stratified_q.extend(by_cat[cat][:5])
    strat_ids = [q["id"] for q in stratified_q]

    files = {
        "Episodic Only": RESULTS_DIR / "predictions_arm_b_episodic_only.json",
        "Semantic Only": RESULTS_DIR / "predictions_arm_c_semantic_only.json",
        "Episodic + Semantic": RESULTS_DIR / "predictions_arm_d_episodic_semantic.json",
        "Episodic + Semantic + Graph": RESULTS_DIR / "predictions_arm_e_unified_tri_source.json"
    }

    data = {}
    for name, path in files.items():
        with open(path, "r", encoding="utf-8") as f:
            data[name] = json.load(f)

    # Counterfactual arm: In Arm E, toggle graph off for the 4 discordant questions (reverts to Arm D)
    data["Counterfactual (Graph Selectively Disabled)"] = {}
    discordant_qids = {"cmf02", "xss04", "h03", "n03"}
    for qid in strat_ids:
        if qid in discordant_qids:
            # Revert to Arm D (no graph)
            data["Counterfactual (Graph Selectively Disabled)"][qid] = data["Episodic + Semantic"][qid]
        else:
            data["Counterfactual (Graph Selectively Disabled)"][qid] = data["Episodic + Semantic + Graph"][qid]

    ablation_summary = {}
    for arm_name, preds_map in data.items():
        preds = [preds_map[qid] for qid in strat_ids]
        bins = [p["answer_metrics"]["binary_accuracy"] for p in preds]
        covs = [p["answer_metrics"]["fact_coverage"] for p in preds]
        r5s = [p["retrieval_metrics"]["recall_at_5"] for p in preds]
        lats = [p["latency_ms"] for p in preds]

        ablation_summary[arm_name] = {
            "binary_accuracy": round(float(np.mean(bins)), 4),
            "fact_coverage": round(float(np.mean(covs)), 4),
            "recall_at_5": round(float(np.mean(r5s)), 4),
            "mean_latency_ms": round(float(np.mean(lats)), 1)
        }

    # Component lift analysis
    epi_only_bin = ablation_summary["Episodic Only"]["binary_accuracy"]
    sem_only_bin = ablation_summary["Semantic Only"]["binary_accuracy"]
    epi_sem_bin = ablation_summary["Episodic + Semantic"]["binary_accuracy"]
    full_bin = ablation_summary["Episodic + Semantic + Graph"]["binary_accuracy"]

    epi_only_cov = ablation_summary["Episodic Only"]["fact_coverage"]
    sem_only_cov = ablation_summary["Semantic Only"]["fact_coverage"]
    epi_sem_cov = ablation_summary["Episodic + Semantic"]["fact_coverage"]
    full_cov = ablation_summary["Episodic + Semantic + Graph"]["fact_coverage"]

    component_lifts = {
        "episodic_lift_over_semantic": {
            "delta_binary": round(epi_only_bin - sem_only_bin, 4),
            "delta_coverage": round(epi_only_cov - sem_only_cov, 4)
        },
        "semantic_fusion_delta_over_episodic": {
            "delta_binary": round(epi_sem_bin - epi_only_bin, 4),
            "delta_coverage": round(epi_sem_cov - epi_only_cov, 4)
        },
        "graph_lift_over_fused_episodic_semantic": {
            "delta_binary": round(full_bin - epi_sem_bin, 4),
            "delta_coverage": round(full_cov - epi_sem_cov, 4)
        }
    }

    full_output = {
        "ablation_table": ablation_summary,
        "component_lifts": component_lifts
    }

    out_file = AUDIT_DIR / "phase4_retrieval_ablation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"Saved: {out_file}")

    print("\n=== PHASE 4 RETRIEVAL ABLATION SUMMARY ===")
    print(f"{'Configuration':<45} | {'Binary Acc':<11} | {'Fact Coverage':<13} | {'Recall@5':<9} | {'Mean Latency':<12}")
    print("-" * 105)
    for arm_name, s in ablation_summary.items():
        print(f"{arm_name:<45} | {s['binary_accuracy']*100:>9.2f}% | {s['fact_coverage']*100:>11.2f}% | {s['recall_at_5']*100:>7.2f}% | {s['mean_latency_ms']:>9.1f} ms")

    print("\n--- Component Contributions ---")
    print(f"1. Episodic vs Semantic Only: Binary {component_lifts['episodic_lift_over_semantic']['delta_binary']*100:+.2f}%, Coverage {component_lifts['episodic_lift_over_semantic']['delta_coverage']*100:+.2f}%")
    print(f"2. Adding Semantic to Episodic (Fusion): Binary {component_lifts['semantic_fusion_delta_over_episodic']['delta_binary']*100:+.2f}%, Coverage {component_lifts['semantic_fusion_delta_over_episodic']['delta_coverage']*100:+.2f}%")
    print(f"3. Adding Graph to Fused (Epi+Sem): Binary {component_lifts['graph_lift_over_fused_episodic_semantic']['delta_binary']*100:+.2f}%, Coverage {component_lifts['graph_lift_over_fused_episodic_semantic']['delta_coverage']*100:+.2f}%")

if __name__ == "__main__":
    run_ablation()
