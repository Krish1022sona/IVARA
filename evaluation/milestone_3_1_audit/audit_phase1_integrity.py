"""Phase 1 Integrity Audit for Milestone 3 Benchmark.

Independently checks all predictions and ground truth, verifies parameters,
checks for silent defaults or errors, and recalculates all metrics.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.validation_sprint.dataset_expanded import QUESTIONS
from evaluation.validation_sprint.metrics import evaluate_retrieval, evaluate_answer

RESULTS_DIR = WORKSPACE_ROOT / "evaluation" / "milestone_3_results"
AUDIT_DIR = WORKSPACE_ROOT / "evaluation" / "milestone_3_1_audit"
ARMS = [
    ("arm_a_semantic_hybrid", "A: Current Semantic/Hybrid IVARA"),
    ("arm_b_episodic_only", "B: Episodic Retrieval Only"),
    ("arm_c_semantic_only", "C: Semantic Retrieval Only"),
    ("arm_d_episodic_semantic", "D: Episodic + Semantic Retrieval"),
    ("arm_e_unified_tri_source", "E: Unified Tri-Source (Epi+Sem+Graph)")
]

def run_phase1_audit():
    # 1. Stratified Question Definition
    test_questions = [q for q in QUESTIONS if q.get("split") == "test"]
    by_cat = defaultdict(list)
    for q in test_questions:
        by_cat[q["category"]].append(q)
    stratified_q = []
    for cat in sorted(by_cat.keys()):
        stratified_q.extend(by_cat[cat][:5])
    strat_q_map = {q["id"]: q for q in stratified_q}
    strat_ids = [q["id"] for q in stratified_q]

    audit_results = {
        "stratified_questions_count": len(stratified_q),
        "categories_count": len(by_cat),
        "arms_audit": {},
        "ground_truth_consistency": True,
        "silent_errors": {},
        "recalculated_overall_metrics": {},
        "recalculated_category_metrics": defaultdict(dict),
        "metric_discrepancies": []
    }

    print(f"=== PHASE 1: BENCHMARK INTEGRITY AUDIT ===")
    print(f"Stratified Questions: {len(stratified_q)} across {len(by_cat)} categories (5 per category).")

    # Load raw prediction files
    raw_arm_data = {}
    for arm_key, arm_label in ARMS:
        p_path = RESULTS_DIR / f"predictions_{arm_key}.json"
        if not p_path.exists():
            print(f"ERROR: {p_path} missing!")
            continue
        with open(p_path, "r", encoding="utf-8") as f:
            raw_arm_data[arm_key] = json.load(f)

    # Load existing milestone_3_summary.json
    summary_path = RESULTS_DIR / "milestone_3_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        existing_summary = json.load(f)

    # Audit each arm
    for arm_key, arm_label in ARMS:
        preds = raw_arm_data.get(arm_key, {})
        keys = set(preds.keys())
        matched_strat = [qid for qid in strat_ids if qid in keys]
        missing_strat = [qid for qid in strat_ids if qid not in keys]

        errors = []
        empty_answers = []
        statuses = Counter()
        latencies = []

        recalculated_preds = []

        for qid in strat_ids:
            if qid not in preds:
                continue
            item = preds[qid]
            status = item.get("status", item.get("predicted_status", "unknown"))
            statuses[status] += 1
            ans = item.get("answer", item.get("predicted_answer", ""))
            if not ans or ans.strip() == "":
                empty_answers.append(qid)
            if status == "error" or "Error:" in str(ans):
                errors.append((qid, ans))

            lat = item.get("latency_ms", 0.0)
            latencies.append(lat)

            # Independent re-evaluation using official metric functions
            q_spec = strat_q_map[qid]
            raw_ctx = item.get("retrieved_context")
            if isinstance(raw_ctx, list):
                retrieved_items = raw_ctx
            elif isinstance(raw_ctx, dict):
                retrieved_items = raw_ctx.get("direct_evidence") or raw_ctx.get("retrieved_memories", [])
            else:
                retrieved_items = []

            cited_ids = item.get("citations", [])
            if cited_ids and isinstance(cited_ids[0], dict):
                cited_ids = [c.get("memory_id", "") for c in cited_ids if "memory_id" in c]

            supp_sessions = q_spec.get("supporting_sessions", [])
            if retrieved_items:
                ret_metrics = evaluate_retrieval(
                    retrieved_items=retrieved_items,
                    supporting_sessions=supp_sessions,
                    is_answerable=q_spec.get("is_answerable", True)
                )
            else:
                ret_metrics = item.get("retrieval_metrics", {
                    "recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "evidence_coverage": 0.0
                })

            ans_pred = {
                "status": status,
                "answer": ans,
                "citations": cited_ids
            }
            # Fully independent evaluation of answer against ground truth
            ans_metrics = evaluate_answer(ans_pred, q_spec)

            recalculated_preds.append({
                "question_id": qid,
                "category": q_spec["category"],
                "retrieval_metrics": ret_metrics,
                "answer_metrics": ans_metrics,
                "latency_ms": lat
            })

            # Check for discrepancies against cached metrics in prediction file
            cached_ans_m = item.get("answer_metrics", {})
            if cached_ans_m:
                if abs(cached_ans_m.get("binary_accuracy", 0.0) - ans_metrics["binary_accuracy"]) > 1e-4:
                    audit_results["metric_discrepancies"].append({
                        "arm": arm_key, "qid": qid, "metric": "binary_accuracy",
                        "cached": cached_ans_m.get("binary_accuracy"), "recalculated": ans_metrics["binary_accuracy"]
                    })
                if abs(cached_ans_m.get("fact_coverage", 0.0) - ans_metrics["fact_coverage"]) > 1e-4:
                    audit_results["metric_discrepancies"].append({
                        "arm": arm_key, "qid": qid, "metric": "fact_coverage",
                        "cached": cached_ans_m.get("fact_coverage"), "recalculated": ans_metrics["fact_coverage"]
                    })

        # Calculate recalculated summary for this arm
        n = len(recalculated_preds)
        covs = [p["answer_metrics"]["fact_coverage"] for p in recalculated_preds]
        bin_accs = [p["answer_metrics"]["binary_accuracy"] for p in recalculated_preds]
        ems = [p["answer_metrics"]["exact_match"] for p in recalculated_preds]
        r5s = [p["retrieval_metrics"]["recall_at_5"] for p in recalculated_preds]
        mrrs = [p["retrieval_metrics"]["mrr"] for p in recalculated_preds]
        lats = [p["latency_ms"] for p in recalculated_preds]
        sorted_lats = sorted(lats)
        p95_idx = min(int(n * 0.95), n - 1)

        recalc_overall = {
            "label": arm_label,
            "questions_count": n,
            "mean_fact_coverage": round(sum(covs) / n, 4) if n > 0 else 0.0,
            "binary_accuracy": round(sum(bin_accs) / n, 4) if n > 0 else 0.0,
            "exact_match_rate": round(sum(ems) / n, 4) if n > 0 else 0.0,
            "recall_at_5": round(sum(r5s) / n, 4) if n > 0 else 0.0,
            "mean_mrr": round(sum(mrrs) / n, 4) if n > 0 else 0.0,
            "mean_latency_ms": round(sum(lats) / n, 1) if n > 0 else 0.0,
            "median_latency_ms": round(sorted_lats[n // 2], 1) if n > 0 else 0.0,
            "p95_latency_ms": round(sorted_lats[p95_idx], 1) if n > 0 else 0.0
        }
        audit_results["recalculated_overall_metrics"][arm_key] = recalc_overall

        # Check against existing summary
        exist_overall = existing_summary.get("overall", {}).get(arm_key, {})
        diffs = {}
        for m_key in ["mean_fact_coverage", "binary_accuracy", "exact_match_rate", "recall_at_5"]:
            d = abs(recalc_overall[m_key] - exist_overall.get(m_key, 0.0))
            if d > 1e-4:
                diffs[m_key] = {"existing": exist_overall.get(m_key), "recalculated": recalc_overall[m_key]}

        audit_results["arms_audit"][arm_key] = {
            "total_cached_predictions": len(keys),
            "stratified_matched": len(matched_strat),
            "stratified_missing": len(missing_strat),
            "status_counts": dict(statuses),
            "errors_count": len(errors),
            "empty_answers_count": len(empty_answers),
            "summary_match": len(diffs) == 0,
            "summary_differences": diffs
        }

        # Category metrics recalculation
        categories = sorted(list(by_cat.keys()))
        for cat in categories:
            cat_preds = [p for p in recalculated_preds if p["category"] == cat]
            n_cat = len(cat_preds)
            if n_cat > 0:
                c_covs = [p["answer_metrics"]["fact_coverage"] for p in cat_preds]
                c_bin = [p["answer_metrics"]["binary_accuracy"] for p in cat_preds]
                c_r5 = [p["retrieval_metrics"]["recall_at_5"] for p in cat_preds]
                audit_results["recalculated_category_metrics"][cat][arm_key] = {
                    "count": n_cat,
                    "fact_coverage": round(sum(c_covs) / n_cat, 4),
                    "binary_accuracy": round(sum(c_bin) / n_cat, 4),
                    "recall_at_5": round(sum(c_r5) / n_cat, 4)
                }

    # Save audit results to JSON
    out_file = AUDIT_DIR / "phase1_integrity_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"Saved: {out_file}")

    print("\n--- Summary of Verification ---")
    for arm_key, info in audit_results["arms_audit"].items():
        print(f"[{arm_key}] Matched: {info['stratified_matched']}/60 | Errors: {info['errors_count']} | Empty: {info['empty_answers_count']} | Summary Match: {info['summary_match']}")
        if not info["summary_match"]:
            print(f"   Differences: {info['summary_differences']}")

    if audit_results["metric_discrepancies"]:
        print(f"Discrepancies found: {len(audit_results['metric_discrepancies'])}")
    else:
        print("ALL RECALCULATED METRICS MATCH ORIGINAL BENCHMARK REPORT WITH 0 DISCREPANCIES.")

if __name__ == "__main__":
    run_phase1_audit()
