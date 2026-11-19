"""
Step 9: Evaluation and Metric Integrity Audit.
Exhaustively audits:
1. Consistency of question IDs across all 4 configurations in Step 7.
2. Alignment between candidate predictions and Gemma local judge records in Step 8.
3. Verification of reported metrics, denominators, and 95% Wilson confidence intervals.
4. Verification of paired win/loss/tie comparisons.
5. Verification of dataset unanswerable labels vs recorded is_abstention flags.
Outputs: evaluation/step9/metric_integrity_audit.json.
"""

import json
import math
from pathlib import Path
from typing import Any

STEP7_DIR = Path("evaluation/data/longmemeval/runs/campaign_v1_pilot")
STEP8_DIR = Path("evaluation/step8")
STEP9_DIR = Path("evaluation/step9")
STEP9_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILE = Path("evaluation/data/longmemeval/longmemeval_s_cleaned.json")

def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - spread), min(1.0, center + spread))

def main():
    print("=== Running Evaluation Integrity Audit ===")
    audit_results = {
        "status": "PASS",
        "checks": {},
        "discrepancies": []
    }

    # 1. Load ground truth dataset
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        ground_truth_dataset = {item["question_id"]: item for item in json.load(f)}

    # 2. Load Step 7 Raw Predictions
    configs = [
        ("Config-A", "config_a", "CONFIG-A"),
        ("Config-B", "config_b", "CONFIG-B"),
        ("Config-C", "config_c", "CONFIG-C"),
        ("Config-D", "config_d", "CONFIG-D")
    ]
    raw_predictions = {}
    qids_by_config = {}

    for cfg_key, folder, prefix in configs:
        res_file = STEP7_DIR / folder / f"results_config_{prefix}.jsonl"
        recs = []
        with open(res_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        raw_predictions[cfg_key] = {r["question_id"]: r for r in recs}
        qids_by_config[cfg_key] = [r["question_id"] for r in recs]

    # Check 1: Question ID alignment and duplicates
    base_qids = qids_by_config["Config-A"]
    qids_consistent = True
    no_duplicates = True
    for cfg_key, qids in qids_by_config.items():
        if len(qids) != len(set(qids)):
            no_duplicates = False
            audit_results["discrepancies"].append(f"Duplicate question IDs found in {cfg_key}")
        if qids != base_qids:
            qids_consistent = False
            audit_results["discrepancies"].append(f"Mismatched question IDs or ordering in {cfg_key}")

    audit_results["checks"]["qid_consistency"] = {
        "passed": qids_consistent and no_duplicates,
        "total_instances_per_config": len(base_qids),
        "no_duplicates": no_duplicates,
        "exact_order_match": qids_consistent
    }

    # Check 2: Unanswerable labels vs dataset
    abstention_alignment = True
    def is_unanswerable_dataset(item: dict) -> bool:
        qid = item.get("question_id", "")
        qtype = item.get("question_type", "").lower()
        ans = str(item.get("answer", "")).lower()
        return "_abs" in qid or "abstention" in qtype or "not enough" in ans or "unanswerable" in ans

    abs_count_dataset = sum(1 for qid in base_qids if is_unanswerable_dataset(ground_truth_dataset[qid]))
    
    for cfg_key, recs in raw_predictions.items():
        for qid, r in recs.items():
            ds_inst = ground_truth_dataset[qid]
            ds_is_abs = is_unanswerable_dataset(ds_inst)
            rec_is_abs = bool(r.get("is_abstention"))
            if ds_is_abs != rec_is_abs:
                abstention_alignment = False
                audit_results["discrepancies"].append(f"Abstention mismatch for {qid} in {cfg_key}: dataset={ds_is_abs}, record={rec_is_abs}")

    audit_results["checks"]["abstention_label_alignment"] = {
        "passed": abstention_alignment,
        "total_abstentions": abs_count_dataset,
        "total_answerable": len(base_qids) - abs_count_dataset,
        "abstention_fraction": round(abs_count_dataset / len(base_qids), 4)
    }

    # Check 3: Alignment between Step 7 raw predictions and Step 8 Gemma local judge
    judge_file = STEP8_DIR / "local_judge_results.json"
    with open(judge_file, "r", encoding="utf-8") as f:
        judge_data = json.load(f)

    judge_preds = judge_data.get("predictions_by_config", {})
    judge_alignment = True
    judge_failures_found = 0

    for cfg_key, _, _ in configs:
        j_cfg = judge_preds.get(cfg_key, {})
        r_cfg = raw_predictions.get(cfg_key, {})
        if len(j_cfg) != len(r_cfg):
            judge_alignment = False
            audit_results["discrepancies"].append(f"Judge count mismatch in {cfg_key}: judge={len(j_cfg)}, raw={len(r_cfg)}")
        for qid, r in r_cfg.items():
            if qid not in j_cfg:
                judge_alignment = False
                audit_results["discrepancies"].append(f"Missing judge judgment for {qid} in {cfg_key}")
                continue
            j_item = j_cfg[qid]
            if j_item.get("judge_failed"):
                judge_failures_found += 1
            # Verify is_abstention consistency
            if j_item.get("is_abstention") != r.get("is_abstention"):
                judge_alignment = False
                audit_results["discrepancies"].append(f"is_abstention mismatch in judge record for {qid} in {cfg_key}")

    audit_results["checks"]["judge_record_alignment"] = {
        "passed": judge_alignment,
        "total_judgments_per_config": {cfg: len(judge_preds.get(cfg, {})) for cfg, _, _ in configs},
        "judge_failures": judge_failures_found
    }

    # Check 4: Recomputation of Reported Gemma Metrics
    reported_metrics = judge_data.get("metrics_by_config", {})
    metrics_recomputed = {}
    metrics_match = True

    for cfg_key, _, _ in configs:
        j_cfg = judge_preds.get(cfg_key, {})
        n_total = len(j_cfg)
        ans_items = [v for v in j_cfg.values() if not v["is_abstention"]]
        abs_items = [v for v in j_cfg.values() if v["is_abstention"]]

        n_corr = sum(1 for v in j_cfg.values() if v["is_correct"])
        n_ans_corr = sum(1 for v in ans_items if v["is_correct"])
        n_abs_corr = sum(1 for v in abs_items if v["is_correct"])

        calc_overall_acc = round((n_corr / n_total) * 100, 2)
        calc_ans_acc = round((n_ans_corr / len(ans_items)) * 100, 2)
        calc_abs_acc = round((n_abs_corr / len(abs_items)) * 100, 2)

        ci_overall = [round(x * 100, 2) for x in compute_wilson_ci(n_corr, n_total)]
        ci_ans = [round(x * 100, 2) for x in compute_wilson_ci(n_ans_corr, len(ans_items))]

        rep = reported_metrics.get(cfg_key, {})
        if (rep.get("overall_accuracy") != calc_overall_acc or
            rep.get("answerable_accuracy") != calc_ans_acc or
            rep.get("abstention_accuracy") != calc_abs_acc or
            rep.get("overall_accuracy_ci95") != ci_overall or
            rep.get("answerable_accuracy_ci95") != ci_ans):
            metrics_match = False
            audit_results["discrepancies"].append(f"Metric recomputation mismatch for {cfg_key}")

        metrics_recomputed[cfg_key] = {
            "total_questions": n_total,
            "answerable_questions": len(ans_items),
            "unanswerable_questions": len(abs_items),
            "total_correct": n_corr,
            "answerable_correct": n_ans_corr,
            "abstention_correct": n_abs_corr,
            "overall_accuracy": calc_overall_acc,
            "overall_ci95": ci_overall,
            "answerable_accuracy": calc_ans_acc,
            "answerable_ci95": ci_ans,
            "abstention_accuracy": calc_abs_acc
        }

    audit_results["checks"]["gemma_metrics_recomputation"] = {
        "passed": metrics_match,
        "recomputed_metrics": metrics_recomputed
    }

    # Check 5: Recomputation of Gemma Paired Comparisons
    reported_pairs = judge_data.get("paired_comparisons", {})
    pairs = [
        ("Config-B", "Config-A"),
        ("Config-C", "Config-B"),
        ("Config-D", "Config-C"),
        ("Config-D", "Config-A")
    ]
    pairs_match = True
    recomputed_pairs = {}

    for c_new, c_base in pairs:
        pair_key = f"{c_new}_vs_{c_base}"
        j_new = judge_preds[c_new]
        j_base = judge_preds[c_base]
        wins, losses, ties = 0, 0, 0
        diffs = []
        for qid in base_qids:
            s_n = 1 if j_new[qid]["is_correct"] else 0
            s_b = 1 if j_base[qid]["is_correct"] else 0
            diffs.append(s_n - s_b)
            if s_n > s_b:
                wins += 1
            elif s_n < s_b:
                losses += 1
            else:
                ties += 1
        net_delta = round((sum(diffs) / len(diffs)) * 100, 2)
        rep_p = reported_pairs.get(pair_key, {})
        if (rep_p.get("wins") != wins or
            rep_p.get("losses") != losses or
            rep_p.get("ties") != ties or
            rep_p.get("net_accuracy_delta_pp") != net_delta):
            pairs_match = False
            audit_results["discrepancies"].append(f"Paired comparison mismatch for {pair_key}")

        recomputed_pairs[pair_key] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "net_accuracy_delta_pp": net_delta
        }

    audit_results["checks"]["paired_comparisons_recomputation"] = {
        "passed": pairs_match,
        "recomputed_pairs": recomputed_pairs
    }

    # Final status
    if audit_results["discrepancies"]:
        audit_results["status"] = "FAIL"

    output_path = STEP9_DIR / "metric_integrity_audit.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print(f"Audit completed. Status: {audit_results['status']}")
    print(f"Total Discrepancies: {len(audit_results['discrepancies'])}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
