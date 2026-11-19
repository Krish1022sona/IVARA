"""
Independent verification and paired statistical analysis of the finalization experiment (N=40).
Computes Wilson 95% CIs, exact McNemar tests, task-type breakdowns, and wins/losses.
"""

import json
import math
from collections import defaultdict
from scipy import stats

def wilson_interval(k: int, n: int, z: float = 1.95996):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(max(0.0, center - spread), 4), round(min(1.0, center + spread), 4))

def main():
    exp_dir = "evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40"
    arms = ["ARM_1_BASELINE", "ARM_2_ONTOLOGY", "ARM_3_DIRECT_TEMPORAL", "ARM_4_UNIFIED_HYBRID"]

    with open("evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json", "r", encoding="utf-8") as f:
        sample_list = json.load(f)
    q_meta = {q["question_id"]: q for q in sample_list}

    preds = {}
    judges = {}
    for arm in arms:
        with open(f"{exp_dir}/predictions_{arm}.jsonl", "r", encoding="utf-8") as f:
            preds[arm] = {line["question_id"]: line for line in (json.loads(l) for l in f)}
        with open(f"{exp_dir}/judge_results_{arm}.jsonl", "r", encoding="utf-8") as f:
            judges[arm] = {line["question_id"]: line for line in (json.loads(l) for l in f)}

    metrics = {}
    for arm in arms:
        total_q = len(judges[arm])
        correct_overall = sum(1 for j in judges[arm].values() if j["is_correct"])
        
        ans_q = [qid for qid, j in judges[arm].items() if not j["is_abstention"]]
        ans_correct = sum(1 for qid in ans_q if judges[arm][qid]["is_correct"])
        
        unans_q = [qid for qid, j in judges[arm].items() if j["is_abstention"]]
        unans_correct = sum(1 for qid in unans_q if judges[arm][qid]["is_correct"])
        
        # Task type breakdown
        tt_counts = defaultdict(lambda: {"total": 0, "correct": 0})
        for qid, j in judges[arm].items():
            tt = j["task_type"]
            tt_counts[tt]["total"] += 1
            if j["is_correct"]:
                tt_counts[tt]["correct"] += 1
                
        # Latencies
        latencies = [p["latency_ms"] / 1000.0 for p in preds[arm].values()]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        
        ci_overall = wilson_interval(correct_overall, total_q)
        ci_ans = wilson_interval(ans_correct, len(ans_q))
        ci_unans = wilson_interval(unans_correct, len(unans_q))
        
        metrics[arm] = {
            "total": total_q,
            "overall_correct": correct_overall,
            "overall_acc": round(correct_overall / total_q, 4),
            "overall_ci95": ci_overall,
            "answerable_total": len(ans_q),
            "answerable_correct": ans_correct,
            "answerable_acc": round(ans_correct / len(ans_q), 4) if ans_q else 0,
            "answerable_ci95": ci_ans,
            "unanswerable_total": len(unans_q),
            "unanswerable_correct": unans_correct,
            "unanswerable_acc": round(unans_correct / len(unans_q), 4) if unans_q else 0,
            "unanswerable_ci95": ci_unans,
            "avg_latency_sec": round(avg_lat, 2),
            "by_task_type": {k: {"total": v["total"], "correct": v["correct"], "acc": round(v["correct"]/v["total"], 3)} for k, v in sorted(tt_counts.items())}
        }

    # Paired comparisons
    paired = {}
    arm_pairs = [
        ("ARM_1_BASELINE", "ARM_2_ONTOLOGY"),
        ("ARM_1_BASELINE", "ARM_3_DIRECT_TEMPORAL"),
        ("ARM_1_BASELINE", "ARM_4_UNIFIED_HYBRID"),
        ("ARM_2_ONTOLOGY", "ARM_3_DIRECT_TEMPORAL"),
        ("ARM_2_ONTOLOGY", "ARM_4_UNIFIED_HYBRID"),
        ("ARM_3_DIRECT_TEMPORAL", "ARM_4_UNIFIED_HYBRID"),
    ]

    for arm_a, arm_b in arm_pairs:
        a_wins = 0
        b_wins = 0
        both_correct = 0
        both_wrong = 0
        a_win_qids = []
        b_win_qids = []
        
        for qid in judges[arm_a]:
            corr_a = judges[arm_a][qid]["is_correct"]
            corr_b = judges[arm_b][qid]["is_correct"]
            if corr_a and corr_b:
                both_correct += 1
            elif not corr_a and not corr_b:
                both_wrong += 1
            elif corr_a and not corr_b:
                a_wins += 1
                a_win_qids.append(qid)
            else:
                b_wins += 1
                b_win_qids.append(qid)
                
        discordant = a_wins + b_wins
        # Exact McNemar binomial test (two-tailed)
        p_val = stats.binomtest(min(a_wins, b_wins), discordant, 0.5, alternative="two-sided").pvalue if discordant > 0 else 1.0
        
        paired[f"{arm_a}__vs__{arm_b}"] = {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "arm_a_wins": a_wins,
            "arm_b_wins": b_wins,
            "arm_a_win_qids": a_win_qids,
            "arm_b_win_qids": b_win_qids,
            "discordant_pairs": discordant,
            "mcnemar_exact_pvalue": round(p_val, 4),
            "statistically_significant": bool(p_val < 0.05)
        }

    output = {
        "metrics": metrics,
        "paired_comparisons": paired
    }

    out_path = f"{exp_dir}/recomputed_metrics_and_paired_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Successfully recomputed and wrote to", out_path)
    print("\n--- METRICS TABLE ---")
    for arm, m in metrics.items():
        print(f"{arm}: Overall={m['overall_acc']*100:.1f}% ({m['overall_correct']}/{m['total']}) CI95={m['overall_ci95']} | Ans={m['answerable_acc']*100:.1f}% | Unans={m['unanswerable_acc']*100:.1f}% | AvgLat={m['avg_latency_sec']}s")
    
    print("\n--- TASK TYPE BREAKDOWN (ACCURACY %) ---")
    all_tt = sorted(list(next(iter(metrics.values()))["by_task_type"].keys()))
    header = f"{'Task Type':<30}" + "".join([f"{arm:<24}" for arm in arms])
    print(header)
    for tt in all_tt:
        row = f"{tt:<30}"
        for arm in arms:
            entry = metrics[arm]["by_task_type"][tt]
            row += f"{entry['acc']*100:.1f}% ({entry['correct']}/{entry['total']})".ljust(24)
        print(row)

    print("\n--- PAIRED COMPARISONS (MCNEMAR EXACT TEST) ---")
    for k, v in paired.items():
        print(f"{k}:")
        print(f"  {v['arm_a']} wins: {v['arm_a_wins']} | {v['arm_b']} wins: {v['arm_b_wins']} | Ties: {v['both_correct'] + v['both_wrong']} (Both correct: {v['both_correct']}, Both wrong: {v['both_wrong']})")
        print(f"  Discordant: {v['discordant_pairs']} | McNemar exact p-value: {v['mcnemar_exact_pvalue']} (Significant: {v['statistically_significant']})")
        if v['arm_a_win_qids']:
            print(f"  {v['arm_a']} unique wins on: {v['arm_a_win_qids']}")
        if v['arm_b_win_qids']:
            print(f"  {v['arm_b']} unique wins on: {v['arm_b_win_qids']}")

if __name__ == "__main__":
    main()
