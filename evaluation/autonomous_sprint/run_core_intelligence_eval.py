"""
Core Intelligence Sprint: Targeted Evaluation on Fresh Held-Out Dataset (N=40).

Evaluates the completed core intelligence pipeline:
- ARM_5_CORE_INTELLIGENCE: Integrated Reasoning (Diverse Retrieval + Direct Seed Temporal + Event-Year Priority + Evidence Gate + Contradiction Analysis)
Against:
- ARM_4_UNIFIED_HYBRID: Prior Best Candidate (Single-session top-k retrieval)
- ARM_1_BASELINE: Dense + BM25

Dataset: evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json
Output: evaluation/autonomous_sprint/runs/core_intelligence_eval/
"""

import sys
import os
os.environ["OLLAMA_LLM_LIBRARY"] = "cuda_v12"
sys.path.insert(0, ".")

import json
import math
import time
from pathlib import Path
from collections import defaultdict
from scipy import stats

from evaluation.autonomous_sprint.preflight import PreflightGate
from evaluation.autonomous_sprint.diagnostic_runner import DiagnosticRunner, ARM_DEFINITIONS


def wilson_interval(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(max(0.0, center - spread), 4), round(min(1.0, center + spread), 4))


def main():
    print("=" * 80)
    print("  IVARA CORE INTELLIGENCE EVALUATION -- TARGETED EXPERIMENT (N=40)")
    print("=" * 80)

    sample_path = Path("evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json")
    expected_sample_sha256 = "5a42338eb5758b7a73c8357b6a3e223b68d38ef27368ada03617865a04cacf53"
    output_dir = Path("evaluation/autonomous_sprint/runs/core_intelligence_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    generator_model = "gpt-oss:latest"
    judge_model = "VladimirGav/gemma4-26b-16GB-VRAM:latest"
    judge_status_label = "diagnostic_local_gemma_not_official"

    eval_arm = "ARM_5_CORE_INTELLIGENCE"

    # 1. PREFLIGHT GATE
    print("\n--- 1. Executing Preflight Gate ---")
    gate = PreflightGate(
        output_dir=output_dir,
        sample_path=sample_path,
        expected_sample_sha256=expected_sample_sha256,
        generator_model=generator_model,
        judge_model=judge_model,
        judge_status_label=judge_status_label
    )
    preflight_meta = gate.validate_all([eval_arm])
    print(f"Preflight Gate PASSED cleanly: {preflight_meta['status']}\n")

    runner = DiagnosticRunner(
        sample_path=sample_path,
        output_dir=output_dir,
        model=generator_model,
        judge_model=judge_model,
        k=5
    )

    manifest_path = runner.write_manifest([eval_arm])
    print(f"Manifest written to {manifest_path}")

    # 2. RUN GENERATION FOR ARM_5
    print(f"\n--- 2. Generating Predictions for {eval_arm} (N=40) ---")
    gen_start = time.time()
    gen_res = runner.run_arm(eval_arm, limit=40, resume=True)
    gen_time_s = round(time.time() - gen_start, 2)
    print(f"Generation Complete: {gen_res['total_instances']} instances in {gen_time_s}s ({gen_time_s/60:.1f} min)")

    # 3. DIAGNOSTIC GEMMA JUDGING
    print(f"\n--- 3. Diagnostic Gemma Judging for {eval_arm} (N=40) ---")
    judge_start = time.time()
    judge_res = runner.evaluate_with_judge(eval_arm, resume=True)
    judge_time_s = round(time.time() - judge_start, 2)
    print(f"Judging Complete: {judge_res['total_instances']} instances in {judge_time_s}s ({judge_time_s/60:.1f} min)")

    # 4. LOAD HISTORICAL ARM_1 AND ARM_4 FOR PAIRED COMPARISON
    hist_dir = Path("evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40")
    with open(sample_path, "r", encoding="utf-8") as f:
        sample_list = json.load(f)
    q_meta = {q["question_id"]: q for q in sample_list}

    # Load ARM_5 predictions & judgments
    with open(output_dir / f"predictions_{eval_arm}.jsonl", "r", encoding="utf-8") as f:
        preds_arm5 = {l["question_id"]: l for l in (json.loads(line) for line in f if line.strip())}
    with open(output_dir / f"judge_results_{eval_arm}.jsonl", "r", encoding="utf-8") as f:
        judges_arm5 = {l["question_id"]: l for l in (json.loads(line) for line in f if line.strip())}

    # Load ARM_4 predictions & judgments
    with open(hist_dir / "predictions_ARM_4_UNIFIED_HYBRID.jsonl", "r", encoding="utf-8") as f:
        preds_arm4 = {l["question_id"]: l for l in (json.loads(line) for line in f if line.strip())}
    with open(hist_dir / "judge_results_ARM_4_UNIFIED_HYBRID.jsonl", "r", encoding="utf-8") as f:
        judges_arm4 = {l["question_id"]: l for l in (json.loads(line) for line in f if line.strip())}

    # Load ARM_1 predictions & judgments
    with open(hist_dir / "predictions_ARM_1_BASELINE.jsonl", "r", encoding="utf-8") as f:
        preds_arm1 = {l["question_id"]: l for l in (json.loads(line) for line in f if line.strip())}
    with open(hist_dir / "judge_results_ARM_1_BASELINE.jsonl", "r", encoding="utf-8") as f:
        judges_arm1 = {l["question_id"]: l for l in (json.loads(line) for line in f if line.strip())}

    # 5. METRICS ANALYSIS
    def calc_metrics(judges_dict, preds_dict):
        total = len(judges_dict)
        correct_total = sum(1 for j in judges_dict.values() if j["is_correct"])
        
        ans_q = [qid for qid, j in judges_dict.items() if not j["is_abstention"]]
        ans_corr = sum(1 for qid in ans_q if judges_dict[qid]["is_correct"])
        
        unans_q = [qid for qid, j in judges_dict.items() if j["is_abstention"]]
        unans_corr = sum(1 for qid in unans_q if judges_dict[qid]["is_correct"])

        tt_breakdown = defaultdict(lambda: {"total": 0, "correct": 0})
        for qid, j in judges_dict.items():
            tt = j["task_type"]
            tt_breakdown[tt]["total"] += 1
            if j["is_correct"]:
                tt_breakdown[tt]["correct"] += 1

        lats = [p["latency_ms"] / 1000.0 for p in preds_dict.values() if "latency_ms" in p]
        avg_lat = sum(lats) / len(lats) if lats else 0.0

        return {
            "total": total,
            "overall_correct": correct_total,
            "overall_pct": round(correct_total / total * 100, 2) if total else 0.0,
            "overall_ci": wilson_interval(correct_total, total),
            "ans_total": len(ans_q),
            "ans_correct": ans_corr,
            "ans_pct": round(ans_corr / len(ans_q) * 100, 2) if ans_q else 0.0,
            "ans_ci": wilson_interval(ans_corr, len(ans_q)),
            "unans_total": len(unans_q),
            "unans_correct": unans_corr,
            "unans_pct": round(unans_corr / len(unans_q) * 100, 2) if unans_q else 0.0,
            "task_types": {k: {"total": v["total"], "correct": v["correct"], "pct": round(v["correct"] / v["total"] * 100, 1)} for k, v in tt_breakdown.items()},
            "avg_latency_s": round(avg_lat, 2)
        }

    m1 = calc_metrics(judges_arm1, preds_arm1)
    m4 = calc_metrics(judges_arm4, preds_arm4)
    m5 = calc_metrics(judges_arm5, preds_arm5)

    # Paired comparisons ARM_5 vs ARM_4
    paired_wins = 0
    paired_losses = 0
    paired_ties = 0
    contingency = [[0, 0], [0, 0]]  # [ARM4 correct, ARM4 wrong] x [ARM5 correct, ARM5 wrong]

    for qid in judges_arm5:
        c4 = judges_arm4[qid]["is_correct"] if qid in judges_arm4 else False
        c5 = judges_arm5[qid]["is_correct"]
        if c5 and not c4:
            paired_wins += 1
        elif not c5 and c4:
            paired_losses += 1
        else:
            paired_ties += 1

        r = 0 if c4 else 1
        c = 0 if c5 else 1
        contingency[r][c] += 1

    b = contingency[0][1]  # ARM4 correct, ARM5 wrong
    c_val = contingency[1][0]  # ARM4 wrong, ARM5 correct
    mcnemar_p = None
    if b + c_val > 0:
        res = stats.binomtest(min(b, c_val), b + c_val, 0.5, alternative="two-sided")
        mcnemar_p = round(res.pvalue, 4)

    # 6. PRINT RESULTS
    print("\n" + "=" * 80)
    print("CONTROLLED EVALUATION RESULTS (N=40 Fresh Held-Out Instances)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'ARM 1 (Baseline)':<18} | {'ARM 4 (Prior Best)':<18} | {'ARM 5 (Core Intel)':<18}")
    print("-" * 90)
    print(f"{'Overall Accuracy':<30} | {m1['overall_pct']}% ({m1['overall_correct']}/40)    | {m4['overall_pct']}% ({m4['overall_correct']}/40)    | {m5['overall_pct']}% ({m5['overall_correct']}/40)")
    print(f"{'  Wilson 95% CI':<30} | {m1['overall_ci']}      | {m4['overall_ci']}      | {m5['overall_ci']}")
    print(f"{'Answerable Accuracy':<30} | {m1['ans_pct']}% ({m1['ans_correct']}/38)    | {m4['ans_pct']}% ({m4['ans_correct']}/38)    | {m5['ans_pct']}% ({m5['ans_correct']}/38)")
    print(f"{'  Wilson 95% CI':<30} | {m1['ans_ci']}      | {m4['ans_ci']}      | {m5['ans_ci']}")
    print(f"{'Abstention Accuracy':<30} | {m1['unans_pct']}% ({m1['unans_correct']}/2)      | {m4['unans_pct']}% ({m4['unans_correct']}/2)      | {m5['unans_pct']}% ({m5['unans_correct']}/2)")
    print(f"{'Avg Latency (sec/query)':<30} | {m1['avg_latency_s']}s              | {m4['avg_latency_s']}s              | {m5['avg_latency_s']}s")
    print("-" * 90)

    print("\nTask-Type Breakdown (Accuracy % [Correct / Total]):")
    all_tts = sorted(list(set(m1["task_types"].keys()) | set(m5["task_types"].keys())))
    for tt in all_tts:
        t1 = m1["task_types"].get(tt, {"pct": 0.0, "correct": 0, "total": 0})
        t4 = m4["task_types"].get(tt, {"pct": 0.0, "correct": 0, "total": 0})
        t5 = m5["task_types"].get(tt, {"pct": 0.0, "correct": 0, "total": 0})
        print(f"  {tt:<28}: ARM1 = {t1['pct']:>5}% ({t1['correct']}/{t1['total']}) | ARM4 = {t4['pct']:>5}% ({t4['correct']}/{t4['total']}) | ARM5 = {t5['pct']:>5}% ({t5['correct']}/{t5['total']})")

    print("\nPaired Analysis (ARM_5 Core Intelligence vs ARM_4 Prior Candidate):")
    print(f"  Paired Wins (ARM_5 correct, ARM_4 wrong)  : {paired_wins}")
    print(f"  Paired Losses (ARM_5 wrong, ARM_4 correct): {paired_losses}")
    print(f"  Paired Ties                               : {paired_ties}")
    print(f"  McNemar Exact Test p-value                : {mcnemar_p}")
    print("=" * 80)

    # 7. WRITE SUMMARY JSON
    summary_path = output_dir / "core_intelligence_evaluation_summary.json"
    summary_payload = {
        "evaluation_tier": "fresh_heldout_40",
        "benchmark_status": "diagnostic_only_not_official_longmemeval",
        "sample_size": 40,
        "models": {
            "generator": generator_model,
            "judge": judge_model,
            "judge_status": judge_status_label
        },
        "runtimes_seconds": {
            "generation": gen_time_s,
            "judging": judge_time_s,
            "total": round(gen_time_s + judge_time_s, 2)
        },
        "arms": {
            "ARM_1_BASELINE": m1,
            "ARM_4_UNIFIED_HYBRID": m4,
            "ARM_5_CORE_INTELLIGENCE": m5
        },
        "paired_comparison_arm5_vs_arm4": {
            "paired_wins": paired_wins,
            "paired_losses": paired_losses,
            "paired_ties": paired_ties,
            "contingency_table": contingency,
            "mcnemar_p_value": mcnemar_p
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\nEvaluation summary written to: {summary_path}")


if __name__ == "__main__":
    main()
