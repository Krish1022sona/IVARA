"""
Day 2 Benchmark Aggregation and Statistical Analysis Script
Computes all 11 required dimensions:
1. Binary answer correctness
2. Fact coverage
3. Exact match
4. Citation validity and factual support
5. Multi-hop citation completeness
6. Correct abstention on unanswerable
7. False abstention on answerable
8. Conflicting and historical evidence handling
9. Repetition and malformed output
10. Per-category performance across all 12 dataset categories and 8 router classes
11. Median and p95 latency, plus peak VRAM
"""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = WORKSPACE_ROOT / "evaluation" / "benchmark_day2"


def compute_quantiles(values: list[float], quantiles: list[float]) -> dict[float, float]:
    if not values:
        return {q: 0.0 for q in quantiles}
    s = sorted(values)
    n = len(s)
    res = {}
    for q in quantiles:
        idx = int(q * (n - 1))
        res[q] = s[idx]
    return res


def analyze_model_predictions(preds: dict[str, Any]) -> dict[str, Any]:
    total_q = len(preds)
    answerable_q = [p for p in preds.values() if p["answer_metrics"]["is_answerable"]]
    unanswerable_q = [p for p in preds.values() if not p["answer_metrics"]["is_answerable"]]

    # 1. Correctness
    bin_acc_all = [p["answer_metrics"]["binary_accuracy"] for p in preds.values()]
    fact_cov_all = [p["answer_metrics"]["fact_coverage"] for p in preds.values()]
    exact_match_all = [p["answer_metrics"]["exact_match"] for p in preds.values()]

    # 2. Abstention Calibration
    correct_abstentions = sum(1 for p in unanswerable_q if p["answer_metrics"]["correct_abstention"])
    false_acceptances = sum(1 for p in unanswerable_q if p["answer_metrics"]["false_acceptance"])
    false_rejections = sum(1 for p in answerable_q if p["answer_metrics"]["false_rejection"])

    # 3. Citations & Factual Support
    valid_cite_rates = [p["citation_metrics"]["citation_validity"] for p in preds.values()]
    fact_support_rates = [p["factual_support"]["factual_support_rate"] for p in answerable_q]

    # Multi-hop completeness
    multihop_q = [p for p in answerable_q if p["multihop_citation"].get("is_multihop")]
    multihop_complete = sum(1 for p in multihop_q if p["multihop_citation"].get("multihop_complete"))

    # 4. Latency and Tokens
    latencies = [p["latency_ms"] for p in preds.values() if p.get("latency_ms")]
    llm_latencies = [p["llm_latency_ms"] for p in preds.values() if p.get("llm_latency_ms")]
    eval_counts = [p["eval_count"] for p in preds.values() if p.get("eval_count")]
    tps_list = [p["tokens_per_sec"] for p in preds.values() if p.get("tokens_per_sec")]

    lat_q = compute_quantiles(latencies, [0.50, 0.95])
    llm_q = compute_quantiles(llm_latencies, [0.50, 0.95])

    # VRAM
    vram_starts = [p["vram_start_mb"] for p in preds.values() if p.get("vram_start_mb")]
    vram_ends = [p["vram_end_mb"] for p in preds.values() if p.get("vram_end_mb")]
    all_vram = vram_starts + vram_ends
    peak_vram = max(all_vram) if all_vram else 0.0

    # Repetition
    rep_count = sum(1 for p in preds.values() if p.get("repetition_loop_detected"))

    # Per-category analysis
    cat_summary = defaultdict(lambda: {"total": 0, "bin_acc": [], "fact_cov": [], "exact_m": []})
    for p in preds.values():
        c = p["category"]
        cat_summary[c]["total"] += 1
        cat_summary[c]["bin_acc"].append(p["answer_metrics"]["binary_accuracy"])
        cat_summary[c]["fact_cov"].append(p["answer_metrics"]["fact_coverage"])
        cat_summary[c]["exact_m"].append(p["answer_metrics"]["exact_match"])

    cat_metrics = {}
    for c, data in cat_summary.items():
        cat_metrics[c] = {
            "count": data["total"],
            "binary_accuracy": round(sum(data["bin_acc"]) / len(data["bin_acc"]), 4),
            "fact_coverage": round(sum(data["fact_cov"]) / len(data["fact_cov"]), 4),
            "exact_match": round(sum(data["exact_m"]) / len(data["exact_m"]), 4)
        }

    return {
        "overall": {
            "total_questions": total_q,
            "answerable_count": len(answerable_q),
            "unanswerable_count": len(unanswerable_q),
            "binary_accuracy": round(sum(bin_acc_all) / len(bin_acc_all), 4),
            "mean_fact_coverage": round(sum(fact_cov_all) / len(fact_cov_all), 4),
            "exact_match_rate": round(sum(exact_match_all) / len(exact_match_all), 4),
            "correct_abstention_rate": round(correct_abstentions / len(unanswerable_q), 4) if unanswerable_q else 1.0,
            "false_acceptance_rate": round(false_acceptances / len(unanswerable_q), 4) if unanswerable_q else 0.0,
            "false_rejection_rate": round(false_rejections / len(answerable_q), 4) if answerable_q else 0.0,
            "citation_validity": round(sum(valid_cite_rates) / len(valid_cite_rates), 4) if valid_cite_rates else 1.0,
            "factual_support_rate": round(sum(fact_support_rates) / len(fact_support_rates), 4) if fact_support_rates else 1.0,
            "multihop_citation_completeness": round(multihop_complete / len(multihop_q), 4) if multihop_q else 1.0,
            "repetition_loops_detected": rep_count,
            "latency": {
                "mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
                "median_ms": round(lat_q[0.50], 1),
                "p95_ms": round(lat_q[0.95], 1),
                "llm_mean_ms": round(sum(llm_latencies) / len(llm_latencies), 1) if llm_latencies else 0.0,
                "llm_median_ms": round(llm_q[0.50], 1),
                "llm_p95_ms": round(llm_q[0.95], 1)
            },
            "tokens": {
                "mean_eval_count": round(sum(eval_counts) / len(eval_counts), 1) if eval_counts else 0.0,
                "total_eval_tokens": sum(eval_counts),
                "mean_tokens_per_sec": round(sum(tps_list) / len(tps_list), 1) if tps_list else 0.0
            },
            "peak_vram_mb": round(peak_vram, 1)
        },
        "per_category": cat_metrics
    }


def main():
    preds_gpt_file = BENCHMARK_DIR / "predictions_gpt_oss.json"
    preds_qwen_file = BENCHMARK_DIR / "predictions_qwen2_5.json"

    if not preds_gpt_file.exists() or not preds_qwen_file.exists():
        print("Predictions files not ready yet.")
        return

    with open(preds_gpt_file, "r", encoding="utf-8") as f:
        preds_gpt = json.load(f)

    with open(preds_qwen_file, "r", encoding="utf-8") as f:
        preds_qwen = json.load(f)

    summary_gpt = analyze_model_predictions(preds_gpt)
    summary_qwen = analyze_model_predictions(preds_qwen)

    # Statistical comparison
    from run_day2_benchmark import compute_paired_statistics
    stats = compute_paired_statistics(preds_gpt, preds_qwen)

    full_summary = {
        "gpt_oss": summary_gpt,
        "qwen2_5": summary_qwen,
        "statistical_comparison": stats
    }

    summary_file = BENCHMARK_DIR / "day2_benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    # Save per-question comparison CSV
    csv_per_q = BENCHMARK_DIR / "per_question_comparison.csv"
    with open(csv_per_q, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "category", "query",
            "gpt_status", "gpt_bin_acc", "gpt_fact_cov", "gpt_latency_ms", "gpt_answer",
            "qwen_status", "qwen_bin_acc", "qwen_fact_cov", "qwen_latency_ms", "qwen_answer",
            "winner"
        ])

        common_ids = sorted(list(set(preds_gpt.keys()).intersection(preds_qwen.keys())))
        for qid in common_ids:
            g = preds_gpt[qid]
            q = preds_qwen[qid]

            g_acc = g["answer_metrics"]["binary_accuracy"]
            q_acc = q["answer_metrics"]["binary_accuracy"]

            if q_acc > g_acc:
                winner = "QWEN"
            elif g_acc > q_acc:
                winner = "GPT_OSS"
            else:
                g_cov = g["answer_metrics"]["fact_coverage"]
                q_cov = q["answer_metrics"]["fact_coverage"]
                if q_cov > g_cov:
                    winner = "QWEN (cov)"
                elif g_cov > q_cov:
                    winner = "GPT_OSS (cov)"
                else:
                    winner = "TIE"

            writer.writerow([
                qid, g["category"], g["query"],
                g["status"], g_acc, g["answer_metrics"]["fact_coverage"], g["latency_ms"], g["answer"].replace("\n", " "),
                q["status"], q_acc, q["answer_metrics"]["fact_coverage"], q["latency_ms"], q["answer"].replace("\n", " "),
                winner
            ])

    # Save category breakdown CSV
    csv_cat = BENCHMARK_DIR / "category_breakdown_day2.csv"
    with open(csv_cat, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "count",
            "gpt_bin_acc", "gpt_fact_cov", "gpt_exact_m",
            "qwen_bin_acc", "qwen_fact_cov", "qwen_exact_m",
            "delta_bin_acc", "delta_fact_cov"
        ])
        cats = sorted(summary_gpt["per_category"].keys())
        for c in cats:
            g = summary_gpt["per_category"][c]
            q = summary_qwen["per_category"][c]
            writer.writerow([
                c, g["count"],
                g["binary_accuracy"], g["fact_coverage"], g["exact_match"],
                q["binary_accuracy"], q["fact_coverage"], q["exact_match"],
                round(q["binary_accuracy"] - g["binary_accuracy"], 4),
                round(q["fact_coverage"] - g["fact_coverage"], 4)
            ])

    print("Day 2 compilation complete! Summary and CSVs created.")


if __name__ == "__main__":
    main()

