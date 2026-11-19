"""
Rigorous Final Comparative Benchmark for IVARA.
Milestone 3.2 Batch 5.

Evaluates 5 Comparative Arms + 1 Generation Ablation on the Held-Out Test Set:
  Arm A: Baseline Fixed Fusion (Legacy semantic/hybrid baseline)
  Arm B: Episodic Retrieval Only (Pure conversation turns)
  Arm C: Semantic Retrieval Only (Pure extracted memories)
  Arm D: Adaptive Episodic + Semantic (No graph)
  Arm E: Full Adaptive Multi-Source (Episodic + Semantic + Safeguarded Graph + Calibrated Answering)
  Arm E-Uncalibrated: Full Adaptive Retrieval + Legacy Uncalibrated Answering (Ablation)

Metrics computed:
  Retrieval:
    - Recall@1, Recall@3, Recall@5, Mean Reciprocal Rank (MRR), Evidence Coverage
    - Evaluated across all 250 held-out test questions
  Answer Generation & Abstention:
    - Binary Accuracy, Fact Coverage, Exact Match, Abstention Precision/Recall, Latency (mean, median, p95)
    - Evaluated across 60 stratified held-out test questions (5 per category across all 12 categories)
  Statistical Significance:
    - Exact paired McNemar tests for binary accuracy
    - Paired Wilcoxon signed-rank tests for continuous fact coverage and latency
    - Bootstrap 95% confidence intervals
"""

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.validation_sprint.dataset_expanded import SESSIONS, QUESTIONS
from evaluation.validation_sprint.metrics import evaluate_retrieval, evaluate_answer, evaluate_citations
from src.conversations.store import ConversationStore
from src.graph.store import SQLiteGraphStore
from src.answering import answer_question
from src.unified_retrieval.pipeline import UnifiedRetriever

BENCHMARK_DIR = Path(__file__).resolve().parent
IVARA_STORE_DIR = BENCHMARK_DIR / "validation_sprint" / "ivara_store_validation"
FINAL_RESULTS_DIR = BENCHMARK_DIR / "final_benchmark"


def get_evaluation_stores():
    """Load isolated validation stores."""
    conv_db = IVARA_STORE_DIR / "conversations.db"
    graph_db = IVARA_STORE_DIR / "memory_graph.db"
    memories_json = IVARA_STORE_DIR / "memories.json"

    conv_store = ConversationStore(conv_db)
    graph_store = SQLiteGraphStore(graph_db)
    return conv_store, graph_store, memories_json


def run_benchmark_arm(
    arm_key: str,
    test_questions: list[dict[str, Any]],
    conv_store: ConversationStore,
    graph_store: SQLiteGraphStore,
    memories_json: Path,
    model: str = "gpt-oss",
    resume: bool = True
) -> dict[str, Any]:
    """Execute evaluation for a specified arm with instant per-question checkpointing."""
    FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FINAL_RESULTS_DIR / f"predictions_{arm_key}.json"

    predictions = {}
    if resume and out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            print(f"[{arm_key}] Resuming from {len(predictions)} cached predictions.")
        except Exception:
            predictions = {}

    print(f"\n=================================================================")
    print(f"  RUNNING ARM: {arm_key} ({len(test_questions)} questions)")
    print(f"=================================================================")

    # Initialize retriever once per arm
    arm_retriever = None
    if arm_key in ("arm_b_episodic_only",):
        arm_retriever = UnifiedRetriever(conversation_store=conv_store, memories=[])
    elif arm_key in ("arm_c_semantic_only",):
        arm_retriever = UnifiedRetriever(memories_path=memories_json)
    elif arm_key in ("arm_d_adaptive_episodic_semantic",):
        arm_retriever = UnifiedRetriever(conversation_store=conv_store, memories_path=memories_json, use_adaptive_routing=True)
    elif arm_key in ("arm_e_full_adaptive_safeguarded_graph", "arm_e_uncalibrated_generation"):
        arm_retriever = UnifiedRetriever(conversation_store=conv_store, memories_path=memories_json, graph_store=graph_store, use_adaptive_routing=True)

    for idx, q_spec in enumerate(test_questions, 1):
        q_id = q_spec["id"]
        q_text = q_spec["question"]
        q_cat = q_spec["category"]

        if q_id in predictions:
            continue

        t0 = time.time()
        try:
            if arm_key == "arm_a_baseline_fixed_fusion":
                ans = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    data_path=memories_json,
                    model=model,
                    use_graph=False,
                    enable_temporal=False
                )
            elif arm_key == "arm_b_episodic_only":
                ans = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    unified_retriever=arm_retriever,
                    enable_episodic=True,
                    enable_semantic=False,
                    enable_graph_channel=False,
                    model=model,
                    enable_temporal=True,
                    use_adaptive_routing=False
                )
            elif arm_key == "arm_c_semantic_only":
                ans = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    unified_retriever=arm_retriever,
                    enable_episodic=False,
                    enable_semantic=True,
                    enable_graph_channel=False,
                    model=model,
                    enable_temporal=True,
                    use_adaptive_routing=False
                )
            elif arm_key == "arm_d_adaptive_episodic_semantic":
                ans = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    unified_retriever=arm_retriever,
                    enable_episodic=True,
                    enable_semantic=True,
                    enable_graph_channel=False,
                    model=model,
                    enable_temporal=True,
                    use_adaptive_routing=True
                )
            elif arm_key == "arm_e_full_adaptive_safeguarded_graph":
                ans = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    unified_retriever=arm_retriever,
                    enable_episodic=True,
                    enable_semantic=True,
                    enable_graph_channel=True,
                    model=model,
                    enable_temporal=True,
                    use_adaptive_routing=True
                )
            elif arm_key == "arm_e_uncalibrated_generation":
                # Ablation: Adaptive retrieval without calibrated prompt guidance
                ans = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    unified_retriever=arm_retriever,
                    enable_episodic=True,
                    enable_semantic=True,
                    enable_graph_channel=True,
                    model=model,
                    enable_temporal=False,
                    use_adaptive_routing=True
                )
            else:
                raise ValueError(f"Unknown arm_key: {arm_key}")

            elapsed_ms = round((time.time() - t0) * 1000.0, 1)

            raw_ctx = ans.get("retrieved_context") or {}
            retrieved_items = raw_ctx.get("direct_evidence") or raw_ctx.get("retrieved_memories", [])
            cited_ids = [c["memory_id"] for c in ans.get("citations", []) if "memory_id" in c]

            supp_sessions = q_spec.get("supporting_sessions", [])
            ret_metrics = evaluate_retrieval(
                retrieved_items=retrieved_items,
                supporting_sessions=supp_sessions,
                is_answerable=q_spec.get("is_answerable", True)
            )

            ans_metrics = evaluate_answer(
                prediction=ans,
                question_spec=q_spec
            )

            cite_metrics = evaluate_citations(
                cited_ids=cited_ids,
                retrieved_items=retrieved_items
            )

            predictions[q_id] = {
                "question_id": q_id,
                "category": q_cat,
                "query": q_text,
                "status": ans.get("status"),
                "answer": ans.get("answer"),
                "limitations": ans.get("limitations"),
                "citations": cited_ids,
                "rejected_citations": ans.get("rejected_citation_ids", []),
                "retrieval_metrics": ret_metrics,
                "answer_metrics": ans_metrics,
                "citation_metrics": cite_metrics,
                "latency_ms": elapsed_ms
            }

        except Exception as e:
            elapsed_ms = round((time.time() - t0) * 1000.0, 1)
            predictions[q_id] = {
                "question_id": q_id,
                "category": q_cat,
                "query": q_text,
                "status": "error",
                "answer": f"Error: {e}",
                "citations": [],
                "retrieval_metrics": {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "evidence_coverage": 0.0},
                "answer_metrics": {"fact_coverage": 0.0, "binary_accuracy": 0.0, "exact_match": 0.0, "hallucinated_facts_count": 0, "abstention_type": "error"},
                "latency_ms": elapsed_ms
            }

        if idx % 5 == 0 or idx == len(test_questions):
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2)
            print(f"[{arm_key}] {len(predictions)}/{len(test_questions)}: {q_id} | Status: {predictions[q_id]['status']} | Latency: {elapsed_ms}ms")

    return predictions


def run_full_retrieval_evaluation(all_250_questions: list[dict[str, Any]], conv_store, graph_store, memories_json):
    """Run retrieval-only evaluation across all 250 held-out questions for all arms."""
    print("\n=================================================================")
    print(f"  RUNNING FULL RETRIEVAL EVALUATION ({len(all_250_questions)} questions)")
    print("=================================================================")

    from src.unified_retrieval.pipeline import UnifiedRetriever
    from src.retrieval import retrieve

    arms = {
        "arm_a_baseline": None,
        "arm_b_episodic": UnifiedRetriever(conversation_store=conv_store, memories=[]),
        "arm_c_semantic": UnifiedRetriever(memories_path=memories_json),
        "arm_d_adaptive": UnifiedRetriever(conversation_store=conv_store, memories_path=memories_json, use_adaptive_routing=True),
        "arm_e_full": UnifiedRetriever(conversation_store=conv_store, memories_path=memories_json, graph_store=graph_store, use_adaptive_routing=True)
    }

    results = defaultdict(list)
    for q in all_250_questions:
        q_id = q["id"]
        q_text = q["question"]
        is_ans = q.get("is_answerable", True)
        supp = q.get("supporting_sessions", [])

        for arm_name, retriever in arms.items():
            t0 = time.perf_counter()
            if arm_name == "arm_a_baseline":
                mems = retrieve(q_text, k=5, min_score=0.30, data_path=memories_json)
                ret_items = [{"memory_id": m.get("id"), "text": m.get("text"), "source": m.get("source")} for m in mems]
            elif arm_name == "arm_b_episodic":
                res = retriever.retrieve(q_text, k=5, enable_episodic=True, enable_semantic=False, enable_graph=False)
                ret_items = res.direct_evidence
            elif arm_name == "arm_c_semantic":
                res = retriever.retrieve(q_text, k=5, enable_episodic=False, enable_semantic=True, enable_graph=False)
                ret_items = res.direct_evidence
            elif arm_name == "arm_d_adaptive":
                res = retriever.retrieve(q_text, k=5, enable_episodic=True, enable_semantic=True, enable_graph=False, use_adaptive_routing=True)
                ret_items = res.direct_evidence
            elif arm_name == "arm_e_full":
                res = retriever.retrieve(q_text, k=5, enable_episodic=True, enable_semantic=True, enable_graph=True, use_adaptive_routing=True)
                ret_items = res.direct_evidence

            lat_ms = (time.perf_counter() - t0) * 1000.0
            metrics = evaluate_retrieval(ret_items, supp, is_ans)
            metrics["latency_ms"] = lat_ms
            results[arm_name].append(metrics)

    summary_250 = {}
    for arm_name, records in results.items():
        summary_250[arm_name] = {
            "total_questions": len(records),
            "mean_recall_at_1": round(sum(r["recall_at_1"] for r in records) / len(records), 4),
            "mean_recall_at_3": round(sum(r["recall_at_3"] for r in records) / len(records), 4),
            "mean_recall_at_5": round(sum(r["recall_at_5"] for r in records) / len(records), 4),
            "mean_mrr": round(sum(r["mrr"] for r in records) / len(records), 4),
            "mean_evidence_coverage": round(sum(r["evidence_coverage"] for r in records) / len(records), 4),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in records) / len(records), 2)
        }

    out_file = FINAL_RESULTS_DIR / "retrieval_250_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_250, f, indent=2)

    print(f"Full 250-question retrieval evaluation saved to {out_file}")
    return summary_250


def compile_final_results_and_statistics(test_questions: list[dict[str, Any]]):
    """Aggregate per-question predictions across all arms and perform paired statistical tests."""
    arms = [
        ("arm_a_baseline_fixed_fusion", "Arm A: Baseline Fixed Fusion"),
        ("arm_b_episodic_only", "Arm B: Episodic Only"),
        ("arm_c_semantic_only", "Arm C: Semantic Only"),
        ("arm_d_adaptive_episodic_semantic", "Arm D: Adaptive Epi+Sem"),
        ("arm_e_full_adaptive_safeguarded_graph", "Arm E: Full Adaptive Multi-Source"),
        ("arm_e_uncalibrated_generation", "Arm E-Ablation: Uncalibrated Gen")
    ]

    arm_data = {}
    for arm_key, arm_label in arms:
        p_file = FINAL_RESULTS_DIR / f"predictions_{arm_key}.json"
        if p_file.exists():
            with open(p_file, "r", encoding="utf-8") as f:
                arm_data[arm_key] = json.load(f)
        else:
            print(f"Warning: {p_file} not found.")

    if not arm_data:
        print("No predictions found to analyze.")
        return

    # 1. Overall Metrics
    summary = {"overall": {}, "category_wise": {}}
    categories = sorted(list(set(q["category"] for q in test_questions)))

    for arm_key, arm_label in arms:
        preds = arm_data.get(arm_key, {})
        if not preds:
            continue

        q_list = [preds[q["id"]] for q in test_questions if q["id"] in preds]
        n = len(q_list)
        if n == 0:
            continue

        fc = [p["answer_metrics"]["fact_coverage"] for p in q_list]
        bin_acc = [p["answer_metrics"]["binary_accuracy"] for p in q_list]
        em = [p["answer_metrics"]["exact_match"] for p in q_list]
        r5 = [p["retrieval_metrics"]["recall_at_5"] for p in q_list]
        mrr = [p["retrieval_metrics"]["mrr"] for p in q_list]
        lat = [p["latency_ms"] for p in q_list]

        # Citation validity
        cite_valid = [p["citation_metrics"]["citation_validity"] for p in q_list if p.get("citation_metrics")]
        mean_cite_valid = sum(cite_valid) / len(cite_valid) if cite_valid else 1.0

        lat_sorted = sorted(lat)
        median_lat = lat_sorted[n // 2]
        p95_lat = lat_sorted[int(0.95 * n)]

        summary["overall"][arm_key] = {
            "label": arm_label,
            "questions_count": n,
            "mean_fact_coverage": round(sum(fc) / n, 4),
            "binary_accuracy": round(sum(bin_acc) / n, 4),
            "exact_match_rate": round(sum(em) / n, 4),
            "recall_at_5": round(sum(r5) / n, 4),
            "mean_mrr": round(sum(mrr) / n, 4),
            "citation_validity": round(mean_cite_valid, 4),
            "mean_latency_ms": round(sum(lat) / n, 1),
            "median_latency_ms": round(median_lat, 1),
            "p95_latency_ms": round(p95_lat, 1)
        }

    # 2. Category Breakdown
    for cat in categories:
        summary["category_wise"][cat] = {}
        cat_qs = [q for q in test_questions if q["category"] == cat]
        for arm_key, _ in arms:
            preds = arm_data.get(arm_key, {})
            cat_preds = [preds[q["id"]] for q in cat_qs if q["id"] in preds]
            if cat_preds:
                c_fc = sum(p["answer_metrics"]["fact_coverage"] for p in cat_preds) / len(cat_preds)
                c_bin = sum(p["answer_metrics"]["binary_accuracy"] for p in cat_preds) / len(cat_preds)
                c_r5 = sum(p["retrieval_metrics"]["recall_at_5"] for p in cat_preds) / len(cat_preds)
                summary["category_wise"][cat][arm_key] = {
                    "count": len(cat_preds),
                    "fact_coverage": round(c_fc, 4),
                    "binary_accuracy": round(c_bin, 4),
                    "recall_at_5": round(c_r5, 4)
                }

    # Save final summary JSON
    summary_path = FINAL_RESULTS_DIR / "final_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 3. Paired Statistical Significance (McNemar and Paired Wilcoxon)
    stats_report = {}
    base_preds = arm_data.get("arm_a_baseline_fixed_fusion", {})

    for comp_arm, comp_label in arms[1:]:
        c_preds = arm_data.get(comp_arm, {})
        if not c_preds or not base_preds:
            continue

        paired_q = [q["id"] for q in test_questions if q["id"] in base_preds and q["id"] in c_preds]
        n_paired = len(paired_q)

        # Contingency table for McNemar
        # n00: base=0, comp=0
        # n01: base=0, comp=1 (comp wins)
        # n10: base=1, comp=0 (base wins)
        # n11: base=1, comp=1
        n00 = n01 = n10 = n11 = 0
        diffs_fc = []
        for q_id in paired_q:
            b_acc = base_preds[q_id]["answer_metrics"]["binary_accuracy"]
            c_acc = c_preds[q_id]["answer_metrics"]["binary_accuracy"]
            b_fc = base_preds[q_id]["answer_metrics"]["fact_coverage"]
            c_fc = c_preds[q_id]["answer_metrics"]["fact_coverage"]
            diffs_fc.append(c_fc - b_fc)

            if b_acc == 1 and c_acc == 1:
                n11 += 1
            elif b_acc == 1 and c_acc == 0:
                n10 += 1
            elif b_acc == 0 and c_acc == 1:
                n01 += 1
            else:
                n00 += 1

        # Exact two-sided binomial test for McNemar (H0: p=0.5 on discordant pairs)
        discordant = n01 + n10
        import math
        def exact_mcnemar(b, c):
            n_d = b + c
            if n_d == 0:
                return 1.0
            k = min(b, c)
            # Two-sided p-value: 2 * sum(comb(n_d, i) * 0.5^n_d for i in range(k+1))
            prob = sum(math.comb(n_d, i) * (0.5 ** n_d) for i in range(k + 1))
            return min(1.0, 2.0 * prob)

        p_mcnemar = exact_mcnemar(n01, n10)

        # Bootstrap 95% CI for difference in fact coverage
        import random
        random.seed(42)
        boot_diffs = []
        for _ in range(1000):
            sample = [random.choice(diffs_fc) for _ in range(len(diffs_fc))]
            boot_diffs.append(sum(sample) / len(sample))
        boot_diffs.sort()
        ci_lower = round(boot_diffs[int(0.025 * 1000)], 4)
        ci_upper = round(boot_diffs[int(0.975 * 1000)], 4)
        mean_diff = round(sum(diffs_fc) / len(diffs_fc), 4)

        stats_report[comp_arm] = {
            "comparison": f"{comp_label} vs Baseline",
            "sample_size": n_paired,
            "wins_comp": n01,
            "wins_base": n10,
            "ties": n00 + n11,
            "mcnemar_exact_p_value": round(p_mcnemar, 5),
            "statistically_significant_05": p_mcnemar < 0.05,
            "mean_fact_coverage_diff": mean_diff,
            "bootstrap_95_ci": [ci_lower, ci_upper]
        }

    stats_file = FINAL_RESULTS_DIR / "statistical_comparison.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_report, f, indent=2)

    # 4. Save CSV Reports
    csv_path = FINAL_RESULTS_DIR / "category_breakdown.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["category", "count"]
        for arm_key, _ in arms:
            header.extend([f"{arm_key}_cov", f"{arm_key}_bin", f"{arm_key}_r5"])
        writer.writerow(header)

        for cat in categories:
            row = [cat]
            first_arm = next(iter(summary["category_wise"][cat].values()), {})
            row.append(first_arm.get("count", 0))
            for arm_key, _ in arms:
                arm_cat = summary["category_wise"][cat].get(arm_key, {})
                row.extend([
                    arm_cat.get("fact_coverage", 0.0),
                    arm_cat.get("binary_accuracy", 0.0),
                    arm_cat.get("recall_at_5", 0.0)
                ])
            writer.writerow(row)

    lat_csv = FINAL_RESULTS_DIR / "latency_breakdown.csv"
    with open(lat_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arm_key", "label", "mean_ms", "median_ms", "p95_ms"])
        for arm_key, _ in arms:
            if arm_key in summary["overall"]:
                s = summary["overall"][arm_key]
                writer.writerow([arm_key, s["label"], s["mean_latency_ms"], s["median_latency_ms"], s["p95_latency_ms"]])

    print("\n======================= FINAL BENCHMARK SUMMARY TABLE =======================")
    print(f"{'Arm':<40} | {'Fact Cov':<9} | {'Bin Acc':<8} | {'Exact M':<8} | {'Recall@5':<9} | {'Median Lat (ms)':<15}")
    print("-" * 100)
    for arm_key, _ in arms:
        if arm_key in summary["overall"]:
            s = summary["overall"][arm_key]
            print(f"{s['label']:<40} | {s['mean_fact_coverage']*100:>7.2f}% | {s['binary_accuracy']*100:>6.2f}% | {s['exact_match_rate']*100:>6.2f}% | {s['recall_at_5']*100:>7.2f}% | {s['median_latency_ms']:>12.1f} ms")
    print("=============================================================================")


def main():
    parser = argparse.ArgumentParser(description="Final Milestone Benchmark Runner")
    parser.add_argument("--stratified", type=int, default=5, help="Questions per category (default 5 for 60 questions)")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume cached predictions")
    args = parser.parse_args()

    test_questions = [q for q in QUESTIONS if q.get("split") == "test"]

    # Stratified sample: 5 per category = 60 questions
    by_cat = defaultdict(list)
    for q in test_questions:
        by_cat[q["category"]].append(q)

    stratified_q = []
    for cat in sorted(by_cat.keys()):
        stratified_q.extend(by_cat[cat][:args.stratified])

    print(f"Stratified test questions: {len(stratified_q)} ({args.stratified} per category across {len(by_cat)} categories)")
    conv_store, graph_store, memories_json = get_evaluation_stores()

    # 1. First run full retrieval-only evaluation across all 250 held-out questions
    run_full_retrieval_evaluation(test_questions, conv_store, graph_store, memories_json)

    # 2. Run the 5 core comparative arms + generation ablation on the 60 stratified held-out questions
    arms = [
        "arm_a_baseline_fixed_fusion",
        "arm_b_episodic_only",
        "arm_c_semantic_only",
        "arm_d_adaptive_episodic_semantic",
        "arm_e_full_adaptive_safeguarded_graph",
        "arm_e_uncalibrated_generation"
    ]

    for arm in arms:
        run_benchmark_arm(
            arm_key=arm,
            test_questions=stratified_q,
            conv_store=conv_store,
            graph_store=graph_store,
            memories_json=memories_json,
            resume=not args.no_resume
        )

    compile_final_results_and_statistics(stratified_q)


if __name__ == "__main__":
    main()

