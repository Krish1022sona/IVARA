"""Controlled Comparative Benchmark Runner for IVARA Milestone 3.

Evaluates 5 comparative arms on the held-out narrative test questions (250 questions):
  Arm A: Current semantic/hybrid IVARA (legacy baseline)
  Arm B: Episodic retrieval only (pure conversation turns)
  Arm C: Semantic retrieval only (pure extracted memories)
  Arm D: Episodic + Semantic retrieval (fused conversation turns + extracted memories)
  Arm E: Unified Tri-Source retrieval (Episodic + Semantic + Knowledge Graph)

Calculates:
  - Fact Coverage
  - Binary Accuracy
  - Exact Match Rate
  - Recall@5
  - Historical-State Retrieval Accuracy
  - Multi-Hop Conversational Accuracy
  - Relationship Retrieval Accuracy
  - Abstention Accuracy
  - Latency (mean, median, p95 in ms)
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.validation_sprint.dataset_expanded import SESSIONS, QUESTIONS
from evaluation.validation_sprint.metrics import evaluate_retrieval, evaluate_answer
from src.conversations.store import ConversationStore
from src.graph.store import SQLiteGraphStore
from src.answering import answer_question
from src.unified_retrieval.fusion import RRFConfig

BENCHMARK_DIR = Path(__file__).resolve().parent
IVARA_STORE_DIR = BENCHMARK_DIR / "validation_sprint" / "ivara_store_validation"
M3_RESULTS_DIR = BENCHMARK_DIR / "milestone_3_results"


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
    M3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = M3_RESULTS_DIR / f"predictions_{arm_key}.json"

    predictions = {}
    if resume and out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            print(f"[{arm_key}] Resuming from {len(predictions)} cached predictions.")
        except Exception:
            predictions = {}

    print(f"\n=======================================================")
    print(f"  RUNNING ARM: {arm_key} ({len(test_questions)} questions)")
    print(f"=======================================================")

    # Initialize unified retriever once per arm for maximum performance
    from src.unified_retrieval.pipeline import UnifiedRetriever
    arm_retriever = None
    if arm_key == "arm_b_episodic_only":
        arm_retriever = UnifiedRetriever(conversation_store=conv_store, memories=[])
    elif arm_key == "arm_c_semantic_only":
        arm_retriever = UnifiedRetriever(memories_path=memories_json)
    elif arm_key == "arm_d_episodic_semantic":
        arm_retriever = UnifiedRetriever(conversation_store=conv_store, memories_path=memories_json)
    elif arm_key == "arm_e_unified_tri_source":
        arm_retriever = UnifiedRetriever(conversation_store=conv_store, memories_path=memories_json, graph_store=graph_store)

    for idx, q_spec in enumerate(test_questions, 1):
        q_id = q_spec["id"]
        q_text = q_spec["question"]
        q_cat = q_spec["category"]

        if q_id in predictions:
            continue

        t0 = time.time()
        try:
            if arm_key == "arm_a_semantic_hybrid":
                # Legacy IVARA semantic/hybrid baseline
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
                # Arm B: Episodic retrieval only
                ans = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    unified_retriever=arm_retriever,
                    enable_episodic=True,
                    enable_semantic=False,
                    enable_graph_channel=False,
                    model=model,
                    enable_temporal=True
                )
            elif arm_key == "arm_c_semantic_only":
                # Arm C: Semantic retrieval only via unified pipeline
                ans = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    unified_retriever=arm_retriever,
                    enable_episodic=False,
                    enable_semantic=True,
                    enable_graph_channel=False,
                    model=model,
                    enable_temporal=True
                )
            elif arm_key == "arm_d_episodic_semantic":
                # Arm D: Episodic + Semantic retrieval
                ans = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    unified_retriever=arm_retriever,
                    enable_episodic=True,
                    enable_semantic=True,
                    enable_graph_channel=False,
                    model=model,
                    enable_temporal=True
                )
            elif arm_key == "arm_e_unified_tri_source":
                # Arm E: Unified Tri-Source (Episodic + Semantic + Graph)
                ans = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    unified_retriever=arm_retriever,
                    enable_episodic=True,
                    enable_semantic=True,
                    enable_graph_channel=True,
                    model=model,
                    enable_temporal=True
                )
            else:
                raise ValueError(f"Unknown arm_key: {arm_key}")

            elapsed_ms = round((time.time() - t0) * 1000.0, 1)

            raw_ctx = ans.get("retrieved_context") or {}
            retrieved_items = raw_ctx.get("direct_evidence") or raw_ctx.get("retrieved_memories", [])
            cited_ids = [c["memory_id"] for c in ans.get("citations", []) if "memory_id" in c]

            # Compute retrieval metrics
            supp_sessions = q_spec.get("supporting_sessions", [])
            ret_metrics = evaluate_retrieval(
                retrieved_items=retrieved_items,
                supporting_sessions=supp_sessions,
                is_answerable=q_spec.get("is_answerable", True)
            )

            # Compute answer metrics
            ans_pred = {
                "status": ans.get("status", "answered"),
                "answer": ans.get("answer", ""),
                "citations": cited_ids
            }
            ans_metrics = evaluate_answer(ans_pred, q_spec)

            predictions[q_id] = {
                "question_id": q_id,
                "category": q_cat,
                "query": q_text,
                "status": ans.get("status", "answered"),
                "answer": ans.get("answer", ""),
                "citations": cited_ids,
                "retrieval_metrics": ret_metrics,
                "answer_metrics": ans_metrics,
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

        if idx % 10 == 0 or idx == len(test_questions):
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2)
            print(f"[{arm_key}] Checkpointed {len(predictions)}/{len(test_questions)} questions ({elapsed_ms}ms last)")

    return predictions


def compile_summary_and_reports(test_questions: list[dict[str, Any]]):
    """Aggregate per-question predictions across all 5 arms and compile deliverables."""
    arms = [
        ("arm_a_semantic_hybrid", "A: Current Semantic/Hybrid IVARA"),
        ("arm_b_episodic_only", "B: Episodic Retrieval Only"),
        ("arm_c_semantic_only", "C: Semantic Retrieval Only"),
        ("arm_d_episodic_semantic", "D: Episodic + Semantic Retrieval"),
        ("arm_e_unified_tri_source", "E: Unified Tri-Source (Epi+Sem+Graph)")
    ]

    arm_data: dict[str, dict[str, Any]] = {}
    for arm_key, arm_label in arms:
        pred_file = M3_RESULTS_DIR / f"predictions_{arm_key}.json"
        if pred_file.exists():
            with open(pred_file, "r", encoding="utf-8") as f:
                arm_data[arm_key] = json.load(f)
        else:
            print(f"Warning: {pred_file} not found.")

    if not arm_data:
        print("No arm data found to compile.")
        return

    categories = sorted(list(set(q["category"] for q in test_questions)))
    n_total = len(test_questions)
    target_qids = set(q["id"] for q in test_questions)

    summary = {
        "overall": {},
        "category_wise": {},
        "latency_metrics": {}
    }

    # 1. Overall Metrics (filtered strictly to evaluated target_qids)
    for arm_key, arm_label in arms:
        if arm_key not in arm_data:
            continue
        preds = [arm_data[arm_key][qid] for qid in target_qids if qid in arm_data[arm_key]]
        n = len(preds)
        if n == 0:
            continue

        covs = [p["answer_metrics"]["fact_coverage"] for p in preds]
        bin_accs = [p["answer_metrics"]["binary_accuracy"] for p in preds]
        ems = [p["answer_metrics"]["exact_match"] for p in preds]
        r5s = [p["retrieval_metrics"]["recall_at_5"] for p in preds]
        mrrs = [p["retrieval_metrics"]["mrr"] for p in preds]
        lats = [p["latency_ms"] for p in preds]
        sorted_lats = sorted(lats)
        p95_idx = min(int(n * 0.95), n - 1)

        summary["overall"][arm_key] = {
            "label": arm_label,
            "questions_count": n,
            "mean_fact_coverage": round(sum(covs) / n, 4),
            "binary_accuracy": round(sum(bin_accs) / n, 4),
            "exact_match_rate": round(sum(ems) / n, 4),
            "recall_at_5": round(sum(r5s) / n, 4),
            "mean_mrr": round(sum(mrrs) / n, 4),
            "mean_latency_ms": round(sum(lats) / n, 1),
            "median_latency_ms": round(sorted_lats[n // 2], 1),
            "p95_latency_ms": round(sorted_lats[p95_idx], 1)
        }

    # 2. Category-wise Breakdown (filtered strictly to target_qids)
    for cat in categories:
        summary["category_wise"][cat] = {}
        for arm_key, _ in arms:
            if arm_key not in arm_data:
                continue
            cat_preds = [
                arm_data[arm_key][qid]
                for qid in target_qids
                if qid in arm_data[arm_key] and arm_data[arm_key][qid]["category"] == cat
            ]
            if not cat_preds:
                continue
            n_cat = len(cat_preds)
            covs = [p["answer_metrics"]["fact_coverage"] for p in cat_preds]
            bin_accs = [p["answer_metrics"]["binary_accuracy"] for p in cat_preds]
            r5s = [p["retrieval_metrics"]["recall_at_5"] for p in cat_preds]

            summary["category_wise"][cat][arm_key] = {
                "count": n_cat,
                "fact_coverage": round(sum(covs) / n_cat, 4),
                "binary_accuracy": round(sum(bin_accs) / n_cat, 4),
                "recall_at_5": round(sum(r5s) / n_cat, 4)
            }

    # Save JSON summary
    summary_path = M3_RESULTS_DIR / "milestone_3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # Save Category Breakdown CSV
    csv_path = M3_RESULTS_DIR / "category_breakdown.csv"
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
    print(f"Saved: {csv_path}")

    # Save Latency Breakdown CSV
    lat_csv = M3_RESULTS_DIR / "latency_breakdown.csv"
    with open(lat_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["arm_key", "label", "mean_ms", "median_ms", "p95_ms"])
        for arm_key, _ in arms:
            if arm_key in summary["overall"]:
                s = summary["overall"][arm_key]
                writer.writerow([arm_key, s["label"], s["mean_latency_ms"], s["median_latency_ms"], s["p95_latency_ms"]])
    print(f"Saved: {lat_csv}")

    # Print summary table
    print("\n================== MILESTONE 3 COMPARATIVE BENCHMARK SUMMARY ==================")
    print(f"{'Arm':<38} | {'Fact Cov':<9} | {'Bin Acc':<8} | {'Exact M':<8} | {'Recall@5':<9} | {'Mean Lat (ms)':<13}")
    print("-" * 95)
    for arm_key, arm_label in arms:
        if arm_key in summary["overall"]:
            s = summary["overall"][arm_key]
            print(f"{s['label']:<38} | {s['mean_fact_coverage']*100:>7.2f}% | {s['binary_accuracy']*100:>6.2f}% | {s['exact_match_rate']*100:>6.2f}% | {s['recall_at_5']*100:>7.2f}% | {s['mean_latency_ms']:>10.1f} ms")
    print("===============================================================================")


def main():
    parser = argparse.ArgumentParser(description="Milestone 3 Benchmark Runner")
    parser.add_argument("--arms", nargs="+", default=["arm_a_semantic_hybrid", "arm_b_episodic_only", "arm_c_semantic_only", "arm_d_episodic_semantic", "arm_e_unified_tri_source"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test questions (for quick validation)")
    parser.add_argument("--stratified", type=int, default=None, help="Sample N questions per category across all categories")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume cached predictions")
    args = parser.parse_args()

    test_questions = [q for q in QUESTIONS if q.get("split") == "test"]
    if args.stratified:
        from collections import defaultdict
        by_cat = defaultdict(list)
        for q in test_questions:
            by_cat[q["category"]].append(q)
        stratified_q = []
        for cat in sorted(by_cat.keys()):
            stratified_q.extend(by_cat[cat][:args.stratified])
        test_questions = stratified_q
    elif args.limit:
        test_questions = test_questions[:args.limit]
    print(f"Total test questions to evaluate: {len(test_questions)}")

    conv_store, graph_store, memories_json = get_evaluation_stores()

    for arm in args.arms:
        run_benchmark_arm(
            arm_key=arm,
            test_questions=test_questions,
            conv_store=conv_store,
            graph_store=graph_store,
            memories_json=memories_json,
            resume=not args.no_resume
        )

    compile_summary_and_reports(test_questions)


if __name__ == "__main__":
    main()
