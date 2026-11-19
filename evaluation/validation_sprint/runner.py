"""Comprehensive Benchmark Runner and Evaluator for Validation Sprint.

Executes and compares 5 experimental configurations:
  Arm A: 'baseline_vector' (Conventional Dense Vector RAG)
  Arm B: 'baseline_hybrid' (Conventional Dense + BM25 Hybrid RAG)
  Arm C: 'baseline_hybrid_speaker_aware' (Fair Hybrid RAG with Speaker Metadata)
  Arm D: 'ivara_hybrid_no_graph' (IVARA Extracted Memories + Ontology, No Graph)
  Arm E: 'ivara_full' (Full IVARA: Memories + Ontology + SQLite Graph + Temporal Claims)

Plus Phase 7 Ablation configurations:
  Abl 1: 'ivara_no_ontology_no_graph'
  Abl 2: 'ivara_ontology_no_graph' (= Arm D)
  Abl 3: 'ivara_graph_no_ontology'
  Abl 4: 'ivara_full_no_temporal'
  Abl 5: 'ivara_full' (= Arm E)
"""

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Optional

from .dataset_expanded import SESSIONS, QUESTIONS
from .baseline_rag_fair import FairConventionalRAGBaseline
from .metrics import evaluate_retrieval, evaluate_answer, evaluate_citations
from .statistical_analysis import (
    wilson_score_interval,
    bootstrap_mean_diff_ci,
    mcnemar_test,
    holm_bonferroni_correction,
    analyze_abstention_and_calibration
)

from src.conversations.store import ConversationStore
from src.graph.store import SQLiteGraphStore
from src.answering import answer_question
from src.memory import sync_memory_graph


BENCHMARK_DIR = Path(__file__).parent
IVARA_STORE_DIR = BENCHMARK_DIR / "ivara_store_validation"
RESULTS_DIR = BENCHMARK_DIR / "results"


def setup_ivara_stores() -> tuple[ConversationStore, SQLiteGraphStore, Path, Path]:
    """Return handles to isolated evaluation stores."""
    IVARA_STORE_DIR.mkdir(parents=True, exist_ok=True)
    conv_db = IVARA_STORE_DIR / "conversations.db"
    graph_db = IVARA_STORE_DIR / "memory_graph.db"
    memories_json = IVARA_STORE_DIR / "memories.json"
    entities_json = IVARA_STORE_DIR / "entities.json"

    conv_store = ConversationStore(conv_db)
    graph_store = SQLiteGraphStore(graph_db)
    return conv_store, graph_store, memories_json, entities_json


def run_evaluation_arm(
    arm_name: str,
    questions: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    model: str = "gpt-oss",
    resume: bool = True
) -> dict[str, Any]:
    """Execute evaluation for a given system arm with instant per-question checkpointing."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"predictions_{arm_name}.json"

    predictions = {}
    if resume and out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            print(f"[{arm_name}] Resuming from {len(predictions)} cached predictions.")
        except Exception:
            predictions = {}

    # Initialize backend
    baseline_rag = None
    conv_store = None
    graph_store = None
    memories_json = None
    entities_json = None

    if arm_name in ("baseline_vector", "baseline_hybrid", "baseline_hybrid_speaker_aware"):
        baseline_rag = FairConventionalRAGBaseline(sessions, model=model)
    else:
        conv_store, graph_store, memories_json, entities_json = setup_ivara_stores()

    print(f"\n--- Running Evaluation: {arm_name} ({len(questions)} questions) ---")

    for idx, q_spec in enumerate(questions, 1):
        q_id = q_spec["id"]
        q_text = q_spec["question"]
        q_cat = q_spec["category"]

        if q_id in predictions:
            continue

        print(f"[{idx}/{len(questions)}] ({arm_name} - {q_cat}) {q_text[:60]}...")
        t_start = time.time()

        pred_result = {}
        retrieved_items = []
        cited_ids = []

        try:
            if arm_name == "baseline_vector":
                rag_out = baseline_rag.answer_question(q_text, k=5, mode="vector")
                retrieved_items = rag_out.get("retrieved_chunks", [])
                cited_ids = rag_out.get("cited_chunk_ids", [])
                pred_result = {
                    "status": rag_out.get("status", "answered"),
                    "answer": rag_out.get("answer", ""),
                    "citations": cited_ids,
                    "retrieved_context": retrieved_items
                }
            elif arm_name == "baseline_hybrid":
                rag_out = baseline_rag.answer_question(q_text, k=5, mode="hybrid")
                retrieved_items = rag_out.get("retrieved_chunks", [])
                cited_ids = rag_out.get("cited_chunk_ids", [])
                pred_result = {
                    "status": rag_out.get("status", "answered"),
                    "answer": rag_out.get("answer", ""),
                    "citations": cited_ids,
                    "retrieved_context": retrieved_items
                }
            elif arm_name == "baseline_hybrid_speaker_aware":
                rag_out = baseline_rag.answer_question(q_text, k=5, mode="hybrid_speaker")
                retrieved_items = rag_out.get("retrieved_chunks", [])
                cited_ids = rag_out.get("cited_chunk_ids", [])
                pred_result = {
                    "status": rag_out.get("status", "answered"),
                    "answer": rag_out.get("answer", ""),
                    "citations": cited_ids,
                    "retrieved_context": retrieved_items
                }
            elif arm_name in ("ivara_hybrid_no_graph", "ivara_ontology_no_graph"):
                iv_out = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    data_path=memories_json,
                    model=model,
                    use_graph=False,
                    enable_temporal=False
                )
                raw_ctx = iv_out.get("retrieved_context") or {}
                retrieved_items = raw_ctx.get("retrieved_memories", [])
                cited_ids = [c["memory_id"] for c in iv_out.get("citations", []) if "memory_id" in c]
                pred_result = {
                    "status": iv_out.get("status", "answered"),
                    "answer": iv_out.get("answer", ""),
                    "citations": cited_ids,
                    "retrieved_context": retrieved_items
                }
            elif arm_name == "ivara_no_ontology_no_graph":
                # Ablation: no ontology entities/relationships, no graph, no temporal
                # Strip structured ontology facts so only raw memory text is available
                with open(memories_json, "r", encoding="utf-8") as mf:
                    raw_mems = json.load(mf)
                stripped_mems = [
                    {k: v for k, v in m.items() if k not in ("entities", "relationships")}
                    for m in raw_mems
                ]
                iv_out = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    memories=stripped_mems,
                    model=model,
                    use_graph=False,
                    enable_temporal=False
                )
                raw_ctx = iv_out.get("retrieved_context") or {}
                retrieved_items = raw_ctx.get("retrieved_memories", [])
                cited_ids = [c["memory_id"] for c in iv_out.get("citations", []) if "memory_id" in c]
                pred_result = {
                    "status": iv_out.get("status", "answered"),
                    "answer": iv_out.get("answer", ""),
                    "citations": cited_ids,
                    "retrieved_context": retrieved_items
                }
            elif arm_name == "ivara_graph_no_ontology":
                # Ablation: graph enabled, temporal enabled
                iv_out = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    data_path=memories_json,
                    model=model,
                    use_graph=True,
                    graph_store=graph_store,
                    enable_temporal=True
                )
                raw_ctx = iv_out.get("retrieved_context") or {}
                retrieved_items = raw_ctx.get("retrieved_memories") or raw_ctx.get("selected_chunks", [])
                cited_ids = [c["memory_id"] for c in iv_out.get("citations", []) if "memory_id" in c]
                pred_result = {
                    "status": iv_out.get("status", "answered"),
                    "answer": iv_out.get("answer", ""),
                    "citations": cited_ids,
                    "retrieved_context": retrieved_items
                }
            elif arm_name == "ivara_full_no_temporal":
                # Ablation: graph enabled, temporal disabled
                iv_out = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    data_path=memories_json,
                    model=model,
                    use_graph=True,
                    graph_store=graph_store,
                    enable_temporal=False
                )
                raw_ctx = iv_out.get("retrieved_context") or {}
                retrieved_items = raw_ctx.get("retrieved_memories") or raw_ctx.get("selected_chunks", [])
                cited_ids = [c["memory_id"] for c in iv_out.get("citations", []) if "memory_id" in c]
                pred_result = {
                    "status": iv_out.get("status", "answered"),
                    "answer": iv_out.get("answer", ""),
                    "citations": cited_ids,
                    "retrieved_context": retrieved_items
                }
            elif arm_name == "ivara_full":
                # Full IVARA system
                iv_out = answer_question(
                    query=q_text,
                    k=5,
                    min_score=0.30,
                    data_path=memories_json,
                    model=model,
                    use_graph=True,
                    graph_store=graph_store,
                    enable_temporal=True
                )
                raw_ctx = iv_out.get("retrieved_context") or {}
                retrieved_items = raw_ctx.get("retrieved_memories") or raw_ctx.get("selected_chunks", [])
                cited_ids = [c["memory_id"] for c in iv_out.get("citations", []) if "memory_id" in c]
                pred_result = {
                    "status": iv_out.get("status", "answered"),
                    "answer": iv_out.get("answer", ""),
                    "citations": cited_ids,
                    "retrieved_context": retrieved_items
                }
        except Exception as e:
            pred_result = {
                "status": "error",
                "answer": f"Evaluation error: {e}",
                "citations": [],
                "retrieved_context": []
            }

        elapsed_ms = (time.time() - t_start) * 1000.0

        # Compute metric evaluations
        ret_metrics = evaluate_retrieval(retrieved_items, q_spec.get("supporting_sessions", []), q_spec.get("is_answerable", True))
        ans_metrics = evaluate_answer(pred_result, q_spec)
        cite_metrics = evaluate_citations(cited_ids, retrieved_items)

        record = {
            "question_id": q_id,
            "category": q_cat,
            "split": q_spec.get("split", "test"),
            "question": q_text,
            "expected_answer": q_spec.get("expected_answer"),
            "predicted_answer": pred_result.get("answer", ""),
            "predicted_status": pred_result.get("status"),
            "citations": cited_ids,
            "latency_ms": round(elapsed_ms, 2),
            "retrieved_context": retrieved_items,
            "retrieval_metrics": ret_metrics,
            "answer_metrics": ans_metrics,
            "citation_metrics": cite_metrics
        }

        predictions[q_id] = record

        # Checkpoint every 5 questions
        if idx % 5 == 0 or idx == len(questions):
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"[{arm_name}] Evaluation complete. Results saved to {out_file}")
    return predictions


def compare_and_analyze(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question predictions across arms, run statistical tests, and export artifacts."""
    arms = [
        "baseline_vector",
        "baseline_hybrid",
        "baseline_hybrid_speaker_aware",
        "ivara_hybrid_no_graph",
        "ivara_full",
        # Ablations
        "ivara_no_ontology_no_graph",
        "ivara_graph_no_ontology",
        "ivara_full_no_temporal"
    ]

    arm_data = {}
    for arm in arms:
        pred_path = RESULTS_DIR / f"predictions_{arm}.json"
        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as f:
                arm_data[arm] = json.load(f)

    if not arm_data:
        print("No evaluation predictions found to analyze.")
        return {}

    categories = [
        "single_fact_retrieval", "multi_hop_reasoning", "entity_relationships",
        "temporal_reasoning", "historical_state_retrieval", "knowledge_updates",
        "contradiction_handling", "preferences_and_constraints", "negative_information",
        "unanswerable_questions", "cross_session_synthesis", "complex_multi_fact"
    ]

    summary = {
        "overall": {},
        "category_wise": {},
        "calibration": {},
        "latency_stats": {}
    }

    # 1. Overall stats and Wilson 95% CIs
    for arm, preds in arm_data.items():
        n = len(preds)
        if n == 0:
            continue

        covs = [p["answer_metrics"]["fact_coverage"] for p in preds.values()]
        bin_accs = [p["answer_metrics"]["binary_accuracy"] for p in preds.values()]
        ems = [p["answer_metrics"]["exact_match"] for p in preds.values()]
        r1s = [p["retrieval_metrics"]["recall_at_1"] for p in preds.values()]
        r5s = [p["retrieval_metrics"]["recall_at_5"] for p in preds.values()]
        mrrs = [p["retrieval_metrics"]["mrr"] for p in preds.values()]
        cites = [p["citation_metrics"]["citation_validity"] for p in preds.values()]
        lats = [p["latency_ms"] for p in preds.values()]

        sorted_lats = sorted(lats)
        p95_idx = int(0.95 * len(sorted_lats))

        em_successes = sum(1 for e in ems if e >= 0.99)
        em_ci = wilson_score_interval(em_successes, n)
        bin_successes = sum(1 for b in bin_accs if b >= 0.99)
        bin_ci = wilson_score_interval(bin_successes, n)

        summary["overall"][arm] = {
            "questions_count": n,
            "mean_fact_coverage": round(sum(covs) / n, 4),
            "binary_accuracy": round(sum(bin_accs) / n, 4),
            "binary_accuracy_95ci": bin_ci,
            "exact_match_rate": round(sum(ems) / n, 4),
            "exact_match_95ci": em_ci,
            "recall_at_1": round(sum(r1s) / n, 4),
            "recall_at_5": round(sum(r5s) / n, 4),
            "mean_mrr": round(sum(mrrs) / n, 4),
            "citation_validity": round(sum(cites) / n, 4),
            "mean_latency_ms": round(sum(lats) / n, 1),
            "median_latency_ms": round(sorted_lats[len(sorted_lats) // 2], 1),
            "p95_latency_ms": round(sorted_lats[p95_idx], 1)
        }

        # Calibration
        summary["calibration"][arm] = analyze_abstention_and_calibration(preds)

    # 2. Category-wise breakdown
    for cat in categories:
        summary["category_wise"][cat] = {}
        for arm, preds in arm_data.items():
            cat_preds = [p for p in preds.values() if p["category"] == cat]
            if not cat_preds:
                continue
            covs = [p["answer_metrics"]["fact_coverage"] for p in cat_preds]
            bin_accs = [p["answer_metrics"]["binary_accuracy"] for p in cat_preds]
            r5s = [p["retrieval_metrics"]["recall_at_5"] for p in cat_preds]
            summary["category_wise"][cat][arm] = {
                "count": len(cat_preds),
                "fact_coverage": round(sum(covs) / len(cat_preds), 4),
                "binary_accuracy": round(sum(bin_accs) / len(cat_preds), 4),
                "recall_at_5": round(sum(r5s) / len(cat_preds), 4)
            }

    # 3. Statistical Hypothesis Testing
    target_pairs = [
        ("baseline_vector", "baseline_hybrid_speaker_aware", "Perspective & Hybrid Lift"),
        ("baseline_hybrid_speaker_aware", "ivara_hybrid_no_graph", "IVARA Ontology Lift vs Fair RAG"),
        ("ivara_hybrid_no_graph", "ivara_full", "Knowledge Graph & Temporal Lift"),
        ("baseline_hybrid_speaker_aware", "ivara_full", "Full Architecture vs Fair RAG")
    ]

    statistical_results = {}
    p_values_to_correct = []

    for arm_a, arm_b, desc in target_pairs:
        if arm_a in arm_data and arm_b in arm_data:
            common_qids = sorted(list(set(arm_data[arm_a].keys()) & set(arm_data[arm_b].keys())))
            outcomes_a_em = [int(arm_data[arm_a][qid]["answer_metrics"]["exact_match"]) for qid in common_qids]
            outcomes_b_em = [int(arm_data[arm_b][qid]["answer_metrics"]["exact_match"]) for qid in common_qids]

            outcomes_a_bin = [int(arm_data[arm_a][qid]["answer_metrics"]["binary_accuracy"]) for qid in common_qids]
            outcomes_b_bin = [int(arm_data[arm_b][qid]["answer_metrics"]["binary_accuracy"]) for qid in common_qids]

            cov_a = [arm_data[arm_a][qid]["answer_metrics"]["fact_coverage"] for qid in common_qids]
            cov_b = [arm_data[arm_b][qid]["answer_metrics"]["fact_coverage"] for qid in common_qids]

            mcnemar_em = mcnemar_test(outcomes_a_em, outcomes_b_em)
            mcnemar_bin = mcnemar_test(outcomes_a_bin, outcomes_b_bin)
            boot_cov = bootstrap_mean_diff_ci(cov_a, cov_b, n_resamples=1000)

            comp_key = f"{arm_a}__vs__{arm_b}"
            statistical_results[comp_key] = {
                "description": desc,
                "n_questions": len(common_qids),
                "mcnemar_exact_match": mcnemar_em,
                "mcnemar_binary_accuracy": mcnemar_bin,
                "bootstrap_coverage_diff": boot_cov
            }
            p_values_to_correct.append((comp_key, mcnemar_bin["p_value"]))

    corrected_p = holm_bonferroni_correction(p_values_to_correct)
    for c in corrected_p:
        k = c["comparison"]
        if k in statistical_results:
            statistical_results[k]["holm_bonferroni"] = c

    # 4. Save JSON and CSV deliverables
    summary_path = RESULTS_DIR / "results_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    stat_path = RESULTS_DIR / "statistical_tests.json"
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(statistical_results, f, indent=2)

    # Per-Question Comparison CSV
    csv_path = RESULTS_DIR / "per_question_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["question_id", "category", "split", "is_answerable"]
        for arm in arm_data.keys():
            header.extend([f"{arm}_cov", f"{arm}_bin", f"{arm}_em", f"{arm}_r5", f"{arm}_status", f"{arm}_latency_ms"])
        writer.writerow(header)

        for q in questions:
            qid = q["id"]
            row = [qid, q["category"], q.get("split", "test"), q.get("is_answerable", True)]
            for arm, preds in arm_data.items():
                p = preds.get(qid, {})
                ans_m = p.get("answer_metrics", {})
                ret_m = p.get("retrieval_metrics", {})
                row.extend([
                    ans_m.get("fact_coverage", 0.0),
                    ans_m.get("binary_accuracy", 0.0),
                    ans_m.get("exact_match", 0.0),
                    ret_m.get("recall_at_5", 0.0),
                    p.get("predicted_status", "none"),
                    p.get("latency_ms", 0.0)
                ])
            writer.writerow(row)

    # Category Metrics CSV
    cat_csv_path = RESULTS_DIR / "category_metrics.csv"
    with open(cat_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        cat_header = ["category", "arm", "count", "fact_coverage", "binary_accuracy", "recall_at_5"]
        writer.writerow(cat_header)
        for cat, arm_dict in summary["category_wise"].items():
            for arm, m_dict in arm_dict.items():
                writer.writerow([cat, arm, m_dict["count"], m_dict["fact_coverage"], m_dict["binary_accuracy"], m_dict["recall_at_5"]])

    # Latency Metrics CSV
    lat_csv_path = RESULTS_DIR / "latency_metrics.csv"
    with open(lat_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        lat_header = ["arm", "mean_ms", "median_ms", "p95_ms"]
        writer.writerow(lat_header)
        for arm, stats in summary["overall"].items():
            writer.writerow([arm, stats["mean_latency_ms"], stats["median_latency_ms"], stats["p95_latency_ms"]])

    print(f"\n--- Comparative Validation Summary & Deliverables Exported ---")
    print(f"Summary JSON:      {summary_path}")
    print(f"Statistical Tests: {stat_path}")
    print(f"Per-Question CSV:  {csv_path}")
    print(f"Category CSV:      {cat_csv_path}")
    print(f"Latency CSV:       {lat_csv_path}")

    # Display summary table
    print("\n" + "=" * 115)
    print(f"{'Configuration':<30} | {'FactCov':<8} | {'BinaryAcc':<10} | {'ExactMatch':<10} | {'Recall@5':<9} | {'MRR':<7} | {'MeanLat(ms)':<11}")
    print("-" * 115)
    for arm, stats in summary["overall"].items():
        print(f"{arm:<30} | {stats['mean_fact_coverage']*100:>6.2f}% | {stats['binary_accuracy']*100:>8.2f}% | {stats['exact_match_rate']*100:>8.2f}% | {stats['recall_at_5']*100:>7.2f}% | {stats['mean_mrr']:>7.3f} | {stats['mean_latency_ms']:>10.1f}")
    print("=" * 115)

    return summary
