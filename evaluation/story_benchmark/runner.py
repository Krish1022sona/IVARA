"""Comprehensive Benchmark Runner and CLI for Multi-Part Narrative Reasoning.

Executes controlled experiments across 4 experimental arms:
  Arm 1: Baseline Vector RAG (pure FAISS dense retrieval)
  Arm 2: Baseline Hybrid RAG (Vector + BM25 RRF without graph/ontology)
  Arm 3: IVARA Hybrid Retrieval (Extracted Memories + Ontology + FAISS, no graph)
  Arm 4: IVARA Full System (Extracted Memories + Ontology + Relevance Graph + Temporal Reasoning)

Measures:
  - Exact match / Key facts coverage
  - Answer accuracy
  - Abstention correctness (rejecting unanswerable, answering answerable)
  - Retrieval Recall@1, Recall@3, Recall@5, MRR, Evidence coverage
  - Citation validity & hallucinated citations
  - Per-query latency (mean, median, P95)
  - Category-wise performance breakdowns
"""

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Optional

from .dataset import export_benchmark_data, SESSIONS, QUESTIONS, STORY_DATA_PATH, GROUND_TRUTH_PATH
from .baseline_rag import ConventionalRAGBaseline
from .metrics import evaluate_retrieval, evaluate_answer, evaluate_citations

from src.conversations.store import ConversationStore
from src.graph.store import SQLiteGraphStore
from src.chat.engine import ChatEngine
from src.answering import answer_question
from src.memory import sync_memory_graph


BENCHMARK_DIR = Path(__file__).parent
IVARA_STORE_DIR = BENCHMARK_DIR / "ivara_store"
RESULTS_DIR = BENCHMARK_DIR / "results"


def setup_ivara_stores() -> tuple[ConversationStore, SQLiteGraphStore, Path, Path]:
    """Initialize isolated evaluation databases and memory stores."""
    IVARA_STORE_DIR.mkdir(parents=True, exist_ok=True)
    conv_db = IVARA_STORE_DIR / "conversations.db"
    graph_db = IVARA_STORE_DIR / "memory_graph.db"
    memories_json = IVARA_STORE_DIR / "memories.json"
    entities_json = IVARA_STORE_DIR / "entities.json"

    if not entities_json.exists():
        initial_entities = [
            {
                "id": "self",
                "canonical_name": "SELF",
                "types": ["Person"],
                "aliases": ["I", "me", "my", "myself"]
            }
        ]
        with open(entities_json, "w", encoding="utf-8") as f:
            json.dump(initial_entities, f, indent=2)

    if not memories_json.exists():
        with open(memories_json, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    conv_store = ConversationStore(conv_db)
    graph_store = SQLiteGraphStore(graph_db)

    return conv_store, graph_store, memories_json, entities_json


def ingest_ivara(
    model: str = "gpt-oss",
    force_clean: bool = False
) -> dict[str, Any]:
    """Ingest the 12 multi-part narrative sessions into isolated IVARA stores."""
    print("\n--- Ingesting Narrative Sessions into IVARA ---")
    if force_clean and IVARA_STORE_DIR.exists():
        import shutil
        shutil.rmtree(IVARA_STORE_DIR)
        print("Cleaned existing IVARA evaluation stores.")

    conv_store, graph_store, memories_json, entities_json = setup_ivara_stores()
    engine = ChatEngine(
        conversation_store=conv_store,
        graph_store=graph_store,
        memories_path=memories_json,
        entities_path=entities_json,
        model=model
    )

    # Ingest each session
    total_turns = 0
    total_memories_extracted = 0
    t0 = time.time()

    for sess in SESSIONS:
        s_id = sess["session_id"]
        title = sess["title"]
        date = sess["date"]
        print(f"Ingesting [{s_id}] ({date}) - '{title}'...")

        # Create or resume conversation
        conv = conv_store.create_conversation(title=f"{s_id}: {title}")

        for turn in sess["turns"]:
            if turn["role"] == "user":
                total_turns += 1
                user_msg = turn["content"]
                res = engine.process_turn(
                    user_text=user_msg,
                    conversation_id=conv.id,
                    auto_extract=True
                )
                if res.memory_extracted:
                    total_memories_extracted += 1

    # Ensure graph is fully synced
    sync_memory_graph(data_path=memories_json, graph_store=graph_store, entity_path=entities_json)

    elapsed = time.time() - t0
    node_count = graph_store.count_nodes()
    edge_count = graph_store.count_edges()

    print(f"\nIngestion Complete in {elapsed:.2f}s:")
    print(f"  Total Dialogue Turns Processed: {total_turns}")
    print(f"  Long-Term Memories Formed:     {total_memories_extracted}")
    print(f"  Knowledge Graph Nodes:         {node_count}")
    print(f"  Knowledge Graph Edges:         {edge_count}")

    return {
        "status": "ingested",
        "duration_seconds": round(elapsed, 2),
        "total_turns": total_turns,
        "memories_count": total_memories_extracted,
        "graph_nodes": node_count,
        "graph_edges": edge_count
    }


def run_evaluation_arm(
    arm_name: str,
    questions: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    model: str = "gpt-oss",
    resume: bool = True
) -> dict[str, Any]:
    """Execute evaluation for a specific experimental arm with checkpointing."""
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

    # Initialize backend based on arm
    baseline_rag = None
    conv_store = None
    graph_store = None
    memories_json = None
    entities_json = None

    if arm_name in ("baseline_vector", "baseline_hybrid"):
        baseline_rag = ConventionalRAGBaseline(sessions, model=model)
    elif arm_name in ("ivara_hybrid_no_graph", "ivara_full"):
        conv_store, graph_store, memories_json, entities_json = setup_ivara_stores()
        # Verify store is populated
        with open(memories_json, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        if not m_data:
            print(f"Warning: IVARA store appears empty. Running ingestion first...")
            ingest_ivara(model=model)
            with open(memories_json, "r", encoding="utf-8") as f:
                m_data = json.load(f)

    print(f"\n--- Running Evaluation: {arm_name} ({len(questions)} questions) ---")
    latencies = []

    for idx, q_spec in enumerate(questions, 1):
        q_id = q_spec["id"]
        q_text = q_spec["question"]
        q_cat = q_spec["category"]

        if q_id in predictions:
            latencies.append(predictions[q_id].get("latency_ms", 0.0))
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
            elif arm_name == "ivara_hybrid_no_graph":
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
            elif arm_name == "ivara_full":
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
                "answer": f"Error evaluating question: {e}",
                "citations": [],
                "retrieved_context": []
            }

        elapsed_ms = (time.time() - t_start) * 1000.0
        latencies.append(elapsed_ms)

        # Compute metric evaluations
        ret_metrics = evaluate_retrieval(retrieved_items, q_spec.get("supporting_sessions", []), q_spec.get("is_answerable", True))
        ans_metrics = evaluate_answer(pred_result, q_spec)
        cite_metrics = evaluate_citations(cited_ids, retrieved_items)

        record = {
            "question_id": q_id,
            "category": q_cat,
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

    # Save final predictions
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"[{arm_name}] Evaluation complete. Results saved to {out_file}")
    return predictions


def compare_and_report(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question predictions across all arms into comparative reports."""
    arms = ["baseline_vector", "baseline_hybrid", "ivara_hybrid_no_graph", "ivara_full"]
    arm_data = {}

    for arm in arms:
        pred_path = RESULTS_DIR / f"predictions_{arm}.json"
        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as f:
                arm_data[arm] = json.load(f)

    if not arm_data:
        print("No evaluation predictions found to compare.")
        return {}

    categories = [
        "single_fact_recall", "multi_hop", "entity_relationship",
        "temporal_reasoning", "historical_state", "contradiction_handling",
        "preference_constraint", "negative_fact", "cross_session_synthesis",
        "unanswerable"
    ]

    summary: dict[str, Any] = {
        "overall": {},
        "category_wise": {},
        "latency_stats": {}
    }

    # 1. Overall stats
    for arm, preds in arm_data.items():
        n = len(preds)
        if n == 0:
            continue
        accs = [p["answer_metrics"]["accuracy"] for p in preds.values()]
        ems = [p["answer_metrics"]["exact_match"] for p in preds.values()]
        r1s = [p["retrieval_metrics"]["recall_at_1"] for p in preds.values()]
        r5s = [p["retrieval_metrics"]["recall_at_5"] for p in preds.values()]
        mrrs = [p["retrieval_metrics"]["mrr"] for p in preds.values()]
        cites = [p["citation_metrics"]["citation_validity"] for p in preds.values()]
        lats = [p["latency_ms"] for p in preds.values()]

        sorted_lats = sorted(lats)
        p95_idx = int(0.95 * len(sorted_lats))

        summary["overall"][arm] = {
            "questions_count": n,
            "mean_accuracy": round(sum(accs) / n, 4),
            "exact_match_rate": round(sum(ems) / n, 4),
            "recall_at_1": round(sum(r1s) / n, 4),
            "recall_at_5": round(sum(r5s) / n, 4),
            "mean_mrr": round(sum(mrrs) / n, 4),
            "citation_validity": round(sum(cites) / n, 4),
            "mean_latency_ms": round(sum(lats) / n, 1),
            "median_latency_ms": round(sorted_lats[len(sorted_lats) // 2], 1),
            "p95_latency_ms": round(sorted_lats[p95_idx], 1)
        }

    # 2. Category-wise stats
    for cat in categories:
        summary["category_wise"][cat] = {}
        for arm, preds in arm_data.items():
            cat_preds = [p for p in preds.values() if p["category"] == cat]
            if not cat_preds:
                continue
            accs = [p["answer_metrics"]["accuracy"] for p in cat_preds]
            r5s = [p["retrieval_metrics"]["recall_at_5"] for p in cat_preds]
            summary["category_wise"][cat][arm] = {
                "count": len(cat_preds),
                "accuracy": round(sum(accs) / len(cat_preds), 4),
                "recall_at_5": round(sum(r5s) / len(cat_preds), 4)
            }

    # Write summary JSON
    summary_path = RESULTS_DIR / "results_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write Per-Question Comparison CSV
    csv_path = RESULTS_DIR / "per_question_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["question_id", "category", "is_answerable"]
        for arm in arm_data.keys():
            header.extend([f"{arm}_acc", f"{arm}_r5", f"{arm}_status", f"{arm}_latency_ms"])
        writer.writerow(header)

        for q in questions:
            qid = q["id"]
            row = [qid, q["category"], q.get("is_answerable", True)]
            for arm, preds in arm_data.items():
                p = preds.get(qid, {})
                ans_m = p.get("answer_metrics", {})
                ret_m = p.get("retrieval_metrics", {})
                row.extend([
                    ans_m.get("accuracy", 0.0),
                    ret_m.get("recall_at_5", 0.0),
                    p.get("predicted_status", "none"),
                    p.get("latency_ms", 0.0)
                ])
            writer.writerow(row)

    print(f"\n--- Comparative Evaluation Summary Exported ---")
    print(f"Summary JSON: {summary_path}")
    print(f"Per-Question CSV: {csv_path}")

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'Arm':<24} | {'Accuracy':<9} | {'ExactMatch':<10} | {'Recall@5':<9} | {'MRR':<7} | {'CitationVal':<11} | {'MeanLat(ms)':<11}")
    print("-" * 95)
    for arm, stats in summary["overall"].items():
        print(f"{arm:<24} | {stats['mean_accuracy']*100:>7.2f}% | {stats['exact_match_rate']*100:>8.2f}% | {stats['recall_at_5']*100:>7.2f}% | {stats['mean_mrr']:>7.3f} | {stats['citation_validity']*100:>9.2f}% | {stats['mean_latency_ms']:>10.1f}")
    print("=" * 95)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Part Narrative Reasoning Benchmark Runner")
    parser.add_argument("--prepare", action="store_true", help="Prepare benchmark story and questions")
    parser.add_argument("--ingest", action="store_true", help="Ingest story sessions into IVARA stores")
    parser.add_argument("--system", type=str, default="all", choices=["all", "baseline_vector", "baseline_hybrid", "ivara_hybrid_no_graph", "ivara_full"])
    parser.add_argument("--run", action="store_true", help="Run benchmark evaluation")
    parser.add_argument("--compare", action="store_true", help="Compare arm predictions and generate summary")
    parser.add_argument("--model", type=str, default="gpt-oss", help="Ollama model name")
    parser.add_argument("--force-clean", action="store_true", help="Clean IVARA evaluation stores before ingestion")

    args = parser.parse_args()

    # Load or generate dataset
    if args.prepare or not STORY_DATA_PATH.exists() or not GROUND_TRUTH_PATH.exists():
        export_benchmark_data()

    with open(STORY_DATA_PATH, "r", encoding="utf-8") as f:
        sessions = json.load(f)
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if args.ingest:
        ingest_ivara(model=args.model, force_clean=args.force_clean)

    if args.run:
        target_arms = [args.system] if args.system != "all" else ["baseline_vector", "baseline_hybrid", "ivara_hybrid_no_graph", "ivara_full"]
        for arm in target_arms:
            run_evaluation_arm(arm, questions, sessions, model=args.model)
        compare_and_report(questions)

    if args.compare:
        compare_and_report(questions)


if __name__ == "__main__":
    main()
