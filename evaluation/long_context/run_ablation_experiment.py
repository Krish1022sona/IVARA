"""
Controlled Ablation Experiment Runner for IVARA Memory Graph:
Isolating the incremental contribution of:
  A: Vector Baseline (Chunked FAISS vector search)
  B: Hybrid Vector + BM25 (RRF fusion of vector + BM25 lexical)
  C: Hybrid + Verified Graph Triples (No broad traversal, 1-hop chunk-grounded triples only)
  D: Hybrid + Relevance-Guided Graph Traversal and Verified Triples (Full Graph-Assisted)
"""

import sys
import io
import time
import json
from pathlib import Path
from typing import Any
import numpy as np
import faiss

# Ensure clean UTF-8 printing
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS_PATH = BASE_DIR / "questions.json"
MULTI_MEMORIES_PATH = BASE_DIR / "multi_episode" / "memories.json"
MULTI_ENTITIES_PATH = BASE_DIR / "multi_episode" / "entities.json"
MULTI_GRAPH_PATH = BASE_DIR / "multi_episode" / "graph.db"

from src.graph.store import SQLiteGraphStore
from src.graph.retrieval import GraphAssistedRetriever, BM25LexicalRetriever
from src.embedding import get_embedding
from evaluation.long_context.corrected_metrics import calculate_retrieval_metrics


def evaluate_controlled_retrieval():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with open(MULTI_MEMORIES_PATH, "r", encoding="utf-8") as f:
        memories = json.load(f)

    store = SQLiteGraphStore(MULTI_GRAPH_PATH)
    retriever = GraphAssistedRetriever(store)

    ep_to_id = {f"Episode {i}": f"m{i:03d}" for i in range(1, 11)}

    configs = [
        "A_vector_baseline",
        "B_hybrid_vector_bm25",
        "C_hybrid_verified_triples",
        "D_hybrid_traversal_triples"
    ]

    results = {}

    # Warm-up pass to eliminate cold start and page faults
    warmup_queries = [q["question"] for q in questions[:5]]
    for wq in warmup_queries:
        _ = retriever.retrieve(wq, k=3, use_graph_expansion=True, mode="full_graph_assisted")
        _ = retriever.retrieve(wq, k=3, use_graph_expansion=False, mode="chunked_vector")
        _ = retriever.retrieve(wq, k=3, use_graph_expansion=False, mode="chunked_hybrid")

    for cfg in configs:
        print(f"\n--- Running Controlled Evaluation for {cfg} ---")
        case_results = []
        latencies_ms = []
        nodes_expanded_list = []
        triples_expanded_list = []

        # 3 repeated passes per question for latency precision
        for q in questions:
            qid = q["id"]
            query = q["question"]
            req_episodes = q.get("required_episodes", [])
            supporting_ids = [ep_to_id[ep] for ep in req_episodes if ep in ep_to_id]

            # Timing measurement across 3 repeated runs
            runs = []
            res_payload = None
            for _ in range(3):
                t0 = time.perf_counter()
                if cfg == "A_vector_baseline":
                    # A: Chunked vector only (no lexical, no graph, no triples)
                    res_payload = retriever.retrieve(
                        query, k=3, use_graph_expansion=False, mode="chunked_vector", include_triples=False
                    )
                elif cfg == "B_hybrid_vector_bm25":
                    # B: Chunked hybrid vector + BM25 RRF (no graph traversal, no triples)
                    res_payload = retriever.retrieve(
                        query, k=3, use_graph_expansion=False, mode="chunked_hybrid", include_triples=False
                    )
                elif cfg == "C_hybrid_verified_triples":
                    # C: Hybrid retrieval + 1-hop verified triples from candidate entities, NO graph traversal
                    res_payload = retriever.retrieve(
                        query, k=3, use_graph_expansion=False, mode="chunked_hybrid", include_triples=True
                    )
                elif cfg == "D_hybrid_traversal_triples":
                    # D: Full Graph-assisted: Seeds + Relevance-guided Traversal + Triples
                    res_payload = retriever.retrieve(
                        query, k=3, use_graph_expansion=True, mode="full_graph_assisted", include_triples=True
                    )
                dt = (time.perf_counter() - t0) * 1000.0
                runs.append(dt)

            latencies_ms.append(float(np.median(runs)))

            # Extract retrieved memory IDs
            direct_ev = res_payload.get("direct_evidence", [])
            ret_m_ids = []
            seen = set()
            for d in direct_ev:
                mid = d.get("memory_id")
                if mid and mid not in seen:
                    seen.add(mid)
                    ret_m_ids.append(mid)

            diag = res_payload.get("diagnostics", {})
            nodes_exp = len(diag.get("graph_expanded_nodes", []))
            # Direct evidence chunks also count as examined nodes
            total_nodes = max(len(direct_ev), nodes_exp)
            nodes_expanded_list.append(total_nodes)

            context_txt = res_payload.get("context_text", "")
            num_triples = sum(1 for line in context_txt.splitlines() if line.strip().startswith("* Fact:"))
            triples_expanded_list.append(num_triples)

            ret_m = calculate_retrieval_metrics(
                retrieved_ids=ret_m_ids,
                supporting_ids=supporting_ids,
                k=3
            )

            case_results.append({
                "id": qid,
                "retrieved_memory_ids": ret_m_ids,
                "supporting_memory_ids": supporting_ids,
                "standard_precision_at_k": ret_m["precision_at_k"],
                "set_precision": ret_m["set_precision"],
                "recall_at_k": ret_m["recall_at_k"],
                "num_triples": num_triples
            })

        ans_cases = [c for c in case_results if c["supporting_memory_ids"]]
        avg_std_prec = sum(c["standard_precision_at_k"] for c in ans_cases) / len(ans_cases)
        avg_set_prec = sum(c["set_precision"] for c in ans_cases) / len(ans_cases)
        avg_recall = sum(c["recall_at_k"] for c in ans_cases) / len(ans_cases)

        metrics = {
            "config_name": cfg,
            "recall_at_k": round(avg_recall, 4),
            "precision_at_k": round(avg_std_prec, 4),
            "set_precision_at_k": round(avg_set_prec, 4),
            "median_latency_ms": round(float(np.median(latencies_ms)), 2),
            "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
            "avg_nodes_expanded": round(float(np.mean(nodes_expanded_list)), 2),
            "avg_triples_expanded": round(float(np.mean(triples_expanded_list)), 2),
            "cases": case_results
        }

        results[cfg] = metrics
        print(f"  Recall@3 (Answerable): {metrics['recall_at_k'] * 100:.1f}%")
        print(f"  Std Precision@3:       {metrics['precision_at_k'] * 100:.1f}%")
        print(f"  Set Precision@3:       {metrics['set_precision_at_k'] * 100:.1f}%")
        print(f"  Median Latency:        {metrics['median_latency_ms']} ms")
        print(f"  Avg Triples Injected:  {metrics['avg_triples_expanded']}")

    output_path = BASE_DIR / "controlled_ablation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved controlled ablation results to {output_path}")
    return results

if __name__ == "__main__":
    evaluate_controlled_retrieval()
