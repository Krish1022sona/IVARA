"""
Comprehensive Benchmark Runner for IVARA Memory Graph Evaluation.

Ablation Configurations:
Config A: Existing Vector Baseline (Episode-level vector search)
Config B: Chunked Vector Retrieval (FAISS vector search over chunks)
Config C: Chunked Hybrid Retrieval (Vector + BM25 lexical over chunks)
Config D: Hybrid Retrieval + Graph Expansion (Vector + BM25 + Relevance-Guided Traversal)
Config E: Full Graph-Assisted Retrieval with Evidence Verification (Vector + BM25 + Entity Seeds + Traversal + Fact Verification)
"""

import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import faiss

from src.graph.store import SQLiteGraphStore
from src.graph.builder import MemoryGraphBuilder
from src.graph.retrieval import GraphAssistedRetriever, BM25LexicalRetriever
from src.retrieval import retrieve as vector_retrieve
from src.answering import answer_question
from evaluation.long_context.corrected_metrics import (
    calculate_retrieval_metrics,
    calculate_comprehensive_abstention,
    evaluate_rubric_fact
)
from src.embedding import get_embedding

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE_DIR / "questions.json"
MULTI_MEMORIES_PATH = BASE_DIR / "multi_episode" / "memories.json"
MULTI_ENTITIES_PATH = BASE_DIR / "multi_episode" / "entities.json"
MULTI_GRAPH_PATH = BASE_DIR / "multi_episode" / "graph.db"

SINGLE_MEMORIES_PATH = BASE_DIR / "single_episode" / "memories.json"
SINGLE_ENTITIES_PATH = BASE_DIR / "single_episode" / "entities.json"
SINGLE_GRAPH_PATH = BASE_DIR / "single_episode" / "graph.db"


def robust_llm_caller(prompt: str, system_prompt: str) -> str:
    import ollama
    from src.answering import AnswerPayload
    concise_sys = system_prompt + "\nCRITICAL: Keep internal reasoning under 50 words. Output the structured JSON payload immediately."
    response = ollama.chat(
        model="gpt-oss",
        messages=[
            {"role": "system", "content": concise_sys},
            {"role": "user", "content": prompt}
        ],
        format=AnswerPayload.model_json_schema(),
        options={
            "temperature": 0,
            "num_ctx": 16384,
            "num_predict": 4096
        }
    )
    return response.message.content


def evaluate_retrieval_configuration(
    config_name: str,
    questions: list[dict[str, Any]],
    memories: list[dict[str, Any]],
    store: SQLiteGraphStore,
    ep_to_id: dict[str, str],
    k: int = 3
) -> dict[str, Any]:
    """
    Evaluate retrieval across 48 questions for a specific configuration.
    """
    case_results = []
    latencies = []
    expanded_counts = []

    # Prepare memories and vector index for episode-level baseline (Config A)
    memo_texts = [m.get("text", "") for m in memories]
    memo_embs = np.array([get_embedding(t) for t in memo_texts], dtype="float32") if memo_texts else None
    memo_index = None
    if memo_embs is not None and memo_embs.ndim == 2 and memo_embs.shape[0] > 0:
        faiss.normalize_L2(memo_embs)
        memo_index = faiss.IndexFlatIP(memo_embs.shape[1])
        memo_index.add(memo_embs)

    # Prepare chunks and vector index for chunked configurations
    chunk_nodes = store.get_nodes_by_type("chunk")
    chunk_texts = [c.properties.get("text", "") for c in chunk_nodes]
    chunk_embs = np.array([get_embedding(t) for t in chunk_texts], dtype="float32") if chunk_texts else None
    chunk_index = None
    if chunk_embs is not None and chunk_embs.ndim == 2 and chunk_embs.shape[0] > 0:
        faiss.normalize_L2(chunk_embs)
        chunk_index = faiss.IndexFlatIP(chunk_embs.shape[1])
        chunk_index.add(chunk_embs)

    bm25 = BM25LexicalRetriever([{"id": c.id, "text": c.properties.get("text", "")} for c in chunk_nodes])
    retriever_full = GraphAssistedRetriever(store)

    for q in questions:
        qid = q["id"]
        query = q["question"]
        req_episodes = q.get("required_episodes", [])
        supporting_ids = [ep_to_id[ep] for ep in req_episodes if ep in ep_to_id]

        t0 = time.perf_counter()
        retrieved_memory_ids = []
        nodes_expanded = 0

        # Configuration Logic:
        if config_name == "config_a_vector_baseline":
            # Episode-level vector search with pre-indexed FAISS
            if memo_index is not None:
                q_emb = np.array([get_embedding(query)], dtype="float32")
                faiss.normalize_L2(q_emb)
                scores, idxs = memo_index.search(q_emb, min(k, len(memories)))
                retrieved_memory_ids = [memories[idx]["id"] for idx in idxs[0] if idx >= 0]
                nodes_expanded = len(idxs[0])
            else:
                raw = vector_retrieve(query, k=k, min_score=None, memories=memories)
                retrieved_memory_ids = [m["id"] for m in raw]
                nodes_expanded = len(raw)

        elif config_name == "config_b_chunked_vector":
            # FAISS vector search over chunks with RRF
            res = retriever_full.retrieve(query, k=k, mode="chunked_vector", use_graph_expansion=False)
            retrieved_memory_ids = [m["id"] for m in res["retrieved_memories"]]
            nodes_expanded = len(res["retrieved_chunks"])

        elif config_name == "config_c_chunked_hybrid":
            # Vector + BM25 RRF over chunks (no graph)
            res = retriever_full.retrieve(query, k=k, mode="chunked_hybrid", use_graph_expansion=False)
            retrieved_memory_ids = [m["id"] for m in res["retrieved_memories"]]
            nodes_expanded = len(res["retrieved_chunks"])

        elif config_name == "config_d_hybrid_graph_expansion":
            # Hybrid + Graph Traversal OR rule (without entity seeds)
            res = retriever_full.retrieve(query, k=k, mode="hybrid_graph_expansion", use_graph_expansion=True, max_traversal_nodes=15)
            retrieved_memory_ids = [m["id"] for m in res["retrieved_memories"]]
            nodes_expanded = len(res["diagnostics"].get("traversal_details", []))

        elif config_name == "config_e_full_graph_assisted":
            # Full Graph-assisted (Vector + BM25 + Entity Seeds + OR Traversal + Verification)
            res = retriever_full.retrieve(query, k=k, mode="full_graph", use_graph_expansion=True, max_traversal_nodes=20)
            retrieved_memory_ids = [m["id"] for m in res["retrieved_memories"]]
            nodes_expanded = len(res["diagnostics"].get("graph_expanded_nodes", []))

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        expanded_counts.append(nodes_expanded)

        ret_m = calculate_retrieval_metrics(
            retrieved_ids=retrieved_memory_ids,
            supporting_ids=supporting_ids,
            k=k
        )

        case_results.append({
            "id": qid,
            "category": q["category"],
            "question": query,
            "supporting_memory_ids": supporting_ids,
            "retrieved_memory_ids": retrieved_memory_ids,
            "standard_precision_at_k": ret_m["precision_at_k"],
            "set_precision": ret_m["set_precision"],
            "recall_at_k": ret_m["recall_at_k"],
            "latency_ms": round((t1 - t0) * 1000.0, 2)
        })

    ans_cases = [c for c in case_results if c["supporting_memory_ids"]]
    unans_cases = [c for c in case_results if not c["supporting_memory_ids"]]

    avg_std_prec = sum(c["standard_precision_at_k"] for c in ans_cases) / len(ans_cases)
    avg_set_prec = sum(c["set_precision"] for c in ans_cases) / len(ans_cases)
    avg_recall = sum(c["recall_at_k"] for c in ans_cases) / len(ans_cases)
    avg_lat = sum(latencies) / len(latencies)
    avg_nodes = sum(expanded_counts) / len(expanded_counts)

    return {
        "config_name": config_name,
        "standard_precision_at_k_answerable": round(avg_std_prec, 4),
        "set_precision_answerable": round(avg_set_prec, 4),
        "recall_at_k_answerable": round(avg_recall, 4),
        "avg_latency_ms": round(avg_lat, 2),
        "avg_nodes_expanded": round(avg_nodes, 1),
        "cases": case_results
    }


def run_answering_evaluation(
    questions: list[dict[str, Any]],
    store: SQLiteGraphStore,
    ep_to_id: dict[str, str],
    k: int = 3
) -> dict[str, Any]:
    """
    Run end-to-end answer generation with Ollama gpt-oss using full graph-assisted retrieval.
    """
    cases = []
    print(f"Generating answers for {len(questions)} benchmark questions via Ollama...")

    for idx, q in enumerate(questions, 1):
        qid = q["id"]
        category = q["category"]
        query = q["question"]
        gold_ans = q["gold_answer"]
        rubric = q.get("rubric", "")
        should_abstain = q.get("should_abstain", False)
        req_episodes = q.get("required_episodes", [])
        supporting_ids = [ep_to_id[ep] for ep in req_episodes if ep in ep_to_id]

        print(f"[{idx:02d}/48] Answering {qid}: {query[:50]}...")
        ans_output = answer_question(
            query=query,
            k=k,
            use_graph=True,
            graph_store=store,
            model="gpt-oss",
            llm_caller=robust_llm_caller
        )

        actual_status = ans_output.get("status", "answered")
        actual_answer = ans_output.get("answer", "")
        cited_citations = ans_output.get("citations", [])
        cited_ids = [c["memory_id"] for c in cited_citations]

        rubric_res = evaluate_rubric_fact(
            qid=qid,
            actual_answer=actual_answer,
            actual_status=actual_status,
            gold_answer=gold_ans,
            rubric=rubric,
            should_abstain=should_abstain
        )

        ret_context = ans_output.get("retrieved_context", {})
        retrieved_ids = [m["id"] for m in ret_context.get("retrieved_memories", [])]

        if cited_ids:
            cit_valid = len(set(cited_ids) & set(retrieved_ids)) / len(cited_ids)
        else:
            cit_valid = 1.0 if should_abstain else 0.0

        cases.append({
            "id": qid,
            "category": category,
            "question": query,
            "gold_answer": gold_ans,
            "actual_answer": actual_answer,
            "actual_status": actual_status,
            "supporting_memory_ids": supporting_ids,
            "retrieved_memory_ids": retrieved_ids,
            "cited_memory_ids": cited_ids,
            "rubric_passed": rubric_res["passed"],
            "rubric_score": rubric_res["score"],
            "rubric_reason": rubric_res["reason"],
            "citation_validity": round(cit_valid, 3)
        })

    abstention_stats = calculate_comprehensive_abstention(cases)
    rubric_pass_rate = sum(1 for c in cases if c["rubric_passed"]) / len(cases)
    avg_cit_validity = sum(c["citation_validity"] for c in cases) / len(cases)

    ans_cases_list = [c for c in cases if c.get("supporting_memory_ids")]
    cases_with_citations = [c for c in cases if c.get("cited_memory_ids")]
    cit_precision = sum(c["citation_validity"] for c in cases_with_citations) / len(cases_with_citations) if cases_with_citations else 1.0
    cit_coverage = len(cases_with_citations) / len(ans_cases_list) if ans_cases_list else 0.0

    # Category breakdown
    cat_summary = {}
    for c in cases:
        cat = c["category"]
        if cat not in cat_summary:
            cat_summary[cat] = {"count": 0, "rubric_passed": 0, "failures": 0}
        cat_summary[cat]["count"] += 1
        if c["rubric_passed"]:
            cat_summary[cat]["rubric_passed"] += 1
        else:
            cat_summary[cat]["failures"] += 1

    for cat in cat_summary:
        cnt = cat_summary[cat]["count"]
        cat_summary[cat]["accuracy"] = round(cat_summary[cat]["rubric_passed"] / cnt, 3)

    return {
        "rubric_accuracy": round(rubric_pass_rate, 4),
        "citation_validity_strict": round(avg_cit_validity, 4),
        "citation_precision": round(cit_precision, 4),
        "citation_coverage": round(cit_coverage, 4),
        "abstention": abstention_stats,
        "category_breakdown": cat_summary,
        "cases": cases
    }


def rebuild_graph(graph_path: Path, memories_path: Path, entities_path: Path) -> SQLiteGraphStore:
    store = SQLiteGraphStore(graph_path)
    store.clear()
    with open(memories_path, "r", encoding="utf-8") as f:
        memos = json.load(f)
    entities = []
    if entities_path.exists():
        with open(entities_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
    builder = MemoryGraphBuilder(store)
    stats = builder.build_from_records(memories=memos, entities=entities)
    print(f"Graph at {graph_path.name} rebuilt: {stats['nodes_count']} nodes, {stats['edges_count']} edges.")
    return store


def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    # 1. Multi-Episode Ingestion Setup & Rebuild
    with open(MULTI_MEMORIES_PATH, "r", encoding="utf-8") as f:
        multi_memories = json.load(f)
    multi_ep_to_id = {f"Episode {idx}": m["id"] for idx, m in enumerate(multi_memories, 1)}
    store_multi = rebuild_graph(MULTI_GRAPH_PATH, MULTI_MEMORIES_PATH, MULTI_ENTITIES_PATH)

    # 2. Single-Episode Ingestion Setup & Rebuild
    with open(SINGLE_MEMORIES_PATH, "r", encoding="utf-8") as f:
        single_memories = json.load(f)
    single_ep_to_id = {f"Episode {idx}": single_memories[0]["id"] for idx in range(1, 11)}
    store_single = rebuild_graph(SINGLE_GRAPH_PATH, SINGLE_MEMORIES_PATH, SINGLE_ENTITIES_PATH)

    configs_to_run = [
        "config_a_vector_baseline",
        "config_b_chunked_vector",
        "config_c_chunked_hybrid",
        "config_d_hybrid_graph_expansion",
        "config_e_full_graph_assisted"
    ]

    print("==================================================================")
    print("STEP 1: ABLATION OF RETRIEVAL CONFIGURATIONS (MULTI-EPISODE)")
    print("==================================================================")
    multi_ablation = {}
    for cfg in configs_to_run:
        print(f"Evaluating {cfg} on Multi-Episode...")
        res = evaluate_retrieval_configuration(cfg, questions, multi_memories, store_multi, multi_ep_to_id)
        multi_ablation[cfg] = res
        print(f"  -> Recall@3: {res['recall_at_k_answerable']*100:.1f}%, P@3: {res['standard_precision_at_k_answerable']*100:.1f}%, Latency: {res['avg_latency_ms']}ms")

    print("\n==================================================================")
    print("STEP 2: ABLATION OF RETRIEVAL CONFIGURATIONS (SINGLE-EPISODE)")
    print("==================================================================")
    single_ablation = {}
    for cfg in configs_to_run:
        print(f"Evaluating {cfg} on Single-Episode...")
        res = evaluate_retrieval_configuration(cfg, questions, single_memories, store_single, single_ep_to_id)
        single_ablation[cfg] = res
        print(f"  -> Recall@3: {res['recall_at_k_answerable']*100:.1f}%, P@3: {res['standard_precision_at_k_answerable']*100:.1f}%, Latency: {res['avg_latency_ms']}ms")

    print("\n==================================================================")
    print("STEP 3: END-TO-END ANSWER GENERATION EVALUATION (CONFIG E)")
    print("==================================================================")
    answering_results = run_answering_evaluation(questions, store_multi, multi_ep_to_id)
    print(f"  -> Overall Rubric Accuracy: {answering_results['rubric_accuracy']*100:.1f}%")
    print(f"  -> Abstention F1: {answering_results['abstention']['abstention_f1']}")
    print(f"  -> False Abstention Rate: {answering_results['abstention']['false_abstention_rate_on_answerables']*100:.1f}%")

    # Clean store connections
    store_multi.close()
    store_single.close()

    # Save consolidated results
    consolidated = {
        "multi_episode_ablation": {
            k: {
                "config_name": v["config_name"],
                "recall_at_k_answerable": v["recall_at_k_answerable"],
                "standard_precision_at_k_answerable": v["standard_precision_at_k_answerable"],
                "set_precision_answerable": v["set_precision_answerable"],
                "avg_latency_ms": v["avg_latency_ms"],
                "avg_nodes_expanded": v["avg_nodes_expanded"]
            }
            for k, v in multi_ablation.items()
        },
        "single_episode_ablation": {
            k: {
                "config_name": v["config_name"],
                "recall_at_k_answerable": v["recall_at_k_answerable"],
                "standard_precision_at_k_answerable": v["standard_precision_at_k_answerable"],
                "set_precision_answerable": v["set_precision_answerable"],
                "avg_latency_ms": v["avg_latency_ms"],
                "avg_nodes_expanded": v["avg_nodes_expanded"]
            }
            for k, v in single_ablation.items()
        },
        "end_to_end_answering": answering_results,
        "multi_ablation_cases": multi_ablation,
        "single_ablation_cases": single_ablation
    }

    out_file = BASE_DIR / "graph_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\nAll benchmark results saved to {out_file.name}")


if __name__ == "__main__":
    main()
