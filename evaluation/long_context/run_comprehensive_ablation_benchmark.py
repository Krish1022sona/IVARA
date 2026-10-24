"""
Comprehensive End-to-End Ablation Benchmark for IVARA Memory Pipeline.

Compares 5 Configurations across 48 Benchmark Questions:
  Config A: Vector Baseline (Chunked FAISS cosine similarity)
  Config B: Hybrid Vector + BM25 (RRF fusion, chunks only, no triples)
  Config C: Hybrid + Verified Triples (Hybrid top-3 chunks + 1-hop verified triples, no traversal)
  Config D: Full Graph-Assisted (Seeds + Relevance-guided Traversal + Triples)
  Config E: Hybrid + Query Decomposition + Selective Graph Evidence (Decomposition + Subquery Hybrid + Selective 1-Hop Traversal)

Evaluates:
  - Retrieval Metrics: Recall@3, Precision@3, Set-Precision@3, Retrieval Latency (median & mean)
  - End-to-End Answering Metrics: Accuracy, Answerable Acc, Unanswerable Acc, False Abstention Rate, Unsupported Answers, Multi-Hop Acc, Temporal Acc, Citation Coverage, ID Validity, Semantic Entailment, End-to-End Latency
"""

import sys
import io
import time
import json
from pathlib import Path
from typing import Any
import numpy as np

# Ensure UTF-8 printing on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS_PATH = BASE_DIR / "questions.json"
MULTI_MEMORIES_PATH = BASE_DIR / "multi_episode" / "memories.json"
MULTI_GRAPH_PATH = BASE_DIR / "multi_episode" / "graph.db"

from src.graph.store import SQLiteGraphStore
from src.graph.retrieval import GraphAssistedRetriever
from src.query.decomposition import QueryDecomposer
from src.answering import answer_question
from evaluation.long_context.corrected_metrics import (
    calculate_retrieval_metrics,
    calculate_comprehensive_abstention,
    evaluate_rubric_fact
)


def robust_llm_caller(prompt: str, system_prompt: str) -> str:
    import ollama
    from src.answering import AnswerPayload
    from src.query.decomposition import DecompositionPayload

    # Determine schema based on system prompt / instruction
    is_decomposition = ("decomposition" in system_prompt.lower() or "decompose" in prompt.lower())
    target_schema = DecompositionPayload.model_json_schema() if is_decomposition else AnswerPayload.model_json_schema()

    concise_sys = system_prompt + "\nCRITICAL: Keep internal reasoning under 40 words. Output the structured JSON payload immediately."
    response = ollama.chat(
        model="gpt-oss",
        messages=[
            {"role": "system", "content": concise_sys},
            {"role": "user", "content": prompt}
        ],
        format=target_schema,
        options={
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 1024
        }
    )
    return response.message.content


def run_benchmark():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with open(MULTI_MEMORIES_PATH, "r", encoding="utf-8") as f:
        memories = json.load(f)

    store = SQLiteGraphStore(MULTI_GRAPH_PATH)
    retriever = GraphAssistedRetriever(store)
    decomposer = QueryDecomposer(llm_caller=robust_llm_caller)

    ep_to_id = {f"Episode {i}": f"m{i:03d}" for i in range(1, 11)}

    # Warm-up pass to eliminate initial cold start
    print("Executing warm-up passes...")
    for wq in [q["question"] for q in questions[:3]]:
        _ = retriever.retrieve(wq, k=3, mode="chunked_vector", include_triples=False)
        _ = retriever.retrieve(wq, k=3, mode="chunked_hybrid", include_triples=False)
        _ = retriever.retrieve(wq, k=3, mode="chunked_hybrid", include_triples=True)
        _ = retriever.retrieve(wq, k=3, mode="full_graph_assisted", include_triples=True)

    print("Warm-up complete.")

    configs = ["A", "B", "C", "D", "E"]
    results = {}

    # --- 1. RETRIEVAL ABLATION (A, B, C, D, E) ---
    print("\n=======================================================")
    print("PHASE 1: CONTROLLED RETRIEVAL EVALUATION (A through E)")
    print("=======================================================")

    for cfg in configs:
        case_retrieval = []
        ret_latencies = []

        for q in questions:
            qid = q["id"]
            query = q["question"]
            req_episodes = q.get("required_episodes", [])
            supporting_ids = [ep_to_id[ep] for ep in req_episodes if ep in ep_to_id]

            # Measure retrieval latency (median over 3 passes)
            pass_times = []
            res_payload = None
            for _ in range(3):
                t0 = time.perf_counter()
                if cfg == "A":
                    res_payload = retriever.retrieve(query, k=3, mode="chunked_vector", use_graph_expansion=False, include_triples=False)
                elif cfg == "B":
                    res_payload = retriever.retrieve(query, k=3, mode="chunked_hybrid", use_graph_expansion=False, include_triples=False)
                elif cfg == "C":
                    res_payload = retriever.retrieve(query, k=3, mode="chunked_hybrid", use_graph_expansion=False, include_triples=True)
                elif cfg == "D":
                    res_payload = retriever.retrieve(query, k=3, mode="full_graph_assisted", use_graph_expansion=True, include_triples=True)
                elif cfg == "E":
                    # For E retrieval measurement: decompose and retrieve_decomposed
                    dec = decomposer.decompose(query)
                    if dec.is_decomposed:
                        res_payload = retriever.retrieve_decomposed(query, dec, k=3)
                    else:
                        res_payload = retriever.retrieve(query, k=3, mode="chunked_hybrid", use_graph_expansion=False, include_triples=True)
                dt = (time.perf_counter() - t0) * 1000.0
                pass_times.append(dt)

            ret_latencies.append(float(np.median(pass_times)))

            ret_m_ids = []
            seen = set()
            for d in res_payload.get("direct_evidence", []):
                mid = d.get("memory_id")
                if mid and mid not in seen:
                    seen.add(mid)
                    ret_m_ids.append(mid)

            ret_m = calculate_retrieval_metrics(
                retrieved_ids=ret_m_ids,
                supporting_ids=supporting_ids,
                k=3
            )
            case_retrieval.append({
                "id": qid,
                "retrieved_memory_ids": ret_m_ids,
                "supporting_memory_ids": supporting_ids,
                "standard_precision_at_k": ret_m["precision_at_k"],
                "set_precision": ret_m["set_precision"],
                "recall_at_k": ret_m["recall_at_k"]
            })

        ans_cases = [c for c in case_retrieval if c["supporting_memory_ids"]]
        avg_std_prec = sum(c["standard_precision_at_k"] for c in ans_cases) / len(ans_cases)
        avg_set_prec = sum(c["set_precision"] for c in ans_cases) / len(ans_cases)
        avg_recall = sum(c["recall_at_k"] for c in ans_cases) / len(ans_cases)

        results[f"retrieval_config_{cfg}"] = {
            "config": cfg,
            "recall_at_k": round(avg_recall, 4),
            "precision_at_k": round(avg_std_prec, 4),
            "set_precision_at_k": round(avg_set_prec, 4),
            "median_retrieval_latency_ms": round(float(np.median(ret_latencies)), 2),
            "mean_retrieval_latency_ms": round(float(np.mean(ret_latencies)), 2),
            "cases": case_retrieval
        }
        print(f"Config {cfg} Retrieval -> Recall@3: {avg_recall*100:.1f}%, Precision@3: {avg_std_prec*100:.1f}%, Median Latency: {np.median(ret_latencies):.2f} ms")

    # --- 2. END-TO-END ANSWERING EVALUATION FOR CONFIG E ---
    print("\n=======================================================")
    print("PHASE 2: END-TO-END ANSWERING EVALUATION FOR CONFIG E")
    print("=======================================================")

    e_cases = []
    e2e_latencies = []
    subquery_counts = []
    model_call_counts = []

    for idx, q in enumerate(questions, 1):
        qid = q["id"]
        category = q["category"]
        query = q["question"]
        gold_ans = q["gold_answer"]
        rubric = q.get("rubric", "")
        should_abstain = q.get("should_abstain", False)
        req_episodes = q.get("required_episodes", [])
        supporting_ids = [ep_to_id[ep] for ep in req_episodes if ep in ep_to_id]

        print(f"[{idx:02d}/48] Config E answering {qid}: {query[:50]}...")
        t0 = time.perf_counter()

        ans_output = answer_question(
            query=query,
            k=3,
            use_graph=True,
            graph_store=store,
            model="gpt-oss",
            llm_caller=robust_llm_caller,
            use_decomposition=True
        )

        dt = (time.perf_counter() - t0) * 1000.0
        e2e_latencies.append(dt)

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
        diag = ret_context.get("diagnostics", {})

        sq_count = diag.get("subquestion_count", 1)
        subquery_counts.append(sq_count)
        # Model calls: 1 (decomposition) + 1 (answering) if decomposed, or 1 if bypassed
        n_model_calls = 2 if diag.get("mode") == "decomposed_hybrid_graph" else 1
        model_call_counts.append(n_model_calls)

        if cited_ids:
            cit_valid = len(set(cited_ids) & set(retrieved_ids)) / len(cited_ids)
        else:
            cit_valid = 1.0 if should_abstain else 0.0

        e_cases.append({
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
            "citation_validity": round(cit_valid, 3),
            "subquestions": diag.get("subquestions", [query]),
            "is_decomposed": diag.get("mode") == "decomposed_hybrid_graph",
            "latency_ms": round(dt, 2)
        })

    # Summary metrics for E
    abstention_stats = calculate_comprehensive_abstention(e_cases)
    rubric_pass_rate = sum(1 for c in e_cases if c["rubric_passed"]) / len(e_cases)
    avg_cit_validity = sum(c["citation_validity"] for c in e_cases) / len(e_cases)

    ans_cases_list = [c for c in e_cases if not questions[int(c["id"][1:])-1].get("should_abstain", False)]
    unans_cases_list = [c for c in e_cases if questions[int(c["id"][1:])-1].get("should_abstain", False)]

    ans_passed = sum(1 for c in ans_cases_list if c["rubric_passed"])
    unans_passed = sum(1 for c in unans_cases_list if c["rubric_passed"])

    cases_with_citations = [c for c in ans_cases_list if c.get("cited_memory_ids")]
    cit_precision = sum(c["citation_validity"] for c in cases_with_citations) / len(cases_with_citations) if cases_with_citations else 1.0
    cit_coverage = len(cases_with_citations) / len(ans_cases_list) if ans_cases_list else 0.0

    # Category breakdown
    cat_summary = {}
    for c in e_cases:
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
        cat_summary[cat]["accuracy"] = round(cat_summary[cat]["rubric_passed"] / cnt, 4)

    results["answering_config_e"] = {
        "overall_accuracy": round(rubric_pass_rate, 4),
        "overall_passed_count": sum(1 for c in e_cases if c["rubric_passed"]),
        "total_count": len(e_cases),
        "answerable_accuracy": round(ans_passed / len(ans_cases_list), 4),
        "answerable_passed_count": ans_passed,
        "answerable_total": len(ans_cases_list),
        "unanswerable_accuracy": round(unans_passed / len(unans_cases_list), 4),
        "unanswerable_passed_count": unans_passed,
        "unanswerable_total": len(unans_cases_list),
        "citation_coverage": round(cit_coverage, 4),
        "citation_precision": round(cit_precision, 4),
        "strict_citation_validity": round(avg_cit_validity, 4),
        "abstention": abstention_stats,
        "category_breakdown": cat_summary,
        "median_e2e_latency_ms": round(float(np.median(e2e_latencies)), 2),
        "mean_e2e_latency_ms": round(float(np.mean(e2e_latencies)), 2),
        "avg_subquestions": round(float(np.mean(subquery_counts)), 2),
        "avg_model_calls": round(float(np.mean(model_call_counts)), 2),
        "cases": e_cases
    }

    output_path = BASE_DIR / "comprehensive_ablation_a_to_e_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n=======================================================")
    print("CONFIG E BENCHMARK COMPLETE")
    print(f"Overall Accuracy: {rubric_pass_rate*100:.2f}% ({sum(1 for c in e_cases if c['rubric_passed'])}/{len(e_cases)})")
    print(f"Answerable Accuracy: {ans_passed/len(ans_cases_list)*100:.2f}% ({ans_passed}/{len(ans_cases_list)})")
    print(f"Unanswerable Accuracy: {unans_passed/len(unans_cases_list)*100:.2f}% ({unans_passed}/{len(unans_cases_list)})")
    print(f"Median E2E Latency: {np.median(e2e_latencies):.2f} ms")
    print(f"Saved complete results to {output_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
