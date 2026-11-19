"""Rerun 250-Question Retrieval Evaluation with Fixed Provenance Mapping.
Computes Recall@1, Recall@3, Recall@5, MRR, Evidence Coverage across all 250 test questions.
Milestone 3.2 Batch 5.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.validation_sprint.dataset_expanded import QUESTIONS
from evaluation.validation_sprint.metrics import evaluate_retrieval, _get_conversation_session_map
from evaluation.run_final_benchmark import get_evaluation_stores
from src.unified_retrieval.pipeline import UnifiedRetriever
from src.retrieval import retrieve, load_memories, build_index, get_embedding
import faiss
import numpy as np

def run_fast_250_retrieval():
    conv_store, graph_store, memories_json = get_evaluation_stores()
    session_map = _get_conversation_session_map()
    test_questions = [q for q in QUESTIONS if q.get("split") == "test"]
    print(f"Total test questions: {len(test_questions)}")

    # Pre-build Baseline FAISS index once for speed
    raw_mems = load_memories(memories_json)
    base_index = build_index(raw_mems)

    retrievers = {
        "arm_b_episodic": UnifiedRetriever(conversation_store=conv_store, memories=[]),
        "arm_c_semantic": UnifiedRetriever(memories_path=memories_json),
        "arm_d_adaptive": UnifiedRetriever(conversation_store=conv_store, memories_path=memories_json, use_adaptive_routing=True),
        "arm_e_full": UnifiedRetriever(conversation_store=conv_store, memories_path=memories_json, graph_store=graph_store, use_adaptive_routing=True)
    }

    results = defaultdict(list)

    for idx, q in enumerate(test_questions, 1):
        q_text = q["question"]
        is_ans = q.get("is_answerable", True)
        supp = q.get("supporting_sessions", [])

        # Arm A: Baseline
        t0 = time.perf_counter()
        if not is_ans or not q_text.strip() or base_index is None:
            arm_a_items = []
        else:
            q_emb = np.array([get_embedding(q_text.strip())], dtype="float32")
            faiss.normalize_L2(q_emb)
            scores, indices = base_index.search(q_emb, min(5, len(raw_mems)))
            arm_a_items = []
            for s, i in zip(scores[0], indices[0]):
                if s >= 0.30:
                    m = dict(raw_mems[i])
                    m["score"] = float(s)
                    arm_a_items.append(m)
        lat_a = (time.perf_counter() - t0) * 1000.0
        m_a = evaluate_retrieval(arm_a_items, supp, is_ans, session_map=session_map)
        m_a["latency_ms"] = lat_a
        results["arm_a_baseline"].append(m_a)

        # Arms B, C, D, E
        for arm_name in ["arm_b_episodic", "arm_c_semantic", "arm_d_adaptive", "arm_e_full"]:
            r = retrievers[arm_name]
            t0 = time.perf_counter()
            if arm_name == "arm_b_episodic":
                res = r.retrieve(q_text, k=5, enable_episodic=True, enable_semantic=False, enable_graph=False)
            elif arm_name == "arm_c_semantic":
                res = r.retrieve(q_text, k=5, enable_episodic=False, enable_semantic=True, enable_graph=False)
            elif arm_name == "arm_d_adaptive":
                res = r.retrieve(q_text, k=5, enable_episodic=True, enable_semantic=True, enable_graph=False, use_adaptive_routing=True)
            elif arm_name == "arm_e_full":
                res = r.retrieve(q_text, k=5, enable_episodic=True, enable_semantic=True, enable_graph=True, use_adaptive_routing=True)

            lat = (time.perf_counter() - t0) * 1000.0
            m_res = evaluate_retrieval(res.direct_evidence, supp, is_ans, session_map=session_map)
            m_res["latency_ms"] = lat
            results[arm_name].append(m_res)

        if idx % 50 == 0 or idx == len(test_questions):
            print(f"Processed {idx}/{len(test_questions)} questions...")

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

    out_file = Path(__file__).resolve().parent / "final_benchmark" / "retrieval_250_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_250, f, indent=2)

    print("\n================== FULL 250-QUESTION RETRIEVAL BENCHMARK ==================")
    print(f"{'Arm':<20} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR':<8} | {'Evidence Cov':<12} | {'Latency (ms)':<12}")
    print("-" * 95)
    for arm_name, s in summary_250.items():
        print(f"{arm_name:<20} | {s['mean_recall_at_1']*100:>8.2f}% | {s['mean_recall_at_3']*100:>8.2f}% | {s['mean_recall_at_5']*100:>8.2f}% | {s['mean_mrr']:>8.4f} | {s['mean_evidence_coverage']*100:>10.2f}% | {s['mean_latency_ms']:>10.2f} ms")
    print("===========================================================================")

if __name__ == "__main__":
    run_fast_250_retrieval()

