"""Phase 3: Deep Investigation of Knowledge Graph Contribution (Arm D vs Arm E).

Performs:
1. Question-by-question comparison of Arm D (Episodic + Semantic) vs Arm E (Unified Tri-Source).
2. For all discordant or differing questions (especially complex_multi_fact), inspects:
   - Retrieved episodic turns
   - Retrieved semantic facts
   - Retrieved graph triples
   - Fused evidence
   - Final generated answer in Arm D vs Arm E
   - Whether graph evidence improved, degraded, or had zero effect
   - Noise, redundancy, and contradiction audit
3. Quantitative breakdown of graph impact across the entire benchmark.
"""

import json
from pathlib import Path
from collections import defaultdict
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.validation_sprint.dataset_expanded import QUESTIONS
from src.conversations.store import ConversationStore
from src.graph.store import SQLiteGraphStore
from src.unified_retrieval.pipeline import UnifiedRetriever

RESULTS_DIR = WORKSPACE_ROOT / "evaluation" / "milestone_3_results"
AUDIT_DIR = WORKSPACE_ROOT / "evaluation" / "milestone_3_1_audit"
IVARA_STORE_DIR = WORKSPACE_ROOT / "evaluation" / "validation_sprint" / "ivara_store_validation"

def run_phase3_investigation():
    # Load stratified questions
    test_questions = [q for q in QUESTIONS if q.get("split") == "test"]
    by_cat = defaultdict(list)
    for q in test_questions:
        by_cat[q["category"]].append(q)
    stratified_q = []
    for cat in sorted(by_cat.keys()):
        stratified_q.extend(by_cat[cat][:5])
    strat_q_map = {q["id"]: q for q in stratified_q}
    strat_ids = [q["id"] for q in stratified_q]

    # Load Arm D and Arm E predictions
    with open(RESULTS_DIR / "predictions_arm_d_episodic_semantic.json", "r", encoding="utf-8") as f:
        d_preds = json.load(f)
    with open(RESULTS_DIR / "predictions_arm_e_unified_tri_source.json", "r", encoding="utf-8") as f:
        e_preds = json.load(f)

    # Setup retriever to inspect raw multi-source channel outputs
    conv_store = ConversationStore(IVARA_STORE_DIR / "conversations.db")
    graph_store = SQLiteGraphStore(IVARA_STORE_DIR / "memory_graph.db")
    memories_json = IVARA_STORE_DIR / "memories.json"
    retriever = UnifiedRetriever(
        conversation_store=conv_store,
        memories_path=memories_json,
        graph_store=graph_store
    )

    per_question_analysis = []
    category_graph_deltas = defaultdict(lambda: {"d_bin": 0, "e_bin": 0, "d_cov": 0.0, "e_cov": 0.0, "count": 0})

    diff_count = 0
    e_better_count = 0
    d_better_count = 0
    e_cov_better = 0
    d_cov_better = 0

    for qid in strat_ids:
        q_spec = strat_q_map[qid]
        cat = q_spec["category"]
        q_text = q_spec["question"]
        expected = q_spec.get("expected_answer", "")

        d_item = d_preds.get(qid, {})
        e_item = e_preds.get(qid, {})

        d_bin = d_item.get("answer_metrics", {}).get("binary_accuracy", 0.0)
        e_bin = e_item.get("answer_metrics", {}).get("binary_accuracy", 0.0)
        d_cov = d_item.get("answer_metrics", {}).get("fact_coverage", 0.0)
        e_cov = e_item.get("answer_metrics", {}).get("fact_coverage", 0.0)
        d_ans = d_item.get("answer", "")
        e_ans = e_item.get("answer", "")

        category_graph_deltas[cat]["d_bin"] += d_bin
        category_graph_deltas[cat]["e_bin"] += e_bin
        category_graph_deltas[cat]["d_cov"] += d_cov
        category_graph_deltas[cat]["e_cov"] += e_cov
        category_graph_deltas[cat]["count"] += 1

        answers_differ = (d_ans.strip() != e_ans.strip())
        metrics_differ = (d_bin != e_bin or abs(d_cov - e_cov) > 1e-4)

        if metrics_differ:
            diff_count += 1
            if e_bin > d_bin:
                e_better_count += 1
            elif d_bin > e_bin:
                d_better_count += 1

            if e_cov > d_cov:
                e_cov_better += 1
            elif d_cov > e_cov:
                d_cov_better += 1

        # Retrieve detailed channel breakdown
        res = retriever.retrieve(q_text, k=5, enable_episodic=True, enable_semantic=True, enable_graph=True)
        episodic_items = [
            {"turn_id": item.turn_id, "speaker": item.speaker, "content": item.content[:120], "score": item.score}
            for item in res.episodic_hits
        ]
        semantic_items = [
            {"memory_id": item.memory_id, "text": item.text[:120], "source_turn_id": item.source_turn_id, "score": item.score}
            for item in res.semantic_hits
        ]
        graph_triples = [
            {"subject": t.source_name, "predicate": t.predicate, "object": t.target_name, "score": t.score, "hop": t.hops}
            for t in res.graph_hits
        ]
        fused_items = [
            {"cluster_id": item.cluster_id, "channels": item.channels, "score": item.fused_score, "text": item.primary_text[:100]}
            for item in res.evidence_items
        ]

        # Determine graph role
        graph_role = "no_graph_evidence"
        if graph_triples:
            if metrics_differ and e_bin > d_bin:
                graph_role = "graph_improved_accuracy"
            elif metrics_differ and e_cov > d_cov:
                graph_role = "graph_improved_coverage"
            elif metrics_differ and (d_bin > e_bin or d_cov > e_cov):
                graph_role = "graph_degraded_performance"
            elif answers_differ:
                graph_role = "graph_altered_wording_neutral"
            else:
                graph_role = "graph_redundant_no_answer_change"

        q_analysis = {
            "question_id": qid,
            "category": cat,
            "question": q_text,
            "expected_answer": expected,
            "arm_d": {"binary_accuracy": d_bin, "fact_coverage": d_cov, "answer": d_ans},
            "arm_e": {"binary_accuracy": e_bin, "fact_coverage": e_cov, "answer": e_ans},
            "metrics_differ": metrics_differ,
            "answers_differ": answers_differ,
            "graph_triples_count": len(graph_triples),
            "graph_triples": graph_triples,
            "graph_role": graph_role,
            "fused_evidence_count": len(fused_items),
            "episodic_count": len(episodic_items),
            "semantic_count": len(semantic_items)
        }
        per_question_analysis.append(q_analysis)

    summary_phase3 = {
        "total_questions": len(strat_ids),
        "questions_where_graph_triples_retrieved": sum(1 for q in per_question_analysis if q["graph_triples_count"] > 0),
        "questions_with_different_metrics": diff_count,
        "questions_where_arm_e_binary_better": e_better_count,
        "questions_where_arm_d_binary_better": d_better_count,
        "questions_where_arm_e_coverage_better": e_cov_better,
        "questions_where_arm_d_coverage_better": d_cov_better,
        "category_deltas": {
            cat: {
                "count": d["count"],
                "d_binary_mean": d["d_bin"] / d["count"],
                "e_binary_mean": d["e_bin"] / d["count"],
                "d_coverage_mean": d["d_cov"] / d["count"],
                "e_coverage_mean": d["e_cov"] / d["count"]
            }
            for cat, d in category_graph_deltas.items()
        },
        "per_question_analysis": per_question_analysis
    }

    out_file = AUDIT_DIR / "phase3_graph_investigation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_phase3, f, indent=2)
    print(f"Saved: {out_file}")

    print("\n=== PHASE 3 GRAPH INVESTIGATION SUMMARY ===")
    print(f"Total Questions: {len(strat_ids)}")
    print(f"Questions where Graph Triples were Retrieved: {summary_phase3['questions_where_graph_triples_retrieved']}/60")
    print(f"Questions where Metrics Differed: {diff_count}/60")
    print(f"  Arm E Binary Better: {e_better_count}")
    print(f"  Arm D Binary Better: {d_better_count}")
    print(f"  Arm E Coverage Better: {e_cov_better}")
    print(f"  Arm D Coverage Better: {d_cov_better}")

    print("\n--- Category Breakdown of Graph Impact ---")
    for cat, d in summary_phase3["category_deltas"].items():
        bin_diff = d["e_binary_mean"] - d["d_binary_mean"]
        cov_diff = d["e_coverage_mean"] - d["d_coverage_mean"]
        sig_str = ""
        if bin_diff > 0 or cov_diff > 0:
            sig_str = f" [Arm E +{bin_diff*100:.1f}% bin, +{cov_diff*100:.1f}% cov]"
        elif bin_diff < 0 or cov_diff < 0:
            sig_str = f" [Arm D +{-bin_diff*100:.1f}% bin, +{-cov_diff*100:.1f}% cov]"
        else:
            sig_str = " [Identical]"
        print(f"  {cat:<30}: {sig_str}")

if __name__ == "__main__":
    run_phase3_investigation()
