"""Phase 5: Deep Failure Analysis for Milestone 3 Audit.

Identifies the 20 worst-performing questions in Arm E (and cross-arms),
inspects the raw evidence, ground truth, and generated answers,
and classifies each failure into the defined failure taxonomy.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import sys
import re

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from evaluation.validation_sprint.dataset_expanded import QUESTIONS
from evaluation.validation_sprint.metrics import _get_conversation_session_map
from src.conversations.store import ConversationStore
from src.graph.store import SQLiteGraphStore
from src.unified_retrieval.pipeline import UnifiedRetriever

RESULTS_DIR = WORKSPACE_ROOT / "evaluation" / "milestone_3_results"
AUDIT_DIR = WORKSPACE_ROOT / "evaluation" / "milestone_3_1_audit"
IVARA_STORE_DIR = WORKSPACE_ROOT / "evaluation" / "validation_sprint" / "ivara_store_validation"

def run_failure_analysis():
    test_questions = [q for q in QUESTIONS if q.get("split") == "test"]
    by_cat = defaultdict(list)
    for q in test_questions:
        by_cat[q["category"]].append(q)
    stratified_q = []
    for cat in sorted(by_cat.keys()):
        stratified_q.extend(by_cat[cat][:5])
    strat_q_map = {q["id"]: q for q in stratified_q}
    strat_ids = [q["id"] for q in stratified_q]

    with open(RESULTS_DIR / "predictions_arm_e_unified_tri_source.json", "r", encoding="utf-8") as f:
        e_preds = json.load(f)

    conv_store = ConversationStore(IVARA_STORE_DIR / "conversations.db")
    graph_store = SQLiteGraphStore(IVARA_STORE_DIR / "memory_graph.db")
    memories_json = IVARA_STORE_DIR / "memories.json"
    retriever = UnifiedRetriever(
        conversation_store=conv_store,
        memories_path=memories_json,
        graph_store=graph_store
    )
    conv_map = _get_conversation_session_map(IVARA_STORE_DIR / "conversations.db")

    # Score questions in Arm E to find the 20 worst
    scored_questions = []
    for qid in strat_ids:
        q_spec = strat_q_map[qid]
        item = e_preds.get(qid, {})
        m = item.get("answer_metrics", {})
        bin_acc = m.get("binary_accuracy", 0.0)
        cov = m.get("fact_coverage", 0.0)
        is_ans = q_spec.get("is_answerable", True)

        scored_questions.append({
            "qid": qid,
            "category": q_spec["category"],
            "binary_accuracy": bin_acc,
            "fact_coverage": cov,
            "is_answerable": is_ans,
            "item": item,
            "q_spec": q_spec
        })

    # Sort: binary_accuracy ascending, fact_coverage ascending
    scored_questions.sort(key=lambda x: (x["binary_accuracy"], x["fact_coverage"]))
    worst_20 = scored_questions[:20]

    failure_records = []
    taxonomy_counts = Counter()

    for idx, sq in enumerate(worst_20, 1):
        qid = sq["qid"]
        q_spec = sq["q_spec"]
        item = sq["item"]
        q_text = q_spec["question"]
        expected = q_spec.get("expected_answer", "")
        predicted = item.get("answer", "")
        cov = sq["fact_coverage"]
        bin_acc = sq["binary_accuracy"]
        cat = sq["category"]
        supp_sessions = q_spec.get("supporting_sessions", [])

        # Retrieve raw diagnostic evidence
        res = retriever.retrieve(q_text, k=5, enable_episodic=True, enable_semantic=True, enable_graph=True)
        ret_sessions = set()
        for hit in res.episodic_hits:
            if hit.conversation_id in conv_map:
                ret_sessions.add(conv_map[hit.conversation_id])
            elif hit.session_id:
                ret_sessions.add(hit.session_id)
        for s_hit in res.semantic_hits:
            if s_hit.raw_source:
                m = re.search(r"session_\d{2}", s_hit.raw_source)
                if m:
                    ret_sessions.add(m.group(0))

        supp_set = set(supp_sessions)
        session_overlap = supp_set & ret_sessions

        # Analyze evidence text vs expected facts
        all_ret_text = " ".join([h.content for h in res.episodic_hits] + [s.text for s in res.semantic_hits])
        key_facts = q_spec.get("key_facts", [])
        facts_in_retrieved = [kf for kf in key_facts if kf.lower() in all_ret_text.lower()]

        # Failure Taxonomy Classification
        if not q_spec.get("is_answerable", True):
            if not item.get("answer_metrics", {}).get("correct_abstention", False):
                classification = "Answer generation failure"
                rationale = "Model failed to abstain on unanswerable question."
                recommendation = "Strengthen sufficiency gate prompt for ungrounded negative claims."
            else:
                classification = "Evaluation/ground-truth issue"
                rationale = "Question was abstained correctly, but low coverage flagged."
                recommendation = "Audit metric calculation on negative questions."
        elif len(session_overlap) == 0:
            classification = "Retrieval failure"
            rationale = f"None of the ground-truth supporting sessions {supp_sessions} were retrieved in top-k context."
            recommendation = "Enhance dense vector and BM25 hybrid indexing or query reformulation."
        elif len(session_overlap) < len(supp_set):
            classification = "Context truncation"
            rationale = f"Cross-session query required sessions {supp_sessions}, but top-k (k=5) and diversity cap (max 3/session) truncated required sessions to {list(ret_sessions)}."
            recommendation = "Implement hierarchical multi-session index and iterative retrieval for multi-session queries."
        elif len(facts_in_retrieved) == len(key_facts):
            classification = "Answer generation failure"
            rationale = "All required key facts were present in retrieved context, but local LLM generated an incomplete or non-matching response."
            recommendation = "Refine prompt templates and output constraints for structured multi-fact synthesis."
        elif cat == "entity_relationships" and len(res.graph_hits) == 0:
            classification = "Entity resolution failure"
            rationale = "Query entities could not be resolved to graph seed nodes, leaving graph triples empty."
            recommendation = "Implement subword alias matching and fuzzy name resolution in SQLiteGraphStore."
        elif cat in ("historical_state_retrieval", "knowledge_updates"):
            classification = "Fusion/ranking failure"
            rationale = "Relevant evidence was found across sessions, but chronological valid-time ranking failed to prioritize the target timeline point."
            recommendation = "Incorporate explicit point-in-time temporal filtering into RRF fusion."
        else:
            classification = "Answer generation failure"
            rationale = "Relevant supporting sessions were retrieved, but generator failed to extract and synthesize the key facts."
            recommendation = "Improve reasoning prompt and chain-of-thought extraction for complex claims."

        taxonomy_counts[classification] += 1

        failure_records.append({
            "rank": idx,
            "question_id": qid,
            "category": cat,
            "question": q_text,
            "expected_answer": expected,
            "predicted_answer": predicted[:200] + ("..." if len(predicted) > 200 else ""),
            "binary_accuracy": bin_acc,
            "fact_coverage": cov,
            "supporting_sessions": supp_sessions,
            "retrieved_sessions": list(ret_sessions),
            "key_facts_in_context": len(facts_in_retrieved),
            "total_key_facts": len(key_facts),
            "classification": classification,
            "rationale": rationale,
            "recommendation": recommendation
        })

    summary_phase5 = {
        "taxonomy_distribution": dict(taxonomy_counts),
        "worst_20_failures": failure_records
    }

    out_file = AUDIT_DIR / "phase5_failure_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_phase5, f, indent=2)
    print(f"Saved: {out_file}")

    print("\n=== PHASE 5 FAILURE TAXONOMY DISTRIBUTION (WORST 20 QUESTIONS) ===")
    for cls, count in taxonomy_counts.most_common():
        print(f"  {cls:<32}: {count:>2} ({count/20*100:.1f}%)")

if __name__ == "__main__":
    from collections import Counter
    run_failure_analysis()
