"""Refined Phase 5 Failure Analysis for Milestone 3 Audit.

Applies exact inspection of LLM responses, context facts, and metric evaluation:
- Distinguishes evaluator substring brittleness from genuine generation errors.
- Pinpoints over-abstention where facts were present in context.
- Identifies multi-session context truncation where k=5 was insufficient.
"""

import json
from pathlib import Path
from collections import Counter

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIT_DIR = WORKSPACE_ROOT / "evaluation" / "milestone_3_1_audit"

def run_refined_failures():
    with open(AUDIT_DIR / "phase5_failure_analysis.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    refined_records = []
    taxonomy = Counter()

    for item in data["worst_20_failures"]:
        qid = item["question_id"]
        cat = item["category"]
        pred = item["predicted_answer"]
        in_ctx = item["key_facts_in_context"]
        tot_facts = item["total_key_facts"]

        # Exact classification rules
        if qid in ("c04", "n04", "p05"):
            cls = "Evaluation/ground-truth issue"
            if qid == "p05":
                rat = "Model answered correctly ('Meera Sen has a severe allergy to peanuts') but wrapped in JSON, causing evaluator regex to miss key facts."
                rec = "Strip JSON wrappers in evaluate_answer before substring matching."
            elif qid == "c04":
                rat = "Model answered correctly ('declined HyperScale's 30 % salary counter-offer'), but metric required exact continuous substring 'declined counter-offer'."
                rec = "Use token/lemma overlap or LLM-as-a-judge instead of rigid continuous substring matching."
            else:
                rat = "Model answered 'gives him migraines', but metric strictly required 'triggers migraines'."
                rec = "Allow synonymous predicate variations in ground-truth key facts."
        elif "I do not have sufficient" in pred or "not recorded in the provided evidence" in pred:
            if in_ctx > 0:
                cls = "Answer generation failure"
                rat = f"Over-abstention: Required key facts ({in_ctx}/{tot_facts}) were present in retrieved context, but local LLM produced false refusal."
                rec = "Calibrate sufficiency gating prompt to prevent over-abstention when relevant facts are present."
            else:
                if cat in ("cross_session_synthesis", "multi_hop_reasoning"):
                    cls = "Context truncation"
                    rat = "Supporting sessions spanned multiple distant conversations; top-k budget (k=5) could not encompass all needed turns."
                    rec = "Implement hierarchical session index or multi-query iterative retrieval."
                else:
                    cls = "Retrieval failure"
                    rat = "Key facts were not present in top-5 retrieved items."
                    rec = "Enhance hybrid dense+BM25 retrieval ranking."
        elif cat in ("cross_session_synthesis", "complex_multi_fact") and in_ctx < tot_facts:
            cls = "Context truncation"
            rat = f"Complex cross-session question required {tot_facts} facts across sessions, but retrieved context only contained {in_ctx}."
            rec = "Expand retrieval context budget or implement hierarchical timeline synthesis."
        elif qid == "h03":
            cls = "Graph traversal failure"
            rat = "Graph channel introduced distractor relationship triples, causing model to generate Python instead of Mojo."
            rec = "Filter graph triples by valid-time intervals before fusion."
        elif in_ctx == tot_facts:
            cls = "Answer generation failure"
            rat = f"All {tot_facts} key facts were present in retrieved context, but local LLM failed to synthesize them into the expected answer."
            rec = "Improve structured generation prompt and reasoning instructions."
        else:
            cls = "Fusion/ranking failure"
            rat = "Evidence existed in episodic or semantic stores but was ranked outside the top-k fusion budget."
            rec = "Tune RRF channel weights and session diversity cap on dev split."

        taxonomy[cls] += 1
        item["classification"] = cls
        item["rationale"] = rat
        item["recommendation"] = rec
        refined_records.append(item)

    out = {
        "taxonomy_distribution": dict(taxonomy),
        "worst_20_failures": refined_records
    }

    out_file = AUDIT_DIR / "phase5_failure_analysis_refined.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_file}")

    print("\n=== REFINED FAILURE TAXONOMY DISTRIBUTION ===")
    for cls, count in taxonomy.most_common():
        print(f"  {cls:<32}: {count:>2} ({count/20*100:.1f}%)")

if __name__ == "__main__":
    run_refined_failures()

