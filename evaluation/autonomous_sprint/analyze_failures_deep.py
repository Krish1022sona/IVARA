"""
Deep failure analysis of ARM_4 (Unified Research Candidate) across the fresh heldout 40 sample.
Categorizes every failure into:
1. Retrieval Recall (target session chunk not in retrieved_chunk_ids)
2. Evidence Selection / Context truncation (chunk retrieved but buried)
3. Answer Generation / Reasoning (chunk in context, but model answered incorrectly)
4. Abstention failure (model improperly abstained or failed to abstain)
5. Judge Evaluation discrepancy (answer is factually correct according to evidence, but judge flagged it)
"""

import json

exp_dir = "evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40"

with open(f"{exp_dir}/predictions_ARM_4_UNIFIED_HYBRID.jsonl", "r", encoding="utf-8") as f:
    preds = {line["question_id"]: line for line in (json.loads(l) for l in f)}

with open(f"{exp_dir}/judge_results_ARM_4_UNIFIED_HYBRID.jsonl", "r", encoding="utf-8") as f:
    judges = {line["question_id"]: line for line in (json.loads(l) for l in f)}

failures = []
for qid, j in judges.items():
    if not j["is_correct"]:
        p = preds[qid]
        gold_rank = p.get("gold_retrieval_rank")
        retrieved = p.get("retrieved_chunk_ids", [])
        cat = "unknown"
        if gold_rank is None or gold_rank > 3:
            cat = "Retrieval Recall Deficit (Gold not in top-3)"
        elif p.get("status") == "insufficient_evidence" and not p.get("is_abstention"):
            cat = "Over-Abstention (Gold retrieved but system abstained)"
        elif p.get("is_abstention") and p.get("status") != "insufficient_evidence":
            cat = "Under-Abstention (Unanswerable query answered anyway)"
        else:
            cat = "Generation / Reasoning Limitation (Gold in top-3 but wrong answer)"
            
        failures.append({
            "qid": qid,
            "task_type": p["task_type"],
            "question": p["question"],
            "gold_answer": p["gold_answer"],
            "hypothesis": p["hypothesis"][:150],
            "gold_rank": gold_rank,
            "status": p.get("status"),
            "category": cat,
            "judge_reason": j["reasoning"][:200]
        })

print(f"Total failures for ARM_4: {len(failures)} / 40 (Accuracy: {40-len(failures)}/40 = {(40-len(failures))/40*100:.1f}%)")

cat_counts = {}
for f in failures:
    cat_counts[f["category"]] = cat_counts.get(f["category"], 0) + 1

print("\n--- FAILURE TAXONOMY ---")
for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"{cat}: {count} ({count/len(failures)*100:.1f}%)")

