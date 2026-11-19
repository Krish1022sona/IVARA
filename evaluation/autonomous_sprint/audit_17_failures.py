"""
Audit script for the 17 incorrect answers in ARM_4_UNIFIED_HYBRID on the fresh_heldout_40 dataset.
"""

import json
from pathlib import Path

exp_dir = Path("evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40")
sample_path = Path("evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json")

with open(sample_path, "r", encoding="utf-8") as f:
    sample_list = json.load(f)
sample_map = {q["question_id"]: q for q in sample_list}

with open(exp_dir / "predictions_ARM_4_UNIFIED_HYBRID.jsonl", "r", encoding="utf-8") as f:
    preds = {line["question_id"]: line for line in (json.loads(l) for l in f)}

with open(exp_dir / "judge_results_ARM_4_UNIFIED_HYBRID.jsonl", "r", encoding="utf-8") as f:
    judges = {line["question_id"]: line for line in (json.loads(l) for l in f)}

incorrect_qids = [qid for qid, j in judges.items() if not j["is_correct"]]
print(f"Total incorrect in ARM_4: {len(incorrect_qids)} / {len(judges)}")

records = []
for qid in incorrect_qids:
    q_data = sample_map[qid]
    p = preds[qid]
    j = judges[qid]
    
    # Check if target session IDs are in retrieved chunks
    target_sids = set(str(s) for s in q_data.get("answer_session_ids", []))
    retrieved_cids = p.get("retrieved_chunk_ids", [])
    gold_rank = p.get("gold_retrieval_rank")
    
    # Check gold session chunk presence in retrieved chunks
    gold_present = False
    if gold_rank is not None and gold_rank <= len(retrieved_cids):
        gold_present = True
    elif any(any(str(ts) in cid for ts in target_sids) for cid in retrieved_cids):
        gold_present = True

    records.append({
        "question_id": qid,
        "task_type": q_data.get("question_type"),
        "question": q_data.get("question"),
        "gold_answer": str(q_data.get("answer")),
        "target_session_ids": list(target_sids),
        "retrieved_chunk_ids": retrieved_cids,
        "gold_retrieval_rank": gold_rank,
        "gold_evidence_present": gold_present,
        "generated_answer": p.get("hypothesis", ""),
        "status": p.get("status"),
        "is_abstention": p.get("is_abstention"),
        "gemma_verdict": "INCORRECT" if not j.get("is_correct") else "CORRECT",
        "gemma_rationale": j.get("reasoning", "")
    })

out_json = exp_dir / "audited_17_failures.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print(f"Saved audit records to {out_json}")
