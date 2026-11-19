import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

with open("evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/detailed_17_audit.json", "r", encoding="utf-8") as f:
    items = json.load(f)

for idx, it in enumerate(items):
    qid = it["question_id"]
    tt = it["task_type"]
    q = it["question"]
    gold = it["gold_answer"]
    gen = it["generated_answer"]
    status = it["status"]
    gold_rank = it["gold_retrieval_rank"]
    gold_in_ret = it["gold_evidence_in_retrieved"]
    rationale = it["gemma_rationale"]
    
    print("=" * 80)
    print(f"[{idx+1}/17] QID: {qid} | Type: {tt}")
    print(f"Question: {q}")
    print(f"Gold: {gold}")
    print(f"Generated: {repr(gen[:160])}")
    print(f"Status: {status} | Gold Rank: {gold_rank} | Target In Retrieved: {gold_in_ret}")
    print(f"Judge Rationale: {repr(rationale[:160])}")
    print("Target Evidence Snippet:")
    for i, t in enumerate(it.get("target_chunks_text", [])[:1]):
        print(f"  [T{i+1}] {repr(t[:200])}")
    print("Retrieved Snippets:")
    for i, r in enumerate(it.get("retrieved_chunks_text", [])[:2]):
        print(f"  [R{i+1}] {repr(r[:200])}")
