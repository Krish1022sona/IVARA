import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')
import json
from collections import Counter

with open("evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/detailed_17_audit.json", "r", encoding="utf-8") as f:
    items = json.load(f)

print("=== SESSION DIVERSITY IN RETRIEVED CHUNKS ===")
for it in items:
    qid = it["question_id"]
    tt = it["task_type"]
    cids = it["retrieved_chunk_ids"]
    target_sids = it["target_session_ids"]
    
    # Extract session IDs from chunk IDs: e.g. sess_answer_67be2c38_2_c001 -> answer_67be2c38_2
    sids = [c.replace("sess_", "").rsplit("_c", 1)[0] for c in cids]
    counts = Counter(sids)
    
    # Check if multiple chunks from one session crowded out an answer session
    has_target = it["gold_evidence_in_retrieved"]
    missing_target_sids = [ts for ts in target_sids if not any(ts in c for c in cids)]
    
    print(f"[{qid}] ({tt}) Chunks: {len(cids)} | Sessions: {dict(counts)} | Target SIDs: {target_sids} | Missing Targets: {missing_target_sids}")
