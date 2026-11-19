"""
Deep inspection script: loads the raw instance sessions, chunks, and exact evidence
for each of the 17 failures to verify categorization.
"""

import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
from evaluation.adapters.longmemeval_adapter import LongMemEvalAdapter
from src.chunking.turn_chunker import TurnAwareChunker

exp_dir = Path("evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40")
sample_path = Path("evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json")

adapter = LongMemEvalAdapter(sample_path, exp_dir)
instances, _ = adapter.load_and_validate()
inst_map = {inst.question_id: inst for inst in instances}
chunker = TurnAwareChunker(window_size=3, overlap=1)

with open(exp_dir / "audited_17_failures.json", "r", encoding="utf-8") as f:
    failures = json.load(f)

print(f"Auditing {len(failures)} failures...")

detailed_audit = []
for f in failures:
    qid = f["question_id"]
    inst = inst_map[qid]
    all_chunks = chunker.chunk_all_sessions(inst.sessions)
    chunk_map = {c.chunk_id: c for c in all_chunks}
    
    # Target chunks (from answer session IDs)
    target_sids = set(str(s) for s in inst.answer_session_ids)
    target_chunks = [c for c in all_chunks if any(str(ts) in c.session_id or str(ts) in c.chunk_id for ts in target_sids)]
    
    # Retrieved chunks
    retrieved_chunks_objs = [chunk_map[cid] for cid in f["retrieved_chunk_ids"] if cid in chunk_map]
    
    # Check if target content actually exists in retrieved chunks
    target_in_retrieved = any(
        any(str(ts) in c.session_id or str(ts) in c.chunk_id for ts in target_sids)
        for c in retrieved_chunks_objs
    )
    
    detailed_audit.append({
        "question_id": qid,
        "task_type": f["task_type"],
        "question": f["question"],
        "gold_answer": f["gold_answer"],
        "generated_answer": f["generated_answer"],
        "status": f["status"],
        "retrieved_chunk_ids": f["retrieved_chunk_ids"],
        "gold_retrieval_rank": f["gold_retrieval_rank"],
        "gold_evidence_in_retrieved": target_in_retrieved,
        "target_session_ids": list(target_sids),
        "target_chunks_text": [tc.text[:300] for tc in target_chunks[:2]],
        "retrieved_chunks_text": [rc.text[:300] for rc in retrieved_chunks_objs[:3]],
        "gemma_rationale": f["gemma_rationale"]
    })

with open(exp_dir / "detailed_17_audit.json", "w", encoding="utf-8") as out_f:
    json.dump(detailed_audit, out_f, indent=2)

print("Saved detailed_17_audit.json successfully.")
