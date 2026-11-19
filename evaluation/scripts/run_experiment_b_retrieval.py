"""
Experiment B: Session-Diversity Retrieval Evaluation across full LongMemEval-S (500 questions).

Compares 4 isolated retrieval strategies on identical pre-chunked sessions and graph stores:
  - Arm B0: Baseline Top-5 (unconstrained)
  - Arm B1: Strict Max 2 Chunks per Session (max_chunks_per_session=2)
  - Arm B2: Adaptive Session Diversity (query-conditioned cap)
  - Arm B3: Session-Aware MMR (relevance - session penalty)

Measures:
  - Recall@1, Recall@3, Recall@5
  - Evidence Coverage Ratio (% of gold sessions retrieved in direct evidence)
  - Task-specific breakdowns (multi-session, temporal, single-session, etc.)
  - Mean retrieval latency
"""

import json
import time
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.adapters.longmemeval_adapter import LongMemEvalAdapter, LongMemEvalInstance
from evaluation.evaluators.longmemeval_evaluator import OFFICIAL_TASKS
from src.chunking import TurnAwareChunker
from src.chunking.graph_retrieval import TurnGraphBuilder, TurnGraphRetriever
from src.graph.store import SQLiteGraphStore


def run_experiment_b(
    dataset_path: str = "evaluation/data/longmemeval/longmemeval_s_cleaned.json",
    output_dir: str = "evaluation/data/longmemeval/runs/experiment_b_retrieval"
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adapter = LongMemEvalAdapter(dataset_path)
    instances, _ = adapter.load_and_validate()
    print(f"Loaded {len(instances)} instances from {dataset_path}")

    # Strategies to evaluate
    strategies = ["B0_baseline", "B1_max2", "B2_adaptive", "B3_session_mmr"]
    results_by_arm = {s: [] for s in strategies}

    start_time = time.time()
    t_retrieval_by_arm = {s: [] for s in strategies}

    for idx, inst in enumerate(instances):
        if idx % 50 == 0 or idx == len(instances) - 1:
            print(f"[{idx+1}/{len(instances)}] Processing {inst.question_id} ({inst.question_type})...")

        # Set up isolated in-memory or temp SQLite store
        db_file = output_path / f"temp_{inst.question_id}.db"
        if db_file.exists():
            db_file.unlink()

        store = SQLiteGraphStore(db_file)
        chunker = TurnAwareChunker(window_size=3, overlap=1)
        chunks = chunker.chunk_all_sessions(inst.sessions)
        builder = TurnGraphBuilder(store)
        builder.build_from_chunks(chunks)

        retriever = TurnGraphRetriever(chunks=chunks, store=store, use_ontology=True)

        # Retrieve a broader pool (k=15) so re-ranking/diversity strategies have candidates
        t0 = time.time()
        pool_candidates, _ = retriever.retrieve(inst.question, k=15, min_score=None)
        t_base = time.time() - t0

        # Build target gold session set
        target_sids = set()
        if inst.answer_session_ids:
            for s in inst.answer_session_ids:
                s_clean = str(s).replace("sess_", "")
                target_sids.add(s_clean)
                target_sids.add(f"sess_{s_clean}")

        # --- Arm B0: Baseline (Unconstrained Top 5) ---
        t_b0_start = time.time()
        b0_selected = pool_candidates[:5]
        t_retrieval_by_arm["B0_baseline"].append((t_base + (time.time() - t_b0_start)) * 1000)

        # --- Arm B1: Strict Max 2 Chunks per Session ---
        t_b1_start = time.time()
        b1_selected = []
        b1_sess_counts = defaultdict(int)
        for c in pool_candidates:
            sid = c.get("session_id") or c.get("memory_id")
            if b1_sess_counts[sid] < 2:
                b1_selected.append(c)
                b1_sess_counts[sid] += 1
                if len(b1_selected) == 5:
                    break
        t_retrieval_by_arm["B1_max2"].append((t_base + (time.time() - t_b1_start)) * 1000)

        # --- Arm B2: Adaptive Session Diversity ---
        t_b2_start = time.time()
        q_lower = inst.question.lower()
        is_multi_hop_query = (
            inst.question_type == "multi-session" or
            any(w in q_lower for w in ["how many", "all the", "list every", "total", "both", "difference", "between", "each time"])
        )
        max_per_sess = 1 if (is_multi_hop_query and len(pool_candidates) >= 5) else 2
        b2_selected = []
        b2_sess_counts = defaultdict(int)
        for c in pool_candidates:
            sid = c.get("session_id") or c.get("memory_id")
            if b2_sess_counts[sid] < max_per_sess:
                b2_selected.append(c)
                b2_sess_counts[sid] += 1
                if len(b2_selected) == 5:
                    break
        # Fallback to pool if underfilled
        if len(b2_selected) < 5:
            for c in pool_candidates:
                if c not in b2_selected:
                    b2_selected.append(c)
                    if len(b2_selected) == 5:
                        break
        t_retrieval_by_arm["B2_adaptive"].append((t_base + (time.time() - t_b2_start)) * 1000)

        # --- Arm B3: Session-Aware MMR ---
        t_b3_start = time.time()
        # Greedy MMR selection: balance relevance score with penalty for already-selected sessions
        lambda_penalty = 0.25
        b3_selected = []
        b3_selected_sids = set()
        remaining_pool = list(pool_candidates)

        for _ in range(min(5, len(remaining_pool))):
            best_idx = -1
            best_mmr_score = -999.0
            for idx_c, c in enumerate(remaining_pool):
                sid = c.get("session_id") or c.get("memory_id")
                sess_penalty = lambda_penalty if sid in b3_selected_sids else 0.0
                score = c.get("score", 0.0) - sess_penalty
                if score > best_mmr_score:
                    best_mmr_score = score
                    best_idx = idx_c
            if best_idx >= 0:
                chosen = remaining_pool.pop(best_idx)
                b3_selected.append(chosen)
                chosen_sid = chosen.get("session_id") or chosen.get("memory_id")
                b3_selected_sids.add(chosen_sid)
        t_retrieval_by_arm["B3_session_mmr"].append((t_base + (time.time() - t_b3_start)) * 1000)

        # Clean up isolated DB
        store.close()
        if db_file.exists():
            try:
                db_file.unlink()
            except Exception:
                pass

        # Evaluate diagnostic metrics for each arm
        selected_map = {
            "B0_baseline": b0_selected,
            "B1_max2": b1_selected,
            "B2_adaptive": b2_selected,
            "B3_session_mmr": b3_selected
        }

        for arm_name, sel_chunks in selected_map.items():
            retrieved_sids = set()
            gold_rank = None
            for r_i, c in enumerate(sel_chunks):
                sid = str(c.get("session_id") or c.get("memory_id") or "")
                sid_clean = sid.replace("sess_", "")
                retrieved_sids.add(sid_clean)
                retrieved_sids.add(f"sess_{sid_clean}")
                if gold_rank is None and (sid_clean in target_sids or f"sess_{sid_clean}" in target_sids):
                    gold_rank = r_i + 1

            if inst.answer_session_ids and not inst.is_abstention:
                unique_gold = {str(s).replace("sess_", "") for s in inst.answer_session_ids}
                retrieved_gold_count = sum(1 for s in unique_gold if s in retrieved_sids)
                cov_ratio = round(retrieved_gold_count / len(unique_gold), 4)
            else:
                cov_ratio = 1.0

            r1 = bool(gold_rank == 1) if not inst.is_abstention else True
            r3 = bool(gold_rank is not None and gold_rank <= 3) if not inst.is_abstention else True
            r5 = bool(gold_rank is not None and gold_rank <= 5) if not inst.is_abstention else True

            results_by_arm[arm_name].append({
                "question_id": inst.question_id,
                "question_type": inst.question_type,
                "is_abstention": inst.is_abstention,
                "gold_retrieval_rank": gold_rank,
                "recall_at_1": r1,
                "recall_at_3": r3,
                "recall_at_5": r5,
                "evidence_coverage_ratio": cov_ratio,
                "gold_session_count": len(inst.answer_session_ids),
                "retrieved_chunk_ids": [c.get("chunk_id") for c in sel_chunks],
                "retrieved_session_ids": [c.get("session_id") for c in sel_chunks]
            })

    # Aggregate metrics across all 4 arms
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": dataset_path,
        "total_instances": len(instances),
        "arms": {}
    }

    for arm_name in strategies:
        records = results_by_arm[arm_name]
        ans_records = [r for r in records if not r["is_abstention"]]
        arm_summary = {
            "overall": {
                "recall_at_1": round(float(np.mean([1 if r["recall_at_1"] else 0 for r in ans_records])), 4),
                "recall_at_3": round(float(np.mean([1 if r["recall_at_3"] else 0 for r in ans_records])), 4),
                "recall_at_5": round(float(np.mean([1 if r["recall_at_5"] else 0 for r in ans_records])), 4),
                "mean_evidence_coverage_ratio": round(float(np.mean([r["evidence_coverage_ratio"] for r in ans_records])), 4),
                "latency_mean_ms": round(float(np.mean(t_retrieval_by_arm[arm_name])), 2),
                "total_answerable": len(ans_records)
            },
            "by_task": {}
        }
        for task in OFFICIAL_TASKS:
            task_records = [r for r in ans_records if r["question_type"] == task]
            if task_records:
                arm_summary["by_task"][task] = {
                    "recall_at_1": round(float(np.mean([1 if r["recall_at_1"] else 0 for r in task_records])), 4),
                    "recall_at_3": round(float(np.mean([1 if r["recall_at_3"] else 0 for r in task_records])), 4),
                    "recall_at_5": round(float(np.mean([1 if r["recall_at_5"] else 0 for r in task_records])), 4),
                    "mean_evidence_coverage_ratio": round(float(np.mean([r["evidence_coverage_ratio"] for r in task_records])), 4),
                    "instances": len(task_records)
                }
        summary["arms"][arm_name] = arm_summary

        # Write per-arm machine-readable results
        with open(output_path / f"results_{arm_name}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    with open(output_path / "summary_experiment_b.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment B complete in {time.time() - start_time:.2f}s!")
    print(json.dumps(summary["arms"], indent=2))
    return summary


if __name__ == "__main__":
    run_experiment_b()

