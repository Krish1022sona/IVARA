"""
Experiment C: Combined Best-Performing Configuration Evaluation on Full LongMemEval-S (500 Questions).

Combines:
  1. Experiment B Best Retrieval: Session-Aware MMR candidate selection (k_pool=15 -> k=5).
  2. Experiment A Best Answering: Ollama num_ctx=8192, query-scoped contradiction filtering,
     bounded temporal claims, strict context character budgeting, and robust JSON recovery.

Evaluates against the frozen 500-question baseline with paired question-level tracking.
"""

import json
import time
import sys
import subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.adapters.longmemeval_adapter import LongMemEvalAdapter
from evaluation.evaluators.longmemeval_evaluator import evaluate_response_locally, LongMemEvalEvaluator, OFFICIAL_TASKS
from evaluation.experiment_a.calibrated_answering_budgeted import answer_question_budgeted
from src.chunking import TurnAwareChunker
from src.chunking.graph_retrieval import TurnGraphBuilder, TurnGraphRetriever
from src.graph.store import SQLiteGraphStore


def get_gpu_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            text=True
        ).strip().split(",")
        return float(out[0].strip()), float(out[1].strip())
    except Exception:
        return 0.0, 0.0


def select_session_mmr_chunks(pool_candidates: list[dict[str, Any]], k: int = 5, lambda_penalty: float = 0.25) -> list[dict[str, Any]]:
    """Greedy Session-Aware MMR chunk selection from candidate pool."""
    selected = []
    selected_sids = set()
    remaining = list(pool_candidates)

    for _ in range(min(k, len(remaining))):
        best_idx = -1
        best_score = -999.0
        for i, c in enumerate(remaining):
            sid = c.get("session_id") or c.get("memory_id")
            penalty = lambda_penalty if sid in selected_sids else 0.0
            score = c.get("score", 0.0) - penalty
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0:
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            chosen_sid = chosen.get("session_id") or chosen.get("memory_id")
            selected_sids.add(chosen_sid)
    return selected


def run_experiment_c(
    dataset_path: str = "evaluation/data/longmemeval/longmemeval_s_cleaned.json",
    baseline_results_path: str = "evaluation/data/longmemeval/runs/full_baseline_config_d/results_config_D.jsonl",
    output_dir: str = "evaluation/data/longmemeval/runs/experiment_c_combined"
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load baseline records for paired analysis
    baseline_records = {}
    with open(baseline_results_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            baseline_records[r["question_id"]] = r

    adapter = LongMemEvalAdapter(dataset_path)
    instances, _ = adapter.load_and_validate()
    references = adapter.get_reference_data(instances)

    results_file = output_path / "results_experiment_c.jsonl"
    summary_file = output_path / "summary_experiment_c.json"

    completed_qids = set()
    existing_results = []
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                completed_qids.add(rec["question_id"])
                existing_results.append(rec)

    print(f"\n{'='*70}")
    print(f"Starting Experiment C (Combined Configuration) on Full LongMemEval-S")
    print(f"Total instances: {len(instances)} | Completed: {len(completed_qids)} | Remaining: {len(instances) - len(completed_qids)}")
    print(f"Output: {output_path}")
    print(f"{'='*70}\n")

    start_time = time.time()
    peak_vram = 0.0

    with open(results_file, "a", encoding="utf-8") as out_f:
        for idx, inst in enumerate(instances):
            if inst.question_id in completed_qids:
                continue

            q_start = time.time()
            vram_used, _ = get_gpu_vram()
            if vram_used > peak_vram:
                peak_vram = vram_used

            print(f"[{idx+1}/{len(instances)}] Processing {inst.question_id} ({inst.question_type})...", end=" ", flush=True)

            db_file = output_path / f"temp_{inst.question_id}.db"
            if db_file.exists():
                db_file.unlink()

            store = SQLiteGraphStore(db_file)
            memories = adapter.convert_instance_to_memories(inst)
            chunker = TurnAwareChunker(window_size=3, overlap=1)
            chunks = chunker.chunk_all_sessions(inst.sessions)
            builder = TurnGraphBuilder(store)
            builder.build_from_chunks(chunks)

            # 1. Session-Aware MMR Retrieval (k_pool=15 -> k=5)
            t_ret_start = time.time()
            retriever = TurnGraphRetriever(chunks=chunks, store=store, use_ontology=True)
            pool_candidates, _ = retriever.retrieve(inst.question, k=15, min_score=None)
            retrieved_chunks = select_session_mmr_chunks(pool_candidates, k=5, lambda_penalty=0.25)
            t_ret_ms = (time.time() - t_ret_start) * 1000

            # 2. Budgeted Calibrated Answering with num_ctx=8192
            t_ans_start = time.time()
            ans_res = answer_question_budgeted(
                query=inst.question,
                retrieved_chunks=retrieved_chunks,
                memories=memories,
                graph_store=store,
                enable_temporal=True,
                model="gpt-oss",
                num_ctx=8192,
                num_predict=1024
            )
            t_ans_ms = (time.time() - t_ans_start) * 1000

            store.close()
            if db_file.exists():
                try:
                    db_file.unlink()
                except Exception:
                    pass

            latency_ms = round((time.time() - q_start) * 1000, 2)
            hypothesis = ans_res.get("answer", "")
            status = ans_res.get("status", "answered")

            heur_score = evaluate_response_locally(
                task=inst.question_type,
                question=inst.question,
                gold_answer=inst.gold_answer,
                hypothesis=hypothesis,
                is_abstention=inst.is_abstention
            )

            # Diagnostic retrieval metrics
            target_sids = set()
            if inst.answer_session_ids:
                for s in inst.answer_session_ids:
                    s_clean = str(s).replace("sess_", "")
                    target_sids.add(s_clean)
                    target_sids.add(f"sess_{s_clean}")

            retrieved_sids = set()
            gold_rank = None
            for r_i, c in enumerate(retrieved_chunks):
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

            base_rec = baseline_records.get(inst.question_id, {})
            base_heur = base_rec.get("local_heuristic_score", False)

            rec = {
                "question_id": inst.question_id,
                "question_type": inst.question_type,
                "is_abstention": inst.is_abstention,
                "question": inst.question,
                "hypothesis": hypothesis,
                "status": status,
                "citations": [c.get("memory_id") for c in ans_res.get("citations", [])],
                "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved_chunks],
                "retrieved_evidence": [c.get("memory_id") for c in retrieved_chunks],
                "gold_retrieval_rank": gold_rank,
                "recall_at_1": r1,
                "recall_at_3": r3,
                "recall_at_5": r5,
                "evidence_coverage_ratio": cov_ratio,
                "local_heuristic_score": heur_score,
                "is_empty_response": ans_res.get("is_empty_response", False),
                "context_overflow": ans_res.get("context_overflow", False),
                "prompt_token_est": ans_res.get("prompt_token_est", 0),
                "latency_ms": latency_ms,
                "stage_latencies_ms": {
                    "retrieval": round(t_ret_ms, 2),
                    "answering": round(t_ans_ms, 2),
                    "total": round(latency_ms, 2)
                },
                # Paired baseline tracking
                "baseline_heuristic_score": base_heur,
                "baseline_hypothesis": base_rec.get("hypothesis", ""),
                "baseline_status": base_rec.get("status", ""),
                "paired_improvement": heur_score and not base_heur,
                "paired_regression": not heur_score and base_heur
            }

            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            existing_results.append(rec)
            completed_qids.add(inst.question_id)

            delta_str = "+1" if (heur_score and not base_heur) else ("-1" if (not heur_score and base_heur) else "=")
            print(f"Done ({latency_ms:.1f}ms) [{status}] heur={heur_score} (base={base_heur} {delta_str}) empty={rec['is_empty_response']}")

    # Evaluation summary
    evaluator = LongMemEvalEvaluator(use_local_heuristic=True)
    metrics = evaluator.evaluate(existing_results, references)

    # Retrieval diagnostics
    ans_records = [r for r in existing_results if not r["is_abstention"]]
    retrieval_diag = {
        "overall": {
            "recall_at_1": round(float(np.mean([1 if r["recall_at_1"] else 0 for r in ans_records])), 4),
            "recall_at_3": round(float(np.mean([1 if r["recall_at_3"] else 0 for r in ans_records])), 4),
            "recall_at_5": round(float(np.mean([1 if r["recall_at_5"] else 0 for r in ans_records])), 4),
            "mean_evidence_coverage_ratio": round(float(np.mean([r["evidence_coverage_ratio"] for r in ans_records])), 4),
            "total_answerable": len(ans_records)
        },
        "by_task": {}
    }
    for task in OFFICIAL_TASKS:
        task_recs = [r for r in ans_records if r["question_type"] == task]
        if task_recs:
            retrieval_diag["by_task"][task] = {
                "recall_at_1": round(float(np.mean([1 if r["recall_at_1"] else 0 for r in task_recs])), 4),
                "recall_at_3": round(float(np.mean([1 if r["recall_at_3"] else 0 for r in task_recs])), 4),
                "recall_at_5": round(float(np.mean([1 if r["recall_at_5"] else 0 for r in task_recs])), 4),
                "mean_evidence_coverage_ratio": round(float(np.mean([r["evidence_coverage_ratio"] for r in task_recs])), 4),
                "instances": len(task_recs)
            }

    # Statistical significance testing: Paired McNemar Test
    # Contingency table:
    #   b: Exp C correct, Base incorrect (wins)
    #   c: Exp C incorrect, Base correct (losses)
    wins = sum(1 for r in existing_results if r["local_heuristic_score"] and not r["baseline_heuristic_score"])
    losses = sum(1 for r in existing_results if not r["local_heuristic_score"] and r["baseline_heuristic_score"])
    ties_correct = sum(1 for r in existing_results if r["local_heuristic_score"] and r["baseline_heuristic_score"])
    ties_incorrect = sum(1 for r in existing_results if not r["local_heuristic_score"] and not r["baseline_heuristic_score"])

    # McNemar's test with continuity correction
    if (wins + losses) > 0:
        chi2 = (abs(wins - losses) - 1.0) ** 2 / (wins + losses)
        p_value = 1.0 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2, p_value = 0.0, 1.0

    # 95% Confidence Interval for Accuracy (Wilson score interval)
    n_total = len(existing_results)
    p_hat = metrics["official_metrics"]["overall_accuracy"]
    z = 1.96
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    half_width = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    ci_lower = max(0.0, round(center - half_width, 4))
    ci_upper = min(1.0, round(center + half_width, 4))

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_evaluated": len(existing_results),
        "peak_vram_mb": peak_vram,
        "metrics": metrics,
        "retrieval_diagnostics": retrieval_diag,
        "paired_statistical_comparison": {
            "baseline_accuracy": 0.192,
            "experiment_c_accuracy": metrics["official_metrics"]["overall_accuracy"],
            "accuracy_difference": round(metrics["official_metrics"]["overall_accuracy"] - 0.192, 4),
            "wins (Exp C=1, Base=0)": wins,
            "losses (Exp C=0, Base=1)": losses,
            "ties_correct (both=1)": ties_correct,
            "ties_incorrect (both=0)": ties_incorrect,
            "mcnemar_chi2": round(float(chi2), 4),
            "mcnemar_p_value": float(p_value),
            "statistically_significant": bool(p_value < 0.001),
            "wilson_95_ci": [ci_lower, ci_upper]
        },
        "empty_response_frequency": round(sum(1 for r in existing_results if r["is_empty_response"]) / len(existing_results), 4),
        "context_overflow_frequency": round(sum(1 for r in existing_results if r["context_overflow"]) / len(existing_results), 4),
        "latency": {
            "mean_ms": round(float(np.mean([r["latency_ms"] for r in existing_results])), 2),
            "min_ms": round(float(np.min([r["latency_ms"] for r in existing_results])), 2),
            "max_ms": round(float(np.max([r["latency_ms"] for r in existing_results])), 2),
            "total_eval_time_s": round(time.time() - start_time, 2)
        }
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Experiment C (Combined Configuration) Evaluation Completed!")
    print(f"Official Overall Accuracy: {metrics['official_metrics']['overall_accuracy']*100:.2f}% (Baseline: 19.20%) [Δ: +{(metrics['official_metrics']['overall_accuracy'] - 0.192)*100:.2f}%]")
    print(f"95% Wilson Confidence Interval: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"Paired Wins / Losses: {wins} wins vs {losses} losses (McNemar p = {p_value:.2e})")
    print(f"Empty Response Frequency: {summary['empty_response_frequency']*100:.2f}% (Baseline: 75.80%)")
    print(f"Recall@5: {retrieval_diag['overall']['recall_at_5']*100:.2f}% | Evidence Coverage: {retrieval_diag['overall']['mean_evidence_coverage_ratio']*100:.2f}%")
    print(f"Peak VRAM: {peak_vram:.1f} MB | Mean Latency: {summary['latency']['mean_ms']:.1f} ms")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    run_experiment_c()

