"""
Runner for Experiment A: Context Budgeting Evaluation.

Evaluates:
  - 198 Cat 5 questions (where baseline suffered empty-response context overflow)
  - 30 Abstention questions (_abs)
  - 30 Answerable control questions
Total: 258 target questions.

Measures:
  - Empty response frequency
  - Context overflow frequency
  - Local heuristic accuracy
  - Abstention accuracy
  - Latency per question
  - Peak VRAM overhead
"""

import json
import time
import sys
import subprocess
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.adapters.longmemeval_adapter import LongMemEvalAdapter
from evaluation.evaluators.longmemeval_evaluator import evaluate_response_locally, LongMemEvalEvaluator
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


def run_experiment_a(
    dataset_path: str = "evaluation/data/longmemeval/longmemeval_s_cleaned.json",
    baseline_results_path: str = "evaluation/data/longmemeval/runs/full_baseline_config_d/results_config_D.jsonl",
    output_dir: str = "evaluation/data/longmemeval/runs/experiment_a_context_budget"
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load baseline records to identify the 198 Cat 5 questions and controls
    baseline_records = {}
    with open(baseline_results_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            baseline_records[r["question_id"]] = r

    cat5_qids = []
    abs_qids = []
    ctrl_qids = []

    for qid, r in baseline_records.items():
        is_abs = r.get("is_abstention", False) or "_abs" in qid
        hyp = r.get("hypothesis", "")
        gold_rank = r.get("gold_retrieval_rank")
        cov = r.get("evidence_coverage_ratio", 1.0)

        if is_abs:
            abs_qids.append(qid)
        elif "Could not reliably synthesize an answer" in hyp and gold_rank is not None and cov == 1.0:
            cat5_qids.append(qid)
        elif not is_abs and len(ctrl_qids) < 30 and r.get("local_heuristic_score", False):
            ctrl_qids.append(qid)

    print(f"Target Selection for Experiment A:")
    print(f"  Cat 5 (Context Overflow): {len(cat5_qids)}")
    print(f"  Abstentions: {len(abs_qids)}")
    print(f"  Controls: {len(ctrl_qids)}")
    target_qids = set(cat5_qids + abs_qids + ctrl_qids)
    print(f"  Total evaluation set: {len(target_qids)}")

    adapter = LongMemEvalAdapter(dataset_path)
    all_instances, _ = adapter.load_and_validate()
    instances = [inst for inst in all_instances if inst.question_id in target_qids]
    references = adapter.get_reference_data(instances)

    results_file = output_path / "results_experiment_a.jsonl"
    summary_file = output_path / "summary_experiment_a.json"

    completed_qids = set()
    existing_results = []
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                completed_qids.add(rec["question_id"])
                existing_results.append(rec)

    print(f"Already completed: {len(completed_qids)} | Remaining: {len(instances) - len(completed_qids)}")

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

            retriever = TurnGraphRetriever(chunks=chunks, store=store, use_ontology=True)
            retrieved_chunks, _ = retriever.retrieve(inst.question, k=5)

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

            rec = {
                "question_id": inst.question_id,
                "question_type": inst.question_type,
                "is_abstention": inst.is_abstention,
                "question": inst.question,
                "hypothesis": hypothesis,
                "status": status,
                "citations": [c.get("memory_id") for c in ans_res.get("citations", [])],
                "local_heuristic_score": heur_score,
                "is_empty_response": ans_res.get("is_empty_response", False),
                "context_overflow": ans_res.get("context_overflow", False),
                "prompt_token_est": ans_res.get("prompt_token_est", 0),
                "latency_ms": latency_ms,
                "baseline_heuristic_score": baseline_records[inst.question_id].get("local_heuristic_score", False),
                "baseline_hypothesis": baseline_records[inst.question_id].get("hypothesis", "")
            }

            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            existing_results.append(rec)
            completed_qids.add(inst.question_id)

            print(f"Done ({latency_ms:.1f}ms) [{status}] heur={heur_score} (base={rec['baseline_heuristic_score']}) empty={rec['is_empty_response']}")

    # Evaluation summary
    evaluator = LongMemEvalEvaluator(use_local_heuristic=True)
    metrics = evaluator.evaluate(existing_results, references)

    # Cat 5 specific comparison
    cat5_results = [r for r in existing_results if r["question_id"] in set(cat5_qids)]
    cat5_base_acc = sum(1 for r in cat5_results if r["baseline_heuristic_score"]) / len(cat5_results) if cat5_results else 0.0
    cat5_new_acc = sum(1 for r in cat5_results if r["local_heuristic_score"]) / len(cat5_results) if cat5_results else 0.0
    cat5_empty_base = sum(1 for r in cat5_results if "Could not reliably synthesize" in r["baseline_hypothesis"]) / len(cat5_results) if cat5_results else 0.0
    cat5_empty_new = sum(1 for r in cat5_results if r["is_empty_response"]) / len(cat5_results) if cat5_results else 0.0

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_evaluated": len(existing_results),
        "peak_vram_mb": peak_vram,
        "metrics": metrics,
        "cat5_subset_comparison": {
            "total_cat5_questions": len(cat5_results),
            "baseline_empty_frequency": round(cat5_empty_base, 4),
            "experiment_a_empty_frequency": round(cat5_empty_new, 4),
            "baseline_accuracy": round(cat5_base_acc, 4),
            "experiment_a_accuracy": round(cat5_new_acc, 4),
            "accuracy_recovery": round(cat5_new_acc - cat5_base_acc, 4)
        },
        "overall_empty_response_frequency": round(sum(1 for r in existing_results if r["is_empty_response"]) / len(existing_results), 4) if existing_results else 0.0,
        "overall_context_overflow_frequency": round(sum(1 for r in existing_results if r["context_overflow"]) / len(existing_results), 4) if existing_results else 0.0,
        "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in existing_results])), 2) if existing_results else 0.0,
        "total_eval_time_s": round(time.time() - start_time, 2)
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Experiment A Evaluation Completed!")
    print(f"Cat 5 Empty Response Frequency: {cat5_empty_base*100:.1f}% -> {cat5_empty_new*100:.1f}%")
    print(f"Cat 5 Accuracy: {cat5_base_acc*100:.1f}% -> {cat5_new_acc*100:.1f}% (+{(cat5_new_acc - cat5_base_acc)*100:.1f}%)")
    print(f"Peak VRAM: {peak_vram:.1f} MB")
    print(f"Overall Accuracy: {metrics['official_metrics']['overall_accuracy']*100:.2f}%")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    run_experiment_a()

