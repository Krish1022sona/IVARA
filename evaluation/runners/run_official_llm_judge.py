"""
Official LongMemEval LLM-as-a-Judge Evaluator.
Executes the verbatim official judging protocol from LongMemEval (xiaowu0162/LongMemEval):
- Constructs the official prompt per task category (single-session-user, single-session-assistant,
  multi-session, temporal-reasoning, knowledge-update, single-session-preference, abstention).
- Queries the local judge LLM at temperature=0.0.
- Preserves raw judge outputs and per-question decisions.
- Computes overall accuracy, task-averaged accuracy, answerable accuracy, and abstention accuracy.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import ollama

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.adapters.longmemeval_adapter import LongMemEvalAdapter
from evaluation.evaluators.longmemeval_evaluator import OFFICIAL_TASKS, get_anscheck_prompt

def call_ollama_judge(prompt: str, model: str = "qwen2.5:14b", port: int = 11434) -> str:
    """Call local Ollama model for judgment."""
    client = ollama.Client(host=f"http://127.0.0.1:{port}")
    response = client.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": 0.0,
            "num_predict": 30,
            "stop": ["\n", "Question:", "Is the model response correct?"]
        }
    )
    return response.get("response", "").strip()

def run_evaluation(
    predictions_path: Path,
    dataset_path: Path,
    output_dir: Path,
    judge_model: str = "qwen2.5:14b"
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_jsonl_path = output_dir / f"official_judge_decisions_{judge_model.replace(':', '_')}.jsonl"
    summary_json_path = output_dir / f"official_judge_summary_{judge_model.replace(':', '_')}.json"

    # Load dataset references
    adapter = LongMemEvalAdapter(dataset_path)
    instances, _ = adapter.load_and_validate()
    ref_map = {inst.question_id: inst for inst in instances}

    # Load predictions
    predictions = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    print(f"=== Starting Official LLM-as-a-Judge Evaluation ===")
    print(f"Judge Model: {judge_model}")
    print(f"Predictions File: {predictions_path} ({len(predictions)} records)")
    print(f"Dataset File: {dataset_path} ({len(ref_map)} instances)")
    print(f"Output Directory: {output_dir}")

    # Check for resuming
    evaluated_ids = set()
    scored_records = []
    if detailed_jsonl_path.exists():
        with open(detailed_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    scored_records.append(item)
                    evaluated_ids.add(item["question_id"])
        print(f"Resuming: found {len(evaluated_ids)} already evaluated instances.")

    out_file = open(detailed_jsonl_path, "a", encoding="utf-8")

    start_time = time.time()
    for idx, pred in enumerate(predictions):
        qid = pred["question_id"]
        if qid in evaluated_ids:
            continue

        inst = ref_map.get(qid)
        if not inst:
            print(f"Warning: Question ID {qid} not found in reference dataset.")
            continue

        task = inst.question_type
        question = inst.question
        gold_answer = inst.gold_answer
        hypothesis = pred.get("hypothesis", "")
        is_abs = inst.is_abstention or "_abs" in qid

        # Build official LongMemEval prompt
        prompt = get_anscheck_prompt(
            task=task,
            question=question,
            answer=gold_answer,
            response=hypothesis,
            abstention=is_abs
        )

        # Call judge
        t0 = time.time()
        raw_output = call_ollama_judge(prompt, model=judge_model)
        judge_latency_ms = (time.time() - t0) * 1000.0

        # Official LongMemEval check: 'yes' in response.lower()
        # Must be careful not to trigger on words like 'yesterday' or 'eyes'
        raw_lower = raw_output.lower()
        is_correct = bool(re.search(r'\byes\b', raw_lower))

        record = {
            "question_id": qid,
            "task": task,
            "is_abstention": is_abs,
            "question": question,
            "gold_answer": gold_answer,
            "hypothesis": hypothesis,
            "judge_model": judge_model,
            "raw_judge_output": raw_output,
            "is_correct": is_correct,
            "judge_latency_ms": round(judge_latency_ms, 2)
        }

        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        scored_records.append(record)
        evaluated_ids.add(qid)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(predictions):
            print(f"[{idx + 1}/{len(predictions)}] QID: {qid} | Correct: {is_correct} | Raw: '{raw_output}' ({judge_latency_ms:.1f}ms)")

    out_file.close()
    total_time = time.time() - start_time
    print(f"Judging complete in {total_time:.1f}s.")

    # Compute official metrics
    type2acc = {t: [] for t in OFFICIAL_TASKS}
    all_acc = []
    abstention_acc = []
    ans_all_acc = []
    ans_type2acc = {t: [] for t in OFFICIAL_TASKS}

    for rec in scored_records:
        t = rec["task"]
        is_abs = rec["is_abstention"]
        val = 1 if rec["is_correct"] else 0

        all_acc.append(val)
        if t in type2acc:
            type2acc[t].append(val)

        if is_abs:
            abstention_acc.append(val)
        else:
            ans_all_acc.append(val)
            if t in ans_type2acc:
                ans_type2acc[t].append(val)

    task_means = [float(np.mean(type2acc[t])) for t in OFFICIAL_TASKS if len(type2acc[t]) > 0]
    ans_task_means = [float(np.mean(ans_type2acc[t])) for t in OFFICIAL_TASKS if len(ans_type2acc[t]) > 0]

    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge_model": judge_model,
        "predictions_file": str(predictions_path),
        "total_evaluated": len(scored_records),
        "overall_accuracy": round(float(np.mean(all_acc)), 4),
        "task_averaged_accuracy": round(float(np.mean(task_means)), 4),
        "abstention_accuracy": round(float(np.mean(abstention_acc)), 4),
        "answerable_accuracy": round(float(np.mean(ans_all_acc)), 4),
        "answerable_task_averaged_accuracy": round(float(np.mean(ans_task_means)), 4),
        "counts": {
            "total": len(all_acc),
            "correct": sum(all_acc),
            "abstention_total": len(abstention_acc),
            "abstention_correct": sum(abstention_acc),
            "answerable_total": len(ans_all_acc),
            "answerable_correct": sum(ans_all_acc)
        },
        "by_task": {
            t: {
                "accuracy": round(float(np.mean(type2acc[t])), 4) if len(type2acc[t]) > 0 else 0.0,
                "correct": sum(type2acc[t]),
                "total": len(type2acc[t])
            }
            for t in OFFICIAL_TASKS
        }
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("OFFICIAL LLM JUDGE EVALUATION SUMMARY")
    print("=" * 50)
    print(json.dumps(summary, indent=2))
    print(f"\nDetailed decisions saved to: {detailed_jsonl_path}")
    print(f"Summary saved to: {summary_json_path}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official LongMemEval LLM-as-a-Judge Runner")
    parser.add_argument("--predictions", required=True, help="Path to predictions .jsonl file")
    parser.add_argument("--dataset", default="evaluation/data/longmemeval/longmemeval_s_cleaned.json", help="Path to reference dataset")
    parser.add_argument("--output-dir", required=True, help="Directory to store judge output files")
    parser.add_argument("--judge-model", default="qwen2.5:14b", help="Ollama model to use as judge")
    args = parser.parse_args()

    run_evaluation(
        predictions_path=Path(args.predictions),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        judge_model=args.judge_model
    )

