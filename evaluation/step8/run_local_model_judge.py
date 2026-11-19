"""
Step 8: Independent Local Model Judge Evaluation.
Uses VladimirGav/gemma4-26b-16GB-VRAM:latest (25.2B parameters) to provide
independent, model-based evaluation of predictions across all 4 configurations.
Distinct from the answer-generation model (gpt-oss), avoiding correlated bias.
"""

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import ollama

ROOT_DIR = Path("evaluation/data/longmemeval/runs/campaign_v1_pilot")
STEP8_DIR = Path("evaluation/step8")
STEP8_DIR.mkdir(parents=True, exist_ok=True)

JUDGE_MODEL = "VladimirGav/gemma4-26b-16GB-VRAM:latest"
OUTPUT_PATH = STEP8_DIR / "local_judge_results.json"

CONFIGS = [
    ("Config-A", "config_a", "CONFIG-A", "Dense + BM25 baseline"),
    ("Config-B", "config_b", "CONFIG-B", "Baseline + ontology guidance"),
    ("Config-C", "config_c", "CONFIG-C", "Ontology + bounded graph expansion"),
    ("Config-D", "config_d", "CONFIG-D", "Full system with temporal reasoning")
]

JUDGE_RUBRIC = """You are a strict, objective evaluation judge assessing whether an AI assistant's response accurately answers a user question according to a reference gold answer.

User Question: {question}
Reference Gold Answer: {gold_answer}
Candidate Prediction: {hypothesis}
Is Abstention Required: {is_abstention}

Evaluation Rules:
1. For answerable questions (Is Abstention Required: False):
   - Return is_correct: true if the prediction accurately captures the factual information stated in the gold answer, including semantic paraphrases, equivalent dates/durations, and spelled-out numbers vs digits.
   - Return is_correct: false if the prediction is factually wrong, contradicts the evidence, fails to provide the requested information, or improperly abstains.
2. For unanswerable questions (Is Abstention Required: True):
   - Return is_correct: true if the candidate correctly identifies that the information is missing, unrecorded, or unanswerable from memory.
   - Return is_correct: false if the candidate invents an answer or provides unsupported factual claims.

Respond ONLY with a valid JSON object matching this schema:
{{
  "is_correct": true,
  "reasoning": "A concise one-sentence factual justification."
}}
"""

def compute_wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - spread), min(1.0, center + spread))

def main():
    print(f"=== Starting Local Model Judge Evaluation ===")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Configurations: {[c[0] for c in CONFIGS]}")

    # Load dataset gold answers
    dataset_path = Path("evaluation/data/longmemeval/longmemeval_s_cleaned.json")
    dataset_by_qid = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for inst in json.load(f):
            dataset_by_qid[inst["question_id"]] = inst

    # Load existing judge results if resuming
    existing_results = {}
    if OUTPUT_PATH.exists():
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            print(f"Loaded existing results from: {OUTPUT_PATH}")
        except Exception:
            existing_results = {}

    judge_data = existing_results.get("predictions_by_config", {})

    total_start = time.time()
    for cfg_name, folder, prefix, desc in CONFIGS:
        print(f"\n--- Judging {cfg_name} ({desc}) ---")
        res_path = ROOT_DIR / folder / f"results_config_{prefix}.jsonl"
        records = []
        with open(res_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if cfg_name not in judge_data:
            judge_data[cfg_name] = {}

        cfg_judgments = judge_data[cfg_name]
        
        for idx, r in enumerate(records, 1):
            qid = r["question_id"]
            if qid in cfg_judgments:
                continue

            gold_inst = dataset_by_qid.get(qid, {})
            gold_ans = str(gold_inst.get("answer", ""))
            is_abs = r["is_abstention"]
            question = r["question"]
            hyp = r["hypothesis"]

            prompt = JUDGE_RUBRIC.format(
                question=question,
                gold_answer=gold_ans,
                hypothesis=hyp,
                is_abstention=is_abs
            )

            t0 = time.time()
            is_correct = False
            reasoning = "Judge parsing error"
            raw_content = ""
            judge_failed = False

            try:
                res = ollama.chat(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    format="json",
                    options={"temperature": 0}
                )
                raw_content = res.message.content
                parsed = json.loads(raw_content)
                is_correct = bool(parsed.get("is_correct", False))
                reasoning = str(parsed.get("reasoning", ""))
            except Exception as e:
                judge_failed = True
                reasoning = f"Exception during judging: {e}"

            lat_ms = (time.time() - t0) * 1000

            cfg_judgments[qid] = {
                "question_id": qid,
                "question_type": r.get("question_type"),
                "is_abstention": is_abs,
                "is_correct": is_correct,
                "reasoning": reasoning,
                "judge_failed": judge_failed,
                "raw_response": raw_content,
                "latency_ms": round(lat_ms, 2)
            }

            status_str = "CORRECT" if is_correct else "INCORRECT"
            print(f"[{idx}/{len(records)}] {qid:12s} ({r.get('question_type')}) -> {status_str} ({lat_ms:.0f}ms)")

            # Save incrementally after each prediction
            summary = {
                "judge_model": JUDGE_MODEL,
                "rubric": JUDGE_RUBRIC,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "predictions_by_config": judge_data
            }
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

    # Compute overall metrics under local model judge
    metrics_by_config = {}
    for cfg_name, _, _, desc in CONFIGS:
        cfg_j = judge_data.get(cfg_name, {})
        n_total = len(cfg_j)
        ans_j = [v for v in cfg_j.values() if not v["is_abstention"]]
        abs_j = [v for v in cfg_j.values() if v["is_abstention"]]

        n_correct = sum(1 for v in cfg_j.values() if v["is_correct"])
        n_ans_correct = sum(1 for v in ans_j if v["is_correct"])
        n_abs_correct = sum(1 for v in abs_j if v["is_correct"])
        n_failures = sum(1 for v in cfg_j.values() if v.get("judge_failed"))

        acc_overall = (n_correct / n_total) * 100 if n_total > 0 else 0.0
        acc_ans = (n_ans_correct / len(ans_j)) * 100 if ans_j else 0.0
        acc_abs = (n_abs_correct / len(abs_j)) * 100 if abs_j else 0.0

        ci_overall = [round(c * 100, 2) for c in compute_wilson_ci(n_correct, n_total)]
        ci_ans = [round(c * 100, 2) for c in compute_wilson_ci(n_ans_correct, len(ans_j))]

        metrics_by_config[cfg_name] = {
            "description": desc,
            "total_instances": n_total,
            "judge_failures": n_failures,
            "overall_accuracy": round(acc_overall, 2),
            "overall_accuracy_ci95": ci_overall,
            "answerable_accuracy": round(acc_ans, 2),
            "answerable_accuracy_ci95": ci_ans,
            "abstention_accuracy": round(acc_abs, 2),
            "abstentions_correct": n_abs_correct,
            "total_abstentions": len(abs_j)
        }

    # Paired comparisons under local model judge
    pairs = [
        ("Config-B", "Config-A", "Ontology vs Baseline"),
        ("Config-C", "Config-B", "Bounded Graph vs Ontology"),
        ("Config-D", "Config-C", "Full System + Temporal vs Bounded Graph"),
        ("Config-D", "Config-A", "Full System vs Baseline"),
    ]
    paired_comparisons = {}
    for c_new, c_base, p_desc in pairs:
        j_new = judge_data.get(c_new, {})
        j_base = judge_data.get(c_base, {})
        wins, losses, ties = 0, 0, 0
        diffs = []
        for qid, v_new in j_new.items():
            if qid not in j_base:
                continue
            s_n = 1 if v_new["is_correct"] else 0
            s_b = 1 if j_base[qid]["is_correct"] else 0
            diffs.append(s_n - s_b)
            if s_n > s_b:
                wins += 1
            elif s_n < s_b:
                losses += 1
            else:
                ties += 1

        net_delta = (sum(diffs) / len(diffs)) * 100 if diffs else 0.0
        paired_comparisons[f"{c_new}_vs_{c_base}"] = {
            "comparison": p_desc,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "net_accuracy_delta_pp": round(net_delta, 2)
        }

    final_output = {
        "evaluation_mode": "local_model_diagnostic",
        "official_benchmark_status": "diagnostic_only_not_official_longmemeval",
        "judge_model": JUDGE_MODEL,
        "rubric": JUDGE_RUBRIC,
        "total_runtime_s": round(time.time() - total_start, 2),
        "metrics_by_config": metrics_by_config,
        "paired_comparisons": paired_comparisons,
        "predictions_by_config": judge_data
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n=== Local Model Judge Complete ===")
    print(f"Results saved to: {OUTPUT_PATH}")
    for c, m in metrics_by_config.items():
        print(f"{c:10s} | Overall: {m['overall_accuracy']:5.2f}% | Answerable: {m['answerable_accuracy']:5.2f}% | Abstention: {m['abstention_accuracy']:5.2f}%")

if __name__ == "__main__":
    main()
