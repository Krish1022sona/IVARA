"""
Step 8: Metric Recomputation and Audit.
Recomputes metrics from raw JSONL files across all 4 configurations:
1. Original strict heuristic (as reported in Step 7).
2. Audited rule evaluator (fixing contraction brittleness and written number matching).
Saves side-by-side comparisons to JSON and CSV.
"""

import json
import math
import re
from pathlib import Path
from typing import Any

ROOT_DIR = Path("evaluation/data/longmemeval/runs/campaign_v1_pilot")
STEP8_DIR = Path("evaluation/step8")
STEP8_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("Config-A", "config_a", "CONFIG-A", "Dense + BM25 baseline"),
    ("Config-B", "config_b", "CONFIG-B", "Baseline + ontology guidance"),
    ("Config-C", "config_c", "CONFIG-C", "Ontology + bounded graph expansion"),
    ("Config-D", "config_d", "CONFIG-D", "Full system with temporal reasoning")
]

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50
}

def extract_numbers(text: str) -> list[int]:
    """Extract both numeric digits and common spelled-out numbers."""
    nums = [int(n) for n in re.findall(r'\b\d+\b', text)]
    text_lower = text.lower()
    for word, val in NUMBER_WORDS.items():
        if re.search(rf'\b{word}\b', text_lower):
            if val not in nums:
                nums.append(val)
    return nums

def evaluate_audited(
    task: str,
    question: str,
    gold_answer: str,
    hypothesis: str,
    is_abstention: bool = False
) -> bool:
    """
    Audited deterministic evaluation rule fixing known flaws:
    1. Robust abstention markers (including uncontracted 'do not have sufficient', etc.)
    2. Spelled-out numbers in temporal reasoning.
    3. Direct containment & token overlap.
    """
    hyp_lower = str(hypothesis).lower().strip()
    gold_lower = str(gold_answer).lower().strip()

    if is_abstention:
        abstention_markers = [
            "insufficient",
            "cannot answer",
            "cannot determine",
            "don't have sufficient",
            "do not have sufficient",
            "don't have enough",
            "do not have enough",
            "not mentioned",
            "no information",
            "not stated",
            "unknown",
            "unanswerable",
            "not available",
            "does not contain",
            "not recorded",
            "no recorded"
        ]
        return any(m in hyp_lower for m in abstention_markers)

    if not hyp_lower:
        return False

    # Direct containment
    if gold_lower in hyp_lower:
        return True

    # Temporal reasoning numeric check
    if task == "temporal-reasoning":
        gold_nums = extract_numbers(gold_lower)
        hyp_nums = extract_numbers(hyp_lower)
        if gold_nums:
            if not hyp_nums:
                return False
            matches = [any(abs(gn - hn) <= 1 for hn in hyp_nums) for gn in gold_nums]
            return all(matches)

    # Token overlap check (75%+ of content words)
    gold_tokens = [t for t in re.findall(r'\b\w+\b', gold_lower) if len(t) > 2]
    if gold_tokens:
        matched = sum(1 for t in gold_tokens if t in hyp_lower)
        if matched / len(gold_tokens) >= 0.75:
            return True

    return False

def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - spread), min(1.0, center + spread))

# Process configurations
results_original = {}
results_audited = {}

dataset_by_qid = {}
with open("evaluation/data/longmemeval/longmemeval_s_cleaned.json", "r", encoding="utf-8") as f:
    for inst in json.load(f):
        dataset_by_qid[inst["question_id"]] = inst

for cfg_name, folder, prefix, desc in CONFIGS:
    res_path = ROOT_DIR / folder / f"results_config_{prefix}.jsonl"
    records = []
    with open(res_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n_total = len(records)
    ans_records = [r for r in records if not r["is_abstention"]]
    abs_records = [r for r in records if r["is_abstention"]]

    # 1. Original heuristic metrics
    orig_total_correct = sum(1 for r in records if r.get("local_heuristic_score") is True)
    orig_ans_correct = sum(1 for r in ans_records if r.get("local_heuristic_score") is True)
    orig_abs_correct = sum(1 for r in abs_records if r.get("local_heuristic_score") is True)

    orig_acc_total = (orig_total_correct / n_total) * 100 if n_total > 0 else 0.0
    orig_acc_ans = (orig_ans_correct / len(ans_records)) * 100 if ans_records else 0.0
    orig_ci_total = [round(c * 100, 2) for c in compute_wilson_ci(orig_total_correct, n_total)]
    orig_ci_ans = [round(c * 100, 2) for c in compute_wilson_ci(orig_ans_correct, len(ans_records))]

    # 2. Audited rule metrics
    aud_scores = {}
    for r in records:
        qid = r["question_id"]
        gold_inst = dataset_by_qid.get(qid, {})
        gold_ans = gold_inst.get("answer", "")
        is_abs = r["is_abstention"]
        score = evaluate_audited(
            task=r["question_type"],
            question=r["question"],
            gold_answer=gold_ans,
            hypothesis=r["hypothesis"],
            is_abstention=is_abs
        )
        aud_scores[qid] = score

    aud_total_correct = sum(1 for q, s in aud_scores.items() if s is True)
    aud_ans_correct = sum(1 for r in ans_records if aud_scores[r["question_id"]] is True)
    aud_abs_correct = sum(1 for r in abs_records if aud_scores[r["question_id"]] is True)

    aud_acc_total = (aud_total_correct / n_total) * 100 if n_total > 0 else 0.0
    aud_acc_ans = (aud_ans_correct / len(ans_records)) * 100 if ans_records else 0.0
    aud_ci_total = [round(c * 100, 2) for c in compute_wilson_ci(aud_total_correct, n_total)]
    aud_ci_ans = [round(c * 100, 2) for c in compute_wilson_ci(aud_ans_correct, len(ans_records))]

    # Pipeline status-based abstention metrics
    status_abs_correct = sum(1 for r in abs_records if r["status"] == "insufficient_evidence")
    status_false_abs = sum(1 for r in ans_records if r["status"] == "insufficient_evidence")
    status_total_abs = sum(1 for r in records if r["status"] == "insufficient_evidence")
    abs_rec = (status_abs_correct / len(abs_records)) * 100 if abs_records else 0.0
    abs_prec = (status_abs_correct / status_total_abs) * 100 if status_total_abs > 0 else 0.0
    false_abs_rate = (status_false_abs / len(ans_records)) * 100 if ans_records else 0.0

    # Retrieval metrics
    rec1 = (sum(1 for r in ans_records if r.get("gold_retrieval_rank") == 1) / len(ans_records)) * 100
    rec3 = (sum(1 for r in ans_records if r.get("gold_retrieval_rank") and r["gold_retrieval_rank"] <= 3) / len(ans_records)) * 100
    rec5 = (sum(1 for r in ans_records if r.get("gold_retrieval_rank") and r["gold_retrieval_rank"] <= 5) / len(ans_records)) * 100

    results_original[cfg_name] = {
        "description": desc,
        "total_instances": n_total,
        "overall_accuracy": round(orig_acc_total, 2),
        "overall_accuracy_ci95": orig_ci_total,
        "answerable_accuracy": round(orig_acc_ans, 2),
        "answerable_accuracy_ci95": orig_ci_ans,
        "abstentions_scored_correct": orig_abs_correct,
        "total_abstentions": len(abs_records)
    }

    results_audited[cfg_name] = {
        "description": desc,
        "total_instances": n_total,
        "overall_accuracy": round(aud_acc_total, 2),
        "overall_accuracy_ci95": aud_ci_total,
        "answerable_accuracy": round(aud_acc_ans, 2),
        "answerable_accuracy_ci95": aud_ci_ans,
        "abstentions_scored_correct": aud_abs_correct,
        "total_abstentions": len(abs_records),
        "pipeline_abstention_recall": round(abs_rec, 2),
        "pipeline_abstention_precision": round(abs_prec, 2),
        "pipeline_false_abstention_rate": round(false_abs_rate, 2),
        "recall_at_1": round(rec1, 2),
        "recall_at_3": round(rec3, 2),
        "recall_at_5": round(rec5, 2)
    }

summary_comparison = {
    "original_strict_heuristic": results_original,
    "audited_rule_evaluator": results_audited
}

with open(STEP8_DIR / "recomputed_metrics_audit.json", "w", encoding="utf-8") as f:
    json.dump(summary_comparison, f, indent=2)

# Write summary CSV
with open(STEP8_DIR / "recomputed_metrics_audit.csv", "w", encoding="utf-8") as f:
    headers = [
        "Config", "Description", 
        "Orig_Overall_Acc", "Orig_Ans_Acc", "Orig_Abs_Correct",
        "Audited_Overall_Acc", "Audited_Ans_Acc", "Audited_Abs_Correct",
        "Recall@1", "Recall@3", "Recall@5",
        "Pipeline_Abs_Recall", "Pipeline_Abs_Prec"
    ]
    f.write(",".join(headers) + "\n")
    for cfg_name, _, _, desc in CONFIGS:
        o = results_original[cfg_name]
        a = results_audited[cfg_name]
        row = [
            cfg_name,
            f'"{desc}"',
            f"{o['overall_accuracy']:.2f}",
            f"{o['answerable_accuracy']:.2f}",
            f"{o['abstentions_scored_correct']}/{o['total_abstentions']}",
            f"{a['overall_accuracy']:.2f}",
            f"{a['answerable_accuracy']:.2f}",
            f"{a['abstentions_scored_correct']}/{a['total_abstentions']}",
            f"{a['recall_at_1']:.2f}",
            f"{a['recall_at_3']:.2f}",
            f"{a['recall_at_5']:.2f}",
            f"{a['pipeline_abstention_recall']:.2f}",
            f"{a['pipeline_abstention_precision']:.2f}"
        ]
        f.write(",".join(row) + "\n")

print(f"Saved audited recomputed metrics to: {STEP8_DIR / 'recomputed_metrics_audit.json'}")
print(f"Saved audited recomputed CSV to: {STEP8_DIR / 'recomputed_metrics_audit.csv'}")
