"""
Deterministic stratified sampler for IVARA Evaluation Campaign v1 Pilot.
Samples 60 questions: exactly 10 questions per task across all 6 tasks.
Stratified by task and answerability (8 answerable + 2 abstention where available).
"""

import json
import random
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "evaluation" / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
PILOT_DIR = PROJECT_ROOT / "evaluation" / "data" / "longmemeval" / "runs" / "campaign_v1_pilot"

def generate_pilot_sample():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group questions by task and answerability
    groups = defaultdict(lambda: {"ans": [], "abs": []})
    for d in data:
        t = d["question_type"]
        is_abs = ("_abs" in d["question_id"]) or (len(d.get("answer_session_ids", [])) == 0)
        qid = d["question_id"]
        if is_abs:
            groups[t]["abs"].append(qid)
        else:
            groups[t]["ans"].append(qid)

    # Deterministic sampling with seed 42
    rng = random.Random(42)
    selected_qids = []
    task_breakdown = {}

    for t in sorted(groups.keys()):
        ans_pool = sorted(groups[t]["ans"])
        abs_pool = sorted(groups[t]["abs"])

        # Determine counts
        if len(abs_pool) >= 2:
            n_abs = 2
            n_ans = 8
        else:
            n_abs = len(abs_pool)
            n_ans = 10 - n_abs

        # Deterministic sample
        sampled_ans = rng.sample(ans_pool, n_ans)
        sampled_abs = rng.sample(abs_pool, n_abs) if n_abs > 0 else []

        task_selected = sorted(sampled_ans + sampled_abs)
        selected_qids.extend(task_selected)
        task_breakdown[t] = {
            "total": len(task_selected),
            "answerable": len(sampled_ans),
            "abstention": len(sampled_abs),
            "qids": task_selected
        }

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_path": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "sampling_method": "deterministic_stratified",
        "random_seed": 42,
        "total_questions": len(selected_qids),
        "total_tasks": len(task_breakdown),
        "task_breakdown": task_breakdown,
        "selected_qids": selected_qids
    }

    manifest_path = PILOT_DIR / "pilot_sample_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated stratified pilot sample with {len(selected_qids)} questions.")
    print(f"Saved manifest to {manifest_path}")
    for t, info in task_breakdown.items():
        print(f"  {t}: {info['total']} (ans: {info['answerable']}, abs: {info['abstention']})")

    return selected_qids

if __name__ == "__main__":
    generate_pilot_sample()
