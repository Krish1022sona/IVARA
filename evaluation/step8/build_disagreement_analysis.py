"""
Step 8: Comprehensive Paired Failure and Disagreement Analysis.
Compares A vs B, B vs C, C vs D, and A vs D across all 60 pilot questions.
Inspects the full disagreement set and outputs machine-readable JSON.
"""

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path("evaluation/data/longmemeval/runs/campaign_v1_pilot")
STEP8_DIR = Path("evaluation/step8")
STEP8_DIR.mkdir(parents=True, exist_ok=True)

# Load raw records
def load_records(config_folder: str, filename_prefix: str) -> dict[str, dict[str, Any]]:
    path = ROOT_DIR / config_folder / f"results_config_{filename_prefix}.jsonl"
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                records[data["question_id"]] = data
    return records

res_a = load_records("config_a", "CONFIG-A")
res_b = load_records("config_b", "CONFIG-B")
res_c = load_records("config_c", "CONFIG-C")
res_d = load_records("config_d", "CONFIG-D")

# Also load gold dataset to get gold answers and notes
dataset_path = Path("evaluation/data/longmemeval/longmemeval_s_cleaned.json")
dataset_by_qid = {}
with open(dataset_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    for inst in raw_data:
        dataset_by_qid[inst["question_id"]] = inst

pairs = [
    ("Config-B", "Config-A", res_b, res_a, "Ontology vs Baseline"),
    ("Config-C", "Config-B", res_c, res_b, "Bounded Graph vs Ontology"),
    ("Config-D", "Config-C", res_d, res_c, "Full System + Temporal vs Bounded Graph"),
    ("Config-D", "Config-A", res_d, res_a, "Full System vs Baseline"),
]

disagreement_summary = {}

for name_new, name_old, data_new, data_old, pair_desc in pairs:
    pair_disagreements = []
    for qid in data_new:
        if qid not in data_old:
            continue
        r_new = data_new[qid]
        r_old = data_old[qid]
        gold_inst = dataset_by_qid.get(qid, {})

        score_new = r_new.get("local_heuristic_score")
        score_old = r_old.get("local_heuristic_score")
        status_new = r_new.get("status")
        status_old = r_old.get("status")
        hyp_new = r_new.get("hypothesis", "").strip()
        hyp_old = r_old.get("hypothesis", "").strip()
        chunks_new = r_new.get("retrieved_chunk_ids", [])
        chunks_old = r_old.get("retrieved_chunk_ids", [])
        gold_rank_new = r_new.get("gold_retrieval_rank")
        gold_rank_old = r_old.get("gold_retrieval_rank")

        has_score_diff = score_new != score_old
        has_status_diff = status_new != status_old
        has_hyp_diff = hyp_new != hyp_old
        has_retrieval_diff = chunks_new != chunks_old or gold_rank_new != gold_rank_old

        if has_score_diff or has_status_diff or has_hyp_diff or has_retrieval_diff:
            # Analyze root cause
            cause = []
            causal_type = "unknown"
            
            if has_retrieval_diff and not has_hyp_diff:
                causal_type = "retrieval_only_invariant_answer"
                cause.append("Retrieved chunks differed but answer model generated identical response.")
            elif has_score_diff:
                # Disagreement in score
                if qid == "184da446":
                    causal_type = "retrieval_ontology_evidence_gain"
                    cause.append("Config-A missed answer evidence (empty answer); Config-B retrieved chunk sess_answer_90993049_c002 via ontology expansion, providing required factual context.")
                elif qid == "dcfa8644":
                    causal_type = "temporal_prompt_structuring_evaluator_limitation"
                    cause.append("Both models found correct answer (14 days). Config-A wrote narrative 'Fourteen days' which failed heuristic regex; Config-D received structured temporal claim [2023-01-10 to 2023-01-24] and concisely responded '14 days' which matched numeric heuristic.")
                elif qid == "gpt4_fa19884c":
                    causal_type = "temporal_prompt_structuring_evaluator_limitation"
                    cause.append("Both models found correct interval (6 days). Config-C wrote verbose narrative 'Six days passed...', failing numeric regex; Config-D wrote concise '6 days', matching regex.")
                elif qid == "2698e78f_abs":
                    causal_type = "evaluator_brittleness_abstention"
                    cause.append("Both models correctly identified question as unanswerable. Config-A used contraction 'don't have sufficient' which matched hardcoded list; Config-D generated formal 'do not have sufficient' which the strict substring heuristic failed to recognize.")
                elif qid == "099778bb":
                    causal_type = "retrieval_distractor_or_hesitation"
                    cause.append("Config-A answered '20%' based on retrieved session. Config-D correctly detected ambiguity and abstained ('I do not have sufficient recorded memories...'). Hypothesis: Temporal claims or graph expansion highlighted conflicting evidence, leading to cautious abstention.")
                elif qid == "gpt4_4fc4f797":
                    causal_type = "generation_stochasticity_or_distractor"
                    cause.append("Config-B matched numeric condition; Config-A generated explanation without explicit numeric anchor.")
                elif qid == "gpt4_4edbafa2":
                    causal_type = "generation_stochasticity_or_distractor"
                    cause.append("Config-A/C matched numeric condition; Config-B omitted exact digit anchor.")
                elif qid == "gpt4_c27434e8_abs":
                    causal_type = "graph_repetition_loop_vs_temporal_stabilization"
                    cause.append("In Config-C, graph expansion introduced adjacent conversational turns that triggered a repetition loop in the local 7B model. In Config-D, temporal claim structuring eliminated the loop, allowing clean abstention.")
                else:
                    causal_type = "unclassified_disagreement"
                    cause.append(f"Score shifted from {score_old} to {score_new}.")
            else:
                causal_type = "phrasing_or_retrieval_variation"
                cause.append("Score was invariant, but phrasing or chunk ordering varied.")

            pair_disagreements.append({
                "question_id": qid,
                "question_type": r_new.get("question_type"),
                "question": r_new.get("question"),
                "gold_answer": gold_inst.get("answer"),
                "is_abstention": r_new.get("is_abstention"),
                "old_config": {
                    "config": name_old,
                    "score": score_old,
                    "status": status_old,
                    "hypothesis": hyp_old,
                    "gold_retrieval_rank": gold_rank_old,
                    "retrieved_chunk_ids": chunks_old,
                    "citations": r_old.get("citations", [])
                },
                "new_config": {
                    "config": name_new,
                    "score": score_new,
                    "status": status_new,
                    "hypothesis": hyp_new,
                    "gold_retrieval_rank": gold_rank_new,
                    "retrieved_chunk_ids": chunks_new,
                    "citations": r_new.get("citations", [])
                },
                "differences": {
                    "score_changed": has_score_diff,
                    "status_changed": has_status_diff,
                    "retrieval_changed": has_retrieval_diff,
                    "hypothesis_changed": has_hyp_diff
                },
                "causal_classification": causal_type,
                "causal_analysis": " ".join(cause)
            })

    score_disagreements = [d for d in pair_disagreements if d["differences"]["score_changed"]]
    disagreement_summary[f"{name_new}_vs_{name_old}"] = {
        "comparison": pair_desc,
        "total_instances_compared": len(data_new),
        "total_disagreements_any": len(pair_disagreements),
        "total_score_disagreements": len(score_disagreements),
        "score_disagreements": score_disagreements,
        "all_disagreements": pair_disagreements
    }

output_path = STEP8_DIR / "disagreement_analysis.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(disagreement_summary, f, indent=2)

print(f"Saved machine-readable disagreement analysis to: {output_path}")
for k, v in disagreement_summary.items():
    print(f"Pair {k:22s}: {v['total_score_disagreements']:2d} score disagreements, {v['total_disagreements_any']:2d} total diffs")
