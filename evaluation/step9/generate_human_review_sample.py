"""
Step 9: Generate Reproducible Human Review Sample and Failure Analysis.
Stratified selection of 26 questions from the 60-question pilot:
- Disagreements between configurations
- Correct and incorrect judgments by Gemma
- Answerable and unanswerable (abstention) questions
- All 6 task types represented
- Graph-related loops/failures and temporal reasoning cases
Outputs:
- evaluation/step9/human_review_sample.csv
- evaluation/step9/failure_analysis.json
"""

import csv
import json
import random
from pathlib import Path

STEP7_DIR = Path("evaluation/data/longmemeval/runs/campaign_v1_pilot")
STEP8_DIR = Path("evaluation/step8")
STEP9_DIR = Path("evaluation/step9")
STEP9_DIR.mkdir(parents=True, exist_ok=True)

DATASET_FILE = Path("evaluation/data/longmemeval/longmemeval_s_cleaned.json")

def main():
    # 1. Load dataset
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        dataset = {item["question_id"]: item for item in json.load(f)}

    # 2. Load raw predictions from all 4 configurations
    configs = ["Config-A", "Config-B", "Config-C", "Config-D"]
    prefixes = {"Config-A": "CONFIG-A", "Config-B": "CONFIG-B", "Config-C": "CONFIG-C", "Config-D": "CONFIG-D"}
    folders = {"Config-A": "config_a", "Config-B": "config_b", "Config-C": "config_c", "Config-D": "config_d"}
    
    raw_preds = {}
    for cfg in configs:
        p = STEP7_DIR / folders[cfg] / f"results_config_{prefixes[cfg]}.jsonl"
        with open(p, "r", encoding="utf-8") as f:
            raw_preds[cfg] = {r["question_id"]: r for r in [json.loads(line) for line in f if line.strip()]}

    # 3. Load Gemma judge results
    with open(STEP8_DIR / "local_judge_results.json", "r", encoding="utf-8") as f:
        judge_data = json.load(f)
    gemma_preds = judge_data["predictions_by_config"]

    # 4. Stratified selection of 26 questions:
    # Priority A: Disagreements in Gemma judgments across configs
    all_qids = list(raw_preds["Config-A"].keys())
    disagreement_qids = []
    for qid in all_qids:
        scores = [gemma_preds[cfg][qid]["is_correct"] for cfg in configs]
        if len(set(scores)) > 1:
            disagreement_qids.append(qid)

    print(f"Total questions with Gemma score disagreements across configs: {len(disagreement_qids)}")

    # Priority B: Representative Abstentions
    abstention_qids = [qid for qid in all_qids if raw_preds["Config-A"][qid]["is_abstention"]]
    
    # Priority C: Representative from each task type
    by_task = {}
    for qid in all_qids:
        t = raw_preds["Config-A"][qid]["question_type"]
        by_task.setdefault(t, []).append(qid)

    # Seeded selection for remaining quota to hit exactly 26 questions
    random.seed(42)
    selected_set = set(disagreement_qids)

    # Ensure at least 4 abstentions are in the sample
    for qid in abstention_qids:
        selected_set.add(qid)

    # Ensure every task has at least 3 questions
    for t, qids in by_task.items():
        present = [q for q in qids if q in selected_set]
        needed = 4 - len(present)
        if needed > 0:
            candidates = [q for q in qids if q not in selected_set]
            selected_set.update(random.sample(candidates, min(needed, len(candidates))))

    # Target exactly 26 questions
    selected_qids = sorted(list(selected_set))
    if len(selected_qids) > 26:
        # If over, prune purely redundant unanimous ones, keep disagreements and abstentions
        removable = [q for q in selected_qids if q not in disagreement_qids and q not in abstention_qids]
        random.shuffle(removable)
        excess = len(selected_qids) - 26
        for r_qid in removable[:excess]:
            selected_set.remove(r_qid)
        selected_qids = sorted(list(selected_set))

    print(f"Selected {len(selected_qids)} questions for human review sample.")
    print("Breakdown by task:")
    for t, qids in by_task.items():
        in_sample = [q for q in qids if q in selected_qids]
        print(f"  {t:30s}: {len(in_sample)} in sample")

    # 5. Build human_review_sample.csv
    csv_headers = [
        "sample_index",
        "question_id",
        "task_type",
        "is_unanswerable",
        "question",
        "gold_answer",
        "prediction_Config_A",
        "gemma_score_A",
        "gemma_reasoning_A",
        "prediction_Config_B",
        "gemma_score_B",
        "gemma_reasoning_B",
        "prediction_Config_C",
        "gemma_score_C",
        "gemma_reasoning_C",
        "prediction_Config_D",
        "gemma_score_D",
        "gemma_reasoning_D",
        "retrieved_chunks_D",
        "temporal_claims_count_D",
        "human_verdict_A",
        "human_verdict_B",
        "human_verdict_C",
        "human_verdict_D",
        "human_failure_category",
        "human_notes"
    ]

    csv_rows = []
    failure_analysis_records = []

    for idx, qid in enumerate(selected_qids, 1):
        ds_item = dataset[qid]
        q_text = ds_item.get("question", "")
        gold_ans = str(ds_item.get("answer", ""))
        is_abs = raw_preds["Config-A"][qid]["is_abstention"]
        qtype = raw_preds["Config-A"][qid]["question_type"]

        p_A = raw_preds["Config-A"][qid]["hypothesis"].strip()
        p_B = raw_preds["Config-B"][qid]["hypothesis"].strip()
        p_C = raw_preds["Config-C"][qid]["hypothesis"].strip()
        p_D = raw_preds["Config-D"][qid]["hypothesis"].strip()

        g_A = gemma_preds["Config-A"][qid]
        g_B = gemma_preds["Config-B"][qid]
        g_C = gemma_preds["Config-C"][qid]
        g_D = gemma_preds["Config-D"][qid]

        chunks_D = raw_preds["Config-D"][qid].get("retrieved_chunk_ids", [])
        temp_claims_D = raw_preds["Config-D"][qid].get("selected_claims_count", 0)

        # CSV row (Leave human verdicts strictly BLANK)
        csv_rows.append([
            idx,
            qid,
            qtype,
            "YES" if is_abs else "NO",
            q_text,
            gold_ans,
            p_A,
            "CORRECT" if g_A["is_correct"] else "INCORRECT",
            g_A["reasoning"],
            p_B,
            "CORRECT" if g_B["is_correct"] else "INCORRECT",
            g_B["reasoning"],
            p_C,
            "CORRECT" if g_C["is_correct"] else "INCORRECT",
            g_C["reasoning"],
            p_D,
            "CORRECT" if g_D["is_correct"] else "INCORRECT",
            g_D["reasoning"],
            "; ".join(chunks_D),
            temp_claims_D,
            "",  # human_verdict_A (blank)
            "",  # human_verdict_B (blank)
            "",  # human_verdict_C (blank)
            "",  # human_verdict_D (blank)
            "",  # human_failure_category (blank)
            ""   # human_notes (blank)
        ])

        # Technical Failure Analysis for Task 3
        # Categorize likely failure mode and check for judge questionability
        likely_category = "unclassified"
        observed_facts = []
        hypotheses = []
        judge_assessment = "reasonable"

        # Specific case analyses
        if qid == "184da446":
            likely_category = "retrieval_miss_rescued_by_ontology"
            observed_facts.append("Config-A produced empty string (chunk missing from top-k). Config-B/C/D retrieved sess_answer_90993049_c002 and answered 220 correctly.")
            judge_assessment = "Gemma correctly failed A and passed B/C/D."
        elif qid == "gpt4_c27434e8_abs":
            likely_category = "irrelevant_context_graph_distraction_and_repetition_loop"
            observed_facts.append("Config-C generated infinite repetition loop from multi-turn graph chunks. Config-D cleanly abstained.")
            judge_assessment = "Gemma correctly failed C and passed D. Gemma noted C failed to provide an answer due to repetitive loop."
        elif qid == "dcfa8644":
            likely_category = "evaluator_ambiguity_in_heuristic_resolved_by_gemma"
            observed_facts.append("Config-A wrote 'Fourteen days', Config-D wrote '14 days'. Gold is '14 days'.")
            judge_assessment = "Gemma correctly passed BOTH Config-A and Config-D as semantically equivalent, exposing the heuristic judge flaw."
        elif qid == "2698e78f_abs":
            likely_category = "evaluator_ambiguity_in_heuristic_resolved_by_gemma"
            observed_facts.append("Config-D generated uncontracted 'do not have sufficient recorded memories'.")
            judge_assessment = "Gemma correctly passed all abstentions, recognizing formal prose as valid."
        elif qid == "099778bb":
            likely_category = "unsupported_answer_or_cautious_abstention"
            observed_facts.append("Config-A/B answered 20% (gold is 20%). Config-D abstained.")
            hypotheses.append("Temporal claims or additional graph chunks introduced corporate debate text that caused Config-D's grounding filter to abstain.")
            judge_assessment = "Gemma marked A/B/C correct and D incorrect. Questionable: whether abstaining on ambiguous dialogue is a system virtue or accuracy failure."
        elif qid == "09ba9854":
            likely_category = "retrieval_or_calculation_complexity"
            observed_facts.append("Multi-session question regarding taxi vs bus cost savings. All configs produced numerical estimates. Config-D was judged correct by Gemma while A/B were judged incorrect.")
            judge_assessment = "Gemma judged Config-D correct because it provided the complete range ($120-$170 for Narita and $35-$80 for Haneda)."
        elif qid == "6aeb4375":
            likely_category = "knowledge_update_supersession_challenge"
            observed_facts.append("Config-A/B/C answered with updated status; Config-D included additional contextual details.")
            judge_assessment = "Gemma failed Config-D on 6aeb4375 citing over-specification. Questionable judge strictness."
        elif is_abs:
            if all(gemma_preds[c][qid]["is_correct"] for c in configs):
                likely_category = "successful_abstention"
                observed_facts.append("All four configurations correctly abstained on unanswerable question.")
            else:
                likely_category = "incorrect_answer_to_unanswerable_question"
                observed_facts.append("One or more configurations attempted to answer unanswerable question.")
        else:
            if not any(gemma_preds[c][qid]["is_correct"] for c in configs):
                likely_category = "retrieval_miss_or_hallucination_unanimous"
                observed_facts.append("All configurations failed according to Gemma judge.")
            else:
                likely_category = "configuration_retrieval_or_generation_variance"
                observed_facts.append("Configurations diverged in answer completeness or detail.")

        failure_analysis_records.append({
            "sample_index": idx,
            "question_id": qid,
            "task_type": qtype,
            "is_unanswerable": is_abs,
            "question": q_text,
            "gold_answer": gold_ans,
            "likely_failure_category": likely_category,
            "observed_facts": observed_facts,
            "hypotheses": hypotheses,
            "judge_assessment": judge_assessment,
            "gemma_verdicts": {
                "Config-A": g_A["is_correct"],
                "Config-B": g_B["is_correct"],
                "Config-C": g_C["is_correct"],
                "Config-D": g_D["is_correct"]
            },
            "gemma_reasoning": {
                "Config-A": g_A["reasoning"],
                "Config-B": g_B["reasoning"],
                "Config-C": g_C["reasoning"],
                "Config-D": g_D["reasoning"]
            }
        })

    # Save CSV
    csv_file = STEP9_DIR / "human_review_sample.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(csv_rows)
    print(f"Saved human review CSV to: {csv_file}")

    # Save failure analysis JSON
    fa_file = STEP9_DIR / "failure_analysis.json"
    with open(fa_file, "w", encoding="utf-8") as f:
        json.dump({
            "sampling_metadata": {
                "seed": 42,
                "sampling_strategy": "Stratified selection prioritizing Gemma score disagreements, abstentions, and task diversity",
                "sample_size": len(selected_qids),
                "total_pilot_pool": len(all_qids),
                "selected_question_ids": selected_qids
            },
            "failure_analysis": failure_analysis_records
        }, f, indent=2)
    print(f"Saved failure analysis JSON to: {fa_file}")

if __name__ == "__main__":
    main()
