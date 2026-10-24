"""
Recalculate all benchmark metrics from existing raw JSON artifacts
using corrected standard definitions.
"""

import json
from pathlib import Path
from evaluation.long_context.corrected_metrics import (
    calculate_retrieval_metrics,
    calculate_comprehensive_abstention,
    evaluate_rubric_fact,
    evaluate_lexical_heuristic
)

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE_DIR / "questions.json"
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions_list = json.load(f)["questions"]
    questions_map = {q["id"]: q for q in questions_list}


def process_result_file(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    k = data.get("k", 3)
    cases = data.get("cases", [])
    corrected_cases = []

    for c in cases:
        qid = c["id"]
        q_meta = questions_map.get(qid, {})
        retrieved_ids = c.get("retrieved_memory_ids", [])
        supporting_ids = c.get("supporting_memory_ids", [])
        actual_ans = c.get("actual_answer", "")
        actual_status = c.get("actual_status", "answered")
        gold_ans = c.get("gold_answer", "")
        rubric = q_meta.get("rubric", "")
        should_abstain = q_meta.get("should_abstain", False)
        expected_status = c.get("expected_status", "answered")
        cited_ids = c.get("cited_memory_ids", [])

        # 1. Retrieval
        ret_metrics = calculate_retrieval_metrics(
            retrieved_ids=retrieved_ids,
            supporting_ids=supporting_ids,
            k=k
        )

        # 2. Fact evaluation
        rubric_res = evaluate_rubric_fact(
            qid=qid,
            actual_answer=actual_ans,
            actual_status=actual_status,
            gold_answer=gold_ans,
            rubric=rubric,
            should_abstain=should_abstain
        )
        legacy_fact = evaluate_lexical_heuristic(
            actual_answer=actual_ans,
            actual_status=actual_status,
            gold_answer=gold_ans,
            expected_status=expected_status,
            should_abstain=should_abstain
        )

        # 3. Grounding
        if cited_ids:
            cit_valid = len(set(cited_ids) & set(retrieved_ids)) / len(cited_ids)
            if supporting_ids:
                cit_supp = len(set(cited_ids) & set(supporting_ids)) / len(cited_ids)
            else:
                cit_supp = 0.0
        else:
            cit_valid = 1.0 if should_abstain else 0.0
            cit_supp = 1.0 if should_abstain else 0.0

        corrected_cases.append({
            "id": qid,
            "category": c.get("category", ""),
            "question": c.get("question", ""),
            "actual_status": actual_status,
            "actual_answer": actual_ans,
            "gold_answer": gold_ans,
            "supporting_memory_ids": supporting_ids,
            "retrieved_memory_ids": retrieved_ids,
            "cited_memory_ids": cited_ids,
            "standard_precision_at_k": ret_metrics["precision_at_k"],
            "set_precision": ret_metrics["set_precision"],
            "recall_at_k": ret_metrics["recall_at_k"],
            "legacy_fact_score": legacy_fact,
            "rubric_fact_score": rubric_res["score"],
            "rubric_passed": rubric_res["passed"],
            "rubric_reason": rubric_res["reason"],
            "citation_validity": round(cit_valid, 4),
            "citation_support_precision": round(cit_supp, 4)
        })

    # Summary metrics
    ans_cases = [c for c in corrected_cases if c["supporting_memory_ids"]]
    unans_cases = [c for c in corrected_cases if not c["supporting_memory_ids"]]

    avg_std_prec_ans = sum(c["standard_precision_at_k"] for c in ans_cases) / len(ans_cases)
    avg_std_prec_all = sum(c["standard_precision_at_k"] for c in corrected_cases) / len(corrected_cases)
    avg_set_prec_ans = sum(c["set_precision"] for c in ans_cases) / len(ans_cases)
    avg_recall_ans = sum(c["recall_at_k"] for c in ans_cases) / len(ans_cases)

    avg_legacy_fact = sum(c["legacy_fact_score"] for c in corrected_cases) / len(corrected_cases)
    avg_rubric_fact = sum(c["rubric_fact_score"] for c in corrected_cases) / len(corrected_cases)
    rubric_pass_rate = sum(1 for c in corrected_cases if c["rubric_passed"]) / len(corrected_cases)

    abstention_stats = calculate_comprehensive_abstention(corrected_cases)

    return {
        "system_type": data.get("system_type", ""),
        "mode": data.get("mode", ""),
        "k": k,
        "total_cases": len(corrected_cases),
        "answerable_cases_count": len(ans_cases),
        "unanswerable_cases_count": len(unans_cases),
        "retrieval": {
            "standard_precision_at_k_answerable": round(avg_std_prec_ans, 4),
            "standard_precision_at_k_overall": round(avg_std_prec_all, 4),
            "set_precision_answerable": round(avg_set_prec_ans, 4),
            "recall_at_k_answerable": round(avg_recall_ans, 4)
        },
        "fact_accuracy": {
            "rubric_accuracy": round(rubric_pass_rate, 4),
            "avg_rubric_fact_score": round(avg_rubric_fact, 4),
            "legacy_heuristic_fact_score": round(avg_legacy_fact, 4)
        },
        "abstention": abstention_stats,
        "cases": corrected_cases
    }


def main():
    files_to_process = [
        "baseline_results_multi_ivara",
        "baseline_results_multi_baseline",
        "baseline_results_single_ivara",
        "baseline_results_single_baseline",
        "results_multi_ivara",
        "results_multi_baseline",
        "results_single_ivara",
        "results_single_baseline"
    ]

    corrected_summary = {}

    for fname in files_to_process:
        in_path = BASE_DIR / f"{fname}.json"
        out_path = BASE_DIR / f"corrected_{fname}.json"
        print(f"Processing {in_path.name} -> {out_path.name}...")
        res = process_result_file(in_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        # Add to summary
        corrected_summary[fname] = {
            "system_type": res["system_type"],
            "mode": res["mode"],
            "retrieval": res["retrieval"],
            "fact_accuracy": res["fact_accuracy"],
            "abstention": res["abstention"]
        }

    summary_out_path = BASE_DIR / "corrected_metrics_summary.json"
    with open(summary_out_path, "w", encoding="utf-8") as f:
        json.dump(corrected_summary, f, indent=2)
    print(f"Saved consolidated corrected summary to {summary_out_path.name}")


if __name__ == "__main__":
    main()
