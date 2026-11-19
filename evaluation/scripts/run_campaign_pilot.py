"""
Phase 3 Pilot Evaluation Runner for IVARA Evaluation Campaign v1.
Executes a 60-question stratified benchmark across 4 configurations:
  - Config-A: Dense + BM25 baseline (B-Chunked)
  - Config-B: Baseline + ontology (B-Chunked-Ontology)
  - Config-C: Ontology + bounded graph expansion (B-Chunked-Graph)
  - Config-D: Full system with temporal reasoning (B-Chunked-Full)

Features:
  - Immutable manifest & resume protection
  - Immediate persistence per question
  - Comprehensive per-stage latencies and observables
  - Paired comparison analysis with confidence intervals
  - Generates machine-readable CSV & JSON summaries
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.runners.run_longmemeval import LongMemEvalRunner

DATASET_PATH = PROJECT_ROOT / "evaluation" / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
PILOT_DIR = PROJECT_ROOT / "evaluation" / "data" / "longmemeval" / "runs" / "campaign_v1_pilot"
MANIFEST_PATH = PILOT_DIR / "pilot_sample_manifest.json"

CONFIGS = [
    ("Config-A", "config_a", "Dense + BM25 baseline"),
    ("Config-B", "config_b", "Baseline + ontology guidance"),
    ("Config-C", "config_c", "Ontology + bounded graph expansion"),
    ("Config-D", "config_d", "Full system with temporal reasoning"),
]

def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.96 if confidence == 0.95 else 1.645
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)

def run_pilot():
    print("=" * 70)
    print("IVARA EVALUATION CAMPAIGN V1 — PHASE 3 PILOT EVALUATION")
    print("=" * 70)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        pilot_manifest = json.load(f)

    target_qids = pilot_manifest["selected_qids"]
    print(f"Loaded {len(target_qids)} stratified target questions across 6 tasks.")

    all_config_results = {}
    all_config_summaries = {}

    for config_name, dir_name, desc in CONFIGS:
        cfg_dir = PILOT_DIR / dir_name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}")
        print(f"RUNNING CONFIGURATION: {config_name} ({desc})")
        print(f"Output: {cfg_dir}")
        print(f"{'='*70}\n")

        runner = LongMemEvalRunner(
            dataset_path=DATASET_PATH,
            output_dir=cfg_dir,
            config=config_name,
            model="gpt-oss",
            k=5,
            use_local_judge=True
        )
        summary = runner.run(target_question_ids=target_qids, resume=True)
        all_config_summaries[config_name] = summary

        # Read back raw results
        results_file = cfg_dir / f"results_config_{config_name.upper()}.jsonl"
        with open(results_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        all_config_results[config_name] = {r["question_id"]: r for r in records}

    # Aggregate & compute detailed metrics
    print("\n" + "=" * 70)
    print("COMPUTING PILOT METRICS & PAIRED COMPARISONS")
    print("=" * 70)

    metrics_by_config = {}
    for config_name, _, desc in CONFIGS:
        records_map = all_config_results[config_name]
        records = [records_map[qid] for qid in target_qids if qid in records_map]

        n_total = len(records)
        ans_records = [r for r in records if not r["is_abstention"]]
        abs_records = [r for r in records if r["is_abstention"]]

        # Accuracy (local heuristic)
        n_correct = sum(1 for r in records if r["local_heuristic_score"] is True)
        acc_overall = n_correct / n_total if n_total > 0 else 0.0
        acc_ci = compute_wilson_ci(n_correct, n_total)

        n_ans_correct = sum(1 for r in ans_records if r["local_heuristic_score"] is True)
        acc_answerable = n_ans_correct / len(ans_records) if ans_records else 0.0
        ans_ci = compute_wilson_ci(n_ans_correct, len(ans_records))

        # Abstention metrics
        # Abstention accuracy: abstained when is_abstention=True
        correct_abstentions = sum(1 for r in abs_records if r["status"] == "insufficient_evidence")
        false_abstentions = sum(1 for r in ans_records if r["status"] == "insufficient_evidence")
        total_abstentions_predicted = sum(1 for r in records if r["status"] == "insufficient_evidence")

        abs_recall = correct_abstentions / len(abs_records) if abs_records else 0.0
        abs_precision = correct_abstentions / total_abstentions_predicted if total_abstentions_predicted > 0 else 0.0
        false_abs_rate = false_abstentions / len(ans_records) if ans_records else 0.0

        # Retrieval Recall@k on answerable instances
        rec1 = sum(1 for r in ans_records if r.get("gold_retrieval_rank") == 1) / len(ans_records) if ans_records else 0.0
        rec3 = sum(1 for r in ans_records if r.get("gold_retrieval_rank") and r["gold_retrieval_rank"] <= 3) / len(ans_records) if ans_records else 0.0
        rec5 = sum(1 for r in ans_records if r.get("gold_retrieval_rank") and r["gold_retrieval_rank"] <= 5) / len(ans_records) if ans_records else 0.0

        # Citations
        has_citations = sum(1 for r in ans_records if len(r.get("citations", [])) > 0)
        citation_coverage = has_citations / len(ans_records) if ans_records else 0.0

        # Latencies
        total_latencies = [r["latency_ms"] for r in records]
        mean_lat = sum(total_latencies) / len(total_latencies) if total_latencies else 0.0
        chunk_lat = [r.get("stage_latencies_ms", {}).get("chunking", 0.0) for r in records]
        graph_lat = [r.get("stage_latencies_ms", {}).get("graph_build", 0.0) for r in records]
        ans_lat = [r.get("stage_latencies_ms", {}).get("answering", 0.0) for r in records]

        # Context tokens
        tokens = [r.get("context_token_count", 0) for r in records]
        mean_tokens = sum(tokens) / len(tokens) if tokens else 0

        # Temporal observables
        temp_acts = sum(1 for r in records if r.get("temporal_activated"))
        total_selected_claims = sum(r.get("selected_claims_count", 0) for r in records)
        total_historical_claims = sum(r.get("historical_claims_count", 0) for r in records)

        metrics_by_config[config_name] = {
            "description": desc,
            "total_questions": n_total,
            "accuracy_overall": round(acc_overall * 100, 2),
            "accuracy_overall_ci95": [round(acc_ci[0] * 100, 2), round(acc_ci[1] * 100, 2)],
            "accuracy_answerable": round(acc_answerable * 100, 2),
            "accuracy_answerable_ci95": [round(ans_ci[0] * 100, 2), round(ans_ci[1] * 100, 2)],
            "recall_at_1": round(rec1 * 100, 2),
            "recall_at_3": round(rec3 * 100, 2),
            "recall_at_5": round(rec5 * 100, 2),
            "abstention_precision": round(abs_precision * 100, 2),
            "abstention_recall": round(abs_recall * 100, 2),
            "false_abstention_rate": round(false_abs_rate * 100, 2),
            "citation_coverage": round(citation_coverage * 100, 2),
            "mean_total_latency_ms": round(mean_lat, 2),
            "mean_chunking_latency_ms": round(sum(chunk_lat) / len(chunk_lat), 2) if chunk_lat else 0.0,
            "mean_graph_build_latency_ms": round(sum(graph_lat) / len(graph_lat), 2) if graph_lat else 0.0,
            "mean_answering_latency_ms": round(sum(ans_lat) / len(ans_lat), 2) if ans_lat else 0.0,
            "mean_context_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
            "temporal_activated_count": temp_acts,
            "total_selected_claims": total_selected_claims,
            "total_historical_claims": total_historical_claims
        }

    # Paired comparisons
    paired_comparisons = {}
    pairs = [
        ("Config-B", "Config-A", "Ontology Guidance vs Dense+BM25 Baseline"),
        ("Config-C", "Config-B", "Bounded Graph Expansion vs Ontology Guidance"),
        ("Config-D", "Config-C", "Temporal Reasoning vs Bounded Graph Expansion"),
        ("Config-D", "Config-A", "Full System vs Dense+BM25 Baseline")
    ]

    for cfg_new, cfg_base, pair_desc in pairs:
        res_new = all_config_results[cfg_new]
        res_base = all_config_results[cfg_base]

        acc_diffs = []
        rec_diffs = []
        lat_diffs = []

        wins = 0
        losses = 0
        ties = 0

        for qid in target_qids:
            if qid not in res_new or qid not in res_base:
                continue
            r_n = res_new[qid]
            r_b = res_base[qid]

            s_n = 1 if r_n["local_heuristic_score"] is True else 0
            s_b = 1 if r_b["local_heuristic_score"] is True else 0
            acc_diffs.append(s_n - s_b)

            if s_n > s_b:
                wins += 1
            elif s_n < s_b:
                losses += 1
            else:
                ties += 1

            if not r_n["is_abstention"]:
                rank_n = 1 if (r_n.get("gold_retrieval_rank") and r_n["gold_retrieval_rank"] <= 5) else 0
                rank_b = 1 if (r_b.get("gold_retrieval_rank") and r_b["gold_retrieval_rank"] <= 5) else 0
                rec_diffs.append(rank_n - rank_b)

            lat_diffs.append(r_n["latency_ms"] - r_b["latency_ms"])

        net_acc_delta = (sum(acc_diffs) / len(acc_diffs)) * 100 if acc_diffs else 0.0
        net_rec_delta = (sum(rec_diffs) / len(rec_diffs)) * 100 if rec_diffs else 0.0
        mean_lat_delta = sum(lat_diffs) / len(lat_diffs) if lat_diffs else 0.0

        paired_comparisons[f"{cfg_new}_vs_{cfg_base}"] = {
            "comparison": pair_desc,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "net_accuracy_delta_pp": round(net_acc_delta, 2),
            "net_recall5_delta_pp": round(net_rec_delta, 2),
            "mean_latency_delta_ms": round(mean_lat_delta, 2)
        }

    # Save summary JSON
    summary_data = {
        "campaign": "IVARA Evaluation Campaign v1 — Pilot",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_variant": "longmemeval_s_cleaned.json",
        "total_instances_evaluated": len(target_qids),
        "configurations": metrics_by_config,
        "paired_comparisons": paired_comparisons
    }

    summary_json_path = PILOT_DIR / "pilot_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save summary CSV
    summary_csv_path = PILOT_DIR / "pilot_summary.csv"
    with open(summary_csv_path, "w", encoding="utf-8") as f:
        headers = [
            "Configuration", "Description", "Overall_Acc", "Overall_Acc_CI95", "Answerable_Acc", "Answerable_Acc_CI95",
            "Recall@1", "Recall@3", "Recall@5", "Abs_Precision", "Abs_Recall", "False_Abs_Rate",
            "Citation_Coverage", "Mean_Total_Lat_ms", "Mean_Chunking_Lat_ms", "Mean_Graph_Lat_ms",
            "Mean_Answering_Lat_ms", "Mean_Tokens", "Temporal_Activated_Count"
        ]
        f.write(",".join(headers) + "\n")
        for cfg_name, m in metrics_by_config.items():
            row = [
                cfg_name,
                f'"{m["description"]}"',
                f'{m["accuracy_overall"]}%',
                f'"{m["accuracy_overall_ci95"][0]}%-{m["accuracy_overall_ci95"][1]}%"',
                f'{m["accuracy_answerable"]}%',
                f'"{m["accuracy_answerable_ci95"][0]}%-{m["accuracy_answerable_ci95"][1]}%"',
                f'{m["recall_at_1"]}%',
                f'{m["recall_at_3"]}%',
                f'{m["recall_at_5"]}%',
                f'{m["abstention_precision"]}%',
                f'{m["abstention_recall"]}%',
                f'{m["false_abstention_rate"]}%',
                f'{m["citation_coverage"]}%',
                str(m["mean_total_latency_ms"]),
                str(m["mean_chunking_latency_ms"]),
                str(m["mean_graph_build_latency_ms"]),
                str(m["mean_answering_latency_ms"]),
                str(m["mean_context_tokens"]),
                str(m["temporal_activated_count"])
            ]
            f.write(",".join(row) + "\n")

    print(f"\nSaved Pilot Summary JSON: {summary_json_path}")
    print(f"Saved Pilot Summary CSV:  {summary_csv_path}")

    print("\n" + "=" * 70)
    print("PHASE 3 PILOT EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_pilot()
