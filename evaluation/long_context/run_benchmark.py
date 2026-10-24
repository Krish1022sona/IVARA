"""
Entry point to run the IVARA Long-Context Benchmark & Stress Test.
"""

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.long_context.harness import (
    evaluate_extraction_quality,
    evaluate_question_set,
    ingest_story_representation,
    parse_story_episodes,
    setup_isolated_stores
)


def main():
    bench_dir = Path(__file__).parent
    story_path = bench_dir / "story.md"
    gold_path = bench_dir / "gold_annotations.json"
    questions_path = bench_dir / "questions.json"
    reports_dir = bench_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("      IVARA Long-Context Memory Benchmark & Stress Test          ")
    print("=================================================================")
    
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_annotations = json.load(f)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    questions = questions_data["questions"]

    # 1. Parse story episodes
    episodes = parse_story_episodes(story_path)
    print(f"Loaded {len(episodes)} chronological episodes from {story_path.name}.")

    # 2. Ingest Multi-Episode Representation
    print("\n--- Phase A: Ingesting Multi-Episode Representation ---")
    multi_dir = bench_dir / "multi_episode"
    multi_stores = setup_isolated_stores(multi_dir, gold_annotations)
    t0 = time.time()
    multi_ingested = ingest_story_representation(episodes, multi_stores, mode="multi")
    multi_ingest_time = round(time.time() - t0, 2)
    print(f"Multi-episode ingestion completed in {multi_ingest_time}s.")

    # Save raw multi extraction artifacts
    with open(bench_dir / "raw_extraction_multi.json", "w", encoding="utf-8") as f:
        json.dump(multi_ingested, f, indent=2)

    # 3. Ingest Single-Episode Representation
    print("\n--- Phase B: Ingesting Single-Episode Representation ---")
    single_dir = bench_dir / "single_episode"
    single_stores = setup_isolated_stores(single_dir, gold_annotations)
    t0 = time.time()
    single_ingested = ingest_story_representation(episodes, single_stores, mode="single")
    single_ingest_time = round(time.time() - t0, 2)
    print(f"Single-episode ingestion completed in {single_ingest_time}s.")

    # Save raw single extraction artifacts
    with open(bench_dir / "raw_extraction_single.json", "w", encoding="utf-8") as f:
        json.dump(single_ingested, f, indent=2)

    # 4. Extraction Quality Evaluation
    print("\n--- Phase C: Measuring Extraction Quality ---")
    multi_ext_metrics = evaluate_extraction_quality(multi_ingested, gold_annotations)
    single_ext_metrics = evaluate_extraction_quality(single_ingested, gold_annotations)
    print("Multi-Episode Extraction Metrics:", json.dumps(multi_ext_metrics, indent=2))
    print("Single-Episode Extraction Metrics:", json.dumps(single_ext_metrics, indent=2))

    # 5. Question Evaluation: Multi-Episode IVARA vs Baseline
    print("\n--- Phase D: Evaluating Multi-Episode (IVARA Pipeline vs Vector Baseline) ---")
    print(f"Running IVARA Pipeline (min_score=0.35, k=3) on {len(questions)} questions...")
    t0 = time.time()
    res_multi_ivara = evaluate_question_set(
        questions=questions,
        memories_path=multi_stores["memories_path"],
        system_type="ivara",
        mode="multi",
        k=3
    )
    print(f"IVARA Multi evaluation completed in {round(time.time() - t0, 2)}s.")
    with open(bench_dir / "results_multi_ivara.json", "w", encoding="utf-8") as f:
        json.dump(res_multi_ivara, f, indent=2)

    print(f"\nRunning Ordinary Vector Baseline (min_score=None, k=3) on {len(questions)} questions...")
    t0 = time.time()
    res_multi_baseline = evaluate_question_set(
        questions=questions,
        memories_path=multi_stores["memories_path"],
        system_type="baseline",
        mode="multi",
        k=3
    )
    print(f"Baseline Multi evaluation completed in {round(time.time() - t0, 2)}s.")
    with open(bench_dir / "results_multi_baseline.json", "w", encoding="utf-8") as f:
        json.dump(res_multi_baseline, f, indent=2)

    # 6. Question Evaluation: Single-Episode IVARA vs Baseline
    print("\n--- Phase E: Evaluating Single-Episode (IVARA Pipeline vs Vector Baseline) ---")
    print(f"Running IVARA Pipeline on Single Episode...")
    t0 = time.time()
    res_single_ivara = evaluate_question_set(
        questions=questions,
        memories_path=single_stores["memories_path"],
        system_type="ivara",
        mode="single",
        k=3
    )
    print(f"IVARA Single evaluation completed in {round(time.time() - t0, 2)}s.")
    with open(bench_dir / "results_single_ivara.json", "w", encoding="utf-8") as f:
        json.dump(res_single_ivara, f, indent=2)

    print(f"\nRunning Ordinary Vector Baseline on Single Episode...")
    t0 = time.time()
    res_single_baseline = evaluate_question_set(
        questions=questions,
        memories_path=single_stores["memories_path"],
        system_type="baseline",
        mode="single",
        k=3
    )
    print(f"Baseline Single evaluation completed in {round(time.time() - t0, 2)}s.")
    with open(bench_dir / "results_single_baseline.json", "w", encoding="utf-8") as f:
        json.dump(res_single_baseline, f, indent=2)

    # 7. Aggregate Metrics Summary
    summary = {
        "multi_episode": {
            "extraction": multi_ext_metrics,
            "ivara_pipeline": {
                "avg_retrieval_recall": res_multi_ivara["avg_retrieval_recall"],
                "avg_retrieval_precision": res_multi_ivara["avg_retrieval_precision"],
                "avg_fact_score": res_multi_ivara["avg_fact_score"],
                "abstention_f1": res_multi_ivara["abstention"]["f1"],
                "abstention_false_positive_rate": res_multi_ivara["abstention"]["false_positive_answering_rate"]
            },
            "ordinary_vector_baseline": {
                "avg_retrieval_recall": res_multi_baseline["avg_retrieval_recall"],
                "avg_retrieval_precision": res_multi_baseline["avg_retrieval_precision"],
                "avg_fact_score": res_multi_baseline["avg_fact_score"],
                "abstention_f1": res_multi_baseline["abstention"]["f1"],
                "abstention_false_positive_rate": res_multi_baseline["abstention"]["false_positive_answering_rate"]
            }
        },
        "single_episode": {
            "extraction": single_ext_metrics,
            "ivara_pipeline": {
                "avg_retrieval_recall": res_single_ivara["avg_retrieval_recall"],
                "avg_retrieval_precision": res_single_ivara["avg_retrieval_precision"],
                "avg_fact_score": res_single_ivara["avg_fact_score"],
                "abstention_f1": res_single_ivara["abstention"]["f1"],
                "abstention_false_positive_rate": res_single_ivara["abstention"]["false_positive_answering_rate"]
            },
            "ordinary_vector_baseline": {
                "avg_retrieval_recall": res_single_baseline["avg_retrieval_recall"],
                "avg_retrieval_precision": res_single_baseline["avg_retrieval_precision"],
                "avg_fact_score": res_single_baseline["avg_fact_score"],
                "abstention_f1": res_single_baseline["abstention"]["f1"],
                "abstention_false_positive_rate": res_single_baseline["abstention"]["false_positive_answering_rate"]
            }
        }
    }

    with open(bench_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=================================================================")
    print("                      BENCHMARK COMPLETE                         ")
    print("=================================================================")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
