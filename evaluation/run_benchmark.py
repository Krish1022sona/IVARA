"""
CLI entry point to execute the IVARA offline memory QA benchmark.

Usage:
    python -m evaluation.run_benchmark
    python -m evaluation.run_benchmark --output-json evaluation/reports/results.json --output-md evaluation/reports/report.md --verbose
"""

import argparse
import json
import sys
from pathlib import Path

from .runner import generate_markdown_report, run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run the IVARA Reproducible Memory QA Benchmark.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="evaluation/reports/benchmark_results.json",
        help="Path to save machine-readable JSON report."
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="evaluation/reports/benchmark_report.md",
        help="Path to save concise Markdown report."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Top-k retrieval parameter (default: 3)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for every individual test case."
    )

    args = parser.parse_args()

    print("=" * 80)
    print("IVARA Reproducible Memory QA Benchmark")
    print("=" * 80)
    print(f"Executing deterministic benchmark across synthetic test suite (k={args.k})...\n")

    results = run_benchmark(k=args.k, verbose=args.verbose)

    overall = results["overall"]
    by_cat = results["by_category"]
    base_a = overall["baseline_a_raw_vector"]
    ivara = overall["ivara_pipeline"]

    print("-" * 80)
    print("OVERALL METRICS SUMMARY:")
    print("-" * 80)
    print(f"Total Cases: {overall['total_cases']} (Answerable: {overall['answerable_cases']}, Unanswerable/Distractor: {overall['unanswerable_cases']})")
    print()
    print(f"{'Metric':<35} | {'Baseline A (Raw Vector)':<25} | {'Baseline B (IVARA Pipeline)':<25}")
    print("-" * 90)
    print(f"{'Retrieval Recall@k':<35} | {base_a['mean_recall_at_k']*100:>23.1f}% | {ivara['mean_recall_at_k']*100:>23.1f}%")
    print(f"{'Retrieval Precision@k':<35} | {base_a['mean_precision_at_k']*100:>23.1f}% | {ivara['mean_precision_at_k']*100:>23.1f}%")
    print(f"{'Unanswerable False Positive Rate':<35} | {base_a['unanswerable_false_positive_rate']*100:>23.1f}% | {ivara['unanswerable_false_positive_rate']*100:>23.1f}%")
    print(f"{'Answer Fact Accuracy':<35} | {'N/A':>23} | {ivara['mean_fact_accuracy']*100:>23.1f}%")
    print(f"{'Citation Precision':<35} | {'N/A':>23} | {ivara['mean_citation_precision']*100:>23.1f}%")
    print(f"{'Unsupported-Claim Rate':<35} | {'N/A':>23} | {ivara['mean_unsupported_claim_rate']*100:>23.1f}%")
    print(f"{'Abstention F1 Score':<35} | {'N/A':>23} | {ivara['abstention_f1']*100:>23.1f}%")
    print("-" * 90)
    print(f"Abstention Precision: {ivara['abstention_precision']*100:.1f}%, Abstention Recall: {ivara['abstention_recall']*100:.1f}%")
    print()

    print("-" * 80)
    print("PER-CATEGORY BREAKDOWN:")
    print("-" * 80)
    print(f"{'Category':<32} | {'Cases':<6} | {'Recall':<8} | {'Precision':<10} | {'Fact Acc':<9} | {'Failures':<8}")
    print("-" * 85)
    for cat, stats in by_cat.items():
        print(f"{cat:<32} | {stats['count']:<6} | {stats['ivara_recall']*100:>6.1f}% | {stats['ivara_precision']*100:>8.1f}% | {stats['fact_accuracy']*100:>7.1f}% | {stats['failures_count']:<8}")
    print("-" * 85)

    if args.verbose:
        print("\n" + "=" * 80)
        print("PER-CASE FAILURE INSPECTION:")
        print("=" * 80)
        failures_found = 0
        for c in results["cases"]:
            if c["failures"]:
                failures_found += 1
                print(f"Case [{c['case_id']}] ({c['category']}):")
                print(f"  Query: {c['query']}")
                print(f"  Supporting IDs: {c['supporting_ids']}, Retrieved: {c['ivara_retrieved']}")
                print(f"  Status: {c['ivara_status']}, Citations: {c['ivara_citations']}")
                print(f"  Failures: {', '.join(c['failures'])}\n")
        if failures_found == 0:
            print("No case failures detected!\n")

    # Save output reports
    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON report to: {json_path}")

    if args.output_md:
        md_path = Path(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = generate_markdown_report(results)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Saved Markdown report to: {md_path}")

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
