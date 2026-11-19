"""Phase 2 Statistical Analysis for Milestone 3 Audit.

Computes:
- Paired differences against Arm A for Arms B, C, D, E.
- 95% bootstrap confidence intervals (10,000 resamples).
- Statistical significance tests:
  - McNemar's exact test (binomial) for binary accuracy.
  - Paired t-test and Wilcoxon signed-rank test for fact coverage.
  - Wilcoxon signed-rank test for Recall@5.
- Effect sizes (Cohen's d_z, Odds Ratio).
- Per-category paired differences and significance.
"""

import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
RESULTS_DIR = WORKSPACE_ROOT / "evaluation" / "milestone_3_results"
AUDIT_DIR = WORKSPACE_ROOT / "evaluation" / "milestone_3_1_audit"

from evaluation.validation_sprint.dataset_expanded import QUESTIONS

def get_stratified_ids():
    test_questions = [q for q in QUESTIONS if q.get("split") == "test"]
    by_cat = defaultdict(list)
    for q in test_questions:
        by_cat[q["category"]].append(q)
    stratified_q = []
    for cat in sorted(by_cat.keys()):
        stratified_q.extend(by_cat[cat][:5])
    return [q["id"] for q in stratified_q], {q["id"]: q["category"] for q in stratified_q}

def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.array(diffs, dtype=float)
    n = len(diffs)
    boot_means = [np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_boot)]
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(low), float(high)

def mcnemar_exact(y_arm, y_base):
    # b: arm=1, base=0; c: arm=0, base=1
    b = sum(1 for a, b_ in zip(y_arm, y_base) if a == 1 and b_ == 0)
    c = sum(1 for a, b_ in zip(y_arm, y_base) if a == 0 and b_ == 1)
    n_discordant = b + c
    if n_discordant == 0:
        p_val = 1.0
        odds_ratio = 1.0
    else:
        # Two-tailed binomial test with p=0.5
        res = stats.binomtest(b, n_discordant, 0.5, alternative='two-sided')
        p_val = res.pvalue
        odds_ratio = float(b / c) if c > 0 else float("inf")
    return {"b": b, "c": c, "p_value": float(p_val), "odds_ratio": odds_ratio}

def paired_t_and_wilcoxon(vals_arm, vals_base):
    diffs = np.array(vals_arm, dtype=float) - np.array(vals_base, dtype=float)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    cohens_d = float(mean_diff / std_diff) if std_diff > 1e-9 else 0.0

    # Paired t-test
    if np.all(diffs == 0):
        t_stat, t_pval = 0.0, 1.0
    else:
        t_stat, t_pval = stats.ttest_rel(vals_arm, vals_base)

    # Wilcoxon signed-rank test
    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) == 0:
        w_stat, w_pval = 0.0, 1.0
    else:
        try:
            w_res = stats.wilcoxon(diffs, alternative='two-sided')
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_pval = 0.0, 1.0

    ci_low, ci_high = bootstrap_ci(diffs)

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "cohens_d": cohens_d,
        "ci_95": [ci_low, ci_high],
        "t_stat": float(t_stat),
        "t_pvalue": float(t_pval),
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_pval
    }

def run_phase2_analysis():
    strat_ids, cat_map = get_stratified_ids()
    n_q = len(strat_ids)

    arm_data = {}
    arms = [
        "arm_a_semantic_hybrid",
        "arm_b_episodic_only",
        "arm_c_semantic_only",
        "arm_d_episodic_semantic",
        "arm_e_unified_tri_source"
    ]
    for arm in arms:
        p_path = RESULTS_DIR / f"predictions_{arm}.json"
        with open(p_path, "r", encoding="utf-8") as f:
            arm_data[arm] = json.load(f)

    # Base: Arm A
    base_preds = [arm_data["arm_a_semantic_hybrid"][qid] for qid in strat_ids]
    base_bin = [p["answer_metrics"]["binary_accuracy"] for p in base_preds]
    base_cov = [p["answer_metrics"]["fact_coverage"] for p in base_preds]
    base_r5 = [p["retrieval_metrics"]["recall_at_5"] for p in base_preds]

    overall_stats = {}

    for arm in arms[1:]:
        arm_preds = [arm_data[arm][qid] for qid in strat_ids]
        arm_bin = [p["answer_metrics"]["binary_accuracy"] for p in arm_preds]
        arm_cov = [p["answer_metrics"]["fact_coverage"] for p in arm_preds]
        arm_r5 = [p["retrieval_metrics"]["recall_at_5"] for p in arm_preds]

        # Binary Accuracy stats
        bin_diffs = np.array(arm_bin) - np.array(base_bin)
        bin_ci_low, bin_ci_high = bootstrap_ci(bin_diffs)
        mcnemar_res = mcnemar_exact(arm_bin, base_bin)

        # Fact Coverage stats
        cov_stats = paired_t_and_wilcoxon(arm_cov, base_cov)

        # Recall@5 stats
        r5_stats = paired_t_and_wilcoxon(arm_r5, base_r5)

        overall_stats[arm] = {
            "binary_accuracy": {
                "mean_arm": float(np.mean(arm_bin)),
                "mean_base": float(np.mean(base_bin)),
                "diff": float(np.mean(bin_diffs)),
                "ci_95": [bin_ci_low, bin_ci_high],
                "mcnemar": mcnemar_res,
                "statistically_significant": mcnemar_res["p_value"] < 0.05
            },
            "fact_coverage": {
                "mean_arm": float(np.mean(arm_cov)),
                "mean_base": float(np.mean(base_cov)),
                "paired_stats": cov_stats,
                "statistically_significant": cov_stats["wilcoxon_pvalue"] < 0.05
            },
            "recall_at_5": {
                "mean_arm": float(np.mean(arm_r5)),
                "mean_base": float(np.mean(base_r5)),
                "paired_stats": r5_stats,
                "statistically_significant": r5_stats["wilcoxon_pvalue"] < 0.05
            }
        }

    # Per-category paired differences
    categories = sorted(list(set(cat_map.values())))
    cat_stats = defaultdict(dict)
    for cat in categories:
        cat_qids = [qid for qid in strat_ids if cat_map[qid] == cat]
        b_cov = [arm_data["arm_a_semantic_hybrid"][qid]["answer_metrics"]["fact_coverage"] for qid in cat_qids]
        b_bin = [arm_data["arm_a_semantic_hybrid"][qid]["answer_metrics"]["binary_accuracy"] for qid in cat_qids]
        b_r5 = [arm_data["arm_a_semantic_hybrid"][qid]["retrieval_metrics"]["recall_at_5"] for qid in cat_qids]

        for arm in arms[1:]:
            a_cov = [arm_data[arm][qid]["answer_metrics"]["fact_coverage"] for qid in cat_qids]
            a_bin = [arm_data[arm][qid]["answer_metrics"]["binary_accuracy"] for qid in cat_qids]
            a_r5 = [arm_data[arm][qid]["retrieval_metrics"]["recall_at_5"] for qid in cat_qids]

            diff_cov = float(np.mean(np.array(a_cov) - np.array(b_cov)))
            diff_bin = float(np.mean(np.array(a_bin) - np.array(b_bin)))
            diff_r5 = float(np.mean(np.array(a_r5) - np.array(b_r5)))

            # Exact binomial test on the 5 items
            mc = mcnemar_exact(a_bin, b_bin)

            cat_stats[cat][arm] = {
                "sample_count": len(cat_qids),
                "diff_binary_accuracy": diff_bin,
                "diff_fact_coverage": diff_cov,
                "diff_recall_at_5": diff_r5,
                "mcnemar_p": mc["p_value"]
            }

    # Arm D vs Arm E Comparison (Critical for Graph Contribution)
    d_preds = [arm_data["arm_d_episodic_semantic"][qid] for qid in strat_ids]
    e_preds = [arm_data["arm_e_unified_tri_source"][qid] for qid in strat_ids]
    d_bin = [p["answer_metrics"]["binary_accuracy"] for p in d_preds]
    e_bin = [p["answer_metrics"]["binary_accuracy"] for p in e_preds]
    d_cov = [p["answer_metrics"]["fact_coverage"] for p in d_preds]
    e_cov = [p["answer_metrics"]["fact_coverage"] for p in e_preds]
    d_vs_e_bin_mc = mcnemar_exact(e_bin, d_bin)
    d_vs_e_cov_stats = paired_t_and_wilcoxon(e_cov, d_cov)

    arm_d_vs_e = {
        "binary_accuracy": {
            "mean_arm_d": float(np.mean(d_bin)),
            "mean_arm_e": float(np.mean(e_bin)),
            "diff": float(np.mean(np.array(e_bin) - np.array(d_bin))),
            "mcnemar": d_vs_e_bin_mc
        },
        "fact_coverage": {
            "mean_arm_d": float(np.mean(d_cov)),
            "mean_arm_e": float(np.mean(e_cov)),
            "paired_stats": d_vs_e_cov_stats
        }
    }

    full_output = {
        "overall_paired_against_arm_a": overall_stats,
        "arm_d_vs_arm_e_graph_effect": arm_d_vs_e,
        "category_wise_paired_stats": cat_stats
    }

    out_file = AUDIT_DIR / "phase2_statistical_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"Saved: {out_file}")

    print("\n=== PHASE 2 STATISTICAL SUMMARY ===")
    print(f"{'Arm':<28} | {'Bin Diff (95% CI)':<22} | {'McNemar p':<10} | {'Cov Diff (95% CI)':<22} | {'Wilcoxon p':<10} | {'Recall@5 Diff':<13} | {'R@5 p':<10}")
    print("-" * 125)
    for arm, s in overall_stats.items():
        b = s["binary_accuracy"]
        c = s["fact_coverage"]
        r = s["recall_at_5"]
        b_ci = f"+{b['diff']*100:.1f}% [{b['ci_95'][0]*100:.1f}%, {b['ci_95'][1]*100:.1f}%]" if b['diff'] >= 0 else f"{b['diff']*100:.1f}% [{b['ci_95'][0]*100:.1f}%, {b['ci_95'][1]*100:.1f}%]"
        c_ci = f"+{c['paired_stats']['mean_diff']*100:.1f}% [{c['paired_stats']['ci_95'][0]*100:.1f}%, {c['paired_stats']['ci_95'][1]*100:.1f}%]"
        r_diff = f"+{(r['mean_arm'] - r['mean_base'])*100:.1f}%"

        print(f"{arm:<28} | {b_ci:<22} | {b['mcnemar']['p_value']:<10.4f} | {c_ci:<22} | {c['paired_stats']['wilcoxon_pvalue']:<10.4e} | {r_diff:<13} | {r['paired_stats']['wilcoxon_pvalue']:<10.4e}")

    print("\n--- Arm D (Episodic+Semantic) vs Arm E (Unified Tri-Source with Graph) ---")
    print(f"Binary Accuracy: Arm D = {arm_d_vs_e['binary_accuracy']['mean_arm_d']*100:.1f}%, Arm E = {arm_d_vs_e['binary_accuracy']['mean_arm_e']*100:.1f}% (Diff: {arm_d_vs_e['binary_accuracy']['diff']*100:.1f}%, McNemar p = {arm_d_vs_e['binary_accuracy']['mcnemar']['p_value']:.4f})")
    print(f"Fact Coverage: Arm D = {arm_d_vs_e['fact_coverage']['mean_arm_d']*100:.1f}%, Arm E = {arm_d_vs_e['fact_coverage']['mean_arm_e']*100:.1f}% (Diff: +{arm_d_vs_e['fact_coverage']['paired_stats']['mean_diff']*100:.2f}%, Paired t p = {arm_d_vs_e['fact_coverage']['paired_stats']['t_pvalue']:.4f}, Wilcoxon p = {arm_d_vs_e['fact_coverage']['paired_stats']['wilcoxon_pvalue']:.4f})")

if __name__ == "__main__":
    run_phase2_analysis()
