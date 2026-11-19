"""Statistical Significance, Calibration, and Hypothesis Testing for Validation Sprint.

Implements:
1. McNemar's paired test for binary correctness (Exact Match and Binary Accuracy) with continuity correction.
2. Bootstrap paired confidence intervals (1,000 resamples) for differences in means.
3. 95% Confidence Intervals for major metrics (Wilson score for binary rates, Student-t for continuous scores).
4. Effect size calculations (Cohen's d for continuous, Cohen's g / Odds Ratio for paired binary).
5. Holm-Bonferroni correction for family-wise error rate control.
6. Calibration & Abstention analysis:
   - FAR (False Acceptance Rate)
   - FRR (False Rejection Rate)
   - Coverage vs Accuracy trade-offs
"""

import json
import math
import random
from typing import Any, Sequence


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% confidence z-score
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


def bootstrap_mean_diff_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 1000,
    ci: float = 0.95
) -> dict[str, float]:
    """Compute paired bootstrap confidence interval for mean(b) - mean(a)."""
    assert len(scores_a) == len(scores_b), "Paired sequences must have identical length"
    n = len(scores_a)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    observed_mean_diff = sum(diffs) / n

    random.seed(42)
    boot_means = []
    for _ in range(n_resamples):
        sample = [random.choice(diffs) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    lower_idx = int(((1.0 - ci) / 2.0) * n_resamples)
    upper_idx = int((1.0 - ((1.0 - ci) / 2.0)) * n_resamples)

    return {
        "observed_mean_diff": round(observed_mean_diff, 4),
        "ci_lower": round(boot_means[lower_idx], 4),
        "ci_upper": round(boot_means[upper_idx], 4)
    }


def mcnemar_test(
    outcomes_a: Sequence[int],
    outcomes_b: Sequence[int],
    continuity_correction: bool = True
) -> dict[str, Any]:
    """
    Perform McNemar's test for paired 2x2 binary outcomes.
    outcomes: 1 for correct, 0 for incorrect.
    Table:
              B=1    B=0
       A=1    n11    n10 (A correct, B wrong)
       A=0    n01    n00 (A wrong, B correct)
    """
    assert len(outcomes_a) == len(outcomes_b), "Outcomes must have identical length"

    n10 = sum(1 for a, b in zip(outcomes_a, outcomes_b) if a == 1 and b == 0)
    n01 = sum(1 for a, b in zip(outcomes_a, outcomes_b) if a == 0 and b == 1)
    discordant = n10 + n01

    if discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n10": 0,
            "n01": 0,
            "discordant": 0,
            "odds_ratio": 1.0,
            "significant_at_05": False
        }

    # Continuity corrected chi-square: (|n10 - n01| - 1)^2 / (n10 + n01)
    if continuity_correction:
        stat = ((abs(n10 - n01) - 1.0) ** 2) / discordant
    else:
        stat = ((n10 - n01) ** 2) / discordant

    # Approximation of chi2 p-value with 1 df
    # Survival function approximation: P(chi2(1) >= stat) = 2 * (1 - Phi(sqrt(stat)))
    z = math.sqrt(stat)
    p_val = math.erfc(z / math.sqrt(2.0))

    odds_ratio = (n01 / n10) if n10 > 0 else (float("inf") if n01 > 0 else 1.0)
    # Cohen's g for paired proportions: g = |P(b) - P(a)| / 2 or (n01 - n10) / (2 * total)
    total = len(outcomes_a)
    cohens_g = abs(n01 - n10) / (2.0 * discordant) if discordant > 0 else 0.0

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "n10_a_better": n10,
        "n01_b_better": n01,
        "discordant": discordant,
        "odds_ratio": round(odds_ratio, 4) if odds_ratio != float("inf") else "inf",
        "cohens_g": round(cohens_g, 4),
        "significant_at_05": p_val < 0.05
    }


def holm_bonferroni_correction(p_values: list[tuple[str, float]]) -> list[dict[str, Any]]:
    """
    Apply Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER).
    p_values: list of (comparison_name, p_value)
    """
    m = len(p_values)
    sorted_p = sorted(p_values, key=lambda x: x[1])

    results = []
    has_failed = False
    for rank, (name, p) in enumerate(sorted_p, 1):
        alpha_adjusted = 0.05 / (m - rank + 1)
        is_sig = (p <= alpha_adjusted) and not has_failed
        if not is_sig:
            has_failed = True
        results.append({
            "comparison": name,
            "raw_p": p,
            "adjusted_alpha": round(alpha_adjusted, 6),
            "is_significant": is_sig
        })
    return results


def analyze_abstention_and_calibration(predictions: dict[str, Any]) -> dict[str, Any]:
    """
    Compute calibration, over-abstention, and false acceptance statistics.
    """
    n_total = len(predictions)
    n_answerable = sum(1 for p in predictions.values() if p.get("answer_metrics", {}).get("is_answerable", True))
    n_unanswerable = n_total - n_answerable

    # Metrics
    correct_answers = 0
    false_rejections = 0  # Abstained on answerable query
    correct_abstentions = 0  # Abstained on unanswerable query
    false_acceptances = 0  # Attempted answer on unanswerable query

    for p in predictions.values():
        ans_m = p.get("answer_metrics", {})
        is_ans = ans_m.get("is_answerable", True)
        abstained = ans_m.get("abstained", False)
        acc = ans_m.get("binary_accuracy", ans_m.get("accuracy", 0.0))

        if is_ans:
            if abstained:
                false_rejections += 1
            else:
                if acc >= 0.70:
                    correct_answers += 1
        else:
            if abstained:
                correct_abstentions += 1
            else:
                false_acceptances += 1

    far = (false_acceptances / n_unanswerable) if n_unanswerable > 0 else 0.0
    frr = (false_rejections / n_answerable) if n_answerable > 0 else 0.0
    coverage = (n_total - (false_rejections + correct_abstentions)) / n_total if n_total > 0 else 0.0

    return {
        "total_questions": n_total,
        "answerable_count": n_answerable,
        "unanswerable_count": n_unanswerable,
        "correct_answered": correct_answers,
        "false_rejections_overabstention": false_rejections,
        "correct_abstentions": correct_abstentions,
        "false_acceptances_hallucination": false_acceptances,
        "false_acceptance_rate_far": round(far, 4),
        "false_rejection_rate_frr": round(frr, 4),
        "effective_coverage": round(coverage, 4)
    }

