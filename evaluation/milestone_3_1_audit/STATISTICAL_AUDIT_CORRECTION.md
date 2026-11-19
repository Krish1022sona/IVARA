# IVARA Milestone 3.1: Statistical Audit Addendum & Correction Note

**Date:** November 15, 2026  
**Subject:** Formal Correction of Discordant Cell Counts in Milestone 3.1 Paired McNemar Analysis  
**Reference Document:** `MILESTONE_3_1_AUDIT_REPORT.md` (Section 2.1)  
**Status:** Approved Formal Addendum  

---

## 1. Summary of Discrepancy & Root Cause

In Section 2.1 of `MILESTONE_3_1_AUDIT_REPORT.md`, the summary table reported paired significance tests for Binary Accuracy against Arm A across 60 stratified held-out test questions ($N = 60$).

A transcription error occurred in the display text for the discordant cell counts column (`Statistic / Discordant`):
- **Reported Text in Markdown Table:**
  - Arm B: `Discordant (11 vs 3)`
  - Arm D: `Discordant (9 vs 3)`
  - Arm E: `Discordant (9 vs 3)`
- **True Evaluated Counts in `phase2_statistical_analysis.json`:**
  - Arm B: $b = 10$ wins, $c = 2$ losses ($10 - 2 = 8$ net gain, $+13.33\%$).
  - Arm D: $b = 7$ wins, $c = 1$ loss ($7 - 1 = 6$ net gain, $+10.00\%$).
  - Arm E: $b = 8$ wins, $c = 2$ losses ($8 - 2 = 6$ net gain, $+10.00\%$).

### Why the Error Occurred
The net gains ($+8, +6, +6$) for 60 questions evaluate to:
- $+8 / 60 = +13.33\%$ (Arm B)
- $+6 / 60 = +10.00\%$ (Arm D)
- $+6 / 60 = +10.00\%$ (Arm E)

In manual drafting of the markdown table text, hypothetical pairs with the same net differences ($11 - 3 = 8$, $9 - 3 = 6$, $9 - 3 = 6$) were transcribed into the table string. However, the exact $p$-values reported in the table (`0.0386`, `0.0703`, `0.1094`) were generated directly by Python (`scipy.stats.binomtest`) from the **true counts** ($10\text{ vs }2$, $7\text{ vs }1$, $8\text{ vs }2$).

---

## 2. Statistical Impact of the Correction

Evaluating two-sided exact McNemar binomial tests under both sets of numbers reveals why this correction is critical:

| Arm | Metric | True Counts ($b \text{ vs } c$) | True Two-Sided $p$-value | Typographical Counts ($b \text{ vs } c$) | Typographical $p$-value | Significance Impact |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm B** (Episodic Only) | Binary Accuracy | **10 vs 2** ($N_{\text{disc}}=12$) | $\mathbf{0.0386}$ ($< 0.05$) | 11 vs 3 ($N_{\text{disc}}=14$) | $0.0574$ ($> 0.05$) | **Crucial:** True counts achieve statistical significance ($\alpha=0.05$); typographical counts fall above threshold. |
| **Arm D** (Episodic + Semantic) | Binary Accuracy | **7 vs 1** ($N_{\text{disc}}=8$) | $\mathbf{0.0703}$ ($< 0.10$) | 9 vs 3 ($N_{\text{disc}}=12$) | $0.1460$ ($> 0.10$) | True counts indicate marginal significance ($\alpha=0.10$); typographical counts lose all significance. |
| **Arm E** (Unified Tri-Source) | Binary Accuracy | **8 vs 2** ($N_{\text{disc}}=10$) | $\mathbf{0.1094}$ ($> 0.10$) | 9 vs 3 ($N_{\text{disc}}=12$) | $0.1460$ ($> 0.10$) | Both reflect non-significance ($\alpha=0.05$). |

### Exact Calculations:
$$\text{Arm B: } P(X \ge 10 \lor X \le 2 \mid n=12, p=0.5) = 2 \times \sum_{k=10}^{12} \binom{12}{k} (0.5)^{12} = 2 \times \frac{66 + 12 + 1}{4096} = \frac{158}{4096} = \mathbf{0.038574}$$
$$\text{Arm D: } P(X \ge 7 \lor X \le 1 \mid n=8, p=0.5) = 2 \times \sum_{k=7}^{8} \binom{8}{k} (0.5)^8 = 2 \times \frac{8 + 1}{256} = \frac{18}{256} = \mathbf{0.0703125}$$
$$\text{Arm E: } P(X \ge 8 \lor X \le 2 \mid n=10, p=0.5) = 2 \times \sum_{k=8}^{10} \binom{10}{k} (0.5)^{10} = 2 \times \frac{45 + 10 + 1}{1024} = \frac{112}{1024} = \mathbf{0.109375}$$

---

## 3. Unanswerable Category Confidence Bound Correction

The Milestone 3 and Milestone 3.1 reports noted that all arms achieved **100% accuracy on the unanswerable question category**.

**Critical Scientific Correction:**
- The stratified benchmark evaluated exactly **5 unanswerable questions** ($n=5$).
- While observing $5/5$ correct abstentions is encouraging, the Wilson score 95% confidence interval for $5/5$ is:
  $$\text{Wilson 95\% CI: } \left[ \frac{5 + \frac{1.96^2}{2} - 1.96\sqrt{\frac{0}{5} + \frac{1.96^2}{20}}}{1 + \frac{1.96^2}{5}}, 1.0 \right] = \mathbf{[56.6\%, 100.0\%]}$$
- **Conclusion:** A sample of 5 questions is statistically underpowered. It is improper to claim "proven zero hallucination" or "perfect abstention" from an $n=5$ sample with a lower confidence bound of 56.6%. The evaluation protocol in Milestone 3.2 must scale unanswerable testing to at least $n=25\text{ to }30$ queries.

---

## 4. Preservation Statement

`MILESTONE_3_1_AUDIT_REPORT.md` is preserved in its original form as a historical audit record. This document (`STATISTICAL_AUDIT_CORRECTION.md`) and the comprehensive design document `MILESTONE_3_2_DESIGN.md` serve as the authoritative statistical baseline.

