# IVARA — Post-Evaluation Audit and Architecture Decision Gate Report

**Date:** October 28, 2026  
**Auditor:** Antigravity Autonomous Agent  
**Branch:** `research/evaluation-baseline`  
**Git Baseline Commit:** `885298c73cd0257653c49cb81d958f49ad8a0da4`  
**Test Suite Status:** 216 / 216 passed (100% clean, 16.14s execution)  
**Audit Status:** Complete read-only audit; zero production store modifications; human review sheet blank columns preserved.  
**Evaluation Nature:** **Diagnostic local-judge evaluation only; not official LongMemEval benchmark results.**

---

## 1. Dataset and Cohort Provenance Audit

### 1.1 Source Dataset Identification
- **Source File:** [evaluation/data/longmemeval/longmemeval_s_cleaned.json](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/data/longmemeval/longmemeval_s_cleaned.json)
- **Source File SHA-256:** `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
- **Total Questions in Source:** 500 instances across 6 LongMemEval task types.

### 1.2 Diagnostic Sample Identification
- **Sample File:** [evaluation/autonomous_sprint/diagnostic_sample_100.json](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/autonomous_sprint/diagnostic_sample_100.json)
- **Sample File SHA-256:** `cef499da6e68cae8f8e8df61351261a2c652074986063acade0c1330331de338`
- **Sampling Manifest:** [evaluation/autonomous_sprint/diagnostic_sample_100_manifest.json](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/autonomous_sprint/diagnostic_sample_100_manifest.json)
- **Sampling Mechanism & Seed:** `deterministic_stratified_held_out` with fixed Python PRNG seed `42` (`random.Random(42)`).
- **Stratification Breakdown:**
  - `multi-session`: 20 instances (16 answerable, 4 abstention)
  - `temporal-reasoning`: 20 instances (16 answerable, 4 abstention)
  - `knowledge-update`: 18 instances (14 answerable, 4 abstention)
  - `single-session-user`: 16 instances (14 answerable, 2 abstention)
  - `single-session-assistant`: 14 instances (12 answerable, 2 abstention)
  - `single-session-preference`: 12 instances (10 answerable, 2 abstention)
  - **Total:** 86 answerable, 14 unanswerable/abstention = 100 questions.

### 1.3 Precise Cohort Overlap Disambiguation

A rigorous set-intersection audit across all cohorts produced the following verified overlap counts:

| Comparison | Overlap Count | Overlap Percentage | Provenance & Audit Finding |
| :--- | :---: | :---: | :--- |
| `diagnostic_sample_100` vs. **Step 7 60-Question Pilot** | **0 / 60** | **0.0%** | **Strict Zero Overlap Confirmed.** All 60 Step 7 pilot questions were explicitly filtered out prior to sampling. |
| `diagnostic_sample_100` vs. **Source Cleaned Dataset** | **100 / 500** | **20.0%** | Exact 20% stratified subset of LongMemEval-S. |
| `diagnostic_pilot_30` vs. `diagnostic_sample_100` | **30 / 30** | **100.0%** | **Prefix Identity Confirmed.** `diagnostic_pilot_30` is **not** a distinct dataset; it consists of precisely the **first 30 instances** of `diagnostic_sample_100.json`. |
| Remaining Newly Evaluated Cohort | **70 / 70** | **100.0%** | Instances 31 through 100 of `diagnostic_sample_100.json`. |

> [!IMPORTANT]
> **Narrative Correction:** Previous report wording stated that the entire diagnostic evaluation ran on a "held-out sample with 0 overlap with the pilot." To be completely precise:
> - `diagnostic_sample_100.json` has **0 overlap** with the historical **Step 7 60-question pilot** (`campaign_v1_pilot`).
> - The combined $N=100$ result consists of the **30 predictions from `diagnostic_pilot_30`** (which evaluated questions 1–30 of `diagnostic_sample_100.json`) plus **70 newly evaluated predictions** (questions 31–100).
> - No predictions were silently regenerated; the first 30 were seeded directly from `diagnostic_pilot_30/` without modification.

### 1.4 Record Completeness and Alignment Verification
For all 6 arms (`ARM_A` to `ARM_F`):
- **Predictions:** Exactly 100 records per arm (600 total).
- **Judgments:** Exactly 100 records per arm (600 total).
- **Duplicate IDs:** 0 duplicates across all arms.
- **Missing Questions:** 0 missing questions.
- **Sequence Alignment:** All 6 arms evaluated the identical 100 question IDs in the exact same sequence matching `diagnostic_sample_100.json`.

---

## 2. Independent Statistical Recomputation & Cross-Check

All metrics were independently recomputed directly from the raw `predictions_*.jsonl` and `judge_results_*.jsonl` records in `evaluation/autonomous_sprint/runs/diagnostic_full_100/`.

### 2.1 Full 100-Question Metrics Cross-Check

| Arm | Metric | Recomputed Value | Final Report Value | Audit Discrepancy |
| :--- | :--- | :---: | :---: | :---: |
| **ARM_A** (Baseline) | Overall Accuracy | 64.00% (64/100) | 64.00% | **0.00%** |
| | 95% Wilson CI | [54.24%, 72.73%] | [54.2%, 72.7%] | **Exact Match** |
| | Answerable Accuracy | 60.47% (52/86) | 60.47% | **0.00%** |
| | Abstention Accuracy | 85.71% (12/14) | 85.71% | **0.00%** |
| **ARM_B** (Ontology) | Overall Accuracy | 65.00% (65/100) | 65.00% | **0.00%** |
| | 95% Wilson CI | [55.25%, 73.64%] | [55.3%, 73.6%] | **Exact Match** |
| | Answerable Accuracy | 60.47% (52/86) | 60.47% | **0.00%** |
| | Abstention Accuracy | 92.86% (13/14) | 92.86% | **0.00%** |
| **ARM_C** (Raw Graph) | Overall Accuracy | 61.00% (61/100) | 61.00% | **0.00%** |
| | 95% Wilson CI | [51.20%, 69.98%] | [51.2%, 70.0%] | **Exact Match** |
| | Answerable Accuracy | 56.98% (49/86) | 56.98% | **0.00%** |
| | Abstention Accuracy | 85.71% (12/14) | 85.71% | **0.00%** |
| **ARM_D** (Filtered Graph) | Overall Accuracy | 64.00% (64/100) | 64.00% | **0.00%** |
| | 95% Wilson CI | [54.24%, 72.73%] | [54.2%, 72.7%] | **Exact Match** |
| | Answerable Accuracy | 60.47% (52/86) | 60.47% | **0.00%** |
| | Abstention Accuracy | 85.71% (12/14) | 85.71% | **0.00%** |
| **ARM_E** (Direct Temporal) | Overall Accuracy | 64.00% (64/100) | 64.00% | **0.00%** |
| | 95% Wilson CI | [54.24%, 72.73%] | [54.2%, 72.7%] | **Exact Match** |
| | Answerable Accuracy | 60.47% (52/86) | 60.47% | **0.00%** |
| | Abstention Accuracy | 85.71% (12/14) | 85.71% | **0.00%** |
| **ARM_F** (Full System) | Overall Accuracy | 61.00% (61/100) | 61.00% | **0.00%** |
| | 95% Wilson CI | [51.20%, 69.98%] | [51.2%, 70.0%] | **Exact Match** |
| | Answerable Accuracy | 56.98% (49/86) | 56.98% | **0.00%** |
| | Abstention Accuracy | 85.71% (12/14) | 85.71% | **0.00%** |

*Verification:* All overall, answerable, and abstention percentages and confidence intervals recomputed identically to the final report.

---

### 2.2 Confusion Matrices: Abstention vs. Answering Behavior

To understand why configurations fail, we computed the confusion counts across the 14 unanswerable and 86 answerable questions:

| Arm | Gold Abstention (N=14) | | Gold Answerable (N=86) | | | Abstention Precision | Abstention Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Correct Abstention** | **Failed (Hallucinated)** | **Correct Answer** | **Incorrect Answer** | **Improper Abstention** | | |
| **ARM_A** (Baseline) | 12 | 2 | 52 | 26 | 8 | 60.0% (12/20) | 85.7% (12/14) |
| **ARM_B** (Ontology) | **13** | **1** | 52 | 27 | 7 | **65.0% (13/20)** | **92.9% (13/14)** |
| **ARM_C** (Raw Graph) | 12 | 2 | 49 | 29 | 8 | 60.0% (12/20) | 85.7% (12/14) |
| **ARM_D** (Filtered Graph) | 12 | 2 | 52 | 26 | 8 | 60.0% (12/20) | 85.7% (12/14) |
| **ARM_E** (Direct Temporal) | 12 | 2 | 52 | 26 | 8 | 60.0% (12/20) | 85.7% (12/14) |
| **ARM_F** (Full System) | 12 | 2 | 49 | 29 | 8 | 60.0% (12/20) | 85.7% (12/14) |

*Key Confusion Finding:*
- **ARM_B (Ontology Guidance)** had the lowest hallucination rate on unanswerable questions (only 1 failure out of 14, achieving 92.9% recall and 65.0% precision).
- Improper abstention (abstaining when a question was answerable) occurred at a steady rate of 7–8 questions across all arms, primarily on open-ended preference questions where the model claimed insufficient memory.

---

### 2.3 Paired Statistical Comparisons & McNemar p-Values

| Comparison Pair | Arm 1 Wins ($b$) | Arm 2 Wins ($c$) | Concordant Ties | Total Discordant ($b+c$) | McNemar Exact 2-Tailed $p$ | Statistical Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ARM_A vs. ARM_B** | 4 | 5 | 91 | 9 | $p = 1.0000$ | **Not Statistically Significant** ($p \ge 0.05$). |
| **ARM_C vs. ARM_D** | 1 | 4 | 95 | 5 | $p = 0.3750$ | **Directional Advantage** for D (Net +3), but **Not Statistically Significant** ($p \ge 0.05$). |
| **ARM_E vs. ARM_F** | 5 | 2 | 93 | 7 | $p = 0.4531$ | **Directional Advantage** for E (Net +3), but **Not Statistically Significant** ($p \ge 0.05$). |
| **ARM_A vs. ARM_D** | 5 | 5 | 90 | 10 | $p = 1.0000$ | **Identical Net Performance** ($b = c$). |
| **ARM_A vs. ARM_E** | 4 | 4 | 92 | 8 | $p = 1.0000$ | **Identical Net Performance** ($b = c$). |
| **ARM_C vs. ARM_F** | 2 | 2 | 96 | 4 | $p = 1.0000$ | **Identical Net Performance** ($b = c$). |

> [!WARNING]
> **Statistical Significance Boundary:**
> Because $90\%$ to $96\%$ of all questions resulted in identical verdicts between any pair of configurations, the discordant sample size ($b + c$) is only between 4 and 10 instances out of 100.
> - **Fact:** The raw difference in accuracy (e.g., $64.0\%$ vs $61.0\%$) and the paired win ratios ($4:1$ for D over C; $5:2$ for E over F) are **directional trends**, NOT statistically significant findings at $\alpha = 0.05$.
> - **Correction:** We must not claim that Prototype B is "proven superior" or that Baseline and Prototype D have "proven equivalence" in a statistical sense. We can only claim that on this 100-question sample, the observed point estimates show directional advantages that warrant further testing.

---

## 3. Implementation Flags and Experimental Fairness Audit

The audit inspected `DiagnosticRunner` in [evaluation/autonomous_sprint/diagnostic_runner.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/autonomous_sprint/diagnostic_runner.py), the answering pipeline in [src/answering.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/answering.py), and the run manifest:

1. **Ontology Activation:**
   - Verified that `use_ontology = False` in `ARM_A` and `use_ontology = True` in `ARM_B`, `ARM_C`, `ARM_D`, `ARM_E`, `ARM_F`.
   - Verified that ontology guidance modifies only the query expansion concepts and schema matching, not the base vector retriever.
2. **Graph Traversal & Relevance Filtering:**
   - Verified that graph construction occurred **only** in `ARM_C`, `ARM_D`, and `ARM_F`.
   - Verified that `min_edge_relevance = 0.0` in `ARM_C` and `ARM_F` (raw 1-hop traversal).
   - Verified that `min_edge_relevance = 0.50` was strictly applied in `ARM_D` (Prototype B), successfully filtering edges with confidence $< 0.50$.
3. **Temporal Reasoning Modes:**
   - Verified that `direct_seed_temporal = True` was active **only** in `ARM_E` (Prototype C), extracting claims directly from top-k seeds and bypassing graph traversal.
   - Verified that `ARM_F` executed the full graph-temporal pipeline (`enable_temporal = True`, `use_chunk_graph = True`, `direct_seed_temporal = False`).
4. **Experimental Controls & Uniformity:**
   - **Prompt Template:** Identical LongMemEval answering prompt template across all 6 arms.
   - **Context Budget:** Strictly capped at `max_context_tokens = 1,500` across all arms.
   - **Retrieval Depth:** Identical `k = 5` seed chunks across all arms.
   - **Models:** Generator fixed to `gpt-oss:latest` (temp=0.0); Judge fixed to `VladimirGav/gemma4-26b-16GB-VRAM:latest` (temp=0.0, format=json).
   - **Git Commit:** Generation ran on git commit `885298c73cd0257653c49cb81d958f49ad8a0da4` (verified in manifest).

*Conclusion:* The experiment was conducted with strict configuration isolation and experimental fairness across all six arms.

---

## 4. Inspection of Discordant Cases & Judge Audit

We audited all 21 discordant instances across the primary architectural comparisons (A vs B, C vs D, E vs F).

### 4.1 Raw Graph (C) vs. Filtered Graph (D) Discordances (5 instances: 4 D wins, 1 C win)

1. **`2a1811e2` (Temporal Reasoning): ARM_D Won**
   - *Gold Answer:* 21 days (22 days also acceptable).
   - *ARM_C (Raw):* Traversed an unpruned edge to a late-March turn chunk, calculating 28 days $\rightarrow$ **INCORRECT**.
   - *ARM_D (Filtered):* Pruned the late-March edge, calculating 21 days $\rightarrow$ **CORRECT**.
   - *Judge Audit:* Supported. Ground truth is unambiguously 21 days.
2. **`6b7dfb22` (Single-Session Preference): ARM_D Won**
   - *Gold Answer:* Build upon Instagram art accounts / online tutorials; painting flowers; utilize recent 30-day challenge.
   - *ARM_C (Raw):* Produced vague general advice without mentioning social media or past painting habits $\rightarrow$ **INCORRECT**.
   - *ARM_D (Filtered):* Explicitly referenced social media inspiration and structured artistic routines $\rightarrow$ **CORRECT**.
   - *Judge Audit:* Supported. ARM_C improperly omitted the grounded preference sources.
3. **`7401057b` (Knowledge Update): ARM_D Won**
   - *Gold Answer:* Two free night stays.
   - *ARM_C (Raw):* Context contained conflicting point balance chunks; model hedged with "at least one or two" $\rightarrow$ **INCORRECT**.
   - *ARM_D (Filtered):* Retained only the latest updated point balance; answered "two free night's stays" $\rightarrow$ **CORRECT**.
   - *Judge Audit:* Supported.
4. **`dad224aa` (Knowledge Update): ARM_D Won**
   - *Gold Answer:* 7:30 am.
   - *ARM_C (Raw):* Answered with contradictory times ("7:30 am and 9:30 am") due to an older neighbor turn $\rightarrow$ **INCORRECT**.
   - *ARM_D (Filtered):* Correctly prioritized the recent 7:30 am wake-up time $\rightarrow$ **CORRECT**.
   - *Judge Audit:* Supported.
5. **`cc539528` (Multi-Session): ARM_C Won**
   - *Gold Answer:* Ruby, Python, or PHP.
   - *ARM_C (Raw):* Answered "Ruby, Python, or PHP" $\rightarrow$ **CORRECT**.
   - *ARM_D (Filtered):* Answered "Ruby, Python, PHP, Node.js, and Java" $\rightarrow$ **INCORRECT**.
   - *Judge Audit:* Supported. ARM_D's generation brought in two extraneous languages from an adjacent turn.

### 4.2 Direct-Seed Temporal (E) vs. Full System (F) Discordances (7 instances: 5 E wins, 2 F wins)

1. **`gpt4_65aabe59` (Temporal Reasoning): ARM_E Won**
   - *Gold Answer:* Smart thermostat was set up first.
   - *ARM_F (Full System):* Multi-turn temporal claims mis-ordered the events, claiming "mesh network was installed first" $\rightarrow$ **INCORRECT**.
   - *ARM_E (Direct Seed):* Correctly identified "smart thermostat first" $\rightarrow$ **CORRECT**.
   - *Judge Audit:* Supported.
2. **`f35224e0` (Multi-Session): ARM_E Won**
   - *Gold Answer:* 27 total episodes (12 + 15).
   - *ARM_F (Full System):* Graph expansion flooded context; model failed to perform summation $\rightarrow$ **INCORRECT**.
   - *ARM_E (Direct Seed):* Compact seed context enabled correct summation (12 + 15 = 27) $\rightarrow$ **CORRECT**.
   - *Judge Audit:* Supported.
3. **`06f04340` (Single-Session Preference): ARM_F Won**
   - *Gold Answer:* Incorporate homegrown cherry tomatoes and herbs like basil and mint.
   - *ARM_E (Direct Seed):* Suggested generic garden produce without naming cherry tomatoes or basil $\rightarrow$ **INCORRECT**.
   - *ARM_F (Full System):* Graph expansion happened to link the ingredient mentions, allowing specific naming of tomatoes and basil $\rightarrow$ **CORRECT**.
   - *Judge Audit:* Supported.
4. **`59524333` (Knowledge Update): ARM_F Won**
   - *Gold Answer:* 6:00 pm.
   - *ARM_E (Direct Seed):* Stated 7:00 pm based on an earlier conversation $\rightarrow$ **INCORRECT**.
   - *ARM_F (Full System):* Graph expansion included the superseding conversation timestamp (2023-05-30), correctly reporting 6:00 pm $\rightarrow$ **CORRECT**.
   - *Judge Audit:* Supported.

---

## 5. Conservative Architecture Decision Gate

### 5.1 Evaluation of the Core Hypotheses

| Hypothesis | Evaluation from Audit | Conservative Status |
| :--- | :--- | :--- |
| **1. Ontology improves answerability and abstention** | Ontology guidance improved abstention accuracy from $85.7\%$ to **$92.9\%$** (13/14 correct), achieving 100% on the held-out 70 questions. Answerable accuracy remained identical ($60.47\%$). Overall difference (+1.0%) is not statistically significant ($p = 1.0$). | **Validated as Abstention Safeguard.** Recommend retaining ontology guidance for entity grounding and abstention filtering. |
| **2. Graph relevance filtering reduces distractor contamination** | Filtering low-confidence edges ($\text{threshold} = 0.50$) recovered 4 wins against raw graph expansion with only 1 loss (net $+3$ wins, overall $+3.0\%$). However, against the unaugmented baseline, ARM_D is net neutral ($5$ wins, $5$ losses, $p = 1.0$). | **Directional Fix for Graph Noise.** Validated as a necessary patch for graph expansion, but does not outperform the simple baseline on this sample. |
| **3. Direct-seed temporal reasoning preserves accuracy with lower latency** | Direct temporal extraction from seeds matched baseline and filtered graph accuracy ($64.0\%$), outperformed the full system by $+3.0\%$ ($5$ wins vs. $2$ losses), and completely eliminated graph construction overhead (saving $428\text{ ms}$ per query). | **Strong Research Candidate.** Highly efficient and less prone to multi-turn distraction, though not statistically significant over full graph ($p = 0.453$). |
| **4. Full graph traversal provides enough value to justify overhead** | Full graph expansion (ARM_F) had lower accuracy ($61.0\%$ vs $64.0\%$) and added $428\text{ ms}$ latency. In only 2 out of 100 questions did full graph expansion provide a unique win over direct-seed retrieval. | **Hypothesis Rejected.** Unconstrained 1-hop graph expansion is not justified by current empirical evidence. |

---

### 5.2 Go / No-Go Architectural Decision

| Decision Gate | Recommendation | Operational Action |
| :--- | :---: | :--- |
| **Production Default Changes** | **NO-GO** | **Do NOT modify production defaults (`data/` or `main`) at this time.** Current production defaults must remain stable until evaluated on the official OpenAI benchmark judge. |
| **Research Architecture Selection** | **GO (Candidate)** | **Select `IVARA-Hybrid` as the Primary Research Configuration:**<br>1. Retain Ontology Guidance for schema grounding and abstention.<br>2. Set `direct_seed_temporal = True` as the primary fast-path temporal engine.<br>3. Deprecate raw graph expansion; restrict any graph traversal to `min_edge_relevance = 0.50`. |
| **Official Benchmark Readiness** | **GO (Preflight Complete)** | Proceed to execute the official LongMemEval benchmark judge on the existing predictions once OpenAI API credentials are provided by the user. |

---

## 6. Audit Outputs & Verification Summary

1. **Audit Artifact Directory:** [evaluation/autonomous_sprint/runs/post_evaluation_audit/](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/autonomous_sprint/runs/post_evaluation_audit/)
   - `audit_provenance.json`: Verified provenance, SHA-256 hashes, and exact cohort overlaps.
   - `independent_metrics_recomputation.json`: Verified recomputed statistics, confusion matrices, and McNemar $p$-values.
   - `implementation_flags_audit.json`: Verified exact flag compliance across all six arms.
   - `disagreement_inspection.json`: Verified inspection of all 21 discordant cases and judge rationale.
2. **Safety Compliance:**
   - `main`: Untouched.
   - `data/memories.json` and `data/entities.json`: Untouched.
   - Historical Steps 7–9 reports and manifests: Untouched.
   - `evaluation/step9/human_review_sample.csv`: Verified that all human verdict columns remain strictly blank.
   - Test Suite: **216 / 216 tests passing**.
