# IVARA — LongMemEval-S Benchmark Evaluation Results

**Benchmark Suite:** LongMemEval-S Cleaned (`longmemeval_s_cleaned.json`, 500 questions across 6 official tasks)  
**Evaluator:** Official LongMemEval LLM-as-a-Judge prompt (`qwen2.5:14b`, `temperature=0.0`)  
**Execution Environment:** Local inference via Ollama (`http://localhost:11434`, `gpt-oss`, `temperature=0.0`)  
**Scope Note:** These results evaluate the **Config D batch benchmark pipeline** (`evaluation/runners/run_longmemeval.py`, `use_chunking=True`, $W=3, O=1$, `question_date` provided). They do **not** represent current interactive CLI chat (`main.py chat`) performance, which uses a separate unchunked `UnifiedRetriever` path (see [`ARCHITECTURE.md`](../ARCHITECTURE.md)).

---

## 1. Overall 500-Question Performance Summary

Metrics below are computed directly from the preserved JSONL outputs in `evaluation/data/longmemeval/runs/final_bakeoff/`:
- **Candidate E (`candidate_e_baseline`):** Config D (`55a273b`) baseline RAG (reused from `integrated_production_run`).
- **Candidate C (`candidate_c_phase3`):** Config D (`55a273b`) `QueryAwareRouter` + proportional session budgeting (reused 133 temporal predictions from `temporal_phase3_experiments` + 367 fresh).
- **Candidate B (`candidate_b_arm_b`):** Config D (`55a273b`) Flat MMR ($\lambda = 0.70$, $\le 2$ chunks/session) + TACC calendar arithmetic + regex/JSON recovery parser (reused 133 temporal predictions from `temporal_phase2b2_experiments` + 367 fresh).

| Architecture | Overall Accuracy ($N=500$) | Macro Task Avg (6 tasks) | Answerable ($N=470$) | Negative Controls ($N=30$ `_abs`) | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Candidate B** | **58.40% (292/500)** | **61.34%** | **56.60% (266/470)** | 86.67% (26/30) | 7.46 s |
| **Candidate C** | 56.80% (284/500) | 60.54% | 54.47% (256/470) | **93.33% (28/30)** | **7.25 s** |
| **Candidate E (Baseline)** | 40.80% (204/500) | 45.10% | 37.87% (178/470) | 86.67% (26/30) | 9.92 s |
| **Net Gain (B vs. E)** | **+17.60 pp (+88 Qs)** | **+16.24 pp** | **+18.73 pp (+88 Qs)** | 0.00 pp | **-2.46 s (-24.8%)** |

---

## 2. Official Task-by-Task Breakdown ($N = 500$)

The 30 abstention (`_abs`) negative-control questions are pooled inside four of the six official tasks (`single-session-user`: 6, `multi-session`: 12, `temporal-reasoning`: 6, `knowledge-update`: 6) and are not a disjoint seventh category.

| Official Task | Total $N$ | Candidate E (Baseline) | Candidate C (Query-Aware) | Candidate B (Arm B) | B vs. E Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `single-session-user` | 70 | 82.86% (58/70) | 90.00% (63/70) | **92.86% (65/70)** | +10.00 pp |
| `single-session-assistant` | 56 | 58.93% (33/56) | 89.29% (50/56) | **91.07% (51/56)** | +32.14 pp |
| `single-session-preference` | 30 | 33.33% (10/30) | **40.00% (12/30)** | 33.33% (10/30) | 0.00 pp |
| `multi-session` | 133 | 31.58% (42/133) | **36.09% (48/133)** | **36.09% (48/133)** | +4.51 pp |
| `knowledge-update` | 78 | 43.59% (34/78) | 58.97% (46/78) | **62.82% (49/78)** | +19.23 pp |
| `temporal-reasoning` | 133 | 20.30% (27/133) | 48.87% (65/133) | **51.88% (69/133)** | +31.58 pp |
| **Total / Macro Task Avg** | **500** | **40.80% (45.10%)** | **56.80% (60.54%)** | **58.40% (61.34%)** | **+17.60 pp (+16.24 pp)** |

---

## 3. Temporal Reasoning Subsets & Category H ($N = 133$)

In IVARA's temporal analyses, the 133 `temporal-reasoning` questions partition into:
- **Category H ($N=41$):** Curated multi-session temporal aggregation subset requiring ordering or comparing events across distinct sessions.
- **Non-Category H ($N=86$):** Single-session temporal reasoning, date arithmetic, and ordering.
- **Abstentions ($N=6$):** Unanswerable temporal negative controls.

| Temporal Sub-Category | $N$ | Candidate E (Baseline) | Candidate B (Arm B) | Candidate C (Query-Aware) |
| :--- | :---: | :---: | :---: | :---: |
| **Category H (Multi-Session Temporal)** | 41 | 0.00% (0/41) | 7.32% (3/41) | **19.51% (8/41)** |
| **Non-Category H (Single-Session Temporal)** | 86 | 25.58% (22/86) | **70.93% (61/86)** | 59.30% (51/86) |
| **Temporal Abstentions** | 6 | 83.33% (5/6) | 83.33% (5/6) | **100.00% (6/6)** |
| **All Temporal Reasoning** | **133** | **20.30% (27/133)** | **51.88% (69/133)** | **48.87% (65/133)** |

*Qualification:* Candidate C recovered +5 Category H questions over Candidate B, but lost 10 Non-Category H single-session questions due to session dispersion from routing misclassifications, yielding a net -4 deficit on temporal reasoning.

---

## 4. Paired Statistical Significance (McNemar Test, Edwards Continuity Correction, $N=500$)

### Candidate B vs. Candidate E
- **Both Correct:** 192 | **Candidate B Only:** 100 | **Candidate E Only:** 12 | **Both Incorrect:** 196
- **$\chi^2$ ($df=1$):** **67.5804** ($p = 2.02 \times 10^{-16}$)

### Candidate B vs. Candidate C
- **Both Correct:** 257 | **Candidate B Only:** 35 | **Candidate C Only:** 27 | **Both Incorrect:** 181
- **$\chi^2$ ($df=1$):** **0.7903** ($p = 0.3740$, not statistically significant)

---

## 5. Retrieval Recall@5, Evidence Coverage & Latency

*Note:* `recall_at_5` measures whether **at least one** gold evidence session appears in the top-5 retrieved chunks over the 470 answerable questions; it does **not** guarantee complete retrieval of every session required for multi-session aggregation.

| Architecture | Mean Latency | Median Latency | Min / Max Latency | Recall@5 ($N=470$) | Mean Evidence Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Candidate B** | 7.46 s | 6.83 s | 4.40 s / 22.01 s | **95.32% (448/470)** | **88.14%** |
| **Candidate C** | **7.25 s** | **6.80 s** | 2.96 s / 17.71 s | **95.32% (448/470)** | 87.73% |
| **Candidate E** | 9.92 s | 9.34 s | 5.90 s / 20.22 s | 94.26% (443/470) | 82.49% |

*Post-Retrieval Bottleneck:* Of Candidate B's 204 failing answerable questions, **184 (90.20%)** had `recall_at_5 == True` (primarily `insufficient_evidence` refusals [103], wrong `answered` [38], wrong `conflicting_evidence` [29], and incomplete `partial` [14]).

---

## 6. Preserved Run Artifacts & SHA-256 Checksums

Checksums identify the exact local files stored under `evaluation/data/longmemeval/runs/final_bakeoff/`:

| Candidate | Prediction File | Size | SHA-256 |
| :--- | :--- | :---: | :--- |
| **Candidate B** | `candidate_b_arm_b/results_config_D.jsonl` | 836,988 B | `ba9e71bec593c71f76d802fe154d21321776defd9fc808f6ade735867e90a9b9` |
| **Candidate C** | `candidate_c_phase3/results_config_D.jsonl` | 845,005 B | `58062320b1c0424d78aa90c84a79866ccbfd33bb233ee1e028b0e4a2ee04a2ee` |
| **Candidate E** | `candidate_e_baseline/results_config_D.jsonl` | 738,071 B | `914227ae70df1b8e00ba421492249e3b65d22fc0acf477b784a81ff8503ae3a3` |
