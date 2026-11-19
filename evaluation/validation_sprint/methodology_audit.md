# Benchmark Methodology Audit: Previous IVARA vs Conventional RAG Evaluation

**Document Status:** Independent Audit  
**Date:** November 2026  
**Subject:** 50-Question Synthetic Narrative Benchmark (`evaluation/story_benchmark/`)  
**Active Branch:** `research/evaluation-baseline`  

---

## 1. Executive Summary & Audit Mandate

The previous benchmark report presented the following metrics across 50 questions evaluated on a local Ollama model (`gpt-oss:latest`):

| Configuration | Reported Accuracy | Reported Exact Match | Reported Recall@5 | Reported MRR | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conventional Vector RAG** | 16.17% | 12.00% | 98.00% | 0.940 | 1,997 ms |
| **Conventional Hybrid RAG** | 16.50% | 14.00% | 100.00% | 0.977 | 2,096 ms |
| **IVARA Hybrid (No Graph)** | 33.70% | 20.00% | 88.00% | 0.840 | 3,628 ms |
| **IVARA Full System** | 39.02% | 24.00% | 92.00% | 0.895 | 3,402 ms |

This independent audit conducted a line-by-line inspection of:
- `evaluation/story_benchmark/dataset.py`
- `evaluation/story_benchmark/baseline_rag.py`
- `evaluation/story_benchmark/runner.py`
- `evaluation/story_benchmark/metrics.py`
- Raw prediction files: `predictions_baseline_vector.json`, `predictions_baseline_hybrid.json`, `predictions_ivara_hybrid_no_graph.json`, `predictions_ivara_full.json`

The audit investigated whether these findings represent true architectural superiority or whether they stem from methodological imbalances, scoring anomalies, or artificial baseline handicaps.

---

## 2. Point-by-Point Investigation

### 2.1 Equivalence of Source Information
**Finding:** **Severe Asymmetry Identified (Artificial Baseline Handicap)**
- **Conventional RAG Input:** Conventional RAG chunked the raw session transcripts into lines formatted as:
  ```
  [2024-01-15] User: I was born in Chandigarh, though my family moved around a bit...
  Assistant: That's interesting! What was it like growing up in Chandigarh?
  ```
- **IVARA Input:** IVARA ingested user turns through `engine.process_turn(...)`, extracting atomic facts into `memories.json` while simultaneously mapping first-person pronouns (`I`, `me`, `my`) to the canonical entity record `SELF` (seeded in `entities.json` with `aliases: ["I", "me", "my", "myself"]`).
- **Consequence:** The questions in the benchmark asked about "Arjun Mehta" or "Arjun" (e.g., `q01`: *"In which city was Arjun Mehta born?"*). In the conventional RAG chunks, the name "Arjun" was absent; only the token "User: I..." was present. Because the RAG system prompt strictly prohibited extrapolating or inventing facts not directly stated, the LLM faithfully refused to assume that the generic "User" was named "Arjun Mehta", forcing an abstention.
- **Verdict:** The low 16% accuracy of conventional RAG was not primarily caused by vector retrieval failure (which had 98–100% Recall@5), but by **deliberately or inadvertently withholding speaker attribution metadata** (`User = Arjun Mehta`) from the baseline.

### 2.2 Equivalence of Generation Models, Prompts, and Context Budgets
**Finding:** **Prompt Discrepancy with Asymmetric Constraints**
- Both systems utilized the same local Ollama model (`gpt-oss:latest`) with `temperature=0.0`.
- Both generated structured JSON responses.
- However, the system prompts were fundamentally different:
  - **Baseline RAG Prompt (`RAG_SYSTEM_PROMPT`)**:
    > "Answer using ONLY facts directly stated or logically inferred in the provided evidence. NEVER invent personal facts, dates, project details, or colleague names not in the evidence... If the provided evidence is completely irrelevant or contains no pertinent facts, set status to 'insufficient_evidence'."
  - **IVARA Answering Prompt (`ANSWERING_SYSTEM_PROMPT`)**:
    > "You are IVARA's evidence-grounded personal memory answering assistant... If the question asks for a count, sequence, time elapsed, or comparison of events or items described in the evidence, carefully deduce it directly from the stated facts and timestamps."
  - Furthermore, IVARA's context injector explicitly prepended ontology and entity knowledge:
    `[Entity]: SELF (canonical_name: SELF, aliases: I, me, my, myself)`.
- **Verdict:** IVARA's prompt actively facilitated entity resolution and deduction, whereas the RAG prompt was heavily biased toward conservative abstention on first-person transcripts.

### 2.3 Separation of Retrieval vs. Generation Measurement
**Finding:** **Valid Separation, Revealing the Diagnostic Dissociation**
- In `metrics.py`, retrieval recall was evaluated independently by checking whether the retrieved chunks or memory units originated from the target supporting sessions (`evaluate_retrieval()`).
- This separation accurately exposed the core phenomenon:
  - Vector RAG Recall@5: **98.00%**
  - Hybrid RAG Recall@5: **100.00%**
  - Vector RAG Answer Accuracy: **16.17%**
  - Hybrid RAG Answer Accuracy: **16.50%**
- This proves that retrieval was not the failure bottleneck in conventional RAG; the failure occurred entirely during generation due to the speaker identity gap.

### 2.4 Information Leakage and Ground-Truth Integrity
**Finding:** **No Ground-Truth Leakage in Indexes, but Index Asymmetry Exists**
- An audit of `baseline_rag.py` and `src/embedding.py` confirmed that neither the ground-truth answers nor the evaluation questions were indexed or leaked into system prompts.
- FAISS indexes and BM25 tables were created strictly from session turns or extracted memories.
- However, IVARA's pre-seeded `entities.json` contained `canonical_name: SELF` with aliases `["I", "me", "my", "myself"]`, an external structural advantage not afforded to RAG.

### 2.5 Explanation of Fractional Accuracy Values ($16.17\%$, $33.70\%$, $39.02\%$)
**Finding:** **Resolved: Continuous Fact-Coverage Scoring vs Binary Exact Match**
- Why were accuracy figures not multiples of $2\%$ ($1/50 = 0.02$)?
- Line 164 of `evaluation/story_benchmark/metrics.py` defines:
  ```python
  fact_coverage = len(found_facts) / len(key_facts) if key_facts else 1.0
  accuracy = round(fact_coverage, 4)
  exact_match = 1.0 if fact_coverage >= 0.99 else 0.0
  ```
- Questions in the benchmark contain varying numbers of required key facts:
  - Questions with 3 key facts yield coverage scores of $0.0$, $0.3333$, $0.6667$, $1.0$.
  - Questions with 2 key facts yield $0.0$, $0.5$, $1.0$.
  - Questions with 5 key facts yield multiples of $0.20$.
  - Questions with 7 key facts yield multiples of $0.1429$.
- Averaging these continuous fractional values across 50 questions produces non-integer percentage scores:
  - `baseline_vector`: $\sum \text{acc} / 50 = 8.0833 / 50 = 0.161666 \rightarrow \mathbf{16.17\%}$
  - `baseline_hybrid`: $\sum \text{acc} / 50 = 8.2499 / 50 = 0.164998 \rightarrow \mathbf{16.50\%}$
  - `ivara_hybrid_no_graph`: $\sum \text{acc} / 50 = 16.8475 / 50 = 0.336950 \rightarrow \mathbf{33.70\%}$
  - `ivara_full`: $\sum \text{acc} / 50 = 19.5118 / 50 = 0.390236 \rightarrow \mathbf{39.02\%}$
- In contrast, `exact_match` was strictly binary ($0.0$ or $1.0$), resulting in exact multiples of $2\%$:
  - `baseline_vector`: $6 / 50 = \mathbf{12.00\%}$
  - `baseline_hybrid`: $7 / 50 = \mathbf{14.00\%}$
  - `ivara_hybrid_no_graph`: $10 / 50 = \mathbf{20.00\%}$
  - `ivara_full`: $12 / 50 = \mathbf{24.00\%}$
- **Verdict:** The scoring logic is mathematically consistent with macro-averaged multi-fact coverage. The report should have clearly labeled this metric as **"Mean Fact Coverage"** rather than "Accuracy" to avoid confusion with binary accuracy.

### 2.6 Independent Recomputation from Raw Predictions
The raw prediction JSON files were parsed and independently recomputed:

| Configuration | Computed Mean Fact Coverage | Computed Exact Match | Computed Recall@5 | Computed MRR | Computed Citation Validity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline_vector` | 0.161666 (16.17%) | 0.1200 (12.0%) | 0.9800 (98.0%) | 0.9400 | 1.0000 |
| `baseline_hybrid` | 0.164998 (16.50%) | 0.1400 (14.0%) | 1.0000 (100.0%)| 0.9767 | 1.0000 |
| `ivara_hybrid_no_graph` | 0.336950 (33.70%) | 0.2000 (20.0%) | 0.8800 (88.0%) | 0.8400 | 1.0000 |
| `ivara_full` | 0.390236 (39.02%) | 0.2400 (24.0%) | 0.9200 (92.0%) | 0.8950 | 0.9800 |

All values match the previously reported metrics to 4 decimal places. No calculation tampering occurred.

### 2.7 Latency Measurement Equivalence
- Both baseline and IVARA latencies measured query-time execution (search + reasoning/synthesis).
- Ingestion, memory extraction, and index construction were excluded from query-time latency in all arms.
- IVARA's mean latency (3,402 ms) was ~1.4s higher than baseline RAG (2,096 ms). This overhead comes from SQLite graph traversal (~25ms) and slightly longer prompt context and token generation (~1,300ms).

---

## 3. Methodological Flaws & Corrective Actions for Sprint 2

The audit identifies three primary methodological defects in the previous benchmark that must be remedied:

1. **Defect 1: Unfair Conventional RAG Baseline (The Perspective Handicap)**
   - *Issue:* RAG was not provided with the trivial metadata mapping `Speaker: Arjun Mehta`.
   - *Remedy:* Implement **Arm C: Fair Conventional Hybrid RAG with Speaker Metadata** where chunks are labeled with speaker identity (`[Date] Speaker: Arjun Mehta (User): ...`) and the query prompt includes the context that the conversation partner is Arjun Mehta.
2. **Defect 2: Small Sample Size ($N=50$) and Lack of Statistical Significance Testing**
   - *Issue:* 50 questions (5 per category) is underpowered. A difference of 2 questions in a category shifts the category score by 40%. No McNemar tests, bootstrap confidence intervals, or corrections for multiple testing were performed.
   - *Remedy:* Expand the benchmark to **at least 250 questions** across 12 granular categories, with formal paired McNemar tests and 1,000-sample bootstrap confidence intervals.
3. **Defect 3: Conflation of Fact-Coverage with Binary Accuracy**
   - *Issue:* Calling partial fact-coverage "Accuracy" obscures whether the assistant actually answered the question correctly or merely outputted one matching keyword.
   - *Remedy:* Report both **Binary Answer Accuracy** (thresholded at $\ge 0.80$ fact coverage or exact semantics) and **Fact-Level Coverage**, alongside **Exact Match (EM)**.

