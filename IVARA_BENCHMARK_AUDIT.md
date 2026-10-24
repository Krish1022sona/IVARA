# IVARA — Independent Benchmark Audit & Verification Report

**Audit Date:** October 24, 2026  
**Auditor:** Autonomous Testing & Verification Engineer (DeepMind Antigravity)  
**Target Repository:** `IVARA` personal AI memory pipeline  
**Target Benchmark:** Long-Context Memory Benchmark (`evaluation/long_context/`)  
**Status:** Audit Complete. All claims independently verified or refuted against raw artifacts. No production data modified. All 85 unit and regression tests passing.

---

## 1. Executive Summary of Audit Findings

An independent, mathematical audit was performed on the IVARA long-context benchmark implementation, saved JSON run artifacts, metric definitions, and recent code modifications.

### Key Audit Conclusions:
1. **Mathematical Denominator Defect in "Precision@3":**
   The benchmark harness implemented `|retrieved ∩ supporting| / |retrieved|` (Set-Precision) rather than standard Information Retrieval `Precision@k = |retrieved ∩ supporting| / k`.
   - In the **single-episode vector baseline**, where the index has exactly 1 memory (`m001`), retrieving $k=3$ returns 1 item (`retrieved = ['m001']`). Because `|retrieved| = 1`, the metric divided 1 match by 1 returned item, claiming **91.7% Precision@3**.
   - Under standard IR definitions ($/ 3$), true Precision@3 on answerable queries is **33.3%** (and **30.6%** across all 48 queries). A database of 1 memory can never exceed 33.3% Precision@3 at $k=3$.
   - In **multi-episode mode**, IVARA was credited with 43.0% precision vs. baseline's 34.0% because IVARA's 0.35 threshold returned fewer items (often 1 or 2), inflating its set purity. Under true Standard Precision@3 on answerable queries, baseline actually retrieved more relevant memories per query (**37.1%** vs. IVARA's **34.8%**).
2. **Fact Scoring Heuristic vs. Semantic Grading:**
   The reported "Fact Score" was not semantic evaluation or LLM-as-a-judge; it was a crude lexical substring overlap heuristic ($0.7 \times \text{token\_ratio} + 0.3 \times \text{status\_match}$).
   - **False Positive Credit:** A completely false answer asserting COBOL or false deployment claims receives **>70% fact score** if it echoes question and gold keywords.
   - **False Negative Penalty:** An exact, concise, 100% correct answer (e.g., `"Neo4j"` for `q15`) received a **0.07 (7%) fact score** because it lacked verbose filler words present in the gold answer sentence.
3. **Abstention Metric Inversion & Misleading Labeling:**
   The reported `abstention_false_positive_rate: 0.0` was mathematically inverted. The harness computed `(unanswerable_cases - TP) / unanswerable_cases` ($FN / 4 = 0.0$), which is actually the **Failure-to-Abstain / Hallucination Rate on Unanswerables**, NOT the False Positive Abstention rate.
   - In reality, in single-episode mode, IVARA falsely abstained on **28 out of 44 answerable queries (63.6% False Abstention Rate)**, while reporting `abstention_false_positive_rate: 0.0`.
4. **Relationship Persistence Discrepancy (10 vs 14) Resolved:**
   The intermediate status update reported 10 persisted relationships because Episode 10 was still being processed by the local LLM. When Episode 10 finished, it contributed 4 relationships (`2 + 2 + 1 + 2 + 0 + 3 + 0 + 0 + 0 + 4 = 14`), matching `raw_extraction_multi.json` and `memories.json`.
   - However, the report's claim that total extracted relationships was **88** and rejection rate was **84.1%** was **factually incorrect**: the post-fix run extracted **193 raw relationships** and persisted **14**, meaning the actual post-fix rejection rate was **92.7%** (down from 98.5% in the baseline run).
5. **Citation Validity vs. Claim Entailment:**
   Neither metric performed Natural Language Inference (NLI) claim entailment. `citation_validity` only verified syntactic presence in the retrieved list, and `claim_entailment` only verified whether the cited memory ID belonged to the gold supporting episode ID.

---

## 2. Discrepancy Matrix

The table below reconciles all claims made in `IVARA_LONG_CONTEXT_EVALUATION_REPORT.md` and earlier execution logs against raw inspected artifacts:

| # | Topic / Claim in Report | Original Claimed Value | Verified Ground Truth | Source Artifact | Explanation & Root Cause |
| :-: | :--- | :---: | :---: | :--- | :--- |
| **1** | Persisted Relationships (Multi) | 14 in Report / 10 in update | **14 stored** (Episodes 1–9: 10, Ep 10: 4) | `multi_episode/memories.json`, `raw_extraction_multi.json` | An intermediate log was captured while Episode 10 was still executing in Ollama. Once Episode 10 finished, it added 4 relationships. Final stored count is 14. |
| **2** | Post-Fix Total Extracted Relationships | 88 | **193 raw relationships** | `raw_extraction_multi.json` | The report author mistakenly recorded 88 (likely conflating entity recall 0.88). Raw extraction contains 193 triples across 10 episodes. |
| **3** | Post-Fix Relationship Rejection Rate | 84.1% | **92.7%** | `raw_extraction_multi.json`, `memories.json` | $(193 - 14) / 193 = 92.7\%$. 179 relationships were rejected by the ontology validator due to open-domain predicates (`role`, `status`, `employed by`). |
| **4** | Single-Episode Baseline Precision@3 | 91.7% | **33.3%** (answerables) / **30.6%** (overall) | `baseline_results_single_baseline.json` | **Denominator Bug:** Divided by `len(retrieved)` (1 memory) instead of rank parameter $k=3$. Mathematically, 1 memory in an index cannot exceed $1/3$ Precision@3. |
| **5** | Multi-Episode IVARA Precision@3 | 43.0% | **34.8%** (answerables) / **31.9%** (overall) | `results_multi_ivara.json` | Harness computed Set-Precision (47.0% on answerables) across returned items, then averaged zeros on unanswerables to get 43.0%. True Standard P@3 is 34.8%. |
| **6** | Multi-Episode Baseline Precision@3 | 34.0% | **37.1%** (answerables) / **34.0%** (overall) | `results_multi_baseline.json` | Baseline always returned 3 items. Under true Standard P@3, baseline (37.1%) actually outperformed IVARA (34.8%) because baseline captured more relevant memories. |
| **7** | Fact Score Methodology | "Fact Accuracy" / "Fact Score" | **Lexical Substring Heuristic** ($0.7 \times \text{tokens} + 0.3 \times \text{status}$) | `harness.py:L434-L449` | Not semantic evaluation. Rewarded keyword-stuffed falsehoods and severely penalized concise 100% correct answers (e.g. `q15` Neo4j = 0.07; `q37` zero = 0.311). |
| **8** | Abstention False Positive Rate | 0.0% across all runs | **63.6%** (Single IVARA), **15.9%** (Multi IVARA) | `results_single_ivara.json`, `results_multi_ivara.json` | Formula was $FN / \text{unans\_cases} = 0.0$ (Hallucination rate on unanswerables). The true False Positive Abstention Rate on answerable queries was 63.6% for single IVARA. |
| **9** | Pure Unanswerable Accuracy | 100.0% | **100.0% (Verified)** | `results_*.json` (`q41`–`q44`) | Verified on all 4 result files: `q41`, `q42`, `q43`, `q44` returned `actual_status: "insufficient_evidence"`. Zero hallucinations on pure unanswerables. |
| **10** | Claim Entailment | "Claim Entailment: 29% - 100%" | **Episode ID Match Only** (`cited_id ∈ supporting_ids`) | `harness.py:L426-L432` | No sentence-to-passage natural language entailment was performed. Verified only that the cited memory ID was from the correct episode. |

---

## 3. Metric-Definition Audit & Mathematical Formulations

### 3.1 Retrieval Precision@k vs. Set-Precision

#### Harness Implementation (Defective):
```python
ret_prec = len(intersection) / len(retrieved_ids) if retrieved_ids else 0.0
```
- **Flaw:** This calculates the **precision of the returned set**, NOT Precision@k.
- When $k=3$ and the database has 1 memory, FAISS returns 1 item (`len(retrieved_ids) = 1`). If that item is relevant, the formula computes $1 / 1 = 1.0$ (100%), creating the illusion of 91.7% Precision@3.
- When IVARA applies a 0.35 threshold and filters out 2 borderline memories, it returns 1 item. If that item is relevant, it receives $1 / 1 = 1.0$, whereas baseline returning 3 items (1 relevant, 2 irrelevant) receives $1 / 3 = 0.333$. Comparing IVARA's $1/1$ against baseline's $1/3$ falsely claimed IVARA had higher Precision@3.

#### Correct Standard IR Definitions Implemented in `evaluation/long_context/corrected_metrics.py`:
$$\text{Standard Precision@}k = \frac{|\text{Retrieved} \cap \text{Relevant}|}{k}$$
$$\text{Set-Precision} = \frac{|\text{Retrieved} \cap \text{Relevant}|}{|\text{Retrieved}|} \quad (\text{if } |\text{Retrieved}| > 0 \text{ else } 0.0)$$

### 3.2 Retrieval Recall@k & Unanswerable Handling

#### Harness Implementation (Defective):
```python
if not supporting_ids:
    ret_recall = 1.0 if not retrieved_ids else 0.0
```
- In standard Information Retrieval, Recall is only defined for queries that have at least one relevant document ($|\text{Relevant}| > 0$).
- Synthetic scoring of unanswerable queries as $1.0$ or $0.0$ distorted aggregate retrieval recall. In single-episode IVARA, crediting $1.0$ on the 4 unanswerables masked how low the answerable recall actually was (25.9% vs. reported 32.1%).

#### Correct Definition:
$$\text{Recall@}k = \frac{|\text{Retrieved} \cap \text{Relevant}|}{|\text{Relevant}|} \quad \text{evaluated strictly over answerable queries } (|\text{Relevant}| > 0)$$

### 3.3 Abstention Confusion Matrix

The benchmark dataset contains:
- 44 **Answerable Questions** (`supporting_memory_ids` non-empty, `should_abstain = False`)
- 4 **Unanswerable Questions** (`supporting_memory_ids` empty, `should_abstain = True`)

#### Rigorous Confusion Matrix Breakdown:
- **True Positive ($TP$):** Unanswerable question correctly abstained (`status == "insufficient_evidence"`).
- **False Positive ($FP$):** Answerable question falsely abstained (`status == "insufficient_evidence"`).
- **False Negative ($FN$):** Unanswerable question answered / hallucinated (`status != "insufficient_evidence"`).
- **True Negative ($TN$):** Answerable question answered (`status != "insufficient_evidence"`).

$$\text{Abstention Precision} = \frac{TP}{TP + FP}$$
$$\text{Abstention Recall} = \frac{TP}{TP + FN}$$
$$\text{False Abstention Rate (on answerables)} = \frac{FP}{FP + TN} = \frac{FP}{44}$$
$$\text{Failure-to-Abstain / Hallucination Rate (on unanswerables)} = \frac{FN}{TP + FN} = \frac{FN}{4}$$

*Audit Note:* In `harness.py`, `false_pos_rate` was computed as `(4 - TP) / 4 = FN / 4`. This is the **Failure-to-Abstain Rate**, NOT the False Positive Abstention Rate.

### 3.4 Fact Accuracy: Heuristic Flaw vs. Rubric Grading

#### Legacy Lexical Heuristic in `harness.py`:
```python
expected_key_tokens = [w.lower().strip(".,;:()") for w in gold_answer.split() if len(w) > 4]
found = sum(1 for token in expected_key_tokens if token in ans_lower)
token_ratio = found / len(expected_key_tokens)
fact_score = round(token_ratio * 0.7 + (0.3 if status_match else 0.0), 3)
```

#### Defect Verification:
1. **False Claim with Keywords (`q01`):**
   - Gold: *"Project Chronos was implemented in Rust to guarantee zero-cost abstractions."*
   - False Answer: *"Project Chronos was implemented in COBOL to guarantee zero-cost abstractions."*
   - Legacy Score: **0.78** (awarded high credit because 5 of 6 long words matched!).
   - Rubric Evaluation: **0.00** (detected COBOL negative constraint violation).
2. **Concise Correct Answer (`q15`):**
   - Question: *"What graph database was selected for Project Nexus?"*
   - Model Answer: `"Neo4j"` (100% correct).
   - Legacy Score: **0.07** (penalized for not repeating 13 other words from gold answer).
   - Rubric Evaluation: **1.00** (awarded full credit for identifying Neo4j).

---

## 4. Before-and-After Comparison Using Verified Artifacts

All metrics were recalculated directly from the preserved raw artifacts without rerunning expensive LLM calls, using `evaluation/long_context/recalculate_all_metrics.py`.

### 4.1 System Comparison Across All 4 Configurations (Post-Fix Run)

| Configuration | Metric | Original Reported Value | Corrected Standard Value | Audit Status |
| :--- | :--- | :---: | :---: | :--- |
| **Multi-Episode IVARA** | Standard Precision@3 (Answerables) | 43.0% (Set-Prec) | **34.8%** | Corrected (/ 3) |
| | Set-Precision (Answerables) | — | **47.0%** | Verified |
| | Recall@3 (Answerables) | 69.4% (All) | **75.8%** | Corrected (Ans only) |
| | Rubric Fact Accuracy | 55.1% (Heuristic) | **79.2%** | Corrected (Rubric) |
| | False Abstention Rate | 0.0% (Inverted) | **15.9% (7 / 44)** | Corrected |
| | Hallucination on Unanswerables | 0.0% | **0.0% (0 / 4)** | Confirmed |
| **Multi-Episode Baseline** | Standard Precision@3 (Answerables) | 34.0% | **37.1%** | Corrected (/ 3) |
| | Set-Precision (Answerables) | 34.0% | **37.1%** | Verified |
| | Recall@3 (Answerables) | 74.7% (All) | **81.4%** | Corrected (Ans only) |
| | Rubric Fact Accuracy | 55.0% (Heuristic) | **83.3%** | Corrected (Rubric) |
| | False Abstention Rate | 0.0% (Inverted) | **13.6% (6 / 44)** | Corrected |
| | Hallucination on Unanswerables | 0.0% | **0.0% (0 / 4)** | Confirmed |
| **Single-Episode IVARA** | Standard Precision@3 (Answerables) | 41.7% (Set-Prec) | **12.1%** | Corrected (/ 3) |
| | Set-Precision (Answerables) | — | **36.4%** | Verified |
| | Recall@3 (Answerables) | 32.1% (All) | **25.9%** | Corrected (Ans only) |
| | Rubric Fact Accuracy | 42.1% (Heuristic) | **39.6%** | Corrected (Rubric) |
| | False Abstention Rate | 0.0% (Inverted) | **63.6% (28 / 44)** | Severe Dilution |
| | Hallucination on Unanswerables | 0.0% | **0.0% (0 / 4)** | Confirmed |
| **Single-Episode Baseline** | Standard Precision@3 (Answerables) | **91.7%** (Flawed) | **33.3%** | Denominator Fixed |
| | Set-Precision (Answerables) | — | **100.0%** | 1 item returned |
| | Recall@3 (Answerables) | 75.5% (All) | **82.4%** | Corrected (Ans only) |
| | Rubric Fact Accuracy | 57.4% (Heuristic) | **93.8%** | High context dump |
| | False Abstention Rate | 0.0% (Inverted) | **4.5% (2 / 44)** | Minimal |
| | Hallucination on Unanswerables | 0.0% | **0.0% (0 / 4)** | Confirmed |

---

## 5. Code-Change and Regression Review

The working tree changes and new test files were audited against all requirements:

### 5.1 `src/extraction/extractor.py`
- **Schema Constraints:** Added `origin: Literal["explicit", "inferred"]` with clear Field descriptions. Replaced ambiguous `str` typing.
- **Evidence Field:** Added `evidence: str | None = Field(default=None)` to capture supporting text spans explicitly.
- **Prompt Guidance:** Added Rules 11, 12, 13 enforcing evidence text containment, strict origin values, and reasoning token budgeting.
- **Context Window Protection:** Added `num_ctx: 16384` and `num_predict: 4096` to prevent chain-of-thought overflow on reasoning models (`gpt-oss`).

### 5.2 `src/ontology/vocabulary.json`
- **Predicate Aliases Registered:**
  - `worked_with`: Added `"collaborated with"`. Inverse: `worked_with` (symmetric).
  - `worked_on`: Added `"assigned to"`, `"co-founded"`, `"spearheaded"`, `"initiated"`. Inverse: `worked_on_by`.
  - `uses`: Added `"uses_framework"`, `"uses_language"`. Inverse: `used_by`.
- **Integrity Verified:**
  - All inverses are fully mapped and symmetric.
  - No alias collisions or duplicate normalized entries exist.
  - Passes `test_ontology.py::test_vocabulary_structural_integrity`.

### 5.3 `src/answering.py`
- **Containerization & Isolation:** Implemented `<evidence_context>` XML container to isolate memory text from instructions.
- **Deterministic Validation:** `validate_citations()` checks cited IDs strictly against `retrieved_context["direct_evidence"]`. Rejects fabricated IDs.
- **Status Demotion Rules:** Demotes status from `answered` to `partial` if an answer makes factual assertions with zero verifiable citations. Does NOT demote valid unanswerable abstentions.
- **Evidence Grounding:** Preserves and enforces verbatim grounding.

### 5.4 Test Suite Verification
- **Total Tests Passing:** **85 passed in 10.70s** (0 failures, 0 regressions).
  - `tests/test_answering.py`: 9 passed
  - `tests/test_benchmark.py`: 10 passed
  - `tests/test_benchmark_metrics.py`: 9 passed (New unit tests for metric definitions)
  - `tests/test_extraction_regression.py`: 3 passed (New regression tests for schema/vocabulary)
  - `tests/test_ontology.py`: 35 passed
  - `tests/test_pipeline_offline.py`: 6 passed
  - `tests/test_retrieval_grounded.py`: 13 passed

---

## 6. Invalid, Unverifiable, or Misleading Conclusions in Original Report

1. **"Ordinary Vector Baseline achieves 91.7% Retrieval Precision@3 in single-episode mode."**
   - **Verdict: INVALID.**
   - **Reason:** Result of a denominator bug ($1/1$ instead of $1/3$). Standard Precision@3 is 33.3%.
2. **"IVARA achieves higher Precision@3 than Baseline in multi-episode mode (43.0% vs. 34.0%)."**
   - **Verdict: MISLEADING.**
   - **Reason:** Compares IVARA's set purity on filtered subsets against baseline's rank-3 precision. Under true Standard Precision@3 on answerables, Baseline achieved 37.1% while IVARA achieved 34.8%. IVARA does, however, achieve higher Set-Precision (47.0% vs. 37.1%).
3. **"Post-fix extraction extracted 88 relationships with an 84.1% rejection rate."**
   - **Verdict: FACTUALLY INACCURATE.**
   - **Reason:** The post-fix run extracted 193 relationships and persisted 14, yielding a **92.7% rejection rate**.
4. **"Abstention False Positive Rate is 0.0%."**
   - **Verdict: MISLEADING & INVERTED.**
   - **Reason:** The metric computed the failure-to-abstain rate on unanswerables ($0/4$). The actual false positive abstention rate on answerables was **15.9%** in multi-episode mode and **63.6%** in single-episode mode.
5. **"Claim Entailment is 29% - 100% across categories."**
   - **Verdict: MISLEADING LABEL.**
   - **Reason:** Only verified that the cited memory ID matched the gold supporting episode ID. Did not evaluate semantic entailment of the claims themselves.

---

## 7. Corrected Conclusions Supported by Evidence

1. **Evidence Grounding and Adversarial Defense Are Genuine Strengths:**
   - On adversarial prompt injection (`q38`, `q39`), IVARA demonstrated 100% safety, successfully rejecting malicious instructions to declare COBOL or Bob as CEO.
   - On pure unanswerable questions (`q41`–`q44`), IVARA achieved 100% abstention precision and recall, with 0 hallucinations.
2. **Relevance Thresholding Causes a Severe Precision/Recall Trade-Off:**
   - Multi-Episode: IVARA's `min_score = 0.35` achieves higher purity in returned items (Set-Precision 47.0% vs. 37.1%), but at the expense of recall (75.8% vs. 81.4%) and false abstentions on 15.9% of answerable questions.
   - Single-Episode: Monolithic memories suffer severe semantic dilution, causing 63.6% of answerable questions to fall below 0.35 and falsely abstain, collapsing recall to 25.9%.
3. **Relationship Persistence Improved by 367%, but Ontology Coverage Remains the Primary Bottleneck:**
   - Fixing schema typing and alias mapping increased persisted relationships from 3 to 14.
   - However, 92.7% of extracted triples (179 of 193) were rejected because domain predicates (`role`, `status`, `employed by`, `joined`, `planned to use`) are absent from `vocabulary.json`.
4. **Single-Hop Vector Retrieval Fails Multi-Hop Reasoning:**
   - Multi-hop questions requiring evidence from separate episodes (`q13`, `q14`, `q15`) suffer low retrieval recall because vector search cannot link entities across documents without graph traversal or iterative retrieval.

---

## 8. Exact Commands for Reproducing the Audit

All audit steps and recalculations can be reproduced directly using the project virtual environment:

### Run the Complete Test Suite (85 tests):
```powershell
.\.venv\Scripts\python.exe -m pytest
```

### Run the Specific Metric Definition Tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_benchmark_metrics.py -v
```

### Run the Extraction Regression Tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_extraction_regression.py -v
```

### Recalculate Corrected Metrics from Raw Artifacts (No LLM rerun needed):
```powershell
.\.venv\Scripts\python.exe -m evaluation.long_context.recalculate_all_metrics
```

### Inspect the Consolidated Corrected Summary:
```powershell
Get-Content evaluation/long_context/corrected_metrics_summary.json | Select-Object -First 50
```

---

## 9. Prioritized List of What Remains to be Tested

Before proceeding to graph-guided retrieval or architectural changes:

1. **High Priority — Semantic NLI Claim Entailment:**
   Evaluate generated claims against retrieved text spans using an NLI classifier or rubric entailment checker rather than ID-matching alone.
2. **High Priority — Dynamic / Calibrated Relevance Thresholding:**
   Test length-normalized or percentile-based relevance thresholds to prevent monolithic documents from causing catastrophic false abstentions (63.6%).
3. **Medium Priority — Ontology Predicate Expansion:**
   Audit and register common corporate/project relationship predicates (`employed_by`, `role_at`, `managed_by`, `joined_on`) to recover the 179 valid relationships currently dropped by the validator.
4. **Medium Priority — Multi-Hop Benchmark Coverage:**
   Test multi-hop retrieval with 2-step iterative retrieval or entity-seed expansion to establish the baseline for future graph-guided retrieval.
