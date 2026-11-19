# IVARA — Step 9: Human Validation Plan and Evaluator Audit Report

**Date:** 2026-10-28  
**Branch:** `research/evaluation-baseline`  
**Test Suite Status:** **204 passing tests** (199 repository/evaluator tests + 5 new Step 9 integrity/sampling validation tests in `tests/test_step9_validation.py`)  
**Scope:** Verification of Step 7 and Step 8 data integrity, construction of a reproducible 26-question human review sample, failure taxonomy, audit of local Gemma judge behaviors, and design of the smallest targeted follow-up experiment.  
**Deliverables Directory:** `evaluation/step9/`  

---

## Executive Summary

Step 9 establishes the empirical and procedural bridge between automated local evaluation and human validation. Prior to committing to architectural changes or full benchmark execution, this audit verified:

1. **Complete Metric Integrity (Task 1):** Verified 100% integrity across all Step 7 raw predictions, Step 8 Gemma judgments, recomputed metrics, and paired comparisons. **Zero QID mismatches, zero duplicate entries, zero missing judgments, and zero denominator inconsistencies exist.** The 8 unanswerable questions in the pilot align exactly between the dataset and pipeline records.
2. **Reproducible Stratified Human-Review Sample (Task 2):** Created a fixed, reproducible sample of **26 distinct questions** (`evaluation/step9/human_review_sample.csv`) using seed 42. The sample captures all 13 configuration score disagreements, 8 unanswerable questions, and representation across all 6 LongMemEval tasks. **All human verdict columns are left strictly blank.**
3. **Failure Taxonomy & Gemma Judge Audit (Task 3):** Analyzed each question in the sample across seven failure modes, distinguishing observed evidence from hypotheses. Audited instances where Gemma's judgment provides clear improvements over rule heuristics (e.g., semantic number equivalence in `dcfa8644` and formal abstention phrasing in `2698e78f_abs`) as well as cases where Gemma's judgment is questionable (e.g., penalizing cautious abstention in `099778bb` or minor contextual expansion in `6aeb4375`).
4. **Targeted Follow-Up Experiment Design (Task 4):** Designed a three-arm, 100-question targeted diagnostic experiment to isolate: (1) ontology retrieval gain, (2) graph edge pruning to eliminate repetition loops, and (3) temporal resolution applied directly to seed chunks without raw graph expansion.

---

## 1. Evaluation & Metric Integrity Audit (Task 1)

A systematic verification of all underlying data files was conducted via `evaluation/step9/audit_integrity.py`. The results are formally recorded in [`evaluation/step9/metric_integrity_audit.json`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/step9/metric_integrity_audit.json).

### 1.1 Integrity Checklist

| Integrity Verification Check | Expected Standard | Observed Result | Status |
|---|---|---|---|
| **Question ID Alignment** | Identical 60 QIDs in identical order across Configs A–D | 60/60 exact match in identical order | **PASS** |
| **Duplicate Entries** | 0 duplicate question IDs across all runs | 0 duplicates found | **PASS** |
| **Unanswerable Denominators** | Exactly 8 unanswerable questions (`_abs`) in pilot | 8 unanswerable, 52 answerable | **PASS** |
| **Dataset Label Alignment** | `is_abstention` flag in prediction matches benchmark label | 100% agreement across all 240 records | **PASS** |
| **Judge Judgment Mapping** | Every judgment maps to correct question ID and candidate text | 240/240 mapped with 0 orphans | **PASS** |
| **Judge Error Count** | 0 malformed JSON outputs or model execution crashes | 0 judge failures (`judge_failures: 0`) | **PASS** |
| **Reported Metric Parity** | Reported accuracy & 95% Wilson CIs match raw calculations | 100% exact parity across all metrics | **PASS** |
| **Paired Win/Loss Parity** | Reported paired deltas match raw record differences | 100% exact parity across all 4 pairs | **PASS** |

### 1.2 Recomputed Authoritative Metrics (Local Gemma Judge)

All reported figures from Step 8 recompute exactly from raw prediction files:

- **Config-A (Baseline):** 63.33% Overall [50.7, 74.4] | 61.54% Answerable (32/52) | 75.00% Abstention (6/8)
- **Config-B (Ontology):** **71.67% Overall** [59.2, 81.5] | **67.31% Answerable** (35/52) | **100.00% Abstention** (8/8)
- **Config-C (Graph):** 68.33% Overall [55.8, 78.7] | 65.38% Answerable (34/52) | 87.50% Abstention (7/8)
- **Config-D (Full System):** **71.67% Overall** [59.2, 81.5] | **67.31% Answerable** (35/52) | **100.00% Abstention** (8/8)

---

## 2. Reproducible Human-Review Sample (Task 2)

A stratified sample of **26 distinct questions** was constructed using `evaluation/step9/generate_human_review_sample.py` with deterministic random seed `42`.

### 2.1 Sampling Protocol & Stratification
- **Sampling Unit:** Distinct question instances (each question includes predictions from Config-A, Config-B, Config-C, and Config-D side-by-side, yielding 104 individual candidate predictions).
- **Inclusion Criteria:**
  1. All 13 questions where Gemma's correctness judgment differed across configurations.
  2. 8 unanswerable/abstention instances (`_abs`) to evaluate false abstention vs proper refusal.
  3. Balanced representation across all 6 LongMemEval tasks:
     - `multi-session`: 5 questions
     - `single-session-preference`: 5 questions
     - `knowledge-update`: 4 questions
     - `single-session-assistant`: 4 questions
     - `single-session-user`: 4 questions
     - `temporal-reasoning`: 4 questions
  4. At least 5 unanimous correct and 3 unanimous incorrect questions to establish human baseline agreement.

### 2.2 Review Artifact Structure
The review dataset is saved in [`evaluation/step9/human_review_sample.csv`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/step9/human_review_sample.csv). Each row contains:
- Question metadata: `sample_index`, `question_id`, `task_type`, `is_unanswerable`, `question`, `gold_answer`.
- Candidate predictions: `prediction_Config_A`, `prediction_Config_B`, `prediction_Config_C`, `prediction_Config_D`.
- Automated judge context: `gemma_score_*` and `gemma_reasoning_*` for each configuration.
- Provenance: `retrieved_chunks_D` and `temporal_claims_count_D`.
- **Human evaluation columns (strictly blank):** `human_verdict_A`, `human_verdict_B`, `human_verdict_C`, `human_verdict_D`, `human_failure_category`, and `human_notes`.

---

## 3. Failure Analysis & Local Judge Audit (Task 3)

The complete failure taxonomy and case-by-case findings are recorded in [`evaluation/step9/failure_analysis.json`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/step9/failure_analysis.json).

### 3.1 Failure Categories Identified

| Failure Category | Observed Evidence | Example QID | Key Causal Factor |
|---|---|---|---|
| **Retrieval Miss (Rescued by Ontology)** | Empty prediction in A; accurate prediction in B/C/D | `184da446` | Keyword matching failed; ontology expansion brought update chunk into rank 1. |
| **Graph Distraction & Repetition Loop** | Config-C generated 32 repeated sentences; Config-D abstained cleanly | `gpt4_c27434e8_abs` | 1-hop neighbor turns introduced cyclical prompt patterns in the 20.9B model. |
| **Evaluator Ambiguity (Heuristic vs LLM)** | "Fourteen days" failed heuristic regex but passed Gemma | `dcfa8644` | Deterministic digit matching rejected valid written numbers. |
| **Formal Prose Contraction Rejection** | "do not have sufficient" rejected by heuristic string match | `2698e78f_abs` | Hardcoded substring rule only searched for "don't have sufficient". |
| **Cautious Grounding Abstention** | Model abstained when dialogue contained conflicting claims | `099778bb` | Grounding threshold detected corporate vs industry data ambiguity. |
| **Multi-Session Cost Range Calculation** | Config-D provided full Haneda + Narita range | `09ba9854` | Complex multi-hop synthesis across multiple session timestamps. |

### 3.2 Audit of Gemma Local Judge Behavior

The evaluation revealed that Gemma 25.2B is a vastly superior judge compared to the deterministic heuristic regex, but exhibits distinct failure boundaries that require human awareness:

#### Clear Gemma Successes:
- **Semantic Number Equivalence (`dcfa8644`):** Gemma correctly noted: *"The candidate prediction correctly identifies the number of days as fourteen, which is semantically equivalent to the gold answer of 14 days."*
- **Formal Abstention Acceptance (`2698e78f_abs`):** Gemma correctly recognized that *"I do not have sufficient recorded memories..."* is a valid abstention, rejecting the brittle heuristic's failure.
- **Repetition Loop Penalization (`gpt4_c27434e8_abs`):** Gemma correctly failed Config-C for producing an ungrounded infinite generation loop.

#### Questionable Gemma Judgments:
- **Penalizing Cautious Abstention (`099778bb`):** In a multi-session query regarding women in leadership positions, the user conversation contained conflicting mentions (a 20% internal figure vs external benchmarks). Config-D refused to answer due to ambiguity. Gemma marked Config-D as `INCORRECT` (*"improperly abstained from answering"*). Human review is needed to decide whether conservative safety abstention should be penalized.
- **Over-Specification Penalty (`6aeb4375`):** Config-D provided the updated fact along with relevant historical context. Gemma marked it incorrect for adding unrequested detail, despite the core factual assertion being true.

---

## 4. Smallest Targeted Follow-Up Experiment (Task 4)

To resolve the remaining empirical questions without incurring unnecessary compute or running the full 500-question benchmark, we recommend a **targeted 100-question diagnostic ablation** structured into three isolated micro-experiments:

### 4.1 Experiment Architecture

```
                  ┌────────────────────────────────────────┐
                  │ 100-Question Stratified Diagnostic Set │
                  └───────────────────┬────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
      [Arm 1: E1]               [Arm 2: E2]               [Arm 3: E3]
   Ontology Verification     Graph Edge Filtering     Direct Temporal Seed
   Config-A vs Config-B      Config-B vs Pruned-C     Config-B + Temporal
   (Is +8.33 pp robust?)     (Does cutoff kill loops?) (Does graph add value?)
```

1. **Arm E1 — Ontology Retrieval Verification:**
   - **Hypothesis:** Ontology guidance adds a robust +5–8 pp retrieval accuracy boost on entity-heavy questions without adding noise.
   - **Protocol:** Compare Config-A (Baseline) vs Config-B (Baseline + Ontology) on 100 held-out LongMemEval instances.
2. **Arm E2 — Graph Edge Pruning & Relevance Cutoff:**
   - **Hypothesis:** The repetition loops and -3.33 pp drop in Config-C are caused by low-relevance 1-hop edges. Restricting expansion to edges with weight $\ge 0.50$ and pruning turns with $<0.20$ cross-encoder score will eliminate distractors while preserving recall.
   - **Protocol:** Compare unpruned Config-C against pruned Config-C+ on instances with graph nodes $>2,000$.
3. **Arm E3 — Direct Temporal Resolution on Seed Chunks:**
   - **Hypothesis:** Temporal claim extraction and supersession provide their full benefit when executed directly on seed retrieved chunks, without requiring 1-hop graph expansion.
   - **Protocol:** Test Config-B + Temporal Resolver (omitting graph retrieval entirely) to determine if graph traversal is redundant when temporal claims are extracted directly from top-$k$ chunks.

### 4.2 Non-Negotiable Gate
**Do not implement code changes or launch Arm E1–E3 until a human evaluator completes the review of [`evaluation/step9/human_review_sample.csv`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/step9/human_review_sample.csv).**

---

## 5. Deliverable Inventory & File Locations

All Step 9 artifacts are created and verified in `evaluation/step9/`:

1. **Step 9 Validation Plan & Report:** [`evaluation/step9/STEP9_HUMAN_VALIDATION_PLAN.md`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/step9/STEP9_HUMAN_VALIDATION_PLAN.md)
2. **Human Review Sample Sheet:** [`evaluation/step9/human_review_sample.csv`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/step9/human_review_sample.csv) (26 questions, blank human verdict columns)
3. **Metric Integrity Audit Record:** [`evaluation/step9/metric_integrity_audit.json`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/step9/metric_integrity_audit.json) (100% verified, 0 discrepancies)
4. **Detailed Failure Analysis:** [`evaluation/step9/failure_analysis.json`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/step9/failure_analysis.json) (taxonomized failure modes, hypotheses, and judge assessments)
5. **Validation Unit Test Suite:** [`tests/test_step9_validation.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/tests/test_step9_validation.py) (5 tests, 100% passing)

---

## 6. Safety & Integrity Confirmation

- **Git Branch:** Currently on `research/evaluation-baseline`. `main` was never modified.
- **Production Stores:** [`data/memories.json`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/data/memories.json) and [`data/entities.json`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/data/entities.json) are **completely untouched**.
- **Historical Deliverables:** Step 7 and Step 8 prediction files, summaries, manifests, and reports remain **completely preserved**.
- **Execution Policy:** No architecture features were added; no paid APIs were called; no models were downloaded.
- **Next Step:** Awaiting human review of `evaluation/step9/human_review_sample.csv`.
