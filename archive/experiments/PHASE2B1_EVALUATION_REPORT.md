# IVARA — Phase 2B.1: Temporal Event Graph Evaluation Report

**Date:** 2026-11-17  
**Author:** Antigravity (DeepMind Team)  
**Branch:** `experiment/temporal-kg-phase2b`  
**Baseline Commit:** `55a273bff20fa0eec1b2d361e8a03b217d147abe`  
**Judge Model:** `qwen2.5:14b` (Temperature = 0.0, Official LongMemEval Judge Prompt)  
**Evaluated Cohort:** All 133 LongMemEval-S Temporal Reasoning Questions  
**Status:** Evaluation Completed & Certified  

---

## 1. Executive Summary

Phase 2B.1 evaluated the isolated **Temporal Event Graph (TEG)** prototype across the complete 133-question LongMemEval-S temporal benchmark.

Under official LLM judging with `qwen2.5:14b` at temperature 0.0:
- **Overall Temporal Accuracy reached 51.88% (69 / 133)**, breaking the 50% milestone on temporal reasoning for the first time in IVARA's evaluation history.
- Compared to the **Phase 1 Flat Baseline (46.62%, 62 / 133)**, Phase 2B.1 achieved a net gain of **+7 questions (+5.26% absolute)**.
- Compared to **Phase 2A.1 Adaptive Hierarchical Retrieval (44.36%, 59 / 133)**, Phase 2B.1 achieved a net gain of **+10 questions (+7.52% absolute)**.
- **Non-Category H accuracy reached an all-time high of 71.74% (66 / 92)**, reversing the severe regressions of Phase 2A/2A.1 and recovering **7 of the 11 regressions**.
- **Abstention Accuracy remained robust at 83.33% (5 / 6)**.
- **Mean Latency was reduced to 8.05 seconds**, down from 9.92s in baseline and 11.2s in Phase 2A.1.

---

## 2. 3-Way Comparative Benchmark Results

| Metric | Phase 1 Flat Baseline | Phase 2A.1 Adaptive | Phase 2B.1 Event Graph | Delta (2B.1 vs P1) | Delta (2B.1 vs 2A.1) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Official Overall Accuracy** | **46.62%** (62/133) | **44.36%** (59/133) | **51.88%** (69/133) | **+5.26% (+7 Qs)** | **+7.52% (+10 Qs)** |
| **Category H (Multi-Session)** | 0.00% (0/41) | 14.63% (6/41) | 7.32% (3/41) | +7.32% (+3 Qs) | -7.31% (-3 Qs) |
| **Non-Category H (Single-Session)** | 67.39% (62/92) | 57.61% (53/92) | **71.74%** (66/92) | **+4.35% (+4 Qs)** | **+14.13% (+13 Qs)** |
| **Abstention Accuracy** | 83.33% (5/6) | 83.33% (5/6) | 83.33% (5/6) | 0.00% (0) | 0.00% (0) |
| **Answerable Accuracy** | 44.88% (57/127) | 42.52% (54/127) | **50.39%** (64/127) | **+5.51% (+7 Qs)** | **+7.87% (+10 Qs)** |
| **Recall@5** | 94.26% | 93.98% | 91.73% | -2.53% | -2.25% |
| **Evidence Coverage Ratio** | 82.49% | 85.12% | **86.93%** | **+4.44%** | **+1.81%** |
| **Mean Context Tokens** | 2,840 | 3,110 | **2,903** | +63 tokens | -207 tokens |
| **Mean End-to-End Latency** | 9.92s | 11.20s | **8.05s** | **-1.87s** | **-3.15s** |

---

## 3. Cohort Transition Accounting

### 3.1 Transition Matrix (Phase 1 Flat $\to$ Phase 2B.1 Event Graph)
```
                          PHASE 1 FLAT BASELINE
                           (62 / 133 = 46.62%)
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
  Phase 1 Correct                                     Phase 1 Incorrect
   (62 questions)                                      (71 questions)
         │                                                   │
   ┌─────┴─────┐                                       ┌─────┴─────┐
   ▼           ▼                                       ▼           ▼
Remained    Regressed                               Recovered   Remained
Correct     (T -> F)                                (F -> T)    Incorrect
(T -> T)    5 questions                             12 questions(F -> F)
57 Qs       (All Non-Cat H)                         (3 Cat H,   59 Qs
                                                     9 Non-Cat H)
```
- **Net Question Movement:** $+12 - 5 = \mathbf{+7\text{ questions}}$.

---

## 4. Resolution of the 11 Phase 2A.1 Regressions

In Phase 2A.1, 11 non-Category H questions regressed due to prompt snippet re-ordering (6 cases) and dense retrieval drift (2 cases). 

In Phase 2B.1, **7 of these 11 regressions are fully recovered**:

| Question ID | Phase 1 | Phase 2A.1 | Phase 2B.1 | Status in 2B.1 | Root Cause Resolution in 2B.1 |
|---|:---:|:---:|:---:|:---:|---|
| `gpt4_fa19884c` | True | False | **True** | **RECOVERED** | Relevance-first layout + regex parsing eliminated formatting fallback. |
| `0db4c65d` | True | False | False | Still Wrong | Distractor dates in context; requires tighter date-filtering. |
| `gpt4_1916e0ea` | True | False | **True** | **RECOVERED** | Event graph linked entity `ent_farmfresh` to cancellation date turn. |
| `gpt4_1e4a8aeb` | True | False | **True** | **RECOVERED** | Restored concise context budget; model emitted exact answer. |
| `6e984301` | True | False | **True** | **RECOVERED** | Fixed prompt layout inversion; LLM parsed correct event date. |
| `gpt4_8e165409` | True | False | **True** | **RECOVERED** | Synthesis protection recovered answer from valid text block. |
| `gpt4_68e94287` | True | False | **True** | **RECOVERED** | Event node linked `#PlankChallenge` across sessions directly. |
| `gpt4_4929293a` | True | False | False | Still Wrong | Model hedges on relative dates; conservative abstention. |
| `gpt4_78cf46a3` | True | False | False | Still Wrong | Judge marked No due to gold answer format mismatch. |
| `gpt4_f420262c` | False | False | False | Still Wrong | 6-flight ordering query requiring broader multi-hop traversal. |
| `gpt4_93159ced_abs`| True | False | **True** | **RECOVERED** | Correctly abstained on unanswerable Google job duration. |

**Recovery Rate on Regressions:** **63.6% (7 / 11)**.

---

## 5. Event Graph Extraction & Relationship Audit (20 Samples)

From a detailed audit across 20 representative multi-session instances:
- **Total Events Extracted:** 1,092 events ($\mu = 54.6$ events / instance).
- **Total Temporal Edges Built:** 15,774 edges ($\mu = 788.7$ edges / instance).
- **Temporal Precision Distribution:**
  - `EXACT_DAY`: 177 events (16.2%) — exact calendar day anchors.
  - `RELATIVE`: 38 events (3.5%) — anchored relative to conversation date.
  - `MONTH`: 287 events (26.3%) — bounded calendar month intervals.
  - `YEAR`: 97 events (8.9%) — bounded calendar year intervals.
  - `UNKNOWN`: 493 events (45.1%) — **uncertainty preserved without date fabrication**.
- **Provenance Integrity:** **100.0% (1,092 / 1,092)** retained valid source session IDs and verbatim text spans.
- **Decoupled Time Invariant:** 599 events demonstrated distinct `valid_from` vs `recorded_at`, confirming that event time is strictly decoupled from ingestion time.
- **Non-Ingestion Rule Compliance:** 0 temporal edges were created between undated events, confirming that conversation logging order is never used as a proxy for real-world event chronology.

---

## 6. Technical Verdict & Failure Attribution

### 6.1 Where the Temporal Event Graph Improved Performance
1. **Entity-Anchored Dense Retrieval Recovery:**
   - In questions like `gpt4_1916e0ea` (FarmFresh cancellation) and `gpt4_68e94287` (#PlankChallenge), dense retrieval suffered semantic drift against distractor conversations. The event graph explicitly linked the entity mention to the event node and its source turn, restoring 100% evidence coverage.
2. **Elimination of Synthesis Fallbacks:**
   - Hardened answer extraction and relevance-first snippet layout recovered 5 formatting fallbacks from Phase 2A.1, driving Non-Category H accuracy to an all-time peak of 71.74%.
3. **Historical State Grounding & Abstention:**
   - Decoupled `valid_from` vs `recorded_at` enabled robust abstention on queries where the user had not stated historical duration (`gpt4_93159ced_abs`).

### 6.2 Remaining Limitations & Areas Inconclusive
1. **High-Cardinality Event Ordering (Category H):**
   - While the graph recovered 3 Category H questions (`b46e15ee`, `gpt4_1a1dc16d`, `gpt4_2f56ae70`), complex 3- to 6-event ordering queries (e.g. `gpt4_7abb270c`: order of 6 museums; `gpt4_f420262c`: order of 6 flights) require complete topological sorting over disjoint sessions that the current 2-hop bounded traversal does not fully resolve.
   - Forcing global hierarchical retrieval (as in Phase 2A) surfaced more sessions (6/41 correct) but damaged single-session performance. The Event Graph achieved a superior overall trade-off (51.88% vs 44.36%) by safeguarding single-session precision, but high-cardinality multi-session ordering remains an open challenge.

---

## 7. Production Database Integrity Check

All production databases remain 100% untouched with bit-exact SHA-256 matches:

| Database File | Expected SHA-256 | Actual SHA-256 | Verification |
|---|---|---|:---:|
| `data/memories.json` | `489e3ea21380...` | `489e3ea21380...` | **PASSED (Bit-Exact)** |
| `data/entities.json` | `51f993fb...` | `51f993fb...` | **PASSED (Bit-Exact)** |
| `data/memory_graph.db`| `f6b9f9cf...` | `f6b9f9cf...` | **PASSED (Bit-Exact)** |
| `data/conversations.db`| `af88c18f...` | `af88c18f...` | **PASSED (Bit-Exact)** |

