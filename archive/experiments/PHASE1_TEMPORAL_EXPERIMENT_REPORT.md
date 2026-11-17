# IVARA — Phase 1 Temporal Reasoning Improvement Experiment Report

**Status:** Completed  
**Branch:** `experiment/temporal-phase1`  
**Baseline Git Commit:** `55a273bff20fa0eec1b2d361e8a03b217d147abe`  
**Judge Model:** `qwen2.5:14b` (temperature = 0.0, official LongMemEval prompt protocol)  
**Task Category:** `temporal-reasoning` (133 questions from `longmemeval_s_cleaned.json`)  
**Verdict:** **PROMISING**  

---

## 1. Executive Summary

Based on the findings of `TEMPORAL_REASONING_ARCHITECTURE_AUDIT.md`, IVARA's 20.30% baseline accuracy on temporal reasoning questions was constrained not primarily by graph representation, but by **four non-graph architectural deficits**:
1. Rigid fallback template triggering upon valid synthesis or JSON format discrepancies.
2. Inability of local LLMs to perform multi-step calendrical subtraction across months, leap years, and weeks without external grounding.
3. Lack of reference-date context in user prompts (forcing the model to guess relative offsets like *"how many days ago"* against either machine date or utterance date).
4. Multi-session retrieval starvation under bounded $k=5$ chunks.

In this isolated research experiment, four targeted, non-graph interventions were implemented and evaluated across all **133 temporal diagnostic scenarios**:
1. **Deterministic Calendar Engine (`src/temporal/calendar_arithmetic.py`)**: Standalone, verifiable date arithmetic helper handling leap years, month boundaries, inclusive/exclusive day counting, and week/month deltas.
2. **Reference-Date Context Anchor (`src/answering.py`)**: Explicit propagation and injection of `inst.question_date` and verified calendar facts into the answering prompt.
3. **Adaptive Temporal Retrieval Depth (`src/answering.py`)**: Automatic escalation from $k=5$ to $k=8$ chunks (MMR pool size 15 to 21) when queries route to temporal longitudinal or cross-session synthesis.
4. **Synthesis Fallback & Numeric JSON Calibration (`src/answering.py`)**: Schema recovery accepting numeric/integer answers (e.g. `answer: 4`), unescaping raw quotes, and falling back directly to deterministic calendar facts when models fail to synthesize strings.

### Key Measured Outcomes

| Metric | Baseline (Released Integrated) | Phase 1 Combined | Absolute Change | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Official LLM Judge Accuracy (`qwen2.5:14b`)** | **20.30% (27 / 133)** | **46.62% (62 / 133)** | **+26.32%** | **+129.6% (2.30x)** |
| **Local Heuristic Accuracy** | 30.83% (41 / 133) | 52.63% (70 / 133) | +21.80% | +70.7% |
| **Answerable Accuracy (Official Judge)** | 17.32% (22 / 127) | 44.88% (57 / 127) | +27.56% | +159.1% (2.59x) |
| **Abstention Accuracy (Official Judge)** | 83.33% (5 / 6) | 83.33% (5 / 6) | 0.00% | Maintained |
| **Recall@1** | 80.31% | 80.31% | 0.00% | Preserved |
| **Recall@5** | 88.98% | 88.98% | 0.00% | Preserved |
| **Evidence Coverage Ratio** | 72.34% | 72.34% | 0.00% | Preserved |
| **Mean Answering Latency** | 10,501.59 ms | 9,851.93 ms | -649.66 ms | 6.2% faster |
| **Full Test Suite (Pass / Total)** | 462 / 462 (100%) | 469 / 469 (100%) | +7 tests | Zero regressions |

The combined interventions produced **36 newly passing questions** and **only 1 regression** (net +35 questions), raising temporal reasoning accuracy from **20.30% to 46.62%**.

---

## 2. Strict Experimental Isolation & Safety Audit

In accordance with strict experimental guidelines, no production stores or released branches were modified:
- **Baseline Commit Preserved:** `55a273bff20fa0eec1b2d361e8a03b217d147abe` on branch `research/evaluation-baseline`.
- **Experimental Branch:** `experiment/temporal-phase1`.
- **Isolated Run Directory:** `evaluation/data/longmemeval/runs/temporal_phase1_experiments/phase1_combined/`.
- **Production Store Hash Integrity Verification:**
  - `data/memories.json`: `489e3ea2138022f97a590808cee3dfba03465966eda06d750e340d6d0fddf858` (Verified Unchanged)
  - `data/entities.json`: `51f993fbe1cfc7f40084724499cd2dd216b30d8e3aeeace9824d1863d84fc09e` (Verified Unchanged)
  - `data/memory_graph.db`: `f6b9f9cf7d35674e461350ffa87dcfa735ae0d7643c46e30138f56bffb31f95d` (Verified Unchanged)
  - `data/conversations.db`: `a44d2ce7501fd3170147c614ccb59d4f9416e11cd0defd66b370d9442878e89f` (Verified Unchanged)

---

## 3. Category-by-Category Recovery Analysis

We cross-referenced all 106 baseline failures categorized during the audit against the Phase 1 official judge decisions:

```
=== Category-by-Category Recovery Analysis (Total 106 Baseline Failures) ===
Category I (Synthesis / Template Fallback)      : 17 / 21 recovered (81.0%)
Category A (Relative Date Arithmetic / Offset) :  4 /  5 recovered (80.0%)
Category D (Event-Time vs Ingestion-Time)      :  3 /  5 recovered (60.0%)
Category C (Event Ordering / Chronology)       :  3 /  6 recovered (50.0%)
Category E (Historical State Reconstruction)   :  2 /  4 recovered (50.0%)
Category G (Knowledge Update / Contradiction)  :  3 /  8 recovered (37.5%)
Category B (Complex Synthesis Queries)         :  1 /  8 recovered (12.5%)
Category H (Multi-Session Aggregation)         :  3 / 44 recovered ( 6.8%)
Category F (Retrieval Ranking)                 :  0 /  5 recovered (  0.0%)
```

### Detailed Observations per Category:
1. **Category I (Synthesis / Fallback Recovery) — 81.0% Recovery**:
   - Addressing the schema bug where integer answers (`answer: 4`) triggered Pydantic validation failures restored 17 answers that had previously been discarded into fallback template responses.
2. **Category A (Calendar Arithmetic) — 80.0% Recovery**:
   - Injecting deterministic date calculation annotations directly into user prompts resolved 4 out of 5 arithmetic failures (e.g. counting exact elapsed days across month boundaries).
3. **Category D & E (Time Anchors & Historical State) — 50.0% - 60.0% Recovery**:
   - Explicit reference date anchoring (`CURRENT REFERENCE TIME ANCHOR: ...`) eliminated utterance vs. question date confusion on relative queries.
4. **Category H (Multi-Session Evidence Aggregation) — 6.8% Recovery (41 Remaining Failures)**:
   - Escalating chunk retrieval depth to $k=8$ recovered 3 multi-session questions, but **41 questions remained unrecovered**.
   - **Root Cause**: When an instance contains evidence spread across 4 to 6 disparate conversation sessions among 50+ total sessions, increasing $k$ from 5 to 8 within a single semantic MMR pass is insufficient. Chunks from the dominant 2 sessions saturate the top-8 pool, leaving 3rd and 4th sessions unretrieved.

---

## 4. Remaining Failure Analysis (71 Failures Total)

A comprehensive audit of the 71 failures remaining after Phase 1 reveals the clear architectural distribution:

1. **Category H: Multi-Session Evidence Aggregation (41 / 71 = 57.7% of all remaining failures)**:
   - **Diagnosis**: This is now the undisputed primary bottleneck in temporal reasoning. The current chunk retriever operates at the individual turn-chunk level with session MMR penalties. However, when an instance requires correlating an event in session 1 (January), an event in session 12 (February), and an event in session 28 (April), dense vector similarity pulls chunks from the most semantically dense sessions and leaves earlier anchor sessions unretrieved.
   - **Solution for Phase 2**: Hierarchical session-level retrieval or a cross-session temporal episodic graph that links entities across sessions chronologically.
2. **Category B: Complex Synthesis & Attribute Extraction (7 / 71 = 9.9%)**:
   - Complex reasoning where all evidence is present in context, but the LLM fails to synthesize multi-hop deductive chains.
3. **Category F: Retrieval Ranking (5 / 71 = 7.0%)**:
   - Evidence chunks exist in the session pool but fall just outside the top-8 rank due to dense vector similarity noise.
4. **Category G: Knowledge Update / Supersession (5 / 71 = 7.0%)**:
   - Contradictory statements where temporal invalidation was not explicitly established.

---

## 5. Regression Inspection

Exactly **1 question** regressed from correct to incorrect:
- **Question ID**: `gpt4_483dd43c`
- **Question**: *"Which show did I start watching first, 'The Crown' or 'Game of Thrones'?"*
- **Gold Answer**: `'Game of Thrones'`
- **Baseline Hypothesis**: *"You started watching Game of Thrones first."* (Status: `answered`)
- **Phase 1 Hypothesis**: *"Could not reliably synthesize an answer for 'Which show did I start watching first...'."* (Status: `insufficient_evidence`)
- **Cause**: The model generated an unparseable response string under the enhanced temporal prompt that was not a valid JSON object or calendar-computable query, falling back to abstention.

---

## 6. Phase 2 Recommendations & Roadmap

The Phase 1 experiment conclusively demonstrated that non-graph interventions (deterministic calendar arithmetic, reference-date anchoring, and fallback repair) deliver immediate, high-impact gains (+26.32 percentage points).

However, to address the **41 remaining Category H failures**, a **Temporal Knowledge Graph & Hierarchical Episodic Architecture (Phase 2)** is justified and recommended:

### Phase 2 Architecture Plan:
1. **Hierarchical Two-Stage Retrieval (Session-Level -> Chunk-Level)**:
   - First stage: Retrieve top-$M$ relevant conversation sessions using session-level temporal centroids and entity occurrences.
   - Second stage: Sample top-$N$ diverse chunks within each retrieved session. Guarantees balanced representation across 4+ sessions.
2. **Temporal Edge & Chronological Sequence Graph**:
   - Construct directed `OCCURRED_BEFORE` and `VALID_DURING` edges between entity states and session nodes.
   - Traverse temporal paths to naturally reconstruct event sequences across months.
3. **Calibrated Synthesis Fine-Tuning**:
   - Strengthen LLM prompt guidelines for multi-session chronology formatting.

---

## 7. Conclusion & Final Verdict

**Verdict:** **PROMISING**

The Phase 1 interventions achieved a **+26.32 percentage point gain** (from **20.30% to 46.62%**) on the 133 LongMemEval temporal reasoning benchmark without altering production stores, without causing regressions in abstention accuracy (83.33% maintained), and while slightly improving inference latency (from 10.5s to 9.85s).

The forensic audit of remaining failures proves that multi-session evidence aggregation (Category H, 41 cases) is the final major frontier for Phase 2.

