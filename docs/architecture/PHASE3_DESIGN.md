# IVARA — Phase 3: Query-Aware Multi-Session Retrieval Design

**Date:** 2026-11-17  
**Author:** Antigravity (DeepMind Team)  
**Branch:** `experiment/query-aware-retrieval-phase3`  
**Baseline Commit:** `55a273bff20fa0eec1b2d361e8a03b217d147abe`  
**Status:** Approved Architecture & Design Specification  

---

## 1. Executive Summary & Problem Formulation

In Phase 2B.2, a strictly controlled 4-arm ablation demonstrated that the Temporal Event Graph prototype contributed **0.00% net accuracy**, with 100% of the benchmark improvement over Phase 1 (+5.26% absolute, 62 → 69/133) driven by the robust synthesis and JSON parsing pipeline in [src/answering.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/answering.py).

However, **Category H (Multi-Session Temporal Reasoning)** remained largely unsolved:
- Phase 1 Flat Baseline: **0 / 41 (0.00%)**
- Phase 2A.1 Adaptive: **6 / 41 (14.63%)**, but suffered from **11 Non-Category H regressions** when forced diversity degraded single-session depth.
- Phase 2B.1 / Arm B: **3 / 41 (7.32%)**, achieving a peak **71.74% (66/92)** on Non-Category H through dense flat MMR ($k=5$), but suffering from **100% retrieval starvation** on multi-session queries because the single highest-scoring session consumed 3–4 chunk slots.

### The Phase 3 Core Objective
Design and implement a **Query-Aware Multi-Session Retrieval Architecture** that:
1. Recovers multi-session temporal reasoning on Category H by ensuring that all required sessions are retrieved and allocated context slots.
2. Completely protects the 71.74% precision on Non-Category H queries by routing single-session and uncertain queries to the proven flat dense MMR pipeline.
3. Dynamically estimates the number of required sessions ($S$) and avoids the fragile "one chunk per session" constraint for two-event comparison queries.
4. Preserves 100% verifiable provenance, chronological snippet ordering, and calibrated abstention behavior.

---

## 2. Architecture Overview

```
                                 User Query
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │   QueryAwareRetrievalRouter   │
                     └───────────────┬───────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           ▼                         ▼                         ▼
   [ SINGLE_SESSION ]         [ MULTI_SESSION ]          [ UNCERTAIN ]
   Confidence >= 0.90         Confidence >= 0.85         Confidence < 0.85
           │                         │                         │
           │                         ▼                         │
           │              Query Sub-clause Decomposer          │
           │              (Extract Subqueries: S1, S2, ...)    │
           │                         │                         │
           │                         ▼                         │
           │              Session-Level Representation         │
           │              (Vector + BM25 Candidate Ranking)    │
           │                         │                         │
           │                         ▼                         │
           │              Dynamic Budget Allocation            │
           │              S = 2: 2 chunks/session + 1 rem      │
           │              S = 3: 1-2 chunks/session            │
           │              S >= 4: 1 chunk/session              │
           │                         │                         │
           │                         ▼                         │
           │              Within-Session Chunk Scoring         │
           │              (Intra-session Dense + BM25)         │
           │                         │                         │
           │                         ▼                         │
           │              Chronological Ordering &             │
           │              Provenance Assembly                  │
           │                         │                         │
           └────────────────►        │        ◄────────────────┘
                                     ▼
                           Grounded Context Window
                           (Bounded <= 8192 Tokens)
                                     │
                                     ▼
                          Phase 2B.2 Arm B Synthesis
                          (Robust Regex + Calendar)
                                     │
                                     ▼
                             Calibrated Answer
```

---

## 3. Router Outcomes & Decision Engine

The router executes deterministic syntactic and semantic pattern matching with three distinct outcomes:

### Outcome 1: `SINGLE_SESSION`
- **Definition:** The query concerns an event or activity located in a single conversational session, anchored by a relative time expression relative to the reference date (e.g. "two weeks ago", "last Friday", "a month ago"), or asks a standard localized factual question.
- **Representative Query Patterns:**
  - `\bhow many (days|weeks|months|years) ago did i\b`
  - `\bhow long ago did i\b`
  - `\bwhich book did i finish a week ago\b`
  - `\bwhat charity event did i participate in a month ago\b`
  - `\bi mentioned [a-z ]+ (two weeks ago|a month ago|last friday)\b`
- **Retrieval Strategy:** Flat Dense MMR (`ChunkRetriever`, $k=5$, `effective_chunks_per_session=3`, $\lambda=0.25$).
- **Rationale:** Preserves full conversational context (2–3 contiguous turns) so the LLM can resolve references and nuances without distraction.

### Outcome 2: `MULTI_SESSION`
- **Definition:** The query explicitly requires evidence from multiple distinct conversational episodes (comparing relative timing of two events, ordering 3–6 historical events, or computing intervals across distinct sessions).
- **Representative Query Patterns:**
  - **Two-Event Relative Comparison:** `which did i do first, X or Y`, `did X happen before Y`, `which happened first, X or Y`.
  - **Two-Event Interval:** `how many (days|weeks|months) passed between X and Y`, `how many (days|weeks) before X did Y happen`.
  - **Sequential N-Event Ordering:** `order of the three events: 'X', 'Y', 'Z'`, `which three events happened in order: X, Y, and Z`, `order of the six museums from earliest to latest`.
  - **Multi-Event Aggregation:** `consecutive days in a row`, `total time spent on X, Y, and Z`.
- **Retrieval Strategy:** `QueryAwareMultiSessionRetriever` with subquery decomposition and dynamic session budgeting.

### Outcome 3: `UNCERTAIN`
- **Definition:** The query lacks definitive multi-session or single-session lexical anchors, or exhibits ambiguous cross-session references.
- **Fallback Rule:** Reverts to standard flat MMR (`ChunkRetriever`). Forced diversity is strictly prohibited without high-confidence multi-session evidence.

---

## 4. Multi-Session Retrieval Mechanism

### 4.1 Sub-clause Decomposition
For queries comparing distinct events, the retriever extracts targeted subqueries:
- **Two-Event Comparison:** Extracts `Subquery_A` and `Subquery_B`.
  - Example: *"Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?"*
  - `Subquery_A`: `"Effective Time Management workshop"`
  - `Subquery_B`: `"Data Analysis using Python webinar"`
- **Quoted Event Triples:** Extracts all quoted or comma-separated event clauses.

### 4.2 Candidate Session Scoring
Each candidate session is scored against the full query and its decomposed subqueries:
$$\text{Score}(\text{Sess}, Q) = 0.55 \cdot \text{VectorSim}(\text{Sess}, Q) + 0.45 \cdot \text{BM25}(\text{Sess}, Q)$$
For queries with subqueries $Q_A$ and $Q_B$:
- Best candidate for Event A: $\arg\max_{\text{Sess}} \text{Score}(\text{Sess}, Q_A)$
- Best candidate for Event B: $\arg\max_{\text{Sess}} \text{Score}(\text{Sess}, Q_B)$
This guarantees that both event sessions are represented among the top candidates, completely eliminating the dense retrieval starvation observed in Phase 2B.1!

### 4.3 Dynamic Chunk Budget Allocation
To prevent single-session dominance while avoiding the 1-chunk-per-session starvation:
- **For $S = 2$ (Two Required Sessions):**
  - Session A: Allocated 2 chunks (top turns).
  - Session B: Allocated 2 chunks (top turns).
  - Remainder: 1 chunk allocated to the highest-scoring turn across either session or candidate Session C.
  - Total: 5 chunks. Both events are deeply represented with contiguous turn context.
- **For $S = 3$ (Three Required Sessions):**
  - Session A: Allocated 2 chunks.
  - Session B: Allocated 2 chunks.
  - Session C: Allocated 1 chunk.
  - Total: 5 chunks.
- **For $S \ge 4$ (High-Cardinality Ordering):**
  - Each selected candidate session is allocated exactly 1 top turn chunk.
  - Total: $\min(S, 6)$ chunks.

### 4.4 Chronological Assembly & Provenance Preservation
All retrieved chunks are sorted chronologically by session timestamp and turn index before passing to context assembly:
1. Preserves temporal narrative flow for the LLM.
2. Every chunk retains its immutable `session_id`, `turn_start_idx`, `timestamp`, and verbatim text.
3. No timestamps or dates are inferred from ingestion order.

---

## 5. Inspectability & Diagnostic Logging

The retriever emits a comprehensive diagnostic object attached to `context["query_aware_routing"]`:
```json
{
  "outcome": "MULTI_SESSION",
  "confidence": 0.95,
  "estimated_required_sessions": 2,
  "pattern_matched": "which_first_or",
  "subqueries": [
    "Effective Time Management workshop",
    "Data Analysis using Python webinar"
  ],
  "selected_sessions": [
    {"session_id": "sess_12", "score": 0.884, "allocated_chunks": 2},
    {"session_id": "sess_38", "score": 0.812, "allocated_chunks": 2}
  ],
  "total_budget": 5,
  "budget_allocation": {
    "sess_12": 2,
    "sess_38": 2,
    "sess_fallback": 1
  }
}
```

---

## 6. Verification and Validation Plan

1. **Unit & Integration Tests (`tests/test_query_aware_retrieval.py`):**
   - Verify router classification on all 8 query categories.
   - Verify subquery extraction across all 13 two-event patterns.
   - Verify dynamic budget allocation ($S=2$, $S=3$, $S \ge 4$).
   - Verify inspectable diagnostic logging.
2. **Benchmark Execution:**
   - Run official 133-question evaluation using the Phase 3 Query-Aware Retriever with Phase 2B.2 Arm B synthesis pipeline.
   - Official `qwen2.5:14b` LLM judge at `temperature=0.0`.
3. **Acceptance Criteria Gate:**
   - Must achieve meaningful Category H recovery ($> 6 / 41$).
   - Non-Category H accuracy must not regress below the Arm B baseline ($66 / 92 = 71.74\%$).
   - Production database hashes must remain bit-exact.

