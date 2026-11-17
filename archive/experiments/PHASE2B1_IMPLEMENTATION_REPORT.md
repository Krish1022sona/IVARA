# IVARA — Phase 2B.1: Temporal Event Graph Implementation Report

**Date:** 2026-11-17  
**Author:** Antigravity (DeepMind Team)  
**Branch:** `experiment/temporal-kg-phase2b`  
**Baseline Commit:** `55a273bff20fa0eec1b2d361e8a03b217d147abe`  
**Status:** Implemented & Verified (14/14 Unit Tests Passing)  

---

## 1. Executive Summary

In accordance with the Phase 2B.1 requirements, an isolated, minimal **Temporal Event Graph (TEG)** prototype has been designed, implemented, and verified in the IVARA codebase. 

The implementation addresses the core findings from `PHASE2A1_FAILURE_FORENSICS.md`:
1. **Explicit Event Nodes with Decoupled Time:** Decoupled event occurrence intervals (`valid_from`, `valid_to`) from conversation ingestion time (`recorded_at`), eliminating date confusion across disjoint dialogue sessions.
2. **Preservation of Uncertainty:** Introduced `TemporalPrecision` (`EXACT_DAY`, `MONTH`, `YEAR`, `RELATIVE`, `UNKNOWN`) to prevent synthetic date fabrication when episodic evidence is approximate or relative.
3. **Strict Non-Ingestion Ordering Safety:** Built a pairwise relationship builder that derives `OCCURRED_BEFORE` and `OCCURRED_AFTER` strictly from grounded interval comparisons or explicit textual markers, never from conversation logging order.
4. **Minimal 4-Relationship Vocabulary:** Implemented `OCCURRED_BEFORE`, `OCCURRED_AFTER`, `SUPERSEDES`, and `INVALIDATES`.
5. **Bounded Graph Traversal & Episodic Supplementation:** The graph retriever supplements episodic chunks with structured timeline ordering and missing cross-session evidence, falling back gracefully to episodic retrieval when confidence is $< 0.4$.
6. **Answer Synthesis & Parsing Protection:** Hardened regex extraction in `src/answering.py` to eliminate malformed JSON fallbacks when the LLM emits valid textual answers.

All 14 unit and integration tests in `tests/test_event_graph.py` pass with 100% success.

---

## 2. Isolation & Safety Verification

### 2.1 Branch Isolation
- Active branch: `experiment/temporal-kg-phase2b`
- Branched from: `55a273bff20fa0eec1b2d361e8a03b217d147abe` (Released Baseline)
- Previous experiment branches preserved: `experiment/temporal-phase1`, `experiment/hierarchical-retrieval-phase2a`, `experiment/adaptive-hierarchical-phase2a1`.

### 2.2 Production Database Immutability Check
All production databases remain 100% untouched with bit-exact SHA-256 matches:

| File | SHA-256 Hash | Status |
|---|---|:---:|
| `data/memories.json` | `489e3ea2138022f97a590808cee3dfba03465966eda06d750e340d6d0fddf858` | **Bit-Exact Match** |
| `data/entities.json` | `51f993fbe1cfc7f40084724499cd2dd216b30d8e3aeeace9824d1863d84fc09e` | **Bit-Exact Match** |
| `data/memory_graph.db` | `f6b9f9cf7d35674e461350ffa87dcfa735ae0d7643c46e30138f56bffb31f95d` | **Bit-Exact Match** |
| `data/conversations.db` | `af88c18f31d768e7a62a91d5c4dd40e1a8271fe4de1ee86e56c4d180021bd11f` | **Bit-Exact Match** |

All benchmark operations run in isolated per-instance databases (`output_dir/stores/{qid}.db`).

---

## 3. Architecture & Implementation Details

### 3.1 Event Representation (`src/temporal/event_graph.py`)
- `EventNode`:
  - `event_id`: Deterministic hash derived from session ID, description, and valid date.
  - `description`: Canonical activity/state descriptor (e.g. `Visited Paris`, `Cancelled FarmFresh`).
  - `entity_ids`: Associated ontology entity references (e.g. `ent_paris`, `ent_farmfresh`).
  - `session_id` & `turn_id`: Precise provenance coordinates.
  - `valid_from` & `valid_to`: Standardized ISO strings (`YYYY-MM-DD`, `YYYY-MM`, or `YYYY`).
  - `recorded_at`: Ingestion date (`session.date`).
  - `precision`: `TemporalPrecision` enum (`EXACT_DAY`, `MONTH`, `YEAR`, `RELATIVE`, `UNKNOWN`).
  - `confidence`: Extraction confidence [0.0, 1.0].
  - `provenance`: Verbatim supporting sentence from the turn.
  - Full serialization/deserialization to `GraphNode` for storage in `SQLiteGraphStore`.

### 3.2 Decoupled Event Extraction (`src/temporal/event_extractor.py`)
- `TemporalEventExtractor`:
  - Scans turns for 11 core personal episodic activity patterns: attendance, travel, relocation, employment, education, subscriptions, and purchases.
  - Resolves calendar expressions deterministically:
    - Absolute dates (`April 6, 2023`, `2023-04-06`) $\to$ `EXACT_DAY`.
    - Relative expressions (`yesterday`, `today`, `3 days ago`, `last week`, `last month`) $\to$ deterministically anchored relative to `recorded_at`.
    - Month expressions (`in July 2023`) $\to$ `MONTH` (`2023-07-01` to `2023-07-31`).
    - Year expressions (`in 2021`) $\to$ `YEAR` (`2021-01-01` to `2021-12-31`).
    - Unanchored / approximate mentions $\to$ `UNKNOWN`, preserving uncertainty without date fabrication.

### 3.3 Strict Non-Ingestion Relationship Construction (`src/temporal/relationship_builder.py`)
- `TemporalRelationshipBuilder`:
  - **Interval Comparisons (`OCCURRED_BEFORE` / `OCCURRED_AFTER`):**
    - Emitted if and only if $E_1.\text{valid\_to} < E_2.\text{valid\_from}$.
    - Dual edge $E_2 \xrightarrow{\text{OCCURRED\_AFTER}} E_1$ created synchronously.
    - Confidence penalization applied if either event has coarse precision (YEAR: 0.85, MONTH: 0.92).
  - **State Supersession (`SUPERSEDES`):**
    - Identifies mutually exclusive state slots (`lived in`, `worked at`) with differing targets (e.g., Boston vs. Seattle) and strictly later dates.
  - **State Invalidation (`INVALIDATES`):**
    - Identifies termination actions (`cancelled`, `terminated`) linked to active memberships or subscriptions.
  - **CRITICAL SAFETY ENFORCEMENT:**
    - If events lack verifiable dates, NO ordering edge is created based on session logging order. Unit test `test_strict_safety_no_ingestion_order_inference` confirms 0 edges emitted for undated sessions.

### 3.4 Bounded Traversal & Retrieval (`src/temporal/event_graph_traversal.py`)
- `TemporalEventGraphRetriever`:
  - Matches candidate query events via keyword and entity overlap.
  - Traverses 1- to 2-hop paths for ordering, supersession, and invalidation queries.
  - Assembles structured `[TEMPORAL EVENT GRAPH EVIDENCE]` block containing:
    - Event descriptions, dates, and precisions.
    - Verbatim source provenance quotes.
    - Verified ordering findings with exact date deltas.
    - Chronological timelines.
  - Returns `supporting_chunk_ids` to supplement the episodic context.
  - Emits `confidence = 0.0` and empty text when graph evidence is insufficient, falling back cleanly to episodic retrieval.

### 3.5 Graph Store & Pipeline Integration
- `src/chunking/graph_retrieval.py`:
  - `TurnGraphBuilder`: Automatically extracts `EventNode`s and `TemporalRelationshipEdge`s during graph indexing and commits them to `SQLiteGraphStore`.
  - `TurnGraphRetriever`: Added Safe Edge Filter 4: traversing `contains_event \to EventNode \to TemporalEdge \to target EventNode \to target Chunk`, allowing graph-guided discovery of cross-session evidence.
- `src/answering.py`:
  - Step 3.6: Executes `TemporalEventGraphRetriever` on the instance store, injecting verified graph evidence into prompt context.
  - Supplements `context["direct_evidence"]` with any supporting chunks discovered through the event graph.
  - Synthesis Protection: Hardened multi-pattern regex fallback to extract answers from truncated or unformatted LLM outputs, preventing unnecessary fallback to `Could not reliably synthesize an answer`.

---

## 4. Test Suite Execution

All 14 tests in `tests/test_event_graph.py` were executed via pytest:

```
tests/test_event_graph.py::TestEventExtraction::test_exact_date_extraction PASSED
tests/test_event_graph.py::TestEventExtraction::test_month_precision_preservation PASSED
tests/test_event_graph.py::TestEventExtraction::test_year_precision_preservation PASSED
tests/test_event_graph.py::TestEventExtraction::test_relative_date_anchoring PASSED
tests/test_event_graph.py::TestEventExtraction::test_unknown_date_preserves_uncertainty PASSED
tests/test_event_graph.py::TestTemporalRelationships::test_occurred_before_and_after_grounded PASSED
tests/test_event_graph.py::TestTemporalRelationships::test_strict_safety_no_ingestion_order_inference PASSED
tests/test_event_graph.py::TestTemporalRelationships::test_event_order_opposite_to_ingestion_order PASSED
tests/test_event_graph.py::TestTemporalRelationships::test_state_supersedes_relationship PASSED
tests/test_event_graph.py::TestTemporalRelationships::test_invalidation_relationship PASSED
tests/test_event_graph.py::TestBoundedGraphTraversalAndRetrieval::test_ordering_traversal_between_two_events PASSED
tests/test_event_graph.py::TestBoundedGraphTraversalAndRetrieval::test_supersession_traversal PASSED
tests/test_event_graph.py::TestBoundedGraphTraversalAndRetrieval::test_fallback_on_insufficient_graph_confidence PASSED
tests/test_event_graph.py::TestStoreIntegration::test_store_population_and_retrieval PASSED

============================= 14 passed in 7.77s ==============================
```

---

## 5. Next Steps: Benchmark Evaluation

With the implementation verified and tested, the prototype is ready for full benchmark evaluation against the 133-question LongMemEval temporal benchmark using the official `qwen2.5:14b` judge:
1. Target Cohort 1: 35 remaining Category H failures.
2. Target Cohort 2: 92 non-Category H questions (tracking recovery of the 11 regressions).
3. Full Cohort: All 133 temporal questions.

