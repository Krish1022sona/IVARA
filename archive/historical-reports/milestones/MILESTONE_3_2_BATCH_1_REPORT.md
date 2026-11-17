# IVARA Milestone 3.2 — Batch 1 Implementation Report: Adaptive Router

**Date:** November 15, 2026  
**Status:** Completed & Verified  
**Active Branch:** `research/evaluation-baseline`  
**Test Suite:** 382/382 Passing Tests (17 new router tests + 365 existing tests)  
**Production Store Hashes:** 100% Unchanged (Zero Data Mutations)

---

## 1. Executive Summary

In Milestone 3.1, our empirical audit demonstrated that while fused multi-source retrieval achieved high retrieval recall, it suffered from context dilution and distractor injection in specific query classes:
- Episodic-only retrieval was superior on conversational context and unanswerable queries.
- Unversioned Knowledge Graph relationships displaced critical temporal evidence and induced hallucinations on historical-state questions.
- Fusing all sources unconditionally harmed answer generation precision.

In accordance with `MILESTONE_3_2_DESIGN.md`, we have implemented **Batch 1: Adaptive Router (`AdaptiveRouter`)**. The router acts as a lightweight, deterministic first-stage gate that inspects user queries in sub-millisecond time and configures retrieval channels (`episodic`, `semantic`, `graph`) based on query semantics, temporal indicators, and relational bridges.

---

## 2. Graph Schema Audit Findings: `valid_from` & `valid_to`

Prior to implementation, we conducted a rigorous inspection of the SQLite knowledge graph store (`data/memory_graph.db` and `evaluation/validation_sprint/ivara_store_validation/memory_graph.db`).

### Schema Inspection
The `edges` table schema is defined as:
```sql
CREATE TABLE edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    relation TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    properties TEXT DEFAULT '{}',
    PRIMARY KEY (source, target, relation),
    FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE
);
```

### Empirical Findings
1. **No Dedicated Columns:** There are no dedicated columns for temporal validity (`valid_from`, `valid_to`) on graph edges.
2. **Properties Inspection:**
   - Out of all edges in production and validation stores, 10 structural claim edges possess `properties` set to `{"valid_from": null, "valid_to": null}`.
   - All domain relationship edges (e.g., `employed_by`, `collaborated_with`) have empty properties `{}`.
3. **Critical Architectural Implication:**
   - Graph edges **do not possess populated valid-time intervals**. They only record an ingestion `timestamp`.
   - The knowledge graph represents current or positive assertions without temporal versioning.
   - **Mandatory Invariant Enforced:** `AdaptiveRouter` must **never** enable knowledge graph retrieval for historical-state queries (`historical_state`), as unversioned current relationships directly contradict past biographical states.

---

## 3. Architecture & Routing Logic (`AdaptiveRouter`)

The implementation resides in `src/unified_retrieval/router.py` with public interfaces exported in `src/unified_retrieval/__init__.py`.

### 3.1 The 8 Canonical Query Categories

| Category | Typical Query Pattern | Episodic | Semantic | Graph | Context Mode |
|---|---|:---:|:---:|:---:|---|
| `single_fact` | "What is Arjun's birthday?", "From which university did Arjun graduate?" | **ON** | **ON** | **OFF** | `single_fact` |
| `historical_state` | "What was Arjun's previous job title before switching?", "Where did he live originally?" | **ON** | **ON** | **OFF** | `historical_state` |
| `temporal_longitudinal` | "Did the promotion occur before or after the relocation to Bengaluru?" | **ON** | **ON** | **OFF** | `temporal_longitudinal` |
| `cross_session_synth` | "Trace Arjun's career progression across all companies.", "Full lifecycle of CacheMesh" | **ON** | **ON** | **OFF** | `cross_session_synth` |
| `entity_relationships` | "What is the relationship between Arjun and Vikram Rao?" | **ON** | **ON** | **COND** | `entity_relationships` |
| `multi_hop_reasoning` | "In which city does the manager who supervised Arjun at HyperScale live?" | **ON** | **ON** | **COND** | `multi_hop` |
| `negative_information` | "Did Arjun undergo knee surgery?", "Does Arjun drink coffee or avoid caffeine?" | **ON** | **ON** | **OFF** | `negative_information` |
| `unanswerable` | "What is the capital of Mars according to Arjun?", "President of the moon in 2099" | **ON** | **OFF** | **OFF** | `unanswerable` |
| `safe_fallback` | Ambiguous queries, low confidence, or unclassified inputs | **ON** | **ON** | **OFF** | `standard` |

*Note: **COND** (Conditional) activates the graph channel only when confidence $\ge 0.65$ and entity mentions or relative clause bridges are cleanly resolved.*

### 3.2 Invariants Enforced by Construction

1. **Episodic Backbone Invariance:**
   Episodic retrieval is **never disabled** (`episodic_enabled = True` across all categories). Episodic turns provide ground-truth conversational evidence and chronological anchoring.
2. **Graph Guard for Historical State:**
   Queries matching historical keywords ("previous", "former", "initially", "before switching", "reverted to") strictly disable graph retrieval.
3. **Negative Information Safeguard:**
   Negative or constraint questions disable the positive-assertion graph store to prevent false-positive distraction. The system preserves the distinction between *absence of evidence* and *evidence of absence*.
4. **Unknown Entity Safeguard:**
   Queries referencing entities absent from the domain index route to `safe_fallback` with standard channels active. They are **never** prematurely classified as `unanswerable`. Unanswerability remains an evidence-sufficiency decision downstream.
5. **Deterministic Sub-Millisecond Execution:**
   Pure compiled regex and token-boundary feature extraction. Zero external API calls, zero local LLM inferences.

---

## 4. Integration into Unified Pipeline

`AdaptiveRouter` is integrated into `UnifiedRetriever` (`src/unified_retrieval/pipeline.py`):

```python
retriever = UnifiedRetriever(
    conversation_store=store,
    memories=memories,
    graph_store=graph_store,
    use_adaptive_routing=False  # Backward compatibility: default is False
)

# Per-query dynamic execution:
result = retriever.retrieve(query, use_adaptive_routing=True)
```

When `use_adaptive_routing` is enabled:
1. `router.route(clean_q)` computes the `RouteDecision`.
2. Channels are filtered: `enable_episodic = enable_episodic and decision.episodic_enabled`, etc.
3. Full routing diagnostics are captured in `result.diagnostics["routing"]`.

---

## 5. Performance & Latency Benchmark

Benchmarked on 200 consecutive routing passes covering all 8 query classes:

- **Total Execution Time (200 queries):** $10.14\text{ ms}$
- **Average Routing Latency:** **$0.0507\text{ ms}$ ($50.7\ \mu\text{s}$ per query)**
- **Throughput:** **$\sim 19,720\text{ queries/second}$**
- **Overhead:** Adds negligible latency ($<0.06\text{ ms}$) to the retrieval pipeline while eliminating unnecessary semantic and graph queries for unanswerable and single-fact queries.

---

## 6. Verification & Test Suite

A dedicated test suite was implemented in `tests/test_adaptive_router.py`:
- `TestAdaptiveRouterCategories`: All 8 canonical query categories tested with realistic benchmarks.
- `TestAdaptiveRouterInvariantsAndEdgeCases`: Safe fallback, unknown entities, empty strings, episodic backbone persistence, serialization, and sub-millisecond execution.
- `TestUnifiedRetrieverIntegration`: Default pipeline invariance, active routing channel masking, and per-query overrides.

### Test Results
```text
tests/test_adaptive_router.py: 17 passed in 6.70s
Full Test Suite: 382 passed in 32.07s (0 failures, 0 errors, 0 regressions)
```

### Production Store Integrity
```text
memories.json:    489e3ea2138022f97a590808cee3dfba03465966eda06d750e340d6d0fddf858 (MATCH)
entities.json:    51f993fbe1cfc7f40084724499cd2dd216b30d8e3aeeace9824d1863d84fc09e (MATCH)
memory_graph.db:  f6b9f9cf7d35674e461350ffa87dcfa735ae0d7643c46e30138f56bffb31f95d (MATCH)
conversations.db: af88c18f31d768e7a62a91d5c4dd40e1a8271fe4de1ee86e56c4d180021bd11f (MATCH)
```

---

## 7. Next Steps & Halt Condition

Milestone 3.2: Batch 1 is **complete**. In accordance with our execution protocol:
- No commits or pushes have been made.
- We have **halted** before Batch 2 (Evidence Selector).
- We await user review and approval before proceeding with Batch 2.

