# IVARA — Step 6: Temporal Memory Pipeline Integration Report

**Date:** October 2026  
**Status:** Complete / Fully Tested  
**Branch:** `research/evaluation-baseline`  
**Working Tree Status:** Clean, isolated on research branch  
**Data Integrity:** Production stores (`data/memories.json`, `data/entities.json`) and historical evaluation artifacts remain completely untouched.

---

## 1. Executive Summary

Step 6 resolved the critical temporal integration gap identified in [`evaluation/STEP5_INTEGRATION_AND_EVALUATION_READINESS.md`](../../../evaluation/STEP5_INTEGRATION_AND_EVALUATION_READINESS.md). In Step 4, `TemporalClaim` representation and `TemporalSupersessionEngine` were implemented as isolated algorithms, but were not automatically invoked or persisted during normal ingestion or LongMemEval graph construction. Callers had to supply `temporal_claims` manually.

In Step 6, we wired temporal claim extraction and conservative supersession end-to-end into the real pipeline:
1. **LongMemEval Benchmark Ingestion:** [`TurnGraphBuilder.build_from_chunks`](../../../src/chunking/graph_retrieval.py) automatically extracts claims from chunks, evaluates pairwise supersession, and persists `claim` nodes, `has_claim` provenance edges, and directed `supersedes` edges to the isolated SQLite graph store.
2. **Production Ingestion:** [`process_and_store_memory`](../../../src/extraction/pipeline.py) and [`MemoryGraphBuilder.build_from_records`](../../../src/graph/builder.py) automatically extract claims and persist them with memory node provenance.
3. **Public Answering Pipeline:** [`answer_question`](../../../src/answering.py) automatically queries `graph_store.get_nodes_by_type("claim")` (with an automated extraction fallback if `graph_store` is omitted), passes active claims to [`QueryTemporalResolver`](../../../src/temporal/resolution.py), formats validated temporal context, and validates citations without callers needing to construct `temporal_claims` manually.
4. **Non-Destructive & Conservative:** Historical claims, unresolved contradictions, explicit/inferred provenance, and unanchored dates are preserved without destructive deletion or date fabrication.
5. **Idempotent Ingestion:** Rebuilding the same input produces zero duplicate claims or edges.
6. **Zero Regressions:** All 187 repository tests pass cleanly.

---

## 2. Exact Files & Functions Changed

| File | Function / Class | Change Description |
| :--- | :--- | :--- |
| `src/temporal/claim.py` | `build_claims_from_text` | Enhanced regex patterns to support named entity subjects (e.g., Alice, Bob), flexible line endings, and verbatim text spans while preserving default user attribution. |
| `src/chunking/graph_retrieval.py` | `TurnGraphBuilder.build_from_chunks` | Added Step 5 to extract claims across all chunks, combine with existing claims, process supersession, and batch upsert `claim` nodes, `has_claim` provenance edges, and directed `supersedes` edges into `SQLiteGraphStore`. |
| `src/graph/builder.py` | `MemoryGraphBuilder.build_from_records` | Added Step 4 to extract claims from memories, resolve supersession, and persist `claim` nodes and `has_claim`/`supersedes` edges into the store. |
| `src/extraction/pipeline.py` | `process_and_store_memory` | Added `graph_store=None` parameter; extracts claims with provenance pointing to `memory["id"]`, attaches `temporal_claims` to returned record, and upserts memory container + claims into `graph_store`. |
| `src/memory.py` | `add_memory` | Added `temporal_claims=None` optional argument; stores `temporal_claims` list in memory record when provided. |
| `src/answering.py` | `answer_question` | Added `enable_temporal: bool = True` parameter; automatically queries `graph_store.get_nodes_by_type("claim")` with on-the-fly chunk/memory fallback; formats temporal resolution context. |
| `src/answering.py` | `validate_citations` | Added temporal claim resolution mapping to ensure cited claim IDs map back to source chunk and memory IDs without false rejections. |
| `tests/test_temporal_integration.py` | Full Test Suite | Created 10 isolated integration tests covering production ingestion, chunk graph construction, answering resolution, genuine updates, ambiguous conflicts, subject isolation, unknown dates, idempotence, citation validation, and disabled baseline behavior. |

---

## 3. Real Execution Paths: Before vs. After

### Before Step 6:
```
[Ingestion]
Raw Chunks -> TurnGraphBuilder.build_from_chunks()
                |-> Populates ChunkNode, EpisodeNode, adjacent_to, mentions_entity
                [DISCONNECT: build_claims_from_text NEVER called]
                [DISCONNECT: No claim nodes or supersedes edges populated]

[Answering]
User Query -> answer_question(..., temporal_claims=None)
                |-> graph_store.get_nodes_by_type("claim") returns []
                [DISCONNECT: QueryTemporalResolver completely bypassed]
                |-> Standard context passed to LLM (cannot resolve knowledge updates)
```

### After Step 6:
```
[Ingestion]
Raw Chunks -> TurnGraphBuilder.build_from_chunks()
                |-> 1. Inserts ChunkNode and EpisodeNode
                |-> 2. Inserts contains_chunk and adjacent_to edges
                |-> 3. Inserts mentions_entity edges
                |-> 4. Calls build_claims_from_text(chunk.text, session_id, chunk_id, timestamp)
                |-> 5. Merges with existing claims in SQLiteGraphStore
                |-> 6. Calls TemporalSupersessionEngine().process_claims(combined_claims)
                |-> 7. Batch upserts:
                       - GraphNode(node_type="claim") with valid intervals & is_current
                       - GraphEdge(edge_type="has_claim", source=chunk_id, target=claim_id)
                       - GraphEdge(edge_type="supersedes", source=C_new, target=C_old)

[Answering]
User Query -> answer_question(..., temporal_claims=None, enable_temporal=True)
                |-> 1. Hybrid Dense + BM25 + Ontology Seed Retrieval
                |-> 2. Query-conditioned bounded 1-hop graph expansion
                |-> 3. Automatic claim retrieval:
                       - Queries graph_store.get_nodes_by_type("claim")
                       - (Fallback: extracts on-the-fly if no graph_store supplied)
                |-> 4. Calls QueryTemporalResolver().resolve(query, active_claims)
                       - Dispatches CURRENT vs. HISTORICAL vs. RELATIVE vs. AMBIGUOUS
                       - Filters active vs. historical superseded claims
                |-> 5. Appends formatted temporal context to evidence container
                |-> 6. Invokes answering LLM (or mock caller)
                |-> 7. validate_citations() resolves chunk, memory, and claim IDs
```

---

## 4. Test Evidence

### 1. Focused Integration Test Suite (`tests/test_temporal_integration.py`):
```
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.7.0
collected 10 items

tests/test_temporal_integration.py::test_production_ingestion_creates_temporal_claims PASSED [ 10%]
tests/test_temporal_integration.py::test_longmemeval_chunk_graph_creates_temporal_claims PASSED [ 20%]
tests/test_temporal_integration.py::test_public_answering_resolves_temporal_queries_automatically PASSED [ 30%]
tests/test_temporal_integration.py::test_genuine_update_creates_edge_and_preserves_older_claim PASSED [ 40%]
tests/test_temporal_integration.py::test_ambiguous_contradictions_remain_unresolved PASSED [ 50%]
tests/test_temporal_integration.py::test_unrelated_subjects_and_predicates_do_not_supersede PASSED [ 60%]
tests/test_temporal_integration.py::test_unknown_dates_are_not_fabricated PASSED [ 70%]
tests/test_temporal_integration.py::test_repeated_ingestion_is_strictly_idempotent PASSED [ 80%]
tests/test_temporal_integration.py::test_citation_validation_with_temporal_claims PASSED [ 90%]
tests/test_temporal_integration.py::test_baseline_behavior_unchanged_when_temporal_disabled PASSED [100%]

============================= 10 passed in 11.27s =============================
```

### 2. Full Focused Temporal & Graph Suite (33 Tests):
```
tests/test_temporal_supersession.py (12 passed)
tests/test_bounded_graph_expansion.py (11 passed)
tests/test_temporal_integration.py (10 passed)
============================= 33 passed in 12.04s =============================
```

### 3. Full Repository Test Suite (187 Tests):
```
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.7.0
rootdir: <project-root>
plugins: anyio-4.14.2
collected 187 items

tests\test_answering.py .........                                        [  4%]
tests\test_benchmark.py ..........                                       [ 10%]
tests\test_benchmark_metrics.py .........                                [ 14%]
tests\test_bounded_graph_expansion.py ...........                        [ 20%]
tests\test_extraction_regression.py ...                                  [ 22%]
tests\test_graph_ablation.py .....                                       [ 25%]
tests\test_graph_store.py .......                                        [ 28%]
tests\test_hybrid_retrieval.py .......                                   [ 32%]
tests\test_longmemeval_adapter.py .........                              [ 37%]
tests\test_longmemeval_runner.py ....                                    [ 39%]
tests\test_memory_graph.py ................                              [ 48%]
tests\test_ontology.py ...................................               [ 66%]
tests\test_pipeline_offline.py ......                                    [ 70%]
tests\test_query_decomposition.py ............                           [ 76%]
tests\test_retrieval_grounded.py .............                           [ 83%]
tests\test_temporal_integration.py ..........                            [ 88%]
tests\test_temporal_supersession.py ............                         [ 95%]
tests\test_turn_chunking.py .........                                    [100%]

============================ 187 passed in 16.12s =============================
```

---

## 5. Limitations of the Deterministic Extractor

Per user requirements, external LLMs were not called during claim extraction. The implementation relies on deterministic pattern matching in `build_claims_from_text`.

### Explicit Capabilities:
- **Residence:** Matches `lived in <loc>`, `moved to <loc>`, `live/lives in <loc>`, with optional explicit year (e.g., `in 2025`).
- **Employment:** Matches `worked at <org>`, `works at <org>`, with optional year.
- **Hobbies:** Matches `My hobby is <hob>`, `I like/enjoy playing <hob>`.
- **Visits:** Matches `visited <place>`, with optional year.
- **Subject Extraction:** Identifies named entities (e.g. `Alice`, `Bob`) or defaults to `user` for first-person statements (`I`, `User: I`).

### Known Boundaries & Limitations:
1. **Phrasing Sensitivity:** Free-form or poetic phrasing (e.g., *"My heart belongs to Bangalore since last spring"*) will not be extracted as a `lives_in` claim without expanding regex patterns.
2. **Implicit Updates:** Idiomatic expressions (e.g., *"Bid farewell to my old flat and unpacked boxes in the new place"*) require semantic understanding that regexes cannot capture without an LLM extractor.
3. **Multi-hop Date Math:** Complex relative temporal anchors (e.g., *"Three years after graduating college"*) remain unanchored (`valid_from=None`) unless explicit calendar years are stated.

---

## 6. Runtime Overhead Measurements

Measured on a multi-session conversational corpus (15 sessions, 50 nodes, 104 edges, 12 claims) via `scratch/benchmark_step5_end_to_end.py`:

| Pipeline Stage | Latency |
| :--- | :--- |
| **1. Turn Chunking** | **0.15 ms** (15 chunks) |
| **2. SQLite Graph Construction** | **5.95 ms** (50 nodes, 104 edges) |
| **3. Temporal Claim & Supersession Engine** | **1.83 ms** (12 claims processed) |
| **Total Ingestion Latency** | **7.93 ms** |
| **Cold-Start Query Latency** | **97.29 ms** |
| **Warm Steady-State Query Mean** | **72.39 ms** |

### Latency Verdict:
Ingestion overhead of temporal claim extraction and supersession is **1.83 ms** (less than 2 milliseconds). Query latency remains under **75 ms** warm, representing an enormous speedup over the historical non-batched ablation (268 seconds).

---

## 7. Safety, Branch, and Invariance Confirmation

- **Git Branch:** `research/evaluation-baseline` (main has not been modified).
- **Production Stores:** `git diff main..HEAD -- data/` returns zero diff (`data/memories.json` and `data/entities.json` remain pristine).
- **Historical Evaluation Artifacts:** All historical evaluation manifests, outputs, reports, and benchmark runs remain completely untouched.
- **Test Suite Pass Rate:** 187/187 tests passing (100%).

---

## 8. Completion & Next Steps

Step 6 implementation and verification are **complete**. Per user constraints, benchmark execution has **not** been launched. We are awaiting user instructions.
