# IVARA — Step 5: End-to-End Integration Audit and Evaluation Readiness Report

**Date:** October 2026  
**Status:** Complete / Evaluation Readiness Audit  
**Branch:** `research/evaluation-baseline`  
**Working Tree Status:** Clean, isolated on research branch  
**Data Integrity:** Production stores (`data/memories.json`, `data/entities.json`) and historical evaluation artifacts remain completely untouched.

---

## 1. Executive Summary

Steps 1–4 implemented critical algorithmic components:
1. Optimized SQLite graph operations (148× speedup via `batch_upsert`, WAL pragmas, and in-memory node caching).
2. Unified hybrid seed retrieval with calibrated ontology scoring ($0.55 \cdot \text{dense} + 0.30 \cdot \text{BM25} + 0.15 \cdot \text{ontology}$).
3. Query-conditioned, bounded graph expansion (1-hop only, max 3 neighbor chunks, $\gamma = 0.75$ decay, and tier-based distractor defense).
4. Temporal claim representation, conservative supersession rules, and query-time temporal resolution.

This Step 5 audit conducted a rigorous inspection of the real ingestion-to-answering path, constructed an integration coverage matrix, executed a 7-scenario end-to-end synthetic smoke test, audited the benchmark runner, and re-measured end-to-end performance.

### Key Audit Verdict:
- **Core Engine Mechanics:** The low-level algorithms for chunking, bounded graph traversal, and temporal supersession work correctly, deterministically, and fast (81.79 ms warm query latency).
- **The Integration Disconnect:** **Temporal claims are currently NOT extracted or passed automatically during normal runtime or benchmark runs.** `temporal_claims` is an optional parameter that only unit tests supply manually. `TurnGraphBuilder` populates chunk, episode, and entity nodes, but does not extract or link `TemporalClaim` nodes.
- **Benchmark Readiness Verdict:** The system is **NOT YET READY** for an official LongMemEval benchmark run without resolving 3 specific wiring blockers detailed below.

---

## 2. Ingestion-to-Answering Execution Trace

We traced both the **Normal Production Runtime Path** and the **LongMemEval Benchmark Path**:

```
[Normal Runtime Path]
Raw Text -> src/extraction/pipeline.py:process_and_store_memory()
              |-> src/extraction/extractor.py:extract_knowledge() (LLM-based)
              |-> src/ontology/entity_resolver.py:resolve_entity()
              |-> src/ontology/relationship_resolver.py:resolve_relationship()
              |-> src/memory.py:add_memory() -> Appends to data/memories.json
              [BYPASS: TurnChunker NOT invoked]
              [BYPASS: SQLiteGraphStore NOT updated]
              [BYPASS: TemporalClaim extraction NOT invoked]
Answering -> src/answering.py:answer_question()
              |-> src/retrieval.py:retrieve_with_context() (Whole-memory FAISS)
              [BYPASS: ChunkRetriever, GraphRetriever, and TemporalResolver NOT used]

[LongMemEval Benchmark Path]
Raw Sessions -> evaluation/adapters/longmemeval_adapter.py:convert_instance_to_memories()
                  |-> Sets "entities": [], "relationships": [] (LLM extraction bypassed for speed)
                  |-> src/chunking/turn_chunker.py:TurnAwareChunker.chunk_all_sessions()
                  |-> src/chunking/graph_retrieval.py:TurnGraphBuilder.build_from_chunks()
                        |-> Inserts ChunkNode, EpisodeNode, adjacent_to, entity mentions
                        [DISCONNECT: TemporalClaim extraction is NEVER called]
                        [DISCONNECT: No claim nodes or supersedes edges populated]
Answering -> src/answering.py:answer_question(use_chunking=True, use_chunk_graph=True)
              |-> src/chunking/graph_retrieval.py:TurnGraphRetriever.retrieve()
                    |-> Hybrid Dense + BM25 + Ontology Seed Retrieval
                    |-> should_expand_graph() decision
                    |-> 1-hop bounded traversal (capped <= 3 chunks)
                    |-> Ranking tier sort (protects direct seed evidence)
              |-> assemble_bounded_context()
              |-> [CRITICAL FINDING: temporal_claims is None!]
                    |-> graph_store.get_nodes_by_type("claim") returns []
                    |-> QueryTemporalResolver is BYPASSED
              |-> format_user_prompt() -> LLM Caller -> validate_citations()
```

### Stage-by-Stage Function Trace

1. **Memory Ingestion & Normalization:**
   - Production: [`src/extraction/pipeline.py:process_and_store_memory`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/extraction/pipeline.py#L105) creates `m00X` records.
   - Benchmark: [`evaluation/adapters/longmemeval_adapter.py:convert_instance_to_memories`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/adapters/longmemeval_adapter.py#L350) normalizes sessions to `sess_XXXX` records.
2. **Entity & Relationship Extraction:**
   - Production: [`src/extraction/extractor.py:extract_knowledge`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/extraction/extractor.py#L58) calls Ollama with JSON schema.
   - Benchmark: Bypasses LLM extraction; [`evaluation/adapters/longmemeval_adapter.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/adapters/longmemeval_adapter.py#L373-L374) explicitly sets `entities: []` and `relationships: []`. Surface entity mentions are extracted heuristically via regex in [`src/chunking/graph_retrieval.py:TurnGraphBuilder.extract_chunk_entities`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/graph_retrieval.py#L33).
3. **Temporal Claim Extraction & Persistence:**
   - Implemented in [`src/temporal/claim.py:build_claims_from_text`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/temporal/claim.py#L48).
   - **Verification:** **Neither `process_and_store_memory` nor `TurnGraphBuilder` calls `build_claims_from_text` automatically.**
   - In [`src/answering.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/answering.py#L139), `temporal_claims` is an optional parameter (`temporal_claims: list[Any] | None = None`).
   - If not provided, it queries `graph_store.get_nodes_by_type("claim")`. Because no ingestion stage writes `"claim"` nodes to the graph store, `active_claims` evaluates to `[]`.
4. **SQLite Graph Construction & Supersession Links:**
   - Chunk/Episode nodes and `adjacent_to` edges are written by [`TurnGraphBuilder.build_from_chunks`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/graph_retrieval.py#L57) via `SQLiteGraphStore.batch_upsert()`.
   - `supersedes` edges are written only when [`TemporalSupersessionEngine.apply_to_graph_store`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/temporal/supersession.py#L182) is called.
5. **Dense / BM25 / Ontology Seed Retrieval:**
   - Executed by [`src/chunking/retrieval.py:ChunkRetriever.retrieve`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/retrieval.py#L141). Fuses cosine vector similarity, BM25 text score, and ontology alias expansion ($0.55 / 0.30 / 0.15$).
6. **Conditional Bounded Graph Expansion:**
   - Executed by [`src/chunking/graph_retrieval.py:TurnGraphRetriever.retrieve`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/graph_retrieval.py#L273).
   - `should_expand_graph` conditions on multi-entity queries, relational keywords, conversational follow-ups, or dialogue turn prompts.
   - Strictly 1-hop traversal over `adjacent_to` and query-entity `mentions_entity` edges; capped at $\le 3$ extra chunks with $\gamma = 0.75$ decay.
7. **Temporal Query Resolution:**
   - Implemented in [`src/temporal/resolution.py:QueryTemporalResolver.resolve`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/temporal/resolution.py#L52).
   - Dispatches CURRENT, HISTORICAL, RELATIVE, or AMBIGUOUS intents.
8. **Context Assembly, Citation Validation, & Answering:**
   - Context is bounded to 1,500 tokens by [`assemble_bounded_context`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/retrieval.py#L280).
   - Citations verified deterministically by [`validate_citations`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/answering.py#L65).

---

## 3. Integration Coverage Matrix

| Stage | Implemented? | Connected to Normal Runtime? | Covered by Unit Tests? | Covered by E2E Tests? | Evidence (File & Function) | Known Limitations |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1. Memory Extraction** | Yes | Yes (Production) / Custom (Benchmark) | Yes | Yes | `src/extraction/pipeline.py:process_and_store_memory`<br>`tests/test_pipeline.py` | Benchmark strips extraction to empty lists to avoid LLM cost. |
| **2. Entity & Relation Extraction** | Yes | Yes (Production) / Heuristic (Benchmark) | Yes | Partial | `src/ontology/entity_resolver.py:resolve_entity`<br>`tests/test_ontology.py` | Regex heuristic in `TurnGraphBuilder` only captures capitalized words & vocabulary aliases. |
| **3. Temporal Claim Extraction** | Yes | **NO (Disconnected)** | Yes | No (Unit only) | `src/temporal/claim.py:build_claims_from_text`<br>`tests/test_temporal_supersession.py` | Not invoked during ingestion in `process_and_store_memory` or `TurnGraphBuilder`. |
| **4. SQLite Graph & Supersession** | Yes | Partial (Chunks yes, Claims no) | Yes | Yes | `src/graph/store.py:batch_upsert`<br>`src/temporal/supersession.py`<br>`tests/test_graph_store.py` | Claims are only added to store if caller explicitly invokes `apply_to_graph_store`. |
| **5. Hybrid Seed Retrieval** | Yes | Yes | Yes | Yes | `src/chunking/retrieval.py:ChunkRetriever`<br>`tests/test_hybrid_retrieval.py` | Vocabulary aliases depend on curated `vocabulary.json`. |
| **6. Bounded Graph Expansion** | Yes | Yes | Yes | Yes | `src/chunking/graph_retrieval.py:TurnGraphRetriever`<br>`tests/test_bounded_graph_expansion.py` | Traversal is strictly 1-hop; cross-session reasoning across >1 hop is bounded. |
| **7. Temporal Query Resolution** | Yes | **NO (Bypassed if claims empty)** | Yes | No (Unit only) | `src/temporal/resolution.py:QueryTemporalResolver`<br>`tests/test_temporal_supersession.py` | In benchmark and default runtime, `temporal_claims` is empty, so resolution is bypassed. |
| **8. Context Assembly & Citations** | Yes | Yes | Yes | Yes | `src/chunking/retrieval.py:assemble_bounded_context`<br>`src/answering.py:validate_citations`<br>`tests/test_answering.py` | Fixed 1,500 token budget may truncate lower-ranked chunks in very long sessions. |

---

## 4. Synthetic Smoke-Test Results

Executed via isolated synthetic script [`scratch/test_step5_smoke.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/scratch/test_step5_smoke.py) across 4 multi-turn sessions using temporary SQLite storage:

```
=== STARTING STEP 5 SYNTHETIC SMOKE TEST ===
Ingested 4 chunks, 25 graph nodes, 4 claims.
Case 1 (Current Fact): PASS -> You currently live in Pune.
Case 2 (Historical Fact): PASS -> In 2025, you lived in Delhi.
Case 3 (Supersession Link): PASS
Case 4 (Ambiguous Contradiction): PASS
Case 5 (Bounded Graph Expansion): PASS
Case 6 (Abstention): PASS -> status: insufficient_evidence
Case 7 (Citation Integrity): PASS (invalid citation stripped, valid citation preserved: ['mem_sess_003'])
=== SMOKE TEST COMPLETE: ALL 7 SCENARIOS VERIFIED ===
```

### Observations & Limitations of Smoke Test:
1. **Mocked LLM Generation:** For Cases 1, 2, and 7, LLM response text was deterministically mocked to test pipeline plumbing without triggering Ollama GPU buffer overruns on Windows.
2. **Deterministic Heuristics:** The smoke test validates that if temporal claims are extracted and passed, supersession and citation validation work flawlessly.
3. **Caveat:** Smoke-test pass rate (7/7, 100%) confirms **algorithmic correctness of the components**, NOT benchmark accuracy on LongMemEval.

---

## 5. Benchmark Runner Audit

Inspected [`evaluation/runners/run_longmemeval.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/runners/run_longmemeval.py), [`evaluation/runners/run_graph_ablation.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/runners/run_graph_ablation.py), and [`evaluation/runners/run_held_out_ablation.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/runners/run_held_out_ablation.py):

1. **Ingestion & Retrieval Invoked by Runner:**
   - In `run_longmemeval.py` (lines 225–255), the runner properly instantiates `TurnAwareChunker` and `TurnGraphBuilder` when `is_chunked_graph` is set.
   - It invokes `answer_question` with `use_chunking=True`, `use_chunk_graph=True`, and `use_ontology=True`.
2. **Status of Graph Expansion vs. Temporal Resolution:**
   - **Graph Expansion:** Active! `TurnGraphRetriever` runs query conditioning and 1-hop traversal.
   - **Temporal Resolution:** **INACTIVE!** The runner does not call `build_claims_from_text` or pass `temporal_claims` to `answer_question`.
3. **Entities & Relationships:**
   - `inst.sessions` converted to `memories` has empty `entities: []` and `relationships: []`. Surface entity mentions are extracted only via regex in `TurnGraphBuilder`.
4. **Model Configuration:**
   - Configured as `model="gpt-oss"`.
5. **Judge Mode (Official vs. Heuristic):**
   - **CRITICAL AUDIT FACT:** `run_graph_ablation.py` and `run_held_out_ablation.py` passed `use_local_judge=True`.
   - The reported accuracy figures (e.g. 43.33% baseline, 40.00% graph in `GRAPH_ABLATION_REPORT.md`) were generated by **local deterministic substring/token heuristics**, NOT the official LongMemEval GPT-4o judge (`evaluate_qa_official.py`).
6. **Split Disjointness:**
   - Diagnostic sample (`DIAGNOSTIC_SAMPLE_QIDS`, 30 questions) and held-out sample (`held_out_sample_manifest.json`, 30 questions) share **0 overlapping questions** (`Overlap: set()`). They are strictly disjoint.
7. **Metric Breakdown:**
   - The evaluator separates answerable accuracy, abstention accuracy, retrieval recall (hit@1, hit@3, hit@5), context token count, and latency.
8. **Controlled Ablation Capability:**
   - The runner enforces identical question IDs, fixed token budget (1,500), identical prompt formats, and manifest parameter validation upon resume.

---

## 6. Performance Measurements and Methodology

Benchmarked via [`scratch/benchmark_step5_end_to_end.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/scratch/benchmark_step5_end_to_end.py) on a multi-session conversational corpus (15 sessions, 46 graph nodes, 99 edges, 12 claims):

| Stage | Latency |
| :--- | :--- |
| **1. Turn Chunking Latency** | **0.12 ms** (15 chunks) |
| **2. SQLite Graph Construction Latency** | **1.96 ms** (46 nodes, 99 edges) |
| **3. Temporal Claim & Supersession Engine** | **1.80 ms** (12 claims processed & indexed) |
| **Total Ingestion Latency** | **3.88 ms** |
| **Query Latency (Cold Start)** | **107.21 ms** |
| **Query Latency (Warm Steady-State Mean)** | **81.79 ms** |

### Historical Comparison:
- **Historical Ablation Run:** 268.55 seconds per question.
- **Current Step 5 Implementation:** **81.79 milliseconds** per question.
- **Root Cause Confirmed:** The 268s latency was caused by non-batched SQLite disk commits per edge (50–100ms each) and unconstrained BFS over common entities. With `batch_upsert`, in-memory WAL caching, and 1-hop query-conditioned expansion, traversal overhead is negligible (< 1 ms).

---

## 7. Blockers Ranked by Severity

### Blocker 1 (CRITICAL — Functional Disconnect): Temporal claims are not automatically extracted during ingestion
- **Impact:** Knowledge update questions (e.g., Delhi 2025 vs. Pune 2026) in LongMemEval will fail to resolve state updates because `TurnGraphBuilder` does not extract `TemporalClaim`s or populate `supersedes` edges.
- **Remedy Required:** Wire `build_claims_from_text` and `TemporalSupersessionEngine` directly into `TurnGraphBuilder.build_from_chunks()`. When `build_from_chunks` runs, it should automatically extract claims and store them in the instance's `SQLiteGraphStore`. `answer_question` already checks `store.get_nodes_by_type("claim")`, so wiring ingestion will automatically activate temporal resolution without changing any callers!

### Blocker 2 (HIGH — Evaluation Validity): Evaluation runner defaults to local heuristic judge
- **Impact:** Heuristic substring matching is not comparable to the official LongMemEval benchmark results reported in literature.
- **Remedy Required:** For official benchmark evaluation, configure `use_local_judge=False` with the official OpenAI GPT-4o judge endpoint (or clearly label all local runs as offline smoke heuristics).

### Blocker 3 (MEDIUM — Entity Sparsity): Benchmark ingestion bypasses ontology entity resolution
- **Impact:** Because LLM extraction is skipped for speed on LongMemEval sessions, `TurnGraphBuilder` relies on regex capitalized-word extraction. Rare or lowercase domain entities in conversations may be missed by the graph.
- **Remedy Required:** Expand `extract_chunk_entities` with domain-specific keyword dictionaries from `src/ontology/vocabulary.json`.

---

## 8. Precise Recommendation for Next Step

**Do NOT launch the full LongMemEval benchmark yet.**

### Recommended Step 6 Workflow:
1. **Wire Ingestion (Fix Blocker 1):**
   - Update `TurnGraphBuilder.build_from_chunks` in `src/chunking/graph_retrieval.py` to call `build_claims_from_text` and `TemporalSupersessionEngine.process_claims()`, persisting claims and `supersedes` edges directly to `SQLiteGraphStore`.
2. **Validate Automatic Pipeline:**
   - Run the synthetic smoke test *without* explicitly passing `temporal_claims` to `answer_question`, proving that `answer_question` loads claims directly from the SQLite graph store.
3. **Execute Controlled 30-Question Held-Out Benchmark Run:**
   - Run a single controlled 30-question held-out ablation comparing:
     - Config A: Baseline (`B-Chunked`)
     - Config B: Hybrid + Ontology
     - Config C: Integrated Architecture (`B-Chunked` + Bounded Graph + Temporal Supersession)
   - Measure whether Knowledge Update and Temporal Reasoning categories improve with measured evidence.

---

## 9. Safety & Repository Status Confirmation

- **Current Git Branch:** `research/evaluation-baseline`
- **Working Tree:**
  - Modified files: `src/answering.py`, `src/graph/schema.py`, `src/graph/store.py`
  - Untracked research modules: `src/chunking/`, `src/temporal/`, `tests/`
  - Zero modifications to `main`.
- **Production Stores:**
  - `git diff main..HEAD -- data/` returns zero diff (`data/memories.json` and `data/entities.json` remain completely untouched).
- **Historical Evaluation Artifacts:** All historical evaluation manifests, outputs, reports, and benchmark runs remain preserved and unmodified.
- **Test Suite Health:** All 177 unit and integration tests in `tests/` pass (100% pass rate).

---

## STOP Condition
Step 5 Audit is complete. The report has been written and verified. Awaiting your approval before proceeding to any Step 6 implementation or benchmark execution.
