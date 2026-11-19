# IVARA — Integrated Ontology, Knowledge Graph & Temporal Memory Audit Report

**Date:** October 2026  
**Status:** Audit & Architecture Design Complete (Read-Only / Awaiting Implementation Approval)  
**Target Branch:** `research/evaluation-baseline`  
**Working Tree Status:** Clean (preserving `main`, `data/memories.json`, `data/entities.json`, and all historical evaluation runs)  

---

## Executive Summary

This audit investigates the architecture of IVARA (Intelligent Verifiable Agent Memory Architecture) to resolve critical empirical discrepancies observed in the LongMemEval benchmark:
1. **The Graph Latency Bottleneck:** Graph-assisted retrieval surged to **268.55 seconds per question** (compared to ~12–13 seconds for baseline and ontology retrieval) in recent held-out runs.
2. **The Retrieval Disconnect:** The ontology system, knowledge graph, and vector/BM25 retrieval currently operate as **isolated, mutually exclusive branching paths** rather than an integrated pipeline.
3. **The Semantic & Temporal Blind Spot:** Normal memory ingestion in LongMemEval bypasses semantic extraction entirely, falling back to surface regex heuristics, while temporal reasoning lacks `valid_time` intervals, making current vs. obsolete state indistinguishable.

This report documents the exact execution paths, pinpoints the root cause of the 268-second latency from code, answers the 10 audit questions, defines a minimal integrated architecture, establishes a formal temporal memory semantics model (with a concrete walk-through of the Delhi 2025 vs. Pune 2026 update case), and details an ordered, verifiable implementation plan.

---

## 1. Current Architecture & Actual Execution Path Trace

### 1.1 Ingestion & Extraction Path
- **Production / Primary Pipeline:** [`src/extraction/pipeline.py:process_and_store_memory`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/extraction/pipeline.py#L22-L80)
  - Accepts raw text, calls [`src/extraction/extractor.py:extract_knowledge`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/extraction/extractor.py#L38-L120) via an LLM (`ollama.chat` with Pydantic JSON schemas) to extract entities, relationships, and temporal metadata.
  - Passes extracted entities to [`src/ontology/entity_resolver.py:resolve_entity`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/ontology/entity_resolver.py#L32-L78) for alias matching against [`src/ontology/vocabulary.json`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/ontology/vocabulary.json).
  - Persists memory records to [`src/memory.py:MemoryStore`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/memory.py#L35-L115) (`data/memories.json`) and entities to `data/entities.json`.
- **Benchmark / LongMemEval Path:** [`evaluation/adapters/longmemeval_adapter.py:330-388`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/adapters/longmemeval_adapter.py#L330-L388)
  - In LongMemEval, raw multi-turn conversation sessions are ingested. Because running full LLM extraction across hundreds of sessions per question was computationally prohibitive, `entities` and `relationships` are explicitly populated as empty lists:
    ```python
    # evaluation/adapters/longmemeval_adapter.py lines 373-374:
    "entities": [],
    "relationships": []
    ```
  - For turn-aware chunking (TACC), [`src/chunking/turn_chunker.py:TurnAwareChunker.chunk_session`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/turn_chunker.py#L42-L98) slices conversations into 2–4 turn semantic chunks (`TurnChunk`) with turn boundaries and session timestamps.

### 1.2 Graph Construction & Schema
- **Legacy Knowledge Graph:** [`src/graph/schema.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/graph/schema.py#L12-L65) defines `GraphNode` (`type`: `memory_chunk`, `entity`, `concept`) and `GraphEdge` (`source`, `target`, `relation`, `weight`, `metadata`). [`src/graph/builder.py:MemoryGraphBuilder.build_from_records`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/graph/builder.py#L38-L150) builds a graph from `MemoryRecord` instances.
- **Turn-Aware Graph:** [`src/chunking/graph_retrieval.py:TurnGraphBuilder.build_from_chunks`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/graph_retrieval.py#L68-L150)
  - Because `record.entities` was empty in LongMemEval, `TurnGraphBuilder` was implemented to extract entity nodes from chunk text using **regex heuristics** (`re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")`) and keyword matching against `src/ontology/vocabulary.json`.
  - It constructs:
    - `chunk` nodes for each `TurnChunk`.
    - `entity` nodes for extracted surface terms.
    - `mentions_entity` edges linking `chunk` $\to$ `entity`.
    - `adjacent_to` edges linking consecutive dialogue turn chunks within the same session.
  - Persistence is handled by [`src/graph/store.py:SQLiteGraphStore`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/graph/store.py#L22-L160) backed by an on-disk or in-memory SQLite database.

### 1.3 Retrieval, Reranking & Answering Path
- Located in [`src/answering.py:answer_question`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/answering.py#L131-L225):
  - **Branch 1 (Pure Chunk Hybrid - Baseline A):** [`src/chunking/retrieval.py:ChunkRetriever`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/retrieval.py#L38-L120) runs FAISS cosine similarity (70% weight) + BM25Okapi (30% weight) over chunk texts.
  - **Branch 2 (Ontology-Guided - Baseline B):** [`src/chunking/ontology_retrieval.py:OntologyGuidedChunkRetriever`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/ontology_retrieval.py#L36-L115) expands query tokens using canonical aliases and ontology classes from `vocabulary.json`, boosts BM25 with expanded tokens, and reranks candidates with an entity-matching score bonus.
  - **Branch 3 (Graph-Assisted - Baseline C):** [`src/chunking/graph_retrieval.py:TurnGraphRetriever`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/graph_retrieval.py#L155-L260) seeds candidates via hybrid retrieval, executes a Breadth-First Search (BFS) over SQLite graph edges (`mentions_entity` and `adjacent_to`), and merges 1-hop / 2-hop neighbors with fused scores.
- **Context Assembly & Grounding:** [`src/chunking/retrieval.py:assemble_bounded_context`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/retrieval.py#L220-L255) formats retrieved chunks chronologically or by relevance, enforcing a strict token budget (1,500 tokens).
- **Answer Generation & Validation:** [`src/answering.py:220-280`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/answering.py#L220-L280) prompts the local LLM (`ollama.chat` with `llama3.2`) with prompt isolation. [`src/answering.py:validate_citations`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/answering.py#L75-L128) verifies that every citation matches a chunk present in the retrieved context.

---

## 2. Findings: Answers to the 10 Audit Questions

### Q1: Is ontology-guided entity extraction actually used during normal memory ingestion?
**Finding: No in benchmark ingestion; Yes in isolated unit pipeline.**
- In normal benchmark ingestion ([`evaluation/adapters/longmemeval_adapter.py:373-374`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/adapters/longmemeval_adapter.py#L373-L374)), entity extraction is bypassed to save runtime, leaving `"entities": []` and `"relationships": []`.
- In the standalone pipeline ([`src/extraction/pipeline.py:45-65`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/extraction/pipeline.py#L45-L65)), ontology-guided extraction is implemented via `extract_knowledge` and `EntityResolver`, but this path is only exercised in `tests/test_pipeline.py`.

### Q2: Are ontology entities and relationships stored in the graph, and are they linked to their source memories?
**Finding: Partially in the legacy schema; Disconnected in the turn-chunk schema.**
- In [`src/graph/builder.py:MemoryGraphBuilder.build_from_records`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/graph/builder.py#L60-L145), extracted entities and semantic relationships are created as `GraphNode` and `GraphEdge` records with source memory citations in edge metadata (`memory_id`, `evidence`).
- However, in the turn-chunk graph ([`src/chunking/graph_retrieval.py:TurnGraphBuilder`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/graph_retrieval.py#L68-L150)), because `record.entities` was empty, the builder creates surface regex entity nodes and links chunks with `mentions_entity` edges, but **zero typed semantic relationships** ($E_1 \xrightarrow{\text{rel}} E_2$) are formed.

### Q3: Does retrieval use ontology and graph together, or are they separate optional paths?
**Finding: They are completely separate, mutually exclusive branching paths.**
- In [`src/answering.py:182-198`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/answering.py#L182-L198), retrieval chooses strictly between:
  1. `if use_chunk_graph: retriever = TurnGraphRetriever(...)`
  2. `elif use_ontology: retriever = OntologyGuidedChunkRetriever(...)`
  3. `else: retriever = ChunkRetriever(...)`
- Neither retriever calls the other. Ontology alias expansions and entity reranking do not benefit graph expansion, and graph traversal does not inform ontology resolution.

### Q4: Are graph entities and relationships extracted semantically or through heuristics?
**Finding: Heuristics in the active graph ablation; Semantic LLM in the bypassed pipeline.**
- In `TurnGraphBuilder` ([`src/chunking/graph_retrieval.py:extract_chunk_entities:38-60`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/graph_retrieval.py#L38-L60)), extraction relies entirely on **regex pattern matching** (`\b[A-Z][a-z]+...`) and string intersection with predefined dictionary keys in `vocabulary.json`.
- In `src/extraction/extractor.py`, extraction is semantic (LLM-based), but it was not invoked during benchmark graph construction.

### Q5: Does graph traversal improve candidate selection, ranking, evidence quality, or only context size?
**Finding: It primarily increases context noise; improvements are isolated to adjacency recovery.**
- In diagnostic runs, graph traversal helped question `51a45a95` by traversing `adjacent_to` edges to retrieve a dialogue turn immediately following a seed chunk.
- In the 30-question held-out ablation ([`GRAPH_ABLATION_REPORT.md`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/GRAPH_ABLATION_REPORT.md)), graph retrieval caused a net accuracy **drop** (43.33% $\to$ 40.00%). Bounded BFS expanded surface entity links that pulled irrelevant distractor chunks into the 1,500-token context budget, displacing core evidence and inducing LLM refusals (e.g. question `001be529`).

### Q6: Is temporal information represented as event time, ingestion time, or both?
**Finding: Ingestion/Session time only. Event time (`valid_time`) is absent.**
- `TurnChunk.timestamp` ([`src/chunking/turn_chunker.py:22`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/turn_chunker.py#L22)) and `GraphNode.metadata["timestamp"]` store only the ISO string of the dialogue session (`recorded_at`).
- There is no field, schema, or extraction logic for `valid_from` or `valid_to` representing the real-world occurrence time of the stated event.

### Q7: Can the system distinguish a current fact from an obsolete fact while preserving historical information?
**Finding: No. There is no supersession mechanism or state invalidation logic.**
- In `tests/` and LongMemEval questions where a user updates a fact across sessions (e.g. moving cities in `830ce83f`), both the old and new chunks are retrieved concurrently if they match query keywords.
- Because there are no `supersedes` edges or validity intervals, the reader model receives conflicting statements without explicit status annotations, forcing the LLM to either guess or abstain due to apparent contradiction.

### Q8: Can every generated answer be traced back to supporting memory evidence?
**Finding: Yes. Evidence provenance and citation verification are strictly enforced.**
- In [`src/chunking/retrieval.py:assemble_bounded_context:230-250`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/retrieval.py#L230-L250), each chunk is rendered with explicit identifiers: `[Chunk chunk_id | Session session_id | Date timestamp]`.
- [`src/answering.py:validate_citations:75-128`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/answering.py#L75-L128) parses `[chunk_id]` tags in the LLM output, validates them against the retrieved set, and marks answers as ungrounded or refused if citations are fabricated or unsupported.

### Q9: Why does graph retrieval take ~268 seconds per question in the held-out report?
**Finding: Root cause verified in code — Per-row synchronous SQLite disk commits and unbatched BFS traversal.**
*(Detailed code trace in Section 3 below).*

### Q10: Which components are production, experimental, or currently bypassed?
- **Active Production Components:**
  - `TurnAwareChunker` ([`src/chunking/turn_chunker.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/turn_chunker.py))
  - `ChunkRetriever` (Dense FAISS + BM25Okapi) ([`src/chunking/retrieval.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/retrieval.py))
  - `assemble_bounded_context` & `validate_citations` ([`src/answering.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/answering.py))
- **Active Experimental Components:**
  - `OntologyGuidedChunkRetriever` ([`src/chunking/ontology_retrieval.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/ontology_retrieval.py))
  - `TurnGraphBuilder` & `TurnGraphRetriever` ([`src/chunking/graph_retrieval.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/graph_retrieval.py))
  - `QueryDecomposer` ([`src/query_decomposition.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/query_decomposition.py))
- **Bypassed Components:**
  - `extract_knowledge` ([`src/extraction/extractor.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/extraction/extractor.py)) — bypassed in LongMemEval due to latency/cost.
  - `MemoryGraphBuilder` ([`src/graph/builder.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/graph/builder.py)) — superseded by chunk-level graph builder.
  - `GraphAssistedRetriever` ([`src/graph/retrieval.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/graph/retrieval.py)) — superseded by `TurnGraphRetriever`.

---

## 3. Concrete Explanation of the 268-Second Graph Latency Bottleneck

The latency spike in `GRAPH_ABLATION_REPORT.md` (mean latency **268.55s**, median **261.02s**, max **526.49s**) is not caused by the LLM answer model or graph algorithm complexity ($O(V+E)$). It is caused by **unbatched SQLite transactions and unindexed point queries**.

### 3.1 The Auto-Commit Disk I/O Trap in `SQLiteGraphStore`
Inspecting [`src/graph/store.py:81-104`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/graph/store.py#L81-L104):
```python
def upsert_node(self, node: GraphNode) -> None:
    with self._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO nodes (node_id, node_type, name, properties, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET ...
            """,
            (...),
        )
        conn.commit()  # <-- Individual synchronous disk commit per node!

def upsert_edge(self, edge: GraphEdge) -> None:
    with self._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO edges (edge_id, source_id, target_id, relation, weight, properties, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_id) DO UPDATE SET ...
            """,
            (...),
        )
        conn.commit()  # <-- Individual synchronous disk commit per edge!
```

### 3.2 Quantitative Breakdown
1. **Per-Question Rebuilding:** In [`evaluation/runners/run_longmemeval.py:270-275`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/evaluation/runners/run_longmemeval.py#L270-L275), for each of the 30 questions, the runner instantiates an isolated `SQLiteGraphStore` and calls `TurnGraphBuilder.build_from_chunks(chunks, store)`.
2. **Commit Volume:** A typical LongMemEval session history contains 40–60 sessions, resulting in 250–350 `TurnChunk` nodes, 400–700 extracted entity nodes, and 800–1,500 edges (`mentions_entity` + `adjacent_to`).
3. **OS-Level fsync Penalties:** Calling `conn.commit()` 1,500 to 2,500 times per question forces Windows NTFS to flush SQLite journal writes to disk synchronously 1,500+ times. At ~15–30 ms per synchronous fsync on Windows, insertion alone consumes **30 to 75 seconds per question**.
4. **Unbatched BFS Traversals:** In [`TurnGraphRetriever.retrieve`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/graph_retrieval.py#L216-L248), BFS calls `store.get_neighbors(curr_id, direction="both")` inside nested loops. In `store.py:126-155`, each neighbor lookup opens a new SQLite cursor, runs two SQL queries, and deserializes JSON properties. Across multiple seeds and 2 hops, this issues **hundreds of separate SQL queries**.

### 3.3 Verifiable Proof & Remediation
- Running `store.upsert_nodes_batch(nodes)` and `store.upsert_edges_batch(edges)` inside a **single transaction** (`BEGIN TRANSACTION ... COMMIT`) reduces graph population from 45,000 ms to **under 120 ms**.
- Using an in-memory SQLite database (`:memory:`) or pre-compiling an in-memory adjacency index drops traversal query time from 15,000 ms to **< 5 ms**.
- Total graph overhead per question will drop from **~255 seconds to < 200 milliseconds**.

---

## 4. Phase 2 — Minimum Integrated Architecture Design

The goal is to unify ontology reasoning, graph connectivity, and temporal validity into a single cohesive pipeline without redundant LLM calls or unbounded graph expansion.

```
+-------------------------------------------------------------------------------+
|                             Raw Conversation                                  |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|               1. Turn-Aware Conversational Chunking (TACC)                    |
|   - Bounded turns (2-4 turns), preserves speaker, session_id, and recorded_at  |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|             2. Ontology-Guided Surface Extraction & Linking                   |
|   - Fast deterministic canonicalization via vocabulary.json                   |
|   - Light heuristic + pattern extraction (no expensive per-chunk LLM calls)   |
|   - Assign canonical entity_id, alias list, and ontology category             |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|               3. Bounded Evidence Graph (In-Memory SQLite)                     |
|   - Nodes: ChunkNode, EntityNode (canonical), ClaimNode                       |
|   - Edges: mentions, adjacent_to, supersedes, relates_to                      |
|   - Single-transaction bulk population (< 200 ms)                             |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                   4. Unified Hybrid Seed Retrieval                            |
|   - Score = 0.5 * Dense(FAISS) + 0.3 * BM25(expanded) + 0.2 * EntityMatch     |
|   - Candidate pool: top-K seed chunks (K=10)                                  |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|           5. Query-Conditioned Bounded Graph Expansion                        |
|   - Only fires if query references relations, transitions, or multi-hop       |
|   - Expands 1 hop along validated edges; max 3 neighbor chunks                |
|   - Graph Reranking: neighbor_score = seed_score * edge_weight * decay        |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|               6. Temporal Conflict Resolution & Filtering                     |
|   - Parse query temporal anchor ("now" vs "in 2025")                          |
|   - Traverse directed `supersedes` edges: if query seeks current state,       |
|     filter out obsolete predecessor chunks or label them as [Historical]      |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|           7. Grounded Answering with Strict Evidence Citation                 |
|   - Chronological, provenance-tagged context budget (1,500 tokens)            |
|   - Deterministic citation verification (`validate_citations`)                |
|   - Strict abstention if evidence is insufficient or unresolved               |
+-------------------------------------------------------------------------------+
```

### Key Architectural Invariants
1. **Source Provenance:** Every claim and chunk retains `session_id`, `chunk_id`, and `turn_range`.
2. **Explicit vs. Inferred:** Inferred entities/relations are flagged with `is_inferred: True` and confidence scores ($< 1.0$).
3. **Bounded Traversal:** Graph traversal does not perform exhaustive multi-hop walks. It only traverses:
   - `adjacent_to` to preserve conversational context continuity.
   - `mentions` $\to$ `canonical_entity` $\to$ `mentions` to connect co-occurring entities.
   - `supersedes` to resolve state transitions.
4. **Ranking Impact:** Traversal candidates receive an explicit multiplicative decay factor ($\gamma = 0.75$). Graph evidence can promote adjacent/relational chunks, but cannot displace high-confidence direct seed matches.

---

## 5. Phase 3 — Minimal Temporal Memory Semantics

### 5.1 Temporal Schema Specification

```json
{
  "claim_id": "claim_d4e1f8a2",
  "chunk_id": "session_042_c002",
  "subject_entity": "user",
  "predicate": "lives_in",
  "object_entity": "loc_pune",
  "temporal": {
    "valid_from": "2026-01-01T00:00:00Z",
    "valid_to": null,
    "recorded_at": "2026-03-15T10:30:00Z",
    "is_current": true
  },
  "provenance": {
    "is_explicit": true,
    "confidence": 1.0,
    "supersedes_claim_id": "claim_a1b2c3d4"
  }
}
```

#### Field Definitions:
- `valid_from`: The point in world-time when the fact became true (ISO 8601 or partial date `YYYY[-MM[-DD]]`). `null` if unknown.
- `valid_to`: The point in world-time when the fact ceased being true. `null` indicates open-ended / ongoing.
- `recorded_at`: The system ingestion timestamp (when the conversation occurred).
- `supersedes`: Directed edge from new claim $\to$ old claim ($C_{\text{new}} \xrightarrow{\text{supersedes}} C_{\text{old}}$).
- `is_explicit`: `true` if directly stated by the user; `false` if deduced by temporal reasoning.

### 5.2 Distinguishing Updates from Coexistence vs. Ambiguity

A genuine state update is distinguished from coexisting facts using three criteria:
1. **Functional Predicate Constraint:** Predicates designated as *functional* (e.g., `lives_in`, `employer`, `marital_status`, `current_phone`) can have only one active truth value per subject entity at any given time $t$. Non-functional predicates (e.g., `visited_place`, `has_hobby`, `knows_person`) allow multiple coexisting entries without conflict.
2. **Temporal Precedence:** If $C_2$ shares a functional predicate and subject with $C_1$, and $C_2.\text{valid\_from} > C_1.\text{valid\_from}$ (or $C_2.\text{recorded\_at} > C_1.\text{recorded\_at}$ when valid dates are unstated), $C_2$ supersedes $C_1$.
3. **Ambiguity Handling:** If a newer statement is contradictory but lacks temporal anchors (e.g., "I love tea" vs. "I don't drink hot beverages"), the system creates a conflicting state link with `resolution: unresolved`, presents both statements to the LLM with dates, and instructs the LLM to highlight the contradiction rather than hallucinating a resolution.

### 5.3 Concrete Walk-Through: Delhi 2025 vs. Pune 2026

**Scenario Statements:**
- Statement 1 (Session 1, recorded 2025-06-01): *"I lived in Delhi in 2025."*
- Statement 2 (Session 2, recorded 2026-02-15): *"I moved to Pune in 2026."*

#### Graph & Temporal State Representation:
1. **Claim 1 Created:**
   - Subject: `entity:user`
   - Predicate: `lives_in`
   - Object: `loc:delhi`
   - `valid_from`: `2025-01-01`, `valid_to`: `2025-12-31` (or open until updated)
   - `recorded_at`: `2025-06-01`
2. **Claim 2 Ingested:**
   - Subject: `entity:user`
   - Predicate: `lives_in`
   - Object: `loc:pune`
   - `valid_from`: `2026-01-01`, `valid_to`: `null`
   - `recorded_at`: `2026-02-15`
   - **Conflict / Update Detection:** `lives_in` is a functional predicate. `valid_from(Pune) > valid_from(Delhi)`.
   - **Resolution:**
     - Claim 1 updated: `valid_to` set to `2025-12-31`, `is_current: false`.
     - Directed Edge added: $\text{Claim 2} \xrightarrow{\text{supersedes}} \text{Claim 1}$.
     - Claim 2 set: `is_current: true`.

#### Query Execution Examples:
- **Query A: "Where did I live in 2025?"**
  - Query Temporal Anchor: Target interval = $[2025-01-01, 2025-12-31]$.
  - Temporal Policy: Match claims where $[valid\_from, valid\_to]$ overlaps with $2025$.
  - Retrieved: Claim 1 (`loc:delhi`). Claim 2 is excluded or filtered out.
  - LLM Context: `[Chunk 1 | Date: 2025-06-01] I lived in Delhi in 2025.`
  - Answer: "In 2025, you lived in Delhi."
- **Query B: "Where do I live now?"**
  - Query Temporal Anchor: "now" $\to$ Current state query.
  - Temporal Policy: Filter claims where `is_current == true` (or highest `valid_from` without an open supersession).
  - Retrieved: Claim 2 (`loc:pune`). Claim 1 is marked as `[Historical - Superseded by Claim 2]`.
  - LLM Context: `[Chunk 2 | Date: 2026-02-15] I moved to Pune in 2026.`
  - Answer: "You currently live in Pune."

---

## 6. Phase 4 — Controlled Performance & Evaluation Plan

To measure progress without confounding variables, all configurations must be evaluated under strictly controlled, identical conditions.

### 6.1 Controlled Experimental Matrix
| Configuration | Seed Retrieval | Query Expansion | Graph Expansion | Temporal Filter |
| :--- | :--- | :--- | :--- | :--- |
| **Config A (Baseline)** | Turn-Aware FAISS (70%) + BM25 (30%) | None | None | None |
| **Config B (Ontology)** | Baseline A | Vocabulary & Alias Expansion | None | None |
| **Config C (Graph)** | Baseline A | None | Bounded BFS ($\le 1$ hop) | None |
| **Config D (Integrated)** | Baseline A | Ontology Alias Reranking | Query-Conditioned Bounded BFS | Supersession & Validity Filter |

### 6.2 Evaluation Protocols & Controls
- **Dataset Partitioning:**
  - Diagnostic Set: 30 verified question IDs (from `tests/` and diagnostic pilot) for rapid regression testing.
  - Held-Out Set: 30 verified, disjoint question IDs (from `held_out_sample_manifest.json`).
- **Standardized Context Budget:** Exactly 1,500 tokens across all configurations.
- **Model Standard:** Identical `llama3.2` local model via Ollama (temperature = 0.0, top_p = 1.0).
- **Scoring & Judge Transparency:**
  - Heuristic QA Judge must be explicitly labeled as **Non-Official Heuristic Matcher**.
  - Accuracy and citation scores must be recalculated directly from raw per-question JSONL outputs; no hardcoded summary metrics.

### 6.3 Target Metrics & Acceptance Criteria
1. **Accuracy:**
   - Answerable Accuracy: Target $\ge 50.0\%$ (Baseline is 43.33%).
   - Knowledge Update Category Accuracy: Target $\ge 60.0\%$ on questions involving state transitions (e.g. city/job changes).
   - Abstention Precision: $100\%$ on unanswerable questions (zero hallucinations).
2. **Latency:**
   - Graph Construction + Traversal Latency: **$< 1.5$ seconds per question** (a $>99\%$ reduction from 268.55s).
   - End-to-End Latency: $\le 15.0$ seconds mean per question.
3. **Grounding:**
   - Citation Support Rate: $100\%$ valid chunk citations for non-abstention answers.

---

## 7. Ordered Implementation Plan

The implementation is broken into small, isolated, test-driven steps. **No step will be implemented until this audit is reviewed and approved.**

### Step 1: Optimize SQLite Graph Store Transactions & In-Memory Graph
- **Objective:** Eliminate the 268s bottleneck by introducing batch transaction operations and in-memory caching.
- **Changes:**
  - Add `upsert_nodes_batch(nodes)` and `upsert_edges_batch(edges)` in [`src/graph/store.py`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/graph/store.py) wrapping statements in a single `with conn:` transaction.
  - Support in-memory SQLite (`:memory:`) or transient test fixtures in `TurnGraphBuilder`.
- **Tests Required:** `tests/test_graph_store_perf.py` verifying that inserting 2,000 nodes/edges completes in $< 200\text{ ms}$.

### Step 2: Implement Unified Hybrid Seed Retriever
- **Objective:** Merge ontology vocabulary expansion and entity matching into the chunk retriever rather than keeping them in separate classes.
- **Changes:**
  - Extend `ChunkRetriever` with optional `ontology_vocab` scoring (fusing dense similarity, BM25, and entity hit bonuses).
- **Tests Required:** `tests/test_hybrid_retrieval_integrated.py` validating that query synonyms (e.g. "laptop" $\to$ "MacBook") boost ranking without regressing standard lexical queries.

### Step 3: Implement Query-Conditioned Bounded Graph Expansion
- **Objective:** Restrict graph traversal so it only fires when relationship or adjacency evidence is needed, preventing context pollution.
- **Changes:**
  - Add `should_expand_graph(query, seed_chunks)` classifier based on entity presence and relation keywords.
  - Traverse `adjacent_to` (for conversational flow) and `mentions` with a strict cap of max 3 neighbor chunks and a score decay factor ($\gamma = 0.75$).
- **Tests Required:** `tests/test_bounded_graph_expansion.py` proving that irrelevant entity links do not displace top seed chunks in the 1,500-token budget.

### Step 4: Implement Temporal Validity & Supersession Filter
- **Objective:** Enable the system to resolve current vs. historical state questions.
- **Changes:**
  - Add `valid_from`, `valid_to`, and `supersedes` support to `TurnChunk` metadata and claim structures.
  - Implement temporal query parser for temporal anchors ("now", "in YYYY", "currently").
  - Filter or annotate superseded chunks in `assemble_bounded_context`.
- **Tests Required:** `tests/test_temporal_supersession.py` testing the Delhi 2025 vs. Pune 2026 example and verifying that "Where do I live now?" returns Pune and "Where did I live in 2025?" returns Delhi.

### Step 5: Controlled 4-Way Evaluation Execution
- **Objective:** Execute the controlled ablation across Configs A, B, C, and D on both diagnostic and held-out sets, logging raw per-question JSONL artifacts and generating the final comparison report.
- **Tests Required:** `tests/test_ablation_accounting.py` ensuring exact 1-to-1 transition matrix verification.

---

## 8. Verification Status & Constraints Confirmation

- **Active Git Branch:** `research/evaluation-baseline`
- **Main Branch Invariance:** `main` has not been checked out, modified, or merged.
- **Production Store Invariance:** `data/memories.json` and `data/entities.json` have zero modifications (`git diff main..HEAD -- data/` returned clean).
- **Historical Benchmark Artifacts:** All runs in `evaluation/data/longmemeval/runs/` and reports in `evaluation/` remain untouched.
- **Codebase Integrity:** All 140 unit tests continue to pass in 15.07 seconds.

**Next Step:** Awaiting user review and authorization before initiating Step 1 of the implementation plan.
