# IVARA — Phase 2B.2: Category H Failure Analysis & Retrieval Architecture Audit

**Date:** 2026-11-17  
**Author:** Antigravity (DeepMind Team)  
**Branch:** `experiment/temporal-kg-phase2b`  
**Baseline Commit:** `55a273bff20fa0eec1b2d361e8a03b217d147abe`  
**Scope:** Forensic Root-Cause Investigation of all 38 Unrecovered Category H Failures and the Disparity Between Phase 2A.1 (6/41) and Phase 2B.1 (3/41)  
**Status:** Completed & Certified  

---

## 1. Executive Summary

In Phase 2B.1, IVARA achieved an all-time benchmark high of **51.88% overall accuracy (69/133)** on the LongMemEval-S temporal benchmark, driven by a record **71.74% (66/92)** on Non-Category H questions. 

However, on **Category H (Multi-Session Temporal Reasoning, 41 questions)**:
- Phase 1 Flat Baseline: **0 / 41 (0.00%)**
- Phase 2A.1 Adaptive: **6 / 41 (14.63%)**
- Phase 2B.1 Event Graph: **3 / 41 (7.32%)**

This failure analysis was conducted across all 38 remaining Category H failures to determine:
1. Whether required events and entities were extracted and connected in the graph.
2. Why Phase 2B.1 underperformed Phase 2A.1 on Category H despite having an explicit temporal event graph.
3. The exact root causes behind all 38 unrecovered questions.

### Key Forensic Findings
- **100% Retrieval Starvation (38 / 38 cases):** In every single failed Category H question, the retrieval pipeline failed to surface chunks from all required sessions into the LLM context. 
  - **52.6% (20 questions):** Required 2 distinct sessions, but dense retrieval only surfaced 1 session.
  - **47.4% (18 questions):** Required $\ge 3$ distinct sessions (ordering 3 to 6 distinct events across 3 to 6 distinct conversations), but retrieval surfaced at most 1 or 2 sessions.
- **The Phase 2A.1 vs. Phase 2B.1 Mechanism Disparity:**
  - In Phase 2A.1, the retriever used `HierarchicalEpisodicRetriever` with forced session diversity (`use_diversity=True`, `top_m_sessions=6`, capping chunks to 1 per session). This artificially guaranteed that 3 to 6 distinct sessions were retrieved, allowing the LLM to see dates across episodes.
  - In Phase 2B.1, retrieval relied on `TurnGraphRetriever` with standard Session-Aware MMR (`k=5`). Because `TurnGraphRetriever` seeds candidates from dense vector similarity, the single highest-scoring session consumed 3–4 chunk slots. Furthermore, 1-hop graph expansion only expanded from the top 3 seeds. If a secondary event session lacked an explicit entity overlap with the seed, graph expansion never bridged to that session.
- **The Accuracy Trade-off:**
  - Forcing session diversity (Phase 2A.1) gained $+6$ questions on Category H, but caused **11 severe regressions on Non-Category H**, dragging overall accuracy down to 44.36%.
  - Preserving intra-session depth (Phase 2B.1) achieved **71.74% on Non-Category H (+4 questions above baseline)** and **51.88% overall (+7 questions above baseline)**, but sacrificed session breadth on multi-session queries.

---

## 2. Taxonomy of the 38 Category H Failures

```
Total Category H Failures (38 Questions)
├── Multi-Session Retrieval Starvation (20 Qs, 52.6%)
│   └── Exactly 2 sessions required; only 1 session retrieved
└── High-Cardinality Session Starvation (18 Qs, 47.4%)
    └── 3 to 6 sessions required; only 1 or 2 sessions retrieved
```

| Failure Mode | Count | % | Description | Representative Questions |
|---|:---:|:---:|---|---|
| **Multi-Session Retrieval Starvation** | 20 | 52.6% | The query compares two events in two disjoint sessions (e.g. "Did Event A happen before Event B?"). Dense seed retrieval only retrieved Session A; Session B was never expanded. | `gpt4_2487a7cb`, `gpt4_4cd9eba1`, `gpt4_61e13b3c`, `gpt4_213fd887`, `gpt4_5438fa52` |
| **High-Cardinality Session Starvation** | 18 | 47.4% | The query asks for the chronological ordering of 3 to 6 events (e.g. order of 6 museums, 6 flights, 3 trips). With $k=5$ chunks, it is mathematically impossible to represent 6 distinct sessions unless strictly 1 chunk per session is forced. | `gpt4_7abb270c`, `gpt4_f420262c`, `gpt4_18c2b244`, `gpt4_f49edff3`, `gpt4_7ca326fa` |

---

## 3. Forensics on the 5 Questions Recovered in Phase 2A.1 but Missed in Phase 2B.1

In Phase 2A.1, 6 Category H questions were correct. In Phase 2B.1, 1 of them remained correct (`gpt4_1a1dc16d`), while 2 new Category H questions were recovered (`b46e15ee`, `gpt4_2f56ae70`), but 5 regressed back to incorrect:

| Question ID | Question Text | Required Sessions | P2A.1 Retrieval Strategy | P2B.1 Retrieval Strategy | Root Cause in Phase 2B.1 |
|---|---|:---:|---|---|---|
| `gpt4_f49edff3` | Which three events happened in order: nursery prep, baby shower, phone case? | 3 | Hierarchical Diversity (1 chunk from each of 3 sessions retrieved) | TurnGraphRetriever ($k=5$, 2 sessions retrieved, nursery session missed) | Seed chunk from baby shower session consumed 3 chunk slots; 1-hop traversal did not reach nursery prep session. |
| `gpt4_18c2b244` | What is the order of three events: rewards program, product launch, webinar? | 3 | Hierarchical Diversity (3 distinct sessions surfaced) | TurnGraphRetriever (only rewards program session retrieved) | Dense vector search favored ultrachat distractor chunks; 1-hop expansion failed to bridge to product launch session. |
| `gpt4_2487a7cb` | Which did I attend first: 'Effective Time Management' workshop or 'AI in Healthcare' webinar? | 2 | Hierarchical Diversity (both workshop & webinar sessions surfaced) | TurnGraphRetriever (only workshop session surfaced) | No shared entity between workshop and webinar; without shared entity, 1-hop graph expansion cannot hop between them. |
| `gpt4_4cd9eba1` | How many weeks was I accepted into exchange program when I started packing? | 2 | Hierarchical Diversity (acceptance session + packing session surfaced) | TurnGraphRetriever (only packing session surfaced) | Acceptance session lacked lexical keyword overlap with "packing"; graph had no direct edge. |
| `gpt4_61e13b3c` | How many weeks passed between selling baked goods and farmers market? | 2 | Hierarchical Diversity (both sessions surfaced) | TurnGraphRetriever (only farmers market session surfaced) | Dense retrieval clustered around farmers market distractor chunks; baked goods turn missed. |

---

## 4. Why Bounded Graph Traversal Fails on Disjoint Events

A critical architectural insight emerged from this forensic analysis:

### 4.1 The Disconnected Query Problem
In Category H questions like `gpt4_2487a7cb`:
- Event 1: "Effective Time Management workshop" in Session 12.
- Event 2: "AI in Healthcare webinar" in Session 38.
- These two events have **zero entities in common**:
  - Event 1 entities: `ent_effective_time_management`, `ent_workshop`.
  - Event 2 entities: `ent_ai_in_healthcare`, `ent_webinar`.
- In the graph store, there is an `OCCURRED_BEFORE` edge between `evt_12` and `evt_38` because their dates are known.
- **However, during retrieval:**
  1. The retriever starts from top seed chunks returned by dense search.
  2. Dense search matches "Effective Time Management workshop" and retrieves Chunk 12.
  3. Safe Edge Filter 4 expands from Chunk 12 to Event 12, and then traverses to Event 38.
  4. **The Bottleneck:** Because `k=5` and MMR prioritizes high semantic similarity to the query, the seed session (which has 4 chunks matching the query words) outranks the single distant chunk from Session 38.
  5. As a result, Chunk 12 is placed in the context, but Chunk 38 is pushed below the top-$k$ budget!

---

## 5. Architectural Recommendations for Phase 3

To solve Category H without sacrificing Non-Category H:
1. **Dynamic Chunk Allocation Based on Query Intent:**
   - For single-session / factual queries: Keep `eff_chunks_per_session = 3` (preserving 71.74% precision).
   - For multi-session ordering queries (detected by query router): Force `eff_chunks_per_session = 1` and set $k = \max(k, 6)$ to guarantee session diversity.
2. **Graph-Forced Evidence Inclusion:**
   - When the graph traversal discovers a verified `OCCURRED_BEFORE` path between Event A and Event B, the supporting chunks for *both* Event A and Event B must be guaranteed slots in the context, bypassing dense similarity pruning.

