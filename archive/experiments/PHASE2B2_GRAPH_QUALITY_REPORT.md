# IVARA — Phase 2B.2: Temporal Event Graph Quality Audit Report

**Date:** 2026-11-17  
**Author:** Antigravity (DeepMind Team)  
**Branch:** `experiment/temporal-kg-phase2b`  
**Baseline Commit:** `55a273bff20fa0eec1b2d361e8a03b217d147abe`  
**Scope:** Deep Forensic Audit of Event Graph Schema, Edge Densities, Redundancy, Consistency, and Provenance across 20 Representative Multi-Session Episodes  
**Status:** Audit Completed & Certified  

---

## 1. Executive Summary

During the Phase 2B.1 benchmark run, an audit of 20 representative multi-session instances revealed that the prototype constructed **1,092 EventNodes** and **15,774 temporal edges** ($\mu = 54.6$ events and $\mu = 788.7$ edges per instance, with a mean of **14.45 edges per event**). 

This investigation was conducted to determine:
1. Whether this high edge density is justified by underlying textual evidence.
2. Whether edge duplication, inverse symmetries, or transitive explosion are present.
3. Whether contradictory temporal relationships or temporal paradoxes (cycles) exist.
4. Whether the event graph preserves provenance and uncertainty without date fabrication.

### Key Audit Findings
- **Dual Inverse Edge Redundancy (100% of Edges):** Exactly 7,887 forward (`OCCURRED_BEFORE`) and 7,887 backward (`OCCURRED_AFTER`) edges were constructed. Every single temporal relationship was stored as a dual inverse pair ($E_1 \to E_2$ and $E_2 \to E_1$), doubling edge storage and traversal overhead with zero informational gain.
- **$\mathcal{O}(N^2)$ Pairwise Tournament Graph Explosion:** The relationship builder evaluated all pairs of dated events. For an instance with $K$ dated events, this constructed $\binom{K}{2} \times 2 = K(K-1)$ edges. In instances with 35–40 dated events, this generated over 1,100–1,500 edges, creating dense cliques over dated events where 30.4% of events have degrees $> 50$ (max degree: 86).
- **Zero Logical Contradictions & Zero Cycles:** Across all 15,774 edges, there were **0 contradictory pairs** (no case where $A \text{ BEFORE } B$ and $B \text{ BEFORE } A$ co-occurred) and **0 invalid date intervals** ($valid\_from \le valid\_to$ in 100% of cases).
- **Exact Provenance & Uncertainty Preservation:** 100.0% of events (1,092 / 1,092) retained verbatim provenance text and source session IDs. 45.1% of events were classified as `UNKNOWN` precision and strictly isolated (degree 0), confirming that the system never fabricates dates or infers order from ingestion sequence.

---

## 2. Graph Volume and Density Metrics

From the 20-sample detailed audit:

| Metric | Measured Value | Analysis |
|---|:---:|---|
| **Total Events Extracted** | 1,092 | Mean $54.6$ events per 50-session instance. |
| **Total Temporal Edges Built** | 15,774 | Mean $788.7$ edges per instance. |
| **Mean Edges per Event** | 14.45 | High density driven by all-pairs comparison. |
| **Mean Node Degree** | 28.9 | Average edges incident per node. |
| **Median Node Degree** | 34.0 | Skewed by dense dated event cliques. |
| **Max Node Degree** | 86.0 | Observed on early dated events compared to all later events. |
| **Isolated Events (Degree 0)** | 493 (45.1%) | Undated events preserved without hallucinated edges. |
| **High-Degree Events (Deg > 50)** | 332 (30.4%) | Dated events participating in all-pairs tournament DAGs. |

---

## 3. Redundancy & Dual Symmetry Analysis

### 3.1 Inverse Pair Symmetry
```
             OCCURRED_BEFORE (Forward)
    [Event A] ────────────────────────► [Event B]
              ◄────────────────────────
             OCCURRED_AFTER  (Backward)
```
- **Exact Duplicate Edges:** **0 (0.0%)**. No identical duplicate tuples $(u, v, rel)$ were inserted.
- **Inverse Symmetric Pairs:** **7,887 pairs (100.0% of all edges)**.
  - Every pair of dated events $(E_1, E_2)$ where $E_1.\text{valid\_to} < E_2.\text{valid\_from}$ produced both:
    1. $E_1 \xrightarrow{\text{OCCURRED\_BEFORE}} E_2$
    2. $E_2 \xrightarrow{\text{OCCURRED\_AFTER}} E_1$
  - In SQLite and property graphs, storing both directions creates $2\times$ index overhead. A directed graph needs only canonical forward edges (`OCCURRED_BEFORE`), as backward queries can simply query incoming edges (`get_incoming_edges`).

---

## 4. The $\mathcal{O}(N^2)$ Pairwise Comparison Problem

### 4.1 Structural Cause
In `src/temporal/relationship_builder.py`:
```python
for i in range(n):
    for j in range(i + 1, n):
        # Compares every event against every other event
```
When an episodic conversation collection has $N_{\text{dated}}$ dated events:
$$\text{Total Edges} = 2 \times \frac{N_{\text{dated}}(N_{\text{dated}} - 1)}{2} = N_{\text{dated}}^2 - N_{\text{dated}}$$

For instance `gpt4_f49edff3`:
- 59 total events, of which 34 were dated.
- $34 \times 33 = 1,122$ temporal edges!
- An event from January 2021 was connected to an event in February 2021, March 2021, April 2021, ..., December 2023.

### 4.2 Is This Edge Density Justified?
**No.** Storing transitive edges ($A < B$, $B < C \implies A < C$) across every pair of chronological events is unnecessary:
1. **Transitivity:** A timeline of 5 events $E_1 < E_2 < E_3 < E_4 < E_5$ requires only 4 consecutive edges ($E_1 \to E_2 \to E_3 \to E_4 \to E_5$) to represent the complete total ordering.
2. The current prototype generates $\binom{5}{2} \times 2 = 20$ edges instead of 4 edges—a **$5\times$ edge inflation**.
3. For retrieval, this dense connectivity creates search noise: when traversing from $E_1$, the retriever sees direct 1-hop links to dozens of unrelated events rather than the immediate successor or predecessor.

---

## 5. Relationship Type & Derivation Distribution

| Relationship Type | Count | % of Total | Derivation Method |
|---|:---:|:---:|---|
| `OCCURRED_BEFORE` | 7,887 | 50.0% | `grounded_interval_comparison` (100%) |
| `OCCURRED_AFTER` | 7,887 | 50.0% | `grounded_interval_comparison` (100%) |
| `SUPERSEDES` | 0 | 0.0% | `state_supersession_rule` (0%) |
| `INVALIDATES` | 0 | 0.0% | `invalidation_rule` (0%) |

### 5.1 Why Were `SUPERSEDES` and `INVALIDATES` Zero in the 20-Sample Audit?
- `SUPERSEDES` requires identical state predicates (`lived in`, `worked at`) with different locations/companies across sessions.
- `INVALIDATES` requires an explicit cancellation/termination event matching a prior subscription.
- While these patterns occurred in specific target questions (such as `gpt4_1916e0ea` FarmFresh cancellation, which succeeded in Phase 2B.1), they are rare across random conversational slices, appearing in $< 2\%$ of conversational turns.
- In contrast, date interval comparisons occurred in almost every session containing activities.

---

## 6. Logical Consistency, Contradictions, and Cycle Analysis

An automated consistency scan was executed across all 15,774 edges:
1. **Direct Contradictions ($A \text{ BEFORE } B \land B \text{ BEFORE } A$):** **0 instances**.
2. **Temporal Cycles (Length 2 or 3):** **0 instances**. The temporal subgraph is a strict Directed Acyclic Graph (DAG).
3. **Invalid Date Spans ($valid\_from > valid\_to$):** **0 instances**.
4. **Confidence Calibration:**
   - Exact Day pairs: confidence = 1.0 (mean: 1.00).
   - Month-precision pairs: confidence = 0.92.
   - Year-precision pairs: confidence = 0.85.
   - Minimum observed confidence: 0.72 (combining extraction uncertainty with year-level precision).

---

## 7. Uncertainty & Provenance Audit

### 7.1 Temporal Precision Breakdown
```
  EXACT_DAY  [███████                     ] 16.2% (177 events)
  RELATIVE   [██                          ]  3.5% (38 events)
  MONTH      [███████████                 ] 26.3% (287 events)
  YEAR       [████                        ]  8.9% (97 events)
  UNKNOWN    [████████████████████        ] 45.1% (493 events)
```
- **Uncertainty Preservation:** 45.1% of events were appropriately marked `UNKNOWN`. None of these events had hallucinated dates, and all remained isolated (degree 0).
- **Decoupling Verification:** 599 events had explicit `valid_from` dates that were completely distinct from the session's `recorded_at` ingestion date.

---

## 8. Recommendations for Graph Architecture Optimization

1. **Eliminate Inverse Dual Edges:** Remove `OCCURRED_AFTER` from edge generation. Use `OCCURRED_BEFORE` exclusively, saving 50% storage and graph traversal cost.
2. **Transition from Complete Bipartite to Sequential Interval Ordering:**
   - Instead of connecting every event $E_i$ to every future event $E_j$, sort dated events by `valid_from` and connect only consecutive events ($E_i \xrightarrow{\text{NEXT\_EVENT}} E_{i+1}$) or events sharing an entity.
   - This reduces edge complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$, cutting edges from ~800/instance to ~50/instance while preserving topological order.
3. **Entity-Restricted Temporal Edges:**
   - Restrict temporal edges between events that either:
     a) Share an entity (e.g. two visits to Paris, or switching jobs between companies), or
     b) Fall within the same thematic activity slot (travel, purchases, sports).

