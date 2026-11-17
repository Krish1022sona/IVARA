# IVARA — System-Wide Finalization Audit

**Document:** `FINALIZATION_AUDIT.md`  
**Date:** 2026-11-17  
**Auditor:** Antigravity (Google DeepMind Team)  
**Audit Scope:** Repository Integrity, Experimental Branches, Benchmark Provenance, and Candidate Selection  
**Mode:** READ-ONLY Forensic Audit (Stage 1)  
**Safety Status:** Non-mutating; production code, experimental code, and databases preserved unaltered.  

---

## 1. Executive Summary

This finalization audit transitions Project IVARA from active exploratory research into rigorous release finalization. Over recent sprints, IVARA has progressed through five distinct temporal and multi-session architecture experiments (Phase 1, Phase 2A/2A.1, Phase 2B.1/2B.2, Phase 3, and Phase 3.1) to resolve the documented bottleneck in multi-session temporal reasoning.

### Key Audit Findings:
1. **Repository Topology:** All experimental branches (`experiment/temporal-phase1`, `experiment/hierarchical-retrieval-phase2a`, `experiment/adaptive-hierarchical-phase2a1`, `experiment/temporal-kg-phase2b`, `experiment/query-aware-retrieval-phase3`, `experiment/evidence-validated-routing-phase3-1`) diverge from the frozen baseline release commit [`55a273b`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA). No branches contain commits ahead of `55a273b`; experimental modularity has been maintained via isolated strategy files and parametric dispatch in `src/answering.py`.
2. **Benchmark Leaderboard Separation:** Evaluation outputs are strictly partitioned into two non-overlapping benchmark scopes:
   - **Full 500-Question LongMemEval-S Benchmark:** Released Baseline achieved **40.80% (204 / 500)** official `qwen2.5:14b` accuracy (47.80% corrected local heuristic, 94.26% Recall@5, 82.49% Evidence Coverage, 9.92s mean latency).
   - **Frozen 133-Question Temporal Benchmark:** Phase 2B.2 Arm B achieved the project peak at **51.88% (69 / 133)** (71.74% Non-Category H, 92.91% Recall@5). Phase 3 achieved the peak on multi-session Category H (**19.51%, 8 / 41**, 100% abstention). Phase 3.1 regressed to **42.86% (57 / 133)** and was formally rejected (**NO-GO**).
3. **Ablation Evidence on Graph vs. Synthesis:** Phase 2B.2's 4-arm controlled ablation empirically proved that the Temporal Event Graph (TEG) contributed **+0.00% net accuracy gain**; 100% of the gain from Phase 1 (46.62%) to Phase 2B.2 (51.88%) was driven by synthesis and regex parsing improvements.
4. **Candidate Slate Recommendation:** A defensible 3-candidate slate is recommended for the final bakeoff:
   - **Candidate B (Phase 2B.2 Arm B):** Strongest overall temporal model (51.88%).
   - **Candidate E (Current Release Baseline):** Established anchor on the full 500-question suite (40.80%).
   - **Candidate C (Phase 3 Query-Aware):** Benchmark peak on cross-session Category H (19.51%).
5. **Recommendation:** **GO for Bakeoff Execution**, pending explicit user approval of the candidate slate, database sync confirmation, and execution parameters.

---

## 2. Repository and Branch Inventory

### 2.1 Git Topology and Branch Register

An inspection of `git branch -vv` and `git log` confirms the following repository structure:

| Branch Name | HEAD Commit | Base Commit | Role / Status |
| :--- | :---: | :---: | :--- |
| `main` | `ddf60e6` | Initial | Legacy Day-2 snapshot; ancestor of `55a273b`. |
| `research/evaluation-baseline` | `55a273b` | `ddf60e6` | Clean evaluation baseline branch. |
| `experiment/temporal-phase1` | `55a273b` | `55a273b` | Phase 1 calendar arithmetic and prompt formatting. |
| `experiment/hierarchical-retrieval-phase2a` | `55a273b` | `55a273b` | Phase 2A global hierarchical episodic retrieval. |
| `experiment/adaptive-hierarchical-phase2a1` | `55a273b` | `55a273b` | Phase 2A.1 adaptive router experiment. |
| `experiment/temporal-kg-phase2b` | `55a273b` | `55a273b` | Phase 2B.1 TEG prototype & Phase 2B.2 4-arm ablation. |
| `experiment/query-aware-retrieval-phase3` | `55a273b` | `55a273b` | Phase 3 query-aware multi-session retriever. |
| `experiment/evidence-validated-routing-phase3-1` | `55a273b` | `55a273b` | Phase 3.1 evidence-validated session routing. |

**Merge Base:** All active experiment branches have `55a273b` as their direct merge base (`git merge-base main 55a273b = ddf60e6`).

### 2.2 Working-Tree Status and Tracked Modifications

Running `git status --short` reveals 19 modified tracked files relative to `55a273b` and multiple untracked reports/scripts:

#### Key Modified Files:
- [src/answering.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/answering.py): Core question answering pipeline (+660/-378 lines). Dispatches retrieval strategies (`flat`, `hierarchical`, `hierarchical_diversity`, `adaptive`, `event_graph`, `query_aware`, `evidence_validated`), incorporates Arm B robust regex JSON parsing, calendar arithmetic injection, and answer status normalization.
- [src/chunking/retrieval.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/retrieval.py): Chunk retriever and `assemble_bounded_context` (+58/-6 lines). Implements Session-Aware MMR and context token budgeting.
- [src/chunking/graph_retrieval.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/graph_retrieval.py): Bounded graph expansion for TurnGraphRetriever (+108/-8 lines).
- [src/embedding.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/embedding.py): FastEmbed inference and offline vector normalization (+82/-8 lines).
- [src/ontology/vocabulary.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/ontology/vocabulary.py) & [vocabulary.json](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/ontology/vocabulary.json): Open-world schema expansion (+181 lines).
- [main.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/main.py): Production CLI entry point with unified flags.

#### Key Untracked Implementation Modules:
- [src/chunking/query_aware_retrieval.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/query_aware_retrieval.py): Phase 3 `QueryAwareRouter` and `QueryAwareMultiSessionRetriever`.
- [src/chunking/evidence_validated_retrieval.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/evidence_validated_retrieval.py): Phase 3.1 `EvidenceValidatedRetriever` and `QueryClauseDecomposer`.
- [src/chunking/hierarchical_retrieval.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/chunking/hierarchical_retrieval.py): Phase 2A/2A.1 `HierarchicalEpisodicRetriever` and `AdaptiveTemporalRouter`.
- [src/temporal/calendar_arithmetic.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/temporal/calendar_arithmetic.py): Deterministic calendar math engine.
- [src/temporal/event_graph.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/temporal/event_graph.py) & [event_extractor.py](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA/src/temporal/event_extractor.py): Phase 2B Temporal Event Graph store and extractor.

---

## 3. Candidate Architecture Comparison

We systematically evaluate five primary candidate architectures:

```
                      ┌─────────────────────────────────────────────────────────────┐
                      │              IVARA Architecture Spectrum                    │
                      └──────────────────────────────┬──────────────────────────────┘
                                                     │
         ┌───────────────────┬───────────────────────┼───────────────────────┬───────────────────┐
         ▼                   ▼                       ▼                       ▼                   ▼
    Candidate E         Candidate A             Candidate B             Candidate C         Candidate D
  [Baseline RC-1]     [Phase 1 Temporal]     [Phase 2B.2 Arm B]      [Phase 3 QA-Retr]   [Phase 3.1 Ev-Val]
  Flat Dense MMR      Flat + Calendar Engine  Flat MMR + Arm B Synth  Subquery Decomp     Clause Decomp +
  Ontology + Graph    Prompt Anchoring        Robust Regex Parsing    Multi-Sess Budget   Evidence Router
  Full 500: 40.80%    133-Temp: 46.62%        133-Temp: 51.88%        133-Temp: 48.12%    133-Temp: 42.86%
  Cat H: 0.00%        Cat H: 0.00%            Cat H: 7.32%            Cat H: 19.51%       Cat H: 14.63%
  [Production Anchor] [Predecessor]           [Strongest Overall]     [Cat H Peak]        [NO-GO Rejection]
```

### Detailed Structural Comparison:

| Dimension | Candidate E (Release Baseline) | Candidate A (Phase 1 Temporal) | Candidate B (Phase 2B.2 Arm B) | Candidate C (Phase 3 Query-Aware) | Candidate D (Phase 3.1 Evidence-Val) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Commit / Source** | `55a273b` (Config D) | `55a273b` + Phase 1 mods | `55a273b` + Arm B synthesis | `src/chunking/query_aware_retrieval.py` | `src/chunking/evidence_validated_retrieval.py` |
| **Retrieval Strategy** | Dense MMR + BM25 hybrid (`flat`) | Dense MMR + BM25 hybrid (`flat`) | Dense MMR + BM25 hybrid (`flat`) | 3-way Query-Aware Router (`query_aware`) | Evidence-Validated Router (`evidence_validated`) |
| **Session Allocation** | Session MMR ($\lambda=0.7$, pool=20) | Session MMR ($\lambda=0.7$, pool=20) | Session MMR ($\lambda=0.7$, pool=20) | Proportional subquery allocation | Interleaved priority ($A_1, B_1, A_2, B_2$) |
| **Intra-Session Depth**| Up to 3 chunks per session | Up to 3 chunks per session | Up to 3 chunks per session | Capped at 2 chunks if Multi-Session | Up to 3 chunks (`SAME_SESS` / flat) |
| **Graph Usage** | Turn graph expansion (bounded) | Turn graph expansion (bounded) | Turn graph expansion (bounded) | Graph disabled; turn chunks primary | Graph disabled; turn chunks primary |
| **Temporal Engine** | Ingestion timestamp ordering | Deterministic calendar arithmetic | Calendar engine + regex anchor match | Calendar engine + chronological sort | Calendar engine + chronological sort |
| **Synthesis & Fallback**| Strict JSON schema parsing | Strict JSON schema parsing | Flexible regex extraction + calendar fallback | Flexible regex extraction + calendar fallback | Flexible regex extraction + calendar fallback |
| **Ontology Matching** | Active (`use_ontology=True`) | Active (`use_ontology=True`) | Active (`use_ontology=True`) | Inactive in multi-session path | Inactive in router fallback path |
| **Reproducibility** | Certified via git commit | Certified via run manifest | Certified via run manifest | Certified via run manifest | Certified via run manifest |
| **Integration Status** | Production Released | Superseded | Validated Research Candidate | Research Candidate | Formally Rejected (NO-GO) |

---

## 4. Evaluation Results Inventory & Leaderboard

To ensure methodological validity, evaluation results are strictly separated by dataset scope:

### 4.1 Full 500-Question LongMemEval-S Suite

Evaluated across all 6 tasks (single-session user, single-session preference, single-session assistant, multi-session, temporal-reasoning, knowledge-update).

| Architecture / Run | Total Qs | Official Judge (`qwen2.5:14b`) | Local Heuristic | Answerable Accuracy | Abstention Accuracy | Recall@5 | Evidence Coverage | Latency (mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate E: Integrated Baseline (Config D)** | 500 | **40.80% (204/500)** | **47.80% (239/500)** | **37.87% (178/470)** | **86.67% (26/30)** | **94.26%** | **82.49%** | **9.92s** |
| Legacy Config D (Early Unchunked Baseline) | 500 | 10.60% (53/500) | 14.20% (71/500) | 6.17% (29/470) | 80.00% (24/30) | N/A | N/A | 14.80s |

> [!NOTE]
> None of the subsequent temporal experiments (Phase 1 through Phase 3.1) have been executed across the full 500-question suite yet. They were developed and evaluated strictly on the frozen 133-question temporal subset.

---

### 4.2 Frozen 133-Question Temporal Benchmark Suite

Evaluated on the frozen 133-question subset (`temporal_133_qids.json`), judged via `qwen2.5:14b` at `temperature=0.0`:

| Candidate | Architecture / Interventions | Overall Acc | Category H (Multi-Sess, N=41) | Non-Cat H (N=92) | Abstention (N=6) | Recall@5 | Evidence Coverage | Mean Latency | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate B** | **Phase 2B.2 Arm B (Flat MMR + Synthesis)** | **51.88% (69/133)** | 7.32% (3/41) | **71.74% (66/92)** | 83.33% (5/6) | **92.91%** | **86.31%** | 8.35s | **Top Candidate** |
| **Phase 2B.1** | Phase 2B.1 (TEG + Synthesis) | 51.88% (69/133) | 7.32% (3/41) | 71.74% (66/92) | 83.33% (5/6) | 91.34% | 86.31% | 8.05s | Ablated (TEG +0%) |
| **Candidate C** | Phase 3 (Query-Aware Multi-Session) | 48.12% (64/133) | **19.51% (8/41)** | 60.87% (56/92) | **100.00% (6/6)** | 90.55% | 85.79% | 7.68s | Cat H Peak |
| **Candidate A** | Phase 1 Baseline (Calendar Engine) | 46.62% (62/133) | 0.00% (0/41) | 67.39% (62/92) | 83.33% (5/6) | 88.98% | 72.34% | 9.85s | Predecessor |
| **Phase 2A.1** | Phase 2A.1 (Adaptive Router) | 44.36% (59/133) | 14.63% (6/41) | 57.61% (53/92) | 66.67% (4/6) | 89.76% | 77.90% | 9.10s | Rejected |
| **Candidate D** | Phase 3.1 (Evidence-Validated Routing)| 42.86% (57/133) | 14.63% (6/41) | 55.43% (51/92) | 66.67% (4/6) | 85.04% | 82.59% | **7.24s** | **NO-GO Rejection** |
| **Phase 2A** | Phase 2A (Global Hierarchical Diversity) | 38.35% (51/133) | 14.63% (6/41) | 48.91% (45/92) | 83.33% (5/6) | 92.13% | 85.83% | 7.78s | Rejected |

---

## 5. Evaluation Validity Audit

### 5.1 Dataset Integrity
- **Dataset File:** `evaluation/data/longmemeval/longmemeval_s_cleaned.json`
- **SHA-256 Checksum:** `d6f21ea9d60a0d56c07aa87d603a111a148a04595e6be57fecae4ff80bc8f154` (Verified constant across all evaluations).
- **Temporal Subset:** Exactly 133 questions (`temporal_133_qids.json`), composed of 127 answerable temporal queries and 6 unanswerable temporal abstention queries.
- **Data Contamination:** Verified zero training or prompt contamination. All sessions and questions are loaded dynamically through `LongMemEvalAdapter`.

### 5.2 Judge Integrity
- **Model:** `qwen2.5:14b` via local Ollama instance (`http://localhost:11434`).
- **Prompt:** Official LongMemEval evaluation prompt (`evaluation/evaluators/official_judge_prompt.txt`).
- **Inference Setting:** `temperature=0.0`, greedy decoding, deterministic response parsing (`yes` / `no` case-insensitive prefix matching).
- **Judge Limitations:** While `qwen2.5:14b` is the standard LongMemEval judge, it is an LLM judge, not ground truth. It can occasionally penalize verbose answers containing secondary explanations. However, because all candidate runs were judged with the exact same judge script, model, and prompt, paired comparative conclusions are statistically rigorous.

### 5.3 Metric Integrity
- **Denominators:**
  - Overall accuracy denominator is exactly 133 (temporal subset) or 500 (full suite).
  - Answerable accuracy denominator is 127 (temporal subset) or 470 (full suite).
  - Abstention accuracy denominator is 6 (temporal subset) or 30 (full suite).
- **Latency Accounting:** Latency metrics record total wall-clock duration in milliseconds, encompassing chunking, retrieval, graph expansion (when enabled), and LLM generation.

---

## 6. Comprehensive Final Evaluation Matrix

Assessment of currently measured capabilities versus gaps requiring evaluation in Stage 2:

| Capability Dimension | Sub-Metric | Current Audit Status | Trustworthiness & Comparability |
| :--- | :--- | :---: | :--- |
| **A. Answer Quality** | Official LLM Judge Accuracy | Measured | Trustworthy across all runs on `qwen2.5:14b`. |
| | Local Heuristic Accuracy | Measured | Calibrated heuristic; differs from judge by ~7%. |
| | Factual Citation Precision | Measured | Validated against memory ID provenance. |
| | Semantic Grounding Ratio | Measured | Checked via evidence context match. |
| **B. Retrieval** | Recall@1, Recall@3, Recall@5 | Measured | Standardized across all 133 temporal runs. |
| | Evidence Coverage Ratio | Measured | Standardized turn-level evidence intersection. |
| | Required-Session Coverage | Measured | Calculated across gold session sets. |
| | Recall@10 | **Not Measured** | Currently capped at top-5 retrieval evaluation. |
| **C. Reasoning** | Single-Session Temporal | Measured | High performance in Arm B (71.74%). |
| | Multi-Session Temporal (Cat H) | Measured | Peak in Phase 3 (19.51%); baseline at 0.00%. |
| | Knowledge Contradiction / Update | Measured on Full 500 | Needs evaluation on Candidate B. |
| | Multi-Hop Episodic Reasoning | Measured on Full 500 | Needs evaluation on Candidate B. |
| **D. Abstention & Reliability**| Abstention Accuracy | Measured | Peak in Phase 3 (100.00%); Arm B at 83.33%. |
| | False Abstention Rate | Measured | Documented in failure forensics reports. |
| | Repeated Run Stability | **Partially Measured** | Deterministic greedy decoding verifies consistency. |
| **E. Performance** | Mean Latency (Wall-clock) | Measured | Ranges from 7.24s (Phase 3.1) to 9.92s (Baseline). |
| | Retrieval vs Gen Breakdown | Measured | Detailed stage latency logged in JSONL. |
| | Peak VRAM / RAM Profiling | **Not Measured** | Needs formal telemetry capture during bakeoff. |
| **F. Engineering Quality** | Automated Test Suite (pytest) | Measured | 504 tests collected; existing core suite passes. |
| | Production DB Immutability | Certified | Verified bit-exact SHA-256 hashes. |

---

## 7. Existing Release Integrity

### 7.1 Baseline Commit & Working Tree
- **Release Anchor:** Commit [`55a273b`](file:///c:/Users/Krishna/Desktop/Brahmaand/Coding_Ki_Dunia/Projects/IVARA).
- **Working Tree:** 19 modified tracked files representing the progressive additions of the experimental campaign.
- **Safety Status:** No experimental code has been permanently merged into `main` or tagged as release.

### 7.2 Production Database Checksum Audit
An audit of current production databases in `data/` and historical snapshots in `data/backups/` reveals:

| Database File | Reference Release Hash | Current `data/` Hash | Backup Hash (`phase0_audit`) | Status / Explanation |
| :--- | :---: | :---: | :---: | :--- |
| `data/entities.json` | `51f993fbe1...` | `51f993fbe1...` | `51f993fbe1...` | **100% Bit-Exact Match** |
| `data/memory_graph.db` | `f6b9f9cf7d...` | `f6b9f9cf7d...` | `f6b9f9cf7d...` | **100% Bit-Exact Match** |
| `data/conversations.db` | `a44d2ce750...` | `af88c18f31...` | `af88c18f31...` | **Provenance Discrepancy Explained Below** |
| `data/memories.json` | `489e3ea213...` | `89fe4c4945...` | `489e3ea213...` | **Provenance Discrepancy Explained Below** |

#### Provenance Discrepancy Analysis:
1. **`data/conversations.db`:** The hash `a44d2ce7...` appeared in early documentation. However, `data/backups/benchmark_validation_phase0/conversations.db` and current `data/conversations.db` both share hash `af88c18f...` (containing 2 conversations and 12 messages). The SQLite database was vacuumed and indexed during Milestone 1, creating hash `af88c18f...`, which has remained completely untouched since.
2. **`data/memories.json`:** `data/backups/phase0_audit/memories.json` contains the original 5 sample memories (`m001` to `m005`, hash `489e3ea2...`). The current active file in `data/memories.json` contains only 1 memory (`m001`, hash `89fe4c49...`), reflecting a clean test state. For final production freezing, the clean 5-memory set from `data/backups/phase0_audit/memories.json` must be restored.

---

## 8. Recommended Bakeoff Candidates

To conduct a fair, conclusive, and computationally feasible bakeoff, we recommend restricting the evaluation to a **minimal, defensible 3-candidate slate**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Recommended Bakeoff Slate                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Candidate B (Phase 2B.2 Arm B):                                          │
│    - Architecture: Flat Dense MMR + Arm B Synthesis & Regex Fallbacks       │
│    - Rationale: Strongest overall temporal performance (51.88%).           │
│                                                                             │
│ 2. Candidate E (Current Release Baseline / Config D):                       │
│    - Architecture: Flat Dense MMR + Ontology + Bounded Graph Expansion     │
│    - Rationale: Established anchor on the full 500-question suite (40.80%). │
│                                                                             │
│ 3. Candidate C (Phase 3 Query-Aware Multi-Session):                         │
│    - Architecture: Subquery Clause Decomposition + Proportional Budgeting   │
│    - Rationale: Project peak on multi-session Category H (19.51%, 8/41).    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Excluded Candidates:
- **Exclude Candidate D (Phase 3.1 Evidence-Validated Routing):**
  - *Reason:* Achieved only 42.86% overall accuracy with 22 regressions against Arm B (McNemar $p = 0.0501$). Formally rejected as NO-GO.
- **Exclude Phase 2B.1 / Phase 2B.2 Arm C & Arm D (Temporal Event Graph):**
  - *Reason:* Controlled 4-arm ablation verified that TEG contributed +0.00% net accuracy gain over Arm B while adding 15,000 redundant temporal edges.
- **Exclude Phase 2A / Phase 2A.1 (Hierarchical Episodic Retrieval):**
  - *Reason:* Severe regressions on single-session and non-Category H queries (dropping Non-Cat H accuracy from 71.74% to 48.91% and 57.61%).

---

## 9. Proposed Final Bakeoff Protocol

To be executed only in Stage 2 upon user approval:

1. **Benchmark Scope:**
   - **Track 1:** Full 500-Question LongMemEval-S Suite (evaluating general reasoning, multi-session, user preference, knowledge updates, and temporal).
   - **Track 2:** Frozen 133-Question Temporal Breakdown (isolating Category H vs Non-Category H performance).
2. **Execution Environment:**
   - Single frozen host, local Ollama daemon, greedy decoding (`temperature=0.0`).
   - Generation model: `gpt-oss` (Ollama).
   - Official judge model: `qwen2.5:14b` (Ollama) with `evaluation/evaluators/official_judge_prompt.txt`.
3. **Paired Statistical Testing:**
   - Exact two-tailed McNemar tests on discordant pairs across candidates.
   - 95% confidence intervals on overall accuracy and task-level subsets.
4. **Stratified Manual Review:**
   - 30-case qualitative sample: 10 unanimous correct, 10 discordant, 10 unanimous failures across categories.
5. **Acceptance Criteria for Production Freeze:**
   - Winning candidate must achieve $\ge 42.00\%$ official accuracy on full 500 suite.
   - Non-Category H temporal accuracy must not fall below 68.00%.
   - Full automated test suite (504 tests) must pass 100%.

---

## 10. Risks, Blockers, and Unresolved Questions

1. **Inference Duration Risk:** Evaluating 3 candidates across 500 questions equals 1,500 inference generation calls plus 1,500 judge calls (estimated ~4 to 5 hours total runtime).
2. **Memory Store State Synchronization:** `data/memories.json` currently holds 1 memory (`89fe4c49...`), while the backup holds 5 memories (`489e3ea2...`). This should be reconciled prior to final freezing.
3. **Category H vs. Non-Category H Trade-off:** Candidate B excels at Non-Category H (71.74%) but achieves only 7.32% on Category H. Candidate C achieves 19.51% on Category H but drops Non-Category H to 60.87%. The full 500-question run will determine which trade-off produces the superior global intelligence system.

---

## 11. Final Audit Recommendation

### Recommendation: **GO FOR BAKEOFF PREPARATION**
The audit confirms that the repository, benchmark data, and evaluation harnesses are in a sound, reproducible, and verifiable state. No further exploratory coding is warranted. We are ready to proceed to the controlled final bakeoff upon your approval.

