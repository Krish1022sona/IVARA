# IVARA vs. Mem0 LongMemEval Comparison Protocol

**Document ID:** `IVARA-EVAL-MEM0-PROTO-01`  
**Date:** 2026-10-27  
**Status:** Approved Protocol Design  

---

## Executive Summary

To prevent misleading claims of superiority, this document defines a scientifically rigorous, two-track comparison protocol for benchmarking IVARA against Mem0 on LongMemEval.

Crucially:
- A comparison of IVARA top-5 chunk retrieval against Mem0 top-200 memory retrieval measures context efficiency, but is not an architecture-only comparison by itself.
- We establish two parallel evaluation tracks to isolate retrieval quality from prompt budgeting and LLM capacity.

---

## Track A: Matched Conditions (Controlled Architectural Isolation)

Track A controls all external variables to isolate differences in memory indexing and retrieval architectures.

| Parameter | Matched Specification | Rationale |
| :--- | :--- | :--- |
| **Benchmark Dataset** | `longmemeval_oracle.json` and `longmemeval_s_cleaned.json` (identical 500 questions) | Standardized test split across both systems |
| **Context Budget ($k$)** | **Matched $k = 5$ items** and **Matched $k = 10$ items** | Tests which system provides higher precision under strict token constraints |
| **Answering LLM** | Standardized `gpt-4o-mini` (or standardized local `gpt-oss` via Ollama) | Eliminates answerer backbone discrepancies |
| **Judge Model & Rubric** | Official `gpt-4o` evaluator with verbatim prompts from `xiaowu0162/LongMemEval` | Guarantees identical grading thresholds |
| **Prompt Template** | Standardized evidence container + question prompt | Eliminates reading style bias |
| **Gold Answer Isolation** | Zero access to `has_answer` or `answer_session_ids` during indexing/retrieval | Preserves strict benchmark integrity |

### Unavoidable Differences in Track A:
- **Index Granularity:** Mem0 stores extracted atomic facts as flat vector records; IVARA stores coherent chunk nodes ($W=120$ words) linked to SQLite graph triples.
- **Backend:** Mem0 OSS uses Qdrant via Docker; IVARA uses in-process FAISS + BM25 + SQLite.

---

## Track B: Native Recommended Configurations (End-to-End System Benchmark)

Track B evaluates each system operating in its documented, recommended production configuration to reflect real-world performance and operational trade-offs.

| Dimension | IVARA Native Recommended | Mem0 Native Recommended |
| :--- | :--- | :--- |
| **Retrieval Strategy** | Config D (Hybrid Vector + BM25 + SQLite Graph Triples) | Hybrid Dense Vector + BM25 + Recency Decay factor (0.3×–1.5×) |
| **Retrieval Context Budget** | **Top-5 evidence chunks** (~600 tokens) | **Top-200 memory items** (~4,000–8,000 tokens) |
| **Abstention Mechanism** | Dual-stage pre-generation relevance gating + citation verification | Prompt-directed LLM refusal in reader prompt |
| **Storage Infrastructure** | Local SQLite graph store + local FAISS index | Qdrant vector database + SQLite metadata (OSS) or Mem0 Cloud |
| **Primary Reported Metrics** | Accuracy, Token Cost per Query, p95 Latency, Citation Grounding | Accuracy, Token Cost per Query, p95 Latency |

---

## Required Comparative Metrics for Both Tracks

1. **Category Breakdown Accuracies:**
   - Knowledge Updates (KU)
   - Temporal Reasoning (TR)
   - Abstention Accuracy (ABS)
   - Multi-Session Reasoning (MR)
   - Information Extraction (IE: Single-Session User & Assistant)
2. **Summary Accuracies:**
   - Task-averaged Accuracy (mean of tasks)
   - Overall Accuracy (micro-average across 500 questions)
3. **Efficiency & Grounding:**
   - Prompt token consumption per query (Mean & p95)
   - End-to-end latency in milliseconds (Mean & Median)
   - Hallucination rate on unanswerables (False Positive Rate)
   - Citation validity rate (percentage of cited memory IDs matching retrieved evidence)

---

## Adoption Criteria for Performance Claims

Before IVARA can claim superiority over Mem0:
1. Both systems must be evaluated on the **exact same 500 questions** using the **exact same judge model**.
2. IVARA must demonstrate a statistically significant gain on **Knowledge Updates** and **Abstention** under Track A without regressions in standard Single-Session recall.
3. If IVARA's accuracy matches Mem0 while consuming $\ge 70\%$ fewer prompt tokens (Top-5 vs Top-200), IVARA is superior in token efficiency and precision, but not unconstrained recall. Both dimensions must be stated explicitly.
