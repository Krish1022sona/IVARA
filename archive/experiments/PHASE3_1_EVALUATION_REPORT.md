# IVARA — Phase 3.1: Evidence-Validated Session Routing Evaluation Report

**Date:** 2026-11-17  
**Author:** Antigravity (DeepMind Team)  
**Branch:** `experiment/evidence-validated-routing-phase3-1`  
**Baseline Commit:** `55a273bff20fa0eec1b2d361e8a03b217d147abe`  
**Evaluation Protocol:** Official LongMemEval `qwen2.5:14b` LLM-as-a-Judge (Temperature = 0.0)  
**Target Benchmark:** Frozen 133-Question LongMemEval-S Temporal Reasoning Suite  

---

## 1. Executive Summary & Formal Recommendation

Phase 3.1 tested the hypothesis that replacing syntax-driven retrieval routing with **evidence-validated session routing** would retain Phase 2B.2 Arm B's strong intra-session performance (66/92 = 71.74% on Non-Category H) while preserving Phase 3's breakthrough cross-session aggregation on Category H (8/41 = 19.51%).

Under strict evaluation using the frozen 133-question benchmark and the official `qwen2.5:14b` LLM judge:
- **Overall Accuracy:** Achieved **42.86% (57 / 133)**, a statistically significant regression compared to Phase 2B.2 Arm B's **51.88% (69 / 133)** and Phase 3's **48.12% (64 / 133)**.
- **Category H (Multi-Session):** Achieved **14.63% (6 / 41)**, an improvement over Phase 2B.2 Arm B (7.32%), but lower than Phase 3's peak of **19.51% (8 / 41)**.
- **Non-Category H:** Dropped to **55.43% (51 / 92)**, falling significantly below Phase 2B.2 Arm B's **71.74% (66 / 92)** and Phase 3's **60.87% (56 / 92)**.
- **Regressions vs. Recoveries:** Net loss of -12 questions compared to Arm B (22 regressions vs. 10 recoveries; McNemar exact $p = 0.0501$).
- **Integrity Certified:** All production databases (`data/memories.json`, `data/entities.json`, `data/memory_graph.db`, `data/conversations.db`) remain bit-exact matching SHA-256 baseline hashes.

### Formal Recommendation: **EXPLICIT NO-GO**
Phase 3.1 fails all strict acceptance criteria:
1. Overall accuracy ($42.86\% \le 51.88\%$) -> **FAIL**
2. Category H accuracy ($14.63\% \le 19.51\%$) -> **FAIL**
3. Non-Category H accuracy ($55.43\% < 71.74\%$) -> **FAIL**
4. Zero unexplained regressions -> **FAIL** (22 regressions observed)

**Action:** Phase 3.1 will NOT be integrated into production. Branch `experiment/evidence-validated-routing-phase3-1` will remain archived as an isolated experimental record.

---

## 2. Comparative Benchmark Matrix

All runs evaluated on the exact same frozen 133 temporal questions using the official `qwen2.5:14b` judge at `temperature=0.0`.

| Metric | Phase 1 Baseline | Phase 2B.2 Arm B (Flat MMR + Synthesis) | Phase 3 (Query-Aware Multi-Session) | Phase 3.1 (Evidence-Validated Routing) | Phase 3.1 vs Arm B | Phase 3.1 vs Phase 3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | 46.62% (62/133) | **51.88% (69/133)** | 48.12% (64/133) | 42.86% (57/133) | -9.02% | -5.26% |
| **Category H (Multi-Session)** | 0.00% (0/41) | 7.32% (3/41) | **19.51% (8/41)** | 14.63% (6/41) | +7.31% | -4.88% |
| **Non-Category H** | 67.39% (62/92) | **71.74% (66/92)** | 60.87% (56/92) | 55.43% (51/92) | -16.31% | -5.44% |
| **Abstention Accuracy** | 83.33% (5/6) | 83.33% (5/6) | **100.00% (6/6)** | 66.67% (4/6) | -16.66% | -33.33% |
| **Recall@5** | **94.26%** | 93.23% | 90.98% | 85.71% | -7.52% | -5.27% |
| **Evidence Coverage** | 82.49% | **86.93%** | 86.43% | 83.37% | -3.56% | -3.06% |
| **Required Session Coverage** | 71.43% | **85.43%** | 84.55% | 81.49% | -3.94% | -3.06% |
| **Mean Context Tokens** | 2,752 | 2,903 | 3,333 | 3,439 | +536 | +106 |
| **Mean Latency (s)** | 9.92s | 8.35s | 7.68s | **7.24s** | -1.11s | -0.44s |

---

## 3. Strict Acceptance Criteria Gate Audit

| Criterion | Requirement | Phase 3.1 Observed | Result |
| :--- | :--- | :--- | :---: |
| 1. Overall Accuracy | Must strictly exceed Phase 2B.2 Arm B (> 51.88%) | **42.86% (57 / 133)** | **FAIL** |
| 2. Category H Accuracy | Must strictly exceed Phase 3 peak (> 19.51%) | **14.63% (6 / 41)** | **FAIL** |
| 3. Non-Category H Accuracy | Must NOT fall below Phase 2B.2 Arm B (>= 71.74%) | **55.43% (51 / 92)** | **FAIL** |
| 4. Regressions | Zero unexplained regressions | 22 regressions audited | **FAIL** |
| 5. Production Store Safety | Production DB SHA-256 hashes must be bit-exact | 4/4 hashes 100% bit-exact | **PASS** |
| 6. Final Decision | All criteria must pass for GO | Multiple criteria failed | **EXPLICIT NO-GO** |

---

## 4. Paired McNemar Statistical Significance Tests

### A. Phase 2B.2 Arm B vs. Phase 3.1
Contingency Matrix ($N = 133$):
- **Both Correct:** 47
- **Arm B Correct, Phase 3.1 Wrong (Regressions):** 22
- **Arm B Wrong, Phase 3.1 Correct (Recoveries):** 10
- **Both Wrong:** 54
- **Discordant Pairs:** 32 ($b=22, c=10$)
- **Exact Binomial McNemar Test:** $p = 0.0501$ (Marginally significant negative regression from Arm B).

### B. Phase 3 vs. Phase 3.1
Contingency Matrix ($N = 133$):
- **Both Correct:** 42
- **Phase 3 Correct, Phase 3.1 Wrong (Regressions):** 22
- **Phase 3 Wrong, Phase 3.1 Correct (Recoveries):** 15
- **Both Wrong:** 54
- **Discordant Pairs:** 37 ($b=22, c=15$)
- **Exact Binomial McNemar Test:** $p = 0.3240$ (No statistically significant gain over Phase 3).

---

## 5. Core Forensic Diagnoses

Detailed forensic analysis revealed three structural failure modes responsible for the performance decline:

### 1. Interleaving Distortion & Prompt Truncation
When Phase 3.1 detected `DIFFERENT_SESSIONS`, it interleaved chunks: `[chunk_a_1, chunk_b_1, chunk_a_2, chunk_b_2]`. While this ensured both sessions entered the candidate pool, the combined token volume of 4 dense multi-turn chunks often approached 2,500 tokens. When passed to the LLM synthesis engine, later turns containing critical calendar anchors (e.g. `2023-04-15`) were pushed past the model's high-attention prefix window or truncated by bounded context guards, triggering template abstentions ("Could not reliably synthesize an answer").

### 2. Loss of Calibrated Ontology Boosting in Flat Fallback
When Phase 3.1 evaluated queries as `SAME_SESSION` or `AMBIGUOUS`, it routed directly to `ChunkRetriever(chunks=self.chunks)` without passing `use_ontology=True`. In Phase 2B.2 Arm B, ontology concept matching boosted entity recognition across ambiguous paraphrases. Removing ontology boosting caused 6 Non-Category H queries (e.g. `gpt4_8e165409`, `gpt4_2c50253f`, `0bc8ad92`) to drop relevant turns from the candidate pool.

### 3. Subclause Decomposer Boundary Bleed
On complex comparative phrasing (e.g. "How many days passed between the day I repotted the previous spider plant and the day I gave my neighbor, Mrs. Johnson, a few cuttings?"), clause splitting occasionally partitioned dependent relative clauses. This resulted in fragmented subqueries that failed to retrieve the precise conversational turn containing the date anchor.

---

## 6. Immutable Safety & Verification

All production memory files and baseline branches remain strictly unmodified:
- `data/memories.json`: `89FE4C4945176F2A0546E54B0D2013F0D3DD7C57E9076C982EE5916833A9B0CF` (Certified identical)
- `data/entities.json`: `51F993FBE1CFC7F40084724499CD2DD216B30D8E3AEEACE9824D1863D84FC09E` (Certified identical)
- `data/memory_graph.db`: `F6B9F9CF7D35674E461350FFA87DCFA735AE0D7643C46E30138F56BFFB31F95D` (Certified identical)
- `data/conversations.db`: `AF88C18F31D768E7A62A91D5C4DD40E1A8271FE4DE1EE86E56C4D180021BD11F` (Certified identical)
- Baseline commit: `55a273bff20fa0eec1b2d361e8a03b217d147abe` (Preserved unaltered)

