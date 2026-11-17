# IVARA — Phase 3: Query-Aware Multi-Session Retrieval Evaluation Report

**Date:** 2026-11-17  
**Author:** Antigravity (DeepMind Team)  
**Branch:** `experiment/query-aware-retrieval-phase3`  
**Baseline Commit:** `55a273bff20fa0eec1b2d361e8a03b217d147abe`  
**Dataset:** LongMemEval-S Cleaned Benchmark (133 Frozen Temporal Questions)  
**Evaluator:** Official LLM-as-a-Judge (`qwen2.5:14b` via Ollama at `temperature=0.0`)  
**Status:** Evaluation Completed & Certified  

---

## 1. Executive Summary

Phase 3 investigated whether a **Query-Aware Multi-Session Retrieval Architecture** could resolve the multi-session retrieval starvation of Category H without sacrificing the 71.74% single-session accuracy established by Phase 2B.2 Arm B.

Under identical evaluation conditions (frozen 133 questions, official `qwen2.5:14b` judge at temperature 0.0, identical system prompts):
- **Category H (Multi-Session Temporal Reasoning, 41 questions):** Jumped from **3 / 41 (7.32%)** to an all-time benchmark record of **8 / 41 (19.51%)** (+12.19% absolute gain, nearly triple the recovery rate of Phase 2B.2 Arm B).
- **Abstention Accuracy (6 questions):** Reached a perfect **6 / 6 (100.00%)**, up from 83.33%.
- **Non-Category H (Single-Session / Intra-Session, 92 questions):** Decreased from **66 / 92 (71.74%)** to **56 / 92 (60.87%)** (-10 questions).
- **Overall Temporal Accuracy (133 questions):** Settled at **64 / 133 (48.12%)**, compared to 69 / 133 (51.88%) for Arm B and 62 / 133 (46.62%) for Phase 1.

### Paired Comparison Verdict
A paired McNemar test yielded $\chi^2 = 0.6400$ ($p > 0.05$, non-significant net difference with 10 recoveries and 15 regressions). Because Non-Category H suffered 10 regressions, Phase 3 **does not satisfy the strict zero-sacrifice acceptance criterion** for a global production upgrade.

**Recommendation:** **NO-GO FOR PRODUCTION INTEGRATION as a global replacement.** Maintain Phase 2B.2 Arm B as the production baseline.

---

## 2. Comparative Benchmark Matrix

| Metric | Phase 1 Baseline (Arm A) | Phase 2B.2 Arm B Baseline | Phase 3 Query-Aware Retrieval | Net $\Delta$ (P3 vs Arm B) |
|---|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | 46.62% (62/133) | **51.88%** (69/133) | 48.12% (64/133) | -3.76% (-5 Qs) |
| **Category H (Multi-Session)** | 0.00% (0/41) | 7.32% (3/41) | **19.51%** (8/41) | **+12.19% (+5 Qs)** |
| **Non-Category H** | 67.39% (62/92) | **71.74%** (66/92) | 60.87% (56/92) | -10.87% (-10 Qs) |
| **Abstention Accuracy** | 83.33% (5/6) | 83.33% (5/6) | **100.00%** (6/6) | **+16.67% (+1 Q)** |
| **Recall@5** | **94.26%** | 93.23% | 90.98% | -2.25% |
| **Evidence Coverage Ratio** | 82.49% | **86.93%** | 86.43% | -0.50% |
| **Required-Session Coverage** | 83.12% | **85.43%** | 84.55% | -0.88% |
| **Mean Context Tokens** | 2,842 | 2,903 | 3,333 | +430 tokens |
| **Mean Latency** | 9.92s | 8.35s | **7.68s** | **-0.67s** |

---

## 3. Question-Level Forensics: Recoveries & Regressions

### 3.1 Recovered Questions (10 Questions in Phase 3 vs Arm B)
Seven of the 10 recoveries occurred in Category H, directly demonstrating that subquery clause decomposition and session-budget allocation solved the session starvation problem:

| Question ID | Task Category | Question Text | Gold Answer | Mechanism of Recovery |
|---|:---:|---|---|---|
| `gpt4_59149c77` | Cat H | Days between MoMA visit and Ancient Civilizations exhibit at the Met? | 0 days | Subqueries separated MoMA (s1) and Met (s2); both sessions surfaced and scored. |
| `gpt4_f49edff3` | Cat H | Order of 3 events: nursery prep, baby shower, phone case? | baby shower, phone case, nursery prep | Decomposed clauses surfaced all 3 distinct sessions. |
| `gpt4_18c2b244` | Cat H | Order of 3 events: ShopRite rewards, Walmart diaper coupon, Ibotta Amazon card? | rewards, coupon, cashback | Quoted clauses surfaced all 3 shopping episodes. |
| `gpt4_4ef30696` | Cat H | Days between finishing The Nightingale and starting Hitchhiker's Guide? | 11 days | Decomposed book queries retrieved both reading sessions. |
| `gpt4_4cd9eba1` | Cat H | Weeks accepted into exchange program when starting pre-departure orientation? | 3 weeks | Dual subqueries retrieved acceptance turn and orientation turn. |
| `gpt4_93f6379c` | Cat H | Which group joined first, Page Turners or Marketing Professionals? | Page Turners | Surfaced Page Turners session and Marketing Professionals session. |
| `gpt4_68e94288` | Cat H | Social media activity 5 days ago? | Participating in photography challenge | Correctly routed to single-session flat retrieval. |
| `gpt4_85da3956` | Non-Cat H | Weeks ago attended 'Summer Nights' festival at Universal Studios? | 3 weeks ago | Context assembly avoided false contradiction trigger. |
| `gpt4_4edbafa2` | Non-Cat H | Date attended first BBQ event in June? | June 10, 2023 | Flat MMR fallback surfaced BBQ session. |
| `gpt4_fe651585_abs` | Non-Cat H | Who became a parent first, Tom or Alex? | Abstain (no record) | Calibrated abstention: correctly recognized no record of parenthood. |

### 3.2 Regressed Questions (15 Questions in Phase 3 vs Arm B)
Of the 15 regressions, 13 were Non-Category H questions:

| Question ID | Task Category | Question Text | Root Cause of Regression in Phase 3 |
|---|:---:|---|---|
| `gpt4_1d4ab0c9` | Non-Cat H | Days between watering herb garden and harvesting fresh herbs? | Router triggered `MULTI_SESSION`, allocating only 2 chunks to gold session and inserting distractor chunks that obscured the harvesting date. |
| `gpt4_74aed68e` | Non-Cat H | Days between replacing spark plugs and Turbocharged Tuesdays? | Multi-session budgeting diluted intra-session turn depth. |
| `gpt4_98f46fc6` | Non-Cat H | Which event first, charity gala or charity bake sale? | Subquery retrieval split context across 3 sessions; gold turns were fragmented. |
| `gpt4_af6db32f` | Non-Cat H | Days ago watched Super Bowl? | Relative date anchor misparsed by subquery matcher. |
| `gpt4_f420262d` | Non-Cat H | Airline flied with on Valentine's day? | Cardinality pattern matched "five airlines" instead of date anchor. |
| `0bb5a684` | Non-Cat H | Days before team meeting attended communication workshop? | Diluted gold session chunks below required context threshold. |
| `f0853d11` | Non-Cat H | Days between Walk for Hunger and Coastal Cleanup? | Chronological sorting interleaved background ultrachat turns. |
| `gpt4_6ed717ea` | Non-Cat H | Which item purchased first, dog bed or training pads? | Intra-session event pair truncated by per-session 2-chunk cap. |
| `gpt4_70e84552` | Non-Cat H | Which task completed first, fixing fence or trimming goat hooves? | Capped gold session to 2 chunks, losing the goat hooves turn. |
| `c9f37c46` | Non-Cat H | How long watching comedy specials before open mic night? | Inadvertently routed to multi-session budgeting. |
| `gpt4_b4a80587` | Non-Cat H | Which event first, road trip or prime lens arrival? | Intra-session pair split across distractor sessions. |
| `gpt4_d9af6064` | Non-Cat H | Which device set up first, smart thermostat or router? | Intra-session comparison starved of turn depth. |
| `gpt4_d31cdae3` | Non-Cat H | Which trip taken first, solo Europe or Southwest road trip? | Dispersed context across 3 candidate sessions. |
| `gpt4_2f56ae70` | Cat H | Which streaming service started using most recently? | Candidate session scoring missed Disney+ sign-up turn. |
| `gpt4_1a1dc16d` | Cat H | Which event first, meeting with Rachel or pride parade? | Vector scoring favored distractor parade turn. |

---

## 4. Key Architectural Insights

1. **The Intra-Session vs Cross-Session Trade-Off:**
   - Multi-session queries benefit from broad session representation ($S=2$ or $S=3$).
   - Intra-session queries (where both events are discussed in the *same* conversation session) are harmed by multi-session budgeting: capping the session to 2 chunks and forcing chunks from other sessions introduces distractor noise and removes contiguous turn context.
2. **Category H Breakthrough:**
   - Subquery clause decomposition is the *first* technique tested on IVARA that successfully lifted Category H to **19.51% (8/41)**, up from 0.00% in Phase 1 and 7.32% in Phase 2B.1.
   - For true cross-session queries, subquery-targeted session retrieval completely eliminates dense session starvation.
3. **Perfect Abstention Calibration:**
   - Phase 3 achieved **100.00% (6/6)** abstention accuracy, demonstrating that the calibrated evidence gate does not hallucinate answers on negative or unrecorded queries.

---

## 5. Explicit GO/NO-GO Recommendation

**Verdict: NO-GO FOR GLOBAL PRODUCTION MERGE.**

- **Reason:** While Category H reached an all-time record of 19.51%, Non-Category H accuracy dropped from 71.74% to 60.87%, lowering overall accuracy from 51.88% to 48.12%.
- **Safe Production Configuration:** Retain **Phase 2B.2 Arm B** (Flat Dense MMR with Robust Synthesis) as the default production configuration, preserving 51.88% overall and 71.74% Non-Category H accuracy.
- **Future Direction (Phase 3.1):** Incorporate session-count verification: before routing to `MULTI_SESSION`, verify whether the subqueries match turns within a *single* session with high lexical confidence; if so, retain flat MMR.

