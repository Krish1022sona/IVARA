# IVARA — Phase 2B.2: Controlled Ablation Report

**Date:** 2026-11-17  
**Author:** Antigravity (DeepMind Team)  
**Branch:** `experiment/temporal-kg-phase2b`  
**Baseline Commit:** `55a273bff20fa0eec1b2d361e8a03b217d147abe`  
**Dataset:** LongMemEval-S Cleaned Benchmark (133 Frozen Temporal Questions)  
**Evaluator:** Official LLM-as-a-Judge (`qwen2.5:14b` via Ollama at `temperature=0.0`)  
**Status:** Audit Completed & Experimentally Verified  

---

## 1. Executive Summary

Phase 2B.1 demonstrated an apparent performance leap from the released Phase 1 baseline of **46.62% (62/133)** to **51.88% (69/133)** (+5.26% absolute). However, concurrent forensic observations raised two critical questions:
1. Did the **Temporal Event Graph (TEG)** and its 15,774 temporal edges drive this accuracy increase, or was it an artifact of synthesis/JSON parsing improvements introduced during the same phase?
2. Why did Category H (Multi-Session Temporal Reasoning) achieve only **3 / 41 (7.32%)** in Phase 2B.1 compared to **6 / 41 (14.63%)** in Phase 2A.1?

To isolate the exact causal factors, we conducted a rigorous, strictly controlled **4-Arm Ablation Experiment** across all 133 temporal questions under identical evaluation conditions (same frozen dataset, same generator model, same official `qwen2.5:14b` judge at temperature 0.0, identical system prompts):
- **Arm A:** Phase 1 Flat Baseline
- **Arm B:** Phase 1 Flat Retrieval + Synthesis / JSON Parsing Improvements Only
- **Arm C:** Phase 1 + Temporal Event Graph Only (without synthesis improvements)
- **Arm D:** Phase 1 + Temporal Event Graph + Synthesis Improvements (Phase 2B.1 run)

### Key Empirical Finding: The Zero Marginal Graph Contribution
The empirical results establish conclusively that:
- **Marginal Gain from Temporal Event Graph Alone (Arm C vs Arm A):** **0.00% (0 questions net)**.
- **Marginal Gain from Synthesis / Parsing Improvements Alone (Arm B vs Arm A):** **+5.26% (+7 questions net, 62 → 69)**.
- **Marginal Gain from Adding TEG on Top of Synthesis (Arm D vs Arm B):** **0.00% (0 questions net, 69 → 69)**. In fact, 131 out of 133 generated hypotheses (98.5%) and **100% of the judge decisions (133/133)** were identical between Arm B (Flat) and Arm D (TEG).

**Conclusion:** The Temporal Event Graph prototype, in its current formulation, contributed **zero net accuracy** to temporal reasoning. The entirety of the 5.26% performance gain was driven by robust JSON parsing, regex answer extraction, and calendar fallback in `src/answering.py`.

---

## 2. 4-Arm Controlled Ablation Matrix

All four arms were evaluated on the exact same 133 temporal questions using the official `qwen2.5:14b` LLM judge:

| Arm | Configuration | Retrieval Strategy | Synthesis Pipeline | Overall Accuracy (133 Qs) | Category H Accuracy (41 Qs) | Non-Category H Accuracy (92 Qs) | Abstention Accuracy (6 Qs) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | Phase 1 Baseline | Flat Dense MMR ($k=5$) | Strict Pydantic JSON | **46.62%** (62/133) | **0.00%** (0/41) | **67.39%** (62/92) | 83.33% (5/6) |
| **B** | Flat + Improved Synthesis | Flat Dense MMR ($k=5$) | Robust Regex + Calendar | **51.88%** (69/133) | **7.32%** (3/41) | **71.74%** (66/92) | 83.33% (5/6) |
| **C** | TEG Only | TurnGraphRetriever ($k=5$) | Strict Pydantic JSON | **46.62%** (62/133) | **0.00%** (0/41) | **67.39%** (62/92) | 83.33% (5/6) |
| **D** | TEG + Improved Synthesis | TurnGraphRetriever ($k=5$) | Robust Regex + Calendar | **51.88%** (69/133) | **7.32%** (3/41) | **71.74%** (66/92) | 83.33% (5/6) |

### Factorial Analysis of Variance

$$\begin{aligned}
\Delta_{\text{Synthesis}} &= \text{Accuracy}(\text{Arm B}) - \text{Accuracy}(\text{Arm A}) = 51.88\% - 46.62\% = \mathbf{+5.26\%} \\
\Delta_{\text{Graph}} &= \text{Accuracy}(\text{Arm C}) - \text{Accuracy}(\text{Arm A}) = 46.62\% - 46.62\% = \mathbf{0.00\%} \\
\Delta_{\text{Interaction}} &= (\text{Arm D} - \text{Arm A}) - (\Delta_{\text{Synthesis}} + \Delta_{\text{Graph}}) = 5.26\% - (5.26\% + 0.00\%) = \mathbf{0.00\%}
\end{aligned}$$

---

## 3. Detailed Question-Level Forensics

### 3.1 What Drove the +7 Net Question Gain (Arm A → Arm B)?
The improvement from 62 to 69 questions was driven by **12 recoveries** and **5 regressions**:

#### Recovered Questions (12 Questions):
1. `71017276`: Recovered from Pydantic parsing exception; regex recovered "4 weeks ago".
2. `gpt4_8279ba02`: Recovered from "Could not reliably synthesize"; calendar suggested "10 days ago".
3. `gpt4_468eb063`: Recovered from formatting markdown wrap; extracted "9 days ago".
4. `5e1b23de`: Recovered from malformed JSON; extracted "3 months ago".
5. `b46e15ee` (Cat H): Recovered "Walk for Hunger" charity event.
6. `gpt4_e414231f`: Recovered 4 days duration.
7. `gpt4_468eb064`: Recovered date calculation.
8. `c8090214`: Recovered exact day calculation.
9. `gpt4_483dd43c`: Recovered relative day comparison.
10. `c9f37c46`: Recovered open-mic attendance timeline.
11. `gpt4_2f56ae70` (Cat H): Recovered book discussion relative timing.
12. `gpt4_1a1dc16d` (Cat H): Recovered multi-session sequence.

#### Regressed Questions (5 Questions):
1. `gpt4_4929293a`: Over-cautious abstention (treated cousin's wedding as undated).
2. `0db4c65d`: Unhandled nested escape character in JSON string.
3. `gpt4_85da3956`: False positive in contradiction engine (treated "just got back" as conflicting with "three weeks ago").
4. `gpt4_78cf46a3`: Inverse chronological deduction error.
5. `gpt4_2f584639`: Unhandled JSON formatting error.

---

## 4. Why Did the Temporal Event Graph Have Zero Net Effect?

Our forensic audits in `PHASE2B2_GRAPH_QUALITY_REPORT.md` and `PHASE2B2_FAILURE_ANALYSIS.md` explain the exact structural mechanisms:

1. **The Disjoint Event Bottleneck:**
   - Multi-session temporal reasoning questions compare events from unrelated domains (e.g. "Did Event A in Session 12 happen before Event B in Session 38?").
   - These events share **no common entities**.
   - Because 1-hop graph traversal expands only along entity edges from the top seed chunks, the graph **never traverses to the second session**.
2. **MMR Seed Dominance:**
   - Even when an edge existed between dated events, dense vector search initialized from query text matches the seed session with multiple high-scoring chunks.
   - Under $k=5$, the seed session consumes 3–4 slots, pushing any graph-discovered distant session chunk below the top-$k$ cutoff.
3. **Redundant Clique Structure ($\mathcal{O}(N^2)$ Pairwise Explosion):**
   - The graph builder generated 15,774 edges by comparing every dated event against every other dated event.
   - 100% of the edges were symmetric inverse pairs (`OCCURRED_BEFORE` and `OCCURRED_AFTER`).
   - This dense connectivity produced a complete tournament DAG where dated nodes had degrees up to 86.
   - During retrieval, this high connectivity generated high graph branching noise rather than targeted chronological stepping.

---

## 5. Architectural Recommendation

| Improvement Direction | Empirical Status | Recommendation |
|---|---|---|
| **Synthesis / JSON Parsing Improvements** | $+5.26\%$ verified gain across benchmark | **APPROVED FOR PRODUCTION INTEGRATION** into core answering pipeline. |
| **Temporal Event Graph Prototype (Phase 2B.1)** | $0.00\%$ net gain, 15,774 redundant edges | **REJECTED FOR PRODUCTION INTEGRATION** in its current form. Do not merge. |
| **Hierarchical Episodic Diversity (Phase 2A.1)** | $+6$ Category H recoveries, but caused $-11$ regressions on factual questions | **REJECTED AS GLOBAL DEFAULT**. Requires query-intent routing. |

### Path Forward for Phase 3:
1. **Promote Arm B Improvements to Production:**
   - Retain robust JSON regex parsing, calendar calculation fallback, and refined abstention heuristics in `src/answering.py`.
2. **Redesign Graph Structure Before Re-evaluating TEG:**
   - Replace $\mathcal{O}(N^2)$ all-pairs edges with **Sequential Interval Ordering** ($E_i \xrightarrow{\text{NEXT\_EVENT}} E_{i+1}$), reducing edges by $\sim 95\%$.
   - Remove redundant inverse edges (`OCCURRED_AFTER`).
   - Implement **Graph-Forced Inclusion**: When a temporal comparison query is identified, force the retrieval budget to allocate chunks evenly across both compared events, bypassing dense similarity pruning.

