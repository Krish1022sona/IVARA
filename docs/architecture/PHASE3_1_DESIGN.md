# IVARA — Phase 3.1: Evidence-Validated Session Routing Design

**Date:** 2026-11-17  
**Author:** Antigravity (DeepMind Team)  
**Branch:** `experiment/evidence-validated-routing-phase3-1`  
**Baseline Commit:** `55a273bff20fa0eec1b2d361e8a03b217d147abe`  
**Status:** Approved Architectural Specification  

---

## 1. Executive Summary & Forensic Motivation

In Phase 3, IVARA proved that **subquery clause decomposition** resolves the multi-session retrieval starvation of Category H, lifting Category H accuracy to an all-time benchmark peak of **19.51% (8 / 41)** and achieving **100.00% (6 / 6)** abstention accuracy.

However, Phase 3 suffered from **15 regressions** (predominantly in Non-Category H, which fell from 71.74% to 60.87%). Our deep forensic audit revealed the exact structural root cause:
- **Syntax-Driven Routing Flaw:** Phase 3 used lexical and grammatical syntax (`between X and Y`, `which happened first`) to trigger `MULTI_SESSION`.
- **The Intra-Session Reality:** In LongMemEval, many comparative queries in Non-Category H discuss both events within the **exact same conversational session** (or closely linked sub-sessions).
- **The Consequence:** Capping the gold session at 2 chunks and forcing chunks from unrelated background sessions diluted contiguous turn depth and interleaved distractor text, causing the LLM to overlook dates that were previously accessible under Flat Dense MMR.

### Core Objective of Phase 3.1
Replace syntax-driven retrieval routing with **Evidence-Validated Session Routing**:
1. Never assume that comparative language implies different sessions.
2. Retrieve candidate sessions independently for each event clause.
3. Classify evidence relationship based on actual retrieved support:
   - `SAME_SESSION`: Evidence indicates both events are discussed in the same session. Route to Flat MMR to preserve contiguous dialogue depth.
   - `DIFFERENT_SESSIONS`: Evidence confirms events reside in distinct sessions with no strong joint session. Route to balanced multi-session retrieval.
   - `AMBIGUOUS`: Low-confidence or conflicting session evidence. Safely fall back to Flat MMR.
4. Guarantee that single-session depth is never capped merely because of comparative phrasing.

---

## 2. Architecture Overview

```
                                  User Query
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │   Query Clause Decomposer    │
                       │   (Extract Clauses A and B)  │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │ Independent Candidate Search │
                       │ Score Sessions for Event A   │
                       │ Score Sessions for Event B   │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │ Evidence-Validated Decider   │
                       │ Evaluate Joint vs Disjoint   │
                       └──────────────┬───────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
    [ SAME_SESSION ]       [ DIFFERENT_SESSIONS ]         [ AMBIGUOUS ]
  Joint Support >= 0.45    Top A != Top B, Joint < 0.40   Confidence < 0.35
            │                         │                         │
            ▼                         ▼                         ▼
     Flat Dense MMR            Balanced Multi-Session     Safe Flat MMR
   Preserves Contiguous        Allocates Chunks Across    Fallback
    Turn Depth (<= 4 chunks)    Both Disjoint Sessions    (Single Session Depth)
            │                         │                         │
            └────────────────►        │        ◄────────────────┘
                                      ▼
                            Bounded Context Window
                                      │
                                      ▼
                            Arm B Synthesis Engine
                                      │
                                      ▼
                                Final Answer
```

---

## 3. Evidence-Validated Decision Rules

For a query comparing Event A and Event B:

### Step 1: Independent Scoring
Let $\text{Hits}_A = \{(S, \text{score}_A(S))\}$ and $\text{Hits}_B = \{(S, \text{score}_B(S))\}$.
Top sessions: $S_A^* = \arg\max \text{score}_A(S)$, $S_B^* = \arg\max \text{score}_B(S)$.

### Step 2: Joint Evidence Evaluation
For any session $S$ present in candidates for both events:
$$\text{JointSupport}(S) = \min(\text{score}_A(S), \text{score}_B(S))$$

### Step 3: Classification

#### A. `SAME_SESSION`
Triggered if:
1. $S_A^* == S_B^*$ (the exact same session is top rank for both events with score $\ge 0.40$), OR
2. $\max_S \text{JointSupport}(S) \ge 0.42$ (a single session contains strong textual evidence for both events).

**Retrieval Action:**
- Route to Flat Dense MMR.
- Set `max_chunks_per_session = 3` (or 4).
- Never cap the gold session; allow full contiguous conversational context.

#### B. `DIFFERENT_SESSIONS`
Triggered if:
1. $S_A^* \ne S_B^*$ AND $\text{score}_A(S_A^*) \ge 0.40$ AND $\text{score}_B(S_B^*) \ge 0.40$, AND
2. $\max_S \text{JointSupport}(S) < 0.40$ (no single session adequately explains both events).

**Retrieval Action:**
- Allocate 2 chunks from $S_A^*$ and 2 chunks from $S_B^*$.
- Do NOT add random ultrachat distractor chunks; fill remaining budget only with verified candidate turns.
- Sort chronologically by session date and turn index.

#### C. `AMBIGUOUS`
Triggered if top scores are low ($< 0.35$) or evidence is conflicting.
**Retrieval Action:** Safe fallback to Flat MMR.

---

## 4. Acceptance Criteria Gate

1. **Overall Accuracy:** Must strictly exceed Phase 2B.2 Arm B baseline of **51.88%** ($> 69 / 133$).
2. **Category H:** Must strictly exceed Phase 3's record of **8 / 41 (19.51%)**.
3. **Non-Category H:** Must NOT fall below Phase 2B.2 Arm B baseline of **66 / 92 (71.74%)**.
4. **Zero Unexplained Regressions:** Every regression must be audited and explained.
5. **No Production Mutation:** Production database hashes must remain bit-exact.

