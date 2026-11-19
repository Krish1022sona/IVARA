# IVARA Retrieval Failure Mode Diagnostic Report

**Source Data:** Evaluated on all 358 development questions from `longmemeval_s_cleaned.json` via out-of-fold candidate pool (`top_k=25`).  
**Artifact Integrity:** Read-only analysis. Evaluated retriever left completely unchanged. Zero modifications to benchmarks, models, or dependencies.

---

## 1. Candidate Ceiling and Failure Instance Summary

Across the 358 development questions, there are **698 gold session requirements** (some questions require 2 to 5 distinct sessions to fully answer).

| Metric | Count | Percentage |
|---|:---:|:---:|
| **Total Gold Session Instances (Denominator)** | **698** | 100.00% |
| **Covered Gold Sessions in Top-25 Candidates** | **655** | **93.84%** |
| **Missed Gold Sessions in Top-25 Candidates** | **43** | **6.16%** |
| **Questions with Full Evidence (All Gold Sessions in Top 25)** | **321 / 358** | **89.66%** |
| **Questions with Partial Evidence (At least 1 Gold Session in Top 25)** | **353 / 358** | **98.60%** |
| **Questions with Zero Evidence (Full Misses in Top 25)** | **5 / 358** | **1.40%** |

---

## 2. Classification of Retrieval Failure Modes

Every one of the 43 missed gold session instances falls into one of three distinct, identifiable failure modes:

### Failure Mode 1: Multi-Session Aggregation Dispersion (32 instances, 74.4% of misses)
- **Description:** Questions that require aggregating instances across 3 to 5 separate conversations (e.g. counting total projects led, road trip hours, doctor visits, furniture assembled, baking events).
- **Observed Behavior:** The primary hybrid retriever successfully retrieves 1, 2, or 3 of the relevant sessions into the top 10, but the remaining 1 or 2 sessions lack sufficiently concentrated keyword density or dense similarity to beat competitive distractors for the remaining top-25 slots.
- **Representative Examples:**
  - `6d550036` (*"How many projects have I led or am currently leading?"*):
    - Retrieved in Top 25: `answer_ec904b3c_1`, `answer_ec904b3c_2`, `answer_ec904b3c_4`
    - Missed: `answer_ec904b3c_3`
  - `aae3761f` (*"How many hours in total did I spend driving to my three road trip destinations combined?"*):
    - Retrieved in Top 25: `answer_526354c8_1`, `answer_526354c8_2`
    - Missed: `answer_526354c8_3`
  - `gpt4_f2262a51` (*"How many different doctors did I visit?"*):
    - Retrieved in Top 25: `answer_55a6940c_2`, `answer_55a6940c_3`
    - Missed: `answer_55a6940c_1`
  - `88432d0a` (*"How many times did I bake something in the past two weeks?"*):
    - Retrieved in Top 25: `answer_733e443a_1`, `answer_733e443a_2`, `answer_733e443a_3`
    - Missed: `answer_733e443a_4`

### Failure Mode 2: Implicit / Conversational Semantic Gap (6 instances, 14.0% of misses)
- **Description:** Queries expressing subjective states, preferences, or conversational small talk without unique named entities or distinctive keywords.
- **Observed Behavior:** Generic multi-turn distractors from ShareGPT/UltraChat dialogue noise receive slightly higher generic similarity scores than the user's specific conversational memory.
- **Representative Examples:**
  - `75832dbd` (*"Can you recommend some recent publications or conferences that I might find interesting?"*):
    - Missed gold session: `answer_d87a6ef8` (Swamped by synthetic academic dialogue distractors).
  - `d6233ab6` (*"I've been feeling nostalgic lately. Do you think it would be a good idea to attend my high school reunion?"*):
    - Missed gold session: `answer_b0fac439` (Swamped by general reunion discussions).

### Failure Mode 3: Temporal Anchor & Metadata Disconnect (5 instances, 11.6% of misses)
- **Description:** Queries with relative time anchors (e.g. *"10 days ago"*, *"in the past few months"*) where the temporal anchor is encoded in the session date metadata rather than the spoken text turn.
- **Observed Behavior:** The dense bi-encoder evaluates chunk text alone without conditioning on the elapsed days between query timestamp and session recording timestamp.
- **Representative Examples:**
  - `gpt4_8279ba03` (*"What kitchen appliance did I buy 10 days ago?"*):
    - Missed gold session: `answer_56521e66_1` (The text mentions buying an appliance, but does not contain the word "10 days ago").

---

## 3. Implications for Graph Expansion

Because **74.4% of all missed evidence** belongs to multi-session queries where 1 to 3 gold sessions were **already retrieved**, 1-hop graph expansion from already-retrieved seed entities and sessions has the structural potential to recover the missing gold sessions without needing to broaden the initial dense search space.
