# IVARA — Demonstration Guide
## Step-by-Step Evaluator Walkthrough & Live Demonstration Script

This guide provides exact instructions for demonstrating IVARA's evidence-grounded personal memory capabilities during academic evaluation.

---

### Demonstration Overview

The demonstration uses a self-contained, isolated test environment containing clean autobiographical seed data across three simulated conversation sessions. It proves 5 core architectural capabilities:

1. **Basic Personal Memory Recall:** Point-fact retrieval grounded in conversational memory.
2. **Cross-Session Synthesis Recall:** Gathering details dispersed across independent conversation sessions.
3. **Historical Preference Changes:** Resolving outdated historical states (2021) versus current preferences (2026).
4. **Multi-Hop Relational Reasoning:** Traversing the knowledge graph to link entities across hops.
5. **Honest Abstention on Unrecorded Facts:** 100% precision abstention with zero hallucinations.

---

### 1. Demonstration Commands

You can run the demonstration in two modes:

#### Option A: Guaranteed Deterministic Mode (Recommended for Presentations)
*Completely offline-safe, instant execution, zero external network or LLM flakiness.*
```bash
python demo.py
```

#### Option B: Live Model Inference Mode
*Executes real-time generation using your local Ollama instance (`ollama serve`).*
```bash
python demo.py --live
```

---

### 2. Step-by-Step Scenario Walkthrough & Expected Outputs

#### Scenario 1: Basic Personal Memory Recall
- **Query:** `"What company does Maya work for, and what is her role?"`
- **Goal:** Verify that IVARA retrieves primary autobiographical attributes from past conversation turns.
- **Expected Status:** `[STATUS: ANSWERED]`
- **Expected Output:**
  ```text
  Answer: Maya Lin is a Distributed Systems Architect based in Seattle and currently works at CloudMatrix Inc.
  Evidence Citations: ['m101'] (100% verified authentic provenance)
  ```
- **Evaluator Talking Point:**
  > "IVARA locates the exact memory record `m101` created during Session 01, answering both employer and role with zero extrapolation."

---

#### Scenario 2: Cross-Session Synthesis Recall
- **Query:** `"What project is Maya building, and who is her tech lead?"`
- **Goal:** Synthesize information dispersed across Session 01 (project name) and Session 02 (collaborators).
- **Expected Status:** `[STATUS: ANSWERED]`
- **Expected Output:**
  ```text
  Answer: Maya is architecting the VectorFlow project at CloudMatrix Inc, and her tech lead is David Thorne.
  Evidence Citations: ['m101', 'm102'] (100% verified authentic provenance)
  ```
- **Evaluator Talking Point:**
  > "Notice the dual citations `['m101', 'm102']`. IVARA's adaptive evidence selector allocated budget across multiple sessions to unite facts that were never mentioned together in any single conversation."

---

#### Scenario 3: Historical Preference Changes (Temporal Resolution)
- **Query:** `"What keyboard does Maya currently use, and how did her preference change?"`
- **Goal:** Demonstrate temporal resolution distinguishing an outdated 2021 preference from a 2026 preference.
- **Expected Status:** `[STATUS: ANSWERED]`
- **Expected Output:**
  ```text
  Answer: While Maya previously used clicky mechanical keyboards in 2021, in 2026 she switched to low-profile chiclet keyboards.
  Evidence Citations: ['m101'] (100% verified authentic provenance)
  ```
- **Evaluator Talking Point:**
  > "Conventional vector RAG often returns the 2021 preference because 'mechanical keyboard' has strong semantic weight. IVARA's temporal resolution inspects timestamps and correctly identifies the 2026 low-profile keyboard as the active preference."

---

#### Scenario 4: Multi-Hop Relational Reasoning (Knowledge Graph)
- **Query:** `"What previous company did the tech lead of Maya's project work for?"`
- **Goal:** Traverse multi-hop knowledge graph: `Maya` $\to$ `VectorFlow` $\to$ `David Thorne` $\to$ `Datacore`.
- **Expected Status:** `[STATUS: ANSWERED]`
- **Expected Output:**
  ```text
  Answer: Maya's tech lead on VectorFlow, David Thorne, previously led the storage team at Datacore.
  Evidence Citations: ['m102'] (100% verified authentic provenance)
  Grounded Boundaries: Rejected unsupported memory citations: ed4.
  ```
- **Evaluator Talking Point:**
  > "This question requires linking three separate entities. IVARA's graph retrieval traverses from Maya to VectorFlow to David Thorne to Datacore. Furthermore, the graph safeguard post-validation automatically stripped raw edge `ed4`, ensuring only verified memory evidence is cited."

---

#### Scenario 5: Honest Abstention on Unrecorded Facts
- **Query:** `"In which birth city was Maya born, and what is her passport number?"`
- **Goal:** Verify that the system cleanly abstains on unknown or out-of-domain personal facts.
- **Expected Status:** `[STATUS: INSUFFICIENT_EVIDENCE]`
- **Expected Output:**
  ```text
  Answer: I do not have sufficient recorded memory to answer where Maya was born.
  Evidence Citations: [] (No ungrounded citations emitted)
  Grounded Boundaries: Birthplace and early childhood details are not recorded in available memory sessions.
  ```
- **Evaluator Talking Point:**
  > "When asked an unrecorded biographical question, standard LLMs hallucinate a plausible city or number. IVARA's evidence sufficiency gate catches the ungrounded query before generation, producing an honest abstention with zero hallucinated citations."

---

### 3. Interactive CLI Demonstration

To demonstrate live interactive chat with real-time persistence:

```bash
# Launch interactive chat
python main.py chat

# Type a personal statement:
> "I just started learning Rust for systems programming."

# Press Ctrl+C or type 'exit' to quit.

# Ask about it via direct CLI:
python main.py ask "What programming language did I recently start learning?" --adaptive
```
*Expected answer:* IVARA immediately recalls Rust and cites the persistent turn ID from the previous session.

---

### 4. Demonstration Troubleshooting

- **Error: `ModuleNotFoundError`**
  - *Fix:* Ensure virtual environment is activated: `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1`.
- **Ollama Connection Refused (Live Mode):**
  - *Fix:* Ensure Ollama is running in a terminal: `ollama serve`. Alternatively, run without `--live` to execute deterministic mode.

