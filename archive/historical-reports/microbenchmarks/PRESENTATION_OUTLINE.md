# IVARA — Presentation Outline
## 11-Slide Academic Presentation Structure
**Target Duration:** 10–12 Minutes  
**Audience:** Academic Committee, Faculty Evaluators & AI Researchers

---

### Slide 1: Title & Executive Hook
- **Slide Title:** IVARA: Evidence-Grounded Autobiographical AI Memory
- **Subtitle:** Solving Context Loss, Context Overflows & Memory Fragmentation in Personal Agents
- **Presenter Information:** IVARA Research Team | November 2026
- **Key Visual:** System Logo + Architectural Split (Persistent Episodic vs Semantic Graph)
- **Speaker Notes (1 min):**
  > "Good morning, committee members. Today's conversational AI assistants suffer from a profound paradox: they can discuss quantum mechanics, but they forget what project you worked on last week. The root cause isn't model size; it's an architectural flaw in how AI memory is designed. Today, we introduce IVARA—a two-layer, evidence-grounded personal memory system that solves context loss and ungrounded hallucinations without sacrificing local privacy."

---

### Slide 2: The Core Problem — Three Fatal Flaws in Existing RAG
- **Slide Title:** Why Conventional Memory Architectures Fail
- **Core Vulnerabilities Identified:**
  1. *Heuristic Extraction Rejection:* Pre-filters discard **48.1% of valid user turns**, permanently losing personal history.
  2. *Schema Rigidity:* Closed-world ontologies reject natural biographical relationships (51 of 54 stored memories lacked edges).
  3. *Context Overflows & Empty Responses:* In unbudgeted retrieval systems, 39.6% of queries crashed on context overflows and 75.8% produced empty string completions.
- **Key Visual:** Diagram showing user turns being rejected by heuristics and prompt windows overflowing.
- **Speaker Notes (1 min):**
  > "When we audited state-of-the-art personal AI pipelines, we uncovered two critical flaws: first, heuristic memory filters permanently reject up to 48% of user messages before saving. Second, without strict context budgeting, large retrieval blocks blow past the model's token limits, causing silent context overflows and empty answers."

---

### Slide 3: The IVARA Architectural Solution
- **Slide Title:** Two-Layer Coordinated Memory Architecture
- **Layer 1: Persistent Episodic Backbone (`conversations.db`):**
  - 100% turn retention in SQLite with FTS5 lexical indexing.
  - Zero information loss; monotonic sequence numbers and session tracking.
- **Layer 2: Open-World Semantic & Knowledge Graph (`memory_graph.db`):**
  - Schema-flexible relationship resolution.
  - Rigorous edge provenance: verbatim evidence spans and confidence scores.
- **Key Visual:** System data flow diagram from user message to twin stores.
- **Speaker Notes (1 min):**
  > "IVARA solves this through two complementary layers. Layer 1 is the Episodic Backbone: every conversation turn is immediately persisted in SQLite with FTS5 indexing. No turns are discarded. Layer 2 is an Open-World Knowledge Graph that extracts entities and biographical relationships with explicit verbatim evidence spans."

---

### Slide 4: Adaptive Multi-Source Retrieval Pipeline
- **Slide Title:** Deterministic Routing & Session-Aware MMR
- **Key Features:**
  - *8-Category Deterministic Router:* Classifies queries in **0.15 ms** (single fact, temporal, multi-hop, unanswerable).
  - *Contiguous Window Expansion:* Expands episodic hits by $\pm 1$ turn to maintain conversational flow.
  - *Session-Aware MMR Re-ranking:* Pools 15 chunks and applies Maximal Marginal Relevance ($\lambda = 0.25$) to prevent token monopolization by single sessions.
- **Key Visual:** The 8 routing categories flowchart and MMR re-ranking diagram.
- **Speaker Notes (1 min):**
  > "Rather than using fixed-k vector search, IVARA employs a sub-millisecond Adaptive Query Router. It identifies whether a query needs a single fact, a temporal timeline, or a relational bridge. To prevent a single conversation from monopolizing the context, our Session-Aware MMR re-ranking diversifies evidence across distinct sessions."

---

### Slide 5: Graph Safeguards & Temporal Invalidation
- **Slide Title:** Preventing Graph Hallucinations
- **Three Non-Negotiable Safeguards:**
  1. *Seed Entity Relevance:* Graph traversal only expands from entities explicitly present in the query or retrieved turns.
  2. *Provenance Verification:* Rejects graph triples lacking verified memory citations.
  3. *Temporal Invalidation:* Strictly suppresses unversioned current graph edges during historical-state queries.
- **Key Visual:** Diagram showing an unanchored graph edge being pruned by the safeguard gate.
- **Speaker Notes (1 min):**
  > "Knowledge graphs are notoriously prone to distractor injection. IVARA enforces three strict graph safeguards: seed entity relevance, evidence verification, and temporal invalidation. If a user asks about a past job, current unversioned graph relationships are strictly suppressed."

---

### Slide 6: Evidence-Grounded Answering & Context Budgeting
- **Slide Title:** Calibrated Generation & Context Stability
- **Core Mechanics:**
  - *Pre-Generation Sufficiency Gate:* Evaluates whether evidence is sufficient, partial, or absent before prompting the model.
  - *Strict Context Budgeting:* Allocates 1,500 tokens for episodic evidence, 250 for temporal claims, and `num_ctx = 8192` in Ollama.
  - *Query-Scoped Contradiction Filtering:* Only flags contradictions directly relevant to query entities.
  - *Citation Post-Validation:* Strips ungrounded keys; guarantees zero citations on abstention.
- **Key Visual:** Context token budget allocation chart (1,500 + 250 + 250).
- **Speaker Notes (1 min):**
  > "To ensure absolute stability, IVARA enforces strict token budgeting: 1,500 tokens for episodic dialogue, 250 for temporal claims, operating inside an 8,192-token context. If evidence is absent, the system produces an honest abstention with zero hallucinated citations."

---

### Slide 7: Evaluation Methodology: Official LongMemEval Benchmark
- **Slide Title:** Rigorous 500-Question Standardized Evaluation
- **Benchmark Specifications:**
  - Standardized **LongMemEval-S** benchmark (500 questions, 6 challenge categories).
  - Official independent LLM judge (`qwen2.5:14b`, `temperature=0.0`) using the verbatim official evaluation prompt (`xiaowu0162/LongMemEval`).
  - Corrected local regex heuristic for offline continuous integration.
  - Zero question/answer leakage; production databases strictly isolated.
- **Key Visual:** LongMemEval dataset category distribution pie chart.
- **Speaker Notes (1 min):**
  > "To evaluate IVARA objectively, we used the full 500-question LongMemEval-S benchmark. We ran the official evaluation using an independent Qwen 2.5 14B model as the LLM judge under zero temperature, ensuring unbiased, reproducible scoring."

---

### Slide 8: Full Benchmark Results — 3.85x Accuracy Gain
- **Slide Title:** LongMemEval-S Official Results (500 Questions)
- **Metrics Comparison Table:**
  - Official LLM-Judge Accuracy: **10.60% $\to$ 40.80% (+30.20%, 3.85x jump)**
  - Task-Averaged Accuracy: **11.60% $\to$ 45.10% (+33.50%)**
  - Answerable Accuracy: **6.17% $\to$ 37.87% (6.14x jump)**
  - Abstention Accuracy: **80.00% $\to$ 86.67% (26/30 correct)**
  - Context Overflow Crashes: **198 $\to$ 0 (100% eliminated)**
  - Empty Completions: **379 $\to$ 0 (100% eliminated)**
  - Retrieval Recall@5: **92.13% $\to$ 94.26%** | Coverage: **72.89% $\to$ 82.49%**
- **Key Visual:** Grouped bar chart showing Baseline vs Integrated Production across all 6 challenge categories.
- **Speaker Notes (1.5 min):**
  > "The results are decisive. Under the official LLM judge, overall accuracy leaped from 10.60% to 40.80%—a 3.85-fold increase. Answerable accuracy jumped over six-fold, from 6.17% to 37.87%. By eliminating context overflows and empty responses, every single question received a valid, evidence-grounded evaluation."

---

### Slide 9: Task Category Breakdown & Grounding Efficacy
- **Slide Title:** Performance Across Memory Challenge Categories
- **Results Highlights:**
  - `single-session-user`: **34.29% $\to$ 82.86% (+48.57%)**
  - `single-session-assistant`: **8.93% $\to$ 58.93% (+50.00%)**
  - `knowledge-update`: **7.69% $\to$ 43.59% (+35.90%)**
  - `multi-session`: **7.52% $\to$ 31.58% (+24.06%)**
  - `temporal-reasoning`: **4.51% $\to$ 20.30% (+15.79%)**
  - Citation Validity: **100.0%** across all answers.
- **Key Visual:** Radar chart showing performance gains across all 6 tasks.
- **Speaker Notes (1.5 min):**
  > "Looking at individual tasks, single-session personal recall surged to 82.9%, assistant recall to 58.9%, and knowledge updates to 43.6%. Furthermore, across all answers generated, citation validity was 100%, with zero ungrounded memory keys emitted."

---

### Slide 10: System Performance, Latency & Resource Footprint
- **Slide Title:** Hardware Efficiency & Local Latency
- **Key Metrics:**
  - Mean Answering Latency: **9.92 seconds** (local Ollama on RTX 5080)
  - SQLite Query Latency (p50): **0.030 ms**
  - FAISS Vector Search (p50): **0.002 ms**
  - Peak VRAM Consumption: **12,899 MB** (3.4 GB safety margin on 16 GB GPU)
  - Base RAM Footprint: **~50 MB**
- **Key Visual:** Latency distribution and VRAM headroom breakdown.
- **Speaker Notes (1 min):**
  > "Because IVARA relies on lightweight SQLite, FTS5, and FAISS, all retrieval operations finish in under a millisecond. End-to-end generation averages 9.92 seconds on commodity GPU hardware, operating with 3.4 GB of VRAM safety headroom."

---

### Slide 11: Conclusion & Submission Readiness
- **Slide Title:** Summary of Contributions
- **Deliverables:**
  - Production-ready two-layer memory architecture with 100% episodic retention.
  - **462 / 462 passing tests** in automated regression suite.
  - Verified 40.80% official LLM-judge benchmark on LongMemEval-S (3.85x gain).
  - Deterministic and live demonstration scripts (`python demo.py`).
  - Bit-for-bit preserved production database files.
- **Final Verdict:** Audited, verified, reproducible, and ready for deployment.
- **Speaker Notes (1 min):**
  > "In summary, IVARA demonstrates that personal AI memory cannot rely on lossy extraction heuristics or unbudgeted context windows. By pairing persistent episodic logging with open-world graphs, session-aware MMR, and context budgeting, IVARA delivers reliable, grounded personal memory. The system is verified across 462 passing tests and reproducible benchmarks. Thank you, and we welcome your questions."
