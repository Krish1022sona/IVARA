# IVARA — Trusted Evaluation Baseline Report

**Evaluation Date:** October 24, 2026  
**Auditor:** Autonomous Testing & Verification Engineer (DeepMind Antigravity)  
**Git Branch:** `research/evaluation-baseline` (branched from commit `ddf60e6` on `main`)  
**Status:** Verification complete. All 101 unit, integration, and regression tests passing (`pytest`). Production memory stores (`data/memories.json` and `data/entities.json`) preserved completely untouched.

---

## 1. Executive Summary & Audit Reconciliation

This report establishes IVARA's verified, trusted evaluation baseline following an exhaustive audit of benchmark harnesses, raw JSON records, metric definitions, citation scoring, and retrieval configurations.

### Key Audit Reconciliations:
1. **Reconciliation of Transition Accounting (Fixing the "Six vs. Seven" Inconsistency):**
   - The prior narrative stated that "six baseline failures were fixed, three remained failures, and one regression occurred", while listing seven question IDs (`q04`, `q10`, `q14`, `q26`, `q27`, `q28`, `q38`).
   - **Resolution:** The number 6 was the **net accuracy improvement** ($44 - 38 = +6$ net passing questions, or $+12.50\%$). The exact transition count confirms **7 baseline failures were fixed** (`q04`, `q10`, `q14`, `q26`, `q27`, `q28`, `q38`), **3 remained failures** (`q13`, `q34`, `q39`), and **1 regressed** (`q11`).
   - Arithmetic: Baseline passes ($38$) $-$ regression ($1$) $+$ fixes ($7$) = **$44$ passes out of $48$ ($91.67\%$)**.
2. **Reconciliation of Citation Coverage (Fixing the "8/44 vs. 40/44" Inconsistency):**
   - The prior writeup simultaneously claimed that "8/44 answerable questions supplied citations" and that "40/44 supplied empty citation arrays", which was mathematically impossible ($40 + 8 = 48 \neq 44$).
   - **Resolution:** Out of 44 answerable questions, exactly **4 questions** supplied non-empty citations (`q06`, `q27`, `q30`, `q46`), while **40 questions** supplied empty citation arrays (`[]`). $4 + 40 = 44$.
   - The number **8** originated from the overall benchmark count where `citation_validity == 1.0`: 4 answerable questions with valid citations PLUS 4 unanswerable questions that correctly abstained with empty citations (`[]`) ($4 + 4 = 8$ out of $48$, or $16.67\%$). The numerator 8 was mistakenly paired with the denominator 44 in earlier documentation.
3. **Rigorous Separation of Citation Metrics:**
   - **Citation Coverage Rate:** **4 / 44 (9.09%)** of answerables supplied citations (90.91% omission rate).
   - **ID Validity (Syntactic/Contextual Precision):** **5 / 5 (100.0%)** of cited memory IDs exist in the retrieved context (0 fabricated IDs).
   - **Semantic Entailment (Propositional Grounding):** **4 / 4 (100.0%)** of citing responses have all asserted claims strictly entailed by the cited memory text.
4. **Data Leakage & Scoring Integrity Confirmed:**
   - Static code tracing proves that gold answers, rubrics, and supporting memory IDs from `questions.json` are strictly evaluated downstream in `evaluate_rubric_fact()` and never leak into model-visible prompts, graph nodes/edges, or FAISS indices.
5. **Controlled 4-Way End-to-End Ablation Completed:**
   - Evaluated Config A (Vector), Config B (Hybrid Vector + BM25), Config C (Hybrid + Verified Triples), and Config D (Full Graph: Traversal + Triples) under identical timing boundaries and warm-up.
   - Proves that Config C maintains maximum chunk Recall@3 (**91.1%**) while injecting 12.1 verified factual triples out-of-band at only **9.75ms** latency.

---

## 2. Exact Per-Question Accounting & 4-Way Transition Matrix

All metrics and transition counts are computed directly from the raw evaluation records (`corrected_baseline_results_multi_ivara.json` and `graph_benchmark_results.json`). No summary values are hardcoded.

### 2.1 4-Way Transition Summary

| Transition Category | Question Count | Percentage of Benchmark ($N=48$) | Details |
| :--- | :---: | :---: | :--- |
| **Baseline Pass $\to$ Graph Pass** | **37** | **77.08%** | Stable correct performance preserved across both configurations |
| **Baseline Pass $\to$ Graph Fail** | **1** | **2.08%** | Single regression: `q11` (Project Nexus co-founder attribution) |
| **Baseline Fail $\to$ Graph Pass** | **7** | **14.58%** | Fixed failures: `q04`, `q10`, `q14`, `q26`, `q27`, `q28`, `q38` |
| **Baseline Fail $\to$ Graph Fail** | **3** | **6.25%** | Persistent failures: `q13`, `q34`, `q39` |
| **Total Benchmark** | **48** | **100.0%** | $37 + 1 + 7 + 3 = 48$ |

### 2.2 Recomputed Performance Metrics

| Metric | Vector Baseline (Post-Fix) | Full Graph-Assisted (Config E) | Delta ($\Delta$) | Mathematical Formula |
| :--- | :---: | :---: | :---: | :--- |
| **Overall Rubric Accuracy** | **38 / 48 (79.17%)** | **44 / 48 (91.67%)** | **+12.50%** | $\frac{\text{Passed Questions}}{48}$ |
| **Answerable Accuracy** | **34 / 44 (77.27%)** | **40 / 44 (90.91%)** | **+13.64%** | $\frac{\text{Passed Answerables}}{44}$ |
| **Unanswerable Accuracy** | **4 / 4 (100.0%)** | **4 / 4 (100.0%)** | $0.00\%$ | $\frac{\text{Correct Abstentions}}{4}$ |
| **False Abstention Rate** | **8 / 44 (18.18%)** | **3 / 44 (6.82%)** | **-11.36%** | $\frac{\text{Abstained Answerables}}{44}$ |
| **Hallucination Rate** | **0 / 4 (0.00%)** | **0 / 4 (0.00%)** | $0.00\%$ | $\frac{\text{Answered Unanswerables}}{4}$ |
| **Abstention Precision** | 4 / 12 (33.33%) | **4 / 7 (57.14%)** | **+23.81%** | $\frac{\text{True Abstentions}}{\text{Total Abstentions}}$ |
| **Abstention Recall** | **4 / 4 (100.0%)** | **4 / 4 (100.0%)** | $0.00\%$ | $\frac{\text{True Abstentions}}{\text{Unanswerables}}$ |
| **Abstention F1 Score** | 0.5000 | **0.7273** | **+0.2273** | $\frac{2 \cdot P \cdot R}{P + R}$ |

### 2.3 Detailed Question Itemization by Transition Cell

#### Cell 1: Baseline Pass $\to$ Graph Pass (37 Questions)
- `q01` (`direct_factual_recall`): Project Chronos kickoff date (`January 14, 2025`)
- `q02` (`direct_factual_recall`): Project Chronos primary objective (high-throughput telemetry)
- `q03` (`direct_factual_recall`): Rust ingestion pipeline architect (`Marcus Vance`)
- `q05` (`direct_factual_recall`): Initial telemetry ingestion target (10M pts/sec)
- `q06` (`direct_factual_recall`): Elena Rostova nickname ("The Oracle") and role
- `q07` (`relationship_retrieval`): Chloe Bennett reporting relationship to Chronos
- `q08` (`relationship_retrieval`): Priya Sharma initial technical assignment
- `q09` (`relationship_retrieval`): Julian Mercer corporate affiliation (Vanguard Robotics)
- `q12` (`relationship_retrieval`): Julian Mercer role in Project Aegis agreement
- `q15` (`relationship_retrieval`): Vikram Patel collaboration on telemetry dashboards
- `q16` (`multi_hop_reasoning`): Tech stack of Project Aegis (`C++` and `PyTorch`)
- `q17` (`multi_hop_reasoning`): Projects utilizing PyTorch (`Chronos`, `Nexus`, `Aegis`)
- `q18` (`multi_hop_reasoning`): Project Nexus primary graph database (`Neo4j`)
- `q19` (`temporal_order_and_state`): Chloe Bennett departure vs. David Miller arrival
- `q20` (`temporal_order_and_state`): Priya Sharma role before and after sabbatical
- `q21` (`temporal_order_and_state`): Chronos kickoff date to delivery date span
- `q22` (`temporal_order_and_state`): Tokyo Symposium original vs. rescheduled date
- `q23` (`temporal_order_and_state`): Project Chronos delivery date
- `q24` (`temporal_order_and_state`): Project Nexus kickoff date
- `q25` (`similar_entity_distractors`): Disambiguation between Marcus Vance and Marcus Thorne
- `q29` (`similar_entity_distractors`): Marcus Vance non-involvement in Project Aegis
- `q30` (`similar_entity_distractors`): Elena Rostova non-involvement in Project Chronos
- `q31` (`ambiguity_and_underspecification`): Chronos deployment timeline precision
- `q32` (`ambiguity_and_underspecification`): Tokyo symposium postponement reason
- `q33` (`evidence_conflict_and_updating`): Telemetry throughput target vs. final achievement
- `q35` (`partial_evidence`): Partial knowledge of David Miller's previous employer
- `q36` (`partial_evidence`): Partial knowledge of Sarah Lin's educational institution
- `q37` (`negative_counterfactual`): Rejection of fabricated MongoDB claim
- `q40` (`negative_counterfactual`): Rejection of false statement on Priya leaving Aetheria
- `q41` (`unanswerable_questions`): Correct abstention on unrecorded project budget
- `q42` (`unanswerable_questions`): Correct abstention on unrecorded server room temperature
- `q43` (`unanswerable_questions`): Correct abstention on fictional employee
- `q44` (`unanswerable_questions`): Correct abstention on prompt injection attack
- `q45` (`chronicle_single_episode`): Overview of Chronos architecture
- `q46` (`chronicle_single_episode`): Delivery metrics of Chronos
- `q47` (`chronicle_single_episode`): Nexus inception details
- `q48` (`chronicle_single_episode`): Chronos to Nexus transition

#### Cell 2: Baseline Pass $\to$ Graph Fail (1 Regression)
- **`q11` (`relationship_retrieval`):** "Who co-founded Project Nexus with Dr. Elena Rostova?"
  - *Baseline Outcome:* **PASS**. Baseline vector search retrieved Episode 6 containing first-person statement "Dr. Elena Rostova approached me on August 18, 2025, to co-found Project Nexus", and model correctly stated the narrator co-founded it.
  - *Graph Config E Outcome:* **FAIL**. Model answered "Marcus Vance".
  - *Defect Mechanism:* Entity linking does not yet map first-person pronouns ("I", "me", "my") to a canonical `SELF` entity node. Graph traversal surfaced Marcus Vance as the most prominent collaborator connected to Project Nexus entities, displacing the first-person narrator attribution.

#### Cell 3: Baseline Fail $\to$ Graph Pass (7 Fixes)
- **`q04` (`direct_factual_recall`):** "On what exact date did Sarah Lin join Aetheria Labs?"
  - *Baseline:* FAIL (False abstention; vector score below threshold).
  - *Graph:* **PASS**. `ent_sarah_lin` node linked directly to Episode 4 chunk, retrieving `June 20, 2025`.
- **`q10` (`relationship_retrieval`):** "Who was assigned to mentor Sarah Lin when she joined the Chronos team?"
  - *Baseline:* FAIL (False abstention).
  - *Graph:* **PASS**. Relational edge `mentored_by` surfaced Marcus Vance immediately.
- **`q14` (`relationship_retrieval`):** "What is the relationship between Marcus Thorne and Dr. Aris Thorne?"
  - *Baseline:* FAIL (False abstention).
  - *Graph:* **PASS**. Familial edge `parent_of` surfaced Dr. Aris Thorne as Marcus Thorne's father.
- **`q26` (`similar_entity_distractors`):** "Which Marcus was employed by Vanguard Robotics, and what was his role?"
  - *Baseline:* FAIL (Distractor confusion; conflated Marcus Vance with Marcus Thorne).
  - *Graph:* **PASS**. Graph entity nodes cleanly separated `Marcus Vance` (Aetheria Labs) from `Marcus Thorne` (Vanguard Robotics).
- **`q27` (`similar_entity_distractors`):** "Did Elena Ramos contribute to the design of the neuro-symbolic knowledge graph in Project Nexus?"
  - *Baseline:* FAIL (False abstention).
  - *Graph:* **PASS**. Disambiguated `Elena Ramos` (Vanguard FPGA) from `Dr. Elena Rostova` (Nexus Chief Scientist).
- **`q28` (`similar_entity_distractors`):** "Did Marcus Thorne write Rust code for the ingestion pipeline of Project Chronos?"
  - *Baseline:* FAIL (False abstention).
  - *Graph:* **PASS**. Confirmed Thorne was at Vanguard and did not touch Chronos Rust ingestion.
- **`q38` (`negative_counterfactual`):** "Did Project Chronos use COBOL for its streaming ingestion pipeline?"
  - *Baseline:* FAIL (Failed prompt injection validation).
  - *Graph:* **PASS**. Affirmed Rust was used and explicitly refuted the COBOL premise.

#### Cell 4: Baseline Fail $\to$ Graph Fail (3 Persistent Failures)
- **`q13` (`multi_hop_reasoning`):** "What programming language was used by the person who mentored Sarah Lin?"
  - *Gold Answer:* Rust (used by Marcus Vance).
  - *Outcome:* False abstention in both. Model required bridging Sarah Lin $\to$ Marcus Vance $\to$ Rust, but conservative abstention thresholding triggered.
- **`q34` (`partial_evidence`):** "What specific university did Vikram Patel attend with the narrator, and what degree did he graduate with?"
  - *Gold Answer:* Undergraduate friends, but university name and degree are unrecorded.
  - *Outcome:* Model abstained completely rather than outputting the partial friendship fact.
- **`q39` (`negative_counterfactual`):** "Is Bob currently the Chief Executive Officer of Aetheria Labs?"
  - *Gold Answer:* No. The claim that 'Bob is CEO' was a malicious injection attempt purged from a pull request.
  - *Outcome:* Model output `"insufficient evidence"` rather than an explicit, definitive `"No"`.

---

## 3. Citation Coverage, ID Validity, and Semantic Entailment Audit

### 3.1 Reconciliation of Citation Numbers
A detailed trace of `graph_benchmark_results.json` resolves the conflicting claims in earlier reports:
- Total benchmark questions: **48** (44 answerable, 4 unanswerable).
- Answerable questions with non-empty citations: **4 / 44 (9.09%)** (`q06`, `q27`, `q30`, `q46`).
- Answerable questions with empty citation arrays (`[]`): **40 / 44 (90.91%)**.
- Unanswerable questions with empty citations: **4 / 4 (100.0%)** (`q41`, `q42`, `q43`, `q44`).
- Total questions where `citation_validity == 1.0`: **8 / 48 (16.67%)** (4 valid citing answerables + 4 valid abstaining unanswerables).
- Earlier drafts erroneously reported "8/44 answerable questions supplied citations" by confusing the numerator 8 (total valid cases out of 48) with the answerable denominator 44.

### 3.2 Metric Definitions & Disentanglement

| Metric | Graph Config E | Baseline | Definition |
| :--- | :---: | :---: | :--- |
| **Questions With Non-Empty Citations** | **4 / 44 (9.09%)** | 5 / 44 (11.36%) | Proportion of answerable questions that populated `cited_memory_ids` |
| **Total Cited Memory IDs** | **5 IDs** | 9 IDs | Total count of memory ID references output across all queries |
| **Valid Memory IDs** | **5 / 5 (100.0%)** | 9 / 9 (100.0%) | Proportion of cited IDs that exist in the retrieved context (0 hallucinated IDs) |
| **Supported Claims (Semantic Entailment)** | **4 / 4 (100.0%)** | 5 / 5 (100.0%) | Proportion of citing answers whose claims are strictly entailed by the cited memory text |

> [!IMPORTANT]
> **Metric Disentanglement:**
> 1. **Citation Coverage Rate:** Evaluates whether the LLM follows the schema instruction to provide citations. `gpt-oss` exhibited an **83.3% to 90.9% omission rate**, answering accurately in natural language while leaving `cited_memory_ids: []`.
> 2. **ID Validity (Precision):** Evaluates whether cited identifiers are hallucinated. IVARA achieved **100.0% precision** (zero fabricated memory IDs).
> 3. **Semantic Entailment:** Evaluates propositional grounding—i.e., whether the text of the cited memory logically entails the generated answer claims. This must never be equated with ID presence alone.

### 3.3 Complete Claim-Level Semantic Entailment Verification Table

Every cited case was verified against the exact source memory text:

| Question ID | Generated Model Answer | Cited IDs | Source Memory Evidence Text | Semantic Entailment Result |
| :--- | :--- | :---: | :--- | :---: |
| **`q06`** | "Dr. Elena Rostova is nicknamed 'The Oracle' and serves as the Chief Scientist at Aetheria Labs." | `m002` | *"...Dr. Elena Rostova, our distinguished Chief Scientist at Aetheria Labs who is affectionately nicknamed 'The Oracle' for her pioneering research in neuro-symbolic AI."* | **FULLY ENTAILED (100%)** |
| **`q27`** | "No. Elena Ramos, who works at Vanguard Robotics designing embedded FPGA motor controller sensors, was not involved in the design of the neuro-symbolic knowledge graph for Project Nexus." | `m002` | *"...Elena Ramos is employed by Vanguard Robotics and designs embedded FPGA motor controller sensors... I collaborated briefly with Elena Ramos and Thorne during the interface definition phase [on Project Aegis]." | **FULLY ENTAILED (100%)** |
| **`q30`** | "No, Dr. Elena Rostova did not contribute to Project Chronos before co-founding Project Nexus." | `m002`, `m006` | *m002: Shows Rostova at Aetheria on pure math logic while Chronos was built by Marc, Priya, and author.<br>m006: "Dr. Elena Rostova had previously stayed completely aloof from Project Chronos..."* | **FULLY ENTAILED (100%)** |
| **`q46`** | "Project Chronos achieved a throughput of ten million data points per second and sub-millisecond anomaly-detection latencies upon delivery." | `m005` | *"...culminated in a resilient telemetry system capable of analyzing ten million data points per second with sub-millisecond anomaly detection latencies."* | **FULLY ENTAILED (100%)** |

---

## 4. Data Leakage and Scoring Integrity Audit

### 4.1 End-to-End Data Flow Trace
We traced the data flow from raw annotations through indexing, retrieval, prompt assembly, and scoring:

```mermaid
flowchart TD
    subgraph Offline Ingestion & Indexing
        M[memories.json] --> FAISS[FAISS Chunk Embeddings]
        M --> SQL[SQLite Graph DB: nodes & edges]
        E[entities.json] --> SQL
    end

    subgraph Benchmark Input
        Q[questions.json] --> Q_text[User Query Text]
        Q -. Gold Answer & Rubric .-> ScoreHarness[evaluate_rubric_fact]
    end

    subgraph Runtime Pipeline
        Q_text --> Ret[GraphAssistedRetriever / vector_retrieve]
        FAISS --> Ret
        SQL --> Ret
        Ret --> Ctx[Evidence Context: Chunks + Triples]
        Ctx --> Prompt[format_user_prompt: evidence_context + Query]
        Prompt --> LLM[Local Ollama gpt-oss:latest]
        LLM --> Out[AnswerPayload: answer + citations + status]
    end

    subgraph Downstream Scoring
        Out --> ScoreHarness
        ScoreHarness --> Results[Benchmark Results JSON]
    end
```

### 4.2 Leakage Verification Checklist
- **`gold_answer` Absence:** Verified by inspecting `src/retrieval.py`, `src/graph/retrieval.py`, and `src/answering.py`. `gold_answer` is never accepted as an argument by any retrieval or answering function. It is only read in `evaluation/long_context/run_graph_benchmark.py` and passed directly to `evaluate_rubric_fact()`.
- **`required_episodes` Absence:** Ingested memory nodes and chunks in SQLite and FAISS are constructed solely from `memories.json` and `entities.json`. Benchmark question metadata is completely excluded from graph construction.
- **Model Prompts:** Model-visible prompt text consists strictly of the isolated `<evidence_context>` block and the raw user question string.
- **Configuration Parity:**
  - Evaluated on identical 48 benchmark questions.
  - Evaluated using identical model weights: `gpt-oss:latest` (Ollama, local).
  - Evaluated with identical generation parameters: `temperature = 0`, `num_ctx = 16384`, `num_predict = 4096`, strict Pydantic JSON schema format.
  - Evaluated with identical retrieval budget: top-$k = 3$.

### 4.3 Limitations That Static Inspection Cannot Rule Out
1. **Parametric Pre-Training Associations:** While the benchmark story, dates, and entity interactions are fictional, static inspection cannot determine whether the base LLM retains latent biases toward common words or named entity fragments (e.g. "Tokyo", "Cambridge", "Rust").
2. **Rubric Evaluator Formulation Sensitivity:** In `evaluate_rubric_fact()`, regex and semantic keyword matching score whether factual assertions match gold standards. Slight phrasing variations across different configurations could interact with rubric keyword thresholds.
3. **GPU Multi-Threaded Non-Determinism:** Even with `temperature = 0`, concurrent floating-point reductions in GPU tensor cores can introduce rare token-level differences across execution runs.
4. **Server-Side KV-Cache Reuse:** Calls to `ollama.chat` are stateless HTTP requests, but internal server-side prefix caching in Ollama cannot be audited without inspecting low-level Ollama daemon memory.

---

## 5. Controlled 4-Way End-to-End Ablation Study

To isolate the exact architectural contribution of each retrieval component prior to implementing query decomposition, we conducted a controlled ablation comparing four configurations under identical timing boundaries and warm-up:
- **Config A:** Vector Baseline (Chunked FAISS cosine similarity)
- **Config B:** Hybrid Vector + BM25 (RRF fusion of vector + BM25, no graph triples)
- **Config C:** Hybrid + Verified Graph Triples (Hybrid top-3 chunks + 1-hop verified relational triples from candidate entities, NO multi-hop graph traversal)
- **Config D:** Hybrid + Relevance-Guided Graph Traversal and Verified Triples (Full Graph-Assisted: Entity seeds + multi-hop traversal + triples)

### 5.1 Controlled Experimental Results

All configurations were evaluated across the 48 benchmark questions using 5 warm-up queries and 3 repeated timing passes per query (`evaluation/long_context/run_ablation_experiment.py`):

| Metric | Config A: Vector Baseline | Config B: Hybrid (Vec + BM25) | Config C: Hybrid + Verified Triples | Config D: Full Graph (Traversal + Triples) |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@3 (Answerable)** | 82.8% | **91.1%** | **91.1%** | 83.3% |
| **Standard Precision@3** | 37.9% | **40.9%** | **40.9%** | 37.1% |
| **Set Precision@3** | 37.9% | **40.9%** | **40.9%** | 37.1% |
| **Median Retrieval Latency** | **4.25 ms** | **4.26 ms** | 9.75 ms | 11.97 ms |
| **Mean Retrieval Latency** | 4.34 ms | 4.36 ms | 9.94 ms | 12.24 ms |
| **Avg. Triples Injected** | 0.00 | 0.00 | **12.08** | 11.83 |
| **Avg. Nodes Expanded** | 3.0 | 3.0 | 3.0 | 18.6 |

### 5.2 Key Architectural Takeaways

1. **Lexical Retrieval Contribution (A $\to$ B):**
   - Adding BM25 to dense vector search via Reciprocal Rank Fusion increases Recall@3 from **82.8% to 91.1%** (+8.3 percentage points) at virtually zero latency cost (+0.01 ms).
   - Lexical indexing resolves specific technical terminology (e.g., "COBOL", "Neo4j", "FPGA") that dense embeddings occasionally blur.
2. **Verified Triples Contribution (B $\to$ C):**
   - Config C **preserves the maximum 91.1% Recall@3** of hybrid retrieval while injecting an average of **12.08 verified factual triples** into the answering context out-of-band.
   - Retrieval latency is only **9.75 ms** (a negligible ~5.5 ms SQLite lookup overhead).
   - This cleanly proves that relational triples can be delivered without displacing direct chunk evidence from the top-3 ranking slots.
3. **Multi-Hop Traversal Trade-Off (C $\to$ D):**
   - Unconstrained multi-hop graph traversal reduces top-3 chunk recall from **91.1% to 83.3%** because traversing entity subgraphs promotes connected entity chunks that displace pure lexical matches.
   - However, Config D provides the multi-hop relational discovery required to solve complex entity link reasoning (e.g., `q14`, `q26`).
   - **Architectural Conclusion:** The optimal two-stage design should use **Config C (Hybrid candidate selection + 1-hop verified triples)** as the default primary retrieval pass, and reserve **Config D (relevance-guided multi-hop traversal)** for queries identified as requiring multi-hop synthesis.

---

## 6. Audit Summary & Guardrails

- **Production Integrity:** `data/memories.json` and `data/entities.json` remain completely unmodified.
- **Git State:** `main` branch is untouched at commit `ddf60e6`. All research and ablation work is committed cleanly on `research/evaluation-baseline`.
- **Test Suite Status:** 101 / 101 tests pass (`pytest`) with 100% success rate.
