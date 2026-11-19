# IVARA — Controlled Ablation Studies & Architectural Isolations

**Authoritative Record of Component-Level Causality & Negative Experimental Results**

---

## 1. 4-Arm Temporal Reasoning & Knowledge Graph Ablation

### Context & Research Hypothesis
Initial experiments showed an increase in temporal reasoning performance from 46.62% to 51.88% (+5.26 pp). To rigorously isolate whether this gain stemmed from multi-hop **Temporal Event Graph (TEG)** traversal or from **deterministic calendar arithmetic and synthesis improvements**, a strictly controlled 4-arm ablation was executed across all 133 frozen temporal questions in LongMemEval-S.

### Experimental Configuration
All arms were evaluated under strictly identical conditions:
- **Dataset:** 133 Frozen Temporal Questions from LongMemEval-S Cleaned.
- **Generator Model:** `gpt-oss` (via local Ollama daemon).
- **Judge Model:** Official `qwen2.5:14b` judge at `temperature=0.0` (greedy decoding).
- **Prompting & Scoring:** Official LongMemEval judge prompt and scoring rubric.

### 4-Arm Definitions:
- **Arm A (Flat Baseline):** Standard dense + lexical hybrid retrieval over conversational chunks, without temporal graph and without calendar arithmetic.
- **Arm B (Flat Retrieval + Calendar Synthesis):** Hybrid retrieval with deterministic calendar arithmetic and chronological evidence formatting, without temporal graph traversal.
- **Arm C (Temporal Event Graph Only):** Hybrid retrieval augmented with Temporal Event Graph traversal (15,774 temporal edges), without calendar synthesis.
- **Arm D (Unified: Temporal Graph + Calendar Synthesis):** Full pipeline combining Temporal Event Graph traversal with deterministic calendar synthesis.

### Empirical Results

| Experimental Arm | Architecture | Correct Answers ($N=133$) | Accuracy (%) | Marginal Gain vs. Arm A |
| :--- | :--- | :---: | :---: | :---: |
| **Arm A** | Flat Baseline | 62 / 133 | 46.62% | Reference |
| **Arm C** | Flat + Temporal Event Graph Only | 62 / 133 | 46.62% | **0.00 pp (0 net questions)** |
| **Arm B** | Flat + Calendar Synthesis Only | 69 / 133 | 51.88% | **+5.26 pp (+7 net questions)** |
| **Arm D** | Graph + Calendar Synthesis | 69 / 133 | 51.88% | **+5.26 pp (+7 net questions)** |

### Critical Scientific Findings
1. **Zero Marginal Graph Contribution:** Arm C produced identical question-by-question outcomes to Arm A (62/133, 0.00 pp difference). Graph traversal over temporal event nodes provided zero net benefit over flat retrieval on this benchmark.
2. **Calendar Synthesis Drove All Improvement:** Arm B and Arm D achieved identical performance (69/133, 51.88%). The entire +5.26 percentage point gain was caused by deterministic calendar delta calculation and chronological ordering before generation.
3. **Graph Quality & Redundancy Audit:** Analysis of the constructed Temporal Event Graph (15,774 edges across 133 questions) revealed that 92.4% of valid relational triples were already directly captured within the retrieved conversational turn chunks ($W=3, O=1$). High-degree hub nodes introduced topical noise rather than novel grounding.
4. **Implication for Production:** The production architecture (Candidate B) intentionally relies on Flat MMR + Calendar Synthesis, eliminating graph traversal overhead and latency penalties without sacrificing accuracy.

---

## 2. Ontology Concept Recognition & Graph Traversal Ablation

A 3-way ablation was conducted on diagnostic ($N=30$) and held-out ($N=30$) slices to measure the individual impact of ontology-based alias expansion and relational graph traversal:
- **Config A (`B-Chunked` Baseline):** Dense FAISS (70%) + Lexical BM25 (30%) on turn chunks ($W=3, O=1$).
- **Config B (`B-Chunked + Ontology`):** Concept recognition, alias expansion, and entity-mention re-ranking without graph expansion.
- **Config C (`B-Chunked + Graph`):** Ingests chunks into `SQLiteGraphStore` with relevance-guided multi-hop traversal.

### Diagnostic & Held-Out Results

| Split | Metric | Config A (Chunked Baseline) | Config B (+ Ontology) | Config C (+ Graph Traversal) |
| :--- | :--- | :---: | :---: | :---: |
| **Diagnostic ($N=30$)** | **Answerable Accuracy** | 38.46% (10/26) | **42.31% (11/26)** (+3.85 pp) | **42.31% (11/26)** (+3.85 pp) |
| | **Overall Accuracy** | 33.33% (10/30) | **36.67% (11/30)** | **36.67% (11/30)** |
| | **Retrieval Recall@5** | 92.31% (24/26) | 92.31% (24/26) | 92.31% (24/26) |
| | **Mean Latency** | 25.71 s | **11.94 s** (Fastest) | 35.70 s |
| **Held-Out ($N=30$)** | **Answerable Accuracy** | **50.00% (13/26)** | **50.00% (13/26)** | 46.15% (12/26) (-3.85 pp) |
| | **Overall Accuracy** | **43.33% (13/30)** | **43.33% (13/30)** | 40.00% (12/30) |
| | **Retrieval Recall@5** | **96.15% (25/26)** | **96.15% (25/26)** | 92.31% (24/26) (-3.84 pp) |

### Key Insights:
- In held-out evaluation, full graph traversal slightly decreased accuracy (40.00% vs. 43.33%) and increased latency by 3x (35.7s vs 11.9s) due to overconnected hub entity dilution.
- Bounded ontology alias expansion provided stable retrieval indexing without the graph density overhead.

---

## 3. Conversational Chunking vs. Whole-Session Embedding Ablation

### The Session Dilution Problem
In preliminary baseline RAG evaluations, entire multi-turn conversation sessions (averaging 1,500–3,000 words across 10–20 dialogue turns) were embedded as single vectors.

### Experimental Comparison

| Strategy | Unit | Prompt Token Count | Recall@5 | Failure Pattern |
| :--- | :--- | :---: | :---: | :--- |
| **Whole-Session Embedding** | Entire Conversation | 5,500–7,000 tokens | 68.4% | Dilution: single needle facts missed by 1 rank; context window overflow. |
| **Turn-Aware Chunking ($W=3, O=1$)** | Sliding Turn Window | 1,200–1,800 tokens | **94.3%** | Concentrated signal: needle turns maintain high vector similarity. |

### Finding:
Turn-Aware Conversational Chunking ($W=3$ turns, overlap $O=1$) resolved context overflow crashes and boosted initial retrieval Recall@5 by +25.9 pp, ensuring relevant facts are not diluted by adjacent conversational filler.

---

## 4. Session Diversity Filtering & Maximal Marginal Relevance (MMR)

When retrieving top-$k$ chunks from chat history, dense conversational clusters often cause 4 out of 5 retrieved slots to be filled by a single verbose session, starving multi-session questions of necessary evidence.

| Method | Diversity Formulation | Constraint | Multi-Session Reasoning Acc | Single-Session Clustering |
| :--- | :--- | :---: | :---: | :---: |
| **Standard Top-$k$** | Pure Cosine Similarity | None | 24.80% (31/125) | High (up to 5 chunks from 1 session) |
| **Flat MMR ($\lambda = 0.70$)** | Cosine - $\lambda \cdot \text{Sim}(c, S)$ | $\le 2$ chunks / session | **36.00% (45/125)** | Bounded ($\le 2$ chunks / session) |

*Conclusion:* Applying Flat MMR with $\lambda = 0.70$ and enforcing a maximum of 2 chunks per session improved Multi-Session Reasoning accuracy by **+11.20 percentage points** across 125 multi-session questions.

---

## 5. Phase 3.1: Evidence-Validated Session Routing (Explicit NO-GO Decision)

### Hypothesis & Routing Mechanism
In Phase 3.1, an adaptive `EvidenceSelector` was introduced to route queries between single-session focused retrieval and multi-session synthesis, applying strict session boundary masks to eliminate cross-session distractors.

### Empirical Results across 133 Temporal Benchmark Questions
- **Phase 1 / Candidate B Flat MMR Baseline:** **51.88% (69/133)**
- **Phase 3.1 Evidence-Validated Router:** **48.12% (64/133)** (-3.76 pp / -5 net questions)

### Discordant Transition Analysis
- **Recoveries (+10 questions):** The router correctly pruned distractors on 10 isolated single-session questions where peripheral sessions had previously crowded the prompt.
- **Regressions (-15 questions):** The router failed on 15 multi-session questions because hard session-level filtering completely pruned peripheral sessions containing subtle chronological anchors or antecedent entity definitions.

### Formal Verdict: Explicit NO-GO
Because the 15 regressions exceeded the 10 recoveries and reduced overall accuracy by -3.76 pp, the architecture was formally rejected for production release. Candidate B's soft Flat MMR diversity scoring was retained instead of hard session masking.
