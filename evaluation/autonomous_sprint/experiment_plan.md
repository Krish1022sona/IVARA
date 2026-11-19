# Experimental Plan: Controlled Evaluation of Retrieval, Filtering, and Temporal Prototypes

**Sprint Deliverable:** Experimental Design & Protocol  
**Target Evaluation Tier:** 100-Question Controlled Diagnostic Sample  
**Target Benchmark:** LongMemEval-S Cleaned  
**Branch:** `research/evaluation-baseline`  

---

## 1. Experimental Motivation & Core Hypotheses

Our Step 7 pilot evaluation (60 questions across Configs A–D) and Step 8–9 audits established three critical empirical facts:
1. **Ontology Guidance is Highly Effective:** Adding ontology query and entity expansion in Config-B produced a **+8.33 pp gain** over Config-A (71.67% vs 63.33%), achieving 100% precision on unanswerable questions.
2. **Unfiltered Graph Expansion Causes Distractors:** Adding 1-hop conversational neighbor chunks in Config-C caused **6 regressed questions** (-3.33 pp drop to 68.33%), including a 32-sentence repetitive hallucination loop in `gpt4_c27434e8_abs`.
3. **Temporal Structuring Rescues Reasoning:** Explicit temporal claim resolution in Config-D recovered 3 of the regressed questions (returning to 71.67%), broke repetition loops, and reduced output token latency.

To systematically isolate the causal drivers and prepare the optimal production architecture, this experiment tests three formal hypotheses:

### Hypothesis A (Pure Seed Retrieval)
> *Ontology retrieval without graph expansion retains the observed retrieval accuracy gains (+8.33 pp) while completely avoiding conversational distractor injections.*

### Hypothesis B (Relevance-Filtered Graph Expansion)
> *Filtering graph traversal edges by query relevance (`min_edge_relevance >= 0.50` and lexical query overlap) prevents dialogue noise and chit-chat turns from entering the context, eliminating repetition loops while preserving multi-turn conversational connections.*

### Hypothesis C (Direct Seed Temporal Extraction)
> *Extracting temporal claims and resolving supersession directly from retrieved top-$k$ seed chunks (`direct_seed_temporal=True`) captures the temporal accuracy and conciseness benefits of Config-D without requiring background graph expansion or graph-store persistence.*

---

## 2. Experimental Arms & Controlled Variables

To isolate each variable, all experimental arms will execute under strictly frozen conditions:
- **Dataset Sample:** 100 questions stratified across all 6 LongMemEval tasks (knowledge-update, temporal-reasoning, multi-session, single-session-preference, single-session-user, single-session-assistant).
- **Answering Model:** Ollama `gpt-oss:latest` (20.91B MoE, temperature 0, context budget 4,096 tokens, max output 1,024 tokens).
- **Evaluation Judge:** Ollama `VladimirGav/gemma4-26b-16GB-VRAM:latest` with the validated Step 8 strict evaluation rubric.
- **Retrieval Budget:** $k=5$ chunks, max context 1,500 tokens, dense weight 0.55, BM25 weight 0.30, ontology weight 0.15.

### Matrix of Experimental Arms

| Arm ID | Configuration Name | Ingestion & Retrieval Strategy | Edge Filter Threshold | Temporal Extraction Source | Primary Variable Tested |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm 1 (Control)** | `Config-A` | Dense (MiniLM) + Lexical (BM25) | None (No Graph) | Disabled | Baseline |
| **Arm 2 (Hypothesis A)** | `Config-B` | Dense + BM25 + Ontology Guidance | None (No Graph) | Disabled | Pure Ontology Guidance |
| **Arm 3 (Control)** | `Config-C` | Ontology + 1-Hop Graph Expansion | `0.0` (Unfiltered) | Disabled | Impact of Raw Graph Expansion |
| **Arm 4 (Hypothesis B)** | `Config-C-Filtered` | Ontology + 1-Hop Graph Expansion | `0.50` (Relevance Filtered) | Disabled | Edge Relevance Filtering |
| **Arm 5 (Hypothesis C)** | `Config-B-Temporal` | Dense + BM25 + Ontology Guidance | None (No Graph) | Top-$k$ Seed Chunks | Direct Seed Temporal Resolution |
| **Arm 6 (Control)** | `Config-D` | Ontology + 1-Hop Graph + Temporal | `0.0` (Unfiltered) | Graph-Wide Claims | Full Existing Architecture |

---

## 3. Quantitative Metrics & Evaluation Criteria

1. **Overall Accuracy (%)**: Accuracy judged by local Gemma judge across all 100 questions (Answerable + Abstentions).
2. **Answerable Accuracy (%)**: Accuracy on the ~84 answerable questions.
3. **Abstention Precision (%)**: Correct abstention rate on the ~16 unanswerable questions.
4. **Graph Regression Rate**: Count and percentage of questions where an arm scores False while `Config-B` scored True (target: 0 regressions).
5. **Repetition Incident Count**: Count of predictions exhibiting $\ge 3$ repeated sentence spans or decoding loops (target: 0).
6. **Execution Latency**: Mean and median end-to-end question latency (ms) and answering stage latency.

---

## 4. Success and Failure Criteria

### Hypothesis A Success Criteria
- `Config-B` maintains overall accuracy $\ge 70.0\%$ with 0 graph-induced false abstentions.
- Confirms that for single-point factual lookup, pure seed retrieval is strictly superior to unfiltered graph expansion.

### Hypothesis B Success Criteria
- `Config-C-Filtered` reduces the 6 graph regressions observed in `Config-C` by $\ge 66\%$ (at least 4 of 6 recovered).
- Completely prevents the repetition loop seen in `gpt4_c27434e8_abs`.
- Retains multi-turn connection wins (`0a34ad58`, `195a1a1b`).

### Hypothesis C Success Criteria
- `Config-B-Temporal` achieves overall accuracy $\ge 71.67\%$, matching or exceeding `Config-D`.
- Achieves 100% precision on temporal abstentions.
- Eliminates graph construction runtime overhead ($t_{\text{graph}} \to 0$ ms).

---

## 5. Execution Prerequisites & Guardrails

1. **Safety Constraints:**
   - Run only when explicitly authorized by the developer.
   - Do NOT run full 500-question benchmark until the 100-question diagnostic sample confirms the winning architecture.
   - All results must be output to a clean directory: `evaluation/data/longmemeval/runs/diagnostic_experiment_v1/`.
2. **Deterministic Reproducibility:**
   - Fixed random seed 42 for sample selection.
   - Manifest recording dataset SHA-256 and git commit hash.
   - Zero modifications to production files (`data/memories.json`, `data/entities.json`).
