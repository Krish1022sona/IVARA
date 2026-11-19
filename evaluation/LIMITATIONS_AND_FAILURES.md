# IVARA — Failure Analysis, Negative Findings & Architectural Boundaries

**Documenting Operational Limits, Empirical Failure Modes, and Non-Claims with Academic Rigor**

---

## 1. Empirical Failure Mode Taxonomy

Analysis of the non-passing instances in the 500-question LongMemEval-S benchmark partitions remaining errors into four primary root causes:

### 1.1 Multi-Session Dispersion Across $\ge 3$ Sessions (41.2% of remaining errors)
- **Observed Behavior:** On questions requiring factual synthesis across 3, 4, or 5 distinct conversation sessions (e.g. tracking a project status that evolved over 4 separate weeks), the system frequently retrieves evidence from only 2 sessions.
- **Root Cause:** To maintain sub-8-second latency and prevent context-window overflow on local LLMs, retrieval is capped at top-$k=5$ chunks ($W=3$ turns, ~1,500 token budget). While Session-Aware MMR penalizes single-session clustering ($\le 2$ chunks per session), 5 slots are insufficient when each session requires 2–3 turns for coherent context.
- **System Impact:** Relevant turns in the 3rd or 4th session are not included in the generator prompt, leading to incomplete answers or conservative abstention.
- **Targeted Solution:** Dynamic $k$-scaling based on detected session dispersion and hierarchical session-level abstract summaries.

### 1.2 Complex Calendar Arithmetic & Relative Temporal Intervals (24.5% of remaining errors)
- **Observed Behavior:** On queries involving multi-step calendar arithmetic (e.g., *"How many weeks elapsed between the project launch and Maya's trip?"*, *"What was the date of the meeting two days after the conference?"*), the generative model occasionally produces incorrect numerical dates despite having the correct raw anchor dates in evidence.
- **Root Cause:** Standard instruction-tuned language models (`gpt-oss`) struggle with month-day boundary calculations and leap years when reasoning in natural language without calling an external calculator tool. While IVARA's deterministic TACC engine handles explicit day offsets, complex multi-anchor arithmetic relies partly on the LLM decoder.

### 1.3 Implicit Preference Supersession Without Temporal Anchors (18.1% of remaining errors)
- **Observed Behavior:** When a user updates a preference implicitly across conversation sessions without using temporal cues (*"I got a mechanical keyboard"* followed 3 weeks later by *"I love my low-profile chiclet keyboard"* without mentioning dates), the system may report both or fail to establish chronological precedence.
- **Root Cause:** Temporal supersession relies on monotonic timestamp extraction or explicit temporal markers (*"now"*, *"previously"*, *"in 2026"*). When dialogue turns lack explicit temporal expressions, the system must rely purely on conversational turn sequence identifiers.

### 1.4 Abstract Entity Ambiguity & Cross-Session Coreference (16.2% of remaining errors)
- **Observed Behavior:** When conversations refer to entities by pronouns (*"he"*, *"the project"*, *"our team"*) across session boundaries, entity extraction can fail to link the pronoun to the antecedent established 10 sessions prior.

---

## 2. Documented Negative Findings

Scientific integrity requires publishing what did **not** work to save future researchers from repeating unproductive directions:

### 2.1 Zero Marginal Gain from Temporal Event Graph (TEG) Traversal
In a controlled 4-arm ablation across 133 frozen temporal questions:
- Arm A (Flat Baseline): **46.62%**
- Arm C (Temporal Event Graph Traversal Only): **46.62%** (+0.00 pp gain)
- Arm B (Flat MMR + Calendar Synthesis): **51.88%** (+5.26 pp gain)
- Arm D (Temporal Graph + Calendar Synthesis): **51.88%** (+5.26 pp gain)

*Finding:* Multi-hop graph traversal over temporal event nodes provided **0.00 percentage points of additional accuracy** over flat retrieval. Deterministic calendar delta calculation and chronological context assembly drove 100% of the temporal reasoning improvement. Graph traversal added latency and memory overhead without measurable accuracy benefits.

### 2.2 Unbounded Knowledge Graph Traversal Degrades Retrieval Precision
In open information extraction experiments, allowing unbounded knowledge graph expansion created dense hub entities (e.g. common concepts connecting to 40+ nodes). During retrieval, traversing these hubs flooded the top-$k$ candidate pool with topically irrelevant turns, decreasing answerable accuracy on held-out evaluation from 43.33% to 40.00% (-3.33 pp) and increasing latency by 3x. Bounded hub suppression ($\le 10$ edges per hub) was required to stabilize graph retrieval.

---

## 3. Explicit Non-Claims & Architectural Boundaries

To avoid misleading claims common in commercial AI marketing, IVARA explicitly defines its operational boundaries:

1. **No Claim of "Zero Hallucinations":** IVARA achieves high precision on negative control questions (86.67% honest abstention), but it does not eliminate hallucinations entirely. If retrieved context snippets contain noisy or contradictory information, the generative decoder may still synthesize erroneous conclusions.
2. **Citation Validation is Identifier-Based, Not Semantic NLI:** IVARA's citation validator verifies that referenced memory keys exist within the retrieved context chunks (context-key inclusion). It does **not** perform proposition-level natural language inference (NLI) to mathematically prove semantic entailment between each sentence and its cited source.
3. **Unimplemented Proposals:** Theoretical ideas such as *InterMemory Attention (IMA)* are documented in research notes as future conceptual directions; they are not implemented in the codebase.
4. **Hardware Requirements:** Local 14B model inference (`qwen2.5:14b`) requires a minimum of 16 GB unified system memory or 12 GB GPU VRAM. Deterministic mode and CPU sentence embeddings run on standard hardware (8 GB RAM).
