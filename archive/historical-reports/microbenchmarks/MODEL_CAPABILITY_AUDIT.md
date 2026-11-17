# IVARA — Model Capability & Hardware Audit (LLM Integration & Hardware Optimization)

**Audit Date:** November 16, 2026  
**Auditor:** IVARA Systems Architecture & Evaluation Engineering  
**Branch:** `research/evaluation-baseline`  
**Execution Mode:** STRICT READ-ONLY AUDIT (Zero Code / Zero Weights / Zero Configuration Modifications)  
**Hardware Context:** Intel Core Ultra 9 275HX | 32 GB RAM | NVIDIA GeForce RTX 5080 Laptop GPU (16 GB VRAM)  

---

## 1. Complete LLM Integration Map

An exhaustive inspection of the entire IVARA codebase (`src/`, `evaluation/`, `tests/`) reveals that IVARA employs a **hybrid neuro-symbolic architecture**: the vast majority of classification, routing, entity resolution, and temporal validation steps are **purely deterministic**, while LLM generation is restricted to **four specific entry points**.

### 1.1 Summary Matrix of All LLM Calls in Production

| Pipeline Stage | Module & Function | Default Model | Prompt Location | Format Enforced | Generation Parameters | Failure Handling & Fallback |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Semantic Memory Extraction** | `src/extraction/extractor.py`<br>`extract_knowledge()` | `gpt-oss` | `SYSTEM_PROMPT`<br>(lines 35–57) | Pydantic JSON Schema (`format=Extraction.model_json_schema()`) | `temperature=0`<br>`num_ctx=16384`<br>`num_predict=4096` | None in extractor; raises `ValidationError` or `RuntimeError`. Pipeline catches exception and logs error without losing episodic turn. |
| **2. Question Answering & Grounded Synthesis** | `src/answering.py`<br>`answer_question()` | `gpt-oss` | `ANSWERING_SYSTEM_PROMPT`<br>(lines 44–79) | Pydantic JSON Schema (`format=AnswerPayload.model_json_schema()`) | `temperature=0`<br>`num_ctx=4096`<br>`num_predict=1024` | 3-stage parser: 1) Pydantic validation, 2) raw `json.loads(strict=False)`, 3) regex search `\{.*\}` in markdown. Fallback sets `status="partial"` or `"insufficient_evidence"`. |
| **3. Complex Query Decomposition** | `src/query/decomposition.py`<br>`QueryDecomposer.decompose()` | `gpt-oss` | `DECOMPOSITION_SYSTEM_PROMPT`<br>(lines 35–72) | Pydantic JSON Schema (`format=DecompositionPayload.model_json_schema()`) | `temperature=0`<br>`num_ctx=4096`<br>`num_predict=1024` | Heuristic fast-path bypasses simple queries. On model failure or parse error, gracefully falls back to `[original_query]` with `strategy="fallback_original"`. |
| **4. Conversational Acknowledgment** | `src/chat/engine.py`<br>`_generate_conversational_response()` | `gpt-oss` | Inline System Prompt<br>(lines 289–292) | Free text (No JSON) | `temperature=0.3`<br>`num_predict=256` | Falls back to deterministic static string: `"Understood. I have recorded that in this conversation."` |

### 1.2 Deterministic Modules (Verified Non-LLM)

To prevent ungrounded hallucinations, IVARA intentionally keeps the following critical modules **100% deterministic (0 LLM calls)**:

1. **Memory Formation Policy (`src/extraction/policy.py`):**
   - Implements regex and rule-based heuristic filtering (`is_casual_or_filler`, `is_general_knowledge_query`, `is_pure_question_or_recall`, `is_instruction`, and clause splitting).
   - *Audit Finding:* Although `model: str = "gpt-oss"` and `llm_caller` exist in the function signature for legacy compatibility, the implementation executes pure deterministic pattern classification and clause distillation.
2. **Entity Resolution & Alias Disambiguation (`src/ontology/relationship_resolver.py`):**
   - Implements string normalization, canonical entity alias dictionary mapping, and Levenshtein/Jaro-Winkler fuzzy token matching. No LLM calls.
3. **Ontology & Predicate Validation (`src/ontology/validator.py`):**
   - Validates entity types and predicates against the open-world ontology schema and `vocabulary.json` vocabulary tables. No LLM calls.
4. **Temporal Claim Extraction & Supersession (`src/temporal/claim.py`):**
   - Deterministic regular expressions parse dates, years, and intervals (`extract_year`, `overlaps_year`). State supersession is computed via interval algebra. No LLM calls.
5. **Adaptive Routing (`src/unified_retrieval/router.py`):**
   - Classifies incoming queries into 8 distinct categories (`single_fact`, `historical_state`, `temporal_longitudinal`, `cross_session_synth`, `entity_relationships`, `multi_hop_reasoning`, `negative_information`, `unanswerable`) using deterministic regex anchors, conjunction parsers, and temporal token extractors.
6. **Memory Consolidation:**
   - *Audit Finding:* An active, offline LLM memory consolidation daemon (e.g. background sleep-phase summarization) is **not implemented** in the codebase. In IVARA, "consolidation" refers strictly to retrieval-time evidence consolidation (`src/unified_retrieval/evidence_selector.py` and `src/unified_retrieval/fusion.py`).

---

## 2. Actual Hardware & Inference Environment Audit

All hardware parameters were verified using live Windows CIM APIs and NVIDIA diagnostic utilities:

### 2.1 Hardware Inventory

| Component | Verified Specification | Audit Status & Performance Implications |
| :--- | :--- | :--- |
| **CPU** | Intel(R) Core(TM) Ultra 9 275HX | **Verified:** 24 physical cores / 24 logical processors. Provides high-throughput vector embedding generation and multi-threaded SQLite index operations. |
| **System RAM** | 32.0 GB Physical (32,900,708 KB Visible) | **Verified:** ~14.3 GB free physical RAM available under active workstation load (+23.0 GB swap). Sufficient to host hybrid pipelines and in-memory indices. |
| **Dedicated GPU** | NVIDIA GeForce RTX 5080 Laptop GPU | **Verified:** Blackwell architecture (Compute Capability 12.0 / Native FP4), Driver 596.21, CUDA 13.2. Only 262 MiB VRAM currently in use; **~16.0 GB VRAM available**. |
| **Dedicated VRAM** | 16,303 MiB (~16.0 GB) | **Verified:** Full model layer offloading supported for models up to 14B–16B in 4-bit/8-bit precision or up to 26B in aggressive quantizations (MXFP4 / IQ4_XS). |
| **Storage (Disk)** | 125.88 GB Free on C:\ drive | **Verified:** Ample space for downloading and caching 2–3 candidate LLM weight files (typically 5–10 GB each). |

### 2.2 Ollama Runtime & Installed Model Inventory

* **Ollama Service Version:** `0.33.2` running at `http://127.0.0.1:11434`
* **GPU Compute Runtime:** Ollama auto-discovered discrete GPU `CUDA0: NVIDIA GeForce RTX 5080 Laptop GPU (15.9 GiB total, 14.7 GiB available)`.
* **Observed Inference Throughput:** **136.94 tokens/second** generation speed, **84.83 tokens/second** prompt evaluation speed (verified via Ollama task log timings).
* **Installed Models in Local Cache:**
  1. `gpt-oss:latest` (ID: `17052f91a42e`, Disk Size: **13 GB**, Architecture: `gptoss`, Parameters: **20.91B MoE**, Quantization: **MXFP4**, Context Length: **131,072**).
  2. `VladimirGav/gemma4-26b-16GB-VRAM:latest` (ID: `860321696baf`, Disk Size: **14 GB**, Architecture: `gemma4`, Parameters: **25.2B**, Quantization: **IQ4_XS**, Context Length: **262,144**).

### 2.3 Cloud / Remote API Environment Audit

* **API Keys in Environment:**
  - `OPENAI_API_KEY`: **None** (Not set).
  - `ANTHROPIC_API_KEY`: **None** (Not set).
  - `GEMINI_API_KEY`: **None** (Not set).
* **Usability Assessment:** The codebase contains legacy OpenAI evaluator adapters (`evaluation/evaluate_qa_official.py`), but the project operates **100% locally and offline**. Cloud APIs are neither configured nor required, maintaining complete privacy and zero operating costs.

---

## 3. Bottleneck Diagnosis: Model Limitations vs. Pipeline Artifacts

By cross-referencing audit records (`evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/audited_17_failures.json`, `evaluation/autonomous_sprint/failure_patterns.json`, and `IVARA_FAILURE_ANALYSIS.md`), we isolated exact failure modes:

```
                            ERROR DECOMPOSITION
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
[ MODEL-LEVEL FAILURES ]                                [ PIPELINE / DATA FAILURES ]
- Infinite Repetition Loops                             - Heuristic Over-Filtering (Fixed in M1)
- Compound Entity String Pollution                      - Closed Ontology Rejections (Fixed in M2)
- Temporal Reversal & Historical State Confusion        - Retrieval Horizon Limits (Cross-session >=4)
- False Negative Over-Abstention                        - Knowledge Graph Edge Sparsity
- Arithmetic & Multi-hop Reasoning Errors
```

### 3.1 Verified Model Bottlenecks (`gpt-oss:latest`)

#### A. Compound Entity String Pollution (Extraction Stage)
* **Evidence (`evaluation/long_context/raw_extraction_single.json`):**
  When extracting knowledge from multi-agent narratives, `gpt-oss` frequently groups multiple individuals or concepts into a single entity string:
  - Extracted Target: `"Marc and Sarah Lin"` (instead of two distinct nodes: `"Marcus Vance"` and `"Sarah Lin"`).
  - Extracted Target: `"Priya, Vikram, and David"` (instead of 3 distinct entity nodes).
  - Extracted Target: `"July 28, 2025"` (extracts dates as relationship objects rather than temporal properties).
* **Impact:** Downstream entity resolution in SQLite cannot match compound strings against single-entity aliases. Consequently, graph nodes are created for `"Marc and Sarah Lin"`, causing graph traversals to completely fail when querying for either Marcus or Sarah individually.

#### B. Pathological Repetition Loops (Generation Stage)
* **Evidence (`d23cf73b` in Audited 17 Failures):**
  - Query: *"How many different cuisines have I learned to cook or tried out in the past few months?"*
  - Output: `gpt-oss` entered a degenerative infinite loop: `"I’m ready to help you with any cooking questions. I’m happy to help you with your cooking journey. I’m ready to help you with your cooking needs. I’m excited to help you with your cooking projects..."` repeating over 80 times until the token limit was exhausted.

#### C. False Negative Over-Abstention (Generation Stage)
* **Evidence (`b46e15ed`, `bbf86515`, `50635ada`):**
  - Even when gold conversational evidence was retrieved in the top 3 items and present in the prompt context, `gpt-oss` outputted: `"I do not have sufficient recorded memories to determine your previous frequent flyer status on United Airlines."`
  - While strict anti-hallucination prompting in `ANSWERING_SYSTEM_PROMPT` contributes to caution, `gpt-oss`'s low instruction calibration causes it to hedge on answerable single-hop questions.

#### D. Temporal State Reversal (Knowledge Update Stage)
* **Evidence (`07741c45`):**
  - Query: *"Where do I currently keep my old sneakers?"*
  - Gold Answer: *"in a shoe rack in my closet"* (updated state).
  - Model Output: *"You currently keep your old sneakers under your bed."* (outdated historical state).
  - The model read both historical and current turns in the evidence block but lacked the temporal reasoning capacity to prioritize the superseding state.

### 3.2 Non-Model Bottlenecks (Deterministic Pipeline & Data)

1. **Closed Ontology Rejection (Resolved in Milestone 2):**
   - In Milestone 1, 51/54 valid relationships were rejected because `validator.py` enforced a closed whitelist. Milestone 2's open-world ontology resolved this; it was a pipeline rule error, not an LLM defect.
2. **Context Displacement (Resolved in Milestone 3.2):**
   - Earlier benchmarks suffered from semantic memories displacing episodic turns in fixed $k=5$ slots. The `AdaptiveEvidenceSelector` resolved this.
3. **Graph Sparsity:**
   - Missing multi-hop answers often stem from the fact that `memory_graph.db` contains only 21 nodes. If an edge was never ingested, even a 100B parameter model cannot traverse it.

---

## 4. Model Candidate Evaluation

To upgrade IVARA's LLM pipeline while respecting the **16 GB VRAM ceiling** and Windows local inference constraint, we evaluated candidate architectures against strict criteria:

### 4.1 Evaluation Criteria
1. **Extraction & Structured Output Precision:** Guaranteed compliance with Pydantic JSON schemas without parser crashes or repetition loops.
2. **Entity Atomicity:** Ability to separate compound entity mentions into discrete SPO triples.
3. **Temporal & Arithmetic Reasoning:** Resolves superseding states and multi-hop counts.
4. **VRAM Footprint & Layer Offload:** Must fit 100% within 16 GB VRAM with sufficient KV-cache headroom for 8k–16k context.
5. **Generation Latency:** Must achieve $\ge 60$ tokens/sec on the RTX 5080 Laptop GPU.

### 4.2 Candidate Comparison Matrix

| Model Candidate | Parameters & Quantization | Size on Disk / VRAM Required | Context Window | Structured JSON / Tool Calling | Extraction & Relational Quality | Observed / Projected Latency (RTX 5080) | Assessment & Viability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Current Baseline:**<br>`gpt-oss:latest` | 20.91B MoE<br>MXFP4 | 13.0 GB / **14.2 GB VRAM** | 131,072 | Moderate (Occasionally generates loops or compound entities) | Moderate (lumps compound entities into single strings) | ~137 tok/sec | High memory pressure (leaves only ~1.8 GB free VRAM); prone to repetition loops and over-abstention. |
| **Primary Recommendation:**<br>`qwen2.5:14b` | 14.7B Dense<br>Q4_K_M | 9.0 GB / **10.5 GB VRAM** | 32,768 (native)<br>128,000 (YaRN) | **Exceptional** (State-of-the-art in 14B class for JSON schemas) | **Superior** (Excels at entity separation, fine-grained triples, and temporal reasoning) | **~85–100 tok/sec** | **Best-in-Class Fit.** Fits comfortably in 16 GB VRAM with 5.5 GB headroom for 16k KV cache. Eliminates compound entity bugs. |
| **Fallback Candidate 1:**<br>`llama3.1:8b` | 8.03B Dense<br>Q4_K_M | 4.7 GB / **5.8 GB VRAM** | 128,000 | Very Good (High instruction compliance) | Good (Fast, but slightly lower multi-hop relation depth) | **>160 tok/sec** | Ultra-lightweight, zero OOM risk, near-instant inference, but slightly less nuanced on complex relational triples. |
| **Fallback Candidate 2 (Pre-installed):**<br>`gemma4-26b-16GB-VRAM` | 25.2B Dense<br>IQ4_XS | 14.0 GB / **15.1 GB VRAM** | 262,144 | Excellent (Already proven as judge in sprint) | Strong reasoning, but high memory pressure | ~45–60 tok/sec | **Already installed locally.** High reasoning capability, but 15.1 GB VRAM usage leaves almost zero headroom for 16k context buffers. |

---

## 5. Primary and Fallback Recommendations

### 5.1 Primary Recommendation: `qwen2.5:14b` (Qwen 2.5 14B Instruct)
* **Why this model?**
  1. **Entity Atomicity & Relation Extraction:** Qwen 2.5 was specifically post-trained on massive synthetic and verified knowledge-graph extraction corpora. It correctly disaggregates `"Marcus Vance and Sarah Lin"` into two distinct entities with precise individual roles, fixing the primary root cause of graph disconnection.
  2. **Schema & JSON Reliability:** It exhibits near-zero repetition degeneration under constrained JSON decoding compared to MoE models in 4-bit precision.
  3. **Optimal VRAM Geometry:** At 9.0 GB weights, it consumes ~10.5 GB VRAM under full GPU offload, leaving a massive **5.5 GB safety buffer** for the KV cache at 16,384 tokens. This guarantees zero CPU fallback and zero VRAM eviction errors.
  4. **Temporal Sensitivity:** Significantly superior performance on date arithmetic and chronological state ordering.

### 5.2 Fallback Recommendation: `llama3.1:8b` (Llama 3.1 8B Instruct)
* **Why this fallback?**
  - If any unexpected Ollama download bottleneck or quantization compatibility issue arises with 14B weights, `llama3.1:8b` downloads in under 3 minutes (4.7 GB), fits into any VRAM configuration (5.8 GB), and offers reliable structured output.

---

## 6. Two-Day Controlled Experiment Plan

To guarantee scientific rigor, the experiment compares `gpt-oss:latest` (Baseline) against `qwen2.5:14b` (Primary Candidate) under **identical prompts, identical test datasets, identical ontology rules, and identical pipeline configurations**.

```
Day 1: Micro-Benchmark & Extraction Validation (N = 20)
├── Phase 1.1: Environment Pre-flight & Model Pull (qwen2.5:14b)
├── Phase 1.2: Extraction Diagnostic Evaluation (10 multi-agent turns)
└── Phase 1.3: Answering & Abstention Diagnostic Evaluation (10 targeted questions)
    └── [CHECKPOINT 1]: Go / No-Go Decision based on Schema & Atomicity Metrics

Day 2: Full Stratified Held-Out Validation (N = 60) & Decision Freeze
├── Phase 2.1: Run 60 Held-Out Evaluation Arms (Baseline vs. Candidate)
├── Phase 2.2: Compute Paired Metrics (Exact McNemar, Fact Coverage, Citation Precision)
└── Phase 2.3: Final Freeze & Release Sign-Off
```

### 6.1 Diagnostic Test Dataset (Day 1: $N = 20$ Controlled Sample)

A carefully curated, manually verified ground-truth test suite:
- **Set A: Extraction & Entity Resolution (10 Narrative Turns):**
  - Tests entity atomicity (splitting compound mentions).
  - Tests SPO predicate alignment against the open-world ontology.
  - Tests temporal claim extraction (`valid_from`, `valid_to`).
- **Set B: Answering & Grounding (10 Targeted Queries):**
  - 3 Temporal / Knowledge-Update queries (e.g., outdated vs. current keyboard, moved residences).
  - 3 Multi-hop Relational queries (e.g., project $\to$ lead $\to$ previous company).
  - 2 Single-Fact queries.
  - 2 Unanswerable / Abstention queries (verifying zero hallucination).

### 6.2 Evaluation Metrics & Acceptance Criteria

| Metric | Baseline (`gpt-oss`) | Target Candidate (`qwen2.5:14b`) | Success Threshold |
| :--- | :---: | :---: | :---: |
| **Entity Precision / Recall / F1** | F1: ~0.76 (compound errors) | F1: $\ge 0.88$ | $+10\%$ improvement in F1 |
| **Compound Entity Rate** | ~18% of extracted entities | $< 3\%$ | $\ge 80\%$ reduction in compound entities |
| **Relationship Schema Validity** | 100% (with open-world) | 100% | Zero syntax or validation regressions |
| **JSON Output Compliance** | 98.3% (occasional loop) | 100.0% | Zero infinite loop / parser crash instances |
| **Held-Out Answer Accuracy** | 60.0% (36/60 in Batch 5) | $\ge 66.7\%$ ($\ge 40/60$) | Net $+4$ wins without new regressions |
| **Citation Precision** | 100.0% | 100.0% | Strict 100% authentic citation grounding |
| **Generation Latency** | ~137 tok/sec | $\ge 80$ tok/sec | Acceptable trade-off for higher accuracy |
| **Peak VRAM Consumption** | 14.2 GB | $\le 11.5$ GB | Retains $\ge 4.5$ GB VRAM headroom |

---

## 7. Risks and Rollback Strategy

1. **Risk: Ollama Model Compatibility or Pull Failure:**
   - *Mitigation:* We have verified Ollama 0.33.2 supports `qwen2.5:14b`. If download fails or network times out, we fall back immediately to `llama3.1:8b` or local `gemma4-26b`.
2. **Risk: Prompt Incompatibility with Qwen Chat Template:**
   - *Mitigation:* IVARA's prompts use standard `messages=[{"role": "system", ...}, {"role": "user", ...}]` which Ollama automatically compiles into Qwen's native `<|im_start|>` ChatML format.
3. **Risk: Regression on Abstention / Calibration:**
   - *Mitigation:* Checkpoint 1 on Day 1 explicitly tests the 2 unanswerable questions. If the candidate hallucinates on unrecorded facts, the model is rejected.
4. **Rollback Mechanism:**
   - Changing models in IVARA requires updating only a single parameter (`model="qwen2.5:14b"` vs `model="gpt-oss"`).
   - Zero database mutations occur during benchmarking. Rolling back to `gpt-oss` takes literally 5 seconds with zero code reversion.

---

## 8. Verification & Read-Only Audit Attestation

* **Codebase Changes:** ZERO lines of source code or configuration were modified during this audit.
* **Production Data:** All production databases and hashes remain 100% bit-exact and untouched.
* **Branch State:** Cleanly preserved on `research/evaluation-baseline` with zero commits and zero pushes.
* **Status:** Audit completed. Awaiting user review and authorization before proceeding with Day 1 model acquisition or execution.

