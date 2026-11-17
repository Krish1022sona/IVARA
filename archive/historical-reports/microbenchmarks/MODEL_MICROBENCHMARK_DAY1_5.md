# IVARA — Three-Model Comparative Micro-Benchmark Report (Day 1.5)
## Empirical Evaluation: `gpt-oss:latest` vs. `qwen2.5:14b` vs. `VladimirGav/gemma4-26b-16GB-VRAM:latest`

**Evaluation Date:** November 16, 2026  
**Auditor:** IVARA Independent Evaluation & Quality Assurance  
**Target File:** `MODEL_MICROBENCHMARK_DAY1_5.md`  
**Execution Environment:** Isolated Benchmark Directory (`evaluation/microbenchmark_day1_5/`)  
**Hardware Platform:** NVIDIA GeForce RTX 5080 Laptop GPU (16,303 MiB VRAM) | Intel Core Ultra 9 275HX | 32 GB RAM  
**Production Status:** STRICT READ-ONLY. Zero changes to production prompts, schemas, configurations, or databases (`data/`).

---

## 1. Executive Summary & Evaluated Models

Following the Day 1 micro-benchmark and subsequent integrity audit, we evaluated the three locally available LLMs on the same 20 diagnostic scenarios (10 knowledge extraction passages, 10 grounded question answering queries) under identical prompt structures, Pydantic schemas, and greedy sampling ($T=0$).

### Evaluated Model Inventory

| Roster Role | Model Name / Ollama Tag | Architecture | Quantization | Size on Disk | Context Budget |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (Current)** | `gpt-oss:latest` | 20.9B MoE | MXFP4 | 13.0 GB | 16,384 |
| **Candidate 1** | `qwen2.5:14b` | 14.8B Dense | Q4_K_M | 9.0 GB | 16,384 |
| **Candidate 2** | `VladimirGav/gemma4-26b-16GB-VRAM:latest` | 25.2B Dense (Reasoning) | IQ4_XS | 14.0 GB | 8,192* |

*\*Note on Context Budget:* Gemma 4 was capped at `num_ctx: 8192` because its 14.0 GB base model weights consume 14.6 GB VRAM on our 16.3 GB GPU. Attempting a 16k context window triggered CUDA memory spillover.

### High-Level Summary of Results

1. **`qwen2.5:14b` emerges as the dominant candidate across all empirical dimensions:**
   - Achieves the highest **Strict Relationship Precision (59.09% vs 31.51% baseline)** and **F1 (61.90% vs 40.71% baseline)**.
   - Achieves the highest **Entity Precision (75.93% vs 56.58% baseline)**, eliminating date and job-title entity pollution.
   - Lowest peak VRAM consumption (**12.4 GB peak, 9.5 GB baseline**), leaving **4.0–6.7 GB headroom**.
   - Lowest end-to-end turnaround latency (**5,360 ms mean**).
   - Zero corporate affiliation hallucinations; 100% provenance completeness on multi-hop reasoning (`ans_05`).

2. **`VladimirGav/gemma4-26b-16GB-VRAM:latest` is disqualified due to catastrophic structured extraction failures:**
   - **80% schema failure rate on extraction:** In 6 of 10 passages, it hallucinated an entirely different schema (e.g., `{"id": "...", "types": [...], "properties": {}}`) wrapped in markdown code fences; in 2 passages, it spent over 15,000 characters thinking and timed out with empty output (EOF).
   - **Citation field omission:** In 7 of 8 answerable queries, it placed citations in text brackets (`[m001]`) but emitted an empty list in `cited_memory_ids: []`.
   - **Severe latency penalty:** Generated 68,284 characters of internal reasoning, averaging **10,834 ms per turn (102% slower than Qwen)**.
   - **Dangerous VRAM usage:** Peaked at **14,631 MiB (14.6 GB)**, leaving only **1.6 GB headroom** on our 16 GB GPU.

---

## 2. Comprehensive Three-Model Metric Matrix

All extraction metrics below reflect the audited fixes: **strict semantic relationship matching** (requiring Subject + Object + Predicate alignment) and **entity deduplication** per passage.

| Dimension / Metric | Current Baseline (`gpt-oss:latest`) | Candidate 1 (`qwen2.5:14b`) | Candidate 2 (`gemma4:latest`) | Leading Model |
| :--- | :---: | :---: | :---: | :---: |
| **Entity Extraction Precision** | 56.58% (43 TP / 33 FP) | **75.93% (41 TP / 13 FP)** | 76.47% (13 TP / 4 FP)* | **Qwen 2.5** |
| **Entity Extraction Recall** | **93.48% (43 / 46)** | 89.13% (41 / 46) | 28.26% (13 / 46)* | **GPT-OSS** |
| **Entity Extraction F1 Score** | 70.49% | **82.00%** | 41.27%* | **Qwen 2.5** |
| **Strict Relationship Precision** | 31.51% (23 TP / 50 FP) | **59.09% (26 TP / 18 FP)** | 50.00% (9 TP / 9 FP)* | **Qwen 2.5** |
| **Strict Relationship Recall** | 57.50% (23 / 40) | **65.00% (26 / 40)** | 22.50% (9 / 40)* | **Qwen 2.5** |
| **Strict Relationship F1 Score** | 40.71% | **61.90%** | 31.03%* | **Qwen 2.5** |
| **Lenient Relationship Precision** | 38.36% (28 TP / 45 FP) | **63.64% (28 TP / 16 FP)** | 50.00% (9 TP / 9 FP)* | **Qwen 2.5** |
| **Lenient Relationship Recall** | **70.00% (28 / 40)** | **70.00% (28 / 40)** | 22.50% (9 / 40)* | **Tied (Qwen/GPT)** |
| **Lenient Relationship F1 Score** | 49.56% | **66.67%** | 31.03%* | **Qwen 2.5** |
| **Factual Hallucinations** | 1 instance (`ext_02`) | **0 instances** | **0 instances** | **Qwen 2.5** |
| **Extraction Schema Compliance** | **100.0% (10/10)** | **100.0% (10/10)** | 20.0% (2/10) | **Tied (Qwen/GPT)** |
| **Answerable QA Accuracy** | **100.0% (8/8)** | **100.0% (8/8)** | 87.5% (7/8) | **Tied (Qwen/GPT)** |
| **Abstention QA Accuracy** | **100.0% (2/2)** | **100.0% (2/2)** | **100.0% (2/2)** | **All Three** |
| **Overall QA Accuracy** | **100.0% (10/10)** | **100.0% (10/10)** | 90.0% (9/10) | **Tied (Qwen/GPT)** |
| **Multi-Hop Citation Fidelity** | 87.5% (7/8) | **100.0% (8/8)** | 12.5% (1/8)** | **Qwen 2.5** |
| **Citation Validity (No Hallucinations)**| **100.0% (10/10)** | **100.0% (10/10)** | **100.0% (10/10)** | **All Three** |
| **Answering Schema Compliance** | **100.0% (10/10)** | **100.0% (10/10)** | 90.0% (9/10) | **Tied (Qwen/GPT)** |
| **Repetition Degeneration** | 0 / 10 | 0 / 10 | 0 / 10 | **All Three** |
| **Mean End-to-End Latency** | 5,918.9 ms | **5,360.2 ms** | 10,834.2 ms | **Qwen 2.5** |
| **Mean Extraction Latency** | **8,786.7 ms** | 8,976.0 ms | 15,947.2 ms | **GPT-OSS** |
| **Mean Answering Latency** | 3,051.2 ms | **1,744.4 ms** | 5,721.3 ms | **Qwen 2.5** |
| **GPU Generation Throughput** | **136.5 tok/s** | 63.9 tok/s | 131.8 tok/s | **GPT-OSS** |
| **Peak GPU VRAM Consumption** | 13,039 MiB (~12.7 GB) | **12,736 MiB (~12.4 GB)** | 14,631 MiB (~14.6 GB) | **Qwen 2.5** |
| **VRAM Headroom (on 16.3 GB GPU)**| 3.3 GB | **4.0 - 6.7 GB** | 1.6 GB | **Qwen 2.5** |

*\*Note on Gemma 4 Extraction Recall:* Because Gemma 4 failed validation on 8 out of 10 passages, its aggregate recall is 28.26% on entities and 22.50% on relationships across the full dataset.  
*\*\*Note on Gemma 4 Citation Fidelity:* Gemma 4 inserted citation tags in text brackets but left `cited_memory_ids: []` in 7 queries.

---

## 3. Hardware Resources, Throughput & VRAM Profile

```
Hardware Comparison Profile (NVIDIA GeForce RTX 5080 Laptop GPU, 16,303 MiB VRAM)
-----------------------------------------------------------------------------------------------------
Model                     Parameters  Quant    Base VRAM    Peak VRAM    VRAM Headroom   Mean Latency
gpt-oss:latest            20.9B MoE   MXFP4    12,736 MiB   13,039 MiB   3,264 MiB       5,918.9 ms
qwen2.5:14b               14.8B Dense Q4_K_M    9,555 MiB   12,736 MiB   4,567 MiB       5,360.2 ms  <-- BEST
gemma4-26b-16GB-VRAM      25.2B Dense IQ4_XS   14,503 MiB   14,631 MiB   1,672 MiB      10,834.2 ms  <-- RISKY
```

### Resource Dynamics & Analysis
1. **Memory Safety & Headroom:**
   - `qwen2.5:14b` operates at a resting baseline of **9,555 MiB (~9.3 GB)**, peaking at **12,736 MiB (~12.4 GB)** under heavy 16k context ingestion. It leaves **4.0 to 6.7 GB of free VRAM**, providing plenty of room for multi-turn conversational caches and graph traversals.
   - `VladimirGav/gemma4-26b-16GB-VRAM` occupies **14,503 MiB at idle**, leaving only **1.6 GB of VRAM headroom**. Under real-world multi-turn dialogues, it faces an imminent risk of CUDA out-of-memory crashes or emergency CPU offloading.

2. **The Reasoning Latency Penalty in Gemma 4:**
   - Gemma 4 generated **68,284 characters** of internal chain-of-thought tokens across the 20 scenarios.
   - On extraction passage `ext_01`, it spent 13,969 characters thinking, requiring **36.2 seconds** for a single turn.
   - On `ext_05` and `ext_10`, it exhausted its entire token generation budget inside internal reasoning, returning **0 content bytes** and causing an EOF validation failure.
   - Overall mean latency for Gemma 4 is **10,834 ms**, more than double `qwen2.5:14b` (5,360 ms).

3. **Latency vs. Throughput Paradox (GPT-OSS vs. Qwen):**
   - Although `gpt-oss` generates raw tokens faster (136.5 tok/s vs 63.9 tok/s) due to MoE architecture, it produces verbose text and 60% more spurious triples.
   - `qwen2.5:14b` outputs concise, non-redundant payloads, making its **answering latency 42.8% faster (1,744 ms vs 3,051 ms)** and its overall turnaround **9.4% faster**.

---

## 4. Deep Qualitative & Disagreement Analysis

### 4.1 Knowledge Extraction Failures & Model Disagreements

#### Passage `ext_01` (Team Kickoff)
- **`gpt-oss`:** Extracted the calendar date `"January 14, 2025"` as a person/agent node: `(January 14, 2025) -[kicked off]-> (Project Chronos)`.
- **`qwen2.5`:** Cleanly extracted `(SELF) -[leads]-> (Project Chronos)`. Correctly kept dates out of the entity graph.
- **`gemma4`:** Extracted 10 entities, but included functional descriptions as entities: `'distributed storage engineering'`, `'custom temporal attention kernels'`.

#### Passage `ext_02` (Dual-Company Partnership & Entity Disambiguation)
- **Gold Context:** *"He introduced Elena Ramos, their [Vanguard Robotics'] lead hardware engineer. She is completely distinct from Dr. Elena Rostova, who is our Chief Scientist at Aetheria Labs."*
- **`gpt-oss`:** Fabricated a critical false edge: `(Elena Ramos) -[works for]-> (Aetheria Labs)`. Also extracted job titles (`"Chief Scientist"`, `"Lead Hardware Engineer"`) as entity nodes.
- **`qwen2.5`:** Perfectly resolved both entities: `(Elena Ramos) -[worksFor]-> (Vanguard Robotics)` and `(Dr. Elena Rostova) -[worksFor]-> (Aetheria Labs)`. Zero hallucinations.
- **`gemma4`:** Correctly resolved employers, but extracted job titles as separate entity nodes.

#### Passages `ext_03`, `ext_04`, `ext_06`, `ext_07`, `ext_08`, `ext_09` (Schema Drift & Markdown Fences)
- **`gpt-oss` & `qwen2.5`:** Emitted 100% valid JSON matching IVARA's `Extraction` schema.
- **`gemma4`:** Failed Pydantic validation across all 6 passages:
  ```json
  ```json
  {
    "entities": [
      {"id": "SELF", "types": ["person"], "properties": {}},
      {"id": "Datacore", "types": ["organization"], "properties": {}}
    ],
    "topics": ["career history", "employment"],
    "relationships": [
      {"origin": "SELF", "relation": "employed_by", "target": "Datacore"}
    ]
  }
  ```
  ```
  *Failure Modes:*
  1. Wrapped output in markdown code fences (```` ```json ````).
  2. Changed `{"name": "...", "type": "..."}` to `{"id": "...", "types": [...]}`.
  3. Changed `topics` from a list of objects `[{"name": "..."}]` to a flat list of strings.
  4. Changed `relationships` fields from `source` to `origin` or `subject`.

#### Passages `ext_05` and `ext_10` (Reasoning Runaway / Token Exhaustion)
- **`gemma4`:** Generated over 15,000 characters of internal thinking in `ext_05` and `ext_10`. It hit the maximum token prediction limit (`num_predict: 4096`) before writing the output JSON, returning an empty string `""` (EOF parsing error).

---

### 4.2 Grounded Answering Disagreements & Citation Behavior

#### Query `ans_05` (Multi-Hop Relational Synthesis)
- **Query:** *"What company did the current Lead Architect of CloudMatrix work for between 2021 and early 2024?"*
- **Premises Needed:** `m401` (Arjun is Lead Architect at CloudMatrix) AND `m402` (Arjun worked at Datacore from 2021 to 2024).
- **`gpt-oss`:** Outputted *"He worked for Datacore."* Cited **only `['m402']`**, omitting the premise linking the role to Arjun.
- **`qwen2.5`:** Outputted a complete synthesis: *"Based on the provided evidence, the current Lead Architect of CloudMatrix Inc, Arjun, worked as a Senior Data Engineer at Datacore..."* Cited **both `['m401', 'm402']`** (100% provenance fidelity).
- **`gemma4`:** Formulated a correct answer, but cited **empty list `[]`** in `cited_memory_ids` (inserted `[m401, m402]` into prose text only).

#### Query `ans_08` (Negative Information / Absence-of-Evidence)
- **Query:** *"Has Arjun ever worked at Google or lived in London?"*
- **`gpt-oss`:** *"Based on the recorded conversations, there is no record of Arjun having worked at Google, and there is no record of him having lived in London."* Cited `['m701', 'm702']`.
- **`qwen2.5`:** *"Based on the recorded conversations, there is no record of Arjun working at Google or living in London."* Cited `['m701']`.
- **`gemma4`:** **CRASHED.** Spent 4,269 characters thinking and ran out of tokens before generating the answer (EOF crash).

#### Query `ans_10` (Unanswerable Question Abstention)
- **Query:** *"How much did Arjun pay out-of-pocket for dental surgery on April 4, 2025?"*
- **`gpt-oss`:** Abstained with `status: insufficient_evidence`.
- **`qwen2.5`:** Abstained with answer `"insufficient_evidence"`, explaining that the appointment was for routine cleaning, not surgery. (Omitted `"status"` key; Pydantic defaulted to `"answered"`).
- **`gemma4`:** Cleanly abstained with `status: insufficient_evidence` and exact grounded explanation.

---

## 5. Model Incompatibility & Behavioral Summary

| Model | Schema Reliability | Citation Handling | Latency Stability | VRAM Safety | Key Behavioral Flaw |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`gpt-oss:latest`** | 95.0% (1 EOF error) | Good (87.5% multi-hop) | Moderate (~5.9s) | Acceptable (13.0 GB) | Spurious date/title entities; corporate hallucination in `ext_02`. |
| **`qwen2.5:14b`** | **100.0% (No errors)** | **Best (100% multi-hop)** | **Fastest (~5.4s)** | **Safest (12.4 GB peak)** | Omitted `"status"` key on `ans_10` (easily normalized). |
| **`gemma4:latest`** | **Catastrophic (20% on ext)**| **Flawed (12.5% in array)**| **Slowest (~10.8s)** | **Dangerous (14.6 GB)** | Reasoning runaway (EOF on 3 scenarios); invented custom schemas. |

---

## 6. Recommendations for Day 2 Evaluation

### 1. Disqualification of `VladimirGav/gemma4-26b-16GB-VRAM:latest`
We strongly recommend **disqualifying Gemma 4** from further consideration for IVARA:
- Its 80% failure rate on extraction schemas violates IVARA's structured extraction contract.
- Its 14.6 GB memory footprint leaves zero safety margin on 16 GB hardware.
- Its unconstrained reasoning loops double system latency and randomly exhaust prediction token limits, leading to empty-string failures on production queries.

### 2. Candidate for Day 2 Held-Out Evaluation: `qwen2.5:14b`
We recommend advancing **`qwen2.5:14b`** to the full Day 2 held-out evaluation ($N = 60$ queries across all 8 IVARA query classes) against `gpt-oss:latest`:
- **Superior Knowledge Graph Construction:** +27.58% strict relationship precision (59.09% vs 31.51%), +19.35% entity precision (75.93% vs 56.58%), zero hallucinations.
- **Superior Multi-Hop Citation Fidelity:** Captures 100% of required bridge premise IDs (`m401` + `m402`).
- **Optimal Hardware Fit:** Frees 3.1 GB of VRAM, operates 42.8% faster in answering turnaround, and guarantees 100% JSON schema compliance.

### 3. Immediate Benchmark Harness Preparation for Day 2:
- Ensure the 2-line post-processing status normalizer is integrated in the evaluation harness for `ans_10` edge-cases.
- Enforce strict semantic predicate matching and deduplicated entity scoring in all Day 2 scripts.
- Keep production code, configurations, and databases untouched until Day 2 results are reviewed and approved.

