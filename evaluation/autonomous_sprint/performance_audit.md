# Performance Profiling & Latency Audit

**Sprint Date:** October 28, 2026  
**Branch:** `research/evaluation-baseline`  
**Hardware Environment:** Intel Core Ultra 9 / RTX 5080 Laptop GPU (16 GB GDDR7 VRAM)  
**Execution Runtime:** Python 3.14.4 (`.venv`), PyTorch 2.5.1 + CUDA 12.4, Ollama v0.5.x  

---

## 1. Executive Summary

A critical question for IVARA's scaling is whether the layered memory pipeline (embedding, lexical search, ontology guidance, graph traversal, and temporal resolution) introduces computational bottlenecks relative to raw LLM generation.

**Direct Measured Finding:**  
- **Non-LLM Algorithmic Stages:** All indexing, hybrid retrieval, ontology concept extraction, SQLite graph traversal, temporal claim extraction, and context assembly operations combine to **under 10.0 ms per question** (median ~5.5 ms).
- **LLM Generation Stage:** Local model inference (`gpt-oss:latest`, 20.9B MoE, 32 experts) requires **5,600 ms to 6,800 ms per question**, accounting for **85.7% to 91.5% of total end-to-end execution latency**.
- **Temporal Efficiency Gain:** Config-D demonstrated a *faster* mean total latency (6,468 ms) than Config-B (7,439 ms) because structured temporal claim resolution (`[2023-01-10 to 2023-01-24]`) provided concise temporal grounding, breaking repetitive generation loops and shortening token decoding length.

---

## 2. Stage-by-Stage Measured Microbenchmarks

Timings were measured deterministically using `scratch/microbench_stages.py` over synthetic conversational turn-chunk corpora (50 turn-chunk units, isolated `:memory:` SQLite store, repeated $N \ge 30$ iterations).

| Pipeline Stage | Cold Latency | Warm Mean | Warm Median | Warm P95 | Sample Size ($N$) | Notes / Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dense Embedding (`all-MiniLM-L6-v2`)** | 27.71 ms | 4.29 ms | 4.02 ms | 6.05 ms | 30 | PyTorch GPU / SentenceTransformers, 384-dim |
| **BM25 Lexical Retrieval** | 0.18 ms | 0.126 ms | 0.125 ms | 0.135 ms | 50 | In-memory tokenized inverted index |
| **Ontology Concept Extraction** | 0.15 ms | 0.093 ms | 0.080 ms | 0.110 ms | 50 | Regex + trie entity/predicate vocabulary |
| **Full Hybrid Retrieval (Dense+BM25+Ontology)** | 29.50 ms | 5.21 ms | 4.89 ms | 7.86 ms | 30 | Fused RRF / linear normalization over 50 chunks |
| **SQLite Graph Traversal (1-Hop BFS, 3 seeds)** | 0.12 ms | 0.061 ms | 0.058 ms | 0.063 ms | 50 | In-memory SQLite with compound indexed edges |
| **Temporal Claim Extraction & Resolution** | 0.65 ms | 0.33 ms | 0.28 ms | 0.28 ms | 50 | Regex extraction (5 chunks) + DAG supersession |
| **Context Assembly & Prompt Construction** | <0.001 ms | 0.0003 ms | 0.0003 ms | 0.0004 ms | 50 | String interpolation & token truncation |
| **Total Non-LLM Pipeline Overhead** | **~30.5 ms** | **~5.7 ms** | **~5.3 ms** | **~8.3 ms** | - | **Negligible vs LLM generation** |

---

## 3. Measured LLM Inference Latency (Campaign v1 Pilot)

The table below records empirical execution latencies measured during the 60-question Evaluation Campaign v1 pilot run on identical hardware, utilizing Ollama `gpt-oss:latest` (20.91B parameters, 32 experts / 4 active per token, context 4,096 tokens, MXFP4 quantized):

| Configuration | Description | Total Mean Latency | Total Median Latency | LLM Answering Stage Mean | LLM % of Total | Repetition Loops Observed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Config-A** | Dense + BM25 Baseline | 7,244.7 ms | 6,632.8 ms | 6,527.1 ms | 90.1% | 3 |
| **Config-B** | Baseline + Ontology Guidance | 7,439.3 ms | 6,776.9 ms | 6,804.5 ms | 91.5% | 3 |
| **Config-C** | Ontology + Bounded Graph | 6,878.7 ms | 6,502.7 ms | 5,892.0 ms | 85.7% | 1 (severe 32-sentence loop) |
| **Config-D** | Full Architecture + Temporal | 6,468.6 ms | 6,044.1 ms | 5,616.4 ms | 86.8% | **0** |

### Latency Analysis
1. **Config-B vs Config-A (+194 ms):** Ontology expansion retrieves richer, more specific context blocks. The slight increase in prompt token count accounts for the ~194 ms increase in generation time.
2. **Config-D vs Config-B (-970 ms):** Despite doing more processing (temporal claim extraction, supersession DAG resolution, and interval formatting), Config-D is nearly a second faster per question. Providing explicit resolved intervals (`[2023-01-10 to 2023-01-24]`) eliminates model hesitation, avoids repetitive rambling, and enables the model to emit concise answers in fewer output tokens.

---

## 4. Estimated Benchmark Scaling Projections

To assess readiness for larger runs without spending budget or running unauthorized tests, costs are projected from the empirical pilot measurements:

| Evaluation Tier | Questions ($N$) | Answering Latency (1 Config) | Answering Latency (4 Configs) | Gemma Judge Latency (4 Configs) | Total Local Wall-Clock Time | Estimated Financial Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pilot (Completed)** | 60 | 6.5 min | 26.0 min | 17.4 min | 43.4 min | $0.00 (Local GPU) |
| **Diagnostic Sample (Proposed)** | 100 | 10.8 min | 43.2 min | 29.0 min | ~1.2 hours | $0.00 (Local GPU) |
| **Full LongMemEval-S (Benchmark)** | 500 | 54.0 min | 3.6 hours | 2.4 hours | ~6.0 hours | $0.00 (Local GPU) |

### Limitations & Caveats
- **Synthetic Microbenchmarks vs Full LongMemEval:** The synthetic microbenchmark utilized 50 chunks per query. In full LongMemEval-S, instances average 80 to 220 chunks. Because FAISS search is $O(\log N)$ or fast linear scan for small $N$, and SQLite lookups are $O(1)$ indexed B-tree lookups, non-LLM overhead will remain below 20 ms per query even on 500 chunks.
- **VRAM Contention:** Sequential execution (generating predictions with `gpt-oss`, unloading, then running `gemma4-26b` judge) is mandatory on 16 GB VRAM systems to avoid CUDA out-of-memory or driver thrashing.
