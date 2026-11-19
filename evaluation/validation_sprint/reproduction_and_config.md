# IVARA Validation Sprint: Benchmark Configuration & Reproduction Guide

**Document ID:** `IVARA-VAL-REPRO-01`  
**Date:** 2026-11-15  
**Branch:** `research/evaluation-baseline`  

---

## 1. System Environment & Model Configuration

### Hardware & Operating System
- **Operating System:** Windows 11 (AMD64)
- **Runtime Shell:** PowerShell / cmd
- **Python Version:** Python 3.14.4 (`.\.venv\Scripts\python.exe`)
- **Isolation:** Dedicated virtual environment; 100% offline local execution (zero cloud calls, zero external APIs).

### Foundation Models
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
  - *Weights Location:* Local Hugging Face / SentenceTransformer cache
  - *Vector Dimension:* 384
  - *Distance Metric:* L2-normalized Inner Product (Cosine Similarity)
  - *Relevance Filtering Threshold:* $0.30$ ($min\_score = 0.30$)
- **Generation Model:** `gpt-oss` via Local Ollama daemon
  - *Server Endpoint:* `http://localhost:11434`
  - *Temperature:* $0.0$ (deterministic greedy decoding)
  - *Context Budget:* $k=5$ chunks/memories (up to 1,500 context tokens)
  - *Output Schema:* Strict JSON with `status`, `answer`, `citations`, `limitations`

---

## 2. Integrity & Cryptographic Checksums

All benchmarks run on isolated stores in `evaluation/validation_sprint/ivara_store_validation/`.
Production stores remain untouched with verified byte-for-byte SHA256 checksums:

| File | SHA-256 Checksum | Verification Status |
| :--- | :--- | :--- |
| `data/memories.json` | `489e3ea2138022f97a590808cee3dfba03465966eda06d750e340d6d0fddf858` | Verified Bit-Exact |
| `data/entities.json` | `51f993fbe1cfc7f40084724499cd2dd216b30d8e3aeeace9824d1863d84fc09e` | Verified Bit-Exact |
| `data/memory_graph.db`| `f6b9f9cf7d35674e461350ffa87dcfa735ae0d7643c46e30138f56bffb31f95d` | Verified Bit-Exact |
| `data/conversations.db`| `af88c18f31d768e7a62a91d5c4dd40e1a8271fe4de1ee86e56c4d180021bd11f`| Verified Bit-Exact |

---

## 3. Benchmark Dataset Summary

- **Source Sessions:** 18 continuous narrative sessions spanning March 2024 through November 2027.
- **Total Questions:** 280 structured questions
  - `dev` split: 30 questions
  - `test` split: 250 held-out questions (used for all reported validation metrics)
- **Question Categories (12 Categories):**
  1. `single_fact_retrieval` (22 questions)
  2. `multi_hop_reasoning` (22 questions)
  3. `entity_relationships` (22 questions)
  4. `temporal_reasoning` (22 questions)
  5. `historical_state_retrieval` (22 questions)
  6. `knowledge_updates` (17 questions)
  7. `contradiction_handling` (17 questions)
  8. `preferences_and_constraints` (17 questions)
  9. `negative_information` (17 questions)
  10. `unanswerable_questions` (22 questions)
  11. `cross_session_synthesis` (25 questions)
  12. `complex_multi_fact` (25 questions)

---

## 4. Reproduction Commands

### Master Automated Runner (All Arms & Statistical Tests)
To run the full suite across all arms and export all deliverables:
```powershell
.\.venv\Scripts\python.exe -m evaluation.validation_sprint.run_all_evaluations
```

### Running Individual Benchmark Arms
Each arm can be executed or resumed independently:

1. **Arm A: Conventional Vector RAG**
   ```powershell
   .\.venv\Scripts\python.exe -c "from evaluation.validation_sprint.dataset_expanded import SESSIONS, QUESTIONS; from evaluation.validation_sprint.runner import run_evaluation_arm; qs = [q for q in QUESTIONS if q['split'] == 'test']; run_evaluation_arm('baseline_vector', qs, SESSIONS, resume=True)"
   ```

2. **Arm B: Conventional Hybrid RAG (Vector + BM25)**
   ```powershell
   .\.venv\Scripts\python.exe -c "from evaluation.validation_sprint.dataset_expanded import SESSIONS, QUESTIONS; from evaluation.validation_sprint.runner import run_evaluation_arm; qs = [q for q in QUESTIONS if q['split'] == 'test']; run_evaluation_arm('baseline_hybrid', qs, SESSIONS, resume=True)"
   ```

3. **Arm C: Fair Speaker-Aware Hybrid RAG**
   ```powershell
   .\.venv\Scripts\python.exe -c "from evaluation.validation_sprint.dataset_expanded import SESSIONS, QUESTIONS; from evaluation.validation_sprint.runner import run_evaluation_arm; qs = [q for q in QUESTIONS if q['split'] == 'test']; run_evaluation_arm('baseline_hybrid_speaker_aware', qs, SESSIONS, resume=True)"
   ```

4. **Arm D: IVARA Hybrid (No Graph, No Temporal)**
   ```powershell
   .\.venv\Scripts\python.exe -c "from evaluation.validation_sprint.dataset_expanded import SESSIONS, QUESTIONS; from evaluation.validation_sprint.runner import run_evaluation_arm; qs = [q for q in QUESTIONS if q['split'] == 'test']; run_evaluation_arm('ivara_hybrid_no_graph', qs, SESSIONS, resume=True)"
   ```

5. **Arm E: IVARA Full Architecture**
   ```powershell
   .\.venv\Scripts\python.exe -c "from evaluation.validation_sprint.dataset_expanded import SESSIONS, QUESTIONS; from evaluation.validation_sprint.runner import run_evaluation_arm; qs = [q for q in QUESTIONS if q['split'] == 'test']; run_evaluation_arm('ivara_full', qs, SESSIONS, resume=True)"
   ```

### Running Statistical Analysis & Metric Aggregations
To recompute all Wilson 95% CIs, McNemar tests, bootstrap difference distributions, and export CSV/JSON summaries from existing predictions:
```powershell
.\.venv\Scripts\python.exe -c "from evaluation.validation_sprint.dataset_expanded import QUESTIONS; from evaluation.validation_sprint.runner import compare_and_analyze; qs = [q for q in QUESTIONS if q['split'] == 'test']; compare_and_analyze(qs)"
```

### Running Unit & Regression Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

