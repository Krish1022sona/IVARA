# IVARA — Release Readiness Checklist
## Production Deployment & Operational Verification Guide
**Milestone:** 3.2 — Batch 5 Release Verification  
**Branch:** `research/evaluation-baseline`  
**Target Version:** IVARA v1.0.0-rc1  
**Date:** November 16, 2026

---

### Pre-Flight Verification Matrix

| Verification Area | Requirement | Measured Result | Status |
| :--- | :--- | :--- | :---: |
| **Test Suite** | $\ge 425$ tests passing, 0 failures, 0 errors | **429 passed** in 23.53s | **PASS** |
| **Realistic Scenarios** | 12 complex personal memory user situations | **12 / 12 passed** deterministic tests | **PASS** |
| **Git Safety** | Clean baseline, 0 commits, 0 pushes | Clean working tree on `research/evaluation-baseline` | **PASS** |
| **Data Integrity** | SHA-256 hashes of 4 production stores unchanged | 4 / 4 hashes match baseline bit-for-bit | **PASS** |
| **Episodic Persistence** | 100% of conversation turns recoverable | Verified via SQLite + FTS5 full-text index | **PASS** |
| **Graph Schema** | Open-world ontology support with zero schema rejects | Verified across canonical and open predicates | **PASS** |
| **Retrieval SLA** | Sub-millisecond SQLite query and FAISS search | SQLite: **0.03 ms**, FAISS: **0.002 ms** | **PASS** |
| **Model Generation SLA** | Response generated in $\le 5$ seconds | Median latency: **2.90 seconds** | **PASS** |
| **Citation Grounding** | 0 hallucinated citations on benchmark | **0 hallucinations** (100.0% citation validity) | **PASS** |
| **Abstention Calibration** | 0 hallucinations on unrecorded/out-of-domain queries | **100.0% correct abstention** rate | **PASS** |
| **Backward Compatibility** | Existing interfaces function with default parameters | `use_adaptive_routing=False` preserves legacy behavior | **PASS** |

---

### 1. Production Database Hashes (Bit-for-Bit Audit)

The four core production databases were verified using SHA-256 cryptographic checksums before and after the evaluation campaign. Zero modifications were made to production data:

- `data/memories.json`:
  `489e3ea2138022f97a590808cee3dfba03465966eda06d750e340d6d0fddf858`
- `data/entities.json`:
  `51f993fbe1cfc7f40084724499cd2dd216b30d8e3aeeace9824d1863d84fc09e`
- `data/memory_graph.db`:
  `f6b9f9cf7d35674e461350ffa87dcfa735ae0d7643c46e30138f56bffb31f95d`
- `data/conversations.db`:
  `a44d2ce7501fd3170147c614ccb59d4f9416e11cd0defd66b370d9442878e89f`

All evaluation runs operated strictly in the isolated sandbox `evaluation/validation_sprint/ivara_store_validation/`.

---

### 2. Backward Compatibility Guarantees

1. **Default Opt-In Routing:**
   In `src/answering.py`, `use_adaptive_routing: bool = False` remains the default parameter. Existing scripts, pipelines, and integrations calling `answer_question(...)` continue to function without modification.
2. **Conversation Storage:**
   `ConversationStore` automatically provisions `episodic_turns` and `fts_turns` tables upon first connection without altering existing `messages` or `conversations` tables.
3. **Graph Storage:**
   `SQLiteGraphStore` open-world predicate resolution is transparent; existing canonical edges retain their types while open-world predicates are normalized seamlessly.

---

### 3. Production Deployment Procedure

To deploy IVARA v1.0.0-rc1 to a production host:

1. **Environment Setup:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Local LLM Host Verification:**
   Ensure Ollama is running with the target model:
   ```bash
   ollama serve
   ollama pull gpt-oss  # or designated local model
   ```
3. **Database Migration Verification:**
   ```bash
   python -c "from src.conversations.store import ConversationStore; ConversationStore('data/conversations.db')"
   ```
4. **Smoke Test Verification:**
   ```bash
   python main.py ask "What company does Arjun work for?" --adaptive
   python main.py ask "In which city was Arjun born?" --adaptive
   ```
5. **Full Test Suite Execution:**
   ```bash
   pytest -q
   ```
   *Expected result: 429 passed in $\le 30$ seconds.*

---

### 4. Rollback Plan

If unexpected regressions occur in production:
1. Revert `use_adaptive_routing=True` to `False` in application configuration. The system immediately reverts to standard single-source hybrid retrieval.
2. If database schema rollback is required, `data/backups/` contains point-in-time snapshots of `conversations.db` and `memory_graph.db`.

