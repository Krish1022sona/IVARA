# IVARA — Chronological Development Log

Single authoritative development history consolidating verified milestones, experimental findings, and cleanup actions.

---

## 1. Early Prototypes & Ontology Foundation (`196387d` – `0e06adc`)
- **Objective:** Build initial local vector memory (`memories.json`, `all-MiniLM-L6-v2`, FAISS `IndexFlatIP`), LLM entity/relation extraction (`gpt-oss`), and ontology validation.
- **Decisions & Files:** Implemented `src/memory.py`, `src/retrieval.py`, `src/embedding.py`, `src/extraction/{extractor,pipeline}.py`, and `src/ontology/*` (29 tracked files at `0e06adc`).
- **Limitations Discovered:** Append-only JSON storage lacked temporal updates/supersession and relational traversal.

## 2. Episodic Store, Graph Retrieval & Unified Pipeline (`ddf60e6` – `55a273b`)
- **Objective:** Preserve verbatim dialogue in SQLite, add bounded knowledge-graph traversal, temporal reasoning, and evaluate against LongMemEval-S.
- **Key Changes:** Added `ConversationStore` (`src/conversations/store.py`), `SQLiteGraphStore` (`src/graph/`), `TurnAwareChunker` & `ChunkRetriever` (`src/chunking/`), `UnifiedRetriever` (`src/unified_retrieval/`), `ChatEngine` (`src/chat/`), and `evaluation/runners/run_longmemeval.py`.
- **Measured Results & Negative Findings:**
  - **4-Arm Temporal Ablation ($N=133$):** Temporal Event Graph traversal alone yielded `+0.00 pp` over flat baseline (`46.62%` -> `46.62%`), while deterministic calendar arithmetic + chronological synthesis drove `+5.26 pp` (`46.62%` -> `51.88%`).
  - **Unbounded Graph Traversal:** Hub overconnectivity degraded held-out accuracy (`43.33%` -> `40.00%`, `-3.33 pp`) and tripled latency, establishing the need for hub suppression (`<= 10` edges) and query-dependent safeguards.
  - **Phase 3.1 Hard Session Routing (`evidence_validated`):** Explicit NO-GO (`-3.76 pp` due to regressions on multi-session questions needing peripheral context).
  - **Mem0 Audit (`885298c`):** Documented IVARA's missing `ADD/UPDATE/DELETE` (AUDN) consolidation lifecycle.

## 3. Stage 2 LongMemEval-S Bakeoff & Freeze (`12cba9b`, `v1.0.0-freeze`)
- **Objective:** Compare Candidate E (Baseline Config D), Candidate B (Arm B: Flat MMR $\lambda=0.70$, $\le 2$ chunks/session + TACC calendar + regex fallback parser), and Candidate C (`QueryAwareRouter`) across all 500 LongMemEval-S questions using `qwen2.5:14b` official judge.
- **Measured Results:** Candidate B scored **58.40% (292/500, 61.34% macro)** vs. Candidate C **56.80% (284/500, 60.54% macro)** and Candidate E **40.80% (204/500, 45.10% macro)** ($\chi^2 = 67.58, p = 2.02 \times 10^{-16}$ for B vs. E). Candidate C gained +5 on Category H (`8/41` vs `3/41`) but lost -10 on single-session temporal (`51/86` vs `61/86`).

## 4. Initial Public GitHub Publication & Curation (`c75ef67` – `7c72a91`)
- **Changes:** Published `main` (`c75ef67`) and ran repository curation (`c5fa41d`, `a4f10d2`, `7c72a91`) excluding local archives/datasets via `.gitignore`.
- **Problems Introduced (Caught in Phase 0 Audit):** Cleanup rewrites in `README.md`, `evaluation/FINAL_RESULTS.md`, and `evaluation/METHODOLOGY.md` introduced numerical errors (299 vs. 292 task sum, inflated Category H claim, wrong Candidate E/McNemar/Recall@5 figures); `data/memories.json` remained tracked in Git.

## 5. Phase 0 Audit & Phase 1A Safe Consolidation (`2026-11-26`)
- **Objective:** Audit the full codebase and conversational bugs without modifying runtime logic, then execute safe repository consolidation, data separation, and documentation truth-alignment.
- **Decisions:**
  - Prioritize reliable terminal conversational chat and memory formation before pursuing `>= 80%` LongMemEval-S.
  - Preserve all graph/ontology/traversal code and research history for IVARA's associative-memory vision; adopt proven open-source solutions where superior (product-first philosophy); keep InterMemory Attention (IMA) as an unverified future hypothesis.
- **Changes Made:**
  - Created `checkpoint/pre-phase1a-cleanup` at `7c72a91` and verified SQLite/JSON backups in sibling `IVARA_ARCHIVE/`.
  - Untracked `data/memories.json` via `git rm --cached` + `.gitignore` (local SHA-256 `28db52d35bb9efba71b1c9bb8f521b4f1d2784bbce621293d9056fc0458883e8` preserved).
  - Removed `IVARA_FILE_TREE.txt`, `scratch/`, and caches; archived 5 zero-test pre-Antigravity scripts, `data/backups/`, `archive/`, 11 redundant docs, and superseded legacy evaluation artifacts to `IVARA_ARCHIVE/`.
  - Reconciled `evaluation/FINAL_RESULTS.md`, `evaluation/METHODOLOGY.md`, and `README.md` directly against raw `final_bakeoff` JSONLs; established `ARCHITECTURE.md`, `DEVELOPMENT_LOG.md`, `ROADMAP.md`, and `AGENTS.md`.
- **Tests & Remaining Limitations:** Offline `pytest` baseline remains `523 passed, 2 failed` (both failures due to unmocked Ollama calls when daemon is off). Conversational routing, pagination, and memory-policy bugs are documented in `ROADMAP.md` for Phase 1B.
