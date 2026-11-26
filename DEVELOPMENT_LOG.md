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

## 6. Phase 1B — Core Conversational Reliability (`2026-11-26`)
- **Objective:** Resolve the Phase 0 interactive chat defects (`ConversationStore` pagination, unconditional QA lookup, self-retrieval/assistant-claim contamination, `conflicting_evidence` dropping, `num_predict=256` budget, question/hypothetical policy false positives) and bring the offline test suite to 100% pass.
- **Important Decisions:**
  - Introduced `get_recent_messages()` and `get_recent_turns()` (token-aware tail queries) rather than breaking forward pagination in `get_messages()`/`get_turns()`.
  - Routed utterances pre-retrieval with `classify_turn_intent()` (5 categories) without adding an extra LLM routing call.
  - Preserved `include_assistant_turns=False` by default for factual recall while enabling `include_assistant_turns=True` for explicit assistant-recall questions.
- **Key Affected Files:** `src/conversations/store.py`, `src/chat/engine.py`, `src/chat/terminal.py`, `src/unified_retrieval/episodic.py`, `src/unified_retrieval/pipeline.py`, `src/answering.py`, `src/extraction/policy.py`, `tests/test_query_aware_retrieval.py`, `tests/test_unified_retrieval.py`, `tests/test_conversation_store.py`, `tests/test_conversational_reliability.py`.
- **Tests & Measured Results:** Offline `pytest -q` passed **`533 passed, 0 failed`** in `15.05s` (fixed the 2 prior Ollama-dependent failures and added 8 regression tests). Live Ollama multi-turn validation could not be executed because the local Ollama server (`v0.33.2`) failed to initialize in this environment (`Unable to init instance`).
- **Remaining Limitations:** Live Ollama generation/latency validation remains pending; `policy.py` lacks cross-turn coreference; `memories.json` remains append-only (Phase 2); `ChatEngine` still uses `UnifiedRetriever` instead of `ChunkRetriever` (Phase 3).
- **Commit References:** Phase 1A committed at `e71f3fd` (`refactor/repo-cleanup`).

## 7. Phase 1B Follow-Up & Live Corrective Patch (`2026-11-26`)
- **Objective:** Address Developer Tooling requirements and fix failures identified in live user terminal transcript (`ivara_smoke_test.txt`):
  1. *Dialogue History Recall vs Memory QA:* Queries about ongoing dialogue ("what were teh last two topics we discussed", "what was my last chat to you") routed to memory QA and searched `data/memories.json` instead of active conversation history.
  2. *Dialogue Recall Route & Context Disambiguation:* Added `dialogue_recall` intent in `classify_turn_intent()` with typo resilience (`teh`). Provided chronological synthesis prompt instructions differentiating current query from preceding turn.
  3. *Memory Formation Safety on Questions:* Question starters without trailing `?` ("what was my last chat to you just now") previously triggered spurious memory `m004` due to `was` matching declarative verbs. Updated `is_pure_question_or_recall()` and `has_personal_declarative_content()` in `policy.py` to identify question clauses regardless of punctuation. Preserved existing owner memories (`m001`–`m004`) untouched in `data/memories.json`.
  4. *Slash Command Error Handling & Alias:* Unknown `/` commands in `src/chat/terminal.py` now output a command error and never pass to the LLM; added `/memory` alias for `/memories`.
  5. *Generation Reliability & Truncation Safeguard:* Prevented empty strings from being silently persisted or returned; increased `conversational_num_predict` to 2048; added thinking-token budget exhaustion fallback.
  6. *Developer Tooling & Sandbox:* Installed official Graphify tool in isolated environment (`.tools/graphify_env/`) and generated deterministic code graph (`graphify-out/`, 2,038 nodes, 5,430 edges) without LLM. Added `--sandbox` / `-s` mode to `main.py chat` and created `scripts/run_isolated_chat.py` for safe manual testing using temporary isolated runtime directories.
  7. *Operating Rules:* Added Rules 16, 17, and 18 to `AGENTS.md` covering Graphify navigation, tiered verification discipline, and mandatory owner manual acceptance testing.
- **Tests & Measured Results:** Full offline test suite passed with `538 passed, 0 failed` in `22.31s`.

## 8. Phase 1C — Conversational Intelligence, Context Unification & Runtime Reliability (`2026-11-26`)
- **Objective:** Replace brittle phrase-list routing with unified context assembly, generalised conversational understanding, configurable assistant personality, creative diversity, and robust Ollama metadata/error handling.
- **Key Architectural Decisions:**
  - **Unified Context Assembly (`TurnContext`):** Assembled `recent_messages` (bounded by `history_message_limit=16`, `history_token_budget=2400`) once per turn and passed it across conversational, mixed-reasoning, and `personal_recall` routes (`<active_conversation_context>` in `src/answering.py` for pronoun/reference resolution without weakening `<evidence_context>` grounding).
  - **Generalised Routing & Deterministic Fast-Path:** Replaced phrase-specific `DIALOGUE_HISTORY_PATTERNS` with structural `DIALOGUE_META_PATTERNS` + `FOLLOWUP_ANAPHORA_PATTERNS`; added deterministic `ConversationStore` lookup for exact previous-message requests (0 LLM calls) and active-dialogue recovery if `personal_recall` abstains on a meta-conversational cue.
  - **Configurable Personality (`AssistantPersona`) & Creative Diversity:** Separated tone/conciseness/anti-clutter rules from factual grounding; added `_is_creative_request()` detection with novelty instructions and `temperature >= 0.85` for creative turns while keeping factual QA at `0.0`.
  - **Ollama Metadata & Thinking-Exhaustion Recovery:** Live diagnosis of `gpt-oss:latest` and `VladimirGav/gemma4-26b-16GB-VRAM:latest` confirmed both are thinking models whose `num_predict` budget is shared with internal `thinking` tokens (explaining empty `content=""` when `done_reason="length"`). Added `_call_ollama_with_metadata()` with a single bounded retry on empty output, explicit `[Note: Response reached the output token limit...]` on `done_reason="length"`, and optional `/debug` diagnostics (`ChatTurnResult.diagnostics`).
- **Key Affected Files:** `src/chat/engine.py`, `src/chat/terminal.py`, `src/answering.py`, `src/extraction/policy.py`, `tests/test_conversational_reliability.py`.
- **Verification Status:**
  - *Implementation:* Complete.
  - *Offline Verification:* `pytest -q` passed **`543 passed, 0 failed`** in `27.29s`.
  - *Live Inference Validation (Isolated Temp Store):* Verified 10-turn sequence on `gpt-oss:latest` (concise explanations, non-stock jokes, `"What subjects have we covered so far?"` answered accurately, deterministic `0.00s` exact previous-message recall, Sherlock Holmes memory formation/recall, Ferrari hypothetical rejection/abstention).
  - *Owner Acceptance Testing:* Pending owner interactive session.
