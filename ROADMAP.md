# IVARA — Living Roadmap

**Current Phase:** Phase 1C Complete (Conversational Intelligence, Context Unification & Runtime Reliability; `543 passed` offline + isolated live `gpt-oss` diagnostic verified; owner manual acceptance testing pending) -> Ready for **Phase 2** (Automatic Memory Formation, Coreference & AUDN Lifecycle, pending owner approval).

---

## Completed Milestones
- **Core Memory & Ontology Baseline (`0e06adc`):** Local extraction, validation, entity/relationship resolution, JSON/FAISS retrieval.
- **Episodic Store, Graph, Temporal & LongMemEval Pipeline (`55a273b` – `12cba9b`):** SQLite `ConversationStore`, `SQLiteGraphStore`, `TurnAwareChunker`, `ChunkRetriever`, TACC calendar arithmetic, 58.40% LongMemEval-S Candidate B freeze.
- **Phase 0 Comprehensive Audit & Phase 1A Hygiene (`e71f3fd`, `2026-11-26`):** Verified conversational bugs, reconciled benchmark numbers from raw JSONLs, untracked `data/memories.json`, consolidated docs, and archived legacy clutter to `IVARA_ARCHIVE/`.
- **Phase 1B Core Conversational Reliability (`2026-11-26`):** Added `get_recent_messages()`/`get_recent_turns()` with SQLite cleanup, `classify_turn_intent()` pre-retrieval routing, active-turn & assistant-claim exclusion, `conflicting_evidence` + local reference-date handling, general-purpose conversational prompt (`num_predict=1024`), question/hypothetical policy guards, and `533 passed` offline `pytest` suite.
- **Phase 1B Follow-Up & Live Corrective Patch (`2026-11-26`):** Integrated official Graphify tool (`.tools/graphify_env/`) with deterministic code graph (`graphify-out/`), added `main.py chat --sandbox` and `scripts/run_isolated_chat.py`, routed active dialogue history questions (`dialogue_recall`) with typo tolerance and turn disambiguation, prevented question-based memory extraction (`m004` regression), hardened slash commands (`/memory` alias + unknown command blocking), guarded against empty/truncated Ollama responses (`num_predict=2048`), added Rules 16-18 to `AGENTS.md`, and passed `538 passed` offline tests.
- **Phase 1C Conversational Intelligence, Context Unification & Runtime Reliability (`2026-11-26`):** Implemented unified `TurnContext` assembly shared across conversational and `personal_recall` routes (`<active_conversation_context>`), generalised structural intent routing + deterministic previous-message lookup, configurable `AssistantPersona`, creative diversity prompt/temperature, Ollama completion metadata + single bounded retry on thinking-token exhaustion, optional `/debug` turn diagnostics, and `543 passed` offline tests.

---

## Known Bugs & Active Technical Debt
1. **Live Ollama Multi-Turn Validation Pending:** Host Ollama daemon failed to start in the Phase 1B environment; live model behavior and response latency still require interactive verification.
2. **Heuristic Memory Policy & Missing Coreference (`src/extraction/policy.py`):** Ignores `llm_caller`, drops cross-turn anaphora (`"It has an RTX 5080."`), and lacks semantic update detection (`is_update=False`).
3. **Append-Only Memory Store (`src/memory.py`):** Lacks `ADD / UPDATE / SUPERSEDE / DELETE` (AUDN) consolidation; `EpisodicRetriever` re-embeds SQLite turns per query.
4. **Benchmark-to-Chat Split & Router Hygiene:** `ChatEngine` uses unchunked `UnifiedRetriever` instead of `ChunkRetriever`; `router.py:169-203` has duplicate `known_entities` init and synthetic test entities (`"hyperscale"`, `"aetheria"`, `"vikram rao"`).

---

## Prioritised Next Steps

### Phase 1B Follow-Up — Live Ollama Smoke Validation
- Run live interactive multi-turn diagnostics against `gpt-oss` once `ollama serve` is active, and measure response latency across general, creative, memory-recall, and mixed turns.

### Phase 2 — Automatic Memory Formation, Coreference & Lifecycle (P1)
- Pass recent conversation context to formation policy & extractor to resolve cross-turn anaphora and reject questions/hypotheticals.
- Replace append-only `memories.json` writes with SQLite-backed `ADD / UPDATE / SUPERSEDE / DELETE` (AUDN) lifecycle and persisted embeddings.
- **Acceptance Criteria:** Cross-turn facts extracted accurately; hypotheticals/questions never stored; preference/fact updates supersede cleanly.

### Phase 3 — Retrieval Unification & Code Consolidation (P1)
- Unify interactive chat onto sliding-window `ChunkRetriever` ($W=3, O=1$, Session-MMR); clean `router.py`; modularize `src/answering.py` and consolidate redundant `src/` modules safely.

### Phase 4 — Terminal UX & Inference Ergonomics (P2)
- Token streaming, `rich` markdown rendering, `prompt_toolkit` input history, startup Ollama `/api/tags` check, asynchronous post-reply extraction.

### Phase 5 — Long-Term Associative Memory & >=80% Benchmark Goal (P2/P3)
- **Product-First Framework Evaluation:** Benchmark proven open-source approaches (e.g. HippoRAG, Microsoft GraphRAG, Mem0) against accuracy, licensing, local latency, VRAM, privacy, and maintainability before deciding whether to adopt, adapt, or replace IVARA's graph components.
- **Associative Memory Research:** Investigate query-dependent graph expansion, weighted/typed edges, spreading activation / Personalized PageRank, dynamic hop budgets, hub-noise prevention, provenance-backed inference, and observed-vs-hypothesis separation.
- **Deferred Innovation (IMA):** Keep InterMemory Attention (IMA) as an unverified theoretical proposal; revisit lightweight external attention retrieval only after strong conversational and retrieval baselines are established.
- **Deferred Out-of-Scope:** GUI, voice interface, and external third-party cloud integrations.
