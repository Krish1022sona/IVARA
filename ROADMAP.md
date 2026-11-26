# IVARA — Living Roadmap

**Current Phase:** Phase 1A Complete (Repository Consolidation & Documentation Hygiene) -> Ready for **Phase 1B** (Conversational Reliability Fixes, pending owner approval).

---

## Completed Milestones
- **Core Memory & Ontology Baseline (`0e06adc`):** Local extraction, validation, entity/relationship resolution, JSON/FAISS retrieval.
- **Episodic Store, Graph, Temporal & LongMemEval Pipeline (`55a273b` – `12cba9b`):** SQLite `ConversationStore`, `SQLiteGraphStore`, `TurnAwareChunker`, `ChunkRetriever`, TACC calendar arithmetic, 58.40% LongMemEval-S Candidate B freeze.
- **Phase 0 Comprehensive Audit & Phase 1A Hygiene (`2026-11-26`):** Verified all conversational bugs, reconciled benchmark numbers from raw JSONLs, untracked `data/memories.json`, consolidated docs, and archived legacy clutter to `IVARA_ARCHIVE/`.

---

## Known Bugs & Active Technical Debt (Verified in Phase 0)
1. **Unconditional Memory Lookup & Intent Blindness (`src/chat/engine.py:27,143-173`):** `is_general_knowledge_query` is unused; greetings (`"How are you?"`) and general questions fall into `insufficient_evidence`, and statements run 2–3 LLM calls.
2. **Self-Retrieval & Assistant Contamination (`src/chat/engine.py:119`, `src/unified_retrieval/episodic.py`):** User turn is saved to `episodic_turns` before retrieval; `EpisodicRetriever` retrieves the current turn (`0.627`) and past assistant errors (`0.575`).
3. **Oldest-6 Pagination Bug (`src/conversations/store.py:666-668, 904-906`):** `get_messages(limit=6, chronological=True)` returns `sequence_num` `1..6` forever after turn 3.
4. **Dropped `conflicting_evidence` Status (`src/chat/engine.py:162`):** Discards valid answers returned with `status="conflicting_evidence"`.
5. **Reasoning Budget Starvation (`src/chat/engine.py:300`):** `num_predict=256` + terse system prompt clips replies on reasoning-capable local models (`gpt-oss`).
6. **Heuristic Memory Policy Errors (`src/extraction/policy.py`):** Ignores `llm_caller`, drops cross-turn facts (`"It has an RTX 5080."`), extracts first-person questions (`"Which GPU does my laptop have?"`) and hypotheticals (`"Suppose I had a dog named Bruno."`), and misses updates (`is_update=False`).
7. **Benchmark-to-Chat Split & Router Hygiene:** `ChatEngine` uses unchunked `UnifiedRetriever` without `question_date`; `router.py:169-203` has duplicate `known_entities` init and synthetic test entities (`"hyperscale"`, `"aetheria"`, `"vikram rao"`).
8. **Offline Test Failures (`2/525`):** `test_query_aware_answering_flow` and `test_12_answer_question_abstention_on_unrecorded_facts` call live Ollama unmocked.

---

## Prioritised Next Steps

### Phase 1B — Core Interactive Chat Reliability (P0)
- Fix `get_messages()`/`get_turns()` tail window and SQLite connection `.close()` lifecycle.
- Add pre-retrieval intent routing (`chitchat`, `general`, `temp_instruction`, `memory_query`, `fact_statement`).
- Persist user turn after retrieval (or exclude active turn ID & assistant self-echoes), accept `"conflicting_evidence"`, pass `question_date`, and raise conversational `num_predict >= 1024`.
- Mock Ollama in the 2 offline-failing tests and add real multi-turn conversational regression tests.
- **Acceptance Criteria:** Live Ollama multi-turn probes (general Qs, jokes, context past turn 6, greetings) behave naturally without false `insufficient_evidence` refusals; `pytest` passes `525/525` offline.

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
