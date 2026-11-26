# IVARA — System Architecture

This document describes IVARA's **actual implemented architecture** as of Phase 1C, distinguishing active interactive runtime paths from batch benchmark paths and clearly identifying known limitations.

---

## 1. Repository Directory Structure

```text
IVARA/
├── .github/                  # Issue & PR templates
├── data/                     # Sanitized seed fixtures (entities.json, memories.example.json) + gitignored runtime state
├── docs/
│   ├── DEMO_GUIDE.md         # Walkthrough for demo.py
│   └── research/             # Academic paper brief
├── evaluation/               # LongMemEval harnesses (adapters, evaluators, runners), methodology, & sealed final_bakeoff/
├── scripts/                  # Manual E2E verification script (test_user_journey.py)
├── src/
│   ├── answering.py          # Prompt assembly, retrieval dispatch, Ollama call, JSON/regex parser, citation ID check
│   ├── embedding.py          # Singleton all-MiniLM-L6-v2 offline-safe loader
│   ├── memory.py             # Append-only JSON persistence (data/memories.json)
│   ├── retrieval.py          # Per-call FAISS IndexFlatIP over memories.json
│   ├── chat/                 # ChatEngine & blocking CLI REPL (terminal.py)
│   ├── conversations/        # SQLite ConversationStore (messages + episodic_turns + FTS5)
│   ├── extraction/           # Rule-based policy.py, LLM extractor.py, validation pipeline.py
│   ├── chunking/             # TurnAwareChunker(W=3, O=1), ChunkRetriever (Candidate B), & experimental strategies
│   ├── graph/                # SQLiteGraphStore, GraphBuilder, GraphSchema, bounded traversal
│   ├── ontology/             # Closed vocabulary (vocabulary.json), entity/relation resolvers, validator
│   ├── query/                # Query decomposition
│   ├── reasoning/            # Contradiction analyzer, evidence integrator, evidence gate
│   ├── temporal/             # Calendar arithmetic, resolution, supersession, event graph modules
│   └── unified_retrieval/    # Episodic/Semantic/Graph fusion (RRF-P), router, safeguards
└── tests/                    # Automated pytest suite
```

---

## 2. Actual Runtime vs. Benchmark Execution Flows

IVARA currently executes **two distinct retrieval paths** depending on entry point:

```mermaid
flowchart TD
    User([CLI User]) --> Main["main.py (chat [--sandbox] / ask / chats / memories)"]
    Main -->|chat| Term["src/chat/terminal.py (slash commands, /debug diagnostics toggle, & exit)"]
    Term --> Engine["src/chat/engine.py: ChatEngine.process_turn()"]
    Engine -->|1. Persist user turn immediately| DB["ConversationStore (data/conversations.db)"]
    Engine -->|2. Unified context + intent| Ctx["TurnContext (_assemble_turn_context, history_limit=16, budget=2400 tok) -> classify_turn_intent()"]
    Ctx -->|memory_management| MemMgmt["Local memory stats / CLI guidance"]
    Ctx -->|exact prev msg| Det["Deterministic ConversationStore lookup (0 LLM calls)"]
    Ctx -->|dialogue_recall| Dlg["Unified TurnContext -> Dialogue-History LLM Synthesis"]
    Ctx -->|personal_recall| Ans["answer_question(use_unified=True, recent_conversation_context, question_date=local_today)"]
    Ans --> UR["UnifiedRetriever (unchunked turns + memories.json + SQLiteGraphStore)"]
    UR --> Gen["Ollama chat(gpt-oss, temp=0, num_ctx=8192, num_predict=1024)"]
    Gen -->|answered / partial / conflicting_evidence + citations| Reply([Grounded Reply])
    Gen -->|insufficient_evidence / unanswerable| Insuff([Abstention Reply / Dialogue Fallback if meta-cue])
    Ctx -->|mixed_personal_reasoning| Mix["Retrieve untrusted memory snippets -> Conversational LLM"]
    Ctx -->|general_knowledge / general_conversation| Conv["Unified TurnContext + AssistantPersona -> Ollama(num_predict=2048, temp=0.7/0.85)"]
    Reply & Insuff & Mix & Conv & MemMgmt & Dlg & Det --> Safe["Metadata check (done_reason, thinking-exhaustion retry, truncation note)"]
    Safe --> Policy["evaluate_memory_formation() (non-recall, non-dialogue user turns only) -> memories.json + memory_graph.db"]

    Bench["evaluation/runners/run_longmemeval.py (Config D)"] -->|use_chunking=True, question_date| AnsB["answer_question()"]
    AnsB --> Chunk["src/chunking/retrieval.py: ChunkRetriever (W=3,O=1 + FAISS + BM25 + Session-MMR)"]
    AnsB --> Cal["src/temporal/calendar_arithmetic.py + resolution.py"]
```

---

## 3. Core Subsystem Responsibilities

| Package / Module | Responsibility | Implementation Notes |
|---|---|---|
| `main.py`, `src/chat/` | CLI commands, interactive REPL, turn orchestration | `ChatEngine` uses `use_unified=True`, `use_adaptive_routing=False`, `min_score=0.30`. Blocking `input()` loop without streaming. |
| `src/conversations/store.py` | Persistent chats, messages, and episodic turns in SQLite (`WAL` + `FTS5`) | Stores turns redundantly in `messages` and `episodic_turns`. |
| `src/extraction/` | Formation gating (`policy.py`), structured LLM extraction (`extractor.py`), resolution & graph sync (`pipeline.py`) | `policy.py` is 100% heuristic (`llm_caller` ignored); `memory.py` is strictly append-only. |
| `src/ontology/` & `src/graph/` | Canonical entity/relation vocabulary, validation, and `SQLiteGraphStore` (`nodes`, `edges`) | Preserved as the foundation for IVARA's associative memory & relationship-driven inference vision. |
| `src/unified_retrieval/` | Tri-source retrieval (episodic + semantic + graph) with `UnifiedEvidenceFuser` (RRF-P) and `GraphSafeguards` | `EpisodicRetriever` embeds SQLite turns dynamically per query; `router.py` contains legacy synthetic entity lists. |
| `src/chunking/` | Sliding-window turn chunking (`TurnAwareChunker`), `ChunkRetriever` (Candidate B), and experimental retrievers | `ChunkRetriever` powers the 58.40% LongMemEval-S benchmark run but is not yet wired into `ChatEngine`. |
| `src/temporal/` & `src/reasoning/` | Deterministic calendar arithmetic, temporal resolution/supersession, contradiction detection, and event-graph modules | Calendar arithmetic drove `+5.26 pp` in temporal ablation; `event_extractor.py` & `relationship_builder.py` currently have no runtime callers. |
| `src/answering.py` | System/user prompt construction, Ollama `AnswerPayload` JSON generation, regex recovery, ID-based citation validation | 1,080-line module handling 6 retrieval branches. Citation validation checks key inclusion only (not NLI). |

---

## 4. Storage & Inference Configuration

- **Storage:**
  - `data/conversations.db`: SQLite database (`conversations`, `messages`, `episodic_turns`, `episodic_turns_fts`).
  - `data/memory_graph.db`: SQLite database (`nodes`, `edges`). Auto-synced from `data/memories.json` in `main.py` only when `nodes == 0` and `memories.json` is non-empty.
  - `data/memories.json`: Untracked runtime JSON array (`src/memory.py`); initialized empty if absent.
  - `data/entities.json`, `data/memories.example.json`: Tracked sanitized seed fixtures.
- **Inference (`Ollama` at `http://localhost:11434`):**
  - Default generator model: `gpt-oss` (`num_ctx=8192`, `num_predict=1024`, `temperature=0` in `answering.py`; `num_ctx=16384`, `num_predict=4096` in `extractor.py`; `num_ctx=8192`, `num_predict=2048`, `temperature=0.7` in `_generate_conversational_response`).
  - Dense embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, offline cache-first).
- **Developer Tooling & Sandbox:**
  - Graphify Code Graph: Official Graphify tool integrated in isolated virtual environment (`.tools/graphify_env/`). Generates deterministic code AST map in `graphify-out/` (2,038 nodes, 5,430 edges) without LLM/paid APIs.
  - Safe Manual Testing: `python main.py chat --sandbox` or `python scripts/run_isolated_chat.py` provisions a temporary isolated directory for manual testing with live Ollama, ensuring zero contamination of production user memories or conversations.

---

## 5. Foundational Associative Memory Direction & Implemented Limitations

- **Associative Memory Vision:** IVARA preserves its graph, ontology, traversal, and safeguard modules to support future query-dependent multi-hop discovery, weighted/typed relationships, spreading activation / Personalized PageRank, hub-noise suppression, and separation of observed facts from inferred hypotheses.
- **Verified Remaining Limitations (Post-Phase 1B):**
  1. `ChatEngine` uses unchunked `UnifiedRetriever` rather than Candidate B's `ChunkRetriever` (Phase 3).
  2. `src/extraction/policy.py` remains regex/heuristic-based without cross-turn coreference resolution (`"It has an RTX 5080."`), and `src/memory.py` is strictly append-only without `UPDATE`/`DELETE` supersession (Phase 2).
  3. `EpisodicRetriever` embeds SQLite turns on the fly per query without a persistent vector index (Phase 2).
  4. `src/unified_retrieval/router.py` still contains legacy synthetic entity lists from earlier test fixtures (Phase 3).
  5. Live Ollama multi-turn inference validation remains pending whenever the local Ollama daemon is stopped.
