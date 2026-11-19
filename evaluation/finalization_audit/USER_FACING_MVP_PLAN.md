# IVARA User-Facing MVP — Architecture Inspection & Implementation Plan

**Date:** 2026-11-14  
**Sprint:** IVARA Finalization — User-Facing MVP  
**Phase:** Phase A / Batch 1 (Architecture Inspection & Plan)  
**Branch:** `research/evaluation-baseline`

---

## 1. System Inspection Findings

### 1.1 Database Schemas & Persistence Mechanisms
- **Long-Term Semantic Memories:**
  - File: `data/memories.json`
  - Managed by: `src/memory.py` (`add_memory`)
  - Schema: List of JSON records containing `id` (`m001`), `text`, `timestamp`, `source`, `type`, `importance`, `confidence`, `entities`, `topics`, `relationships`, `temporal_claims`, and `diagnostics`.
- **Canonical Entities & Aliases:**
  - File: `data/entities.json`
  - Managed by: `src/ontology/entity_store.py` (`load_entities`, `save_entities`, `create_entity`)
  - Schema: List of JSON records with `id` (`e001`), `canonical_name`, `aliases`, `types`.
- **Relational & Temporal Memory Graph:**
  - File: `data/memory_graph.db` (SQLite)
  - Managed by: `src/graph/store.py` (`SQLiteGraphStore`)
  - Schema:
    - `nodes`: `id TEXT PRIMARY KEY`, `node_type TEXT`, `label TEXT`, `properties TEXT` (JSON)
    - `edges`: `id TEXT PRIMARY KEY`, `source_id TEXT`, `target_id TEXT`, `edge_type TEXT`, `origin TEXT`, `confidence REAL`, `evidence TEXT`, `memory_id TEXT`, `chunk_id TEXT`, `timestamp TEXT`, `properties TEXT` (JSON)
    - Foreign keys: `FOREIGN KEY (source_id) REFERENCES nodes (id) ON DELETE CASCADE`, `FOREIGN KEY (target_id) REFERENCES nodes (id) ON DELETE CASCADE`.
- **Embeddings & Vector Index (FAISS):**
  - Architecture: **Fully in-memory and dynamic.**
  - Managed by: `src/retrieval.py` (`build_index`, `retrieve`)
  - Finding: FAISS indices (`IndexFlatIP`) are built dynamically from loaded memory records. **There are no binary `.faiss` files stored on disk.** Modifying or deleting records in `data/memories.json` immediately and automatically updates the search index on the next query.
- **Conversation History:**
  - Finding: **No persistent conversation storage currently exists.**
  - `data/` contains only `memories.json` and `entities.json`. Existing turn-chunking code in `src/chunking/turn_chunker.py` was used only for benchmark dataset parsing.
  - Decision: Create an isolated SQLite database `data/conversations.db` storing conversations and messages, completely decoupled from long-term semantic memories.

---

### 1.2 Extracted Memory Persistence Flow
- Unstructured conversation enters `src/extraction/pipeline.py:process_and_store_memory()`.
- `src/extraction/extractor.py:extract_knowledge()` calls the local LLM (`gpt-oss`) to extract entities, topics, and explicit/inferred relationships.
- `src/extraction/pipeline.py:process_extraction()` validates extraction schemas, canonicalizes entities via `src/ontology/entity_resolver.py`, and validates relationship predicates via `src/ontology/validator.py`.
- `src/memory.py:add_memory()` persists the validated record to `data/memories.json`.
- `src/temporal/claim.py:build_claims_from_text()` extracts temporal statements and valid intervals.
- `src/temporal/supersession.py:TemporalSupersessionEngine` reconciles updates/supersessions and syncs nodes/edges into `SQLiteGraphStore`.
- Provenance: The `source` field will link directly to `conv_{conversation_id}_msg_{message_id}`.

---

### 1.3 Retrieval and Answering Integration
- `src/answering.py:answer_question()` already implements:
  - Hybrid semantic retrieval (`retrieve_with_context()`).
  - Evidence gating (`EvidenceGate`) to determine if memories sufficiently ground an answer.
  - Temporal supersession and contradiction analysis (`ContradictionAnalyzer`).
  - Prompt construction with citation guidelines (`[m001]`).
  - LLM invocation with Ollama (`gpt-oss`) or mock caller.
  - Deterministic citation verification (`validate_citations()`).
  - Grounded abstention when no relevant memories exist.
- Conversational Integration:
  - In chat mode, if the query relates to past memories, `answer_question()` provides the grounded facts and citations.
  - If the query is conversational chitchat or follow-up discussion, the assistant uses recent conversation context and responds naturally without hallucinating fake memory citations.

---

### 1.4 Deletion Dependencies & Consistency Rules
1. **Deleting a Conversation (`chat delete <id>`):**
   - Deletes conversation metadata and messages from `data/conversations.db`.
   - *Memory Policy:* Long-term memories formed during that chat represent enduring user knowledge and remain in `data/memories.json` by default (with their historical `source` provenance intact). If `--purge-memories` is explicitly passed, memories whose `source` matches that conversation are deleted.
2. **Clearing Conversations (`chats clear`):**
   - Wipes `data/conversations.db`.
   - Never touches `data/memories.json`, `data/entities.json`, or `data/memory_graph.db`.
3. **Deleting a Memory (`memories delete <id>`):**
   - Removes memory record from `data/memories.json`.
   - Removes corresponding memory node from `data/memory_graph.db` (`DELETE FROM nodes WHERE id = ?`).
   - Removes edges where `memory_id = ?` or `chunk_id = ?`.
   - Removes temporal claim nodes derived from that memory.
   - Dynamic FAISS index on next query automatically excludes the deleted memory.
4. **Clearing Memories (`memories clear`):**
   - Clears `data/memories.json`.
   - Clears graph store via `graph_store.clear()`.
   - Never touches `data/conversations.db` or protected benchmark datasets.
5. **Confirmation Safeguard:**
   - Destructive operations (`chats clear`, `memories clear`) prompt the user `[y/N]`, or accept `--yes` for automated scripting.

---

## 2. Reusable Code Assets

| Component | Source File | Reusability |
| :--- | :--- | :--- |
| Grounded QA & Answering | `src/answering.py` | 100% reusable; provides grounded synthesis and citation validation. |
| Memory Ingestion Pipeline | `src/extraction/pipeline.py` | 100% reusable (`process_and_store_memory`). |
| LLM Knowledge Extraction | `src/extraction/extractor.py` | Reusable with Ollama `gpt-oss`. |
| Memory Persistence | `src/memory.py` | Reusable; add `delete_memory`, `clear_memories`, `get_memory`. |
| Persistent Graph Store | `src/graph/store.py` | Reusable (`SQLiteGraphStore`); add `delete_memory_nodes_and_edges`. |
| Semantic Retrieval | `src/retrieval.py` | 100% reusable (`retrieve_with_context`, `build_index`). |
| Entity Resolution | `src/ontology/entity_resolver.py` | 100% reusable. |
| Temporal Supersession | `src/temporal/supersession.py` | 100% reusable. |

---

## 3. Targeted Implementation Roadmap (Phases B – F)

- **Batch 2 — Conversation Persistence (`src/conversations/store.py`)**
  - Implement SQLite conversation store (`data/conversations.db`):
    - Table `conversations` (`id`, `title`, `created_at`, `updated_at`).
    - Table `messages` (`id`, `conversation_id`, `role`, `content`, `timestamp`, `metadata`).
  - Methods: `create_conversation`, `get_conversation`, `list_conversations`, `add_message`, `get_messages`, `delete_conversation`, `clear_conversations`.
  - Add unit tests verifying CRUD operations, persistence across process restarts, and isolation from memories.

- **Batch 3 — Interactive Chat Engine & Memory Extraction Integration**
  - Build `src/chat/engine.py` coordinating:
    - User input handling and conversational context assembly.
    - Querying memory via `answer_question` for grounded recall.
    - Calling Ollama `gpt-oss` for natural conversational responses.
    - Memory formation filter: extracting facts using `process_and_store_memory` when meaningful personal facts are stated.
    - Graceful fallback when Ollama is offline or unavailable.

- **Batch 4 — Memory Management & Unified CLI (`main.py`)**
  - Implement memory CRUD in `src/memory.py` (`delete_memory`, `clear_memories`, `list_memories`).
  - Add graph deletion in `src/graph/store.py` (`delete_memory_graph_data`).
  - Wire CLI in `main.py` using `argparse`:
    - `python main.py` / `python main.py chat`
    - `python main.py chats list` / `open <id>` / `delete <id>` / `clear`
    - `python main.py memories list` / `search <query>` / `show <id>` / `delete <id>` / `clear` / `stats`
  - Strict `--yes` confirmation flag for destructive operations.

- **Batch 5 — Comprehensive Integration Verification**
  - Write test suite (`tests/test_conversation_mvp.py`) covering:
    - Starting, resuming, and persisting conversations across restarts.
    - Memory extraction, storage, and cross-session retrieval.
    - Independent chat deletion vs memory deletion.
    - Offline / unavailable Ollama handling.
    - Deletion consistency across SQLite graph, memories.json, and in-memory FAISS.
  - Run full existing 248-test regression suite.

