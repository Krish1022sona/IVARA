# IVARA — Milestone 1 Code Review & Robustness Audit

**Review Date:** 2026-11-15  
**Review Target:** Persistent Episodic Memory Layer (Milestone 1)  
**Status:** Validated, Hardened, and Regression Tested  

---

## 1. Code Review of the 10 Critical Invariants

### 1.1 Invariant 1: Consistency Between `messages` and `episodic_turns`
- **Audit:** In `src/conversations/store.py`, every turn-addition entry point (`add_turn`, `add_turns_batch`) wraps both table insertions (`episodic_turns` and `messages`) inside an atomic transaction:
  ```python
  # Insert episodic turn
  conn.execute("INSERT INTO episodic_turns (...) VALUES (...)")
  # Synchronize to messages table
  conn.execute("INSERT OR REPLACE INTO messages (...) VALUES (...)")
  ```
  Legacy methods (`add_message`, `add_messages_batch`) delegate directly to `add_turn` / `add_turns_batch`.
  In `ChatEngine.process_turn`, when memory extraction succeeds, the metadata update updates both `episodic_turns` and `messages` within a single SQLite transaction:
  ```python
  conn.execute("UPDATE episodic_turns SET metadata = ? WHERE turn_id = ?", ...)
  conn.execute("UPDATE messages SET metadata = ? WHERE id = ?", ...)
  ```
  Cascade deletes on `conversations(id)` affect both tables simultaneously.
- **Finding:** No divergence path exists between `messages` and `episodic_turns`.

---

### 1.2 Invariant 2: Duplicate Messages During Retries
- **Audit:** `add_turn` checks for duplicate `turn_id` before inserting:
  ```python
  cur = conn.execute("SELECT turn_id FROM episodic_turns WHERE turn_id = ?", (t_id,))
  if cur.fetchone():
      raise ValueError(f"Turn with ID '{t_id}' already exists.")
  ```
  When a caller provides an idempotency key (`turn_id`), resubmitting the identical key causes a clean `ValueError`, preventing duplicate rows or corrupted sequences.
- **Finding:** Idempotency is strictly enforced when `turn_id` is supplied. Auto-generated IDs without an idempotency key will create new turns, which is standard transactional behavior for distinct messages.

---

### 1.3 Invariant 3: Turn Chaining Correctness Under Concurrent Writes
- **Audit:** Under concurrent multi-threaded writes in SQLite, two threads could read the same `MAX(sequence_num)` or last turn ID before either committed, potentially creating duplicate sequence numbers or branching in the double-linked list.
- **Hardening Applied:** 
  1. Configured connection with `check_same_thread=False` to support multi-threaded access.
  2. Introduced an internal re-entrant lock `self._write_lock = threading.RLock()` in `ConversationStore`.
  3. All mutating writes (`add_turn`, `add_turns_batch`, `create_conversation`, `delete_conversation`, `clear_conversations`) execute under `with self._write_lock, conn:`.
  4. Added regression test `test_concurrent_turn_insertion` executing 20 concurrent thread insertions into the same conversation.
- **Result:** Sequence numbers were verified strictly contiguous ($1 \dots 20$) with continuous, unbranched double-linked pointers (`prev_turn_id`, `next_turn_id`).

---

### 1.4 Invariant 4: Foreign Keys and Transaction Configuration
- **Audit:**
  - `PRAGMA foreign_keys = ON` is explicitly executed on connection initialization.
  - Foreign key constraints:
    - `FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE`
    - `FOREIGN KEY (prev_turn_id) REFERENCES episodic_turns(turn_id) ON DELETE SET NULL`
    - `FOREIGN KEY (next_turn_id) REFERENCES episodic_turns(turn_id) ON DELETE SET NULL`
  - Transactions: All writes run within `with conn:`, ensuring automatic `COMMIT` on completion and `ROLLBACK` on any exception.
- **Finding:** Foreign keys and ACID transactions are correctly configured and verified.

---

### 1.5 Invariant 5: FTS5 Fallback Correctness
- **Audit:** `_init_db` creates `episodic_turns_fts` inside a `try...except sqlite3.OperationalError` block. If SQLite lacks FTS5 support, `self._fts_enabled` is set to `False`.
  In `search_turns`:
  - If `_fts_enabled` is True, it queries the FTS virtual table.
  - If FTS returns 0 rows, raises an error, or if `_fts_enabled` is False, execution falls through directly to the standard `LIKE '%query%'` query.
- **Hardening & Verification:** Added `test_fts_disabled_like_fallback_and_partial_migration`, explicitly setting `store._fts_enabled = False` and confirming exact lexical search matches.

---

### 1.6 Invariant 6: Legacy Database Migrations Safety & Idempotency
- **Audit:** The initial migration checked `if msg_count > 0 and turn_count == 0:`. While safe for blank databases, this would skip backfills if a database had partially migrated messages.
- **Hardening Applied:** Refactored migration in `_init_db()` to be per-row idempotent:
  ```sql
  SELECT m.* FROM messages m
  WHERE m.id NOT IN (SELECT turn_id FROM episodic_turns)
  ORDER BY m.conversation_id, m.sequence_num ASC
  ```
  Unmigrated messages are chained seamlessly onto any existing episodic turns for that conversation.
- **Verification:** Verified via `test_12_migration_from_legacy_messages` and `test_fts_disabled_like_fallback_and_partial_migration`.

---

### 1.7 Invariant 7: Assistant Response Persistence After Failures
- **Audit:** In `src/chat/engine.py`, the retrieval and answering step is wrapped in `try...except Exception as e:`.
  If answering fails (e.g. Ollama offline, model timeout, memory retrieval error):
  - `answer_text` is set to an informative error message.
  - `turn_status = "error"`.
  - Execution proceeds to Step 3, persisting the assistant turn into `episodic_turns` with `metadata={"status": "error", "citations": []}`.
  - Step 4 (memory extraction) is wrapped in its own `try...except`, ensuring extraction failures never affect conversational persistence.
- **Finding:** Assistant error responses and conversational turns are reliably persisted.

---

### 1.8 Invariant 8: Provenance Links Reliability
- **Audit:**
  - Memory extraction provenance:
    - User turn: `user_turn.turn_id`
    - Conversation: `conv.id`
    - Ingested source string: `f"{conv.id}_{user_turn.turn_id}"`
    - Stored on memory: `mem["source"]`
  - Turn metadata provenance:
    - `episodic_turns.metadata["extracted_memory_id"] = mem["id"]`
    - `episodic_turns.metadata["memory_category"] = policy_decision.category.value`
- **Finding:** Bidirectional provenance linking is established and persisted.

---

### 1.9 Invariant 9: `pytest.ini` Modification Audit
- **Audit:** The only change to `pytest.ini` was appending `pythonpath = .`.
  - `testpaths`, `norecursedirs`, `python_files`, `python_classes`, and `python_functions` were completely untouched.
  - Purpose: Allows `pytest` to find modules in the project root across different execution environments without modifying test collection behavior.
- **Finding:** Zero change to test discovery, scoping, or filters.

---

### 1.10 Invariant 10: Complexity of Context Retrieval Query
- **Audit:** In `get_context_window(turn_id, window_size=k)`:
  ```sql
  SELECT * FROM episodic_turns
  WHERE conversation_id = ? AND sequence_num BETWEEN ? AND ?
  ORDER BY sequence_num ASC
  ```
- **Index:** `idx_episodic_conv_seq ON episodic_turns(conversation_id, sequence_num ASC)`
- **Query Plan (`EXPLAIN QUERY PLAN`):**
  `SEARCH episodic_turns USING INDEX idx_episodic_conv_seq (conversation_id=? AND sequence_num>? AND sequence_num<?)`
  - Direct B-Tree descent to `min_seq` takes $O(\log N)$ time.
  - Scanning $2k+1$ entries takes $O(k)$ time.
  - Direct traversal of the index preserves `ORDER BY sequence_num ASC`, eliminating any temporary sort table ($O(1)$ sort overhead).
- **Finding:** Claimed $O(\log N + k)$ retrieval complexity is mathematically and empirically accurate.

---

## 2. Test Suite Status
- Total passing tests: **339 passed** (previously 337 + 2 new targeted concurrency & migration tests).
- 0 regressions across all suites.

