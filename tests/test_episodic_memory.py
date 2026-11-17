"""Comprehensive tests for Milestone 1: Persistent Episodic Memory Layer in IVARA.

Verifies:
1. User-turn persistence.
2. Assistant-turn persistence.
3. Persistence when semantic extraction rejects message.
4. Correct timestamps (ISO 8601 UTC).
5. Strict chronological ordering (sequence_num).
6. Double-linked previous/next turn chaining (prev_turn_id, next_turn_id).
7. Conversation isolation (turns and context windows never cross conversations).
8. Context-window retrieval (+/- 1 and configurable k, with edge boundaries).
9. Duplicate prevention (re-submitting identical turn_id).
10. Database restart persistence across connection close and reopen.
11. Interrupted-session / rollback recovery on transaction failure.
12. Migration compatibility (existing messages migrated into episodic_turns).
"""

from datetime import datetime
import json
import sqlite3
import pytest
from src.conversations.store import ConversationStore, Conversation, Message, EpisodicTurn
from src.chat.engine import ChatEngine, ChatTurnResult


def test_01_user_turn_persistence(tmp_path):
    """Scenario 1: User turn is persisted in episodic_turns with all core fields."""
    db_file = tmp_path / "episodic_test.db"
    store = ConversationStore(db_file)
    conv = store.create_conversation("User Turn Test")

    turn = store.add_turn(
        conversation_id=conv.id,
        role="user",
        content="I prefer light roast coffee in the mornings.",
        speaker="Krishna",
        session_id="sess_001",
        metadata={"client": "cli"}
    )

    assert turn.turn_id.startswith("turn_")
    assert turn.conversation_id == conv.id
    assert turn.speaker == "Krishna"
    assert turn.speaker_role == "user"
    assert turn.content == "I prefer light roast coffee in the mornings."
    assert turn.session_id == "sess_001"
    assert turn.metadata == {"client": "cli"}
    assert turn.sequence_num == 1

    # Verify retrieval from store
    fetched = store.get_turn(turn.turn_id)
    assert fetched is not None
    assert fetched.turn_id == turn.turn_id
    assert fetched.content == turn.content
    assert fetched.speaker == "Krishna"
    store.close()


def test_02_assistant_turn_persistence(tmp_path):
    """Scenario 2: Assistant turn is persisted in episodic_turns with citations and status."""
    db_file = tmp_path / "episodic_test.db"
    store = ConversationStore(db_file)
    conv = store.create_conversation("Assistant Turn Test")

    turn = store.add_turn(
        conversation_id=conv.id,
        role="assistant",
        content="Understood. I will remember your preference for light roast coffee.",
        speaker="IVARA",
        metadata={"status": "answered", "citations": ["mem_001"]}
    )

    assert turn.turn_id.startswith("turn_")
    assert turn.conversation_id == conv.id
    assert turn.speaker == "IVARA"
    assert turn.speaker_role == "assistant"
    assert turn.metadata["status"] == "answered"
    assert turn.metadata["citations"] == ["mem_001"]

    fetched = store.get_turn(turn.turn_id)
    assert fetched is not None
    assert fetched.metadata["status"] == "answered"
    store.close()


def test_03_persistence_when_semantic_extraction_rejects(tmp_path):
    """Scenario 3: Episodic turn persists even when semantic extraction rejects the turn."""
    db_file = tmp_path / "episodic_test.db"
    mem_file = tmp_path / "memories.json"
    mem_file.write_text("[]", encoding="utf-8")
    ent_file = tmp_path / "entities.json"
    ent_file.write_text("{}", encoding="utf-8")

    store = ConversationStore(db_file)
    engine = ChatEngine(
        conversation_store=store,
        memories_path=mem_file,
        entities_path=ent_file,
        llm_caller=lambda q, r: "Acknowledged."
    )

    # Turn with casual phrasing or third-person colleague statement
    # that standard memory extraction policy rejects
    casual_statement = "Good morning! Just dropping in to say hi."
    res = engine.process_turn(casual_statement, auto_extract=True)

    # Semantic memory should reject casual greeting
    assert res.memory_extracted is False
    assert res.extracted_memory_id is None

    # But episodic layer must strictly preserve it!
    assert res.user_turn is not None
    assert res.user_turn.content == casual_statement
    assert store.count_turns() == 2  # user + assistant

    fetched_user_turn = store.get_turn(res.user_turn.turn_id)
    assert fetched_user_turn is not None
    assert fetched_user_turn.content == casual_statement

    # Second turn: third-person collaboration detail (e.g. SIH turn 3)
    colleague_turn = "He handles the backend while I work on the frontend."
    res2 = engine.process_turn(colleague_turn, auto_extract=True)
    # Check that it is securely in the episodic layer regardless of extraction
    assert res2.user_turn is not None
    assert res2.user_turn.content == colleague_turn
    assert store.get_turn(res2.user_turn.turn_id) is not None
    store.close()


def test_04_correct_timestamps_iso8601(tmp_path):
    """Scenario 4: Timestamps follow valid ISO 8601 UTC format."""
    db_file = tmp_path / "episodic_test.db"
    store = ConversationStore(db_file)
    conv = store.create_conversation("Timestamp Test")

    turn = store.add_turn(conv.id, "user", "Testing timestamp formatting.")
    # Must be parseable ISO 8601
    dt = datetime.fromisoformat(turn.timestamp)
    assert dt is not None
    # Must have timezone info
    assert dt.tzinfo is not None
    store.close()


def test_05_strict_chronological_ordering(tmp_path):
    """Scenario 5: Turns strictly increment sequence numbers monotonically."""
    db_file = tmp_path / "episodic_test.db"
    store = ConversationStore(db_file)
    conv = store.create_conversation("Ordering Test")

    t1 = store.add_turn(conv.id, "user", "First turn")
    t2 = store.add_turn(conv.id, "assistant", "Second turn")
    t3 = store.add_turn(conv.id, "user", "Third turn")
    t4 = store.add_turn(conv.id, "assistant", "Fourth turn")

    assert t1.sequence_num == 1
    assert t2.sequence_num == 2
    assert t3.sequence_num == 3
    assert t4.sequence_num == 4

    turns = store.get_turns(conv.id, chronological=True)
    assert [t.sequence_num for t in turns] == [1, 2, 3, 4]

    rev_turns = store.get_turns(conv.id, chronological=False)
    assert [t.sequence_num for t in rev_turns] == [4, 3, 2, 1]
    store.close()


def test_06_double_linked_chaining(tmp_path):
    """Scenario 6: Double-linked previous/next turn pointers are maintained transactionally."""
    db_file = tmp_path / "episodic_test.db"
    store = ConversationStore(db_file)
    conv = store.create_conversation("Chaining Test")

    t1 = store.add_turn(conv.id, "user", "Turn 1")
    t2 = store.add_turn(conv.id, "assistant", "Turn 2")
    t3 = store.add_turn(conv.id, "user", "Turn 3")

    # Refresh from database
    f1 = store.get_turn(t1.turn_id)
    f2 = store.get_turn(t2.turn_id)
    f3 = store.get_turn(t3.turn_id)

    # T1 pointers
    assert f1.prev_turn_id is None
    assert f1.next_turn_id == t2.turn_id

    # T2 pointers
    assert f2.prev_turn_id == t1.turn_id
    assert f2.next_turn_id == t3.turn_id

    # T3 pointers
    assert f3.prev_turn_id == t2.turn_id
    assert f3.next_turn_id is None

    # Test neighbor lookup
    prev_n, next_n = store.get_turn_neighbors(t2.turn_id)
    assert prev_n is not None and prev_n.turn_id == t1.turn_id
    assert next_n is not None and next_n.turn_id == t3.turn_id

    # Edge neighbor lookups
    prev_1, next_1 = store.get_turn_neighbors(t1.turn_id)
    assert prev_1 is None
    assert next_1.turn_id == t2.turn_id

    prev_3, next_3 = store.get_turn_neighbors(t3.turn_id)
    assert prev_3.turn_id == t2.turn_id
    assert next_3 is None
    store.close()


def test_07_conversation_isolation(tmp_path):
    """Scenario 7: Turns and context windows never cross conversation boundaries."""
    db_file = tmp_path / "episodic_test.db"
    store = ConversationStore(db_file)
    conv_a = store.create_conversation("Conversation A")
    conv_b = store.create_conversation("Conversation B")

    t_a1 = store.add_turn(conv_a.id, "user", "Secret from Conv A")
    t_b1 = store.add_turn(conv_b.id, "user", "Topic in Conv B")
    t_a2 = store.add_turn(conv_a.id, "assistant", "Reply in Conv A")

    # Conv A should only have 2 turns
    turns_a = store.get_turns(conv_a.id)
    assert len(turns_a) == 2
    assert all(t.conversation_id == conv_a.id for t in turns_a)

    # Conv B should only have 1 turn
    turns_b = store.get_turns(conv_b.id)
    assert len(turns_b) == 1
    assert turns_b[0].conversation_id == conv_b.id

    # Context window from A must never contain turns from B
    window = store.get_context_window(t_a1.turn_id, window_size=5)
    assert all(t.conversation_id == conv_a.id for t in window)
    assert not any(t.turn_id == t_b1.turn_id for t in window)
    store.close()


def test_08_context_window_retrieval(tmp_path):
    """Scenario 8: Context-window retrieval (+/- window_size) with edge cases."""
    db_file = tmp_path / "episodic_test.db"
    store = ConversationStore(db_file)
    conv = store.create_conversation("Window Test")

    turns = []
    for i in range(1, 6):
        role = "user" if i % 2 == 1 else "assistant"
        turns.append(store.add_turn(conv.id, role, f"Turn {i}"))

    # Target: Turn 3, window_size=1 -> [Turn 2, Turn 3, Turn 4]
    w1 = store.get_context_window(turns[2].turn_id, window_size=1)
    assert [t.content for t in w1] == ["Turn 2", "Turn 3", "Turn 4"]

    # Target: Turn 3, window_size=2 -> [Turn 1, Turn 2, Turn 3, Turn 4, Turn 5]
    w2 = store.get_context_window(turns[2].turn_id, window_size=2)
    assert [t.content for t in w2] == ["Turn 1", "Turn 2", "Turn 3", "Turn 4", "Turn 5"]

    # Target: Turn 1 (left boundary), window_size=1 -> [Turn 1, Turn 2]
    w_left = store.get_context_window(turns[0].turn_id, window_size=1)
    assert [t.content for t in w_left] == ["Turn 1", "Turn 2"]

    # Target: Turn 5 (right boundary), window_size=1 -> [Turn 4, Turn 5]
    w_right = store.get_context_window(turns[4].turn_id, window_size=1)
    assert [t.content for t in w_right] == ["Turn 4", "Turn 5"]

    # Target: window_size=0 -> [Turn 3]
    w0 = store.get_context_window(turns[2].turn_id, window_size=0)
    assert [t.content for t0 in [w0] for t in t0] == ["Turn 3"]

    # Negative window_size should raise ValueError
    with pytest.raises(ValueError):
        store.get_context_window(turns[2].turn_id, window_size=-1)

    # Non-existent turn returns empty list
    assert store.get_context_window("turn_nonexistent") == []
    store.close()


def test_09_duplicate_prevention(tmp_path):
    """Scenario 9: Attempting to insert duplicate turn_id raises error."""
    db_file = tmp_path / "episodic_test.db"
    store = ConversationStore(db_file)
    conv = store.create_conversation("Duplicate Test")

    t1 = store.add_turn(conv.id, "user", "Original turn", turn_id="turn_fixed_001")
    assert t1.turn_id == "turn_fixed_001"

    with pytest.raises(ValueError, match="already exists"):
        store.add_turn(conv.id, "user", "Duplicate turn", turn_id="turn_fixed_001")

    # Verify count remains 1
    assert store.count_turns(conv.id) == 1
    store.close()


def test_10_database_restart_persistence(tmp_path):
    """Scenario 10: All data and pointers survive database close and reopen."""
    db_file = tmp_path / "persistent_episodic.db"
    store1 = ConversationStore(db_file)
    conv = store1.create_conversation("Restart Test")

    t1 = store1.add_turn(conv.id, "user", "Turn 1 before restart")
    t2 = store1.add_turn(conv.id, "assistant", "Turn 2 before restart")
    store1.close()

    # Reopen fresh connection
    store2 = ConversationStore(db_file)
    assert store2.count_turns(conv.id) == 2

    reloaded_t1 = store2.get_turn(t1.turn_id)
    reloaded_t2 = store2.get_turn(t2.turn_id)

    assert reloaded_t1 is not None and reloaded_t1.content == "Turn 1 before restart"
    assert reloaded_t2 is not None and reloaded_t2.content == "Turn 2 before restart"
    assert reloaded_t1.next_turn_id == t2.turn_id
    assert reloaded_t2.prev_turn_id == t1.turn_id

    # Append new turn after restart
    t3 = store2.add_turn(conv.id, "user", "Turn 3 after restart")
    assert t3.sequence_num == 3
    assert t3.prev_turn_id == t2.turn_id
    assert store2.get_turn(t2.turn_id).next_turn_id == t3.turn_id
    store2.close()


def test_11_rollback_recovery_on_failure(tmp_path):
    """Scenario 11: Transaction rollback preserves data integrity on failure."""
    db_file = tmp_path / "episodic_test.db"
    store = ConversationStore(db_file)
    conv = store.create_conversation("Rollback Test")

    t1 = store.add_turn(conv.id, "user", "Existing turn", turn_id="turn_good_1")
    assert store.count_turns(conv.id) == 1

    # Attempt a batch insert where the second item causes a duplicate key error
    bad_batch = [
        {"role": "assistant", "content": "Batch turn 1", "turn_id": "turn_batch_1"},
        {"role": "user", "content": "Duplicate key turn", "turn_id": "turn_good_1"},
    ]

    with pytest.raises(ValueError):
        store.add_turns_batch(conv.id, bad_batch)

    # Entire batch must be rolled back: neither turn_batch_1 nor duplicate was committed
    assert store.count_turns(conv.id) == 1
    assert store.get_turn("turn_batch_1") is None
    # Previous turn pointer still intact
    assert store.get_turn(t1.turn_id).next_turn_id is None
    store.close()


def test_12_migration_from_legacy_messages(tmp_path):
    """Scenario 12: Existing database with only messages table is automatically migrated."""
    db_file = tmp_path / "legacy.db"

    # Create raw database with legacy schema
    conn = sqlite3.connect(str(db_file))
    with conn:
        conn.execute("""
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        conn.execute("INSERT INTO conversations VALUES ('conv_legacy', 'Legacy Session', '2026-11-01T10:00:00Z', '2026-11-01T10:05:00Z', '{}')")
        conn.execute("INSERT INTO messages VALUES ('msg_1', 'conv_legacy', 'user', 'Legacy message 1', '2026-11-01T10:00:00Z', 1, '{}')")
        conn.execute("INSERT INTO messages VALUES ('msg_2', 'conv_legacy', 'assistant', 'Legacy message 2', '2026-11-01T10:01:00Z', 2, '{}')")
    conn.close()

    # Now open with ConversationStore: migration must backfill episodic_turns
    store = ConversationStore(db_file)
    assert store.count_turns("conv_legacy") == 2

    turn1 = store.get_turn("msg_1")
    turn2 = store.get_turn("msg_2")

    assert turn1 is not None
    assert turn1.content == "Legacy message 1"
    assert turn1.speaker_role == "user"
    assert turn1.sequence_num == 1
    assert turn1.prev_turn_id is None
    assert turn1.next_turn_id == "msg_2"

    assert turn2 is not None
    assert turn2.content == "Legacy message 2"
    assert turn2.speaker_role == "assistant"
    assert turn2.sequence_num == 2
    assert turn2.prev_turn_id == "msg_1"
    assert turn2.next_turn_id is None
    store.close()


def test_search_and_filter_turns(tmp_path):
    """Verify search_turns and filter_turns capabilities."""
    db_file = tmp_path / "search_filter.db"
    store = ConversationStore(db_file)
    conv = store.create_conversation("Search Conv")

    store.add_turn(conv.id, "user", "I am planning a trip to Kyoto in April.", speaker="Alice", timestamp="2026-04-01T08:00:00Z")
    store.add_turn(conv.id, "assistant", "Kyoto is beautiful during cherry blossom season.", speaker="IVARA", timestamp="2026-04-01T08:01:00Z")
    store.add_turn(conv.id, "user", "I will book a ryokan near Gion.", speaker="Alice", timestamp="2026-04-01T08:02:00Z")

    # Search
    kyoto_results = store.search_turns("Kyoto", conversation_id=conv.id)
    assert len(kyoto_results) >= 1
    assert any("Kyoto" in t.content for t in kyoto_results)

    # Filter by speaker
    alice_turns = store.filter_turns(conversation_id=conv.id, speaker="Alice")
    assert len(alice_turns) == 2
    assert all(t.speaker == "Alice" for t in alice_turns)

    # Filter by time
    filtered_time = store.filter_turns(
        conversation_id=conv.id,
        start_time="2026-04-01T08:01:00Z",
        end_time="2026-04-01T08:02:00Z"
    )
    assert len(filtered_time) == 2
    store.close()


def test_concurrent_turn_insertion(tmp_path):
    """Verify thread-safe sequential turn insertion under concurrent threads."""
    import concurrent.futures
    db_file = tmp_path / "concurrent.db"
    store = ConversationStore(db_file)
    conv = store.create_conversation("Concurrent Conv")

    def worker(idx: int):
        return store.add_turn(
            conversation_id=conv.id,
            role="user",
            content=f"Concurrent message {idx}",
            turn_id=f"turn_concurrent_{idx:03d}"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(1, 21)]
        results = [f.result() for f in futures]

    assert len(results) == 20
    turns = store.get_turns(conv.id, chronological=True)
    assert len(turns) == 20

    # Ensure sequence numbers are strictly contiguous 1..20
    seqs = [t.sequence_num for t in turns]
    assert seqs == list(range(1, 21))

    # Ensure double-linked list has zero breaks or circular forks
    for i in range(len(turns)):
        curr_t = turns[i]
        expected_prev = turns[i - 1].turn_id if i > 0 else None
        expected_next = turns[i + 1].turn_id if i < len(turns) - 1 else None
        assert curr_t.prev_turn_id == expected_prev
        assert curr_t.next_turn_id == expected_next
    store.close()


def test_fts_disabled_like_fallback_and_partial_migration(tmp_path):
    """Verify LIKE fallback when FTS is disabled and verify per-row idempotent partial migration."""
    db_file = tmp_path / "fallback.db"
    store = ConversationStore(db_file)
    conv = store.create_conversation("Fallback Session")

    store.add_turn(conv.id, "user", "Searching for astrophysics terms in episodic storage.")
    
    # Force FTS disabled
    store._fts_enabled = False
    matches = store.search_turns("astrophysics", conversation_id=conv.id)
    assert len(matches) == 1
    assert "astrophysics" in matches[0].content

    # Partial migration test:
    # Directly inject an unmigrated message into messages table
    conn = store._get_connection()
    with conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at, sequence_num, metadata) "
            "VALUES ('msg_unmigrated_99', ?, 'user', 'Direct unmigrated message', '2026-11-15T12:00:00Z', 2, '{}')",
            (conv.id,)
        )
    store.close()

    # Reopen: _init_db should backfill only the missing message without errors
    store2 = ConversationStore(db_file)
    migrated_turn = store2.get_turn("msg_unmigrated_99")
    assert migrated_turn is not None
    assert migrated_turn.content == "Direct unmigrated message"
    assert migrated_turn.sequence_num == 2
    store2.close()


