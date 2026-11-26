"""Unit tests for the dedicated SQLite ConversationStore."""

import hashlib
from pathlib import Path
import pytest
from src.conversations.store import ConversationStore, Conversation, Message


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_create_conversation_and_retrieve(tmp_path):
    """Verify creating a conversation persists metadata and retrieves accurately."""
    db_file = tmp_path / "test_conv.db"
    store = ConversationStore(db_file)

    conv = store.create_conversation(title="Test Project Planning", metadata={"category": "work"})
    assert conv.id.startswith("conv_")
    assert conv.title == "Test Project Planning"
    assert conv.metadata == {"category": "work"}
    assert conv.created_at == conv.updated_at

    fetched = store.get_conversation(conv.id)
    assert fetched is not None
    assert fetched.id == conv.id
    assert fetched.title == "Test Project Planning"
    assert fetched.metadata["category"] == "work"
    store.close()


def test_append_multiple_messages_chronological_order(tmp_path):
    """Verify appending messages preserves role and strict chronological sequence."""
    db_file = tmp_path / "test_conv.db"
    store = ConversationStore(db_file)

    conv = store.create_conversation("Multi-turn Session")
    
    m1 = store.add_message(conv.id, "user", "Hello IVARA, can you remember this project?")
    m2 = store.add_message(conv.id, "assistant", "Yes, I can remember your project details.", metadata={"model": "gpt-oss"})
    m3 = store.add_message(conv.id, "user", "We are using PyTorch and SQLite.")
    m4 = store.add_message(conv.id, "assistant", "Noted, PyTorch and SQLite recorded.", metadata={"citations": []})

    assert m1.sequence_num == 1
    assert m2.sequence_num == 2
    assert m3.sequence_num == 3
    assert m4.sequence_num == 4

    messages = store.get_messages(conv.id, chronological=True)
    assert len(messages) == 4
    assert [m.sequence_num for m in messages] == [1, 2, 3, 4]
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0].content == "Hello IVARA, can you remember this project?"
    assert messages[3].content == "Noted, PyTorch and SQLite recorded."
    assert messages[1].metadata == {"model": "gpt-oss"}

    # Pagination test
    first_two = store.get_messages(conv.id, limit=2, offset=0)
    assert len(first_two) == 2
    assert first_two[0].sequence_num == 1
    assert first_two[1].sequence_num == 2

    next_two = store.get_messages(conv.id, limit=2, offset=2)
    assert len(next_two) == 2
    assert next_two[0].sequence_num == 3
    assert next_two[1].sequence_num == 4
    store.close()


def test_persistence_across_reopen(tmp_path):
    """Verify data survives database closing and reopening."""
    db_file = tmp_path / "persistent_conv.db"
    
    store1 = ConversationStore(db_file)
    c = store1.create_conversation("Persistent Chat")
    store1.add_message(c.id, "user", "Message before restart")
    store1.close()

    # Reopen same file
    store2 = ConversationStore(db_file)
    reloaded_c = store2.get_conversation(c.id)
    assert reloaded_c is not None
    assert reloaded_c.title == "Persistent Chat"

    reloaded_msgs = store2.get_messages(c.id)
    assert len(reloaded_msgs) == 1
    assert reloaded_msgs[0].content == "Message before restart"
    store2.close()


def test_update_conversation_metadata(tmp_path):
    """Verify updating title and metadata updates timestamps appropriately."""
    db_file = tmp_path / "test_conv.db"
    store = ConversationStore(db_file)

    c = store.create_conversation("Initial Title", metadata={"v": 1})
    updated = store.update_conversation(c.id, title="Renamed Title", metadata={"v": 2, "pinned": True})

    assert updated is not None
    assert updated.title == "Renamed Title"
    assert updated.metadata == {"v": 2, "pinned": True}
    assert updated.updated_at >= c.updated_at
    store.close()


def test_delete_conversation_cascades_messages(tmp_path):
    """Verify deleting a conversation removes it and cascade-deletes all its messages."""
    db_file = tmp_path / "test_conv.db"
    store = ConversationStore(db_file)

    c1 = store.create_conversation("Chat 1")
    c2 = store.create_conversation("Chat 2")

    store.add_message(c1.id, "user", "Message in Chat 1")
    store.add_message(c2.id, "user", "Message in Chat 2")

    assert store.count_conversations() == 2
    assert store.count_messages() == 2

    deleted = store.delete_conversation(c1.id)
    assert deleted is True

    # c1 is gone
    assert store.get_conversation(c1.id) is None
    assert store.get_messages(c1.id) == []
    assert store.count_messages(c1.id) == 0

    # c2 is untouched
    assert store.get_conversation(c2.id) is not None
    assert len(store.get_messages(c2.id)) == 1
    assert store.count_conversations() == 1
    store.close()


def test_invalid_conversation_id_handling(tmp_path):
    """Verify appropriate errors and None returns for nonexistent IDs."""
    db_file = tmp_path / "test_conv.db"
    store = ConversationStore(db_file)

    assert store.get_conversation("nonexistent_id") is None
    assert store.delete_conversation("nonexistent_id") is False
    assert store.update_conversation("nonexistent_id", title="New") is None
    assert store.get_messages("nonexistent_id") == []

    with pytest.raises(KeyError):
        store.add_message("nonexistent_id", "user", "Hello")

    with pytest.raises(ValueError):
        c = store.create_conversation("Valid")
        store.add_message(c.id, "invalid_role", "Hello")  # type: ignore

    store.close()


def test_transaction_failure_rollback(tmp_path):
    """Verify failed multi-turn batch inserts rollback atomically without partial writes."""
    db_file = tmp_path / "test_conv.db"
    store = ConversationStore(db_file)
    c = store.create_conversation("Atomic Batch")

    store.add_message(c.id, "user", "Existing valid message")
    initial_count = store.count_messages(c.id)
    assert initial_count == 1

    # Attempt a batch where the second message has an invalid role
    invalid_batch = [
        {"role": "user", "content": "Valid turn 1"},
        {"role": "invalid_role", "content": "This will fail validation"},
    ]

    with pytest.raises(ValueError):
        store.add_messages_batch(c.id, invalid_batch)

    # Confirm strict rollback: no partial writes occurred
    after_count = store.count_messages(c.id)
    assert after_count == initial_count
    messages = store.get_messages(c.id)
    assert len(messages) == 1
    assert messages[0].content == "Existing valid message"
    store.close()


def test_long_term_memories_and_graph_remain_untouched(tmp_path):
    """Verify conversation operations leave production memories and entities completely untouched."""
    mem_sha_before = sha256_file("data/memories.json")
    ent_sha_before = sha256_file("data/entities.json")

    # Perform extensive conversation operations
    db_file = tmp_path / "isolation_check.db"
    store = ConversationStore(db_file)
    c = store.create_conversation("Isolation Test")
    for i in range(10):
        store.add_message(c.id, "user" if i % 2 == 0 else "assistant", f"Turn {i}")
    store.clear_conversations()
    store.close()

    mem_sha_after = sha256_file("data/memories.json")
    ent_sha_after = sha256_file("data/entities.json")

    assert mem_sha_before == mem_sha_after
    assert ent_sha_before == ent_sha_after


def test_recent_history_exceeding_six_messages_and_token_budget(tmp_path):
    """Verify get_recent_messages returns the newest tail in chronological order without breaking legacy get_messages."""
    db_file = tmp_path / "recent_hist.db"
    with ConversationStore(db_file) as store:
        c1 = store.create_conversation("Long Chat")
        c2 = store.create_conversation("Other Session")
        store.add_message(c2.id, "user", "Isolated other session message")

        msgs = []
        for i in range(1, 17):
            role = "user" if i % 2 == 1 else "assistant"
            msgs.append(store.add_message(c1.id, role, f"Message {i}"))

        # Legacy forward pagination remains intact
        oldest_six = store.get_messages(c1.id, limit=6, chronological=True)
        assert [m.sequence_num for m in oldest_six] == [1, 2, 3, 4, 5, 6]
        offset_slice = store.get_messages(c1.id, limit=2, offset=4, chronological=True)
        assert [m.sequence_num for m in offset_slice] == [5, 6]

        # Recent-history method returns latest 6 in chronological order
        recent_six = store.get_recent_messages(c1.id, limit=6, max_tokens=None)
        assert [m.sequence_num for m in recent_six] == [11, 12, 13, 14, 15, 16]

        # Exclude current message id ensures current turn is not duplicated
        without_last = store.get_recent_messages(
            c1.id, limit=6, max_tokens=None, exclude_message_id=msgs[-1].id
        )
        assert [m.sequence_num for m in without_last] == [10, 11, 12, 13, 14, 15]

        # Token budget caps history cleanly while keeping chronological order
        tight_budget = store.get_recent_messages(c1.id, limit=16, max_tokens=20)
        assert 1 <= len(tight_budget) < 16
        assert tight_budget[-1].sequence_num == 16
        assert [m.sequence_num for m in tight_budget] == sorted(m.sequence_num for m in tight_budget)

        # Session isolation
        assert len(store.get_recent_messages(c2.id, limit=6)) == 1

    # Session resumption across a reopened store
    with ConversationStore(db_file) as reopened:
        resumed = reopened.get_recent_messages(c1.id, limit=4, max_tokens=None)
        assert [m.sequence_num for m in resumed] == [13, 14, 15, 16]


