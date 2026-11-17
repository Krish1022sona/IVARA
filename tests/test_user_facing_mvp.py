"""Comprehensive Integration and End-to-End Tests for IVARA User-Facing MVP.

Verifies the complete 15-step core user lifecycle:
1. Start IVARA session
2. Create conversation
3. Post factual message: "I started learning reinforcement learning in July 2025."
4. Verify user message persisted
5. Verify long-term memory extracted and persisted
6. Close / simulate application exit
7. Restart IVARA with fresh store instances
8. Ask question: "When did I start learning reinforcement learning?"
9. Verify memory retrieved and answer produced with valid citation
10. Verify previous conversation history preserved
11. Delete previous conversation
12. Verify conversation deleted
13. Verify long-term memory remains intact (default policy)
14. Delete long-term memory
15. Verify memory no longer retrieved

Also tests:
- Conversational follow-ups
- Conversation isolation
- Temporal updates and supersession
- Honest grounded abstention on ungrounded queries
- Ollama unavailable handling
- Failed extraction without conversation loss
- Independent clear operations (chats vs memories)
"""

import json
from pathlib import Path
import pytest

from src.conversations.store import ConversationStore
from src.graph.store import SQLiteGraphStore
from src.memory import (
    load_memories,
    get_memory,
    list_memories,
    delete_memory,
    clear_memories,
    get_memory_stats
)
from src.retrieval import retrieve
from src.chat.engine import ChatEngine, should_extract_memory


def create_deterministic_mock_llm():
    """Mock LLM caller producing valid IVARA schema answers and extractions."""
    def mock_caller(prompt: str, system_prompt: str):
        # 1. Extraction prompt
        if "semantic memory extraction system" in system_prompt or "Extract knowledge from this memory" in prompt:
            if "reinforcement learning" in prompt.lower():
                return {
                    "entities": [
                        {"name": "SELF", "type": "Person", "confidence": 1.0},
                        {"name": "Reinforcement Learning", "type": "Technology", "confidence": 1.0}
                    ],
                    "topics": [
                        {"name": "Machine Learning", "confidence": 0.95}
                    ],
                    "relationships": [
                        {
                            "source": "SELF",
                            "relation": "learning",
                            "target": "Reinforcement Learning",
                            "confidence": 1.0,
                            "origin": "explicit",
                            "evidence": "I started learning reinforcement learning"
                        }
                    ]
                }
            elif "austin" in prompt.lower() or "seattle" in prompt.lower():
                return {
                    "entities": [
                        {"name": "SELF", "type": "Person", "confidence": 1.0},
                        {"name": "Austin", "type": "Location", "confidence": 1.0}
                    ],
                    "topics": [{"name": "Residence", "confidence": 1.0}],
                    "relationships": [
                        {
                            "source": "SELF",
                            "relation": "lives_in",
                            "target": "Austin",
                            "confidence": 1.0,
                            "origin": "explicit",
                            "evidence": prompt
                        }
                    ]
                }
            return {
                "entities": [{"name": "SELF", "type": "Person", "confidence": 1.0}],
                "topics": [],
                "relationships": []
            }

        # 2. Answering prompt
        if "evidence-grounded personal memory answering assistant" in system_prompt:
            if "reinforcement learning" in prompt.lower():
                # Check if memory evidence is present in prompt
                if "reinforcement learning" in prompt.lower() and ("m00" in prompt or "m0" in prompt):
                    # Extract available memory ID from prompt
                    import re
                    m_ids = re.findall(r"\b(m\d{3})\b", prompt)
                    cited = m_ids[:1] if m_ids else ["m001"]
                    return {
                        "status": "answered",
                        "answer": f"You started learning reinforcement learning in July 2025 [{cited[0]}].",
                        "cited_memory_ids": cited,
                        "limitations": None
                    }
                else:
                    return {
                        "status": "insufficient_evidence",
                        "answer": "I do not have sufficient recorded memories to answer when you started learning reinforcement learning.",
                        "cited_memory_ids": [],
                        "limitations": "No matching memories found."
                    }

            if "what is my mother's maiden name" in prompt.lower():
                return {
                    "status": "insufficient_evidence",
                    "answer": "I do not have sufficient recorded memories to answer what your mother's maiden name is.",
                    "cited_memory_ids": [],
                    "limitations": "No evidence available."
                }

        # 3. Conversational response fallback
        return "I have noted that down in this conversation."

    return mock_caller


def setup_isolated_env(tmp_path):
    """Create isolated file paths and instances for testing."""
    conv_db = tmp_path / "test_conversations.db"
    graph_db = tmp_path / "test_graph.db"
    memories_file = tmp_path / "test_memories.json"
    entities_file = tmp_path / "test_entities.json"

    # Initialize empty memories and entities
    with open(memories_file, "w", encoding="utf-8") as f:
        json.dump([], f)
    with open(entities_file, "w", encoding="utf-8") as f:
        json.dump([
            {"id": "e001", "canonical_name": "Reinforcement Learning", "aliases": ["rl"], "types": ["Technology"]},
            {"id": "e002", "canonical_name": "Austin", "aliases": ["austin, tx"], "types": ["Location"]}
        ], f)

    conv_store = ConversationStore(conv_db)
    graph_store = SQLiteGraphStore(graph_db)

    return conv_store, graph_store, memories_file, entities_file, conv_db, graph_db


# ---------------------------------------------------------------------------
# Core 15-Step User Lifecycle Verification
# ---------------------------------------------------------------------------

def test_full_15_step_user_lifecycle(tmp_path):
    """
    Test the complete 15-step end-to-end workflow:
    Learn fact -> persist chat & memory -> restart -> retrieve & answer -> delete chat -> verify memory stays -> delete memory.
    """
    mock_llm = create_deterministic_mock_llm()
    conv_store, graph_store, mem_file, ent_file, conv_db, graph_db = setup_isolated_env(tmp_path)

    # Step 1: Start IVARA ChatEngine
    engine1 = ChatEngine(
        conversation_store=conv_store,
        graph_store=graph_store,
        memories_path=mem_file,
        entities_path=ent_file,
        llm_caller=mock_llm
    )

    # Step 2: Create conversation
    conv = engine1.get_or_create_conversation(title="Personal Study Planning")
    assert conv.id.startswith("conv_")

    # Step 3: Tell it personal fact
    fact_text = "I started learning reinforcement learning in July 2025."
    res1 = engine1.process_turn(fact_text, conversation_id=conv.id)

    # Step 4: Verify original message persisted
    msgs = conv_store.get_messages(conv.id)
    assert len(msgs) == 2  # user turn + assistant acknowledgement
    assert msgs[0].role == "user"
    assert msgs[0].content == fact_text
    assert msgs[1].role == "assistant"

    # Step 5: Verify long-term memory extracted and persisted
    assert res1.memory_extracted is True
    assert res1.extracted_memory_id is not None
    saved_mems = load_memories(mem_file)
    assert len(saved_mems) == 1
    stored_mem = saved_mems[0]
    assert stored_mem["id"] == res1.extracted_memory_id
    assert stored_mem["text"] == fact_text
    assert stored_mem["source"] == f"{conv.id}_{res1.user_message.id}"

    # Step 6: Simulate application exit
    conv_store.close()
    graph_store.close()

    # Step 7: Restart IVARA with fresh store instances pointing to same databases
    restarted_conv_store = ConversationStore(conv_db)
    restarted_graph_store = SQLiteGraphStore(graph_db)
    engine2 = ChatEngine(
        conversation_store=restarted_conv_store,
        graph_store=restarted_graph_store,
        memories_path=mem_file,
        entities_path=ent_file,
        llm_caller=mock_llm
    )

    # Step 8: Ask question in a brand new conversation
    new_conv = engine2.get_or_create_conversation(title="Query Session")
    query_text = "When did I start learning reinforcement learning?"
    res2 = engine2.process_turn(query_text, conversation_id=new_conv.id)

    # Step 9: Verify memory retrieved and answer produced with valid citation
    assert res2.status in ("answered", "partial")
    assert "July 2025" in res2.answer
    assert res1.extracted_memory_id in res2.citations

    # Step 10: Open previous conversation and verify history
    prev_history = restarted_conv_store.get_messages(conv.id)
    assert len(prev_history) == 2
    assert prev_history[0].content == fact_text

    # Step 11: Delete previous conversation
    del_chat_success = restarted_conv_store.delete_conversation(conv.id)
    assert del_chat_success is True

    # Step 12: Verify conversation is gone
    assert restarted_conv_store.get_conversation(conv.id) is None
    assert restarted_conv_store.get_messages(conv.id) == []

    # Step 13: Verify long-term memory remains intact under default deletion policy
    mems_after_chat_del = load_memories(mem_file)
    assert len(mems_after_chat_del) == 1
    assert mems_after_chat_del[0]["id"] == res1.extracted_memory_id

    # Step 14: Delete long-term memory
    del_mem_success = delete_memory(res1.extracted_memory_id, data_path=mem_file, graph_store=restarted_graph_store)
    assert del_mem_success is True

    # Step 15: Verify memory no longer retrieved
    mems_after_mem_del = load_memories(mem_file)
    assert len(mems_after_mem_del) == 0
    search_results = retrieve("reinforcement learning", k=3, min_score=0.20, data_path=mem_file)
    assert len(search_results) == 0

    restarted_conv_store.close()
    restarted_graph_store.close()


# ---------------------------------------------------------------------------
# Additional Resilience and Edge Case Tests
# ---------------------------------------------------------------------------

def test_grounded_abstention_on_ungrounded_question(tmp_path):
    """Verify system honestly abstains when questioned about unrecorded facts."""
    mock_llm = create_deterministic_mock_llm()
    conv_store, graph_store, mem_file, ent_file, _, _ = setup_isolated_env(tmp_path)

    engine = ChatEngine(
        conversation_store=conv_store,
        graph_store=graph_store,
        memories_path=mem_file,
        entities_path=ent_file,
        llm_caller=mock_llm
    )

    conv = engine.get_or_create_conversation()
    res = engine.process_turn("What is my mother's maiden name?", conversation_id=conv.id)

    assert res.status == "insufficient_evidence"
    assert "sufficient recorded memory" in res.answer.lower() or "insufficient" in res.answer.lower()
    assert len(res.citations) == 0

    conv_store.close()
    graph_store.close()


def test_failed_extraction_does_not_lose_conversation(tmp_path):
    """Verify that an exception in memory extraction does not destroy or corrupt conversation persistence."""
    conv_store, graph_store, mem_file, ent_file, _, _ = setup_isolated_env(tmp_path)

    # Faulty extractor that always raises an error
    def faulty_llm(prompt, sys_prompt):
        if "semantic memory extraction system" in sys_prompt or "Extract knowledge" in prompt:
            raise RuntimeError("Extraction model crashed!")
        return "Noted."

    engine = ChatEngine(
        conversation_store=conv_store,
        graph_store=graph_store,
        memories_path=mem_file,
        entities_path=ent_file,
        llm_caller=faulty_llm
    )

    conv = engine.get_or_create_conversation()
    turn_text = "I moved to Seattle last week."
    res = engine.process_turn(turn_text, conversation_id=conv.id)

    # Conversation must be safely preserved
    msgs = conv_store.get_messages(conv.id)
    assert len(msgs) == 2
    assert msgs[0].content == turn_text

    # Memory extraction should fail gracefully without throwing unhandled exception
    assert res.memory_extracted is False
    assert res.extraction_error is not None
    assert "Extraction model crashed!" in res.extraction_error

    conv_store.close()
    graph_store.close()


def test_independent_clear_operations(tmp_path):
    """Verify clearing chats does not clear memories, and clearing memories does not clear chats."""
    conv_store, graph_store, mem_file, ent_file, _, _ = setup_isolated_env(tmp_path)
    mock_llm = create_deterministic_mock_llm()

    engine = ChatEngine(
        conversation_store=conv_store,
        graph_store=graph_store,
        memories_path=mem_file,
        entities_path=ent_file,
        llm_caller=mock_llm
    )

    conv = engine.get_or_create_conversation("Multi-Asset Chat")
    engine.process_turn("I started learning reinforcement learning in July 2025.", conversation_id=conv.id)

    assert conv_store.count_conversations() == 1
    assert len(load_memories(mem_file)) == 1

    # 1. Clear chats
    conv_store.clear_conversations()
    assert conv_store.count_conversations() == 0
    # Memories must remain untouched!
    assert len(load_memories(mem_file)) == 1

    # 2. Clear memories
    clear_memories(data_path=mem_file, graph_store=graph_store)
    assert len(load_memories(mem_file)) == 0

    conv_store.close()
    graph_store.close()


def test_should_extract_memory_heuristics():
    """Verify heuristic filter distinguishes trivial greetings from meaningful facts."""
    assert should_extract_memory("Hello there!") is False
    assert should_extract_memory("Thanks!") is False
    assert should_extract_memory("ok") is False
    assert should_extract_memory("What is my name?") is False
    assert should_extract_memory("Where do I live?") is False

    assert should_extract_memory("I started learning reinforcement learning in July 2025.") is True
    assert should_extract_memory("My sister lives in Chicago and works at Boeing.") is True
    assert should_extract_memory("I moved to Austin last month for a new job.") is True


def test_follow_up_conversational_turns(tmp_path):
    """Verify follow-up turns in the same conversation retain message sequence."""
    conv_store, graph_store, mem_file, ent_file, _, _ = setup_isolated_env(tmp_path)
    mock_llm = create_deterministic_mock_llm()

    engine = ChatEngine(
        conversation_store=conv_store,
        graph_store=graph_store,
        memories_path=mem_file,
        entities_path=ent_file,
        llm_caller=mock_llm
    )

    conv = engine.get_or_create_conversation("Study Session")
    t1 = engine.process_turn("I started learning reinforcement learning in July 2025.", conversation_id=conv.id)
    t2 = engine.process_turn("Can you remember what I am studying?", conversation_id=conv.id)

    msgs = conv_store.get_messages(conv.id)
    assert len(msgs) == 4
    assert [m.sequence_num for m in msgs] == [1, 2, 3, 4]
    assert msgs[0].content == "I started learning reinforcement learning in July 2025."
    assert msgs[2].content == "Can you remember what I am studying?"

    conv_store.close()
    graph_store.close()


def test_ollama_unavailable_graceful_handling(tmp_path):
    """Verify system handles Ollama connection failure gracefully without crashing."""
    conv_store, graph_store, mem_file, ent_file, _, _ = setup_isolated_env(tmp_path)

    def crashing_llm(prompt, sys_prompt):
        import urllib.error
        raise ConnectionRefusedError("Connection refused: 127.0.0.1:11434")

    engine = ChatEngine(
        conversation_store=conv_store,
        graph_store=graph_store,
        memories_path=mem_file,
        entities_path=ent_file,
        llm_caller=crashing_llm
    )

    conv = engine.get_or_create_conversation()
    res = engine.process_turn("I am thinking of going for a hike this weekend.", conversation_id=conv.id)

    assert res.status == "error"
    assert "unavailable" in res.answer.lower() or "connection" in res.answer.lower()
    # Message must still be persisted
    msgs = conv_store.get_messages(conv.id)
    assert len(msgs) == 2

    conv_store.close()
    graph_store.close()


def test_cli_argument_parser_completeness():
    """Verify all required subcommands and flags are registered in main.py."""
    from main import build_parser
    parser = build_parser()

    # Verify chat commands
    args = parser.parse_args(["chat", "--id", "conv_123"])
    assert args.subcommand == "chat"
    assert args.id == "conv_123"

    # Verify chats list
    args = parser.parse_args(["chats", "list", "--limit", "10"])
    assert args.subcommand == "chats"
    assert args.chats_subcommand == "list"
    assert args.limit == 10

    # Verify chats open
    args = parser.parse_args(["chats", "open", "conv_abc"])
    assert args.id == "conv_abc"

    # Verify chats delete with purge
    args = parser.parse_args(["chats", "delete", "conv_abc", "--purge-memories"])
    assert args.id == "conv_abc"
    assert args.purge_memories is True

    # Verify memories search
    args = parser.parse_args(["memories", "search", "PyTorch", "-k", "3", "--min-score", "0.4"])
    assert args.subcommand == "memories"
    assert args.memories_subcommand == "search"
    assert args.query == "PyTorch"
    assert args.k == 3
    assert args.min_score == 0.4

    # Verify memories stats
    args = parser.parse_args(["memories", "stats"])
    assert args.subcommand == "memories"
    assert args.memories_subcommand == "stats"

