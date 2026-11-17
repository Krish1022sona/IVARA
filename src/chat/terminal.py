"""Interactive Terminal Interface for IVARA Personal AI Memory Assistant."""

import sys
from typing import Optional

from .engine import ChatEngine, ChatTurnResult
from ..conversations.store import ConversationStore, DEFAULT_CONVERSATIONS_DB
from ..memory import list_memories, get_memory_stats, DATA_PATH as DEFAULT_MEMORIES_PATH


BANNER = """
================================================================================
           IVARA -- PERSONAL AI MEMORY & REASONING ASSISTANT
================================================================================
Local Models: Ollama (gpt-oss / gemma4) | Memory Storage: SQLite & FAISS
Type '/help' for in-chat commands, or '/exit' to return to terminal.
================================================================================
"""

HELP_TEXT = """
In-Chat Commands:
  /help       - Show this help message
  /exit, /quit- Exit chat session
  /new        - Start a fresh conversation session
  /chats      - List recent conversation sessions
  /memories   - List recently saved long-term memories
  /info       - Display current conversation details & stats
"""


def run_interactive_chat(
    engine: Optional[ChatEngine] = None,
    conversation_id: Optional[str] = None
) -> None:
    """Run continuous interactive terminal session."""
    chat_engine = engine or ChatEngine()
    conv = chat_engine.get_or_create_conversation(conversation_id)

    print(BANNER)
    print(f"Active Session: {conv.id} | Title: \"{conv.title}\"")
    
    # If resuming existing conversation, show last few messages
    existing_msgs = chat_engine.conv_store.get_messages(conv.id, limit=6, chronological=True)
    if existing_msgs:
        print("\n--- Recent Conversation History ---")
        for m in existing_msgs:
            prefix = "You" if m.role == "user" else "IVARA"
            print(f"[{prefix}]: {m.content}")
        print("-----------------------------------\n")

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession paused. Goodbye!")
            break

        if not user_input:
            continue

        # Check slash commands
        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            print("Session saved. Goodbye!")
            break

        elif cmd == "/help":
            print(HELP_TEXT)
            continue

        elif cmd == "/new":
            conv = chat_engine.conv_store.create_conversation("New Session")
            print(f"\nStarted new conversation: {conv.id}")
            continue

        elif cmd == "/chats":
            convs = chat_engine.conv_store.list_conversations(limit=10)
            print("\nRecent Conversations:")
            for c in convs:
                active_marker = " * " if c.id == conv.id else "   "
                print(f"{active_marker}{c.id}: \"{c.title}\" (Updated: {c.updated_at[:19]})")
            continue

        elif cmd == "/memories":
            memos = list_memories(data_path=chat_engine.memories_path)
            print(f"\nLong-Term Memories ({len(memos)} total):")
            for m in memos[-10:]:
                print(f"  [{m.get('id')}]: {m.get('text')}")
            continue

        elif cmd == "/info":
            msg_count = chat_engine.conv_store.count_messages(conv.id)
            stats = get_memory_stats(data_path=chat_engine.memories_path, graph_store=chat_engine.graph_store)
            print(f"\nConversation: {conv.id}")
            print(f"  Title: \"{conv.title}\"")
            print(f"  Messages in this chat: {msg_count}")
            print(f"  Total long-term memories in IVARA: {stats.get('total_memories', 0)}")
            continue

        # Execute conversation turn
        try:
            result = chat_engine.process_turn(user_input, conversation_id=conv.id)
            
            # Print response
            print(f"\nIVARA > {result.answer}")

            # Print citations if grounded in memory
            if result.citations:
                print(f"  [Citations: {', '.join(result.citations)}]")

            # Print memory storage notice if fact was formed
            if result.memory_extracted:
                print(f"  [Memory Formed: {result.extracted_memory_id}]")
            elif result.extraction_error:
                print(f"  (Note: memory extraction deferred: {result.extraction_error})")

        except Exception as e:
            print(f"\nIVARA > Error processing turn: {e}")

