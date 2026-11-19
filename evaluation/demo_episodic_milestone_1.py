"""Milestone 1 Demonstration: Persistent Episodic Memory Layer in IVARA.

Demonstrates:
1. 6-turn SIH collaborative dialogue ingestion.
2. Complete episodic retention across user and assistant turns.
3. Strict sequence numbers and double-linked sequence chaining.
4. Retention of Turn 3 ("He handles the backend...") despite semantic extraction rejection.
5. Context-window retrieval (+/- 1 turn) around Turn 3.
6. Lexical search across episodic turns.
"""

import json
from pathlib import Path
import tempfile
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conversations.store import ConversationStore
from src.chat.engine import ChatEngine


def run_demonstration():
    print("=" * 80)
    print("IVARA MILESTONE 1 DEMONSTRATION: PERSISTENT EPISODIC MEMORY LAYER")
    print("=" * 80)

    # Use isolated temporary environment to guarantee zero impact on production data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_path = temp_path / "demo_conversations.db"
        memories_path = temp_path / "demo_memories.json"
        entities_path = temp_path / "demo_entities.json"

        memories_path.write_text("[]", encoding="utf-8")
        entities_path.write_text("{}", encoding="utf-8")

        store = ConversationStore(db_path)
        
        # Inject deterministic conversational mock assistant
        def mock_llm(prompt: str, system: str) -> str:
            if "started working with Rahul" in prompt:
                return "That sounds like an exciting hackathon team. What will you be building?"
            elif "handles the backend" in prompt:
                return "Clear division of labor is crucial. How are you integrating the frontend and backend?"
            elif "FastAPI" in prompt:
                return "FastAPI is excellent for fast async Python backends. Great choice!"
            return "Acknowledged."

        engine = ChatEngine(
            conversation_store=store,
            memories_path=memories_path,
            entities_path=entities_path,
            llm_caller=mock_llm
        )

        conv = store.create_conversation("SIH Hackathon Project")
        print(f"\n[Created Session]: ID={conv.id}, Title='{conv.title}'")

        # 6-Turn SIH Conversation
        dialogue = [
            "I started working with Rahul on our SIH project.",
            "He handles the backend while I work on the frontend.",
            "We eventually decided to use FastAPI."
        ]

        print("\n" + "-" * 80)
        print("SIMULATING 6-TURN INTERACTIVE CONVERSATION")
        print("-" * 80)

        turn_results = []
        for user_utterance in dialogue:
            res = engine.process_turn(user_utterance, conversation_id=conv.id, auto_extract=True)
            turn_results.append(res)
            print(f"\n[Turn {res.user_turn.sequence_num}] User: {res.user_turn.content}")
            print(f"  -> Semantic Extracted: {res.memory_extracted} (ID: {res.extracted_memory_id})")
            print(f"[Turn {res.assistant_turn.sequence_num}] Assistant: {res.assistant_turn.content}")

        # Verification 1: Total Turns in Episodic Store
        all_turns = store.get_turns(conv.id, chronological=True)
        print("\n" + "=" * 80)
        print("VERIFICATION 1: COMPLETE EPISODIC RETENTION")
        print("=" * 80)
        print(f"Total Episodic Turns Persisted: {len(all_turns)} (Expected: 6)")
        assert len(all_turns) == 6, f"Expected 6 turns, got {len(all_turns)}"

        for t in all_turns:
            print(f"  Seq {t.sequence_num} [{t.speaker}]: {t.content[:50]}... (prev={t.prev_turn_id[-6:] if t.prev_turn_id else 'None'}, next={t.next_turn_id[-6:] if t.next_turn_id else 'None'})")

        # Verification 2: Double-Linked Sequence Chaining
        print("\n" + "=" * 80)
        print("VERIFICATION 2: DOUBLE-LINKED SEQUENCE CHAINING")
        print("=" * 80)
        for i in range(len(all_turns)):
            curr_t = all_turns[i]
            expected_prev = all_turns[i - 1].turn_id if i > 0 else None
            expected_next = all_turns[i + 1].turn_id if i < len(all_turns) - 1 else None
            assert curr_t.prev_turn_id == expected_prev, f"Turn {curr_t.turn_id} prev mismatch"
            assert curr_t.next_turn_id == expected_next, f"Turn {curr_t.turn_id} next mismatch"
        print("  [SUCCESS] All double-linked list pointers (prev_turn_id, next_turn_id) verified.")

        # Verification 3: Turn 3 Independent Persistence
        print("\n" + "=" * 80)
        print("VERIFICATION 3: INDEPENDENT RETENTION OF SEMANTICALLY REJECTED TURN")
        print("=" * 80)
        turn_3 = all_turns[2]  # 0-indexed: index 2 is Sequence 3
        res_turn_3 = turn_results[1]
        print(f"Target Statement: '{turn_3.content}'")
        print(f"Semantic Extracted: {res_turn_3.memory_extracted}")
        print(f"Episodic Layer Status: Preserved in SQLite with turn_id={turn_3.turn_id}")
        assert res_turn_3.memory_extracted is False, "Turn 3 was expected to be rejected by semantic policy"
        assert store.get_turn(turn_3.turn_id) is not None, "Turn 3 missing from episodic store!"
        print("  [SUCCESS] Turn 3 was rejected by semantic policy but is 100% preserved in episodic memory.")

        # Verification 4: Context Window Expansion (+/- 1 turn)
        print("\n" + "=" * 80)
        print("VERIFICATION 4: CONTEXT WINDOW EXPANSION AROUND TURN 3 (+/- 1 turn)")
        print("=" * 80)
        context_window = store.get_context_window(turn_3.turn_id, window_size=1)
        print(f"Retrieved Context Window Size: {len(context_window)} turns")
        assert len(context_window) == 3, f"Expected 3 turns in window, got {len(context_window)}"
        for w_turn in context_window:
            prefix = "--> TARGET:" if w_turn.turn_id == turn_3.turn_id else "    CONTEXT:"
            print(f"  {prefix} Seq {w_turn.sequence_num} [{w_turn.speaker}]: {w_turn.content}")

        assert context_window[0].sequence_num == 2
        assert context_window[1].sequence_num == 3
        assert context_window[2].sequence_num == 4
        print("  [SUCCESS] Bounded context window (+/- 1 turn) correctly reconstructed.")

        # Verification 5: Lexical Search Across Turns
        print("\n" + "=" * 80)
        print("VERIFICATION 5: EPISODIC SEARCH")
        print("=" * 80)
        search_terms = ["Rahul", "backend", "FastAPI"]
        for query in search_terms:
            matches = store.search_turns(query, conversation_id=conv.id)
            print(f"Query: '{query}' -> Found {len(matches)} match(es)")
            for m in matches:
                print(f"    Seq {m.sequence_num} [{m.speaker}]: {m.content}")
            assert len(matches) >= 1, f"Expected matches for '{query}'"

        store.close()
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY: ALL CHECKS PASSED")
        print("=" * 80)


if __name__ == "__main__":
    run_demonstration()

