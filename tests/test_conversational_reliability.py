"""Regression tests for Phase 1B Core Conversational Reliability.

Covers:
- Pre-retrieval intent routing (general conversation, general knowledge, personal recall, mixed reasoning, memory management)
- Multi-turn context beyond 6 messages without duplicating the active user turn
- Self-retrieval exclusion and assistant-claim isolation vs. legitimate assistant recall
- Grounded answer status handling (including conflicting_evidence) and local reference date
- Memory-policy safety on standalone questions and hypotheticals
- Clear error reporting when Ollama is unavailable
"""

import json
import sys
import types

from src.conversations.store import ConversationStore
from src.chat.engine import ChatEngine, classify_turn_intent
from src.extraction.policy import evaluate_memory_formation
from src.memory import add_memory
from src.unified_retrieval.episodic import EpisodicRetriever


def test_intent_routing_avoids_memory_refusal_on_general_and_chitchat(tmp_path):
    """General knowledge, greetings, and creative follow-ups route conversationally in a single LLM call."""
    calls = []

    def mock_llm(prompt, system_prompt):
        calls.append((prompt, system_prompt))
        return "Here is a helpful natural reply."

    with ConversationStore(tmp_path / "conv.db") as store:
        mem_path = tmp_path / "memories.json"
        mem_path.write_text("[]", encoding="utf-8")
        engine = ChatEngine(
            conversation_store=store,
            memories_path=mem_path,
            entities_path=tmp_path / "entities.json",
            llm_caller=mock_llm,
        )
        conv = engine.get_or_create_conversation("General Chat")

        for u in [
            "Hello!",
            "How are you?",
            "Explain why the sky is blue.",
            "Explain it in more detail.",
            "Give me an analogy.",
            "Tell me a programming joke.",
            "Tell me another one.",
        ]:
            calls.clear()
            res = engine.process_turn(u, conversation_id=conv.id, auto_extract=False)
            assert res.status == "conversational", f"Unexpected status for {u!r}: {res.status}"
            assert "do not have enough verified memories" not in res.answer.lower()
            assert len(calls) == 1, f"Expected 1 LLM call for {u!r}, got {len(calls)}"


def test_long_conversation_history_retains_recent_turns_and_no_duplicate_current(tmp_path):
    """After >6 messages, conversational prompt includes recent turns and current message exactly once."""
    captured_prompts = []

    def mock_llm(prompt, system_prompt):
        captured_prompts.append(prompt)
        return f"Reply #{len(captured_prompts)}"

    with ConversationStore(tmp_path / "conv.db") as store:
        mem_path = tmp_path / "memories.json"
        mem_path.write_text("[]", encoding="utf-8")
        engine = ChatEngine(
            conversation_store=store,
            memories_path=mem_path,
            entities_path=tmp_path / "entities.json",
            llm_caller=mock_llm,
            history_message_limit=6,
        )
        conv = engine.get_or_create_conversation("Joke Continuity")

        for i in range(1, 6):
            engine.process_turn(f"Turn {i} topic", conversation_id=conv.id, auto_extract=False)

        captured_prompts.clear()
        res = engine.process_turn("Tell me a different joke now.", conversation_id=conv.id, auto_extract=False)
        assert res.status == "conversational"
        last_prompt = captured_prompts[-1]
        assert "Turn 4 topic" in last_prompt
        assert "Turn 5 topic" in last_prompt
        assert "Turn 1 topic" not in last_prompt
        assert last_prompt.count("Tell me a different joke now.") == 1


def test_self_retrieval_and_assistant_claim_contamination_prevented(tmp_path):
    """Active user turn is excluded from retrieval; unsupported assistant claims are excluded by default."""
    with ConversationStore(tmp_path / "conv.db") as store:
        c = store.create_conversation("Self Contamination")
        t_user_fact = store.add_turn(c.id, "user", "I moved to Pune in January 2025.", speaker="user")
        t_asst_claim = store.add_turn(
            c.id, "assistant", "I told you a joke about Python decorators and recursion.", speaker="IVARA"
        )
        t_active_q = store.add_turn(c.id, "user", "Where did I move in January 2025?", speaker="user")

        retriever = EpisodicRetriever(store=store)

        # Active query turn must not retrieve itself, and assistant turns are excluded by default
        hits_default = retriever.retrieve(
            "Where did I move in January 2025?",
            k=5,
            min_score=0.2,
            exclude_turn_ids={t_active_q.turn_id},
            include_assistant_turns=False,
        )
        hit_ids_default = {h.turn_id for h in hits_default}
        assert t_active_q.turn_id not in hit_ids_default
        assert t_asst_claim.turn_id not in hit_ids_default
        assert t_user_fact.turn_id in hit_ids_default

        # Legitimate assistant recall works when explicitly requested
        hits_asst = retriever.retrieve(
            "Which jokes have you told me about Python decorators?",
            k=5,
            min_score=0.2,
            exclude_turn_ids={t_active_q.turn_id},
            include_assistant_turns=True,
        )
        assert any(h.turn_id == t_asst_claim.turn_id for h in hits_asst)


def test_personal_recall_mixed_reasoning_and_conflicting_evidence(tmp_path):
    """Verify personal recall, conflicting_evidence preservation, unrecorded abstention, and mixed reasoning."""
    mem_path = tmp_path / "memories.json"
    mem_path.write_text("[]", encoding="utf-8")
    add_memory(
        text="I drink coffee every morning.",
        timestamp="2024-01-10",
        source="chat",
        type="preference",
        importance=0.8,
        data_path=mem_path,
    )
    add_memory(
        text="I switched to drinking matcha tea every morning.",
        timestamp="2026-01-10",
        source="chat",
        type="preference",
        importance=0.8,
        data_path=mem_path,
    )

    def mock_llm(prompt, system_prompt):
        if "evidence-grounded personal memory answering assistant" in system_prompt:
            if "CURRENT REFERENCE TIME ANCHOR:" in prompt and "coffee" in prompt.lower():
                return json.dumps({
                    "status": "conflicting_evidence",
                    "answer": "Earlier records mention coffee, while a 2026 record mentions matcha tea.",
                    "cited_memory_ids": ["m001", "m002"],
                    "limitations": "Conflicting morning drink preferences across dates.",
                })
            return json.dumps({
                "status": "insufficient_evidence",
                "answer": "I do not have enough verified memories to answer that.",
                "cited_memory_ids": [],
                "limitations": "No relevant memory found.",
            })
        # Mixed personal reasoning prompt
        assert "PERSONAL MEMORY CONTEXT" in system_prompt
        return "Based on your morning drink preferences, you could try a ceremonial matcha whisk."

    with ConversationStore(tmp_path / "conv.db") as store:
        engine = ChatEngine(
            conversation_store=store,
            memories_path=mem_path,
            entities_path=tmp_path / "entities.json",
            llm_caller=mock_llm,
        )
        conv = engine.get_or_create_conversation("Memory & Mixed")

        # 1. Conflicting evidence preserved (not discarded!)
        res_conflict = engine.process_turn("What is my favorite morning drink?", conversation_id=conv.id, auto_extract=False)
        assert res_conflict.status == "conflicting_evidence"
        assert "matcha" in res_conflict.answer.lower()
        assert len(res_conflict.citations) >= 1

        # 2. Unrecorded personal fact abstains honestly
        res_unrec = engine.process_turn("What is my passport number?", conversation_id=conv.id, auto_extract=False)
        assert res_unrec.status in ("insufficient_evidence", "unanswerable")

        # 3. Mixed personal + general reasoning
        res_mixed = engine.process_turn("Based on my coffee and matcha drink preference, suggest a gift for me.", conversation_id=conv.id, auto_extract=False)
        assert res_mixed.status == "conversational"
        assert "matcha" in res_mixed.answer.lower()


def test_memory_policy_blocks_standalone_questions_and_hypotheticals():
    """Standalone questions and hypotheticals are not extracted, while embedded facts are preserved."""
    assert evaluate_memory_formation("Which GPU does my laptop have?").should_extract is False
    assert evaluate_memory_formation("Suppose I had a dog named Bruno.").should_extract is False
    assert evaluate_memory_formation("What if I moved to London next year?").should_extract is False
    assert evaluate_memory_formation("I might move to Bangalore next month.").should_extract is False

    # Embedded factual disclosure in a compound message is still extracted
    mixed = evaluate_memory_formation("I bought a new laptop yesterday, which GPU is best for it?")
    assert mixed.should_extract is True


def test_ollama_unavailable_reports_clear_error_and_persists_user_turn(tmp_path, monkeypatch):
    """When Ollama fails, ChatEngine returns status='error' with actionable text and preserves the user turn."""
    def failing_chat(*args, **kwargs):
        raise ConnectionError("Connection refused")

    monkeypatch.setattr("src.chat.engine.ollama_chat", failing_chat)

    with ConversationStore(tmp_path / "conv.db") as store:
        mem_path = tmp_path / "memories.json"
        mem_path.write_text("[]", encoding="utf-8")
        engine = ChatEngine(
            conversation_store=store,
            memories_path=mem_path,
            entities_path=tmp_path / "entities.json",
            llm_caller=None,
        )
        conv = engine.get_or_create_conversation("Error Handling")
        res = engine.process_turn("Explain quantum entanglement.", conversation_id=conv.id, auto_extract=False)
        assert res.status == "error"
        assert "ollama" in res.answer.lower()
        assert store.count_messages(conv.id) == 2


def test_intent_classifier_categories():
    """Verify classify_turn_intent distinguishes assistant recall, personal recall, mixed, GK, and management."""
    assert classify_turn_intent("List my saved memories") == ("memory_management", False)
    assert classify_turn_intent("Which jokes have you already told me?") == ("personal_recall", True)
    assert classify_turn_intent("When did I start learning Python?") == ("personal_recall", False)
    assert classify_turn_intent("Based on my budget, recommend a laptop") == ("mixed_personal_reasoning", False)
    assert classify_turn_intent("Explain how FAISS works") == ("general_knowledge", False)
    assert classify_turn_intent("How are you?") == ("general_conversation", False)


def test_dialogue_recall_intent_and_memory_safety():
    """Questions about active dialogue history route to dialogue_recall and are never extracted as memories."""
    # Intent classification
    assert classify_turn_intent("what were the last two topics we discussed?")[0] == "dialogue_recall"
    assert classify_turn_intent("what were teh last two topics we discussed?")[0] == "dialogue_recall"
    assert classify_turn_intent("what was my last chat to you just now")[0] == "dialogue_recall"
    assert classify_turn_intent("what did i just say")[0] == "dialogue_recall"
    assert classify_turn_intent("what was the last thing i asked")[0] == "dialogue_recall"

    # Memory formation policy rejection
    assert evaluate_memory_formation("what was my last chat to you just now").should_extract is False
    assert evaluate_memory_formation("what were teh last two topics we discussed?").should_extract is False
    assert evaluate_memory_formation("when did i had the urge to eat it").should_extract is False
    assert evaluate_memory_formation("what were the topics we discussed").should_extract is False


def test_dialogue_recall_execution_provides_history_and_instructions(tmp_path):
    """Dialogue recall queries receive preceding conversation history and instructions distinguishing the current query."""
    captured = []

    def mock_llm(prompt, system_prompt):
        captured.append((prompt, system_prompt))
        return "Your last chat was asking about the chocolate waffle."

    with ConversationStore(tmp_path / "conv.db") as store:
        mem_path = tmp_path / "memories.json"
        mem_path.write_text("[]", encoding="utf-8")
        engine = ChatEngine(
            conversation_store=store,
            memories_path=mem_path,
            entities_path=tmp_path / "entities.json",
            llm_caller=mock_llm,
        )
        conv = engine.get_or_create_conversation("Dialogue History Test")

        # Prior turns
        engine.process_turn("when did i eat a chocolate waffle", conversation_id=conv.id, auto_extract=False)
        engine.process_turn("when did i had the urge to eat it", conversation_id=conv.id, auto_extract=False)

        captured.clear()
        res = engine.process_turn("What subjects have we covered so far?", conversation_id=conv.id, auto_extract=True)

        assert res.intent == "dialogue_recall"
        assert res.status == "conversational"
        assert res.memory_extracted is False
        assert len(captured) == 1

        prompt, sys_prompt = captured[0]
        assert "ACTIVE DIALOGUE RECALL" in sys_prompt
        assert "IMMEDIATELY PRECEDING" in sys_prompt
        assert "when did i had the urge to eat it" in prompt
        # Current message itself must not be in the preceding history list
        history_part = prompt.split("Current user message:")[0]
        assert history_part.count("What subjects have we covered so far?") == 0


def test_empty_response_handling_never_saves_empty_strings(tmp_path):
    """Empty or whitespace responses from LLM trigger graceful fallback and are never stored as empty messages."""
    def mock_empty_llm(prompt, system_prompt):
        return ""

    with ConversationStore(tmp_path / "conv.db") as store:
        mem_path = tmp_path / "memories.json"
        mem_path.write_text("[]", encoding="utf-8")
        engine = ChatEngine(
            conversation_store=store,
            memories_path=mem_path,
            entities_path=tmp_path / "entities.json",
            llm_caller=mock_empty_llm,
        )
        conv = engine.get_or_create_conversation("Empty Fallback Test")
        res = engine.process_turn("explain that example in greater detail", conversation_id=conv.id, auto_extract=False)

        assert res.answer.strip() != ""
        assert "wasn't able to generate a complete response" in res.answer.lower()
        asst_msg = store.get_recent_messages(conv.id, limit=1)[0]
        assert asst_msg.content.strip() != ""


def test_terminal_slash_command_error_and_alias(monkeypatch, capsys):
    """Terminal rejects unknown slash commands with an error message and aliases /memory to /memories."""
    from src.chat.terminal import run_interactive_chat

    inputs = iter(["/foobar", "/memory", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    class DummyEngine:
        def __init__(self):
            self.memories_path = "data/memories.example.json"
            self.graph_store = None
            self.conv_store = DummyStore()

        def get_or_create_conversation(self, cid=None):
            return types.SimpleNamespace(id="conv_test", title="Test")

    class DummyStore:
        def get_recent_messages(self, cid, limit=6):
            return []

    run_interactive_chat(engine=DummyEngine())
    captured = capsys.readouterr().out
    assert "Unknown command '/foobar'. Type '/help' for available commands." in captured
    assert "Long-Term Memories" in captured
    assert "Session saved. Goodbye!" in captured


def test_sandbox_mode_uses_isolated_storage(monkeypatch):
    """Sandbox mode launches with a temporary directory and does not access production DBs."""
    from main import run_sandbox_chat
    import src.chat.terminal

    ran_isolated = []

    def mock_run_chat(engine=None, conversation_id=None):
        ran_isolated.append(str(engine.conv_store.db_path))
        assert "ivara_sandbox_" in str(engine.conv_store.db_path)
        assert "ivara_sandbox_" in str(engine.memories_path)

    monkeypatch.setattr(src.chat.terminal, "run_interactive_chat", mock_run_chat)
    monkeypatch.setattr("main.run_interactive_chat", mock_run_chat)

    run_sandbox_chat()
    assert len(ran_isolated) == 1


def test_generalised_dialogue_meta_and_followup_routing():
    """Varied unseen phrasing for dialogue recall and follow-ups routes without hardcoding exact strings."""
    for q in [
        "What subjects have we covered so far?",
        "Which themes did we talk about earlier?",
        "what were teh things we went over?",
        "Remind me what we were just covering",
        "Repeat the message I sent before this one.",
    ]:
        intent, include_asst = classify_turn_intent(q)
        assert intent == "dialogue_recall", f"Failed for {q!r}: got {intent}"
        assert include_asst is True

    # Personal ownership / fact questions still route to personal_recall
    assert classify_turn_intent("Do I own a sailboat?")[0] == "personal_recall"
    assert classify_turn_intent("Which novel did I finish last month?")[0] == "personal_recall"

    # Anaphoric follow-ups route to general_conversation when recent history is present
    dummy_history = [types.SimpleNamespace(role="user", content="Explain photosynthesis.")]
    assert classify_turn_intent("Make that shorter.", recent_messages=dummy_history)[0] == "general_conversation"
    assert classify_turn_intent("What about the second one?", recent_messages=dummy_history)[0] == "general_conversation"


def test_deterministic_exact_previous_message_recall(tmp_path):
    """Exact previous-message requests return the verbatim preceding user turn from ConversationStore without an LLM call."""
    llm_calls = []

    def mock_llm(prompt, system_prompt):
        llm_calls.append(prompt)
        return "LLM reply"

    with ConversationStore(tmp_path / "conv.db") as store:
        mem_path = tmp_path / "memories.json"
        mem_path.write_text("[]", encoding="utf-8")
        engine = ChatEngine(
            conversation_store=store,
            memories_path=mem_path,
            entities_path=tmp_path / "entities.json",
            llm_caller=mock_llm,
        )
        conv = engine.get_or_create_conversation("Exact Prev Msg")
        engine.process_turn("Explain orbital mechanics briefly.", conversation_id=conv.id, auto_extract=False)
        engine.process_turn("What subjects have we covered so far?", conversation_id=conv.id, auto_extract=False)

        llm_calls.clear()
        res = engine.process_turn("Repeat the message I sent before this one.", conversation_id=conv.id, auto_extract=True)

        assert res.answer == "What subjects have we covered so far?"
        assert len(llm_calls) == 0, "Expected deterministic store lookup without invoking LLM"
        assert res.memory_extracted is False
        assert res.diagnostics["completion_reason"] == "deterministic_store_lookup"


def test_personal_recall_receives_active_conversation_context_for_pronouns(tmp_path):
    """Personal recall passes active conversation context so pronouns ('it', 'that') resolve without losing grounding."""
    captured_prompts = []

    def mock_llm(prompt, system_prompt):
        captured_prompts.append(prompt)
        return json.dumps({
            "status": "insufficient_evidence",
            "answer": "Based on the recorded conversations, there is no record of when you bought the telescope.",
            "cited_memory_ids": [],
            "limitations": "No purchase date recorded.",
        })

    with ConversationStore(tmp_path / "conv.db") as store:
        mem_path = tmp_path / "memories.json"
        mem_path.write_text("[]", encoding="utf-8")
        add_memory(
            text="I have a Celestron telescope in my study.",
            timestamp="2026-02-01",
            source="chat",
            type="fact",
            importance=0.8,
            data_path=mem_path,
        )
        engine = ChatEngine(
            conversation_store=store,
            memories_path=mem_path,
            entities_path=tmp_path / "entities.json",
            llm_caller=mock_llm,
        )
        conv = engine.get_or_create_conversation("Pronoun Recall")
        # Establish active topic in conversation
        store.add_turn(conv.id, "user", "Where is my Celestron telescope?", speaker="user")
        store.add_turn(conv.id, "assistant", "It is in your study.", speaker="IVARA")

        captured_prompts.clear()
        res = engine.process_turn("When did I buy that telescope?", conversation_id=conv.id, auto_extract=False)

        assert res.intent == "personal_recall"
        assert len(captured_prompts) == 1
        assert "<active_conversation_context>" in captured_prompts[0]
        assert "Where is my Celestron telescope?" in captured_prompts[0]
        assert res.diagnostics["retrieval_performed"] is True


def test_creative_diversity_prompt_and_temperature(tmp_path, monkeypatch):
    """Creative requests use novelty instructions and elevated temperature while keeping factual QA conservative."""
    recorded_opts = []

    def fake_ollama_chat(model, messages, options):
        recorded_opts.append((messages, options))
        return {
            "message": {"content": "Why do programmers prefer dark mode? Because light attracts bugs."},
            "done_reason": "stop",
            "eval_count": 20,
            "prompt_eval_count": 50,
        }

    monkeypatch.setattr("src.chat.engine.ollama_chat", fake_ollama_chat)

    with ConversationStore(tmp_path / "conv.db") as store:
        mem_path = tmp_path / "memories.json"
        mem_path.write_text("[]", encoding="utf-8")
        engine = ChatEngine(
            conversation_store=store,
            memories_path=mem_path,
            entities_path=tmp_path / "entities.json",
            conversational_temperature=0.7,
        )
        conv = engine.get_or_create_conversation("Creative Test")
        engine.process_turn("Tell me a joke.", conversation_id=conv.id, auto_extract=False)
        res = engine.process_turn("Tell me another one.", conversation_id=conv.id, auto_extract=False)

        assert len(recorded_opts) == 2
        msgs2, opts2 = recorded_opts[1]
        assert opts2["temperature"] >= 0.85
        assert "CREATIVE DIVERSITY" in msgs2[0]["content"]
        assert "scarecrow" in msgs2[0]["content"].lower()
        assert res.diagnostics["completion_reason"] == "stop"


def test_ollama_thinking_exhaustion_retry_and_truncation_handling(tmp_path, monkeypatch):
    """Empty output from thinking exhaustion triggers one bounded retry; length truncation is flagged explicitly."""
    call_count = {"n": 0}

    def fake_ollama_chat(model, messages, options):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call exhausts num_predict in thinking with empty content
            return {
                "message": {"content": "", "thinking": "long internal reasoning..."},
                "done_reason": "length",
                "eval_count": options["num_predict"],
                "prompt_eval_count": 40,
            }
        # Retry produces content, but hits length limit
        return {
            "message": {"content": "Here is the concise partial answer"},
            "done_reason": "length",
            "eval_count": options["num_predict"],
            "prompt_eval_count": 60,
        }

    monkeypatch.setattr("src.chat.engine.ollama_chat", fake_ollama_chat)

    with ConversationStore(tmp_path / "conv.db") as store:
        mem_path = tmp_path / "memories.json"
        mem_path.write_text("[]", encoding="utf-8")
        engine = ChatEngine(
            conversation_store=store,
            memories_path=mem_path,
            entities_path=tmp_path / "entities.json",
            conversational_num_predict=512,
        )
        conv = engine.get_or_create_conversation("Truncation Test")
        res = engine.process_turn("Explain quantum tunneling.", conversation_id=conv.id, auto_extract=False)

        assert call_count["n"] == 2
        assert "Here is the concise partial answer" in res.answer
        assert "output token limit" in res.answer.lower()
        assert res.diagnostics["completion_reason"] == "length"
        assert res.diagnostics["eval_count"] == 2048



