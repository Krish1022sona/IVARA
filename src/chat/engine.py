"""Interactive Chat Engine for IVARA.

Orchestrates conversational sessions, grounded memory retrieval, conversational response
generation via local Ollama models (or injectable mock callers), and automatic long-term
memory formation with strict error isolation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable, Optional

from ..conversations.store import ConversationStore, Conversation, Message, EpisodicTurn, DEFAULT_CONVERSATIONS_DB
from ..memory import DATA_PATH as DEFAULT_MEMORIES_PATH, load_memories
from ..ontology.entity_store import DATA_PATH as DEFAULT_ENTITIES_PATH
from ..graph.store import SQLiteGraphStore
from ..answering import answer_question
from ..extraction.pipeline import process_and_store_memory
from ..extraction.policy import (
    MemoryCategory,
    MemoryFormationDecision,
    evaluate_memory_formation,
    is_casual_or_filler,
    is_pure_question_or_recall,
    is_general_knowledge_query
)

try:
    from ollama import chat as ollama_chat
except ImportError:
    ollama_chat = None


def is_trivial_or_greeting(text: str) -> bool:
    """Check if the text is a trivial greeting, filler, or acknowledgement."""
    return is_casual_or_filler(text)


def is_pure_question(text: str) -> bool:
    """Check if the utterance is purely an inquiry without declarative personal statements."""
    return is_pure_question_or_recall(text)


def should_extract_memory(text: str, existing_memories: Optional[list[dict]] = None) -> bool:
    """
    Determine whether a user utterance contains potential long-term personal facts.
    Rejects trivial greetings, pure questions, and short acknowledgements.
    """
    decision = evaluate_memory_formation(text, existing_memories=existing_memories)
    return decision.should_extract


@dataclass
class ChatTurnResult:
    """Result of an interactive conversation turn."""
    user_message: Message
    assistant_message: Message
    answer: str
    status: str  # 'answered', 'partial', 'conversational', 'insufficient_evidence', 'error'
    citations: list[str] = field(default_factory=list)
    memory_extracted: bool = False
    extracted_memory_id: Optional[str] = None
    extraction_error: Optional[str] = None
    user_turn: Optional[EpisodicTurn] = None
    assistant_turn: Optional[EpisodicTurn] = None


class ChatEngine:
    """Core conversational engine integrating SQLite history and memory intelligence."""

    def __init__(
        self,
        conversation_store: Optional[ConversationStore] = None,
        graph_store: Optional[SQLiteGraphStore] = None,
        memories_path: Optional[str | Path] = None,
        entities_path: Optional[str | Path] = None,
        model: str = "gpt-oss",
        llm_caller: Optional[Callable[[str, str], Any]] = None,
        use_unified: bool = True,
        use_adaptive_routing: bool = False
    ):
        self.conv_store = conversation_store or ConversationStore(DEFAULT_CONVERSATIONS_DB)
        self.graph_store = graph_store
        self.memories_path = memories_path or DEFAULT_MEMORIES_PATH
        self.entities_path = entities_path or DEFAULT_ENTITIES_PATH
        self.model = model
        self.llm_caller = llm_caller
        self.use_unified = use_unified
        self.use_adaptive_routing = use_adaptive_routing

    def get_or_create_conversation(self, conversation_id: Optional[str] = None, title: Optional[str] = None) -> Conversation:
        """Get an existing conversation or create a new session."""
        if conversation_id:
            conv = self.conv_store.get_conversation(conversation_id)
            if conv is not None:
                return conv
        return self.conv_store.create_conversation(title=title or "Chat Session")

    def process_turn(
        self,
        user_text: str,
        conversation_id: Optional[str] = None,
        auto_extract: bool = True
    ) -> ChatTurnResult:
        """
        Execute a full conversational turn:
          1. Persist user message immediately in SQLite.
          2. Retrieve relevant memories and perform grounded answering / conversational response.
          3. Persist assistant message in SQLite.
          4. If eligible, extract and store long-term memory with provenance.
        """
        clean_text = user_text.strip()
        conv = self.get_or_create_conversation(conversation_id)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Step 1: Persist user turn immediately in SQLite episodic layer
        user_turn = self.conv_store.add_turn(
            conversation_id=conv.id,
            role="user",
            content=clean_text,
            speaker="user"
        )
        user_msg = Message(
            id=user_turn.turn_id,
            conversation_id=conv.id,
            role="user",
            content=clean_text,
            created_at=user_turn.timestamp,
            sequence_num=user_turn.sequence_num,
            metadata=user_turn.metadata
        )

        # Step 2: Answer or respond
        answer_text = ""
        citations: list[str] = []
        turn_status = "answered"

        try:
            # Query memory using IVARA evidence-grounded answering
            use_graph = bool(self.graph_store is not None and self.graph_store.count_nodes() > 0)
            qa_res = answer_question(
                query=clean_text,
                k=5,
                min_score=0.30,
                data_path=self.memories_path,
                model=self.model,
                llm_caller=self.llm_caller,
                use_graph=use_graph,
                graph_store=self.graph_store,
                enable_temporal=True,
                conversation_store=self.conv_store,
                use_unified=self.use_unified,
                use_adaptive_routing=self.use_adaptive_routing
            )

            qa_status = qa_res.get("status", "insufficient_evidence")
            qa_citations = qa_res.get("citations", [])
            qa_answer = qa_res.get("answer", "").strip()

            if qa_status in ("answered", "partial") and qa_citations:
                answer_text = qa_answer
                citations = [c["memory_id"] for c in qa_citations if "memory_id" in c]
                turn_status = qa_status
            elif is_pure_question(clean_text):
                # Honest abstention for ungrounded question
                answer_text = qa_answer or f"I do not have enough verified memories to answer '{clean_text}'."
                turn_status = "insufficient_evidence"
            else:
                # Conversational acknowledgement for statements / general chat
                turn_status = "conversational"
                answer_text = self._generate_conversational_response(conv.id, clean_text)

        except Exception as e:
            # Check for Ollama connection error
            err_msg = str(e).lower()
            if "connection" in err_msg or "connect" in err_msg or "refused" in err_msg:
                answer_text = "Local Ollama service appears unavailable. Please ensure `ollama serve` is running."
                turn_status = "error"
            else:
                answer_text = f"I encountered an issue processing your request: {e}"
                turn_status = "error"

        # Step 3: Persist assistant response turn in SQLite episodic layer
        assistant_turn = self.conv_store.add_turn(
            conversation_id=conv.id,
            role="assistant",
            content=answer_text,
            speaker="assistant",
            metadata={"status": turn_status, "citations": citations}
        )
        assistant_msg = Message(
            id=assistant_turn.turn_id,
            conversation_id=conv.id,
            role="assistant",
            content=answer_text,
            created_at=assistant_turn.timestamp,
            sequence_num=assistant_turn.sequence_num,
            metadata=assistant_turn.metadata
        )

        # Step 4: Automatic memory formation (governed by memory formation policy)
        memory_extracted = False
        extracted_memory_id = None
        extraction_error = None

        if auto_extract:
            existing_mems = load_memories(self.memories_path)
            policy_decision = evaluate_memory_formation(
                clean_text,
                existing_memories=existing_mems,
                model=self.model,
                llm_caller=self.llm_caller
            )
            if policy_decision.should_extract:
                try:
                    conv_tag = conv.id if conv.id.startswith("conv_") else f"conv_{conv.id}"
                    msg_tag = user_turn.turn_id if user_turn.turn_id.startswith("turn_") else f"turn_{user_turn.turn_id}"
                    target_text = policy_decision.extracted_text or clean_text
                    importance = 0.9 if policy_decision.is_update else 0.8
                    mem = process_and_store_memory(
                        text=target_text,
                        timestamp=now_iso[:10],  # YYYY-MM-DD
                        source=f"{conv_tag}_{msg_tag}",
                        memory_type=policy_decision.memory_type,
                        importance=importance,
                        memory_store_path=self.memories_path,
                        entity_store_path=self.entities_path,
                        graph_store=self.graph_store,
                        llm_caller=self.llm_caller,
                        model=self.model
                    )
                    if mem and mem.get("id"):
                        memory_extracted = True
                        extracted_memory_id = mem["id"]
                        # Update turn and message metadata with extracted memory reference and category
                        meta = dict(user_turn.metadata or {})
                        meta["extracted_memory_id"] = extracted_memory_id
                        meta["memory_category"] = policy_decision.category.value
                        meta["is_update"] = policy_decision.is_update
                        user_turn.metadata = meta
                        user_msg.metadata = meta
                        # Update in SQLite
                        conn = self.conv_store._get_connection()
                        with conn:
                            conn.execute(
                                "UPDATE episodic_turns SET metadata = ? WHERE turn_id = ?",
                                (json.dumps(meta), user_turn.turn_id)
                            )
                            conn.execute(
                                "UPDATE messages SET metadata = ? WHERE id = ?",
                                (json.dumps(meta), user_turn.turn_id)
                            )
                except Exception as extract_err:
                    # Memory extraction error must NEVER corrupt or discard the conversation
                    extraction_error = str(extract_err)

        return ChatTurnResult(
            user_message=user_msg,
            assistant_message=assistant_msg,
            answer=answer_text,
            status=turn_status,
            citations=citations,
            memory_extracted=memory_extracted,
            extracted_memory_id=extracted_memory_id,
            extraction_error=extraction_error,
            user_turn=user_turn,
            assistant_turn=assistant_turn
        )

    def _generate_conversational_response(self, conversation_id: str, user_text: str) -> str:
        """Generate conversational acknowledgment or response incorporating recent context."""
        recent_msgs = self.conv_store.get_messages(conversation_id, limit=6, chronological=True)
        
        # If injectable mock is provided, use it
        if self.llm_caller is not None:
            mock_out = self.llm_caller(user_text, "Conversational Assistant")
            if isinstance(mock_out, str):
                return mock_out
            elif isinstance(mock_out, dict) and "answer" in mock_out:
                return mock_out["answer"]

        # If Ollama is available, call local model
        if ollama_chat is not None:
            messages = [
                {
                    "role": "system",
                    "content": "You are IVARA, a helpful and respectful Personal AI Memory Assistant. "
                               "Acknowledge the user's statement naturally and concisely. "
                               "Do not invent facts about yourself or claim memories you don't possess."
                }
            ]
            for m in recent_msgs:
                messages.append({"role": m.role, "content": m.content})
            
            resp = ollama_chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.3, "num_predict": 256}
            )
            return resp.message.content.strip()

        # Deterministic fallback acknowledgement
        return f"Understood. I have recorded that in this conversation."
