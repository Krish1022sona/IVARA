"""
Comprehensive unit and integration tests for calibrated evidence-grounded answer generation
(Milestone 3.2 Batch 4).

All tests run fully offline with mock LLM callers and isolated memory/turn fixtures,
guaranteeing 0 external calls, instant execution, and zero mutation of production databases.
"""

import json
import pytest
from unittest.mock import MagicMock

from src.answering import (
    AnswerPayload,
    answer_question,
    format_user_prompt,
    validate_citations,
    _normalize_citation_id,
    ANSWERING_SYSTEM_PROMPT
)
from src.conversations.store import EpisodicTurn
from src.unified_retrieval.router import AdaptiveRouter, RouteDecision
from src.unified_retrieval.pipeline import UnifiedRetriever, UnifiedRetrievalResult
from src.unified_retrieval.evidence_selector import (
    SelectedEvidenceItem,
    SelectionResult,
    EpisodicEvidenceItem,
    SemanticEvidenceItem,
    GraphEvidenceItem
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_episodic_turns():
    return [
        EpisodicTurn(
            turn_id="msg_001",
            conversation_id="conv_01",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="I was born in Chandigarh on August 14, 1999.",
            timestamp="2022-01-10T10:00:00Z",
            sequence_num=1,
            session_id="session_01"
        ),
        EpisodicTurn(
            turn_id="msg_002",
            conversation_id="conv_01",
            speaker="Rohan Sharma",
            speaker_role="assistant",
            content="Happy to be your college roommate at IIITM Gwalior!",
            timestamp="2022-01-10T10:01:00Z",
            sequence_num=2,
            session_id="session_01"
        ),
        EpisodicTurn(
            turn_id="msg_003",
            conversation_id="conv_02",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="I lived in Indiranagar, Bengaluru from July 2022 until late 2024.",
            timestamp="2024-11-01T10:00:00Z",
            sequence_num=1,
            session_id="session_02"
        ),
        EpisodicTurn(
            turn_id="msg_004",
            conversation_id="conv_02",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="In December 2024, I moved to HSR Layout Sector 2.",
            timestamp="2024-12-15T10:00:00Z",
            sequence_num=2,
            session_id="session_02"
        ),
        EpisodicTurn(
            turn_id="msg_005",
            conversation_id="conv_03",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="Rohan built the cloud backend for our AgriVision project using FastAPI.",
            timestamp="2023-05-10T14:00:00Z",
            sequence_num=1,
            session_id="session_03"
        ),
        EpisodicTurn(
            turn_id="msg_006",
            conversation_id="conv_04",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="In Zurich, I do not own an automobile or car. I use the S-Bahn, Tram 4, and my cargo e-bike.",
            timestamp="2026-10-01T09:00:00Z",
            sequence_num=1,
            session_id="session_04"
        )
    ]


@pytest.fixture
def sample_semantic_memories():
    return [
        {
            "id": "m001",
            "text": "Arjun was born in Chandigarh on August 14, 1999.",
            "timestamp": "2022-01-10T10:00:00Z",
            "source": "msg_001",
            "entities": [{"canonical_name": "Arjun Mehta", "types": ["Person"]}],
            "relationships": []
        },
        {
            "id": "m002",
            "text": "Rohan Sharma was Arjun's college roommate at IIITM Gwalior.",
            "timestamp": "2022-01-10T10:01:00Z",
            "source": "msg_002",
            "entities": [{"canonical_name": "Rohan Sharma", "types": ["Person"]}],
            "relationships": []
        },
        {
            "id": "m003",
            "text": "Arjun lived in Indiranagar before moving to HSR Layout in Dec 2024.",
            "timestamp": "2024-11-01T10:00:00Z",
            "source": "msg_003",
            "entities": [],
            "relationships": []
        }
    ]


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestCalibratedAnswering:
    """Comprehensive unit testing of calibrated answer generation."""

    def test_direct_factual_recall(self, sample_episodic_turns, sample_semantic_memories):
        """1. Direct factual recall with valid episodic turn citation."""
        def mock_llm(prompt, sys_prompt):
            assert "born in Chandigarh" in prompt
            return json.dumps({
                "status": "answered",
                "answer": "Arjun Mehta was born in Chandigarh on August 14, 1999 [msg_001].",
                "cited_memory_ids": ["msg_001"],
                "limitations": None
            })

        res = answer_question(
            query="In which city was Arjun Mehta born?",
            episodic_turns=sample_episodic_turns,
            memories=sample_semantic_memories,
            llm_caller=mock_llm,
            use_adaptive_routing=True
        )

        assert res["status"] == "answered"
        assert "Chandigarh" in res["answer"]
        assert len(res["citations"]) == 1
        assert res["citations"][0]["memory_id"] in ("msg_001", "m001")
        assert res["rejected_citation_ids"] == []

    def test_historical_state_resolution(self, sample_episodic_turns, sample_semantic_memories):
        """2. Historical state query returns prior state and notes transition."""
        def mock_llm(prompt, sys_prompt):
            assert "Indiranagar" in prompt
            assert "historical-state" in prompt.lower() or "historical" in prompt.lower()
            return json.dumps({
                "status": "answered",
                "answer": "Before moving to HSR Layout in December 2024, Arjun lived in Indiranagar, Bengaluru [msg_003].",
                "cited_memory_ids": ["msg_003"],
                "limitations": None
            })

        res = answer_question(
            query="Where did Arjun live originally before moving to HSR Layout Sector 2?",
            episodic_turns=sample_episodic_turns,
            memories=sample_semantic_memories,
            llm_caller=mock_llm,
            use_adaptive_routing=True
        )

        assert res["status"] == "answered"
        assert "Indiranagar" in res["answer"]
        assert len(res["citations"]) >= 1

    def test_contradictory_statements_unresolved(self):
        """3. Unresolved factual contradictions trigger status 'conflicting_evidence'."""
        competing_memories = [
            {"id": "m_c1", "text": "Arjun's base salary at HyperScale was 18 LPA.", "timestamp": None, "source": "turn_1"},
            {"id": "m_c2", "text": "Arjun's base salary at HyperScale was 24 LPA.", "timestamp": None, "source": "turn_2"}
        ]

        def mock_llm(prompt, sys_prompt):
            return json.dumps({
                "status": "conflicting_evidence",
                "answer": "There are conflicting records regarding Arjun's base salary: one record states 18 LPA [m_c1] while another states 24 LPA [m_c2].",
                "cited_memory_ids": ["m_c1", "m_c2"],
                "limitations": "Unresolved contradiction between records."
            })

        res = answer_question(
            query="What was Arjun's base salary at HyperScale?",
            memories=competing_memories,
            llm_caller=mock_llm
        )

        assert res["status"] == "conflicting_evidence"
        assert "18 LPA" in res["answer"] and "24 LPA" in res["answer"]
        assert len(res["citations"]) == 2

    def test_multiple_speakers_and_attribution(self, sample_episodic_turns):
        """4. Multiple speakers: user first-person facts correctly mapped to Arjun Mehta."""
        def mock_llm(prompt, sys_prompt):
            assert "Arjun Mehta" in prompt
            assert "Rohan Sharma" in prompt
            return json.dumps({
                "status": "answered",
                "answer": "Rohan Sharma was Arjun's college roommate at IIITM Gwalior [msg_002].",
                "cited_memory_ids": ["msg_002"],
                "limitations": None
            })

        res = answer_question(
            query="Who was Arjun's college roommate?",
            episodic_turns=sample_episodic_turns,
            llm_caller=mock_llm,
            use_adaptive_routing=True
        )

        assert res["status"] == "answered"
        assert "Rohan Sharma" in res["answer"]

    def test_multi_hop_reasoning(self, sample_episodic_turns):
        """5. Multi-hop conversational bridge: roommate -> AgriVision backend -> FastAPI."""
        def mock_llm(prompt, sys_prompt):
            assert "FastAPI" in prompt
            return json.dumps({
                "status": "answered",
                "answer": "Rohan Sharma built the cloud backend for AgriVision using FastAPI [msg_005].",
                "cited_memory_ids": ["msg_005"],
                "limitations": None
            })

        res = answer_question(
            query="What framework did the person who managed the cloud backend for AgriVision use?",
            episodic_turns=sample_episodic_turns,
            llm_caller=mock_llm,
            use_adaptive_routing=True
        )

        assert res["status"] == "answered"
        assert "FastAPI" in res["answer"]

    def test_partially_supported_answers(self, sample_episodic_turns):
        """6. Partially supported query returns status 'partial' and records missing details."""
        def mock_llm(prompt, sys_prompt):
            return json.dumps({
                "status": "partial",
                "answer": "Arjun was born in Chandigarh on August 14, 1999 [msg_001]. However, his birth hospital is not recorded.",
                "cited_memory_ids": ["msg_001"],
                "limitations": "The specific hospital of birth is not mentioned in the recorded memories."
            })

        res = answer_question(
            query="In which city and hospital was Arjun born?",
            episodic_turns=sample_episodic_turns,
            llm_caller=mock_llm,
            use_adaptive_routing=True
        )

        assert res["status"] == "partial"
        assert "Chandigarh" in res["answer"]
        assert "hospital" in res["limitations"].lower()

    def test_unanswerable_query_fast_abstention(self):
        """7. Out-of-domain or unanswerable query immediately abstains without LLM invocation."""
        mock_llm = MagicMock()

        res = answer_question(
            query="What is the capital of Mars according to Arjun?",
            episodic_turns=[],
            memories=[],
            llm_caller=mock_llm,
            use_adaptive_routing=True
        )

        assert res["status"] == "unanswerable"
        assert "don't have any recorded memory" in res["answer"].lower() or "unrecorded" in res["limitations"].lower()
        # Crucial: LLM must not have been invoked
        mock_llm.assert_not_called()

    def test_negative_information_framing(self, sample_episodic_turns):
        """8. Negative information query employs absence-of-evidence framing."""
        def mock_llm(prompt, sys_prompt):
            assert "absence-of-evidence" in prompt.lower() or "cargo e-bike" in prompt.lower()
            return json.dumps({
                "status": "answered",
                "answer": "Based on the recorded conversations, Arjun does not own a car or Tesla in Switzerland; he commutes via S-Bahn, Tram 4, and an e-bike [msg_006].",
                "cited_memory_ids": ["msg_006"],
                "limitations": None
            })

        res = answer_question(
            query="Does Arjun own an electric passenger car such as a Tesla in Switzerland?",
            episodic_turns=sample_episodic_turns,
            llm_caller=mock_llm,
            use_adaptive_routing=True
        )

        assert res["status"] == "answered"
        assert "does not own" in res["answer"].lower()

    def test_missing_evidence_immediate_abstention(self):
        """9. When no memories match query, immediately abstain with insufficient_evidence."""
        mock_llm = MagicMock()

        res = answer_question(
            query="Tell me about the conference in Tokyo",
            episodic_turns=[],
            memories=[],
            llm_caller=mock_llm
        )

        assert res["status"] == "insufficient_evidence"
        assert "sufficient" in res["answer"].lower()
        mock_llm.assert_not_called()

    def test_invalid_and_hallucinated_citations_rejected(self, sample_episodic_turns):
        """10. Hallucinated or fake citations are rejected and moved to rejected_citation_ids."""
        def mock_llm(prompt, sys_prompt):
            return json.dumps({
                "status": "answered",
                "answer": "Arjun was born in Chandigarh [fake_mem_999].",
                "cited_memory_ids": ["fake_mem_999"],
                "limitations": None
            })

        res = answer_question(
            query="Where was Arjun born?",
            episodic_turns=sample_episodic_turns,
            llm_caller=mock_llm,
            use_adaptive_routing=True
        )

        # Status must be demoted to partial because valid citations were 0
        assert res["status"] == "partial"
        assert "fake_mem_999" in res["rejected_citation_ids"]
        assert len(res["citations"]) == 0
        assert "Rejected unsupported" in res["limitations"] or "lacked verifiable" in res["limitations"]

    def test_graph_and_episodic_disagreement(self, sample_episodic_turns):
        """11. Direct episodic conversational turns take precedence over conflicting graph claims."""
        def mock_llm(prompt, sys_prompt):
            # System prompt instructs that conversational turns take precedence
            assert "prioritize the conversational statement" in sys_prompt
            return json.dumps({
                "status": "answered",
                "answer": "Rohan Sharma was Arjun's college roommate [msg_002].",
                "cited_memory_ids": ["msg_002"],
                "limitations": None
            })

        res = answer_question(
            query="Who was Arjun's college roommate?",
            episodic_turns=sample_episodic_turns,
            llm_caller=mock_llm,
            use_adaptive_routing=True
        )

        assert res["status"] == "answered"
        assert "Rohan Sharma" in res["answer"]

    def test_oversized_context_prompt_truncation(self):
        """12. Format_user_prompt safely truncates context exceeding max_prompt_tokens."""
        huge_context = {
            "context_text": "This is a long memory context. " * 1500,
            "diagnostics": {"routing": {"category": "single_fact"}}
        }

        prompt = format_user_prompt(
            query="Tell me about everything",
            context=huge_context,
            max_prompt_tokens=500
        )

        assert "[... Context truncated to preserve model token limit ...]" in prompt
        # Approx 500 tokens * 3.5 = 1750 chars max for context block
        assert len(prompt) < 3000

    def test_malformed_model_output_fallback(self, sample_episodic_turns):
        """13. Model returning unstructured or garbage text triggers safe fallback."""
        def mock_llm_garbage(prompt, sys_prompt):
            return "This is completely unparseable garbage output with no JSON {{"

        res = answer_question(
            query="Where was Arjun born?",
            episodic_turns=sample_episodic_turns,
            llm_caller=mock_llm_garbage,
            use_adaptive_routing=True
        )

        assert res["status"] in ("partial", "insufficient_evidence")
        assert "malformed" in res["limitations"].lower()

    def test_citation_normalization(self):
        """14. _normalize_citation_id handles various brackets and prefixes."""
        assert _normalize_citation_id("Turn: msg_001") == "001"
        assert _normalize_citation_id("turn_002") == "002"
        assert _normalize_citation_id("['m003']") == "003"
        assert _normalize_citation_id("[msg_004]") == "004"
