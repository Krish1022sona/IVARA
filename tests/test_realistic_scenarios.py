"""
Robustness and Realistic User Scenarios Test Suite (Milestone 3.2 Batch 5 - Phase 7).

Verifies 12 realistic personal-memory situations:
 1. A person changes their college, job, or residence (temporal transition).
 2. A person expresses conflicting preferences over time.
 3. Two people have similar names (entity disambiguation / distinctness).
 4. A statement is corrected in a later conversation.
 5. A user refers to an event indirectly.
 6. A conversation contains multiple speakers (assistant vs user attribution).
 7. A relevant fact appears across multiple sessions.
 8. A relationship is implied but never explicitly confirmed (explicit vs inferred distinction).
 9. A user asks about something never discussed (honest abstention).
10. A user asks a multi-hop question requiring multiple pieces of evidence.
11. A user asks what they previously believed versus what they believe now (epistemic change).
12. A user asks whether something happened, but the available evidence is incomplete (partial answer).

Guarantees IVARA does not confuse:
- Unknown with false
- Missing with negative
- Inferred with explicit
- Historical with current
- Candidate relationships with confirmed relationships
- Assistant statements with user statements
"""

import json
import pytest
from unittest.mock import MagicMock

from src.conversations.store import EpisodicTurn
from src.answering import answer_question
from src.unified_retrieval.pipeline import UnifiedRetriever


class TestRealisticUserScenarios:
    """Deterministic validation of 12 realistic memory situations."""

    def test_scenario_01_job_or_residence_transition(self):
        """1. A person changes their job or residence over time."""
        turns = [
            EpisodicTurn(turn_id="t1", conversation_id="c1", speaker="user", speaker_role="user",
                         content="In 2022 I started as Senior Engineer at HyperScale Systems.",
                         timestamp="2022-07-01T10:00:00Z", sequence_num=1, session_id="s1"),
            EpisodicTurn(turn_id="t2", conversation_id="c2", speaker="user", speaker_role="user",
                         content="In February 2025 I switched to Aetheria AI as Principal Architect.",
                         timestamp="2025-02-15T10:00:00Z", sequence_num=1, session_id="s2")
        ]
        def mock_llm(prompt, sys_prompt):
            assert "HyperScale" in prompt and "Aetheria" in prompt
            return json.dumps({
                "status": "answered",
                "answer": "Arjun previously worked at HyperScale Systems before switching to Aetheria AI in February 2025 [t1, t2].",
                "cited_memory_ids": ["t1", "t2"],
                "limitations": None
            })

        res = answer_question("What was my previous job before Aetheria AI?", episodic_turns=turns, memories=[], llm_caller=mock_llm, use_adaptive_routing=True, min_score=0.10)
        assert res["status"] == "answered"
        assert "HyperScale" in res["answer"]
        assert len(res["citations"]) == 2

    def test_scenario_02_conflicting_preferences_over_time(self):
        """2. A person expresses conflicting preferences over time (resolved temporally)."""
        turns = [
            EpisodicTurn(turn_id="t1", conversation_id="c1", speaker="user", speaker_role="user",
                         content="I prefer tea over coffee; coffee gives me jitters.",
                         timestamp="2023-01-10T09:00:00Z", sequence_num=1, session_id="s1"),
            EpisodicTurn(turn_id="t2", conversation_id="c2", speaker="user", speaker_role="user",
                         content="Lately in Zurich I drink single-origin espresso every morning and love it.",
                         timestamp="2026-11-01T08:30:00Z", sequence_num=1, session_id="s2")
        ]
        def mock_llm(prompt, sys_prompt):
            return json.dumps({
                "status": "answered",
                "answer": "Arjun previously preferred tea in 2023 [t1], but in Zurich as of late 2026 he enjoys single-origin espresso [t2].",
                "cited_memory_ids": ["t1", "t2"],
                "limitations": "Preference shifted over time."
            })

        res = answer_question("What hot beverage do I prefer?", episodic_turns=turns, memories=[], llm_caller=mock_llm, use_adaptive_routing=True)
        assert res["status"] == "answered"
        assert "tea" in res["answer"].lower() and "espresso" in res["answer"].lower()

    def test_scenario_03_similar_names_disambiguation(self):
        """3. Two people have similar names (Rohan Sharma vs Rohan Verma)."""
        turns = [
            EpisodicTurn(turn_id="t1", conversation_id="c1", speaker="user", speaker_role="user",
                         content="Rohan Sharma was my college roommate at IIITM Gwalior.",
                         timestamp="2022-01-15T10:00:00Z", sequence_num=1, session_id="s1"),
            EpisodicTurn(turn_id="t2", conversation_id="c2", speaker="user", speaker_role="user",
                         content="Rohan Verma is a client contact at CloudScale Ventures.",
                         timestamp="2025-04-10T11:00:00Z", sequence_num=1, session_id="s2")
        ]
        def mock_llm(prompt, sys_prompt):
            return json.dumps({
                "status": "answered",
                "answer": "Rohan Sharma was Arjun's college roommate [t1], whereas Rohan Verma is a client at CloudScale Ventures [t2].",
                "cited_memory_ids": ["t1"],
                "limitations": None
            })

        res = answer_question("Who was my college roommate named Rohan?", episodic_turns=turns, memories=[], llm_caller=mock_llm, use_adaptive_routing=True)
        assert res["status"] == "answered"
        assert "Sharma" in res["answer"]

    def test_scenario_04_statement_corrected_in_later_conversation(self):
        """4. A factual statement is explicitly corrected in a later conversation."""
        turns = [
            EpisodicTurn(turn_id="t1", conversation_id="c1", speaker="user", speaker_role="user",
                         content="I think my graduation GPA was 8.4.",
                         timestamp="2023-02-01T10:00:00Z", sequence_num=1, session_id="s1"),
            EpisodicTurn(turn_id="t2", conversation_id="c2", speaker="user", speaker_role="user",
                         content="I checked my transcript: my actual graduation GPA was 8.62, not 8.4.",
                         timestamp="2023-05-15T14:00:00Z", sequence_num=1, session_id="s2")
        ]
        def mock_llm(prompt, sys_prompt):
            return json.dumps({
                "status": "answered",
                "answer": "Arjun's final graduation GPA was 8.62, correcting an earlier estimate of 8.4 [t2].",
                "cited_memory_ids": ["t2"],
                "limitations": None
            })

        res = answer_question("What was my graduation GPA?", episodic_turns=turns, memories=[], llm_caller=mock_llm, use_adaptive_routing=True)
        assert res["status"] == "answered"
        assert "8.62" in res["answer"]

    def test_scenario_05_indirect_event_reference(self):
        """5. A user refers to an event indirectly (the big move to Europe)."""
        turns = [
            EpisodicTurn(turn_id="t1", conversation_id="c1", speaker="user", speaker_role="user",
                         content="In September 2026, I relocated to Zurich, Switzerland to join the ETH research lab.",
                         timestamp="2026-09-01T12:00:00Z", sequence_num=1, session_id="s1")
        ]
        def mock_llm(prompt, sys_prompt):
            assert "Zurich" in prompt
            return json.dumps({
                "status": "answered",
                "answer": "The relocation to Europe occurred in September 2026 when Arjun moved to Zurich, Switzerland [t1].",
                "cited_memory_ids": ["t1"],
                "limitations": None
            })

        res = answer_question("When did my big move to Europe happen?", episodic_turns=turns, memories=[], llm_caller=mock_llm, use_adaptive_routing=True)
        assert res["status"] == "answered"
        assert "September 2026" in res["answer"] or "Zurich" in res["answer"]

    def test_scenario_06_multiple_speakers_attribution(self):
        """6. Multiple speakers: statements by the assistant must not be confused with user claims."""
        turns = [
            EpisodicTurn(turn_id="t1", conversation_id="c1", speaker="Assistant", speaker_role="assistant",
                         content="Have you ever considered relocating to Tokyo?",
                         timestamp="2024-05-01T10:00:00Z", sequence_num=1, session_id="s1"),
            EpisodicTurn(turn_id="t2", conversation_id="c1", speaker="Arjun", speaker_role="user",
                         content="No, I have never visited Tokyo or lived in Japan.",
                         timestamp="2024-05-01T10:01:00Z", sequence_num=2, session_id="s1")
        ]
        def mock_llm(prompt, sys_prompt):
            assert "Assistant" in prompt and "Arjun" in prompt
            return json.dumps({
                "status": "answered",
                "answer": "Arjun stated that he has never visited Tokyo or lived in Japan [t2]. The assistant only asked if he had considered it [t1].",
                "cited_memory_ids": ["t2"],
                "limitations": None
            })

        res = answer_question("Did I live in Tokyo?", episodic_turns=turns, memories=[], llm_caller=mock_llm, use_adaptive_routing=True)
        assert res["status"] == "answered"
        assert "never" in res["answer"].lower()

    def test_scenario_07_fact_across_multiple_sessions(self):
        """7. A relevant fact synthesis requires combining evidence across distinct sessions."""
        turns = [
            EpisodicTurn(turn_id="t1", conversation_id="c1", speaker="user", speaker_role="user",
                         content="We kicked off Project CacheMesh at HyperScale in October 2022.",
                         timestamp="2022-10-05T10:00:00Z", sequence_num=1, session_id="s1"),
            EpisodicTurn(turn_id="t2", conversation_id="c2", speaker="user", speaker_role="user",
                         content="After two years in production, we deprecated CacheMesh in December 2024.",
                         timestamp="2024-12-10T14:00:00Z", sequence_num=1, session_id="s2")
        ]
        def mock_llm(prompt, sys_prompt):
            assert "October 2022" in prompt and "December 2024" in prompt
            return json.dumps({
                "status": "answered",
                "answer": "Project CacheMesh ran from October 2022 [t1] until its deprecation in December 2024 [t2].",
                "cited_memory_ids": ["t1", "t2"],
                "limitations": None
            })

        res = answer_question("What was the full duration of CacheMesh?", episodic_turns=turns, memories=[], llm_caller=mock_llm, use_adaptive_routing=True)
        assert res["status"] == "answered"
        assert "October 2022" in res["answer"] and "December 2024" in res["answer"]
        assert len(res["citations"]) == 2

    def test_scenario_08_implied_vs_explicit_relationship(self):
        """8. An implied relationship is qualified as inference rather than confirmed fact."""
        mems = [
            {
                "id": "m100",
                "text": "Arjun attended Sneha Roy's wedding and gave a speech alongside Rohan.",
                "timestamp": "2024-08-15",
                "source": "t1",
                "relationships": [
                    {
                        "source": "Arjun",
                        "predicate": "knows",
                        "target": "Sneha Roy",
                        "origin": "inferred",
                        "evidence": "attended wedding and gave speech"
                    }
                ]
            }
        ]
        def mock_llm(prompt, sys_prompt):
            assert "INFERRED RELATIONSHIPS" in prompt or "inferred" in prompt.lower()
            return json.dumps({
                "status": "answered",
                "answer": "The memory implies Arjun knows Sneha Roy because he attended her wedding and gave a speech [m100], though no closer relation is explicitly stated.",
                "cited_memory_ids": ["m100"],
                "limitations": "Relationship is inferred from event attendance."
            })

        res = answer_question("What is my relationship to Sneha Roy?", memories=mems, llm_caller=mock_llm)
        assert res["status"] == "answered"
        assert "implies" in res["answer"].lower() or "inferred" in res["answer"].lower()

    def test_scenario_09_never_discussed_honest_abstention(self):
        """9. A user asks about something never discussed (Mars rover hobby)."""
        mock_llm = MagicMock()
        res = answer_question("What is my hobby regarding Mars rovers?", episodic_turns=[], memories=[], llm_caller=mock_llm, use_adaptive_routing=True)
        assert res["status"] in ("unanswerable", "insufficient_evidence")
        assert len(res["citations"]) == 0
        mock_llm.assert_not_called()

    def test_scenario_10_multi_hop_bridge_question(self):
        """10. Multi-hop inquiry requires traversing two distinct conversation turns."""
        turns = [
            EpisodicTurn(turn_id="t1", conversation_id="c1", speaker="user", speaker_role="user",
                         content="Priya Sharma was the principal mentor for our SIH team.",
                         timestamp="2021-09-01T10:00:00Z", sequence_num=1, session_id="s1"),
            EpisodicTurn(turn_id="t2", conversation_id="c2", speaker="user", speaker_role="user",
                         content="Priya Sharma relocated to Munich in 2024 to join BMW Tech.",
                         timestamp="2024-03-15T11:00:00Z", sequence_num=1, session_id="s2")
        ]
        def mock_llm(prompt, sys_prompt):
            assert "Priya Sharma" in prompt and "Munich" in prompt
            return json.dumps({
                "status": "answered",
                "answer": "Arjun's SIH mentor, Priya Sharma [t1], relocated to Munich in 2024 to work at BMW Tech [t2].",
                "cited_memory_ids": ["t1", "t2"],
                "limitations": None
            })

        res = answer_question("In which city does my SIH mentor currently live?", episodic_turns=turns, memories=[], llm_caller=mock_llm, use_adaptive_routing=True)
        assert res["status"] == "answered"
        assert "Munich" in res["answer"]
        assert len(res["citations"]) == 2

    def test_scenario_11_epistemic_past_belief_vs_present(self):
        """11. A user asks what they previously believed versus what they know now."""
        turns = [
            EpisodicTurn(turn_id="t1", conversation_id="c1", speaker="user", speaker_role="user",
                         content="I used to think NoSQL was strictly superior to relational databases.",
                         timestamp="2022-03-10T10:00:00Z", sequence_num=1, session_id="s1"),
            EpisodicTurn(turn_id="t2", conversation_id="c2", speaker="user", speaker_role="user",
                         content="Now I realize PostgreSQL with JSONB solves 95% of use cases with full ACID safety.",
                         timestamp="2025-01-20T16:00:00Z", sequence_num=1, session_id="s2")
        ]
        def mock_llm(prompt, sys_prompt):
            return json.dumps({
                "status": "answered",
                "answer": "Arjun previously believed NoSQL was strictly superior to relational databases [t1], but now favors PostgreSQL with JSONB for ACID guarantees [t2].",
                "cited_memory_ids": ["t1", "t2"],
                "limitations": None
            })

        res = answer_question("What did I previously believe about databases compared to now?", episodic_turns=turns, memories=[], llm_caller=mock_llm, use_adaptive_routing=True)
        assert res["status"] == "answered"
        assert "NoSQL" in res["answer"] and "PostgreSQL" in res["answer"]

    def test_scenario_12_incomplete_evidence_partial_answer(self):
        """12. A user asks whether something happened, but available evidence is incomplete."""
        turns = [
            EpisodicTurn(turn_id="t1", conversation_id="c1", speaker="user", speaker_role="user",
                         content="I submitted the paper 'CacheMesh: Distributed Tiered Invalidation' to NSDI 2024.",
                         timestamp="2024-01-10T11:00:00Z", sequence_num=1, session_id="s1")
        ]
        def mock_llm(prompt, sys_prompt):
            return json.dumps({
                "status": "partial",
                "answer": "The records confirm Arjun submitted the CacheMesh paper to NSDI 2024 [t1], but whether it was ultimately accepted or rejected is not recorded.",
                "cited_memory_ids": ["t1"],
                "limitations": "Final acceptance or rejection status is missing from recorded memories."
            })

        res = answer_question("Was my CacheMesh paper accepted at NSDI 2024?", episodic_turns=turns, memories=[], llm_caller=mock_llm, use_adaptive_routing=True)
        assert res["status"] == "partial"
        assert "submitted" in res["answer"].lower()
        assert "acceptance" in res["limitations"].lower()
