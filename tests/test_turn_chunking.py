"""
Tests for Turn-Aware Conversational Chunking (TACC) and B-Chunked Pipeline.

Covers:
- Chunk boundaries, windowing, and overlap logic.
- Preservation of speaker roles, session IDs, and timestamp metadata.
- Stable chunk IDs and provenance links back to original turns.
- Hybrid vector + BM25 chunk retrieval and turn-level mapping.
- Context token budget enforcement and compact evidence assembly.
- Gold-label isolation (no gold answers or labels in chunks).
- Verification that Config B baseline behavior remains unchanged.
"""

from datetime import datetime
import pytest

from src.chunking.turn_chunker import TurnChunk, TurnAwareChunker
from src.chunking.retrieval import ChunkRetriever, assemble_bounded_context
from src.answering import answer_question


class TestTurnAwareChunker:
    """Tests for turn-level windowing and metadata preservation."""

    def test_chunk_boundaries_and_overlap(self):
        chunker = TurnAwareChunker(window_size=3, overlap=1)
        turns = [
            {"role": "user", "content": "Turn 0: I love hiking in the mountains."},
            {"role": "assistant", "content": "Turn 1: The mountains are great for hiking."},
            {"role": "user", "content": "Turn 2: Especially in autumn when leaves turn red."},
            {"role": "assistant", "content": "Turn 3: Autumn scenery is stunning."},
            {"role": "user", "content": "Turn 4: I took my camera with me."},
            {"role": "assistant", "content": "Turn 5: Did you take landscape photos?"}
        ]

        chunks = chunker.chunk_session(
            session_id="session_test_01",
            turns=turns,
            timestamp="2023/10/15 (Sun) 14:00",
            parsed_datetime=datetime(2023, 10, 15, 14, 0)
        )

        # Total 6 turns with W=3, O=1 (step=2):
        # Chunk 0: turns 0, 1, 2
        # Chunk 1: turns 2, 3, 4 (overlap at turn 2)
        # Chunk 2: turns 4, 5 (overlap at turn 4)
        assert len(chunks) == 3

        assert chunks[0].turn_start_idx == 0
        assert chunks[0].turn_end_idx == 2
        assert "Turn 0" in chunks[0].text
        assert "Turn 1" in chunks[0].text
        assert "Turn 2" in chunks[0].text

        assert chunks[1].turn_start_idx == 2
        assert chunks[1].turn_end_idx == 4
        assert "Turn 2" in chunks[1].text
        assert "Turn 3" in chunks[1].text
        assert "Turn 4" in chunks[1].text

        assert chunks[2].turn_start_idx == 4
        assert chunks[2].turn_end_idx == 5
        assert "Turn 4" in chunks[2].text
        assert "Turn 5" in chunks[2].text

    def test_speaker_and_session_metadata_preservation(self):
        chunker = TurnAwareChunker(window_size=3, overlap=1)
        turns = [
            {"role": "user", "content": "Hello there."},
            {"role": "assistant", "content": "Hi! How can I help?"}
        ]
        chunks = chunker.chunk_session(
            session_id="sess_alpha_99",
            turns=turns,
            timestamp="2023/05/20 (Sat) 09:30"
        )

        assert len(chunks) == 1
        c = chunks[0]
        assert c.session_id == "alpha_99"
        assert c.memory_id == "sess_alpha_99"
        assert c.timestamp == "2023/05/20 (Sat) 09:30"
        assert "User: Hello there." in c.text
        assert "Assistant: Hi! How can I help?" in c.text

    def test_short_sessions_and_empty_turns(self):
        chunker = TurnAwareChunker(window_size=3, overlap=1)

        # Single-turn session
        chunks_1 = chunker.chunk_session("sess_single", [{"role": "user", "content": "One turn only."}])
        assert len(chunks_1) == 1
        assert chunks_1[0].turn_start_idx == 0
        assert chunks_1[0].turn_end_idx == 0

        # Session with empty turns
        chunks_empty = chunker.chunk_session("sess_empty", [])
        assert len(chunks_empty) == 0

    def test_stable_chunk_ids(self):
        chunker = TurnAwareChunker(window_size=3, overlap=1)
        turns = [{"role": "user", "content": f"Message {i}"} for i in range(7)]
        chunks = chunker.chunk_session("sess_fixed_id", turns)

        # 7 turns with window 3, step 2 -> 3 chunks covering 0-2, 2-4, 4-6
        expected_ids = ["sess_fixed_id_c000", "sess_fixed_id_c001", "sess_fixed_id_c002"]
        assert [c.chunk_id for c in chunks] == expected_ids


class TestChunkRetrieverAndContextAssembly:
    """Tests for hybrid chunk retrieval and bounded context assembly."""

    @pytest.fixture
    def sample_chunks(self):
        chunker = TurnAwareChunker(window_size=3, overlap=1)
        session_a = [
            {"role": "user", "content": "I adopted a golden retriever puppy named Barnaby."},
            {"role": "assistant", "content": "Barnaby is a wonderful name for a golden retriever!"},
            {"role": "user", "content": "He loves chewing on his squeaky tennis ball."}
        ]
        session_b = [
            {"role": "user", "content": "I am preparing for a marathon in Chicago."},
            {"role": "assistant", "content": "Marathon training requires great discipline."},
            {"role": "user", "content": "My target completion time is under four hours."}
        ]
        session_c = [
            {"role": "user", "content": "I redeemed a $5 coupon on coffee creamer last Sunday at Target."},
            {"role": "assistant", "content": "That is great! Target coupons save a lot."},
            {"role": "user", "content": "I used the Target Cartwheel app."}
        ]

        chunks = []
        chunks.extend(chunker.chunk_session("sess_dog", session_a, timestamp="2023/04/01"))
        chunks.extend(chunker.chunk_session("sess_run", session_b, timestamp="2023/04/10"))
        chunks.extend(chunker.chunk_session("sess_target", session_c, timestamp="2023/04/15"))
        return chunks

    def test_chunk_retrieval_returns_relevant_turn_chunks(self, sample_chunks):
        retriever = ChunkRetriever(sample_chunks)
        results = retriever.retrieve("What is the name of my puppy?", k=2)

        assert len(results) > 0
        top_hit = results[0]
        assert top_hit["memory_id"] == "sess_dog"
        assert "Barnaby" in top_hit["text"]

    def test_lexical_bm25_matches_exact_identifiers(self, sample_chunks):
        retriever = ChunkRetriever(sample_chunks)
        results = retriever.retrieve("Where did I redeem a $5 coupon on coffee creamer?", k=1)

        assert len(results) == 1
        top_hit_sess = results[0]["memory_id"]
        assert top_hit_sess == "sess_target"
        assert "Target" in results[0]["text"]
        assert "coupon" in results[0]["text"]

    def test_context_token_budget_enforcement(self, sample_chunks):
        retriever = ChunkRetriever(sample_chunks)
        results = retriever.retrieve("Target coupon and Chicago marathon", k=3)

        # Enforce strict small budget of 60 tokens
        context = assemble_bounded_context("Target coupon and Chicago marathon", results, max_context_tokens=60)
        assert context["has_sufficient_evidence"] is True
        assert context["context_token_count"] <= 80  # Header words + 1 chunk
        assert len(context["direct_evidence"]) >= 1

    def test_gold_label_isolation_in_chunks(self, sample_chunks):
        for c in sample_chunks:
            # Confirm no test leakage attributes exist
            assert not hasattr(c, "gold_answer")
            assert not hasattr(c, "has_answer")
            assert not hasattr(c, "answer_session_ids")


class TestConfigBRemainsUnchanged:
    """Verifies that Config B baseline execution path remains strictly untouched."""

    def test_config_b_default_behavior_preserved(self):
        memories = [
            {"id": "m1", "text": "I like dark roast coffee.", "timestamp": "2023-01-01"}
        ]
        # Calling answer_question with default arguments (use_chunking=False)
        result = answer_question(
            query="What coffee do I like?",
            k=1,
            memories=memories,
            llm_caller=lambda p, s: {"status": "answered", "answer": "Dark roast coffee.", "cited_memory_ids": ["m1"]}
        )

        assert result["status"] == "answered"
        assert "Dark roast" in result["answer"]
        assert len(result["citations"]) == 1
