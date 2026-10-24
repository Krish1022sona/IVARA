import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval import (
    DEFAULT_RELEVANCE_THRESHOLD,
    build_index,
    load_memories,
    prepare_evidence_context,
    retrieve,
    retrieve_with_context,
)


class TestEvidenceGroundedRetrieval(unittest.TestCase):
    """
    Focused automated unit and integration tests for evidence-grounded memory retrieval.
    Verifies relevance filtering, provenance preservation, empty/malformed safety,
    and evidence context preparation without requiring live Ollama.
    """

    def setUp(self):
        # Create an isolated temporary directory for test fixtures
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.temp_memories_path = self.temp_path / "memories.json"

        # Record real data files state to guarantee zero mutation
        self.real_data_dir = Path(__file__).parent.parent / "data"
        self.real_memories_path = self.real_data_dir / "memories.json"
        self.real_entities_path = self.real_data_dir / "entities.json"

        self.real_memories_before = (
            self.real_memories_path.read_text(encoding="utf-8")
            if self.real_memories_path.exists()
            else None
        )
        self.real_entities_before = (
            self.real_entities_path.read_text(encoding="utf-8")
            if self.real_entities_path.exists()
            else None
        )

        # Seed isolated test memories
        self.test_memories = [
            {
                "id": "m101",
                "text": "I trained an LSTM neural network using PyTorch for speech recognition.",
                "timestamp": "2026-03-15",
                "source": "lab_notes",
                "type": "project_log",
                "importance": 0.9,
                "confidence": 1.0,
                "entities": [
                    {"canonical_name": "SELF", "types": ["Person"], "entity_id": "self"},
                    {"canonical_name": "PyTorch", "types": ["Framework"], "entity_id": "e001"},
                    {"canonical_name": "LSTM", "types": ["Concept"], "entity_id": "e002"}
                ],
                "topics": [{"name": "Deep Learning", "confidence": 0.95}],
                "relationships": [
                    {
                        "source": {"type": "entity", "id": "self"},
                        "predicate": "uses",
                        "target": {"type": "entity", "id": "e001"},
                        "origin": "explicit",
                        "evidence": "using PyTorch",
                        "confidence": 1.0
                    },
                    {
                        "source": {"type": "entity", "id": "self"},
                        "predicate": "worked_on",
                        "target": {"type": "entity", "id": "e002"},
                        "origin": "inferred",
                        "evidence": "trained an LSTM neural network",
                        "confidence": 0.85
                    }
                ]
            },
            {
                "id": "m102",
                "text": "I collaborated with Rahul on our smart attendance project.",
                "timestamp": "2026-04-10",
                "source": "hackathon",
                "type": "collaboration",
                "importance": 0.8,
                "confidence": 1.0,
                "entities": [
                    {"canonical_name": "SELF", "types": ["Person"], "entity_id": "self"},
                    {"canonical_name": "Rahul", "types": ["Person"], "entity_id": "e003"}
                ],
                "topics": [{"name": "Attendance System", "confidence": 0.85}],
                "relationships": [
                    {
                        "source": {"type": "entity", "id": "self"},
                        "predicate": "worked_with",
                        "target": {"type": "entity", "id": "e003"},
                        "origin": "explicit",
                        "evidence": "collaborated with Rahul",
                        "confidence": 0.95
                    }
                ]
            }
        ]

        with open(self.temp_memories_path, "w", encoding="utf-8") as f:
            json.dump(self.test_memories, f, indent=2)

    def tearDown(self):
        self.temp_dir.cleanup()

        # Enforce that real user data was untouched
        if self.real_memories_before is not None:
            self.assertEqual(
                self.real_memories_path.read_text(encoding="utf-8"),
                self.real_memories_before,
                "SAFETY VIOLATION: Real data/memories.json was altered during testing!"
            )
        if self.real_entities_before is not None:
            self.assertEqual(
                self.real_entities_path.read_text(encoding="utf-8"),
                self.real_entities_before,
                "SAFETY VIOLATION: Real data/entities.json was altered during testing!"
            )

    # -----------------------------------------------------------------------
    # 1. Relevance Filtering & Thresholds
    # -----------------------------------------------------------------------

    def test_relevant_memories_returned_with_scores(self):
        """
        Query relevant to PyTorch/LSTM returns m101 with an attached score >= threshold.
        """
        results = retrieve(
            "What deep learning models did I build with PyTorch?",
            k=3,
            min_score=DEFAULT_RELEVANCE_THRESHOLD,
            memories=self.test_memories
        )

        self.assertGreaterEqual(len(results), 1)
        top_match = results[0]
        self.assertEqual(top_match["id"], "m101")
        self.assertIn("score", top_match)
        self.assertGreaterEqual(top_match["score"], DEFAULT_RELEVANCE_THRESHOLD)

    def test_irrelevant_memories_filtered_below_threshold(self):
        """
        Query completely unrelated to stored memories (e.g., video games)
        filters out all memories and returns an empty list.
        """
        results = retrieve(
            "What video games do I play on PlayStation?",
            k=3,
            min_score=DEFAULT_RELEVANCE_THRESHOLD,
            memories=self.test_memories
        )

        self.assertEqual(results, [])

    def test_return_fewer_than_k_when_fewer_qualify(self):
        """
        When k=3 but only 1 memory is topically relevant, return only 1 result.
        """
        results = retrieve(
            "Who is Rahul?",
            k=3,
            min_score=DEFAULT_RELEVANCE_THRESHOLD,
            memories=self.test_memories
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "m102")

    def test_configured_threshold_calibration(self):
        """
        Adjusting min_score calibrates precision:
        - A strict threshold (0.80) rejects moderate matches.
        - A permissive threshold (0.15) admits them.
        """
        query = "Who did I work with on projects?"

        # Moderate match
        standard = retrieve(query, k=3, min_score=0.30, memories=self.test_memories)
        self.assertGreaterEqual(len(standard), 1)

        # Ultra-strict threshold rejects matches below 0.90
        strict = retrieve(query, k=3, min_score=0.90, memories=self.test_memories)
        self.assertEqual(strict, [])

    def test_filtering_disabled_when_min_score_is_none(self):
        """
        Setting min_score=None preserves raw top-k FAISS behavior without filtering.
        """
        results = retrieve(
            "What video games do I play?",
            k=2,
            min_score=None,
            memories=self.test_memories
        )

        # Unconditionally returns up to k items
        self.assertEqual(len(results), 2)
        # Even with negative or near-zero similarity
        self.assertIn("score", results[0])

    # -----------------------------------------------------------------------
    # 2. Provenance Preservation
    # -----------------------------------------------------------------------

    def test_provenance_metadata_preserved_completely(self):
        """
        Retrieved memories must preserve all original fields without alteration.
        """
        results = retrieve(
            "Tell me about PyTorch",
            k=1,
            min_score=0.30,
            memories=self.test_memories
        )

        self.assertEqual(len(results), 1)
        memo = results[0]

        # Key provenance attributes
        self.assertEqual(memo["id"], "m101")
        self.assertEqual(memo["text"], "I trained an LSTM neural network using PyTorch for speech recognition.")
        self.assertEqual(memo["timestamp"], "2026-03-15")
        self.assertEqual(memo["source"], "lab_notes")
        self.assertEqual(memo["type"], "project_log")
        self.assertEqual(memo["importance"], 0.9)
        self.assertEqual(memo["confidence"], 1.0)

        # Entities preserved
        self.assertEqual(len(memo["entities"]), 3)
        self.assertEqual(memo["entities"][1]["canonical_name"], "PyTorch")

        # Relationships preserved with origin and evidence
        self.assertEqual(len(memo["relationships"]), 2)
        rel_explicit = memo["relationships"][0]
        self.assertEqual(rel_explicit["predicate"], "uses")
        self.assertEqual(rel_explicit["origin"], "explicit")
        self.assertEqual(rel_explicit["evidence"], "using PyTorch")

        rel_inferred = memo["relationships"][1]
        self.assertEqual(rel_inferred["predicate"], "worked_on")
        self.assertEqual(rel_inferred["origin"], "inferred")
        self.assertEqual(rel_inferred["evidence"], "trained an LSTM neural network")

    # -----------------------------------------------------------------------
    # 3. Edge Cases & Robustness
    # -----------------------------------------------------------------------

    def test_empty_and_whitespace_query_returns_empty(self):
        self.assertEqual(retrieve("", memories=self.test_memories), [])
        self.assertEqual(retrieve("   \t\n  ", memories=self.test_memories), [])
        self.assertEqual(retrieve(None, memories=self.test_memories), [])

    def test_empty_memory_store_returns_empty(self):
        self.assertEqual(retrieve("PyTorch", memories=[]), [])

    def test_nonexistent_data_path_returns_empty(self):
        fake_path = self.temp_path / "does_not_exist.json"
        self.assertEqual(retrieve("PyTorch", data_path=fake_path), [])

    def test_malformed_json_in_data_path_returns_empty(self):
        bad_file = self.temp_path / "bad.json"
        bad_file.write_text("{ corrupt json ", encoding="utf-8")
        self.assertEqual(retrieve("PyTorch", data_path=bad_file), [])

    def test_build_index_with_empty_or_invalid_memories(self):
        self.assertIsNone(build_index([]))
        self.assertIsNone(build_index(None))

    # -----------------------------------------------------------------------
    # 4. Evidence-Aware Answer Preparation
    # -----------------------------------------------------------------------

    def test_evidence_context_sufficient_evidence(self):
        """
        When relevant memories exist, prepare_evidence_context produces
        structured direct evidence, inferred evidence, and sufficient status.
        """
        ctx = retrieve_with_context(
            "What deep learning models did I build with PyTorch?",
            k=2,
            min_score=DEFAULT_RELEVANCE_THRESHOLD,
            memories=self.test_memories
        )

        self.assertTrue(ctx["has_sufficient_evidence"])
        self.assertEqual(ctx["status"], "sufficient_evidence")
        self.assertEqual(ctx["retrieved_memories_count"], 1)

        # Direct evidence contains raw text and explicit facts
        self.assertEqual(len(ctx["direct_evidence"]), 1)
        direct_entry = ctx["direct_evidence"][0]
        self.assertEqual(direct_entry["memory_id"], "m101")
        self.assertIn("trained an LSTM", direct_entry["text"])

        # Direct facts contain explicit relationship with verbatim evidence
        self.assertEqual(len(direct_entry["direct_facts"]), 1)
        self.assertEqual(direct_entry["direct_facts"][0]["predicate"], "uses")
        self.assertEqual(direct_entry["direct_facts"][0]["evidence"], "using PyTorch")
        self.assertEqual(direct_entry["direct_facts"][0]["origin"], "explicit")

        # Inferred evidence is cleanly separated with premise citation
        self.assertEqual(len(ctx["inferred_evidence"]), 1)
        inf_entry = ctx["inferred_evidence"][0]
        self.assertEqual(inf_entry["predicate"], "worked_on")
        self.assertEqual(inf_entry["origin"], "inferred")
        self.assertEqual(inf_entry["premise_evidence"], "trained an LSTM neural network")

        # Formatted context text includes sections
        self.assertIn("DIRECTLY SUPPORTED EVIDENCE", ctx["context_text"])
        self.assertIn("INFERRED RELATIONSHIPS (DERIVED)", ctx["context_text"])

    def test_evidence_context_insufficient_evidence(self):
        """
        When no memories qualify, prepare_evidence_context marks status
        as insufficient without inventing facts.
        """
        ctx = retrieve_with_context(
            "What video games do I play?",
            k=2,
            min_score=DEFAULT_RELEVANCE_THRESHOLD,
            memories=self.test_memories
        )

        self.assertFalse(ctx["has_sufficient_evidence"])
        self.assertEqual(ctx["status"], "insufficient_evidence")
        self.assertEqual(ctx["retrieved_memories_count"], 0)
        self.assertEqual(ctx["direct_evidence"], [])
        self.assertEqual(ctx["inferred_evidence"], [])
        self.assertIn("Insufficient evidence", ctx["message"])
        self.assertIn("Insufficient evidence", ctx["context_text"])


if __name__ == "__main__":
    unittest.main()
