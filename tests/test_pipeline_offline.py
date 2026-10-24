import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.extraction.extractor import Extraction, Entity, Topic, Relationship
from src.extraction.pipeline import process_extraction, process_and_store_memory
from src.ontology.entity_store import load_entities, save_entities


class TestPipelineOffline(unittest.TestCase):
    """
    Automated offline integration tests for the extraction-to-memory-ingestion pipeline.
    Uses mocked extraction and isolated temporary data fixtures so local Ollama is not required
    and real production files in data/ are strictly untouched.
    """

    def setUp(self):
        # Create an isolated temporary directory for test storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.temp_memories_file = self.temp_path / "memories.json"
        self.temp_entities_file = self.temp_path / "entities.json"

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

    def tearDown(self):
        # Cleanup isolated temp storage
        self.temp_dir.cleanup()

        # Assert real data files were never modified during tests
        if self.real_memories_before is not None:
            self.assertEqual(
                self.real_memories_path.read_text(encoding="utf-8"),
                self.real_memories_before,
                "SAFETY VIOLATION: Real data/memories.json was modified during testing!"
            )
        if self.real_entities_before is not None:
            self.assertEqual(
                self.real_entities_path.read_text(encoding="utf-8"),
                self.real_entities_before,
                "SAFETY VIOLATION: Real data/entities.json was modified during testing!"
            )

    def test_offline_pipeline_successful_ingestion(self):
        """
        Test that valid entities, topics, and relationships flow from extraction
        through validation and resolution into isolated memory storage.
        """
        # Seed isolated entity store
        save_entities([
            {"id": "e001", "canonical_name": "SELF", "types": ["Person"], "aliases": []},
            {"id": "e002", "canonical_name": "IVARA", "types": ["Project"], "aliases": []}
        ], path=self.temp_entities_file)

        memory_text = "I designed and built IVARA for evidence-grounded memory."

        mock_extraction = Extraction(
            entities=[
                Entity(name="SELF", type="Person", confidence=1.0),
                Entity(name="IVARA", type="Project", confidence=1.0)
            ],
            topics=[
                Topic(name="AI Memory", confidence=0.9)
            ],
            relationships=[
                Relationship(
                    source="SELF",
                    relation="worked_on",
                    target="IVARA",
                    confidence=0.95,
                    origin="explicit",
                    evidence="built IVARA"
                )
            ]
        )

        memo = process_and_store_memory(
            text=memory_text,
            timestamp="2026-10-22T10:00:00Z",
            source="test_runner",
            memory_type="project_update",
            importance=0.9,
            memory_store_path=self.temp_memories_file,
            entity_store_path=self.temp_entities_file,
            extraction=mock_extraction
        )

        # Verify returned memory object
        self.assertEqual(memo["id"], "m001")
        self.assertEqual(memo["text"], memory_text)
        self.assertEqual(len(memo["entities"]), 2)
        self.assertEqual(len(memo["relationships"]), 1)

        rel = memo["relationships"][0]
        self.assertEqual(rel["predicate"], "worked_on")
        self.assertEqual(rel["source"]["id"], "self")
        self.assertEqual(rel["target"]["id"], "e002")
        self.assertEqual(rel["evidence"], "built IVARA")

        # Diagnostics observable and clean
        self.assertIn("diagnostics", memo)
        self.assertEqual(memo["diagnostics"]["extraction_errors"], [])
        self.assertEqual(memo["diagnostics"]["invalid_resolved_relationships"], [])
        self.assertEqual(memo["diagnostics"]["unresolved_relationships"], [])

        # Verify file persistence in temp path
        with open(self.temp_memories_file, "r", encoding="utf-8") as f:
            stored_memos = json.load(f)
        self.assertEqual(len(stored_memos), 1)
        self.assertEqual(stored_memos[0]["id"], "m001")

    def test_offline_pipeline_rejects_ungrounded_evidence_and_records_diagnostics(self):
        """
        Test that relationships with ungrounded evidence fail validation,
        are recorded in diagnostics, and are NOT persisted into memory relationships.
        """
        memory_text = "I built an LSTM model using PyTorch."

        mock_extraction = Extraction(
            entities=[
                Entity(name="SELF", type="Person", confidence=1.0),
                Entity(name="PyTorch", type="Technology", confidence=1.0)
            ],
            topics=[],
            relationships=[
                Relationship(
                    source="SELF",
                    relation="uses",
                    target="PyTorch",
                    confidence=1.0,
                    origin="explicit",
                    evidence="fabricated evidence that does not exist in text"
                )
            ]
        )

        memo = process_and_store_memory(
            text=memory_text,
            timestamp="2026-10-22T10:00:00Z",
            source="test_runner",
            memory_type="observation",
            importance=0.5,
            memory_store_path=self.temp_memories_file,
            entity_store_path=self.temp_entities_file,
            extraction=mock_extraction
        )

        # Invalid relationship MUST NOT be persisted
        self.assertEqual(memo["relationships"], [])

        # Failure remains fully observable to caller in diagnostics
        self.assertIn("diagnostics", memo)
        self.assertTrue(len(memo["diagnostics"]["extraction_errors"]) > 0)
        self.assertTrue(any("not grounded" in e for e in memo["diagnostics"]["extraction_errors"]))
        self.assertEqual(len(memo["diagnostics"]["invalid_extracted_relationships"]), 1)

    def test_offline_pipeline_rejects_unresolved_endpoints_and_records_diagnostics(self):
        """
        Test that relationships whose endpoints cannot be resolved produce
        observable diagnostics and do NOT create dangling references.
        """
        memory_text = "I met with Sarah to discuss ontology rules."

        mock_extraction = Extraction(
            entities=[
                Entity(name="SELF", type="Person", confidence=1.0)
                # Sarah is intentionally omitted from entities
            ],
            topics=[],
            relationships=[
                Relationship(
                    source="SELF",
                    relation="worked_with",
                    target="Sarah",
                    confidence=0.9,
                    origin="explicit",
                    evidence="met with Sarah"
                )
            ]
        )

        memo = process_and_store_memory(
            text=memory_text,
            timestamp="2026-10-22T10:00:00Z",
            source="test_runner",
            memory_type="meeting",
            importance=0.6,
            memory_store_path=self.temp_memories_file,
            entity_store_path=self.temp_entities_file,
            extraction=mock_extraction
        )

        # Relationship cannot create dangling reference
        self.assertEqual(memo["relationships"], [])

        # Unresolved relationship diagnostic is recorded
        unresolved = memo["diagnostics"]["unresolved_relationships"]
        self.assertEqual(len(unresolved), 1)
        self.assertFalse(unresolved[0]["resolved"])
        self.assertIn("Sarah", unresolved[0]["error"])

    def test_offline_pipeline_incompatible_type_safety_preserves_canonical_entities(self):
        """
        Test that an extracted entity with an incompatible type cannot merge with
        an existing canonical entity of the same name.
        """
        # Canonical entity in isolated store: Python as ProgrammingLanguage
        save_entities([
            {"id": "e001", "canonical_name": "Python", "types": ["ProgrammingLanguage"], "aliases": []}
        ], path=self.temp_entities_file)

        memory_text = "Python (a colleague) helped me review the codebase."

        mock_extraction = Extraction(
            entities=[
                Entity(name="Python", type="Person", confidence=0.9)
            ],
            topics=[],
            relationships=[]
        )

        knowledge = process_extraction(
            mock_extraction,
            text=memory_text,
            entity_store_path=self.temp_entities_file
        )

        resolved_ent = knowledge["entities"][0]
        # Must NOT merge into e001 because Person is incompatible with ProgrammingLanguage
        self.assertIsNone(resolved_ent["entity_id"])
        self.assertEqual(resolved_ent["status"], "candidate")

        # Canonical entity store remains intact
        stored_entities = load_entities(path=self.temp_entities_file)
        self.assertEqual(len(stored_entities), 1)
        self.assertEqual(stored_entities[0]["types"], ["ProgrammingLanguage"])

    def test_offline_pipeline_inferred_relationship_premise_evidence_flow(self):
        """
        Test that an inferred relationship citing grounded premise evidence
        is accepted with an informative diagnostic warning.
        """
        memory_text = "I built an LSTM using PyTorch."

        mock_extraction = Extraction(
            entities=[
                Entity(name="SELF", type="Person", confidence=1.0),
                Entity(name="LSTM", type="Technology", confidence=0.9)
            ],
            topics=[],
            relationships=[
                Relationship(
                    source="SELF",
                    relation="worked_on",
                    target="LSTM",
                    confidence=0.8,
                    origin="inferred",
                    evidence="built an LSTM"
                )
            ]
        )

        memo = process_and_store_memory(
            text=memory_text,
            timestamp="2026-10-22T10:00:00Z",
            source="test_runner",
            memory_type="development",
            importance=0.7,
            memory_store_path=self.temp_memories_file,
            entity_store_path=self.temp_entities_file,
            extraction=mock_extraction
        )

        self.assertEqual(len(memo["relationships"]), 1)
        self.assertEqual(memo["relationships"][0]["origin"], "inferred")
        self.assertEqual(memo["relationships"][0]["evidence"], "built an LSTM")

    def test_offline_pipeline_partial_corruption_protection(self):
        """
        Verify that a mixed extraction containing both valid and invalid items
        persists only the valid items, preserves diagnostics, and keeps the store
        completely uncorrupted.
        """
        memory_text = "I built IVARA using Python."

        mock_extraction = Extraction(
            entities=[
                Entity(name="SELF", type="Person", confidence=1.0),
                Entity(name="IVARA", type="Project", confidence=1.0),
                Entity(name="Python", type="Technology", confidence=1.0)
            ],
            topics=[],
            relationships=[
                # Valid relationship
                Relationship(
                    source="SELF",
                    relation="worked_on",
                    target="IVARA",
                    confidence=1.0,
                    origin="explicit",
                    evidence="built IVARA"
                ),
                # Invalid relationship: unknown predicate
                Relationship(
                    source="SELF",
                    relation="teleported_to",
                    target="Python",
                    confidence=1.0,
                    origin="explicit",
                    evidence="using Python"
                )
            ]
        )

        memo = process_and_store_memory(
            text=memory_text,
            timestamp="2026-10-22T10:00:00Z",
            source="test_runner",
            memory_type="mixed_ingestion",
            importance=0.8,
            memory_store_path=self.temp_memories_file,
            entity_store_path=self.temp_entities_file,
            extraction=mock_extraction
        )

        # Exactly 1 valid relationship is saved
        self.assertEqual(len(memo["relationships"]), 1)
        self.assertEqual(memo["relationships"][0]["predicate"], "worked_on")

        # The invalid one is caught and reported in diagnostics
        self.assertTrue(len(memo["diagnostics"]["extraction_errors"]) > 0)
        self.assertTrue(any("teleported_to" in e for e in memo["diagnostics"]["extraction_errors"]))

        # Verify physical file integrity
        with open(self.temp_memories_file, "r", encoding="utf-8") as f:
            stored_data = json.load(f)
        self.assertEqual(len(stored_data), 1)
        self.assertEqual(len(stored_data[0]["relationships"]), 1)


if __name__ == "__main__":
    unittest.main()
