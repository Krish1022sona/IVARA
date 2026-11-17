"""Targeted tests for Milestone 2: Open-World Ontology Validation and Relationship Persistence.

Verifies:
1. Existing software predicates.
2. Biographical predicates.
3. Preference predicates.
4. Location relationships.
5. Employment and education relationships.
6. Relationships involving third-party entities.
7. Unknown but structurally valid predicates (open-world).
8. Malformed / noisy / stop-word predicates rejected.
9. Missing subjects or objects rejected.
10. Self-referential relationships rejected.
11. Entity resolution failures handled gracefully.
12. Relationship resolution failures handled gracefully.
13. Graph persistence and retrieval of open-world relationships.
14. Provenance preservation (raw predicate, status, evidence, memory ID).
"""

from pathlib import Path
import pytest
import sqlite3
import tempfile

from src.ontology.vocabulary import (
    load_vocabulary,
    validate_predicate,
    normalize_predicate_name,
    validate_predicate_structure,
    get_canonical_predicate,
    is_valid_predicate
)
from src.ontology.validator import (
    validate_extracted_relationship,
    validate_resolved_relationship,
    validate_extraction
)
from src.ontology.relationship_resolver import resolve_relationship
from src.extraction.pipeline import process_extraction, process_and_store_memory
from src.graph.store import SQLiteGraphStore
from src.graph.builder import MemoryGraphBuilder


def test_01_existing_software_predicates():
    """Existing core predicates and aliases resolve to canonical vocabulary entries."""
    for raw in ["worked_on", "built", "uses", "utilizes", "worked_with", "collaborated with", "employed_by", "works at"]:
        res = validate_predicate(raw, allow_open_world=True)
        assert res.is_valid is True
        assert res.status == "canonical"
        assert res.canonical_predicate in ("worked_on", "uses", "worked_with", "employed_by")


def test_02_biographical_predicates():
    """Biographical predicates validate with proper canonicalization and structure."""
    bio_cases = [
        ("born_in", "born_in"),
        ("born on", "born_on"),
        ("lives_in", "lives_in"),
        ("roommate_of", "roommate_of"),
        ("has_sibling", "has_sibling"),
        ("has_pet", "has_pet")
    ]
    for raw, expected in bio_cases:
        res = validate_predicate(raw, allow_open_world=True)
        assert res.is_valid is True
        assert res.canonical_predicate == expected
        assert res.status in ("canonical", "open_world")


def test_03_preference_predicates():
    """Preference and behavioral predicates validate and retain evidence."""
    pref_cases = ["prefers", "prefers_over", "dislikes", "transitioned_to", "transitioned_from"]
    for p in pref_cases:
        res = validate_predicate(p, allow_open_world=True)
        assert res.is_valid is True
        assert res.canonical_predicate in (p, "prefers")

    # Verify in relationship validation
    text = "I transitioned to a 100% vegan diet in September 2027."
    rel = {
        "source": "SELF",
        "relation": "transitioned_to",
        "target": "vegan diet",
        "confidence": 0.95,
        "origin": "explicit",
        "evidence": "transitioned to a 100% vegan diet"
    }
    val_res = validate_extracted_relationship(rel, memory_text=text)
    assert val_res.is_valid is True


def test_04_location_relationships():
    """Location and movement predicates validate properly."""
    loc_cases = ["located_in", "located_at", "moved_to", "relocates_to", "resides_in"]
    for p in loc_cases:
        res = validate_predicate(p, allow_open_world=True)
        assert res.is_valid is True
        assert res.canonical_predicate is not None

    text = "In August 2026, Arjun relocated to Zurich for his research role."
    rel = {
        "source": "Arjun",
        "relation": "relocates_to",
        "target": "Zurich",
        "confidence": 0.95,
        "origin": "explicit",
        "evidence": "relocated to Zurich"
    }
    val_res = validate_extracted_relationship(rel, memory_text=text)
    assert val_res.is_valid is True


def test_05_employment_and_education_relationships():
    """Employment and academic trajectory predicates validate."""
    cases = ["graduated_from", "studied_at", "mentoring", "mentored_by", "holds_position", "served_as"]
    for p in cases:
        res = validate_predicate(p, allow_open_world=True)
        assert res.is_valid is True


def test_06_relationships_involving_third_party_entities():
    """Relationships where neither endpoint is SELF validate correctly."""
    text = "Rohan Sharma was Kaalu's primary veterinarian in Bangalore."
    rel = {
        "source": "Rohan Sharma",
        "relation": "veterinarian_of",
        "target": "Kaalu",
        "confidence": 0.9,
        "origin": "explicit",
        "evidence": "primary veterinarian"
    }
    val_res = validate_extracted_relationship(rel, memory_text=text, rules={"open_world_predicates": True})
    assert val_res.is_valid is True
    # 'veterinarian_of' is open_world
    assert any("open-world" in w.lower() for w in val_res.warnings)


def test_07_unknown_but_structurally_valid_predicates():
    """Novel predicates outside any dictionary validate as open-world relationships."""
    novel_predicates = [
        "co_authored_paper_with",
        "allergic_to",
        "won_championship_at",
        "speaks_fluently",
        "invested_in",
        "nominated_for"
    ]
    for p in novel_predicates:
        res = validate_predicate(p, allow_open_world=True)
        assert res.is_valid is True
        assert res.status == "open_world"
        assert res.canonical_predicate == p

    # When open_world is disabled, must be rejected
    res_disabled = validate_predicate("co_authored_paper_with", allow_open_world=False)
    assert res_disabled.is_valid is False
    assert res_disabled.status == "invalid"


def test_08_malformed_predicates_rejected():
    """Malformed, stop-word, metadata, or injection predicates are strictly rejected."""
    bad_predicates = [
        "",
        "   ",
        "at",           # stop-word
        "in",           # stop-word
        "on",           # stop-word
        "to",           # stop-word
        "with",         # stop-word
        "of",           # stop-word
        "type",         # metadata token
        "status",       # metadata token
        "null",         # metadata token
        "12345",        # numeric only
        "DROP TABLE",   # SQL / special characters
        "<script>",     # HTML injection
        "a",            # single character
        "!@#$%",        # punctuation
    ]
    for bad in bad_predicates:
        res = validate_predicate(bad, allow_open_world=True)
        assert res.is_valid is False
        assert res.status == "invalid"
        assert res.error is not None

        # Verify validator rejects it
        rel = {
            "source": "SELF",
            "relation": bad,
            "target": "Zurich",
            "confidence": 1.0,
            "origin": "explicit",
            "evidence": "Zurich"
        }
        val_res = validate_extracted_relationship(rel, memory_text="I was in Zurich.")
        assert val_res.is_valid is False


def test_09_missing_subjects_or_objects_rejected():
    """Relationships with missing source or target fail extraction validation."""
    # Missing source
    rel_no_src = {
        "source": "",
        "relation": "born_in",
        "target": "Chandigarh",
        "confidence": 1.0,
        "origin": "explicit",
        "evidence": "born in Chandigarh"
    }
    assert validate_extracted_relationship(rel_no_src, memory_text="I was born in Chandigarh.").is_valid is False

    # Missing target
    rel_no_tgt = {
        "source": "SELF",
        "relation": "born_in",
        "target": "  ",
        "confidence": 1.0,
        "origin": "explicit",
        "evidence": "born in Chandigarh"
    }
    assert validate_extracted_relationship(rel_no_tgt, memory_text="I was born in Chandigarh.").is_valid is False


def test_10_self_referential_relationship_rejected():
    """Relationships where source equals target are rejected as invalid self-loops."""
    rel_self_loop = {
        "source": "PyTorch",
        "relation": "uses",
        "target": "pytorch",
        "confidence": 1.0,
        "origin": "explicit",
        "evidence": "PyTorch uses pytorch"
    }
    res = validate_extracted_relationship(rel_self_loop, memory_text="PyTorch uses pytorch")
    assert res.is_valid is False
    assert any("identical" in e.lower() for e in res.errors)

    # Resolved stage self-loop
    res_rel = {
        "source": {"type": "entity", "id": "e_pytorch"},
        "predicate": "uses",
        "target": {"type": "entity", "id": "e_pytorch"},
        "confidence": 1.0,
        "origin": "explicit",
        "evidence": "PyTorch uses PyTorch"
    }
    assert validate_resolved_relationship(res_rel, memory_text="PyTorch uses PyTorch").is_valid is False


def test_11_entity_resolution_failure_handling():
    """When an entity cannot be resolved, relationship resolver records diagnostic without silent drops."""
    resolved_entities = [
        {"mention": "Arjun", "entity_id": "ent_arjun", "canonical_name": "Arjun", "types": ["Person"], "status": "matched"}
    ]
    # Kaalu is not in resolved_entities
    rel = {
        "source": "Arjun",
        "relation": "has_pet",
        "target": "Kaalu",
        "confidence": 0.95,
        "origin": "explicit",
        "evidence": "Arjun adopted Kaalu"
    }
    res = resolve_relationship(rel, resolved_entities)
    assert res["status"] == "unresolved"
    assert res["resolved"] is False
    assert "Kaalu" in res["error"]
    assert res["raw_predicate"] == "has_pet"


def test_12_relationship_resolution_success_and_provenance():
    """Resolved relationship preserves raw_predicate, canonical predicate, and status."""
    resolved_entities = [
        {"mention": "Arjun", "entity_id": "ent_arjun", "canonical_name": "Arjun", "types": ["Person"], "status": "matched"},
        {"mention": "Zurich", "entity_id": None, "canonical_name": "Zurich", "types": ["Location"], "status": "candidate"}
    ]
    rel = {
        "source": "Arjun",
        "relation": "relocates_to",
        "target": "Zurich",
        "confidence": 0.95,
        "origin": "explicit",
        "evidence": "relocated to Zurich"
    }
    res = resolve_relationship(rel, resolved_entities)
    assert res["status"] == "resolved"
    assert res["resolved"] is True
    assert res["predicate"] in ("lives_in", "relocates_to")
    assert res["raw_predicate"] == "relocates_to"
    assert res["predicate_status"] in ("canonical", "open_world")
    assert res["source"]["type"] == "entity"
    assert res["target"]["type"] == "candidate"


def test_13_graph_persistence_and_retrieval(tmp_path):
    """End-to-end test: Memory with open-world relationships is stored and retrievable in SQLiteGraphStore."""
    db_file = tmp_path / "test_graph.db"
    store = SQLiteGraphStore(db_file)
    builder = MemoryGraphBuilder(store)

    memory_record = {
        "id": "m_test_001",
        "text": "Arjun adopted a street dog named Kaalu in Bangalore.",
        "timestamp": "2024-06-15",
        "source": "conv_001_turn_001",
        "type": "episodic",
        "importance": 0.85,
        "relationships": [
            {
                "source": {"type": "entity", "id": "ent_arjun", "canonical_name": "Arjun"},
                "predicate": "has_pet",
                "raw_predicate": "has pet",
                "predicate_status": "open_world",
                "target": {"type": "candidate", "mention": "Kaalu", "canonical_name": "Kaalu"},
                "confidence": 0.95,
                "origin": "explicit",
                "evidence": "adopted a street dog named Kaalu"
            },
            {
                "source": {"type": "candidate", "mention": "Kaalu", "canonical_name": "Kaalu"},
                "predicate": "adopted_in",
                "raw_predicate": "adopted in",
                "predicate_status": "open_world",
                "target": {"type": "candidate", "mention": "Bangalore", "canonical_name": "Bangalore"},
                "confidence": 0.90,
                "origin": "explicit",
                "evidence": "in Bangalore"
            }
        ]
    }

    stats = builder.build_from_records([memory_record])
    assert stats["nodes_count"] >= 3  # Memory container, chunks, entities
    assert stats["edges_count"] >= 2

    # Query edges from graph store
    pet_edges = store.get_edges_by_type("has_pet")
    assert len(pet_edges) == 1
    edge = pet_edges[0]
    assert edge.source_id == "ent_arjun"
    assert "kaalu" in edge.target_id.lower()
    assert edge.memory_id == "m_test_001"
    assert edge.evidence == "adopted a street dog named Kaalu"
    assert edge.confidence == 0.95

    # Query outgoing edges from Arjun
    out_edges = store.get_outgoing_edges("ent_arjun")
    out_types = [e.edge_type for e in out_edges]
    assert "has_pet" in out_types

    # Query adopted_in edges
    adopted_edges = store.get_edges_by_type("adopted_in")
    assert len(adopted_edges) == 1
    assert "kaalu" in adopted_edges[0].source_id.lower()
    assert "bangalore" in adopted_edges[0].target_id.lower()
    assert edge.source_id == "ent_arjun"
    assert "kaalu" in edge.target_id.lower()
    assert edge.memory_id == "m_test_001"
    assert edge.evidence == "adopted a street dog named Kaalu"
    assert edge.confidence == 0.95
    store.close()


def test_14_pipeline_open_world_end_to_end(tmp_path):
    """Full extraction pipeline validation preserves open-world relationship and diagnostics."""
    mem_path = tmp_path / "mems.json"
    ent_path = tmp_path / "ents.json"
    mem_path.write_text("[]", encoding="utf-8")
    ent_path.write_text("{}", encoding="utf-8")

    mock_extraction = {
        "entities": [
            {"name": "Arjun", "type": "Person", "confidence": 1.0},
            {"name": "Chandigarh", "type": "Location", "confidence": 1.0}
        ],
        "relationships": [
            {
                "source": "Arjun",
                "relation": "born_in",
                "target": "Chandigarh",
                "confidence": 0.98,
                "origin": "explicit",
                "evidence": "born in Chandigarh"
            },
            {
                "source": "Arjun",
                "relation": "at",  # Malformed stop-word
                "target": "Chandigarh",
                "confidence": 0.98,
                "origin": "explicit",
                "evidence": "born in Chandigarh"
            }
        ],
        "topics": []
    }

    res = process_extraction(
        mock_extraction,
        text="Arjun was born in Chandigarh.",
        entity_store_path=ent_path
    )

    # born_in must be accepted into resolved relationships
    assert len(res["relationships"]) == 1
    rel = res["relationships"][0]
    assert rel["predicate"] == "born_in"
    assert rel["raw_predicate"] == "born_in"
    assert rel["predicate_status"] in ("canonical", "open_world")

    # 'at' must be rejected and logged into diagnostics
    invalid_extracted = res["diagnostics"]["invalid_extracted_relationships"]
    assert len(invalid_extracted) == 1
    bad_rel, errors = invalid_extracted[0]
    assert bad_rel["relation"] == "at"
    assert any("stop-word" in e.lower() or "preposition" in e.lower() for e in errors)
