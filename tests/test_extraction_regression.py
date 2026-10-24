"""
Regression tests verifying fixes for the extraction validation boundary,
relationship schema definitions, and predicate canonicalization.
"""

import pytest
from src.extraction.extractor import Extraction, Entity, Topic, Relationship
from src.ontology.validator import validate_extraction
from src.ontology.vocabulary import get_canonical_predicate, is_valid_predicate


def test_relationship_schema_origin_validation():
    """Verify Relationship schema enforces origin as 'explicit' or 'inferred'."""
    # Valid explicit relationship
    rel_exp = Relationship(
        source="SELF",
        relation="worked_on",
        target="Project Chronos",
        confidence=1.0,
        origin="explicit",
        evidence="I officially initiated Project Chronos"
    )
    assert rel_exp.origin == "explicit"
    assert rel_exp.evidence == "I officially initiated Project Chronos"

    # Valid inferred relationship
    rel_inf = Relationship(
        source="Priya Sharma",
        relation="worked_on",
        target="Project Nexus",
        confidence=0.9,
        origin="inferred",
        evidence="joined Project Nexus as Lead Neuro-Symbolic Model Engineer"
    )
    assert rel_inf.origin == "inferred"

    # Invalid origin string must raise ValidationError
    with pytest.raises(Exception):
        Relationship(
            source="Priya Sharma",
            relation="worked_on",
            target="Project Chronos",
            confidence=0.95,
            origin="Priya was assigned to Chronos",  # Evidence text incorrectly placed in origin
            evidence=None
        )


def test_new_predicate_aliases_canonicalization():
    """Verify common predicate variants canonicalize to ontology predicates."""
    assert get_canonical_predicate("collaborated_with") == "worked_with"
    assert get_canonical_predicate("collaborated with") == "worked_with"
    assert get_canonical_predicate("assigned_to") == "worked_on"
    assert get_canonical_predicate("assigned to") == "worked_on"
    assert get_canonical_predicate("co-founded") == "worked_on"
    assert get_canonical_predicate("spearheaded") == "worked_on"
    assert get_canonical_predicate("uses_framework") == "uses"
    assert get_canonical_predicate("uses_language") == "uses"

    assert is_valid_predicate("collaborated_with") is True
    assert is_valid_predicate("assigned_to") is True
    assert is_valid_predicate("uses_framework") is True


def test_validation_boundary_with_compliant_relationships():
    """Verify validate_extraction accepts relationships with valid origin and evidence."""
    memory_text = "I initiated Project Chronos at Aetheria Labs using Rust and PyTorch. Priya Sharma collaborated with me."
    extraction = Extraction(
        entities=[
            Entity(name="SELF", type="Person", confidence=1.0),
            Entity(name="Project Chronos", type="Project", confidence=1.0),
            Entity(name="Aetheria Labs", type="Organization", confidence=1.0),
            Entity(name="Priya Sharma", type="Person", confidence=1.0),
            Entity(name="Rust", type="Technology", confidence=1.0)
        ],
        topics=[
            Topic(name="Project Initiation", confidence=1.0)
        ],
        relationships=[
            Relationship(
                source="SELF",
                relation="worked_on",
                target="Project Chronos",
                confidence=1.0,
                origin="explicit",
                evidence="I initiated Project Chronos at Aetheria Labs"
            ),
            Relationship(
                source="Priya Sharma",
                relation="collaborated_with",
                target="SELF",
                confidence=0.95,
                origin="explicit",
                evidence="Priya Sharma collaborated with me"
            ),
            Relationship(
                source="Project Chronos",
                relation="uses_language",
                target="Rust",
                confidence=1.0,
                origin="explicit",
                evidence="using Rust and PyTorch"
            )
        ]
    )

    result = validate_extraction(extraction, memory_text=memory_text)
    assert result["is_valid"] is True
    assert len(result["errors"]) == 0
    assert len(result.get("invalid_relationships", [])) == 0
    assert len(result["relationships"]) == 3
