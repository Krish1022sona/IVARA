from ..ontology.entity_resolver import (
    build_entity_embeddings,
    resolve_entity
)
from ..ontology.entity_store import load_entities

from ..ontology.relationship_resolver import (
    resolve_relationship
)

from datetime import datetime

from ..memory import add_memory
from .extractor import extract_knowledge

def process_extraction(extraction):

    entities = load_entities()

    embeddings = build_entity_embeddings(
        entities
    )

    resolved_entities = []

    for entity in extraction.entities:

        result = resolve_entity(
            name=entity.name,
            entity_type=entity.type,
            entities=entities,
            embeddings=embeddings
        )

        resolved_entities.append({
            "mention": entity.name,
            "entity_id": result["entity"].get("id"),
            "canonical_name": result["entity"]["canonical_name"],
            "types": result["entity"]["types"],
            "confidence": entity.confidence,
            "resolution_score": result["score"],
            "method": result["method"],
            "status": result["status"]
        })

    resolved_relationships = []

    for relationship in extraction.relationships:

        resolved = resolve_relationship(
            relationship,
            resolved_entities
        )

        if resolved is not None:
            resolved_relationships.append(
                resolved
            )

    return {
        "entities": resolved_entities,
        "relationships": resolved_relationships
    }

def process_and_store_memory(
    text,
    timestamp,
    source,
    memory_type,
    importance
):
    # Extract knowledge
    extraction = extract_knowledge(text)

    # Resolve entities + relationships
    knowledge = process_extraction(
        extraction
    )

    # Store structured memory
    memory = add_memory(
        text=text,
        timestamp=timestamp,
        source=source,
        type=memory_type,
        importance=importance,
        entities=knowledge["entities"],
        topics=[
            topic.model_dump()
            for topic in extraction.topics
        ],
        relationships=knowledge["relationships"]
    )

    return memory