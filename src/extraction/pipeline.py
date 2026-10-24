from ..ontology.entity_resolver import (
    build_entity_embeddings,
    resolve_entity
)
from ..ontology.entity_store import load_entities
from ..ontology.relationship_resolver import (
    resolve_relationship
)
from ..ontology.validator import (
    validate_extraction,
    validate_resolved_relationship
)

from datetime import datetime

from ..memory import add_memory
from .extractor import extract_knowledge


def process_extraction(extraction, text=None, entity_store_path=None):
    """
    Process an extraction bundle: validate extraction, resolve entities against canonical store,
    resolve relationship endpoints, and validate resolved relationships.
    """
    diagnostics = {
        "extraction_errors": [],
        "extraction_warnings": [],
        "invalid_extracted_relationships": [],
        "unresolved_relationships": [],
        "invalid_resolved_relationships": []
    }

    # 1. Validation Boundary A: Validate raw extraction bundle
    val_summary = validate_extraction(extraction, memory_text=text)
    if not val_summary["is_valid"]:
        diagnostics["extraction_errors"].extend(val_summary["errors"])
    diagnostics["extraction_warnings"].extend(val_summary["warnings"])
    diagnostics["invalid_extracted_relationships"].extend(val_summary.get("invalid_relationships", []))

    # Load canonical entities from store
    entities = load_entities(path=entity_store_path)
    embeddings = build_entity_embeddings(entities)

    # 2. Resolve entities (use valid entities from validation stage)
    candidate_entities = val_summary.get("entities", [])
    resolved_entities = []

    for entity in candidate_entities:
        ent_name = entity.get("name") if isinstance(entity, dict) else entity.name
        ent_type = entity.get("type") if isinstance(entity, dict) else entity.type
        ent_conf = entity.get("confidence", 1.0) if isinstance(entity, dict) else getattr(entity, "confidence", 1.0)

        result = resolve_entity(
            name=ent_name,
            entity_type=ent_type,
            entities=entities,
            embeddings=embeddings
        )

        resolved_entities.append({
            "mention": ent_name,
            "entity_id": result["entity"].get("id"),
            "canonical_name": result["entity"]["canonical_name"],
            "types": result["entity"]["types"],
            "confidence": ent_conf,
            "resolution_score": result["score"],
            "method": result["method"],
            "status": result["status"]
        })

    # 3. Resolve relationships
    candidate_relationships = val_summary.get("relationships", [])
    resolved_relationships = []

    for relationship in candidate_relationships:
        resolved = resolve_relationship(
            relationship,
            resolved_entities
        )

        if resolved is not None:
            if resolved.get("status") == "resolved":
                # Validation Boundary B: Validate resolved relationship
                rel_val = validate_resolved_relationship(resolved, memory_text=text)
                if rel_val.is_valid:
                    # Clean internal status tag before persistence
                    clean_rel = {k: v for k, v in resolved.items() if k not in ("status", "resolved")}
                    resolved_relationships.append(clean_rel)
                else:
                    diagnostics["invalid_resolved_relationships"].append({
                        "relationship": resolved,
                        "errors": rel_val.errors
                    })
            else:
                # Unresolved relationship: record diagnostic instead of silent loss
                diagnostics["unresolved_relationships"].append(resolved)

    return {
        "entities": resolved_entities,
        "relationships": resolved_relationships,
        "diagnostics": diagnostics
    }


def process_and_store_memory(
    text,
    timestamp,
    source,
    memory_type,
    importance,
    memory_store_path=None,
    entity_store_path=None,
    extraction=None
):
    # Extract knowledge (or use provided extraction model)
    if extraction is None:
        extraction = extract_knowledge(text)

    # Resolve entities + relationships with validation boundaries
    knowledge = process_extraction(
        extraction,
        text=text,
        entity_store_path=entity_store_path
    )

    # Store structured memory
    topics_data = []
    if hasattr(extraction, "topics") and extraction.topics:
        for topic in extraction.topics:
            topics_data.append(topic.model_dump() if hasattr(topic, "model_dump") else topic)

    memory = add_memory(
        text=text,
        timestamp=timestamp,
        source=source,
        type=memory_type,
        importance=importance,
        entities=knowledge["entities"],
        topics=topics_data,
        relationships=knowledge["relationships"],
        diagnostics=knowledge.get("diagnostics"),
        data_path=memory_store_path
    )

    return memory
