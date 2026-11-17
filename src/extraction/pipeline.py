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
    extraction=None,
    graph_store=None,
    llm_caller=None,
    model="gpt-oss"
):
    # Extract knowledge (or use provided extraction model)
    if extraction is None:
        extraction = extract_knowledge(text, model=model, llm_caller=llm_caller)

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

    # Extract temporal claims from text with provenance pointing to memory ID
    from ..temporal.claim import build_claims_from_text
    extracted_claims = build_claims_from_text(
        text=text,
        session_id=str(source or memory["id"]),
        chunk_id=memory["id"],
        recorded_at=timestamp
    )
    claims_data = [c.to_dict() for c in extracted_claims]
    memory["temporal_claims"] = claims_data

    # Update graph store if supplied
    if graph_store is not None:
        from ..graph.builder import MemoryGraphBuilder
        builder = MemoryGraphBuilder(graph_store)
        builder.build_from_records([memory], entities=load_entities(path=entity_store_path))

        if extracted_claims:
            from ..temporal.supersession import TemporalSupersessionEngine
            from ..graph.schema import TemporalClaim, GraphNode, GraphEdge
            engine = TemporalSupersessionEngine()

            existing_claims = []
            if hasattr(graph_store, "get_nodes_by_type"):
                for cn in graph_store.get_nodes_by_type("claim"):
                    p = cn.properties
                    existing_claims.append(TemporalClaim(
                        claim_id=cn.id,
                        subject=p.get("subject", ""),
                        predicate=p.get("predicate", ""),
                        object=p.get("object", ""),
                        valid_from=p.get("valid_from"),
                        valid_to=p.get("valid_to"),
                        recorded_at=p.get("recorded_at"),
                        source_chunk_id=p.get("source_chunk_id"),
                        is_explicit=p.get("is_explicit", True),
                        confidence=p.get("confidence", 1.0),
                        supersedes_claim_id=p.get("supersedes_claim_id"),
                        evidence=p.get("evidence"),
                        properties=dict(p)
                    ))

            claims_by_id = {c.claim_id: c for c in existing_claims}
            for c in extracted_claims:
                claims_by_id[c.claim_id] = c
            combined = list(claims_by_id.values())
            processed = engine.process_claims(combined)

            nodes = [c.to_graph_node() for c in processed]

            # Ensure memory container node exists in store to satisfy foreign keys
            if not graph_store.get_node(memory["id"]):
                nodes.append(GraphNode(
                    id=memory["id"],
                    node_type="memory",
                    label=f"Memory {memory['id']}",
                    properties={"text": text, "timestamp": timestamp, "source": source}
                ))

            edges = []
            for c in processed:
                src_node_id = c.source_chunk_id or memory["id"]
                edges.append(GraphEdge(
                    id=f"edge_claim_{src_node_id}_{c.claim_id}",
                    source_id=src_node_id,
                    target_id=c.claim_id,
                    edge_type="has_claim",
                    origin="explicit" if c.is_explicit else "inferred",
                    confidence=c.confidence,
                    timestamp=c.valid_from or c.recorded_at,
                    properties={
                        "valid_from": c.valid_from,
                        "valid_to": c.valid_to,
                        "recorded_at": c.recorded_at
                    }
                ))
                if c.supersedes_claim_id:
                    edges.append(GraphEdge(
                        id=f"edge_sup_{c.claim_id}_{c.supersedes_claim_id}",
                        source_id=c.claim_id,
                        target_id=c.supersedes_claim_id,
                        edge_type="supersedes",
                        origin="explicit" if c.is_explicit else "inferred",
                        confidence=c.confidence,
                        evidence=c.evidence,
                        timestamp=c.valid_from or c.recorded_at,
                        properties={
                            "valid_from": c.valid_from,
                            "valid_to": c.valid_to,
                            "recorded_at": c.recorded_at,
                            "reason": c.properties.get("supersession_reason")
                        }
                    ))
            if hasattr(graph_store, "batch_upsert"):
                graph_store.batch_upsert(nodes=nodes, edges=edges)
            elif hasattr(graph_store, "upsert_nodes_batch"):
                graph_store.upsert_nodes_batch(nodes)
                graph_store.upsert_edges_batch(edges)
            else:
                for n in nodes:
                    graph_store.upsert_node(n)
                for e in edges:
                    graph_store.upsert_edge(e)

    return memory
