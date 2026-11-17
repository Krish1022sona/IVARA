from .vocabulary import get_canonical_predicate, validate_predicate


def find_resolved_entity(name, resolved_entities):
    if not name or not isinstance(name, str) or not resolved_entities:
        return None

    name_clean = name.strip()
    name_lower = name_clean.lower()

    # 1. First-person pronoun / special SELF handling
    if name_lower in ("self", "i", "me", "my", "myself", "user"):
        for entity in resolved_entities:
            if entity.get("entity_id") == "self" or entity.get("canonical_name", "").upper() == "SELF":
                return entity

    # 2. Match by mention (case-insensitive)
    for entity in resolved_entities:
        if entity.get("mention", "").strip().lower() == name_lower:
            return entity

    # 3. Match by canonical_name (case-insensitive)
    for entity in resolved_entities:
        if entity.get("canonical_name", "").strip().lower() == name_lower:
            return entity

    # 4. Match by entity_id
    for entity in resolved_entities:
        ent_id = entity.get("entity_id")
        if ent_id and str(ent_id).strip().lower() == name_lower:
            return entity

    return None


def build_entity_reference(entity):
    if entity.get("status") == "matched" and entity.get("entity_id"):
        return {
            "type": "entity",
            "id": entity["entity_id"]
        }

    return {
        "type": "candidate",
        "mention": entity.get("mention") or entity.get("canonical_name", ""),
        "types": entity.get("types", [])
    }


def resolve_relationship(
    relationship,
    resolved_entities
):
    """
    Resolve relationship endpoints against resolved entities.
    Returns a resolved relationship dictionary, or a diagnostic record if endpoints cannot be resolved.
    """
    # Extract fields from dict or Pydantic model
    if hasattr(relationship, "source"):
        raw_source = relationship.source
        raw_relation = getattr(relationship, "relation", None) or getattr(relationship, "predicate", None)
        raw_target = relationship.target
        confidence = getattr(relationship, "confidence", 1.0)
        origin = getattr(relationship, "origin", "explicit")
        evidence = getattr(relationship, "evidence", None)
    elif isinstance(relationship, dict):
        raw_source = relationship.get("source")
        raw_relation = relationship.get("relation") or relationship.get("predicate")
        raw_target = relationship.get("target")
        confidence = relationship.get("confidence", 1.0)
        origin = relationship.get("origin", "explicit")
        evidence = relationship.get("evidence")
    else:
        return {
            "status": "unresolved",
            "resolved": False,
            "error": f"Invalid relationship input type: {type(relationship)}",
            "predicate": None
        }

    # Canonicalize predicate via vocabulary or open-world normalization
    pred_res = validate_predicate(raw_relation, allow_open_world=True) if raw_relation else None
    canonical_pred = pred_res.canonical_predicate if (pred_res and pred_res.is_valid) else raw_relation
    predicate_status = pred_res.status if pred_res else "unknown"

    source = find_resolved_entity(
        raw_source,
        resolved_entities
    )

    target = find_resolved_entity(
        raw_target,
        resolved_entities
    )

    # If any endpoint cannot be resolved, produce an explicit diagnostic record (never drop silently)
    if source is None or target is None:
        missing = []
        if source is None:
            missing.append(f"source '{raw_source}'")
        if target is None:
            missing.append(f"target '{raw_target}'")

        return {
            "status": "unresolved",
            "resolved": False,
            "error": f"Could not resolve {', '.join(missing)} to an entity in the current context.",
            "source_raw": raw_source,
            "predicate": canonical_pred,
            "raw_predicate": raw_relation,
            "predicate_status": predicate_status,
            "target_raw": raw_target,
            "confidence": confidence,
            "origin": origin,
            "evidence": evidence
        }

    return {
        "status": "resolved",
        "resolved": True,
        "source": build_entity_reference(source),
        "predicate": canonical_pred,
        "raw_predicate": raw_relation,
        "predicate_status": predicate_status,
        "target": build_entity_reference(target),
        "confidence": confidence,
        "origin": origin,
        "evidence": evidence
    }
