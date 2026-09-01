def resolve_relationship(
    relationship,
    resolved_entities
):

    source = find_resolved_entity(
        relationship.source,
        resolved_entities
    )

    target = find_resolved_entity(
        relationship.target,
        resolved_entities
    )

    if source is None or target is None:
        return None

    return {
        "source": build_entity_reference(source),
        "predicate": relationship.relation,
        "target": build_entity_reference(target),
        "confidence": relationship.confidence,
        "origin": relationship.origin
    }


def build_entity_reference(entity):

    if entity["status"] == "matched":
        return {
            "type": "entity",
            "id": entity["entity_id"]
        }

    return {
        "type": "candidate",
        "mention": entity["mention"],
        "types": entity["types"]
    }


def find_resolved_entity(
    name,
    resolved_entities
):

    for entity in resolved_entities:

        if entity["mention"].lower() == name.lower():
            return entity

    return None