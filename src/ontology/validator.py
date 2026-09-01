from .loader import (
    load_entity_types,
    load_relationships,
    load_rules
)


ENTITY_TYPES = load_entity_types()
RELATIONSHIPS = load_relationships()
RULES = load_rules()



def validate_entity(entity):
    required = ["name", "type"]

    if not all(field in entity for field in required):
        return False

    if not entity["name"]:
        return False

    if not entity["type"]:
        return False

    return True


def validate_relationship(relationship):
    required = [
        "source",
        "relation",
        "target"
    ]

    if not all(field in relationship for field in required):
        return False

    if not relationship["source"]:
        return False

    if not relationship["relation"]:
        return False

    if not relationship["target"]:
        return False

    return True

def validate_extraction(data):

    valid_entities = [
        entity
        for entity in data.get("entities", [])
        if validate_entity(entity)
    ]

    valid_relationships = [
        relation
        for relation in data.get("relationships", [])
        if validate_relationship(relation)
    ]

    return {
        "entities": valid_entities,
        "topics": data.get("topics", []),
        "relationships": valid_relationships
    }