from .loader import (
    load_entity_types,
    load_relationships,
    load_rules
)


ENTITY_TYPES = load_entity_types()
RELATIONSHIPS = load_relationships()
RULES = load_rules()


def is_valid_entity_type(entity_type):
    return entity_type in ENTITY_TYPES


def is_valid_relationship(relation):
    return relation in RELATIONSHIPS


def is_valid_relationship_types(
    relation,
    source_type,
    target_type
):
    if relation not in RULES:
        return False

    rule = RULES[relation]

    source_valid = (
        "*" in rule["source_types"]
        or source_type in rule["source_types"]
    )

    target_valid = (
        "*" in rule["target_types"]
        or target_type in rule["target_types"]
    )

    return source_valid and target_valid


def get_inverse_relationship(relation):
    if relation not in RELATIONSHIPS:
        return None

    return RELATIONSHIPS[relation]["inverse"]

def validate_entity(entity):
    """
    Validate one extracted entity.
    """

    if not is_valid_entity_type(entity["type"]):
        return False

    return True

def validate_relationship(relationship, entity_types):
    """
    Validate one extracted relationship.
    """

    relation = relationship["relation"]

    if not is_valid_relationship(relation):
        return False

    source = relationship["source"]
    target = relationship["target"]

    source_type = entity_types.get(source)
    target_type = entity_types.get(target)

    if source_type is None or target_type is None:
        return False

    return is_valid_relationship_types(
        relation,
        source_type,
        target_type
    )

def validate_extraction(data):
    valid_entities = []
    valid_relationships = []

    entity_types = {}

    # Validate entities
    for entity in data.get("entities", []):
        if validate_entity(entity):
            valid_entities.append(entity)
            entity_types[entity["name"]] = entity["type"]

    # Validate relationships
    for relationship in data.get("relationships", []):
        if validate_relationship(
            relationship,
            entity_types
        ):
            valid_relationships.append(relationship)

    return {
        "entities": valid_entities,
        "topics": data.get("topics", []),
        "relationships": valid_relationships
    }

