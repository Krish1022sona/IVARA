from src.ontology.validator import (
    is_valid_entity_type,
    is_valid_relationship,
    is_valid_relationship_types,
    get_inverse_relationship
)


print(is_valid_entity_type("Person"))
print(is_valid_entity_type("Pokemon"))

print(is_valid_relationship("friend_of"))
print(is_valid_relationship("eats"))

print(
    is_valid_relationship_types(
        "friend_of",
        "Person",
        "Person"
    )
)

print(
    is_valid_relationship_types(
        "friend_of",
        "Technology",
        "Person"
    )
)

print(get_inverse_relationship("parent_of"))
print(get_inverse_relationship("friend_of"))