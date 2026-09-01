import json

from src.ontology.entity_resolver import (
    build_entity_embeddings,
    find_best_match
)


with open("../data/entities.json", "r") as f:
    entities = json.load(f)


embeddings = build_entity_embeddings(entities)


test_names = [
    "torch",
    "pytorch framework",
    "python programming",
    "Tata Motors company",
    "Flutter"
]


for name in test_names:

    entity, score = find_best_match(
        name,
        entities,
        embeddings
    )

    print(
        f"{name:25} -> "
        f"{entity['canonical_name'] if entity else None:15} "
        f"{score:.3f}"
    )