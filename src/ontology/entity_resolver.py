import numpy as np
from .entity_store import create_entity, get_self
from ..embedding import get_embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def normalize(text):
    return text.lower().strip()


def find_exact_match(name, entities):
    name = normalize(name)

    for entity in entities:
        if normalize(entity["canonical_name"]) == name:
            return entity

        for alias in entity.get("aliases", []):
            if normalize(alias) == name:
                return entity

    return None


def build_entity_embeddings(entities):

    return [
        get_embedding(entity["canonical_name"])
        for entity in entities
    ]


def find_best_match(name, entities, embeddings):

    if not entities:
        return None, 0.0

    query_embedding = get_embedding(name)

    best_entity = None
    best_score = -1.0

    for entity, embedding in zip(
        entities,
        embeddings
    ):
        score = cosine_similarity(
            query_embedding,
            embedding
        )

        if score > best_score:
            best_score = score
            best_entity = entity

    return best_entity, float(best_score)


def resolve_entity(
    name,
    entity_type,
    entities,
    embeddings,
    threshold=0.80
):
    # 0. Extract SELF
    if name.strip().upper() == "SELF":
        return {
            "status": "matched",
            "entity": get_self(),
            "score": 1.0,
            "method": "special"
        }

    # 1. Exact name / alias
    exact_match = find_exact_match(
        name,
        entities
    )

    if exact_match:
        return {
            "status": "matched",
            "entity": exact_match,
            "score": 1.0,
            "method": "exact"
        }

    # 2. Semantic similarity
    entity, score = find_best_match(
        name,
        entities,
        embeddings
    )

    if entity and score >= threshold:
        return {
            "status": "matched",
            "entity": entity,
            "score": score,
            "method": "semantic"
        }

    # 3. Create new entity
    return {
        "status": "candidate",
        "entity": {
            "canonical_name": name,
            "aliases": [],
            "types": [entity_type]
        },
        "score": score,
        "method": "new"
    }