import numpy as np
from .entity_store import create_entity, get_self
from .vocabulary import is_type_compatible, get_canonical_type
from ..embedding import get_embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def normalize(text):
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def find_exact_match(name, entities, entity_type=None):
    name_norm = normalize(name)

    # First pass: exact match with type compatibility
    for entity in entities:
        cand_name = normalize(entity.get("canonical_name", ""))
        cand_aliases = [normalize(a) for a in entity.get("aliases", [])]

        if cand_name == name_norm or name_norm in cand_aliases:
            if entity_type is None or is_type_compatible(entity_type, entity.get("types", [])):
                return entity

    return None


def build_entity_embeddings(entities):
    return [
        get_embedding(entity["canonical_name"])
        for entity in entities
    ]


def find_best_match(name, entities, embeddings, entity_type=None):
    if not entities:
        return None, 0.0

    query_embedding = get_embedding(name)

    best_entity = None
    best_score = -1.0

    for entity, embedding in zip(entities, embeddings):
        # Type compatibility filter
        if entity_type is not None:
            if not is_type_compatible(entity_type, entity.get("types", [])):
                continue

        score = cosine_similarity(query_embedding, embedding)

        if score > best_score:
            best_score = score
            best_entity = entity

    return best_entity, float(best_score) if best_score != -1.0 else 0.0


def resolve_entity(
    name,
    entity_type,
    entities,
    embeddings,
    threshold=0.80
):
    if not name or not isinstance(name, str):
        return {
            "status": "candidate",
            "entity": {
                "canonical_name": "",
                "aliases": [],
                "types": [entity_type] if entity_type else []
            },
            "score": 0.0,
            "method": "invalid"
        }

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
        entities,
        entity_type=entity_type
    )

    if exact_match:
        return {
            "status": "matched",
            "entity": exact_match,
            "score": 1.0,
            "method": "exact"
        }

    # 2. Semantic similarity (with type compatibility filter)
    entity, score = find_best_match(
        name,
        entities,
        embeddings,
        entity_type=entity_type
    )

    if entity and score >= threshold:
        return {
            "status": "matched",
            "entity": entity,
            "score": score,
            "method": "semantic"
        }

    # 3. Create candidate entity (unpersisted)
    canon_type = get_canonical_type(entity_type) if entity_type else None
    assigned_types = [canon_type] if canon_type else ([entity_type] if entity_type else [])

    return {
        "status": "candidate",
        "entity": {
            "canonical_name": name.strip(),
            "aliases": [],
            "types": assigned_types
        },
        "score": score,
        "method": "new"
    }
