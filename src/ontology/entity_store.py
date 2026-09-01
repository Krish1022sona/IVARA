import json
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent.parent / "data" / "entities.json"


def load_entities():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def save_entities(entities):
    with open(DATA_PATH, "w") as f:
        json.dump(entities, f, indent=2)


def create_entity(name, entity_types):
    entities = load_entities()

    # Generate a new ID
    entity_id = f"e{len(entities) + 1:03d}"

    entity = {
        "id": entity_id,
        "canonical_name": name,
        "aliases": [],
        "types": entity_types
    }

    entities.append(entity)
    save_entities(entities)

    return entity

def get_self():
    return {
        "id": "self",
        "canonical_name": "SELF",
        "aliases": [],
        "types": ["Person"]
    }

