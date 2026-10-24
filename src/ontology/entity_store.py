import json
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent.parent / "data" / "entities.json"


def load_entities(path=None):
    p = Path(path) if path else DATA_PATH
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_entities(entities, path=None):
    p = Path(path) if path else DATA_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2)


def create_entity(name, entity_types, path=None):
    entities = load_entities(path=path)

    # Generate a new ID
    entity_id = f"e{len(entities) + 1:03d}"

    entity = {
        "id": entity_id,
        "canonical_name": name,
        "aliases": [],
        "types": entity_types
    }

    entities.append(entity)
    save_entities(entities, path=path)

    return entity

def get_self():
    return {
        "id": "self",
        "canonical_name": "SELF",
        "aliases": [],
        "types": ["Person"]
    }

