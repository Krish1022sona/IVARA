import json
from pathlib import Path


ONTOLOGY_DIR = Path(__file__).parent


def load_json(filename):
    with open(ONTOLOGY_DIR / filename, "r") as f:
        return json.load(f)


def load_entity_types():
    return load_json("entities.json")["types"]


def load_relationships():
    return load_json("relationships.json")["relationships"]


def load_rules():
    return load_json("rules.json")["relationship_rules"]

if __name__ == "__main__":
    print(load_entity_types())
    print(load_relationships())
    print(load_rules())