import json
from pathlib import Path


ONTOLOGY_DIR = Path(__file__).parent


def load_json(filename):
    with open(ONTOLOGY_DIR / filename, "r") as f:
        return json.load(f)


def load_entity_schema():
    return load_json("entities.json")


def load_relationship_schema():
    return load_json("relationships.json")


def load_rules():
    return load_json("rules.json")