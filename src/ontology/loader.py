import json
from pathlib import Path


ONTOLOGY_DIR = Path(__file__).parent


def load_json(filename):
    file_path = ONTOLOGY_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Ontology configuration file not found: {filename}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in ontology configuration file '{filename}': {e}") from e


def load_entity_schema():
    schema = load_json("entities.json")
    required_sections = ["entity", "type", "special_entities"]
    missing = [sec for sec in required_sections if sec not in schema]
    if missing:
        raise ValueError(f"Missing required schema sections in entities.json: {missing}")
    return schema


def load_relationship_schema():
    schema = load_json("relationships.json")
    required_sections = ["relationship", "predicate"]
    missing = [sec for sec in required_sections if sec not in schema]
    if missing:
        raise ValueError(f"Missing required schema sections in relationships.json: {missing}")
    return schema


def load_rules():
    data = load_json("rules.json")
    if "rules" not in data:
        raise ValueError("Missing required schema section in rules.json: 'rules'")
    return data


def load_entity_types():
    from .vocabulary import load_vocabulary
    vocab = load_vocabulary()
    return list(vocab.get("types", {}).keys())


def load_relationships():
    from .vocabulary import load_vocabulary
    vocab = load_vocabulary()
    return vocab.get("predicates", {})
