import json
from pathlib import Path


VOCAB_PATH = Path(__file__).parent / "vocabulary.json"


def load_vocabulary():
    with open(VOCAB_PATH, "r") as f:
        return json.load(f)


def save_vocabulary(vocabulary):
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocabulary, f, indent=2)


def add_type(name, aliases=None, parent=None):
    vocabulary = load_vocabulary()

    if name not in vocabulary["types"]:
        vocabulary["types"][name] = {
            "aliases": aliases or [],
            "parent": parent
        }

        save_vocabulary(vocabulary)


def add_predicate(name, aliases=None, inverse=None):
    vocabulary = load_vocabulary()

    if name not in vocabulary["predicates"]:
        vocabulary["predicates"][name] = {
            "aliases": aliases or [],
            "inverse": inverse
        }

        save_vocabulary(vocabulary)