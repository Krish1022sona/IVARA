import json
from pathlib import Path

import faiss
import numpy as np

from .embedding import get_embedding


DATA_PATH = Path(__file__).parent.parent / "data" / "memories.json"


def load_memories():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def build_index(memories):
    embeddings = np.array(
        [get_embedding(memory["text"]) for memory in memories],
        dtype="float32"
    )

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve(query, k=3):
    memories = load_memories()

    if not memories:
        return []

    index = build_index(memories)

    query_embedding = np.array(
        [get_embedding(query)],
        dtype="float32"
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)

    return [
        memories[index]
        for index in indices[0]
        if index != -1
    ]