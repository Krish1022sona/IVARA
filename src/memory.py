import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "memories.json"


def add_memory(
    text,
    timestamp,
    source,
    type,
    importance,
    entities=None,
    topics=None,
    relationships=None,
    confidence=1.0
):
    with open(DATA_PATH, "r") as f:
        memos = json.load(f)

    M = len(memos)

    memo = {
        "id": f"m{M + 1:03d}",
        "text": text,
        "timestamp": timestamp,
        "source": source,
        "type": type,
        "importance": importance,
        "confidence": confidence,

        "entities": entities or [],
        "topics": topics or [],
        "relationships": relationships or []
    }

    memos.append(memo)

    with open(DATA_PATH, "w") as f:
        json.dump(memos, f, indent=2)

    return memo