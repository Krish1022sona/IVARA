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
    confidence=1.0,
    diagnostics=None,
    data_path=None,
    temporal_claims=None
):
    target_path = Path(data_path) if data_path else DATA_PATH

    if not target_path.exists():
        memos = []
    else:
        with open(target_path, "r", encoding="utf-8") as f:
            try:
                memos = json.load(f)
            except json.JSONDecodeError:
                memos = []

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

    if temporal_claims is not None:
        memo["temporal_claims"] = temporal_claims

    if diagnostics is not None:
        memo["diagnostics"] = diagnostics

    memos.append(memo)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(memos, f, indent=2)

    return memo
