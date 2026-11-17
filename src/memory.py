import json
from pathlib import Path
from typing import Any, Optional

DATA_PATH = Path(__file__).parent.parent / "data" / "memories.json"


def load_memories(data_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load all memories from the specified or default JSON file."""
    target_path = Path(data_path) if data_path else DATA_PATH
    if not target_path.exists():
        return []
    with open(target_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []


def save_memories(memories: list[dict[str, Any]], data_path: str | Path | None = None) -> None:
    """Save memories list atomically to the specified JSON file."""
    target_path = Path(data_path) if data_path else DATA_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2)
    temp_path.replace(target_path)


def get_memory(memory_id: str, data_path: str | Path | None = None) -> Optional[dict[str, Any]]:
    """Retrieve a single memory by ID."""
    memos = load_memories(data_path=data_path)
    for m in memos:
        if m.get("id") == memory_id:
            return m
    return None


def list_memories(data_path: str | Path | None = None) -> list[dict[str, Any]]:
    """List all stored memories."""
    return load_memories(data_path=data_path)


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
    memos = load_memories(data_path=target_path)

    # Compute stable monotonically increasing ID
    max_num = 0
    for m in memos:
        mid = str(m.get("id", ""))
        if mid.startswith("m") and mid[1:].isdigit():
            try:
                max_num = max(max_num, int(mid[1:]))
            except ValueError:
                pass
    new_id = f"m{max(max_num + 1, len(memos) + 1):03d}"

    memo = {
        "id": new_id,
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
    save_memories(memos, data_path=target_path)

    return memo


def delete_memory(
    memory_id: str,
    data_path: str | Path | None = None,
    graph_store: Any | None = None
) -> bool:
    """
    Delete a memory record from JSON and purge its graph nodes and edges.
    
    Returns True if the memory existed and was deleted, False otherwise.
    """
    target_path = Path(data_path) if data_path else DATA_PATH
    memos = load_memories(data_path=target_path)

    original_count = len(memos)
    memos = [m for m in memos if m.get("id") != memory_id]

    if len(memos) == original_count:
        return False

    save_memories(memos, data_path=target_path)

    if graph_store is not None and hasattr(graph_store, "delete_memory_nodes_and_edges"):
        graph_store.delete_memory_nodes_and_edges(memory_id)

    return True


def clear_memories(
    data_path: str | Path | None = None,
    graph_store: Any | None = None
) -> int:
    """
    Clear all long-term memories from JSON and the graph store.
    
    Returns count of cleared memories.
    """
    target_path = Path(data_path) if data_path else DATA_PATH
    memos = load_memories(data_path=target_path)
    count = len(memos)

    save_memories([], data_path=target_path)

    if graph_store is not None and hasattr(graph_store, "clear"):
        graph_store.clear()

    return count


def get_memory_stats(
    data_path: str | Path | None = None,
    graph_store: Any | None = None
) -> dict[str, Any]:
    """Return statistical summary of stored memories and graph elements."""
    memos = load_memories(data_path=data_path)
    sources = set(m.get("source", "unknown") for m in memos)

    stats = {
        "total_memories": len(memos),
        "unique_sources": len(sources),
        "entities_in_memories": sum(len(m.get("entities", [])) for m in memos),
        "relationships_in_memories": sum(len(m.get("relationships", [])) for m in memos),
        "temporal_claims_in_memories": sum(len(m.get("temporal_claims", [])) for m in memos),
    }

    if graph_store is not None:
        stats["graph_nodes"] = graph_store.count_nodes() if hasattr(graph_store, "count_nodes") else 0
        stats["graph_edges"] = graph_store.count_edges() if hasattr(graph_store, "count_edges") else 0

    return stats


def sync_memory_graph(
    data_path: str | Path | None = None,
    entity_path: str | Path | None = None,
    graph_store: Any | None = None
) -> dict[str, int]:
    """
    Synchronize stored memories and canonical entities into the persistent SQLite graph store.
    Ensures all existing memories, chunks, entities, and relationships are reflected in the graph.
    """
    if graph_store is None:
        return {"nodes_count": 0, "edges_count": 0}

    from .graph.builder import MemoryGraphBuilder
    from .ontology.entity_store import load_entities

    mems = load_memories(data_path=data_path)
    ents = load_entities(path=entity_path)

    builder = MemoryGraphBuilder(graph_store)
    return builder.build_from_records(mems, ents)
