import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .embedding import get_embedding


DATA_PATH = Path(__file__).parent.parent / "data" / "memories.json"

# Relevance threshold calibrated for SentenceTransformer all-MiniLM-L6-v2 cosine similarity.
# Inner product on L2-normalized vectors produces cosine similarity in range [-1.0, 1.0].
# - Scores >= 0.70 indicate strong semantic alignment.
# - Scores 0.35 - 0.70 indicate topical or contextual relevance.
# - Scores < 0.35 represent topical drift or spurious semantic noise.
DEFAULT_RELEVANCE_THRESHOLD = 0.35


def load_memories(data_path: str | Path | None = None) -> list[dict]:
    """
    Load stored memories from the configured or specified JSON path.
    Returns an empty list if the file is missing, empty, or malformed.
    """
    target = Path(data_path) if data_path else DATA_PATH
    if not target.exists():
        return []

    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def build_index(memories: list[dict]):
    """
    Construct an in-memory FAISS IndexFlatIP index over L2-normalized memory text embeddings.
    Returns None if memories is empty or contains no valid text.
    """
    if not memories:
        return None

    texts = [m.get("text", "") if isinstance(m, dict) else str(m) for m in memories]
    raw_embeddings = [get_embedding(t) for t in texts]

    if not raw_embeddings:
        return None

    embeddings = np.array(raw_embeddings, dtype="float32")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return None

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve(
    query: str,
    k: int = 3,
    min_score: float | None = DEFAULT_RELEVANCE_THRESHOLD,
    memories: list[dict] | None = None,
    data_path: str | Path | None = None
) -> list[dict]:
    """
    Retrieve top-k relevant memories using cosine similarity with relevance filtering.

    Args:
        query: Search query text. Empty or whitespace-only queries return [].
        k: Maximum number of memories to return (default 3).
        min_score: Minimum cosine similarity score required (default 0.35).
                   Set to None to disable relevance filtering and return raw top-k.
        memories: Optional in-memory list of memory records (overrides data_path).
        data_path: Optional custom path to memories JSON file (used if memories is None).

    Returns:
        List of qualifying memory dictionaries, enriched with 'score' (cosine similarity),
        preserving all original provenance metadata (id, timestamp, source, entities, relationships).
    """
    # 1. Safe query validation
    if not query or not isinstance(query, str) or not query.strip():
        return []

    # 2. Memory loading
    if memories is None:
        memories = load_memories(data_path=data_path)

    if not memories or k <= 0:
        return []

    # 3. Build FAISS index
    index = build_index(memories)
    if index is None:
        return []

    # 4. Generate normalized query embedding
    query_text = query.strip()
    query_embedding = np.array(
        [get_embedding(query_text)],
        dtype="float32"
    )

    if query_embedding.ndim != 2 or query_embedding.shape[0] == 0:
        return []

    faiss.normalize_L2(query_embedding)

    # 5. Search top candidates
    search_k = min(int(k), len(memories))
    if search_k <= 0:
        return []

    scores, indices = index.search(query_embedding, search_k)

    # 6. Filter by relevance threshold and preserve full provenance
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(memories):
            continue

        float_score = float(score)

        # Relevance filtering: skip memories that do not meet the threshold
        if min_score is not None and float_score < min_score:
            continue

        # Preserve full provenance and attach relevance score
        memo = dict(memories[idx])
        memo["score"] = round(float_score, 4)
        results.append(memo)

    return results


def prepare_evidence_context(query: str, retrieved_memories: list[dict]) -> dict[str, Any]:
    """
    Prepare a structured evidence-grounded context from retrieved memories.
    Serves as the foundation for hallucination-resistant answer generation.

    Distinguishes:
      - Directly supported evidence (raw episode text, explicit relationships, verbatim evidence)
      - Inferred evidence (relationships marked with origin='inferred', citing premise evidence)
      - Insufficient evidence status when no memories meet the relevance threshold.
    """
    if not retrieved_memories:
        return {
            "query": query,
            "has_sufficient_evidence": False,
            "status": "insufficient_evidence",
            "message": f"Insufficient evidence: No relevant memories found to answer '{query}'.",
            "direct_evidence": [],
            "inferred_evidence": [],
            "entities": [],
            "retrieved_memories": [],
            "retrieved_memories_count": 0,
            "context_text": f"Query: {query}\nStatus: Insufficient evidence in personal memory store."
        }

    direct_evidence = []
    inferred_evidence = []
    seen_entities = {}

    for memo in retrieved_memories:
        memo_id = memo.get("id", "unknown")
        memo_text = memo.get("text", "")
        memo_ts = memo.get("timestamp")
        memo_src = memo.get("source")
        memo_score = memo.get("score")

        memo_direct_facts = []

        # Process relationships
        for rel in memo.get("relationships", []):
            origin = rel.get("origin", "explicit")
            pred = rel.get("predicate") or rel.get("relation")
            src = rel.get("source")
            tgt = rel.get("target")
            ev = rel.get("evidence")

            src_label = src.get("mention") or src.get("id") or str(src) if isinstance(src, dict) else str(src)
            tgt_label = tgt.get("mention") or tgt.get("id") or str(tgt) if isinstance(tgt, dict) else str(tgt)

            fact_entry = {
                "memory_id": memo_id,
                "statement": f"{src_label} {pred} {tgt_label}",
                "predicate": pred,
                "source": src,
                "target": tgt,
                "evidence": ev,
                "confidence": rel.get("confidence", 1.0)
            }

            if origin == "inferred":
                inferred_evidence.append({
                    **fact_entry,
                    "origin": "inferred",
                    "premise_evidence": ev
                })
            else:
                memo_direct_facts.append({
                    **fact_entry,
                    "origin": "explicit"
                })

        # Process entities
        for ent in memo.get("entities", []):
            ent_key = ent.get("canonical_name") or ent.get("mention") or ent.get("entity_id")
            if ent_key and ent_key not in seen_entities:
                seen_entities[ent_key] = {
                    "canonical_name": ent.get("canonical_name"),
                    "mention": ent.get("mention"),
                    "types": ent.get("types", []),
                    "entity_id": ent.get("entity_id"),
                    "memory_id": memo_id
                }

        direct_evidence.append({
            "memory_id": memo_id,
            "text": memo_text,
            "timestamp": memo_ts,
            "source": memo_src,
            "score": memo_score,
            "direct_facts": memo_direct_facts
        })

    # Build human/LLM-readable grounding context text
    count = len(retrieved_memories)
    context_lines = [
        f"Query: {query}",
        f"Status: Sufficient evidence ({count} relevant episode{'s' if count != 1 else ''} found)\n",
        "--- DIRECTLY SUPPORTED EVIDENCE ---"
    ]
    for d in direct_evidence:
        context_lines.append(f"[{d['memory_id']}] (relevance: {d['score']}, timestamp: {d['timestamp']}): \"{d['text']}\"")
        for f in d["direct_facts"]:
            context_lines.append(f"  * Fact: {f['statement']} (evidence: \"{f['evidence']}\")")

    if inferred_evidence:
        context_lines.append("\n--- INFERRED RELATIONSHIPS (DERIVED) ---")
        for inf in inferred_evidence:
            context_lines.append(
                f"  * [{inf['memory_id']}] Inferred: {inf['statement']} (derived from premise: \"{inf['premise_evidence']}\")"
            )

    context_text = "\n".join(context_lines)

    return {
        "query": query,
        "has_sufficient_evidence": True,
        "status": "sufficient_evidence",
        "message": f"Found {len(retrieved_memories)} relevant memories.",
        "direct_evidence": direct_evidence,
        "inferred_evidence": inferred_evidence,
        "entities": list(seen_entities.values()),
        "retrieved_memories": retrieved_memories,
        "retrieved_memories_count": len(retrieved_memories),
        "context_text": context_text
    }


def retrieve_with_context(
    query: str,
    k: int = 3,
    min_score: float | None = DEFAULT_RELEVANCE_THRESHOLD,
    memories: list[dict] | None = None,
    data_path: str | Path | None = None
) -> dict[str, Any]:
    """
    Execute evidence-grounded retrieval and return the structured evidence context.
    """
    memories_list = retrieve(query, k=k, min_score=min_score, memories=memories, data_path=data_path)
    return prepare_evidence_context(query, memories_list)