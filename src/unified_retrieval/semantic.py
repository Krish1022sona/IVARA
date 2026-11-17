"""Semantic Retrieval Engine for IVARA.

Wraps and standardizes the core semantic memory retriever (src.retrieval),
exposing structured evidence items with memory IDs, content, entities,
canonical and open-world relationships, timestamps, and episodic source turn IDs.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Optional

from ..retrieval import retrieve as base_retrieve, DEFAULT_RELEVANCE_THRESHOLD


@dataclass
class SemanticEvidenceItem:
    """Represents a retrieved semantic memory record enriched with structured metadata."""
    memory_id: str
    text: str
    timestamp: Optional[str]
    source_turn_id: Optional[str]
    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    topics: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    channel: str = "semantic"
    raw_source: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "timestamp": self.timestamp,
            "source_turn_id": self.source_turn_id,
            "entities": self.entities,
            "relationships": self.relationships,
            "topics": self.topics,
            "score": round(self.score, 4),
            "channel": self.channel,
            "raw_source": self.raw_source,
            "confidence": self.confidence
        }


class SemanticRetriever:
    """
    Standardized semantic memory retriever.
    Wraps existing vector search over memories.json, extracting provenance,
    entity mentions, and grounded relationship facts.
    """

    def __init__(
        self,
        memories: Optional[list[dict[str, Any]]] = None,
        data_path: Optional[str | Path] = None,
        default_min_score: float = DEFAULT_RELEVANCE_THRESHOLD
    ):
        self.memories = memories
        self.data_path = data_path
        self.default_min_score = default_min_score

    @staticmethod
    def extract_source_turn_id(source_field: Optional[str]) -> Optional[str]:
        """
        Extract the underlying turn ID or message ID from a memory's source attribute.
        e.g., 'session_01_msg_msg_a12484571c4f' -> 'msg_a12484571c4f'
              'conv_8b62d30c356f_msg_50478aedb404' -> 'msg_50478aedb404'
              'turn_12345' -> 'turn_12345'
        """
        if not source_field or not isinstance(source_field, str):
            return None

        # Look for explicit turn_... or msg_... token
        m = re.search(r"\b(turn_[a-zA-Z0-9_]+|msg_[a-zA-Z0-9_]+)\b", source_field)
        if m:
            return m.group(1)
        return source_field

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: Optional[float] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> list[SemanticEvidenceItem]:
        """
        Execute semantic memory retrieval with metadata enrichment and temporal filtering.
        """
        effective_min_score = min_score if min_score is not None else self.default_min_score

        raw_results = base_retrieve(
            query=query,
            k=k * 2 if (start_time or end_time) else k,
            min_score=effective_min_score,
            memories=self.memories,
            data_path=self.data_path
        )

        results: list[SemanticEvidenceItem] = []
        for memo in raw_results:
            m_ts = memo.get("timestamp")

            # Temporal filtering
            if start_time and m_ts and m_ts < start_time:
                continue
            if end_time and m_ts and m_ts > end_time:
                continue

            raw_source = memo.get("source")
            turn_id = self.extract_source_turn_id(raw_source)

            results.append(SemanticEvidenceItem(
                memory_id=memo.get("id", "m000"),
                text=memo.get("text", ""),
                timestamp=m_ts,
                source_turn_id=turn_id,
                entities=memo.get("entities", []),
                relationships=memo.get("relationships", []),
                topics=memo.get("topics", []),
                score=memo.get("score", 0.0),
                channel="semantic",
                raw_source=raw_source,
                confidence=memo.get("confidence", 1.0)
            ))

            if len(results) >= k:
                break

        return results

