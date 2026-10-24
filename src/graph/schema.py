"""
Memory Graph Schema Definitions for IVARA.

Defines logical node types (MemoryNode, ChunkNode, EntityNode, EpisodeNode)
and typed edges with provenance, confidence, and temporal metadata.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """Base schema for any node in the IVARA Memory Graph."""
    id: str = Field(description="Unique, stable node identifier")
    node_type: Literal["chunk", "memory", "entity", "episode"] = Field(description="Logical node classification")
    label: str = Field(description="Human-readable label or entity name")
    properties: dict[str, Any] = Field(default_factory=dict, description="Arbitrary attributes and metadata")


class ChunkNode(BaseModel):
    """Fact-bearing chunk node derived from an episode or memory."""
    id: str = Field(description="Unique chunk identifier, e.g. chunk_m001_00")
    memory_id: str = Field(description="Parent memory ID, e.g. m001")
    episode_id: str | None = Field(default=None, description="Source episode ID or title")
    text: str = Field(description="Verbatim chunk text")
    chunk_index: int = Field(default=0, description="0-indexed position within parent memory")
    char_start: int = Field(default=0, description="Character start offset in parent text")
    char_end: int = Field(default=0, description="Character end offset in parent text")
    timestamp: str | None = Field(default=None, description="Event or memory timestamp")
    origin: Literal["explicit", "inferred"] = Field(default="explicit", description="Provenance origin")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    entity_ids: list[str] = Field(default_factory=list, description="IDs of canonical entities referenced in this chunk")

    def to_graph_node(self) -> GraphNode:
        return GraphNode(
            id=self.id,
            node_type="chunk",
            label=f"Chunk {self.chunk_index} of {self.memory_id}",
            properties={
                "memory_id": self.memory_id,
                "episode_id": self.episode_id,
                "text": self.text,
                "chunk_index": self.chunk_index,
                "char_start": self.char_start,
                "char_end": self.char_end,
                "timestamp": self.timestamp,
                "origin": self.origin,
                "confidence": self.confidence,
                "entity_ids": self.entity_ids
            }
        )


class EntityNode(BaseModel):
    """Canonical ontology entity node (Person, Organization, Technology, Project, etc.)."""
    id: str = Field(description="Canonical entity ID, e.g. ent_marcus_vance")
    canonical_name: str = Field(description="Primary canonical entity name")
    aliases: list[str] = Field(default_factory=list, description="Known name aliases or mentions")
    types: list[str] = Field(default_factory=list, description="Ontology types, e.g. ['Person']")

    def to_graph_node(self) -> GraphNode:
        return GraphNode(
            id=self.id,
            node_type="entity",
            label=self.canonical_name,
            properties={
                "canonical_name": self.canonical_name,
                "aliases": self.aliases,
                "types": self.types
            }
        )


class EpisodeNode(BaseModel):
    """Chronological episode or narrative container node."""
    id: str = Field(description="Unique episode identifier, e.g. ep_001")
    title: str = Field(description="Episode title")
    timestamp: str | None = Field(default=None, description="Date or temporal interval")
    source: str = Field(default="chronicle", description="Source channel or provenance")
    importance: float = Field(default=1.0, description="Memory importance score")

    def to_graph_node(self) -> GraphNode:
        return GraphNode(
            id=self.id,
            node_type="episode",
            label=self.title,
            properties={
                "title": self.title,
                "timestamp": self.timestamp,
                "source": self.source,
                "importance": self.importance
            }
        )


class GraphEdge(BaseModel):
    """Typed, validated edge connecting two nodes in the Memory Graph."""
    id: str = Field(description="Unique, deterministic edge ID")
    source_id: str = Field(description="Source node ID")
    target_id: str = Field(description="Target node ID")
    edge_type: str = Field(description="Predicate or structural edge type")
    origin: Literal["explicit", "inferred", "structural"] = Field(default="explicit", description="Relationship provenance")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship confidence")
    evidence: str | None = Field(default=None, description="Verbatim supporting text span")
    memory_id: str | None = Field(default=None, description="Supporting memory ID")
    chunk_id: str | None = Field(default=None, description="Supporting chunk ID")
    timestamp: str | None = Field(default=None, description="Valid time or event date")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional edge metadata")
