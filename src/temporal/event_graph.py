"""
Temporal Event Graph Schema Definitions for IVARA.

Defines EventNode, TemporalRelationshipEdge, TemporalPrecision, and TemporalRelationshipType
with full provenance, valid intervals, recording time, and bounded relationships.
"""

from enum import Enum
import hashlib
from typing import Any, Literal
from pydantic import BaseModel, Field

from ..graph.schema import GraphNode, GraphEdge


class TemporalPrecision(str, Enum):
    """Temporal granularity and certainty classification."""
    EXACT_DAY = "EXACT_DAY"    # Standard calendar day: YYYY-MM-DD
    MONTH = "MONTH"            # Month level: YYYY-MM (spans 1st to end of month)
    YEAR = "YEAR"              # Year level: YYYY (spans Jan 1 to Dec 31)
    RELATIVE = "RELATIVE"      # Relative to recorded_at date
    UNKNOWN = "UNKNOWN"        # Approximate, unanchored, or unknown time


class TemporalRelationshipType(str, Enum):
    """Minimal, bounded temporal relationship vocabulary."""
    OCCURRED_BEFORE = "OCCURRED_BEFORE"
    OCCURRED_AFTER = "OCCURRED_AFTER"
    SUPERSEDES = "SUPERSEDES"
    INVALIDATES = "INVALIDATES"


class EventNode(BaseModel):
    """
    Fact-bearing event node in the Temporal Event Graph.
    Strictly decouples event occurrence time from conversation recording/ingestion time.
    """
    event_id: str = Field(description="Unique, deterministic event identifier")
    description: str = Field(description="Canonical summary of event")
    entity_ids: list[str] = Field(default_factory=list, description="IDs of entities linked to this event")
    session_id: str = Field(description="Source session or conversation ID")
    turn_id: str | None = Field(default=None, description="Source turn or chunk ID")
    event_time_text: str | None = Field(default=None, description="Verbatim temporal phrase in source text")
    valid_from: str | None = Field(default=None, description="Standardized ISO date: YYYY-MM-DD, YYYY-MM, or YYYY")
    valid_to: str | None = Field(default=None, description="Standardized ISO date: YYYY-MM-DD, YYYY-MM, or YYYY")
    recorded_at: str | None = Field(default=None, description="Timestamp of conversation ingestion")
    precision: TemporalPrecision = Field(default=TemporalPrecision.UNKNOWN, description="Temporal precision level")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    provenance: str = Field(default="", description="Verbatim quote containing event and date")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")

    def to_graph_node(self) -> GraphNode:
        """Serialize to universal GraphNode for SQLiteGraphStore."""
        return GraphNode(
            id=self.event_id,
            node_type="event",
            label=self.description[:80],
            properties={
                "description": self.description,
                "entity_ids": self.entity_ids,
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "event_time_text": self.event_time_text,
                "valid_from": self.valid_from,
                "valid_to": self.valid_to,
                "recorded_at": self.recorded_at,
                "precision": self.precision.value if isinstance(self.precision, TemporalPrecision) else self.precision,
                "confidence": self.confidence,
                "provenance": self.provenance,
                **self.properties
            }
        )

    @classmethod
    def from_graph_node(cls, node: GraphNode) -> "EventNode":
        """Deserialize from GraphNode."""
        p = node.properties
        return cls(
            event_id=node.id,
            description=p.get("description", node.label),
            entity_ids=p.get("entity_ids", []),
            session_id=p.get("session_id", ""),
            turn_id=p.get("turn_id"),
            event_time_text=p.get("event_time_text"),
            valid_from=p.get("valid_from"),
            valid_to=p.get("valid_to"),
            recorded_at=p.get("recorded_at"),
            precision=TemporalPrecision(p.get("precision", "UNKNOWN")),
            confidence=float(p.get("confidence", 1.0)),
            provenance=p.get("provenance", ""),
            properties={k: v for k, v in p.items() if k not in {
                "description", "entity_ids", "session_id", "turn_id",
                "event_time_text", "valid_from", "valid_to", "recorded_at",
                "precision", "confidence", "provenance"
            }}
        )


class TemporalRelationshipEdge(BaseModel):
    """
    Directed temporal relationship between two EventNodes.
    Retains source provenance and confidence.
    """
    id: str = Field(description="Deterministic edge identifier")
    source_id: str = Field(description="Source EventNode ID")
    target_id: str = Field(description="Target EventNode ID")
    relationship: TemporalRelationshipType = Field(description="Temporal relationship type")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in this edge")
    evidence: str | None = Field(default=None, description="Textual reason or calculation basis")
    provenance_chunk_ids: list[str] = Field(default_factory=list, description="IDs of chunks supporting this relationship")
    derived_by: Literal[
        "grounded_interval_comparison",
        "explicit_claim",
        "state_supersession_rule",
        "invalidation_rule"
    ] = Field(description="Derivation method (never purely ingestion order)")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional edge metadata")

    def to_graph_edge(self) -> GraphEdge:
        """Serialize to universal GraphEdge for SQLiteGraphStore."""
        return GraphEdge(
            id=self.id,
            source_id=self.source_id,
            target_id=self.target_id,
            edge_type=self.relationship.value if isinstance(self.relationship, TemporalRelationshipType) else self.relationship,
            origin="explicit" if self.derived_by == "explicit_claim" else "inferred",
            confidence=self.confidence,
            evidence=self.evidence,
            properties={
                "derived_by": self.derived_by,
                "provenance_chunk_ids": self.provenance_chunk_ids,
                **self.properties
            }
        )

    @classmethod
    def from_graph_edge(cls, edge: GraphEdge) -> "TemporalRelationshipEdge":
        """Deserialize from GraphEdge."""
        p = edge.properties
        return cls(
            id=edge.id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            relationship=TemporalRelationshipType(edge.edge_type),
            confidence=edge.confidence,
            evidence=edge.evidence,
            provenance_chunk_ids=p.get("provenance_chunk_ids", []),
            derived_by=p.get("derived_by", "grounded_interval_comparison"),
            properties={k: v for k, v in p.items() if k not in {"derived_by", "provenance_chunk_ids"}}
        )


def generate_event_id(session_id: str, desc: str, valid_from: str | None) -> str:
    """Generate a deterministic, stable Event ID."""
    raw = f"{session_id}:{desc.strip().lower()}:{valid_from or 'none'}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"evt_{h}"


def generate_temporal_edge_id(source_id: str, target_id: str, rel: TemporalRelationshipType) -> str:
    """Generate a deterministic edge ID."""
    rel_str = rel.value if isinstance(rel, TemporalRelationshipType) else rel
    raw = f"{source_id}:{rel_str}:{target_id}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"tedge_{h}"

