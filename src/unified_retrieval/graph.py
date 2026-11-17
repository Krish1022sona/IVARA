"""Graph Retrieval Engine for IVARA.

Provides bounded, relevance-guided knowledge graph retrieval over SQLiteGraphStore.
Identifies query-relevant seed entities, performs cycle-free multi-hop traversal,
preserves edge directionality, and extracts true domain entity-relationship triples
with full provenance (evidence span, memory ID, confidence, canonical/open-world status).
"""

from dataclasses import dataclass, field
import re
from typing import Any, Optional

from ..graph.store import SQLiteGraphStore
from ..graph.schema import GraphNode, GraphEdge
from ..graph.traversal import traverse_relevance_guided


@dataclass
class GraphEvidenceItem:
    """Represents a retrieved knowledge graph relationship triple with complete provenance."""
    triple_id: str
    source_id: str
    source_name: str
    source_type: str
    predicate: str
    target_id: str
    target_name: str
    target_type: str
    direction: str  # "outgoing" | "incoming"
    hops: int
    confidence: float
    score: float
    evidence: Optional[str] = None
    memory_id: Optional[str] = None
    predicate_status: str = "canonical"  # "canonical" | "open_world"
    channel: str = "graph"
    formatted_statement: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "triple_id": self.triple_id,
            "source": {
                "id": self.source_id,
                "name": self.source_name,
                "type": self.source_type
            },
            "predicate": self.predicate,
            "target": {
                "id": self.target_id,
                "name": self.target_name,
                "type": self.target_type
            },
            "direction": self.direction,
            "hops": self.hops,
            "confidence": self.confidence,
            "score": round(self.score, 4),
            "evidence": self.evidence,
            "memory_id": self.memory_id,
            "predicate_status": self.predicate_status,
            "channel": self.channel,
            "formatted_statement": self.formatted_statement,
            "metadata": self.metadata
        }


class GraphRetriever:
    """
    Retrieves grounded knowledge graph relationships for a query.
    Extracts seed entities, executes bounded relevance-guided traversal,
    and returns directed, provenance-backed domain triples.
    """

    def __init__(
        self,
        store: SQLiteGraphStore,
        max_depth: int = 2,
        max_nodes: int = 15,
        min_confidence: float = 0.5,
        k_rrf: int = 60,
        user_name: Optional[str] = None
    ):
        self.store = store
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.min_confidence = min_confidence
        self.k_rrf = k_rrf
        self.user_name = user_name

        # Cache entity nodes for efficient seed matching
        self.entity_nodes: list[GraphNode] = []
        try:
            self.entity_nodes = self.store.get_nodes_by_type("entity")
        except Exception:
            self.entity_nodes = []

    def identify_seed_entities(
        self,
        query: str,
        additional_entity_names: Optional[list[str]] = None
    ) -> list[str]:
        """
        Identify seed entity IDs matching the query text or additional entity mentions.
        Gracefully handles missing or ambiguous entities without errors.
        """
        clean_q = query.lower()
        seed_ids: list[str] = []
        seen = set()

        # 1. Match from stored entity nodes
        for ent in self.entity_nodes:
            props = ent.properties or {}
            c_name = props.get("canonical_name", ent.label or "").lower().strip()
            aliases = [a.lower().strip() for a in props.get("aliases", []) if a]

            # Check if canonical name or aliases appear as whole tokens in query
            is_match = False
            user_tokens = [" i ", " me ", " my ", " myself ", " user ", " self "]
            if self.user_name:
                user_tokens.append(f" {self.user_name.lower().strip()} ")
            if c_name and len(c_name) >= 3 and c_name in clean_q:
                is_match = True
            elif any(a and len(a) >= 3 and a in clean_q for a in aliases):
                is_match = True
            elif ent.id.lower() == "self" and any(tok in clean_q for tok in user_tokens):
                is_match = True

            if is_match and ent.id not in seen:
                seen.add(ent.id)
                seed_ids.append(ent.id)

        # 2. Match from additional supplied entity names (e.g. from episodic or semantic hits)
        if additional_entity_names:
            for extra in additional_entity_names:
                extra_clean = extra.lower().strip()
                for ent in self.entity_nodes:
                    props = ent.properties or {}
                    c_name = props.get("canonical_name", ent.label or "").lower().strip()
                    aliases = [a.lower().strip() for a in props.get("aliases", []) if a]
                    if extra_clean == c_name or extra_clean in aliases or extra_clean == ent.id.lower():
                        if ent.id not in seen:
                            seen.add(ent.id)
                            seed_ids.append(ent.id)

        return seed_ids

    def retrieve(
        self,
        query: str,
        k: int = 10,
        seed_entity_ids: Optional[list[str]] = None,
        additional_entity_names: Optional[list[str]] = None
    ) -> list[GraphEvidenceItem]:
        """
        Execute bounded graph traversal from seed entities and extract directed domain triples.
        """
        clean_q = query.strip()
        if not clean_q or not self.store:
            return []

        # 1. Discover seed entities
        seeds = list(seed_entity_ids or [])
        if not seeds:
            seeds = self.identify_seed_entities(clean_q, additional_entity_names=additional_entity_names)

        if not seeds:
            return []

        # 2. Execute bounded relevance-guided traversal
        traversal = traverse_relevance_guided(
            store=self.store,
            query=clean_q,
            seed_node_ids=seeds,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes
        )

        discovered_edges: list[GraphEdge] = traversal.get("discovered_edges", [])
        discovered_nodes = {n.id: n for n in traversal.get("discovered_nodes", [])}

        # Also inspect direct edges for each seed entity to guarantee 1-hop coverage
        for s_id in seeds:
            direct_edges = self.store.get_outgoing_edges(s_id) + self.store.get_incoming_edges(s_id)
            for de in direct_edges:
                if de.id not in [e.id for e in discovered_edges]:
                    discovered_edges.append(de)

        # 3. Extract and filter domain relationship triples
        triples: list[GraphEvidenceItem] = []
        seen_triple_keys = set()

        for edge in discovered_edges:
            # Filter out structural navigation edges
            e_type = edge.edge_type
            if e_type in ("contains_chunk", "adjacent_to", "contains_memory", "mentions_entity", "precedes"):
                continue
            if e_type.startswith("grounds_"):
                continue

            src_node = discovered_nodes.get(edge.source_id) or self.store.get_node(edge.source_id)
            tgt_node = discovered_nodes.get(edge.target_id) or self.store.get_node(edge.target_id)

            if not src_node or not tgt_node:
                continue

            # Ensure endpoints are entities or claims
            src_type = src_node.node_type
            tgt_type = tgt_node.node_type
            if src_type not in ("entity", "claim") or tgt_type not in ("entity", "claim"):
                continue

            src_name = src_node.properties.get("canonical_name", src_node.label or src_node.id)
            tgt_name = tgt_node.properties.get("canonical_name", tgt_node.label or tgt_node.id)

            # Prevent self-referential triples
            if src_name.lower().strip() == tgt_name.lower().strip():
                continue

            # Check confidence
            conf = edge.confidence if edge.confidence is not None else 1.0
            if conf < self.min_confidence:
                continue

            # Deduplication key
            t_key = (src_node.id, e_type, tgt_node.id)
            if t_key in seen_triple_keys:
                continue
            seen_triple_keys.add(t_key)

            # Determine direction relative to query seed entities
            direction = "outgoing" if edge.source_id in seeds else (
                "incoming" if edge.target_id in seeds else "undirected"
            )

            # Check hop distance
            hops = 1 if (edge.source_id in seeds or edge.target_id in seeds) else 2

            # Score calculation with hop-decay
            base_score = 0.85 if hops == 1 else 0.60
            if edge.evidence and any(tok in edge.evidence.lower() for tok in clean_q.lower().split() if len(tok) > 3):
                base_score += 0.10
            final_score = min(1.0, base_score * (0.8 ** (hops - 1)) * conf)

            # Predicate status: canonical vs open-world
            pred_status = edge.properties.get("predicate_status", "canonical")

            statement = f"{src_name} {e_type} {tgt_name}"

            triples.append(GraphEvidenceItem(
                triple_id=edge.id,
                source_id=src_node.id,
                source_name=src_name,
                source_type=src_type,
                predicate=e_type,
                target_id=tgt_node.id,
                target_name=tgt_name,
                target_type=tgt_type,
                direction=direction,
                hops=hops,
                confidence=conf,
                score=final_score,
                evidence=edge.evidence,
                memory_id=edge.memory_id,
                predicate_status=pred_status,
                channel="graph",
                formatted_statement=statement,
                metadata=edge.properties
            ))

        # Sort by score descending
        triples.sort(key=lambda t: t.score, reverse=True)
        return triples[:k]

