"""Evidence Fusion Engine for IVARA.

Implements Reciprocal Rank Fusion with Provenance (RRF-P) to unify evidence across:
- Episodic conversation turns
- Semantic extracted memories
- Knowledge graph relationship triples

Guarantees:
1. Provenance-aware deduplication: links semantic memories to their source episodic turns,
   preventing duplicate counting while capturing multi-channel confirmation.
2. Independent evidence preservation: distinct historical states and independent sessions
   are preserved rather than arbitrarily collapsed.
3. Diversity protection: caps dominance from any single conversational session.
4. Full source attribution and configurable channel weighting.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Optional

from .episodic import EpisodicEvidenceItem
from .semantic import SemanticEvidenceItem
from .graph import GraphEvidenceItem


@dataclass
class RRFConfig:
    """Configuration parameters for Reciprocal Rank Fusion with Provenance."""
    k_rrf: int = 60
    weight_episodic: float = 1.0
    weight_semantic: float = 1.2
    weight_graph: float = 0.9
    max_items_per_session: int = 3
    relevance_floor: float = 0.20


@dataclass
class UnifiedEvidenceItem:
    """Consolidated evidence item representing fused information with complete provenance."""
    cluster_id: str
    primary_text: str
    timestamp: Optional[str]
    session_id: Optional[str]
    fused_score: float
    channels: list[str] = field(default_factory=list)
    episodic_turn: Optional[EpisodicEvidenceItem] = None
    semantic_memory: Optional[SemanticEvidenceItem] = None
    graph_triples: list[GraphEvidenceItem] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "primary_text": self.primary_text,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "fused_score": round(self.fused_score, 5),
            "channels": self.channels,
            "citations": self.citations,
            "episodic_turn": self.episodic_turn.to_dict() if self.episodic_turn else None,
            "semantic_memory": self.semantic_memory.to_dict() if self.semantic_memory else None,
            "graph_triples": [t.to_dict() for t in self.graph_triples],
            "metadata": self.metadata
        }


class UnifiedEvidenceFuser:
    """
    Fuses evidence candidates from episodic, semantic, and graph retrieval channels
    using Reciprocal Rank Fusion with Provenance (RRF-P).
    """

    def __init__(self, config: Optional[RRFConfig] = None):
        self.config = config or RRFConfig()

    def fuse(
        self,
        episodic_items: list[EpisodicEvidenceItem],
        semantic_items: list[SemanticEvidenceItem],
        graph_items: list[GraphEvidenceItem],
        top_k: int = 5
    ) -> list[UnifiedEvidenceItem]:
        """
        Merge, deduplicate, score, and rank multi-channel evidence.
        """
        k_rrf = self.config.k_rrf
        w_epi = self.config.weight_episodic
        w_sem = self.config.weight_semantic
        w_graph = self.config.weight_graph

        # 1. Map episodic turns by turn_id
        turns_by_id: dict[str, tuple[int, EpisodicEvidenceItem]] = {
            item.turn_id: (rank, item) for rank, item in enumerate(episodic_items, 1)
        }

        # 2. Map semantic memories by memory_id
        mems_by_id: dict[str, tuple[int, SemanticEvidenceItem]] = {
            item.memory_id: (rank, item) for rank, item in enumerate(semantic_items, 1)
        }

        # 3. Track clusters of consolidated evidence
        # Key: cluster_id (derived from memory_id or turn_id)
        clusters: dict[str, dict[str, Any]] = {}

        # First pass: cluster semantic memories and link to their source episodic turns
        consumed_turn_ids: set[str] = set()

        for sem_rank, sem in enumerate(semantic_items, 1):
            cluster_id = f"cluster_{sem.memory_id}"
            src_turn_id = sem.source_turn_id

            linked_turn = None
            epi_rank = None
            if src_turn_id and src_turn_id in turns_by_id:
                epi_rank, linked_turn = turns_by_id[src_turn_id]
                consumed_turn_ids.add(src_turn_id)

            # RRF score calculation
            sem_score = w_sem / (k_rrf + sem_rank)
            epi_score = (w_epi / (k_rrf + epi_rank)) if epi_rank is not None else 0.0
            total_score = sem_score + epi_score

            citations = [sem.memory_id]
            if linked_turn:
                citations.append(linked_turn.turn_id)

            channels = ["semantic"]
            if linked_turn:
                channels.append("episodic")

            session_id = None
            if linked_turn and linked_turn.session_id:
                session_id = linked_turn.session_id
            elif sem.raw_source and "session_" in sem.raw_source:
                m_sess = re.search(r"(session_\d+)", sem.raw_source)
                if m_sess:
                    session_id = m_sess.group(1)

            clusters[cluster_id] = {
                "cluster_id": cluster_id,
                "primary_text": sem.text,
                "timestamp": sem.timestamp,
                "session_id": session_id,
                "fused_score": total_score,
                "channels": channels,
                "episodic_turn": linked_turn,
                "semantic_memory": sem,
                "graph_triples": [],
                "citations": citations,
                "metadata": {"linked_via": "source_turn_id" if linked_turn else "standalone_semantic"}
            }

        # Second pass: add unconsumed episodic turns
        for epi_rank, turn in enumerate(episodic_items, 1):
            if turn.turn_id in consumed_turn_ids:
                continue

            cluster_id = f"cluster_{turn.turn_id}"
            epi_score = w_epi / (k_rrf + epi_rank)

            clusters[cluster_id] = {
                "cluster_id": cluster_id,
                "primary_text": turn.formatted_text or turn.content,
                "timestamp": turn.timestamp,
                "session_id": turn.session_id,
                "fused_score": epi_score,
                "channels": ["episodic"],
                "episodic_turn": turn,
                "semantic_memory": None,
                "graph_triples": [],
                "citations": [turn.turn_id],
                "metadata": {"linked_via": "standalone_episodic"}
            }

        # Third pass: link or add graph triples
        for graph_rank, triple in enumerate(graph_items, 1):
            graph_score = w_graph / (k_rrf + graph_rank)
            linked = False

            # Check if triple links to an existing semantic memory
            if triple.memory_id:
                target_cluster_id = f"cluster_{triple.memory_id}"
                if target_cluster_id in clusters:
                    clusters[target_cluster_id]["graph_triples"].append(triple)
                    clusters[target_cluster_id]["fused_score"] += graph_score
                    if "graph" not in clusters[target_cluster_id]["channels"]:
                        clusters[target_cluster_id]["channels"].append("graph")
                    linked = True

            # If not linked to an existing memory, create a standalone graph cluster
            if not linked:
                c_id = f"cluster_graph_{triple.triple_id}"
                clusters[c_id] = {
                    "cluster_id": c_id,
                    "primary_text": triple.formatted_statement + (f" (evidence: \"{triple.evidence}\")" if triple.evidence else ""),
                    "timestamp": None,
                    "session_id": None,
                    "fused_score": graph_score,
                    "channels": ["graph"],
                    "episodic_turn": None,
                    "semantic_memory": None,
                    "graph_triples": [triple],
                    "citations": [triple.memory_id] if triple.memory_id else [f"graph_{triple.triple_id}"],
                    "metadata": {"linked_via": "standalone_graph"}
                }

        # 4. Diversity Filtering and Ranking
        # Sort all clusters by fused_score descending
        sorted_clusters = sorted(
            clusters.values(),
            key=lambda c: c["fused_score"],
            reverse=True
        )

        results: list[UnifiedEvidenceItem] = []
        session_counts: dict[str, int] = {}
        max_per_session = self.config.max_items_per_session

        for cl in sorted_clusters:
            sess = cl.get("session_id")

            # Apply session diversity cap if session_id is identifiable
            if sess and max_per_session > 0:
                current_count = session_counts.get(sess, 0)
                if current_count >= max_per_session:
                    continue
                session_counts[sess] = current_count + 1

            results.append(UnifiedEvidenceItem(
                cluster_id=cl["cluster_id"],
                primary_text=cl["primary_text"],
                timestamp=cl["timestamp"],
                session_id=sess,
                fused_score=cl["fused_score"],
                channels=cl["channels"],
                episodic_turn=cl["episodic_turn"],
                semantic_memory=cl["semantic_memory"],
                graph_triples=cl["graph_triples"],
                citations=cl["citations"],
                metadata=cl["metadata"]
            ))

            if len(results) >= top_k:
                break

        return results

