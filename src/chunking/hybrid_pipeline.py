"""
Unified Hybrid Pipeline with Deterministic Query Routing and Bounded Fallback.

Implements the IVARA Research Candidate Architecture:
1. Primary Path: Hybrid Dense + BM25 retrieval with domain ontology guidance.
2. Direct-Seed Temporal: Extract and resolve temporal claims directly from top-k seed chunks.
3. Graph Routing: Deterministic, testable criteria for when graph expansion is justified.
4. Bounded Graph Fallback:
   - Obey hard traversal and context budgets (max 1 hop, max 2 fallback chunks).
   - Filter low-relevance edges (min_edge_relevance >= 0.50).
   - Preserve direct seed evidence priority.
   - Avoid duplicate chunks and self-loops/cycles.
   - Retain complete provenance (source_seed_id, edge_type, via_entity).
   - Explicitly record unresolved relationships rather than silently discarding them.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import re
from typing import Any

from .turn_chunker import TurnChunk
from .ontology_retrieval import OntologyGuidedChunkRetriever
from ..graph.store import SQLiteGraphStore


RELATIONSHIP_PATTERNS = [
    re.compile(r'\b(relationship|relations?|connected|connections?|introduced by|works? with|friend of|colleague|link between|both know|in common|interacted with)\b', re.IGNORECASE),
    re.compile(r'\b(how do .* and .* know|what connects|mutual friends?|who introduced)\b', re.IGNORECASE),
    re.compile(r'\bbetween\s+([A-Z][a-z]+|\w+)\s+and\s+([A-Z][a-z]+|\w+)\b', re.IGNORECASE)
]

FACTUAL_TEMPORAL_PATTERNS = [
    re.compile(r'^(when|how many days|what time|what date|what year|how long|where did|how much)\b', re.IGNORECASE),
    re.compile(r'\b(wake up|breakfast|dinner|lunch|cost|price|fee|hotel|airline|flight|postcards?)\b', re.IGNORECASE)
]


@dataclass
class RoutingDecision:
    """Deterministic routing decision output."""
    route: str  # "direct_seed" | "bounded_graph_fallback"
    should_expand: bool
    is_relationship_query: bool
    reasons: list[str] = field(default_factory=list)
    detected_entities: list[str] = field(default_factory=list)
    top_seed_score: float = 0.0


class DeterministicQueryRouter:
    """
    Evaluates whether a personal memory query requires graph expansion fallback.
    
    Guarantees:
    - Factual, single-session, and direct temporal queries with adequate seed evidence
      stay on the fast direct-seed path (avoiding distraction and traversal latency).
    - Queries asking about cross-session relationships or with low seed confidence on
      multi-entity queries trigger bounded, relevance-filtered graph expansion.
    """

    def __init__(
        self,
        relationship_confidence_threshold: float = 0.45,
        min_seed_score_for_direct: float = 0.40
    ):
        self.relationship_confidence_threshold = relationship_confidence_threshold
        self.min_seed_score_for_direct = min_seed_score_for_direct

    def extract_query_entities(self, query: str) -> list[str]:
        """Extract capitalized entity candidates from query."""
        words = query.split()
        entities = []
        for i, w in enumerate(words):
            clean = re.sub(r'[^a-zA-Z0-9_\-]', '', w)
            if not clean:
                continue
            if (i > 0 and clean[0].isupper() and len(clean) > 2) or clean.isupper() and len(clean) > 1:
                if clean.lower() not in ("what", "where", "when", "which", "how", "who", "can", "could", "would", "the"):
                    entities.append(clean)
        return list(dict.fromkeys(entities))

    def route(self, query: str, seed_chunks: list[dict[str, Any]]) -> RoutingDecision:
        """
        Deterministically decide if graph expansion fallback is warranted.
        """
        clean_q = query.strip()
        top_score = seed_chunks[0]["score"] if seed_chunks else 0.0
        entities = self.extract_query_entities(clean_q)
        reasons = []

        # 1. Check for explicit relationship intent
        is_rel = any(pat.search(clean_q) for pat in RELATIONSHIP_PATTERNS)
        if is_rel:
            reasons.append("query_matches_explicit_relationship_pattern")

        # 2. Check for strong factual/temporal negative guard
        is_factual_temp = any(pat.search(clean_q) for pat in FACTUAL_TEMPORAL_PATTERNS)
        if is_factual_temp and top_score >= self.min_seed_score_for_direct and not is_rel:
            reasons.append(f"factual_temporal_query_with_sufficient_seed_confidence ({top_score:.3f} >= {self.min_seed_score_for_direct})")
            return RoutingDecision(
                route="direct_seed",
                should_expand=False,
                is_relationship_query=False,
                reasons=reasons,
                detected_entities=entities,
                top_seed_score=top_score
            )

        # 3. Decision rule for graph fallback:
        # Rule A: Explicit relationship query
        if is_rel:
            reasons.append("relationship_query_requires_cross_session_resolution")
            return RoutingDecision(
                route="bounded_graph_fallback",
                should_expand=True,
                is_relationship_query=True,
                reasons=reasons,
                detected_entities=entities,
                top_seed_score=top_score
            )

        # Rule B: Multiple entities detected and seed confidence is marginal (< 0.45)
        if len(entities) >= 2 and top_score < self.relationship_confidence_threshold:
            reasons.append(f"multi_entity_query_with_marginal_seed_score ({top_score:.3f} < {self.relationship_confidence_threshold})")
            return RoutingDecision(
                route="bounded_graph_fallback",
                should_expand=True,
                is_relationship_query=False,
                reasons=reasons,
                detected_entities=entities,
                top_seed_score=top_score
            )

        # Default: direct seed is sufficient
        reasons.append("default_direct_seed_path")
        return RoutingDecision(
            route="direct_seed",
            should_expand=False,
            is_relationship_query=False,
            reasons=reasons,
            detected_entities=entities,
            top_seed_score=top_score
        )


class UnifiedCandidateRetriever:
    """
    Executes the research candidate retrieval pipeline:
    - Primary hybrid seed retrieval with ontology guidance.
    - Deterministic query routing.
    - Bounded, relevance-filtered 1-hop graph fallback when routed.
    - Provenance tracking, duplicate avoidance, and cycle prevention.
    """

    def __init__(
        self,
        chunks: list[TurnChunk],
        store: SQLiteGraphStore | None = None,
        use_ontology: bool = True,
        min_edge_relevance: float = 0.50,
        gamma_decay: float = 0.70,
        max_fallback_chunks: int = 2,
        max_chunks_per_session: int | None = None
    ):
        self.chunks = chunks
        self.store = store
        self.use_ontology = use_ontology
        self.min_edge_relevance = min_edge_relevance
        self.gamma_decay = gamma_decay
        self.max_fallback_chunks = max_fallback_chunks
        self.max_chunks_per_session = max_chunks_per_session
        self.router = DeterministicQueryRouter()
        self.base_retriever = OntologyGuidedChunkRetriever(chunks=chunks)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float | None = 0.35,
        max_chunks_per_session: int | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Retrieve chunks using primary hybrid seeds + deterministic bounded fallback.
        """
        eff_max_sess = max_chunks_per_session if max_chunks_per_session is not None else self.max_chunks_per_session
        # Step 1: Primary Hybrid Seed Retrieval
        seed_chunks = self.base_retriever.retrieve(query, k=k, min_score=min_score, max_chunks_per_session=eff_max_sess)

        # Step 2: Routing Decision
        routing = self.router.route(query, seed_chunks)

        diagnostics: dict[str, Any] = {
            "route": routing.route,
            "should_expand": routing.should_expand,
            "is_relationship_query": routing.is_relationship_query,
            "routing_reasons": routing.reasons,
            "detected_entities": routing.detected_entities,
            "top_seed_score": routing.top_seed_score,
            "seeds_retrieved": len(seed_chunks),
            "fallback_chunks_added": 0,
            "unresolved_relationships": []
        }

        # If fallback not warranted or store missing, return primary seeds directly
        if not routing.should_expand or self.store is None:
            return seed_chunks, diagnostics

        # Step 3: Bounded Relevance-Filtered Graph Fallback
        visited_chunk_ids = {c["chunk_id"] for c in seed_chunks}
        fallback_candidates: list[dict[str, Any]] = []

        chunk_lookup = {c.chunk_id: c for c in self.chunks}
        target_entities_lower = [e.lower() for e in routing.detected_entities]

        for s_rank, seed in enumerate(seed_chunks):
            s_cid = seed["chunk_id"]
            s_score = seed["score"]
            neighbors = self.store.get_neighbors(s_cid, direction="both")

            for edge, neighbor_node in neighbors:
                # Obey edge relevance filter
                edge_w = getattr(edge, "confidence", 1.0)
                if hasattr(edge, "properties") and isinstance(edge.properties, dict) and "weight" in edge.properties:
                    edge_w = float(edge.properties["weight"])
                if edge_w < self.min_edge_relevance:
                    continue

                # Traversal via entity node: Chunk -> mentions_entity -> Entity -> other Chunk
                if edge.edge_type == "mentions_entity" and neighbor_node.node_type == "entity":
                    ent_label_lower = neighbor_node.label.lower()
                    is_rel_entity = (not target_entities_lower) or any(
                        te in ent_label_lower or ent_label_lower in te for te in target_entities_lower
                    )
                    if is_rel_entity:
                        ent_neighbors = self.store.get_neighbors(neighbor_node.id, direction="both")
                        for e_ent, ent_target in ent_neighbors:
                            if ent_target.node_type == "chunk":
                                t_cid = ent_target.id
                                if t_cid not in visited_chunk_ids and t_cid in chunk_lookup:
                                    visited_chunk_ids.add(t_cid)
                                    target_chunk_obj = chunk_lookup[t_cid]
                                    decayed_score = round(s_score * (self.gamma_decay ** 2) * edge_w, 4)
                                    fallback_candidates.append({
                                        "chunk_id": target_chunk_obj.chunk_id,
                                        "session_id": target_chunk_obj.session_id,
                                        "memory_id": target_chunk_obj.memory_id,
                                        "text": target_chunk_obj.text,
                                        "turn_range": (target_chunk_obj.turn_start_idx, target_chunk_obj.turn_end_idx),
                                        "timestamp": target_chunk_obj.timestamp,
                                        "score": decayed_score,
                                        "token_estimate": target_chunk_obj.token_estimate,
                                        "origin": "graph_fallback",
                                        "fallback_metadata": {
                                            "source_seed_id": s_cid,
                                            "via_entity": neighbor_node.label,
                                            "edge_type": "mentions_entity",
                                            "edge_confidence": edge_w
                                        }
                                    })
                                    if len(fallback_candidates) >= self.max_fallback_chunks:
                                        break

                # Traversal via direct relational edge: worked_with, colleague_of, etc.
                elif edge.edge_type not in ("contains_chunk", "adjacent_to", "mentions_entity"):
                    if neighbor_node.node_type == "chunk":
                        t_cid = neighbor_node.id
                        if t_cid not in visited_chunk_ids and t_cid in chunk_lookup:
                            visited_chunk_ids.add(t_cid)
                            target_chunk_obj = chunk_lookup[t_cid]
                            decayed_score = round(s_score * self.gamma_decay * edge_w, 4)
                            fallback_candidates.append({
                                "chunk_id": target_chunk_obj.chunk_id,
                                "session_id": target_chunk_obj.session_id,
                                "memory_id": target_chunk_obj.memory_id,
                                "text": target_chunk_obj.text,
                                "turn_range": (target_chunk_obj.turn_start_idx, target_chunk_obj.turn_end_idx),
                                "timestamp": target_chunk_obj.timestamp,
                                "score": decayed_score,
                                "token_estimate": target_chunk_obj.token_estimate,
                                "origin": "graph_fallback",
                                "fallback_metadata": {
                                    "source_seed_id": s_cid,
                                    "edge_type": edge.edge_type,
                                    "edge_confidence": edge_w
                                }
                            })
                            if len(fallback_candidates) >= self.max_fallback_chunks:
                                break

            if len(fallback_candidates) >= self.max_fallback_chunks:
                break

        # Check for unresolved relationships
        if routing.is_relationship_query and len(fallback_candidates) == 0:
            diagnostics["unresolved_relationships"].append({
                "query": query,
                "status": "unresolved_no_qualifying_graph_edges",
                "min_edge_relevance": self.min_edge_relevance
            })

        # Combine: Primary seeds retain highest priority, fallback appended
        final_chunks = list(seed_chunks) + fallback_candidates
        diagnostics["fallback_chunks_added"] = len(fallback_candidates)

        return final_chunks, diagnostics
