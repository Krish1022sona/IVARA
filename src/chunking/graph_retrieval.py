"""
Turn-Aware Graph Construction and Traversal Retrieval for IVARA.

Implements B-Chunked + Graph:
1. TurnGraphBuilder: populates SQLiteGraphStore directly from TurnChunk instances,
   creating ChunkNodes, EpisodeNodes, adjacent_to edges, EntityNodes, and mentions_entity edges.
2. TurnGraphRetriever: runs relevance-guided graph traversal over TurnChunks and fuses
   hop-decayed graph scores to recover multi-hop and adjacent evidence.
"""

from collections import defaultdict
import re
from typing import Any
import numpy as np

from .turn_chunker import TurnChunk
from .ontology_retrieval import OntologyGuidedChunkRetriever
from .retrieval import select_session_mmr_chunks
from ..graph.schema import GraphNode, GraphEdge, EntityNode, EpisodeNode
from ..graph.store import SQLiteGraphStore
from ..graph.traversal import traverse_relevance_guided
from ..ontology.vocabulary import load_vocabulary, normalize_name


class TurnGraphBuilder:
    """
    Populates an isolated SQLiteGraphStore from TurnChunk units and canonical entities.
    """

    def __init__(self, store: SQLiteGraphStore):
        self.store = store
        self.vocabulary = load_vocabulary()

    def extract_chunk_entities(self, text: str) -> list[str]:
        """
        Extract candidate entities (proper nouns, alphanumeric codes, and ontology terms).
        """
        entities = set()
        # Capitalized phrases and alphanumeric codes
        words = text.split()
        for i, w in enumerate(words):
            w_clean = re.sub(r'[^a-zA-Z0-9_\-]', '', w)
            if not w_clean:
                continue
            if (i > 0 and w_clean[0].isupper() and len(w_clean) > 2) or re.search(r'[A-Z0-9]+-[A-Z0-9]+', w_clean):
                entities.add(w_clean)

        # Domain terms from ontology
        text_lower = text.lower()
        types_dict = self.vocabulary.get("types", {})
        for type_name, type_info in types_dict.items():
            for alias in type_info.get("aliases", []):
                if len(alias) > 3 and alias.lower() in text_lower:
                    entities.add(alias)

        return list(entities)

    def build_from_chunks(self, chunks: list[TurnChunk]) -> dict[str, int]:
        """
        Build the graph directly from TurnChunk instances.
        """
        sessions_seen: set[str] = set()
        chunks_by_session: dict[str, list[TurnChunk]] = defaultdict(list)
        entity_id_map: dict[str, str] = {}

        nodes_to_insert: dict[str, GraphNode] = {}
        edges_to_insert: dict[str, GraphEdge] = {}

        for chunk in chunks:
            chunks_by_session[chunk.session_id].append(chunk)

            # 1. Chunk Node
            c_node = GraphNode(
                id=chunk.chunk_id,
                node_type="chunk",
                label=f"Chunk {chunk.chunk_id}",
                properties={
                    "text": chunk.text,
                    "session_id": chunk.session_id,
                    "memory_id": chunk.memory_id,
                    "timestamp": chunk.timestamp,
                    "turn_start": chunk.turn_start_idx,
                    "turn_end": chunk.turn_end_idx,
                    "token_estimate": chunk.token_estimate
                }
            )
            nodes_to_insert[c_node.id] = c_node

            # 2. Episode / Session Node
            if chunk.session_id not in sessions_seen:
                sessions_seen.add(chunk.session_id)
                ep_node = EpisodeNode(
                    id=f"sess_{chunk.session_id}",
                    title=f"Session {chunk.session_id}",
                    timestamp=chunk.timestamp,
                    source=f"session_{chunk.session_id}"
                )
                ep_graph_node = ep_node.to_graph_node()
                nodes_to_insert[ep_graph_node.id] = ep_graph_node

            # Edge: Session -> Chunk (contains_chunk)
            e_sess = GraphEdge(
                id=f"edge_sess_chunk_{chunk.session_id}_{chunk.chunk_id}",
                source_id=f"sess_{chunk.session_id}",
                target_id=chunk.chunk_id,
                edge_type="contains_chunk",
                origin="structural",
                confidence=1.0,
                timestamp=chunk.timestamp
            )
            edges_to_insert[e_sess.id] = e_sess

            # 3. Entity extraction and linking
            chunk_entities = self.extract_chunk_entities(chunk.text)
            for ent_name in chunk_entities:
                ent_norm = ent_name.lower()
                if ent_norm not in entity_id_map:
                    ent_id = f"ent_{re.sub(r'[^a-zA-Z0-9_]', '_', ent_norm)}"
                    ent_node = GraphNode(
                        id=ent_id,
                        node_type="entity",
                        label=ent_name,
                        properties={"canonical_name": ent_name}
                    )
                    nodes_to_insert[ent_id] = ent_node
                    entity_id_map[ent_norm] = ent_id

                target_ent_id = entity_id_map[ent_norm]

                # Edge: Chunk -> Entity (mentions_entity)
                e_ment = GraphEdge(
                    id=f"edge_mentions_{chunk.chunk_id}_{target_ent_id}",
                    source_id=chunk.chunk_id,
                    target_id=target_ent_id,
                    edge_type="mentions_entity",
                    origin="explicit",
                    confidence=1.0,
                    chunk_id=chunk.chunk_id,
                    memory_id=chunk.memory_id,
                    timestamp=chunk.timestamp
                )
                edges_to_insert[e_ment.id] = e_ment

        # 4. Intra-session sequential chunk adjacency (adjacent_to)
        for sess_id, s_chunks in chunks_by_session.items():
            s_chunks.sort(key=lambda c: c.turn_start_idx)
            for i in range(len(s_chunks) - 1):
                c_prev = s_chunks[i]
                c_next = s_chunks[i + 1]
                e_adj = GraphEdge(
                    id=f"edge_adj_{c_prev.chunk_id}_{c_next.chunk_id}",
                    source_id=c_prev.chunk_id,
                    target_id=c_next.chunk_id,
                    edge_type="adjacent_to",
                    origin="structural",
                    confidence=1.0,
                    timestamp=c_prev.timestamp
                )
                edges_to_insert[e_adj.id] = e_adj

        # 5. Extract and resolve temporal claims across chunks
        from ..temporal.claim import build_claims_from_text
        from ..temporal.supersession import TemporalSupersessionEngine
        from ..graph.schema import TemporalClaim

        all_extracted_claims: list[TemporalClaim] = []
        for chunk in chunks:
            c_claims = build_claims_from_text(
                text=chunk.text,
                session_id=chunk.session_id,
                chunk_id=chunk.chunk_id,
                recorded_at=chunk.timestamp
            )
            all_extracted_claims.extend(c_claims)

        # Merge with existing claims in store for cross-batch supersession
        existing_claims: list[TemporalClaim] = []
        if hasattr(self.store, "get_nodes_by_type"):
            for cn in self.store.get_nodes_by_type("claim"):
                p = cn.properties
                existing_claims.append(TemporalClaim(
                    claim_id=cn.id,
                    subject=p.get("subject", ""),
                    predicate=p.get("predicate", ""),
                    object=p.get("object", ""),
                    valid_from=p.get("valid_from"),
                    valid_to=p.get("valid_to"),
                    recorded_at=p.get("recorded_at"),
                    source_chunk_id=p.get("source_chunk_id"),
                    is_explicit=p.get("is_explicit", True),
                    confidence=p.get("confidence", 1.0),
                    supersedes_claim_id=p.get("supersedes_claim_id"),
                    evidence=p.get("evidence"),
                    properties=dict(p)
                ))

        claims_by_id = {c.claim_id: c for c in existing_claims}
        for c in all_extracted_claims:
            claims_by_id[c.claim_id] = c
        combined_claims = list(claims_by_id.values())

        if combined_claims:
            engine = TemporalSupersessionEngine()
            processed_claims = engine.process_claims(combined_claims)

            for claim in processed_claims:
                claim_node = claim.to_graph_node()
                nodes_to_insert[claim_node.id] = claim_node

                # Provenance edge: chunk -> claim
                if claim.source_chunk_id:
                    e_claim = GraphEdge(
                        id=f"edge_claim_{claim.source_chunk_id}_{claim.claim_id}",
                        source_id=claim.source_chunk_id,
                        target_id=claim.claim_id,
                        edge_type="has_claim",
                        origin="explicit" if claim.is_explicit else "inferred",
                        confidence=claim.confidence,
                        chunk_id=claim.source_chunk_id,
                        timestamp=claim.valid_from or claim.recorded_at,
                        properties={
                            "valid_from": claim.valid_from,
                            "valid_to": claim.valid_to,
                            "recorded_at": claim.recorded_at
                        }
                    )
                    edges_to_insert[e_claim.id] = e_claim

                # Directed supersedes edge: C_new -> C_old
                if claim.supersedes_claim_id:
                    e_sup = GraphEdge(
                        id=f"edge_sup_{claim.claim_id}_{claim.supersedes_claim_id}",
                        source_id=claim.claim_id,
                        target_id=claim.supersedes_claim_id,
                        edge_type="supersedes",
                        origin="explicit" if claim.is_explicit else "inferred",
                        confidence=claim.confidence,
                        evidence=claim.evidence,
                        timestamp=claim.valid_from or claim.recorded_at,
                        properties={
                            "valid_from": claim.valid_from,
                            "valid_to": claim.valid_to,
                            "recorded_at": claim.recorded_at,
                            "reason": claim.properties.get("supersession_reason")
                        }
                    )
                    edges_to_insert[e_sup.id] = e_sup

        # 6. Extract and build temporal event nodes and relationships
        from ..temporal.event_extractor import TemporalEventExtractor
        from ..temporal.relationship_builder import TemporalRelationshipBuilder

        event_extractor = TemporalEventExtractor()
        all_events = []
        for chunk in chunks:
            c_events = event_extractor.extract_events_from_chunk(
                text=chunk.text,
                session_id=chunk.session_id,
                chunk_id=chunk.chunk_id,
                recorded_at=chunk.timestamp
            )
            all_events.extend(c_events)

        unique_events = {e.event_id: e for e in all_events}
        for evt in unique_events.values():
            evt_node = evt.to_graph_node()
            nodes_to_insert[evt_node.id] = evt_node
            # Link chunk -> event (contains_event)
            if evt.turn_id:
                e_link = GraphEdge(
                    id=f"edge_has_evt_{evt.turn_id}_{evt.event_id}",
                    source_id=evt.turn_id,
                    target_id=evt.event_id,
                    edge_type="contains_event",
                    origin="explicit",
                    confidence=evt.confidence,
                    chunk_id=evt.turn_id,
                    timestamp=evt.valid_from or evt.recorded_at
                )
                edges_to_insert[e_link.id] = e_link

            # Link event -> entity if entity node exists
            for ent_id in evt.entity_ids:
                if ent_id in nodes_to_insert:
                    e_ent = GraphEdge(
                        id=f"edge_evt_ent_{evt.event_id}_{ent_id}",
                        source_id=evt.event_id,
                        target_id=ent_id,
                        edge_type="mentions_entity",
                        origin="explicit",
                        confidence=evt.confidence,
                        timestamp=evt.valid_from or evt.recorded_at
                    )
                    edges_to_insert[e_ent.id] = e_ent

        rel_builder = TemporalRelationshipBuilder()
        temporal_edges = rel_builder.build_relationships(list(unique_events.values()))
        for tedge in temporal_edges:
            g_edge = tedge.to_graph_edge()
            edges_to_insert[g_edge.id] = g_edge

        node_list = list(nodes_to_insert.values())
        edge_list = list(edges_to_insert.values())

        if hasattr(self.store, "batch_upsert"):
            self.store.batch_upsert(nodes=node_list, edges=edge_list)
        elif hasattr(self.store, "upsert_nodes_batch"):
            self.store.upsert_nodes_batch(node_list)
            self.store.upsert_edges_batch(edge_list)
        else:
            for n in node_list:
                self.store.upsert_node(n)
            for e in edge_list:
                self.store.upsert_edge(e)

        return {
            "nodes": len(nodes_to_insert),
            "edges": len(edges_to_insert),
            "claim_nodes": len(combined_claims),
            "chunks": len(chunks)
        }


def should_expand_graph(
    query: str,
    seed_chunks: list[dict[str, Any]],
    query_concepts: dict[str, list[str]] | None = None
) -> tuple[bool, str]:
    """
    Determine whether graph expansion should be executed for the given query.

    Returns:
        (should_expand: bool, reason: str)
    """
    q_clean = query.strip()
    q_lower = q_clean.lower()

    concepts = query_concepts or {}
    entities = concepts.get("entities", [])
    predicates = concepts.get("predicates", [])

    # 1. Multi-entity query (requires connecting multiple named entities)
    if len(entities) >= 2:
        return True, f"Multi-entity query ({len(entities)} entities detected: {entities})"

    # 2. Relational, collaborative, or transition keywords
    relational_keywords = [
        "worked with", "collaborated", "colleague", "reports to", "reporting to",
        "manager", "mentor", "friend", "family", "parent", "child",
        "assigned to", "works at", "works for", "employed by",
        "moved to", "relocated", "transitioned", "switched", "changed",
        "before", "after", "later", "what happened next", "then",
        "connect", "relationship", "association", "linked"
    ]
    for kw in relational_keywords:
        if kw in q_lower:
            return True, f"Relational/transition keyword matched: '{kw}'"

    if len(predicates) >= 1:
        return True, f"Ontology predicate matched: {predicates}"

    # 3. Conversational follow-up or outcome queries
    followup_keywords = [
        "coupon", "discount", "what percentage", "how much", "saved",
        "what did they say", "what happened next", "warranty code", "registration code",
        "confirmation code", "what was the code",
        "follow up", "follow-up", "outcome"
    ]
    for kw in followup_keywords:
        if kw in q_lower:
            return True, f"Conversational follow-up signal matched: '{kw}'"

    # 4. Incomplete seed turn: top seed chunk ends with dialogue question/prompt
    if seed_chunks:
        top_text = seed_chunks[0].get("text", "")
        last_line = top_text.strip().split("\n")[-1] if top_text.strip() else ""
        if "?" in last_line:
            return True, "Top seed chunk ends with unanswered question; adjacent turn needed"

    return False, "Ordinary factual query; graph expansion bypassed to prevent noise"


class TurnGraphRetriever:
    """
    B-Chunked + Graph retriever:
    Combines unified hybrid seed retrieval (dense + BM25 + ontology) with
    query-conditioned, bounded graph expansion (1-hop only, max 3 neighbor chunks, gamma=0.75 decay).
    """

    def __init__(
        self,
        chunks: list[TurnChunk],
        store: SQLiteGraphStore,
        vector_weight: float = 0.55,
        bm25_weight: float = 0.30,
        ontology_weight: float = 0.15,
        use_ontology: bool = True,
        max_graph_chunks: int = 3,
        gamma_decay: float = 0.75,
        k_rrf: int = 60,
        min_edge_relevance: float = 0.0
    ):
        self.chunks = chunks
        self.store = store
        self.use_ontology = use_ontology
        self.max_graph_chunks = max_graph_chunks
        self.gamma_decay = gamma_decay
        self.k_rrf = k_rrf
        self.min_edge_relevance = min_edge_relevance
        self.chunk_by_id = {c.chunk_id: c for c in chunks}

        # Unified hybrid seed retriever from Step 2
        from .retrieval import ChunkRetriever
        self.seed_retriever = ChunkRetriever(
            chunks=chunks,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            ontology_weight=ontology_weight,
            use_ontology=use_ontology
        )
        self.ontology_retriever = self.seed_retriever

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float | None = None,
        max_traversal_depth: int = 1,
        max_traversal_nodes: int = 15,
        use_session_mmr: bool = True,
        mmr_lambda: float = 0.25,
        mmr_pool_size: int = 15
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Execute query-conditioned, bounded graph chunk retrieval with Session-Aware MMR.
        Returns:
            (retrieved_chunks, graph_diagnostics)
        """
        clean_query = query.strip()
        if not clean_query or not self.chunks:
            return [], {}

        # 1. Direct Hybrid (+ optional Ontology) Candidate Generation using unified ChunkRetriever
        search_k = max(k * 3, mmr_pool_size) if use_session_mmr else max(k * 2, 10)
        seed_candidates = self.seed_retriever.retrieve(
            clean_query, k=search_k, min_score=None, use_session_mmr=False
        )
        if not seed_candidates:
            return [], {
                "expansion_executed": False,
                "decision_reason": "no_seed_candidates",
                "nodes_expanded": 0,
                "edges_traversed": 0,
                "extra_chunks_introduced": 0
            }

        # Extract concepts from seed retriever or query
        query_concepts = seed_candidates[0].get("detected_concepts", {})
        if not query_concepts:
            from .ontology_retrieval import extract_query_concepts
            query_concepts = extract_query_concepts(clean_query)

        # 2. Query Conditioning Decision
        should_expand, reason = should_expand_graph(clean_query, seed_candidates, query_concepts)

        diagnostics: dict[str, Any] = {
            "expansion_executed": should_expand,
            "decision_reason": reason,
            "seed_count": len(seed_candidates),
            "extra_chunks_introduced": 0,
            "nodes_expanded": 0,
            "edges_traversed": 0,
            "traversed_details": []
        }

        # If expansion is not needed or store has no nodes, return clean seed candidates
        if not should_expand or self.store is None or self.store.count_nodes() == 0:
            filtered = [c for c in seed_candidates if min_score is None or c["score"] >= min_score]
            if use_session_mmr and len(filtered) > k:
                pool = filtered[:max(k, mmr_pool_size)]
                return select_session_mmr_chunks(pool, k=k, lambda_penalty=mmr_lambda), diagnostics
            return filtered[:k], diagnostics

        # 3. Bounded 1-Hop Graph Expansion from Top Seed Chunks
        top_seeds = seed_candidates[:3]
        seed_cids = {c["chunk_id"] for c in seed_candidates}

        graph_candidates: dict[str, dict[str, Any]] = {}
        target_query_entities = [e.lower() for e in query_concepts.get("entities", [])]

        nodes_expanded = 0
        edges_traversed = 0

        for seed in top_seeds:
            s_cid = seed["chunk_id"]
            s_score = seed["score"]
            nodes_expanded += 1

            # Get 1-hop neighbors from SQLiteGraphStore
            neighbors = self.store.get_neighbors(s_cid, direction="both")
            for edge, neighbor_node in neighbors:
                edges_traversed += 1

                # Safe Edge Filter 0: edge relevance threshold (Prototype B)
                if self.min_edge_relevance > 0.0:
                    edge_w = getattr(edge, "confidence", 1.0)
                    if hasattr(edge, "properties") and isinstance(edge.properties, dict) and "weight" in edge.properties:
                        edge_w = float(edge.properties["weight"])
                    if edge_w < self.min_edge_relevance:
                        continue
                    if edge.edge_type == "adjacent_to":
                        target_text = neighbor_node.properties.get("text", "").lower()
                        q_words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', clean_query)]
                        if q_words:
                            overlap = sum(1 for w in set(q_words) if w in target_text)
                            if overlap == 0 and (len(target_text.split()) < 10 or "ok" in target_text or "yes" in target_text or "sure" in target_text):
                                continue

                # Safe Edge Filter 1: adjacent_to (within same session)
                if edge.edge_type == "adjacent_to":
                    target_chunk_id = neighbor_node.id
                    decayed_score = s_score * self.gamma_decay
                    if target_chunk_id not in graph_candidates or decayed_score > graph_candidates[target_chunk_id]["decayed_score"]:
                        graph_candidates[target_chunk_id] = {
                            "decayed_score": decayed_score,
                            "source_seed_id": s_cid,
                            "edge_type": "adjacent_to",
                            "node": neighbor_node
                        }
                        diagnostics["traversed_details"].append({
                            "from": s_cid,
                            "to": target_chunk_id,
                            "edge": "adjacent_to",
                            "decayed_score": round(decayed_score, 4)
                        })

                # Safe Edge Filter 2: mentions_entity -> entity -> other chunk
                # ONLY if entity is explicitly named in query
                elif edge.edge_type == "mentions_entity" and neighbor_node.node_type == "entity":
                    ent_label_lower = neighbor_node.label.lower()
                    ent_canon_lower = neighbor_node.properties.get("canonical_name", "").lower()
                    is_query_entity = any(
                        qe in ent_label_lower or qe in ent_canon_lower or ent_label_lower in qe
                        for qe in target_query_entities
                    )

                    if is_query_entity:
                        ent_neighbors = self.store.get_neighbors(neighbor_node.id, direction="both")
                        for e_ent, ent_target in ent_neighbors:
                            edges_traversed += 1
                            if ent_target.node_type == "chunk" and ent_target.id != s_cid:
                                decayed_score = s_score * (self.gamma_decay ** 2)
                                if ent_target.id not in graph_candidates or decayed_score > graph_candidates[ent_target.id]["decayed_score"]:
                                    graph_candidates[ent_target.id] = {
                                        "decayed_score": decayed_score,
                                        "source_seed_id": s_cid,
                                        "edge_type": f"mentions_entity:{neighbor_node.label}",
                                        "node": ent_target
                                    }
                                    diagnostics["traversed_details"].append({
                                        "from": s_cid,
                                        "via": neighbor_node.label,
                                        "to": ent_target.id,
                                        "edge": "mentions_entity",
                                        "decayed_score": round(decayed_score, 4)
                                    })

                # Safe Edge Filter 4: contains_event -> EventNode -> TemporalEdge -> other EventNode -> other chunk
                elif edge.edge_type == "contains_event" and neighbor_node.node_type == "event":
                    evt_neighbors = self.store.get_neighbors(neighbor_node.id, direction="both")
                    for e_evt, evt_target in evt_neighbors:
                        edges_traversed += 1
                        if evt_target.node_type == "event" and e_evt.edge_type in ("OCCURRED_BEFORE", "OCCURRED_AFTER", "SUPERSEDES", "INVALIDATES"):
                            # Get chunks containing target event
                            target_evt_neighbors = self.store.get_neighbors(evt_target.id, direction="both")
                            for e_chk, chk_target in target_evt_neighbors:
                                edges_traversed += 1
                                if chk_target.node_type == "chunk" and chk_target.id != s_cid:
                                    decayed_score = s_score * (self.gamma_decay ** 2)
                                    if chk_target.id not in graph_candidates or decayed_score > graph_candidates[chk_target.id]["decayed_score"]:
                                        graph_candidates[chk_target.id] = {
                                            "decayed_score": decayed_score,
                                            "source_seed_id": s_cid,
                                            "edge_type": f"temporal:{e_evt.edge_type}",
                                            "node": chk_target
                                        }
                                        diagnostics["traversed_details"].append({
                                            "from": s_cid,
                                            "via_event": neighbor_node.label,
                                            "to": chk_target.id,
                                            "edge": e_evt.edge_type,
                                            "decayed_score": round(decayed_score, 4)
                                        })

                # Safe Edge Filter 3: semantic relational edges (worked_with, worked_on, employed_by, uses)
                elif edge.edge_type not in ("contains_chunk", "adjacent_to", "mentions_entity", "contains_event"):
                    if neighbor_node.node_type == "chunk":
                        decayed_score = s_score * self.gamma_decay
                        if neighbor_node.id not in graph_candidates or decayed_score > graph_candidates[neighbor_node.id]["decayed_score"]:
                            graph_candidates[neighbor_node.id] = {
                                "decayed_score": decayed_score,
                                "source_seed_id": s_cid,
                                "edge_type": edge.edge_type,
                                "node": neighbor_node
                            }

        # 4. Conservative Scoring & Merging
        # A. Direct candidates retain direct score + small graph confirmation bonus (if confirmed)
        merged_results: list[dict[str, Any]] = []
        for cand in seed_candidates:
            c_id = cand["chunk_id"]
            d_score = cand["score"]
            g_info = graph_candidates.get(c_id)
            if g_info:
                final_score = min(1.0, d_score + 0.10 * g_info["decayed_score"])
                cand_copy = dict(cand)
                cand_copy["score"] = round(final_score, 4)
                cand_copy["direct_score"] = round(d_score, 4)
                cand_copy["graph_score"] = round(g_info["decayed_score"], 4)
                cand_copy["graph_edge"] = g_info["edge_type"]
                merged_results.append(cand_copy)
            else:
                cand_copy = dict(cand)
                cand_copy["direct_score"] = round(d_score, 4)
                cand_copy["graph_score"] = 0.0
                merged_results.append(cand_copy)

        # B. Pure indirect graph discoveries (chunks NOT already in seed_candidates)
        # Prioritize adjacent_to edges (conversational continuity) over heuristic cross-session entity links
        pure_indirect = [
            (c_id, info)
            for c_id, info in graph_candidates.items()
            if c_id not in seed_cids
        ]
        pure_indirect.sort(
            key=lambda x: (
                0 if x[1]["edge_type"] == "adjacent_to" else 1,
                -x[1]["decayed_score"],
                x[0]
            )
        )
        # Strictly capped by max_graph_chunks (max 3 chunks)
        selected_indirect = pure_indirect[:self.max_graph_chunks]

        for c_id, info in selected_indirect:
            chunk = self.chunk_by_id.get(c_id)
            if not chunk:
                node = self.store.get_node(c_id)
                if node and node.node_type == "chunk":
                    props = node.properties
                    chunk = TurnChunk(
                        chunk_id=c_id,
                        session_id=props.get("session_id", ""),
                        memory_id=props.get("memory_id", ""),
                        turn_start_idx=props.get("turn_start", 0),
                        turn_end_idx=props.get("turn_end", 0),
                        total_session_turns=props.get("turn_end", 0) + 1,
                        timestamp=props.get("timestamp", ""),
                        parsed_datetime=None,
                        turns=[],
                        text=props.get("text", ""),
                        word_count=len(props.get("text", "").split()),
                        token_estimate=props.get("token_estimate", 100)
                    )

            if chunk:
                g_score = info["decayed_score"]
                item = {
                    "chunk": chunk,
                    "chunk_id": chunk.chunk_id,
                    "memory_id": chunk.memory_id,
                    "session_id": chunk.session_id,
                    "score": round(float(g_score), 4),
                    "direct_score": 0.0,
                    "graph_score": round(float(g_score), 4),
                    "graph_edge": info["edge_type"],
                    "turn_range": (chunk.turn_start_idx, chunk.turn_end_idx),
                    "timestamp": chunk.timestamp,
                    "text": chunk.text,
                    "token_estimate": chunk.token_estimate
                }
                merged_results.append(item)

        # 5. Deterministic Ranking
        # Tier 0: Direct evidence OR adjacent sequential conversational turn (safe intra-session context)
        # Tier 1: Pure indirect entity-linked chunks (cross-session heuristic links that must never displace direct evidence)
        def _ranking_tier(item: dict[str, Any]) -> int:
            if item.get("direct_score", 0.0) > 0.0:
                return 0
            if item.get("graph_edge") == "adjacent_to":
                return 0
            return 1

        merged_results.sort(
            key=lambda x: (
                _ranking_tier(x),
                -x["score"],
                x["turn_range"][0],
                x["chunk_id"]
            )
        )

        valid_candidates = [
            c for c in merged_results
            if min_score is None or c["score"] >= min_score
        ]

        if use_session_mmr and len(valid_candidates) > k:
            pool = valid_candidates[:max(k, mmr_pool_size)]
            final_results = select_session_mmr_chunks(pool, k=k, lambda_penalty=mmr_lambda)
        else:
            final_results = valid_candidates[:k]

        diagnostics["nodes_expanded"] = nodes_expanded
        diagnostics["edges_traversed"] = edges_traversed
        diagnostics["extra_chunks_introduced"] = len(selected_indirect)

        return final_results, diagnostics
