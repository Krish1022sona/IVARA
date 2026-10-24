"""
Memory Graph Builder for IVARA.

Constructs an isolated, persistent Memory Graph in SQLite from stored memories,
canonical entities, and validated extractions.
"""

from pathlib import Path
from typing import Any
import re

from .schema import GraphNode, GraphEdge, EntityNode, EpisodeNode
from .store import SQLiteGraphStore
from .chunker import chunk_memory
from ..ontology.vocabulary import get_canonical_predicate


class MemoryGraphBuilder:
    """
    Constructs and populates the Memory Graph from memories and entity stores.
    """

    def __init__(self, store: SQLiteGraphStore):
        self.store = store

    def build_from_records(
        self,
        memories: list[dict[str, Any]],
        entities: list[dict[str, Any]] | None = None
    ) -> dict[str, int]:
        """
        Populate the graph from memory records and canonical entities.

        Returns statistics on nodes and edges created.
        """
        entities = entities or []

        # 1. Populate Canonical Entity Nodes
        ent_id_map = {}
        for ent in entities:
            e_id = ent.get("id") or f"ent_{re.sub(r'[^a-zA-Z0-9_]', '_', ent['canonical_name'].lower())}"
            canon_name = ent.get("canonical_name", "Unknown")
            aliases = ent.get("aliases", [])
            types = ent.get("types", [])

            ent_node = EntityNode(
                id=e_id,
                canonical_name=canon_name,
                aliases=aliases,
                types=types
            )
            self.store.upsert_node(ent_node.to_graph_node())
            ent_id_map[canon_name.lower()] = e_id
            for a in aliases:
                ent_id_map[a.lower()] = e_id

        # 2. Track episodes for temporal sequencing
        episodes_by_id = {}
        episode_order = []

        # 3. Process each memory and its chunks
        for memo in memories:
            m_id = memo.get("id", "m000")
            m_text = memo.get("text", "")
            m_ts = memo.get("timestamp")
            m_src = memo.get("source", "chronicle")
            m_type = memo.get("type", "episodic")
            m_imp = memo.get("importance", 1.0)

            # A. Create Memory Container Node
            memo_node = GraphNode(
                id=m_id,
                node_type="memory",
                label=f"Memory {m_id}",
                properties={
                    "text": m_text,
                    "timestamp": m_ts,
                    "source": m_src,
                    "type": m_type,
                    "importance": m_imp
                }
            )
            self.store.upsert_node(memo_node)

            # B. Manage Episode Node if source or episode field indicates episode
            ep_raw = memo.get("episode") or (m_src if "episode" in str(m_src).lower() else None)
            ep_key = str(ep_raw) if ep_raw else None
            if ep_key and ep_key not in episodes_by_id:
                ep_node = EpisodeNode(
                    id=f"ep_{len(episodes_by_id) + 1:02d}",
                    title=str(ep_key),
                    timestamp=m_ts,
                    source=str(m_src or ep_key),
                    importance=m_imp
                )
                self.store.upsert_node(ep_node.to_graph_node())
                episodes_by_id[ep_key] = ep_node.id
                episode_order.append((ep_node.id, m_ts))

            if ep_key:
                ep_node_id = episodes_by_id[ep_key]
                # Edge: Episode -> Memory
                self.store.upsert_edge(GraphEdge(
                    id=f"edge_ep_mem_{ep_node_id}_{m_id}",
                    source_id=ep_node_id,
                    target_id=m_id,
                    edge_type="contains_memory",
                    origin="structural",
                    confidence=1.0,
                    timestamp=m_ts
                ))

            # C. Chunk Memory and insert Chunk Nodes
            chunks = chunk_memory(memo, target_words=120)
            prev_chunk_id = None

            for c in chunks:
                self.store.upsert_node(c.to_graph_node())

                # Edge: Memory -> Chunk
                self.store.upsert_edge(GraphEdge(
                    id=f"edge_mem_chunk_{m_id}_{c.id}",
                    source_id=m_id,
                    target_id=c.id,
                    edge_type="contains_chunk",
                    origin="structural",
                    confidence=1.0,
                    timestamp=m_ts
                ))

                # Edge: Sequential Chunk Ordering (c_i -> c_{i+1})
                if prev_chunk_id is not None:
                    self.store.upsert_edge(GraphEdge(
                        id=f"edge_adj_{prev_chunk_id}_{c.id}",
                        source_id=prev_chunk_id,
                        target_id=c.id,
                        edge_type="adjacent_to",
                        origin="structural",
                        confidence=1.0,
                        timestamp=m_ts
                    ))
                prev_chunk_id = c.id

                # Edges: Chunk -> Entity links
                for raw_e_id in c.entity_ids:
                    target_ent_id = ent_id_map.get(str(raw_e_id).lower(), str(raw_e_id))
                    if not self.store.get_node(target_ent_id):
                        self.store.upsert_node(GraphNode(
                            id=target_ent_id,
                            node_type="entity",
                            label=str(raw_e_id),
                            properties={"canonical_name": str(raw_e_id)}
                        ))
                        ent_id_map[str(raw_e_id).lower()] = target_ent_id

                    self.store.upsert_edge(GraphEdge(
                        id=f"edge_chunk_ent_{c.id}_{target_ent_id}",
                        source_id=c.id,
                        target_id=target_ent_id,
                        edge_type="mentions_entity",
                        origin="explicit",
                        confidence=1.0,
                        chunk_id=c.id,
                        memory_id=m_id,
                        timestamp=m_ts
                    ))

            # D. Insert Semantic Relationship Edges from extracted facts
            for rel in memo.get("relationships", []):
                pred_raw = rel.get("predicate") or rel.get("relation")
                pred = get_canonical_predicate(pred_raw) or pred_raw
                src = rel.get("source")
                tgt = rel.get("target")

                src_name = src.get("canonical_name") or src.get("mention") or src.get("id") or str(src) if isinstance(src, dict) else str(src)
                tgt_name = tgt.get("canonical_name") or tgt.get("mention") or tgt.get("id") or str(tgt) if isinstance(tgt, dict) else str(tgt)

                # Map to entity node IDs
                src_ent_id = ent_id_map.get(src_name.lower())
                tgt_ent_id = ent_id_map.get(tgt_name.lower())

                # If entity nodes don't exist yet, create lightweight entity nodes
                if not src_ent_id:
                    src_ent_id = f"ent_{re.sub(r'[^a-zA-Z0-9_]', '_', src_name.lower())}"
                    self.store.upsert_node(GraphNode(id=src_ent_id, node_type="entity", label=src_name, properties={"canonical_name": src_name}))
                    ent_id_map[src_name.lower()] = src_ent_id
                if not tgt_ent_id:
                    tgt_ent_id = f"ent_{re.sub(r'[^a-zA-Z0-9_]', '_', tgt_name.lower())}"
                    self.store.upsert_node(GraphNode(id=tgt_ent_id, node_type="entity", label=tgt_name, properties={"canonical_name": tgt_name}))
                    ent_id_map[tgt_name.lower()] = tgt_ent_id

                ev = rel.get("evidence")
                origin = rel.get("origin", "explicit")
                conf = rel.get("confidence", 1.0)

                # Determine which chunk contains this relationship's evidence
                matching_chunk_id = None
                if ev and isinstance(ev, str) and ev.strip():
                    ev_clean = ev.strip().lower()
                    for c in chunks:
                        if ev_clean in c.text.lower():
                            matching_chunk_id = c.id
                            break
                # If evidence cannot be located in any chunk, retain memory-level provenance
                # and leave matching_chunk_id as None (unresolved chunk association)

                edge_id = f"edge_rel_{m_id}_{src_ent_id}_{pred}_{tgt_ent_id}"
                self.store.upsert_edge(GraphEdge(
                    id=edge_id,
                    source_id=src_ent_id,
                    target_id=tgt_ent_id,
                    edge_type=pred,
                    origin=origin,
                    confidence=conf,
                    evidence=ev,
                    memory_id=m_id,
                    chunk_id=matching_chunk_id,
                    timestamp=m_ts
                ))

                # Only link supporting chunk to target entity if evidence was verified in that chunk
                if matching_chunk_id is not None:
                    self.store.upsert_edge(GraphEdge(
                        id=f"edge_ground_{matching_chunk_id}_{tgt_ent_id}_{pred}",
                        source_id=matching_chunk_id,
                        target_id=tgt_ent_id,
                        edge_type=f"grounds_{pred}",
                        origin=origin,
                        confidence=conf,
                        evidence=ev,
                        memory_id=m_id,
                        chunk_id=matching_chunk_id,
                        timestamp=m_ts
                    ))

        # 4. Insert Chronological Ordering Edges between Episodes using validated timestamps
        valid_episodes = []
        for ep_node_id, raw_ts in episode_order:
            parsed_t = self._parse_timestamp(raw_ts)
            if parsed_t is not None:
                valid_episodes.append((ep_node_id, parsed_t, raw_ts))

        # Sort strictly by validated chronological timestamp (independent of input order)
        valid_episodes.sort(key=lambda x: x[1])

        # Create precedes edges between strictly chronologically ordered episodes
        for idx in range(len(valid_episodes) - 1):
            ep_prev, t_prev, raw_prev = valid_episodes[idx]
            ep_next, t_next, raw_next = valid_episodes[idx + 1]
            if t_prev < t_next:
                self.store.upsert_edge(GraphEdge(
                    id=f"edge_time_order_{ep_prev}_{ep_next}",
                    source_id=ep_prev,
                    target_id=ep_next,
                    edge_type="precedes",
                    origin="structural",
                    confidence=1.0,
                    properties={"prev_time": raw_prev, "next_time": raw_next}
                ))

        return {
            "nodes_count": self.store.count_nodes(),
            "edges_count": self.store.count_edges()
        }

    def _parse_timestamp(self, ts_str: str | None) -> float | None:
        """Parse ISO dates like '2025-01-14', '2025-06-20', '2025-01' into comparable floats."""
        if not ts_str or not isinstance(ts_str, str):
            return None
        match = re.search(r'\b(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?\b', ts_str.strip())
        if match:
            year = int(match.group(1))
            month = int(match.group(2) or 1)
            day = int(match.group(3) or 1)
            try:
                from datetime import datetime
                return datetime(year, month, day).timestamp()
            except ValueError:
                return None
        return None
