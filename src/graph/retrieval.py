"""
Graph-Assisted Retrieval Engine for IVARA.

Features:
1. Hybrid Semantic Vector Search (FAISS) + Lexical Search (BM25) on chunks.
2. Configurable Reciprocal Rank Fusion (RRF) for principled score combination.
3. Relevance-Guided Graph Traversal with hop-decay discounting (preventing weak graph nodes from displacing direct evidence).
4. Chunk-level evidence context (avoids dumping full parent episodes; preserves chunk offsets and traceable provenance).
5. Accurate relationship evidence reconstruction: outputs Subject-Predicate-Object triples with verbatim evidence and memory/chunk IDs.
   Strictly filters out internal grounding and structural edges (`grounds_*`, `mentions_entity`, `adjacent_to`).
"""

import math
import re
from typing import Any
import numpy as np
import faiss

from .store import SQLiteGraphStore
from .schema import GraphNode, GraphEdge
from .traversal import traverse_relevance_guided
from ..embedding import get_embedding


class BM25LexicalRetriever:
    """Lightweight BM25 lexical retriever for exact terms, names, and identifiers."""

    def __init__(self, corpus: list[dict[str, Any]]):
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.avg_doc_len = 0.0
        self.doc_tokens = []
        self.doc_freqs = {}

        if self.doc_count > 0:
            total_len = 0
            for doc in corpus:
                tokens = self._tokenize(doc.get("text", ""))
                self.doc_tokens.append(tokens)
                total_len += len(tokens)
                for t in set(tokens):
                    self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
            self.avg_doc_len = total_len / self.doc_count

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in re.findall(r'\b[a-zA-Z0-9_\-+#]{2,}\b', text)]

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, int]]:
        q_tokens = self._tokenize(query)
        if not q_tokens or self.doc_count == 0:
            return []

        scores = []
        k1 = 1.5
        b = 0.75

        for idx, doc_toks in enumerate(self.doc_tokens):
            doc_len = len(doc_toks)
            score = 0.0
            tok_counts = {}
            for t in doc_toks:
                tok_counts[t] = tok_counts.get(t, 0) + 1

            for qt in q_tokens:
                if qt in tok_counts:
                    df = self.doc_freqs.get(qt, 1)
                    idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
                    tf = tok_counts[qt]
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / (self.avg_doc_len or 1.0))))

            if score > 0.0:
                scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]


class GraphAssistedRetriever:
    """
    Coordinates hybrid vector-lexical candidate generation, Reciprocal Rank Fusion (RRF),
    and relevance-guided graph traversal.
    """

    def __init__(
        self,
        store: SQLiteGraphStore,
        k_rrf: int = 60,
        weight_vector: float = 1.0,
        weight_lexical: float = 1.0
    ):
        self.store = store
        self.k_rrf = k_rrf
        self.weight_vector = weight_vector
        self.weight_lexical = weight_lexical

        self.chunk_nodes = self.store.get_nodes_by_type("chunk")
        self.entity_nodes = self.store.get_nodes_by_type("entity")
        self.memory_nodes = self.store.get_nodes_by_type("memory")

        self.chunk_by_id = {c.id: c for c in self.chunk_nodes}
        self.memory_by_id = {m.id: m for m in self.memory_nodes}
        self.entity_by_id = {e.id: e for e in self.entity_nodes}

        self._init_vector_index()
        self._init_lexical_index()

    def _init_vector_index(self):
        self.chunk_embeddings = None
        self.faiss_index = None

        if not self.chunk_nodes:
            return

        texts = [c.properties.get("text", "") for c in self.chunk_nodes]
        embeddings = [get_embedding(t) for t in texts]
        arr = np.array(embeddings, dtype="float32")

        if arr.ndim == 2 and arr.shape[0] > 0:
            faiss.normalize_L2(arr)
            dim = arr.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(arr)
            self.chunk_embeddings = arr

    def _init_lexical_index(self):
        corpus = [{"id": c.id, "text": c.properties.get("text", "")} for c in self.chunk_nodes]
        self.bm25 = BM25LexicalRetriever(corpus)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        use_graph_expansion: bool = True,
        mode: str = "full_graph_assisted",
        max_traversal_nodes: int = 15,
        max_traversal_depth: int = 2
    ) -> dict[str, Any]:
        """
        Execute candidate retrieval, RRF score fusion, graph expansion, and evidence assembly.
        """
        clean_query = query.strip()
        if not clean_query:
            return self._empty_response(query)

        diagnostics = {
            "mode": mode,
            "vector_candidates": [],
            "lexical_candidates": [],
            "rrf_fused_candidates": [],
            "entity_seeds": [],
            "graph_expanded_nodes": [],
            "graph_edges_traversed": [],
            "stopping_reason": "completed"
        }

        # 1. Semantic Vector Retrieval over Chunks
        vec_ranks: dict[str, int] = {}
        if self.faiss_index is not None and len(self.chunk_nodes) > 0:
            q_emb = np.array([get_embedding(clean_query)], dtype="float32")
            faiss.normalize_L2(q_emb)
            search_k = min(10, len(self.chunk_nodes))
            scores, indices = self.faiss_index.search(q_emb, search_k)

            for rank_idx, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
                if idx >= 0 and idx < len(self.chunk_nodes):
                    chunk_id = self.chunk_nodes[idx].id
                    vec_ranks[chunk_id] = rank_idx
                    diagnostics["vector_candidates"].append({
                        "chunk_id": chunk_id,
                        "rank": rank_idx,
                        "cosine_similarity": round(float(score), 4)
                    })

        # 2. Lexical Retrieval (BM25) over Chunks
        lex_ranks: dict[str, int] = {}
        if mode not in ("chunked_vector", "vector_only"):
            lexical_hits = self.bm25.search(clean_query, top_k=10)
            for rank_idx, (score, idx) in enumerate(lexical_hits, 1):
                chunk_id = self.chunk_nodes[idx].id
                lex_ranks[chunk_id] = rank_idx
                diagnostics["lexical_candidates"].append({
                    "chunk_id": chunk_id,
                    "rank": rank_idx,
                    "bm25_score": round(score, 3)
                })

        # 3. Reciprocal Rank Fusion (RRF) for Direct Evidence
        all_candidate_chunks = set(vec_ranks.keys()) | set(lex_ranks.keys())
        direct_scores: dict[str, float] = {}

        for c_id in all_candidate_chunks:
            rrf_vec = (self.weight_vector / (self.k_rrf + vec_ranks[c_id])) if c_id in vec_ranks else 0.0
            rrf_lex = (self.weight_lexical / (self.k_rrf + lex_ranks[c_id])) if c_id in lex_ranks else 0.0
            direct_scores[c_id] = rrf_vec + rrf_lex

        for c_id, score in sorted(direct_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
            diagnostics["rrf_fused_candidates"].append({
                "chunk_id": c_id,
                "rrf_score": round(score, 5)
            })

        # 4. Entity-Aware Seed Identification (Config E only)
        entity_seeds = set()
        if mode in ("full_graph", "full_graph_assisted", "entity_aware"):
            q_lower = clean_query.lower()
            for ent in self.entity_nodes:
                c_name = ent.properties.get("canonical_name", "").lower()
                aliases = [a.lower() for a in ent.properties.get("aliases", [])]
                if (c_name and c_name in q_lower) or any(a and a in q_lower for a in aliases):
                    entity_seeds.add(ent.id)
                    diagnostics["entity_seeds"].append(ent.id)

        # 5. Graph Expansion via Relevance-Guided Traversal
        # Determine traversal seeds: top RRF chunks + identified entity nodes
        top_seed_chunks = sorted(direct_scores.keys(), key=lambda x: direct_scores[x], reverse=True)[:5]
        traversal_seeds = list(set(top_seed_chunks) | entity_seeds)

        graph_scores: dict[str, float] = {}
        traversal_details = []

        if use_graph_expansion and traversal_seeds and mode not in ("chunked_vector", "chunked_hybrid"):
            traversal_res = traverse_relevance_guided(
                store=self.store,
                query=clean_query,
                seed_node_ids=traversal_seeds,
                max_depth=max_traversal_depth,
                max_nodes=max_traversal_nodes
            )
            diagnostics["graph_expanded_nodes"] = [n.id for n in traversal_res["discovered_nodes"]]
            diagnostics["graph_edges_traversed"] = [e.edge_type for e in traversal_res["discovered_edges"]]
            traversal_details = traversal_res["diagnostics"]
            diagnostics["traversal_details"] = traversal_details

            # Calculate hop-decayed graph candidate score
            # Prevents weak graph candidates from blindly outranking direct evidence!
            for step in traversal_details:
                to_node = step["to_node"]
                depth = step["depth"]
                step_score = step["score"]
                # Decay factor: 0.5 per hop
                decayed = (step_score * (0.5 ** depth)) / (self.k_rrf + 5)

                node = self.store.get_node(to_node)
                if node and node.node_type == "chunk":
                    graph_scores[to_node] = max(graph_scores.get(to_node, 0.0), decayed)

        # 6. Final Score Fusion
        # Chunks with direct evidence retain primary RRF trust.
        # Graph connectivity adds a principled support bonus.
        # Pure indirect graph discoveries (e.g. multi-hop targets) enter with decayed discovery score.
        final_chunk_scores: dict[str, float] = {}
        all_evaluated_chunks = set(direct_scores.keys()) | set(graph_scores.keys())

        for c_id in all_evaluated_chunks:
            d_score = direct_scores.get(c_id, 0.0)
            g_score = graph_scores.get(c_id, 0.0)

            if d_score > 0.0:
                # Direct hit with graph confirmation bonus (up to +20%)
                final_chunk_scores[c_id] = d_score + (0.5 * g_score)
            else:
                # Pure indirect multi-hop discovery
                final_chunk_scores[c_id] = g_score

        # Rank all candidate chunks
        sorted_chunks = sorted(
            final_chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top chunks, ensuring up to k distinct memories are represented
        # This prevents multiple chunks of one episode from displacing a supporting chunk of another episode
        selected_chunk_ids = []
        seen_memory_ids = set()
        for c_id, score in sorted_chunks:
            chunk_node = self.chunk_by_id.get(c_id) or self.store.get_node(c_id)
            if not chunk_node:
                continue
            m_id = chunk_node.properties.get("memory_id")
            if m_id not in seen_memory_ids:
                seen_memory_ids.add(m_id)
                selected_chunk_ids.append(c_id)
            elif len(selected_chunk_ids) < k:
                selected_chunk_ids.append(c_id)

            if len(seen_memory_ids) >= k and len(selected_chunk_ids) >= k:
                break

        # Build Selected Chunk Evidence Objects
        selected_chunk_objs = []
        for c_id in selected_chunk_ids:
            chunk_node = self.chunk_by_id.get(c_id) or self.store.get_node(c_id)
            if not chunk_node:
                continue
            props = chunk_node.properties
            selected_chunk_objs.append({
                "chunk_id": chunk_node.id,
                "memory_id": props.get("memory_id", "m000"),
                "episode_id": props.get("episode_id"),
                "text": props.get("text", ""),
                "chunk_index": props.get("chunk_index", 0),
                "char_start": props.get("char_start", 0),
                "char_end": props.get("char_end", len(props.get("text", ""))),
                "timestamp": props.get("timestamp"),
                "score": round(final_chunk_scores.get(c_id, 0.0), 5)
            })

        # 7. Semantic Relationship Triple Reconstruction
        # Gather TRUE semantic relationships (Entity -> Entity)
        # Excludes internal/structural edges: grounds_*, mentions_entity, adjacent_to, contains_*
        direct_facts = []
        inferred_facts = []
        seen_fact_signatures = set()

        # Check relationships connected to entities mentioned in selected chunks or seeds
        relevant_entity_ids = set(entity_seeds)
        for ch in selected_chunk_objs:
            ch_node = self.chunk_by_id.get(ch["chunk_id"])
            if ch_node:
                relevant_entity_ids.update(ch_node.properties.get("entity_ids", []))
            # Also discover entities linked in the store directly to this chunk (mentions or grounds)
            chunk_edges = self.store.get_outgoing_edges(ch["chunk_id"])
            for ce in chunk_edges:
                if ce.edge_type == "mentions_entity" or ce.edge_type.startswith("grounds_"):
                    relevant_entity_ids.add(ce.target_id)

        # Query semantic edges connected to relevant entities
        for ent_id in relevant_entity_ids:
            edges = self.store.get_outgoing_edges(ent_id) + self.store.get_incoming_edges(ent_id)
            for e in edges:
                # Filter out internal/structural edges
                if e.edge_type in ("contains_chunk", "adjacent_to", "contains_memory", "mentions_entity", "precedes"):
                    continue
                if e.edge_type.startswith("grounds_"):
                    continue

                # Ensure endpoints are Entity nodes
                src_node = self.store.get_node(e.source_id)
                tgt_node = self.store.get_node(e.target_id)
                if not src_node or not tgt_node:
                    continue
                if src_node.node_type != "entity" or tgt_node.node_type != "entity":
                    continue

                sig = (src_node.label, e.edge_type, tgt_node.label, e.memory_id)
                if sig in seen_fact_signatures:
                    continue
                seen_fact_signatures.add(sig)

                fact_obj = {
                    "statement": f"{src_node.label} {e.edge_type} {tgt_node.label}",
                    "subject": src_node.label,
                    "predicate": e.edge_type,
                    "object": tgt_node.label,
                    "memory_id": e.memory_id,
                    "chunk_id": e.chunk_id,
                    "evidence": e.evidence,
                    "confidence": e.confidence,
                    "origin": e.origin,
                    "timestamp": e.timestamp
                }

                if e.origin == "inferred":
                    inferred_facts.append(fact_obj)
                else:
                    direct_facts.append(fact_obj)

        # 8. Build Evidence Context Lines (Chunk-Level, Not Full Episode Text)
        context_text = self._format_chunk_context(clean_query, selected_chunk_objs, direct_facts, inferred_facts)

        # 9. Format direct_evidence for downstream answering & citation validation
        # Provides chunk-level text and offsets, mapped to both memory_id and chunk_id
        direct_evidence = []
        for ch in selected_chunk_objs:
            # Associate direct facts relevant to this chunk/memory
            ch_facts = [f for f in direct_facts if f.get("chunk_id") == ch["chunk_id"] or f.get("memory_id") == ch["memory_id"]]
            direct_evidence.append({
                "memory_id": ch["memory_id"],
                "chunk_id": ch["chunk_id"],
                "text": ch["text"],
                "char_start": ch["char_start"],
                "char_end": ch["char_end"],
                "timestamp": ch["timestamp"],
                "score": ch["score"],
                "direct_facts": ch_facts
            })

        # Collect unique parent memory records represented
        unique_memory_ids = list(dict.fromkeys(ch["memory_id"] for ch in selected_chunk_objs))
        retrieved_memories = []
        for m_id in unique_memory_ids:
            memo_node = self.memory_by_id.get(m_id) or self.store.get_node(m_id)
            if memo_node:
                retrieved_memories.append({
                    "id": m_id,
                    "text": memo_node.properties.get("text", ""),
                    "timestamp": memo_node.properties.get("timestamp"),
                    "source": memo_node.properties.get("source"),
                    "score": max((ch["score"] for ch in selected_chunk_objs if ch["memory_id"] == m_id), default=0.0)
                })

        return {
            "query": clean_query,
            "has_sufficient_evidence": len(selected_chunk_objs) > 0,
            "status": "sufficient_evidence" if selected_chunk_objs else "insufficient_evidence",
            "direct_evidence": direct_evidence,
            "inferred_evidence": inferred_facts,
            "retrieved_chunks": selected_chunk_objs,
            "retrieved_memories": retrieved_memories,
            "retrieved_memories_count": len(retrieved_memories),
            "context_text": context_text,
            "diagnostics": diagnostics
        }

    def _format_chunk_context(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        direct_facts: list[dict[str, Any]],
        inferred_facts: list[dict[str, Any]]
    ) -> str:
        if not chunks:
            return f"Query: {query}\nStatus: Insufficient evidence in personal memory store."

        count = len(chunks)
        lines = [
            f"Query: {query}",
            f"Status: Sufficient evidence ({count} relevant chunk{'s' if count != 1 else ''} found)\n",
            "--- DIRECTLY SUPPORTED EVIDENCE (CHUNKS) ---"
        ]

        for ch in chunks:
            ts_str = f", timestamp: {ch['timestamp']}" if ch.get('timestamp') else ""
            lines.append(f"[{ch['chunk_id']} | Source: {ch['memory_id']}]{ts_str} (score: {ch['score']:.4f}): \"{ch['text']}\"")

        if direct_facts:
            lines.append("\n--- VERIFIED RELATIONSHIP TRIPLES ---")
            for f in direct_facts:
                ev_str = f" (evidence: \"{f['evidence']}\" in [{f.get('chunk_id') or f.get('memory_id')}])" if f.get('evidence') else ""
                lines.append(f"  * Fact: {f['statement']}{ev_str}")

        if inferred_facts:
            lines.append("\n--- INFERRED RELATIONSHIPS (UNVERIFIED / DERIVED) ---")
            for inf in inferred_facts:
                lines.append(f"  * [Inferred] {inf['statement']} (derived from premise: \"{inf.get('evidence')}\")")

        return "\n".join(lines)

    def _empty_response(self, query: str) -> dict[str, Any]:
        return {
            "query": query,
            "has_sufficient_evidence": False,
            "status": "insufficient_evidence",
            "direct_evidence": [],
            "inferred_evidence": [],
            "retrieved_chunks": [],
            "retrieved_memories": [],
            "retrieved_memories_count": 0,
            "context_text": f"Query: {query}\nStatus: Insufficient evidence in personal memory store.",
            "diagnostics": {}
        }
