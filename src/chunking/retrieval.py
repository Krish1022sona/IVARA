"""
Chunk Retrieval and Compact Context Assembly for IVARA TACC.

Provides FAISS semantic search + BM25 lexical search over TurnChunk units,
with context-token budget enforcement and dialogue snippet consolidation.
"""

from collections import defaultdict
import math
import re
from typing import Any
import faiss
import numpy as np

from .turn_chunker import TurnChunk, TurnAwareChunker
from ..embedding import get_embedding


class BM25ChunkSearcher:
    """Lightweight BM25 lexical retriever tailored for conversational turn chunks."""

    def __init__(self, chunks: list[TurnChunk]):
        self.chunks = chunks
        self.doc_count = len(chunks)
        self.avg_doc_len = 0.0
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)

        if self.doc_count > 0:
            total_tokens = 0
            for chunk in chunks:
                tokens = self._tokenize(chunk.text)
                self.doc_tokens.append(tokens)
                total_tokens += len(tokens)
                for t in set(tokens):
                    self.doc_freqs[t] += 1
            self.avg_doc_len = total_tokens / self.doc_count

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in re.findall(r'\b[a-zA-Z0-9_\-+$]{2,}\b', text)]

    def search(
        self,
        query: str,
        top_k: int = 15,
        expanded_tokens: list[str] | None = None
    ) -> list[tuple[float, int]]:
        q_tokens = self._tokenize(query)
        if expanded_tokens:
            for t in expanded_tokens:
                for tok in self._tokenize(t):
                    if tok not in q_tokens:
                        q_tokens.append(tok)

        if not q_tokens or self.doc_count == 0:
            return []

        scores: list[tuple[float, int]] = []
        k1 = 1.5
        b = 0.75

        for idx, doc_toks in enumerate(self.doc_tokens):
            doc_len = len(doc_toks)
            score = 0.0
            tok_counts = defaultdict(int)
            for t in doc_toks:
                tok_counts[t] += 1

            for qt in q_tokens:
                if qt in tok_counts:
                    df = self.doc_freqs.get(qt, 0)
                    idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
                    tf = tok_counts[qt]
                    num = tf * (k1 + 1.0)
                    den = tf + k1 * (1.0 - b + b * (doc_len / (self.avg_doc_len or 1.0)))
                    score += idf * (num / den)

            if score > 0.0:
                scores.append((round(score, 4), idx))

        scores.sort(key=lambda x: (-x[0], x[1]))
        return scores[:top_k]


class ChunkRetriever:
    """
    Hybrid Vector + BM25 + Ontology retriever operating over TurnChunk units.
    """

    def __init__(
        self,
        chunks: list[TurnChunk],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        ontology_weight: float = 0.15,
        use_ontology: bool = False,
        vocabulary: dict[str, Any] | None = None
    ):
        self.chunks = chunks
        self.use_ontology = use_ontology

        if self.use_ontology:
            self.ontology_weight = ontology_weight
            total = vector_weight + bm25_weight + ontology_weight
            if total > 0:
                self.vector_weight = vector_weight / total
                self.bm25_weight = bm25_weight / total
                self.ontology_weight = ontology_weight / total
            else:
                self.vector_weight = 0.55
                self.bm25_weight = 0.30
                self.ontology_weight = 0.15
        else:
            self.vector_weight = vector_weight
            self.bm25_weight = bm25_weight
            self.ontology_weight = 0.0

        self.vocabulary = vocabulary
        self.index: faiss.IndexFlatIP | None = None
        self.bm25: BM25ChunkSearcher | None = None

        if self.chunks:
            self._build_indexes()

    def _build_indexes(self):
        """Construct in-memory FAISS and BM25 indexes."""
        # 1. FAISS Index
        texts = [c.text for c in self.chunks]
        raw_embeddings = [get_embedding(t) for t in texts]
        embeddings = np.array(raw_embeddings, dtype="float32")

        if embeddings.ndim == 2 and embeddings.shape[0] > 0:
            faiss.normalize_L2(embeddings)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)

        # 2. BM25 Index
        self.bm25 = BM25ChunkSearcher(self.chunks)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float | None = None,
        max_chunks_per_session: int | None = None,
        use_session_mmr: bool = True,
        mmr_lambda: float = 0.25,
        mmr_pool_size: int = 15
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k chunks combining cosine similarity, BM25, and optional ontology matching.
        """
        if not query or not query.strip() or not self.chunks or self.index is None:
            return []

        clean_query = query.strip()
        search_k = min(max(k * 3, 15), len(self.chunks))

        # 1. Concept Extraction (if ontology is enabled)
        concepts: dict[str, list[str]] = {"entities": [], "aliases": [], "predicates": []}
        expanded_tokens: list[str] = []
        if self.use_ontology:
            from .ontology_retrieval import extract_query_concepts
            concepts = extract_query_concepts(clean_query, self.vocabulary)
            expanded_tokens = concepts.get("aliases", [])

        # 2. Vector search on clean query (unpolluted by alias tokens)
        q_emb = np.array([get_embedding(clean_query)], dtype="float32")
        faiss.normalize_L2(q_emb)
        vec_scores, vec_indices = self.index.search(q_emb, search_k)

        candidate_scores: dict[int, dict[str, float]] = defaultdict(lambda: {"vec": 0.0, "bm25": 0.0, "ont": 0.0})

        # Max vector score for normalization
        max_vec = 0.0
        for score, idx in zip(vec_scores[0], vec_indices[0]):
            if idx != -1 and idx < len(self.chunks):
                s_float = float(score)
                candidate_scores[idx]["vec"] = s_float
                if s_float > max_vec:
                    max_vec = s_float

        # 3. BM25 search (with expanded alias tokens if ontology enabled)
        max_bm25 = 0.0
        if self.bm25 is not None:
            bm25_hits = self.bm25.search(clean_query, top_k=search_k, expanded_tokens=expanded_tokens)
            for b_score, idx in bm25_hits:
                candidate_scores[idx]["bm25"] = b_score
                if b_score > max_bm25:
                    max_bm25 = b_score

        # 4. Calibrated Ontology Entity and Concept Matching
        if self.use_ontology:
            target_entities = [e.lower() for e in concepts.get("entities", [])]
            target_predicates = [p.lower() for p in concepts.get("predicates", [])]
            target_aliases = [a.lower() for a in concepts.get("aliases", [])]

            for idx in candidate_scores.keys():
                chunk_text_lower = self.chunks[idx].text.lower()

                # Component A: Entity match
                s_ent = 0.0
                if target_entities:
                    matched_ent = sum(1 for e in target_entities if e in chunk_text_lower)
                    s_ent = matched_ent / len(target_entities)

                # Component B: Predicate match
                s_pred = 0.0
                if target_predicates:
                    matched_pred = sum(1 for p in target_predicates if p in chunk_text_lower)
                    s_pred = matched_pred / len(target_predicates)

                # Component C: Alias match
                s_alias = 0.0
                if target_aliases:
                    matched_alias = sum(1 for a in target_aliases if a in chunk_text_lower)
                    s_alias = min(1.0, matched_alias / min(3, len(target_aliases)))

                # Weighted normalized ontology match score in [0.0, 1.0]
                if target_entities and target_predicates:
                    ont_score = 0.50 * s_ent + 0.25 * s_pred + 0.25 * s_alias
                elif target_entities:
                    ont_score = 0.65 * s_ent + 0.35 * s_alias
                elif target_predicates:
                    ont_score = 0.50 * s_pred + 0.50 * s_alias
                elif target_aliases:
                    ont_score = s_alias
                else:
                    ont_score = 0.0

                candidate_scores[idx]["ont"] = ont_score

        # 5. Fused scoring
        fused_results = []
        for idx, scores in candidate_scores.items():
            norm_vec = max(0.0, scores["vec"])
            norm_bm25 = (scores["bm25"] / max_bm25) if max_bm25 > 0 else 0.0

            if self.use_ontology:
                norm_ont = scores["ont"]
                fused_score = (
                    (self.vector_weight * norm_vec) +
                    (self.bm25_weight * norm_bm25) +
                    (self.ontology_weight * norm_ont)
                )
            else:
                fused_score = (self.vector_weight * norm_vec) + (self.bm25_weight * norm_bm25)

            if min_score is not None and fused_score < min_score:
                continue

            chunk = self.chunks[idx]
            result_item: dict[str, Any] = {
                "chunk": chunk,
                "chunk_id": chunk.chunk_id,
                "memory_id": chunk.memory_id,
                "session_id": chunk.session_id,
                "score": round(float(fused_score), 4),
                "vector_score": round(scores["vec"], 4),
                "bm25_score": round(scores["bm25"], 4),
                "turn_range": (chunk.turn_start_idx, chunk.turn_end_idx),
                "timestamp": chunk.timestamp,
                "text": chunk.text,
                "token_estimate": chunk.token_estimate
            }
            if self.use_ontology:
                result_item["ontology_score"] = round(scores["ont"], 4)
                result_item["ontology_bonus"] = round(scores["ont"], 4)
                result_item["detected_concepts"] = concepts

            fused_results.append(result_item)

        # Deterministic sorting: by score descending, then turn_start ascending, then chunk_id ascending
        fused_results.sort(key=lambda x: (-x["score"], x["turn_range"][0], x["chunk_id"]))
        if max_chunks_per_session is not None and max_chunks_per_session > 0:
            diverse_results: list[dict[str, Any]] = []
            session_counts: dict[str, int] = defaultdict(int)
            for item in fused_results:
                sid = item.get("session_id") or item.get("memory_id")
                if sid:
                    if session_counts[sid] >= max_chunks_per_session:
                        continue
                    session_counts[sid] += 1
                diverse_results.append(item)
                if len(diverse_results) >= k:
                    break
            return diverse_results

        # Session-Aware MMR selection when enabled
        if use_session_mmr and len(fused_results) > k:
            candidate_pool = fused_results[:max(k, mmr_pool_size)]
            return select_session_mmr_chunks(candidate_pool, k=k, lambda_penalty=mmr_lambda)

        return fused_results[:k]


def select_session_mmr_chunks(
    candidates: list[dict[str, Any]],
    k: int = 5,
    lambda_penalty: float = 0.25
) -> list[dict[str, Any]]:
    """
    Session-Aware Maximal Marginal Relevance selection.
    Balances chunk relevance with diversity across conversational sessions.
    Penalizes subsequent chunks from an already-selected session by lambda_penalty.
    Preserves existing top-k retrieval as an exact fallback when lambda_penalty=0.0.
    """
    if not candidates or k <= 0:
        return []
    if len(candidates) <= k or lambda_penalty <= 0.0:
        return candidates[:k]

    selected: list[dict[str, Any]] = []
    selected_sids: set[str] = set()
    remaining = list(candidates)

    for _ in range(min(k, len(remaining))):
        best_idx = -1
        best_score = -999.0
        for i, c in enumerate(remaining):
            sid = str(c.get("session_id") or c.get("memory_id") or "")
            penalty = lambda_penalty if sid in selected_sids else 0.0
            score = c.get("score", 0.0) - penalty
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0:
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            chosen_sid = str(chosen.get("session_id") or chosen.get("memory_id") or "")
            if chosen_sid:
                selected_sids.add(chosen_sid)

    return selected


def assemble_bounded_context(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    max_context_tokens: int = 1500,
    max_chunks_per_session: int | None = None
) -> dict[str, Any]:
    """
    Assemble a compact, bounded context from retrieved turn chunks,
    respecting an explicit token budget while preserving session provenance.
    """
    if not retrieved_chunks:
        return {
            "query": query,
            "has_sufficient_evidence": False,
            "status": "insufficient_evidence",
            "message": f"Insufficient evidence: No relevant chunks found for '{query}'.",
            "direct_evidence": [],
            "retrieved_memories_count": 0,
            "retrieved_chunks_count": 0,
            "context_token_count": 0,
            "context_text": f"Query: {query}\nStatus: Insufficient evidence in personal memory store."
        }

    # Budget-aware chunk selection
    selected: list[dict[str, Any]] = []
    current_tokens = 0
    session_counts: dict[str, int] = defaultdict(int)

    for item in retrieved_chunks:
        sid = item.get("session_id") or item.get("memory_id")
        if max_chunks_per_session is not None and max_chunks_per_session > 0 and sid:
            if session_counts[sid] >= max_chunks_per_session:
                continue
        est = item.get("token_estimate", 100)
        # Always allow at least the top-1 chunk; then check budget
        if selected and (current_tokens + est > max_context_tokens):
            continue
        selected.append(item)
        if sid:
            session_counts[sid] += 1
        current_tokens += est

    # Group selected chunks by session to present coherent snippets
    session_snippets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in selected:
        session_snippets[s["memory_id"]].append(s)

    context_lines = [
        f"Query: {query}",
        f"Status: Sufficient evidence ({len(selected)} relevant chunk{'s' if len(selected) != 1 else ''} found)\n",
        "--- DIRECTLY SUPPORTED EVIDENCE ---"
    ]

    direct_evidence = []
    # Sort sessions chronologically by timestamp so multi-session context is presented in temporal order
    sorted_sessions = sorted(
        session_snippets.items(),
        key=lambda item: (str(item[1][0].get("timestamp", "")), item[0])
    )
    for mem_id, chunks_in_sess in sorted_sessions:
        # Sort chunks within the session chronologically by start turn
        chunks_in_sess.sort(key=lambda c: c["turn_range"][0])
        for c in chunks_in_sess:
            start_t, end_t = c["turn_range"]
            header = f"[{c['chunk_id']}] (Session: {c['session_id']}, Turns {start_t}-{end_t}, Date: {c['timestamp']}, Relevance: {c['score']}):"
            context_lines.append(header)
            context_lines.append(f'"{c["text"]}"\n')

            direct_evidence.append({
                "memory_id": c["memory_id"],
                "chunk_id": c["chunk_id"],
                "session_id": c["session_id"],
                "text": c["text"],
                "turn_range": (start_t, end_t),
                "timestamp": c["timestamp"],
                "score": c["score"]
            })

    context_text = "\n".join(context_lines).strip()
    actual_words = len(context_text.split())
    measured_tokens = max(1, int(actual_words / 0.75))

    return {
        "query": query,
        "has_sufficient_evidence": len(selected) > 0,
        "status": "sufficient_evidence",
        "message": f"Found {len(selected)} relevant turn chunks.",
        "direct_evidence": direct_evidence,
        "retrieved_memories_count": len(session_snippets),
        "retrieved_chunks_count": len(selected),
        "context_token_count": measured_tokens,
        "context_text": context_text
    }
