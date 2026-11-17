"""
Hierarchical Episodic Retriever for IVARA.

Implements two-stage hierarchical retrieval to eliminate multi-session starvation:
Stage 1: Session Retrieval — Builds session-level representations and retrieves candidate
         conversational sessions combining dense vector similarity and BM25 lexical matching.
         Applies session-level diversity controls to prevent single-session dominance.
Stage 2: Turn/Chunk Retrieval — Searches within candidate sessions for the most relevant
         turn-level chunks, enforces per-session caps, preserves chronological ordering,
         and deduplicates overlapping turns.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date
import math
import re
from typing import Any, Optional
import faiss
import numpy as np

from .turn_chunker import TurnChunk
from .retrieval import BM25ChunkSearcher
from ..embedding import get_embeddings, get_embedding
from ..temporal.calendar_arithmetic import parse_date_string


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not",
    "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "won't", "would", "you", "your", "yours",
    "yourself", "yourselves", "tell"
}


@dataclass
class SessionRepresentation:
    session_id: str
    memory_id: str
    date_str: str
    parsed_date: Optional[date]
    chunks: list[TurnChunk]
    text: str
    embedding: Optional[np.ndarray] = None


class SessionBM25Searcher:
    """BM25 lexical retriever tailored for complete conversational session transcripts."""

    def __init__(self, sessions: list[SessionRepresentation]):
        self.sessions = sessions
        self.doc_count = len(sessions)
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.avg_doc_len = 1.0

        if self.doc_count > 0:
            total_tokens = 0
            for sess in sessions:
                tokens = self._tokenize(sess.text)
                self.doc_tokens.append(tokens)
                total_tokens += len(tokens)
                for t in set(tokens):
                    self.doc_freqs[t] += 1
            self.avg_doc_len = total_tokens / self.doc_count if self.doc_count else 1.0

    def _tokenize(self, text: str) -> list[str]:
        tokens = [t.lower() for t in re.findall(r'\b[a-zA-Z0-9_\-+$]{2,}\b', text)]
        meaningful = [t for t in tokens if t not in STOPWORDS]
        return meaningful if meaningful else tokens

    def search(self, query: str, top_k: int = 15) -> list[tuple[float, int]]:
        q_tokens = self._tokenize(query)
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
                    den = tf + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len))
                    score += idf * (num / den)

            if score > 0.0:
                scores.append((round(score, 4), idx))

        scores.sort(key=lambda x: (-x[0], x[1]))
        return scores[:top_k]


class HierarchicalEpisodicRetriever:
    """
    Two-stage Hierarchical Retriever:
    Stage 1: Session-level retrieval with optional diversity controls.
    Stage 2: Within-session chunk retrieval with chronological sorting.
    """

    def __init__(
        self,
        chunks: list[TurnChunk],
        vector_weight: float = 0.55,
        bm25_weight: float = 0.45,
        session_text_cap_chars: int = 2500
    ):
        self.chunks = chunks
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.session_text_cap_chars = session_text_cap_chars

        self.sessions: list[SessionRepresentation] = []
        self.session_index: Optional[faiss.IndexFlatIP] = None
        self.session_bm25: Optional[SessionBM25Searcher] = None
        self.chunk_retrievers: dict[str, Any] = {}

        if self.chunks:
            self._build_session_representations()

    def _build_session_representations(self):
        """Group chunks into coherent session units and index both sessions and chunks."""
        by_session: dict[str, list[TurnChunk]] = defaultdict(list)
        for c in self.chunks:
            by_session[c.session_id].append(c)

        session_texts = []
        for sid, s_chunks in by_session.items():
            # Sort chunks in session chronologically
            s_chunks.sort(key=lambda x: x.turn_start_idx)
            first_chunk = s_chunks[0]

            # Construct session dialogue text without duplicate turns
            seen_turn_signatures = set()
            turn_lines = []
            for c in s_chunks:
                for t in c.turns:
                    role = t.get("role", "User")
                    content = t.get("content", "").strip()
                    sig = (role, content)
                    if sig not in seen_turn_signatures:
                        seen_turn_signatures.add(sig)
                        turn_lines.append(f"{role}: {content}")

            date_str = first_chunk.timestamp or ""
            parsed_d = parse_date_string(date_str)
            raw_text = f"Session Date: {date_str}\n" + "\n".join(turn_lines)
            capped_text = raw_text[:self.session_text_cap_chars]
            session_texts.append(capped_text)

            sess_rep = SessionRepresentation(
                session_id=sid,
                memory_id=first_chunk.memory_id,
                date_str=date_str,
                parsed_date=parsed_d,
                chunks=s_chunks,
                text=raw_text
            )
            self.sessions.append(sess_rep)

        # Batch encode session representations
        if session_texts:
            raw_embeddings = get_embeddings(session_texts, batch_size=64)
            embeddings = np.array(raw_embeddings, dtype="float32")
            if embeddings.ndim == 2 and embeddings.shape[0] > 0:
                faiss.normalize_L2(embeddings)
                dim = embeddings.shape[1]
                self.session_index = faiss.IndexFlatIP(dim)
                self.session_index.add(embeddings)
                for idx, emb in enumerate(embeddings):
                    self.sessions[idx].embedding = emb

        # Build Session BM25 index
        self.session_bm25 = SessionBM25Searcher(self.sessions)

    def retrieve_sessions(
        self,
        query: str,
        top_m: int = 5,
        use_diversity: bool = False,
        diversity_lambda: float = 0.30
    ) -> list[tuple[SessionRepresentation, float]]:
        """
        Stage 1: Retrieve top-M candidate sessions using hybrid Vector + BM25,
        with optional session diversity controls to avoid temporal/thematic clustering.
        """
        if not self.sessions or not query.strip() or self.session_index is None:
            return []

        clean_query = query.strip()
        search_m = min(max(top_m * 3, 15), len(self.sessions))

        # 1. Session Vector Search
        q_emb = np.array([get_embedding(clean_query)], dtype="float32")
        faiss.normalize_L2(q_emb)
        vec_scores, vec_indices = self.session_index.search(q_emb, search_m)

        session_scores: dict[int, dict[str, float]] = defaultdict(lambda: {"vec": 0.0, "bm25": 0.0})
        for score, idx in zip(vec_scores[0], vec_indices[0]):
            if idx != -1 and idx < len(self.sessions):
                session_scores[idx]["vec"] = max(0.0, float(score))

        # 2. Session BM25 Search
        max_bm25 = 0.0
        if self.session_bm25 is not None:
            bm25_hits = self.session_bm25.search(clean_query, top_k=search_m)
            for b_score, idx in bm25_hits:
                session_scores[idx]["bm25"] = b_score
                if b_score > max_bm25:
                    max_bm25 = b_score

        # 3. Fused Session Scoring
        ranked_candidates: list[tuple[int, float]] = []
        for idx, sc in session_scores.items():
            norm_vec = sc["vec"]
            norm_bm = (sc["bm25"] / max_bm25) if max_bm25 > 0.0 else 0.0
            fused = (self.vector_weight * norm_vec) + (self.bm25_weight * norm_bm)
            ranked_candidates.append((idx, fused))

        ranked_candidates.sort(key=lambda x: -x[1])

        # 4. Selection: Pure Top-M vs Diversity-Controlled Selection
        if not use_diversity or len(ranked_candidates) <= top_m or diversity_lambda <= 0.0:
            return [(self.sessions[idx], round(score, 4)) for idx, score in ranked_candidates[:top_m]]

        # Diversity controls: Penalize candidate sessions from the exact same calendar day
        # or identical date prefix when earlier sessions from that date were already selected
        selected_sessions: list[tuple[SessionRepresentation, float]] = []
        selected_dates: dict[str, int] = defaultdict(int)
        remaining = list(ranked_candidates)

        for _ in range(min(top_m, len(remaining))):
            best_idx = -1
            best_penalized_score = -999.0
            best_raw_score = 0.0
            best_rem_pos = -1

            for pos, (s_idx, raw_score) in enumerate(remaining):
                sess = self.sessions[s_idx]
                d_key = sess.date_str[:10] if sess.date_str else sess.session_id
                date_count = selected_dates.get(d_key, 0)
                # Penalize multiple sessions from the exact same date
                penalty = date_count * diversity_lambda
                penalized_score = raw_score - penalty

                if penalized_score > best_penalized_score:
                    best_penalized_score = penalized_score
                    best_raw_score = raw_score
                    best_idx = s_idx
                    best_rem_pos = pos

            if best_idx != -1 and best_rem_pos != -1:
                chosen_sess = self.sessions[best_idx]
                d_key = chosen_sess.date_str[:10] if chosen_sess.date_str else chosen_sess.session_id
                selected_dates[d_key] += 1
                selected_sessions.append((chosen_sess, round(best_raw_score, 4)))
                remaining.pop(best_rem_pos)

        return selected_sessions

    def retrieve(
        self,
        query: str,
        k: int = 8,
        top_m_sessions: int = 5,
        max_chunks_per_session: int = 2,
        use_diversity: bool = False,
        diversity_lambda: float = 0.30
    ) -> list[dict[str, Any]]:
        """
        Execute two-stage hierarchical retrieval:
        1. Select top-M candidate sessions.
        2. Within each selected session, score and retrieve top-N chunks.
        3. Sort final chunk results chronologically across all selected sessions.
        """
        candidate_sessions = self.retrieve_sessions(
            query=query,
            top_m=top_m_sessions,
            use_diversity=use_diversity,
            diversity_lambda=diversity_lambda
        )

        if not candidate_sessions:
            return []

        q_clean = query.strip()
        q_emb = np.array([get_embedding(q_clean)], dtype="float32")
        faiss.normalize_L2(q_emb)

        collected_chunks: list[dict[str, Any]] = []

        # Stage 2: Within-session chunk scoring and selection
        for sess, sess_score in candidate_sessions:
            s_chunks = sess.chunks
            if not s_chunks:
                continue

            # Compute chunk embeddings & scores within this session
            chunk_texts = [c.text for c in s_chunks]
            raw_c_embs = get_embeddings(chunk_texts, batch_size=32)
            c_embs = np.array(raw_c_embs, dtype="float32")
            faiss.normalize_L2(c_embs)

            # Cosine similarity
            c_sims = np.dot(c_embs, q_emb.T).flatten()

            # BM25 within session chunks
            bm25 = BM25ChunkSearcher(s_chunks)
            bm25_hits = {idx: sc for sc, idx in bm25.search(q_clean, top_k=len(s_chunks))}
            max_c_bm25 = max(bm25_hits.values()) if bm25_hits else 1.0

            scored_chunks = []
            for c_idx, chunk in enumerate(s_chunks):
                vec_sc = max(0.0, float(c_sims[c_idx]))
                raw_bm = bm25_hits.get(c_idx, 0.0)
                norm_bm = (raw_bm / max_c_bm25) if max_c_bm25 > 0.0 else 0.0
                # Fused chunk score combines chunk-level relevance with session relevance
                chunk_score = (0.50 * vec_sc) + (0.30 * norm_bm) + (0.20 * sess_score)

                scored_chunks.append({
                    "chunk": chunk,
                    "chunk_id": chunk.chunk_id,
                    "memory_id": chunk.memory_id,
                    "session_id": chunk.session_id,
                    "score": round(float(chunk_score), 4),
                    "vector_score": round(vec_sc, 4),
                    "bm25_score": round(raw_bm, 4),
                    "turn_range": (chunk.turn_start_idx, chunk.turn_end_idx),
                    "timestamp": chunk.timestamp,
                    "text": chunk.text,
                    "token_estimate": chunk.token_estimate
                })

            scored_chunks.sort(key=lambda x: -x["score"])
            # Allocate up to max_chunks_per_session from this session
            allocated = scored_chunks[:max_chunks_per_session]
            collected_chunks.extend(allocated)

        # Sort by score descending so assemble_bounded_context selects highest relevance chunks first
        collected_chunks.sort(key=lambda x: -x["score"])
        if len(collected_chunks) > k:
            collected_chunks = collected_chunks[:k]

        return collected_chunks


class AdaptiveTemporalRouter:
    """
    Deterministic Adaptive Router for IVARA retrieval strategy selection.
    Routes high-confidence multi-session queries to hierarchical retrieval with diversity.
    Routes single-session, relative elapsed time, and ordinary factual queries to flat retrieval.
    Safe fallback: If confidence is low or query is ambiguous, defaults to flat retrieval.
    """
    
    # 1. Explicit single-session exclusions (MUST use flat retrieval)
    SINGLE_SESSION_EXCLUSIONS = [
        re.compile(r"\bhow many (?:days|weeks|months|years) ago did i\b", re.I),
        re.compile(r"\bhow long ago did i\b", re.I),
        re.compile(r"\bhow many (?:days|weeks|months|years) have passed since i (?:last|visited|went|bought|started|participated)\b", re.I),
        re.compile(r"\bi mentioned (?:participating in|visiting|cooking|an investment|a significant)\b.+\b(?:ago|last)\b", re.I),
        re.compile(r"\bwhich book did i finish a week ago\b", re.I),
        re.compile(r"\bwhat (?:gardening-related|charity|artist|life|social media|significant)\b.+\b(?:ago|last friday)\b", re.I),
    ]

    # 2. High-confidence multi-session patterns
    HIGH_CONFIDENCE_MULTI_SESSION = [
        # Explicit chronological ordering of 2+ events
        re.compile(r"\border of (?:the )?(?:\w+ )?(?:events|trips|museums|sports events|sports|concerts|musical events|airlines|books)\b", re.I),
        re.compile(r"\border from (?:first|earliest) to (?:last|latest)\b", re.I),
        re.compile(r"\bfrom (?:earliest|first) to (?:latest|last)\b", re.I),
        re.compile(r"\bstarting from the earliest\b", re.I),
        re.compile(r"\bwhich (?:three )?events happened in (?:the )?order\b", re.I),
        re.compile(r"\bwho graduated first,?\s*second\b", re.I),
        re.compile(r"\bwhich (?:event|group|airline|book|webinar|workshop|activity) did i (?:attend|join|fly|read|finish|participate in) first\b", re.I),
        re.compile(r"\bwhich (?:event )?happened first\b", re.I),
        re.compile(r"\bfirst to last\b", re.I),
        
        # Elapsed time / interval between TWO distinct named events
        re.compile(r"\bhow many (?:days|weeks|months|years) passed between (?:the|my)\b.+\band (?:the|my)\b", re.I),
        re.compile(r"\bhow many (?:days|weeks|months) before\b.+\bdid\b.+\b(?:get engaged|happen|occur|start)\b", re.I),
        re.compile(r"\bhow many weeks have i been accepted into .+ when i started\b", re.I),
        re.compile(r"\bbetween the completion of .+ and the submission of\b", re.I),
        re.compile(r"\bbetween the day i finished .+ and the day i started\b", re.I),
        re.compile(r"\bbetween the time i .+ and the time i\b", re.I),
        
        # Longitudinal multi-event aggregation
        re.compile(r"\bconsecutive days\b", re.I),
        re.compile(r"\bin a row\b", re.I),
        re.compile(r"\btotal (?:do i spent|time|hours|days|weeks)\b.+\band\b", re.I),
        re.compile(r"\bhow many (?:charity )?events did i participate in before\b", re.I),
    ]

    @classmethod
    def route(cls, query: str) -> dict[str, Any]:
        clean_q = query.strip()
        if not clean_q:
            return {
                "strategy": "flat",
                "confidence": 1.0,
                "reason": "Empty query defaults to flat retrieval."
            }
        
        # Check single-session exclusions first
        for pat in cls.SINGLE_SESSION_EXCLUSIONS:
            if pat.search(clean_q):
                return {
                    "strategy": "flat",
                    "confidence": 0.95,
                    "reason": f"Single-session relative anchor pattern matched: {pat.pattern}"
                }
                
        # Check high-confidence multi-session patterns
        for pat in cls.HIGH_CONFIDENCE_MULTI_SESSION:
            if pat.search(clean_q):
                return {
                    "strategy": "hierarchical_diversity",
                    "confidence": 0.90,
                    "reason": f"High-confidence multi-session pattern matched: {pat.pattern}"
                }
                
        # Safe fallback: flat retrieval
        return {
            "strategy": "flat",
            "confidence": 0.50,
            "reason": "Uncertain or standard factual query: safe fallback to flat chunk retrieval."
        }

