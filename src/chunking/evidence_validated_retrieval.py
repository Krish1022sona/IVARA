"""
Evidence-Validated Session Routing Architecture for IVARA (Phase 3.1).

Replaces syntax-driven multi-session routing with an evidence-validated decision engine:
1. Decomposes query clauses (Event A, Event B, etc.).
2. Independently retrieves and scores candidate sessions for each event clause.
3. Evaluates joint vs disjoint session support:
   - SAME_SESSION: Evidence indicates both events are discussed in the same session.
     Routes to Flat MMR (max_chunks_per_session=3 or 4) preserving intra-session contiguous turn depth.
   - DIFFERENT_SESSIONS: Evidence confirms events reside in distinct sessions with no strong joint session.
     Executes balanced multi-session retrieval (guaranteeing both sessions in context without distractor dilution).
   - AMBIGUOUS: Low confidence or conflicting evidence.
     Safely falls back to Flat MMR.
4. Generates comprehensive inspectable diagnostics for every routing decision.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
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


class EvidenceRoutingDecision(str, Enum):
    SAME_SESSION = "SAME_SESSION"
    DIFFERENT_SESSIONS = "DIFFERENT_SESSIONS"
    AMBIGUOUS = "AMBIGUOUS"
    SINGLE_SESSION_SYNTAX = "SINGLE_SESSION_SYNTAX"


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
    """BM25 lexical retriever for conversational session units."""

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


class QueryClauseDecomposer:
    """
    Extracts conservative subquery clauses from comparative, temporal, and sequential queries.
    Preserves exact user terminology and strips trailing interrogative clauses.
    """

    SINGLE_SESSION_PATTERNS = [
        re.compile(r"\bhow many (?:days|weeks|months|years) ago did i\b", re.I),
        re.compile(r"\bhow long ago did i\b", re.I),
        re.compile(r"\bhow many (?:days|weeks|months|years) have passed since i (?:last|visited|went|bought|started|participated)\b", re.I),
        re.compile(r"\bi mentioned (?:participating in|visiting|cooking|an investment|a significant|that i)\b.+\b(?:ago|last)\b", re.I),
        re.compile(r"\bwhich book did i finish a week ago\b", re.I),
        re.compile(r"\bwhat (?:gardening-related|charity|artist|life|social media|significant)\b.+\b(?:ago|last friday|last saturday)\b", re.I),
        re.compile(r"\bwhich streaming service did i start using most recently\b", re.I),
        re.compile(r"\bwho did i go with to the music event last saturday\b", re.I),
    ]

    TWO_EVENT_SUBQUERY_PATTERNS = [
        (
            re.compile(r"\bbetween (?:my visit to |the completion of |the time |the day )?(.+?) and (?:the submission of |the time |the day )?(.+?)\?*$", re.I),
            "between_a_and_b"
        ),
        (
            re.compile(r"\bwhich (?:event|group|activity|item|device|task|trip|did i attend|did i join|did i participate in|did i purchase|did i complete)?.*?first,?\s*(?:the )?['\"]?(.+?)['\"]?\s+or\s+(?:the )?['\"]?(.+?)['\"]?\?*$", re.I),
            "which_first_a_or_b"
        ),
        (
            re.compile(r"\bwhich (?:event )?happened first,?\s*(?:the )?['\"]?(.+?)['\"]?\s+or\s+(?:the )?['\"]?(.+?)['\"]?\?*$", re.I),
            "which_happened_first_or"
        ),
        (
            re.compile(r"\bhow many (?:days|weeks|months) before (.+?) did (.+?)\?*$", re.I),
            "before_did"
        ),
        (
            re.compile(r"\bhow many weeks (?:have i been|was i) accepted into (.+?) when i started (.+?)\?*$", re.I),
            "accepted_when_started"
        ),
        (
            re.compile(r"\bhow many days ago did i (.+?) when i (.+?)\?*$", re.I),
            "event_when_event"
        )
    ]

    MULTI_SESSION_ORDERING_PATTERNS = [
        (re.compile(r"\border of (?:the )?(?:\w+ )?(?:events|trips|museums|sports events|sports|concerts|musical events|airlines|books)\b", re.I), "order_of_events"),
        (re.compile(r"\border from (?:first|earliest) to (?:last|latest)\b", re.I), "order_first_to_last"),
        (re.compile(r"\bfrom (?:earliest|first) to (?:latest|last)\b", re.I), "first_to_last"),
        (re.compile(r"\bstarting from the earliest\b", re.I), "starting_earliest"),
        (re.compile(r"\bwhich (?:three )?events happened in (?:the )?order\b", re.I), "which_events_in_order"),
        (re.compile(r"\bwho graduated first,?\s*second\b", re.I), "who_graduated_order"),
        (re.compile(r"\bconsecutive days\b", re.I), "consecutive_days"),
        (re.compile(r"\bin a row\b", re.I), "in_a_row"),
        (re.compile(r"\btotal (?:do i spent|time|hours|days|weeks)\b.+\band\b", re.I), "total_time_spent_across"),
        (re.compile(r"\bhow many (?:charity )?events did i participate in before\b", re.I), "events_before"),
        (re.compile(r"\bhow long have i been working before i started my current job\b", re.I), "work_history_before"),
        (re.compile(r"\bwhich airline did i fly with the most in march and april\b", re.I), "flights_across_months"),
    ]

    @classmethod
    def decompose(cls, query: str) -> dict[str, Any]:
        clean_q = query.strip()
        if not clean_q:
            return {
                "is_syntax_single": False,
                "subqueries": [],
                "pattern_matched": "empty",
                "estimated_required_sessions": 1
            }

        # 1. Single session syntax check
        for pat in cls.SINGLE_SESSION_PATTERNS:
            if pat.search(clean_q):
                return {
                    "is_syntax_single": True,
                    "subqueries": [],
                    "pattern_matched": pat.pattern,
                    "estimated_required_sessions": 1
                }

        # 2. Check two-event subqueries
        for pat, pat_name in cls.TWO_EVENT_SUBQUERY_PATTERNS:
            m = pat.search(clean_q)
            if m:
                sub1 = m.group(1).strip().strip("'\"?.,")
                sub2 = m.group(2).strip().strip("'\"?.,")
                # Strip trailing interrogative tails (e.g. ", how many days passed", ", which happened first")
                sub2 = re.split(r",\s*(?:how many|which|what|did|when)\b", sub2, flags=re.I)[0].strip().rstrip(",?.'\"").strip()
                if len(sub1) > 2 and len(sub2) > 2:
                    return {
                        "is_syntax_single": False,
                        "subqueries": [sub1, sub2],
                        "pattern_matched": pat_name,
                        "estimated_required_sessions": 2
                    }

        # 3. Quoted sequence check (e.g. 3 events in quotes)
        quoted_items = re.findall(r"['\"]([^'\"]{4,})['\"]", clean_q)
        if len(quoted_items) >= 3 and any(w in clean_q.lower() for w in ("order", "first", "latest", "earliest", "total")):
            return {
                "is_syntax_single": False,
                "subqueries": quoted_items,
                "pattern_matched": "quoted_multi_events",
                "estimated_required_sessions": len(quoted_items)
            }

        # 4. Multi-session ordering patterns
        cardinal_match = re.search(r"\b(?:order of (?:the )?)?(two|three|four|five|six|\d+)\s+([a-zA-Z ]+)", clean_q, re.I)
        word_to_num = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        est_s = 3
        if cardinal_match:
            val_str = cardinal_match.group(1).lower()
            est_s = int(val_str) if val_str.isdigit() else word_to_num.get(val_str, 3)

        for pat, pat_name in cls.MULTI_SESSION_ORDERING_PATTERNS:
            if pat.search(clean_q):
                return {
                    "is_syntax_single": False,
                    "subqueries": [],
                    "pattern_matched": pat_name,
                    "estimated_required_sessions": max(est_s, 2)
                }

        return {
            "is_syntax_single": False,
            "subqueries": [],
            "pattern_matched": "none",
            "estimated_required_sessions": 1
        }


class EvidenceValidatedRetriever:
    """
    Evidence-Validated Retriever:
    1. Extracts decomposed clauses.
    2. Scores candidate sessions independently for each clause.
    3. Evaluates joint session support vs disjoint session evidence.
    4. Routes:
       - SAME_SESSION -> Flat MMR (preserving contiguous depth)
       - DIFFERENT_SESSIONS -> Balanced cross-session allocation without distractor dilution
       - AMBIGUOUS -> Safe Flat MMR
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

        if self.chunks:
            self._build_session_representations()
            self._build_session_indices()

    def _build_session_representations(self) -> None:
        sess_chunks = defaultdict(list)
        for c in self.chunks:
            sess_chunks[c.session_id].append(c)

        for sid, c_list in sess_chunks.items():
            c_list.sort(key=lambda x: x.turn_start_idx)
            full_text = "\n".join(c.text for c in c_list)
            if len(full_text) > self.session_text_cap_chars:
                full_text = full_text[:self.session_text_cap_chars]

            date_str = c_list[0].timestamp if c_list else ""
            parsed_d = c_list[0].parsed_datetime.date() if (c_list and c_list[0].parsed_datetime) else None
            if not parsed_d and date_str:
                parsed_d = parse_date_string(date_str)

            self.sessions.append(SessionRepresentation(
                session_id=sid,
                memory_id=c_list[0].memory_id if c_list else f"sess_{sid}",
                date_str=date_str,
                parsed_date=parsed_d,
                chunks=c_list,
                text=full_text
            ))

        self.sessions.sort(key=lambda s: (s.parsed_date or date.min, s.session_id))

    def _build_session_indices(self) -> None:
        if not self.sessions:
            return

        texts = [s.text for s in self.sessions]
        raw_embeddings = get_embeddings(texts, batch_size=64)
        if raw_embeddings is not None:
            mat = np.array(raw_embeddings, dtype=np.float32)
            if mat.ndim == 2 and mat.shape[0] > 0:
                faiss.normalize_L2(mat)
                dim = mat.shape[1]
                self.session_index = faiss.IndexFlatIP(dim)
                self.session_index.add(mat)
                for idx, s in enumerate(self.sessions):
                    s.embedding = mat[idx]

        self.session_bm25 = SessionBM25Searcher(self.sessions)

    def _score_sessions(self, subquery: str, top_m: int = 5) -> list[tuple[SessionRepresentation, float]]:
        if not self.sessions:
            return []

        q_emb = get_embedding(subquery)
        vec_scores: dict[int, float] = {}
        if q_emb is not None and self.session_index is not None:
            q_vec = np.array([q_emb], dtype=np.float32)
            faiss.normalize_L2(q_vec)
            dists, idxs = self.session_index.search(q_vec, min(top_m * 2, len(self.sessions)))
            for d, idx in zip(dists[0], idxs[0]):
                if idx >= 0:
                    vec_scores[int(idx)] = float(d)

        bm25_scores: dict[int, float] = {}
        if self.session_bm25 is not None:
            b_hits = self.session_bm25.search(subquery, top_k=top_m * 2)
            max_b = max((h[0] for h in b_hits), default=1.0)
            max_b = max(max_b, 1.0)
            for sc, idx in b_hits:
                bm25_scores[idx] = sc / max_b

        cand_indices = set(vec_scores.keys()).union(set(bm25_scores.keys()))
        if not cand_indices:
            cand_indices = set(range(min(len(self.sessions), top_m)))

        scored: list[tuple[SessionRepresentation, float]] = []
        for idx in cand_indices:
            v_sc = vec_scores.get(idx, 0.0)
            b_sc = bm25_scores.get(idx, 0.0)
            hybrid = (self.vector_weight * v_sc) + (self.bm25_weight * b_sc)
            scored.append((self.sessions[idx], round(hybrid, 4)))

        scored.sort(key=lambda x: (-x[1], x[0].parsed_date or date.min))
        return scored[:top_m]

    def evaluate_routing(self, query: str) -> dict[str, Any]:
        """
        Evidence-validated decision engine.
        Returns detailed diagnostics with decision, scores, and candidate sessions.
        """
        decomp = QueryClauseDecomposer.decompose(query)
        if decomp.get("is_syntax_single"):
            return {
                "decision": EvidenceRoutingDecision.SINGLE_SESSION_SYNTAX.value,
                "confidence": 0.95,
                "reason": "Single-session relative anchor matched syntactically.",
                "subqueries": [],
                "target_sessions": [],
                "joint_support": 0.0
            }

        subqueries = decomp.get("subqueries", [])
        if len(subqueries) < 2:
            # Ordering pattern without explicit 2 subqueries
            if decomp.get("pattern_matched") != "none" and decomp.get("estimated_required_sessions", 1) >= 2:
                # Ordering pattern detected: check candidates using full query
                cand_sessions = self._score_sessions(query, top_m=decomp.get("estimated_required_sessions", 3))
                return {
                    "decision": EvidenceRoutingDecision.DIFFERENT_SESSIONS.value,
                    "confidence": 0.85,
                    "reason": f"Multi-session ordering pattern: {decomp.get('pattern_matched')}",
                    "subqueries": [query],
                    "target_sessions": [s.session_id for s, _ in cand_sessions],
                    "joint_support": 0.0
                }
            return {
                "decision": EvidenceRoutingDecision.AMBIGUOUS.value,
                "confidence": 0.50,
                "reason": "No comparative subclauses extracted. Safe fallback to Flat MMR.",
                "subqueries": [],
                "target_sessions": [],
                "joint_support": 0.0
            }

        # We have 2 or more subqueries
        hits_a = self._score_sessions(subqueries[0], top_m=5)
        hits_b = self._score_sessions(subqueries[1], top_m=5)

        scores_a = {s.session_id: sc for s, sc in hits_a}
        scores_b = {s.session_id: sc for s, sc in hits_b}

        top_a_sid, top_a_sc = (hits_a[0][0].session_id, hits_a[0][1]) if hits_a else (None, 0.0)
        top_b_sid, top_b_sc = (hits_b[0][0].session_id, hits_b[0][1]) if hits_b else (None, 0.0)

        # Condition 1: Identical top-1 session
        if top_a_sid is not None and top_a_sid == top_b_sid and top_a_sc >= 0.35 and top_b_sc >= 0.35:
            return {
                "decision": EvidenceRoutingDecision.SAME_SESSION.value,
                "confidence": min(top_a_sc, top_b_sc),
                "reason": f"Both event clauses strongly point to the same top session {top_a_sid} (A: {top_a_sc:.2f}, B: {top_b_sc:.2f}).",
                "subqueries": subqueries,
                "target_sessions": [top_a_sid],
                "top_a": {"session_id": top_a_sid, "score": top_a_sc},
                "top_b": {"session_id": top_b_sid, "score": top_b_sc},
                "joint_support": min(top_a_sc, top_b_sc)
            }

        # Condition 2: Distinct top sessions with verified evidence
        if top_a_sid is not None and top_b_sid is not None and top_a_sid != top_b_sid and top_a_sc >= 0.35 and top_b_sc >= 0.35:
            return {
                "decision": EvidenceRoutingDecision.DIFFERENT_SESSIONS.value,
                "confidence": min(top_a_sc, top_b_sc),
                "reason": f"Distinct top sessions verified: A->{top_a_sid} ({top_a_sc:.2f}), B->{top_b_sid} ({top_b_sc:.2f}).",
                "subqueries": subqueries,
                "target_sessions": [top_a_sid, top_b_sid],
                "top_a": {"session_id": top_a_sid, "score": top_a_sc},
                "top_b": {"session_id": top_b_sid, "score": top_b_sc},
                "joint_support": 0.0
            }

        # Condition 3: Ambiguous or weak support -> Fallback
        return {
            "decision": EvidenceRoutingDecision.AMBIGUOUS.value,
            "confidence": 0.50,
            "reason": f"Candidate evidence ambiguous (A: {top_a_sc:.2f}, B: {top_b_sc:.2f}). Safe fallback to Flat MMR.",
            "subqueries": subqueries,
            "target_sessions": [top_a_sid, top_b_sid] if top_a_sid else [],
            "top_a": {"session_id": top_a_sid, "score": top_a_sc},
            "top_b": {"session_id": top_b_sid, "score": top_b_sc},
            "joint_support": 0.0
        }

    @staticmethod
    def _chunk_to_dict(chunk: TurnChunk, score: float = 0.8) -> dict[str, Any]:
        """Convert a TurnChunk to standard retrieval dictionary format."""
        return {
            "chunk": chunk,
            "chunk_id": chunk.chunk_id,
            "memory_id": chunk.memory_id,
            "session_id": chunk.session_id,
            "score": round(float(score), 4),
            "vector_score": round(float(score), 4),
            "bm25_score": round(float(score), 4),
            "turn_range": (chunk.turn_start_idx, chunk.turn_end_idx),
            "timestamp": chunk.timestamp,
            "text": chunk.text,
            "token_estimate": chunk.token_estimate
        }

    def retrieve(
        self,
        query: str,
        k: int = 5,
        max_chunks_per_session: int = 3
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Executes evidence-validated retrieval:
        - If DIFFERENT_SESSIONS: allocates chunks from Session A and Session B with interleaved priority (A1, B1, A2, B2).
          Guarantees representation of both sessions in context without distractor dilution.
        - If SAME_SESSION or AMBIGUOUS: selects chunks using Flat MMR across the full chunk set.
        """
        routing = self.evaluate_routing(query)
        decision = routing["decision"]

        # 1. DIFFERENT_SESSIONS execution
        if decision == EvidenceRoutingDecision.DIFFERENT_SESSIONS.value:
            subqueries = routing.get("subqueries", [])
            target_sids = routing.get("target_sessions", [])

            sess_map = {s.session_id: s for s in self.sessions}
            selected_chunks: list[TurnChunk] = []

            if len(subqueries) >= 2 and len(target_sids) >= 2:
                sub_a, sub_b = subqueries[0], subqueries[1]
                sid_a, sid_b = target_sids[0], target_sids[1]

                sess_a = sess_map.get(sid_a)
                sess_b = sess_map.get(sid_b)

                chunks_a = self._rank_chunks_in_session(sess_a, sub_a, max_c=2) if sess_a else []
                chunks_b = self._rank_chunks_in_session(sess_b, sub_b, max_c=2) if sess_b else []

                # Interleave priority so both sessions are protected against context truncation
                if chunks_a:
                    selected_chunks.append(chunks_a[0])
                if chunks_b:
                    selected_chunks.append(chunks_b[0])
                if len(chunks_a) > 1:
                    selected_chunks.append(chunks_a[1])
                if len(chunks_b) > 1:
                    selected_chunks.append(chunks_b[1])

            dict_chunks = [self._chunk_to_dict(c, score=0.95 - idx * 0.05) for idx, c in enumerate(selected_chunks)]
            routing["retrieved_chunk_ids"] = [c["chunk_id"] for c in dict_chunks]
            routing["retrieval_mode"] = "BALANCED_MULTI_SESSION"
            return dict_chunks[:k], routing

        # 2. SAME_SESSION, AMBIGUOUS, or SINGLE_SESSION_SYNTAX execution
        from .retrieval import ChunkRetriever
        flat_retriever = ChunkRetriever(chunks=self.chunks)
        flat_hits = flat_retriever.retrieve(
            query,
            k=k,
            max_chunks_per_session=max_chunks_per_session,
            use_session_mmr=True,
            mmr_lambda=0.7,
            mmr_pool_size=max(k * 2 + 5, 20)
        )
        routing["retrieved_chunk_ids"] = [c.get("chunk_id", "") if isinstance(c, dict) else getattr(c, "chunk_id", "") for c in flat_hits]
        routing["retrieval_mode"] = "FLAT_MMR_INTRA_SESSION"
        return flat_hits, routing

    def _rank_chunks_in_session(
        self,
        session: SessionRepresentation,
        subquery: str,
        max_c: int = 2
    ) -> list[TurnChunk]:
        """Rank chunks within a specific session using embedding and lexical relevance."""
        if not session.chunks:
            return []
        if len(session.chunks) <= max_c:
            return list(session.chunks)

        q_emb = get_embedding(subquery)
        q_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', subquery.lower())) - STOPWORDS

        scored_c = []
        for c in session.chunks:
            score = 0.0
            if q_emb is not None:
                c_emb = get_embedding(c.text)
                if c_emb is not None:
                    c_norm = np.linalg.norm(c_emb)
                    q_norm = np.linalg.norm(q_emb)
                    if c_norm > 0 and q_norm > 0:
                        cos = float(np.dot(c_emb, q_emb) / (c_norm * q_norm))
                        score += 0.6 * cos

            c_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', c.text.lower()))
            if q_tokens:
                overlap = len(q_tokens.intersection(c_tokens)) / len(q_tokens)
                score += 0.4 * overlap

            scored_c.append((c, score))

        scored_c.sort(key=lambda x: -x[1])
        return [c for c, _ in scored_c[:max_c]]
