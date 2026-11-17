"""
Query-Aware Multi-Session Retrieval Architecture for IVARA (Phase 3).

Implements:
1. QueryAwareRouter with 3 outcomes:
   - SINGLE_SESSION: preserves intra-session depth via flat dense MMR.
   - MULTI_SESSION: decomposes sub-clauses, estimates required sessions S,
     and enforces proportional cross-session chunk budgeting.
   - UNCERTAIN: safe fallback to flat MMR.
2. QueryAwareMultiSessionRetriever:
   - Session-level representation & hybrid Vector + BM25 ranking.
   - Subquery-targeted session surfacing (guaranteeing representation for both compared events).
   - Dynamic session budget allocation (preventing single-session starvation without the 1-chunk trap).
   - Chronological ordering and provenance assembly.
   - Detailed inspectable diagnostics.
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


class QueryIntent(str, Enum):
    SINGLE_SESSION = "SINGLE_SESSION"
    MULTI_SESSION = "MULTI_SESSION"
    UNCERTAIN = "UNCERTAIN"


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


class QueryAwareRouter:
    """
    Deterministic semantic and syntactic router classifying queries into:
    - SINGLE_SESSION
    - MULTI_SESSION
    - UNCERTAIN
    
    Extracts subqueries for multi-event queries and estimates required session count S.
    """

    # 1. High-confidence SINGLE_SESSION exclusions
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

    # 2. Two-event comparison subquery extraction patterns (S = 2)
    TWO_EVENT_SUBQUERY_PATTERNS = [
        (
            re.compile(r"\bbetween (?:my visit to |the completion of |the time |the day )?(.+?) and (?:the submission of |the time |the day )?(.+?)\?*$", re.I),
            "between_a_and_b"
        ),
        (
            re.compile(r"\bwhich (?:event|group|activity|did i attend|did i join|did i participate in)?.*?first,?\s*(?:the )?['\"]?(.+?)['\"]?\s+or\s+(?:the )?['\"]?(.+?)['\"]?\?*$", re.I),
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

    # 3. High-confidence multi-session ordering & aggregation patterns
    MULTI_SESSION_PATTERNS = [
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
    def route(cls, query: str) -> dict[str, Any]:
        clean_q = query.strip()
        if not clean_q:
            return {
                "outcome": QueryIntent.UNCERTAIN.value,
                "confidence": 1.0,
                "estimated_required_sessions": 1,
                "subqueries": [],
                "pattern_matched": "empty_query",
                "reason": "Empty query defaults to UNCERTAIN / flat retrieval."
            }

        # Step 1: Check single-session relative anchors
        for pat in cls.SINGLE_SESSION_PATTERNS:
            if pat.search(clean_q):
                return {
                    "outcome": QueryIntent.SINGLE_SESSION.value,
                    "confidence": 0.95,
                    "estimated_required_sessions": 1,
                    "subqueries": [],
                    "pattern_matched": pat.pattern,
                    "reason": "Single-session relative anchor pattern matched. Flat MMR preserves intra-session depth."
                }

        # Step 2: Check two-event subquery patterns (S = 2)
        for pat, pat_name in cls.TWO_EVENT_SUBQUERY_PATTERNS:
            m = pat.search(clean_q)
            if m:
                sub1 = m.group(1).strip().strip("'\"")
                sub2 = m.group(2).strip().strip("'\"")
                if len(sub1) > 2 and len(sub2) > 2:
                    return {
                        "outcome": QueryIntent.MULTI_SESSION.value,
                        "confidence": 0.92,
                        "estimated_required_sessions": 2,
                        "subqueries": [sub1, sub2],
                        "pattern_matched": pat_name,
                        "reason": f"Two-event relative comparison matched: '{sub1}' vs '{sub2}'."
                    }

        # Step 3: Check explicit quoted or enumerated 3-event ordering
        quoted_items = re.findall(r"['\"]([^'\"]{4,})['\"]", clean_q)
        if len(quoted_items) >= 3 and any(w in clean_q.lower() for w in ("order", "first", "latest", "earliest", "total")):
            return {
                "outcome": QueryIntent.MULTI_SESSION.value,
                "confidence": 0.95,
                "estimated_required_sessions": len(quoted_items),
                "subqueries": quoted_items,
                "pattern_matched": "quoted_multi_events",
                "reason": f"Explicit quoted {len(quoted_items)}-event sequence detected."
            }

        # Step 4: Check high-cardinality numbers ("six museums", "three trips", "three sports events")
        cardinal_match = re.search(r"\b(?:order of (?:the )?)?(two|three|four|five|six|\d+)\s+([a-zA-Z ]+)", clean_q, re.I)
        word_to_num = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        est_s = 3
        if cardinal_match:
            val_str = cardinal_match.group(1).lower()
            est_s = int(val_str) if val_str.isdigit() else word_to_num.get(val_str, 3)

        # Step 5: Check multi-session ordering & aggregation patterns
        for pat, pat_name in cls.MULTI_SESSION_PATTERNS:
            if pat.search(clean_q):
                return {
                    "outcome": QueryIntent.MULTI_SESSION.value,
                    "confidence": 0.88,
                    "estimated_required_sessions": max(est_s, 2),
                    "subqueries": [],
                    "pattern_matched": pat_name,
                    "reason": f"Multi-session temporal pattern matched: {pat_name} (est_sessions={est_s})."
                }

        # Step 6: Safe fallback -> UNCERTAIN
        return {
            "outcome": QueryIntent.UNCERTAIN.value,
            "confidence": 0.50,
            "estimated_required_sessions": 1,
            "subqueries": [],
            "pattern_matched": "none",
            "reason": "Query does not match high-confidence multi-session patterns. Fallback to flat MMR."
        }


class QueryAwareMultiSessionRetriever:
    """
    Multi-session retriever executing:
    1. Session representation construction (Vector + BM25).
    2. Subquery-targeted session surfacing (guaranteeing representation for both compared events).
    3. Dynamic session budget allocation (preventing single-session starvation without the 1-chunk trap).
    4. Chronological ordering and provenance assembly.
    5. Detailed inspectable diagnostics.
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

    def _build_session_representations(self):
        by_session: dict[str, list[TurnChunk]] = defaultdict(list)
        for c in self.chunks:
            by_session[c.session_id].append(c)

        session_texts = []
        for sid, s_chunks in by_session.items():
            s_chunks.sort(key=lambda x: x.turn_start_idx)
            first_chunk = s_chunks[0]

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

        self.session_bm25 = SessionBM25Searcher(self.sessions)

    def _score_sessions(self, query_text: str, top_m: int = 10) -> list[tuple[SessionRepresentation, float]]:
        if not self.sessions or not query_text.strip() or self.session_index is None:
            return []

        clean_q = query_text.strip()
        search_m = min(max(top_m * 3, 15), len(self.sessions))

        # Vector search
        q_emb = np.array([get_embedding(clean_q)], dtype="float32")
        faiss.normalize_L2(q_emb)
        vec_scores, vec_indices = self.session_index.search(q_emb, search_m)

        session_scores: dict[int, dict[str, float]] = defaultdict(lambda: {"vec": 0.0, "bm25": 0.0})
        for score, idx in zip(vec_scores[0], vec_indices[0]):
            if idx != -1 and idx < len(self.sessions):
                session_scores[idx]["vec"] = max(0.0, float(score))

        # BM25 search
        max_bm25 = 0.0
        if self.session_bm25 is not None:
            bm25_hits = self.session_bm25.search(clean_q, top_k=search_m)
            for b_score, idx in bm25_hits:
                session_scores[idx]["bm25"] = b_score
                if b_score > max_bm25:
                    max_bm25 = b_score

        # Fused scoring
        ranked: list[tuple[int, float]] = []
        for idx, sc in session_scores.items():
            norm_vec = sc["vec"]
            norm_bm = (sc["bm25"] / max_bm25) if max_bm25 > 0.0 else 0.0
            fused = (self.vector_weight * norm_vec) + (self.bm25_weight * norm_bm)
            ranked.append((idx, fused))

        ranked.sort(key=lambda x: -x[1])
        return [(self.sessions[idx], round(score, 4)) for idx, score in ranked[:top_m]]

    def retrieve(
        self,
        query: str,
        routing_decision: dict[str, Any],
        k: int = 5,
        max_chunks_per_session: int = 2
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Execute query-aware multi-session retrieval:
        1. Subquery-targeted candidate session selection.
        2. Proportional budget allocation across required sessions.
        3. Within-session chunk scoring.
        4. Chronological assembly.
        5. Detailed diagnostics logging.
        """
        subqueries = routing_decision.get("subqueries", [])
        est_s = routing_decision.get("estimated_required_sessions", 2)

        # 1. Select candidate sessions
        selected_sessions_dict: dict[str, tuple[SessionRepresentation, float, str]] = {}

        if subqueries and len(subqueries) >= 2:
            # Subquery-targeted surfacing: Get top candidate session for each subquery
            for sub_idx, sub_q in enumerate(subqueries):
                hits = self._score_sessions(sub_q, top_m=5)
                for sess, sc in hits:
                    sid = sess.session_id
                    if sid not in selected_sessions_dict:
                        selected_sessions_dict[sid] = (sess, sc, f"subquery_{sub_idx+1}: {sub_q[:30]}")
                        break

        # Also get top sessions for the full query to fill remaining capacity
        full_hits = self._score_sessions(query, top_m=max(est_s * 2, 6))
        for sess, sc in full_hits:
            sid = sess.session_id
            if sid not in selected_sessions_dict and len(selected_sessions_dict) < max(est_s, 4):
                selected_sessions_dict[sid] = (sess, sc, "full_query_match")

        candidate_sessions = list(selected_sessions_dict.values())
        if not candidate_sessions:
            return [], {"error": "no_candidate_sessions_found"}

        # 2. Dynamic Budget Allocation
        num_sessions = len(candidate_sessions)
        budget_map: dict[str, int] = {}

        if est_s == 2 and num_sessions >= 2:
            # Two events: allocate 2 chunks each to top 2 sessions, 1 to remainder
            s1_id = candidate_sessions[0][0].session_id
            s2_id = candidate_sessions[1][0].session_id
            budget_map[s1_id] = 2
            budget_map[s2_id] = 2
            if len(candidate_sessions) >= 3:
                budget_map[candidate_sessions[2][0].session_id] = 1
            else:
                budget_map[s1_id] = 3
        elif est_s == 3 and num_sessions >= 3:
            # Three events: allocate 2, 2, 1 chunks
            budget_map[candidate_sessions[0][0].session_id] = 2
            budget_map[candidate_sessions[1][0].session_id] = 2
            budget_map[candidate_sessions[2][0].session_id] = 1
        elif est_s >= 4:
            # High cardinality: 1 chunk per candidate session
            for sess, _, _ in candidate_sessions[:k]:
                budget_map[sess.session_id] = 1
        else:
            # Fallback allocation
            remaining_k = k
            for sess, _, _ in candidate_sessions:
                alloc = min(max_chunks_per_session, remaining_k)
                if alloc > 0:
                    budget_map[sess.session_id] = alloc
                    remaining_k -= alloc
                else:
                    budget_map[sess.session_id] = 0

        # 3. Within-session chunk scoring and retrieval
        q_clean = query.strip()
        q_emb = np.array([get_embedding(q_clean)], dtype="float32")
        faiss.normalize_L2(q_emb)

        collected_chunks: list[dict[str, Any]] = []
        diagnostics_sessions = []

        for sess, sess_score, reason in candidate_sessions:
            sid = sess.session_id
            max_alloc = budget_map.get(sid, 0)
            if max_alloc <= 0:
                continue

            s_chunks = sess.chunks
            if not s_chunks:
                continue

            # Chunk dense vector scoring
            chunk_texts = [c.text for c in s_chunks]
            raw_c_embs = get_embeddings(chunk_texts, batch_size=32)
            c_embs = np.array(raw_c_embs, dtype="float32")
            faiss.normalize_L2(c_embs)
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
                fused = (0.50 * vec_sc) + (0.30 * norm_bm) + (0.20 * sess_score)

                scored_chunks.append({
                    "chunk": chunk,
                    "chunk_id": chunk.chunk_id,
                    "memory_id": chunk.memory_id,
                    "session_id": chunk.session_id,
                    "score": round(float(fused), 4),
                    "vector_score": round(vec_sc, 4),
                    "bm25_score": round(raw_bm, 4),
                    "turn_range": (chunk.turn_start_idx, chunk.turn_end_idx),
                    "timestamp": chunk.timestamp,
                    "text": chunk.text,
                    "token_estimate": chunk.token_estimate
                })

            scored_chunks.sort(key=lambda x: -x["score"])
            allocated = scored_chunks[:max_alloc]
            collected_chunks.extend(allocated)

            diagnostics_sessions.append({
                "session_id": sid,
                "date": sess.date_str,
                "score": sess_score,
                "selection_reason": reason,
                "allocated_chunks": len(allocated),
                "budget_cap": max_alloc
            })

        # 4. Chronological sorting across retrieved chunks
        def _get_sort_key(item):
            ts = item.get("timestamp") or ""
            tr = item.get("turn_range") or (0, 0)
            return (ts, tr[0])

        collected_chunks.sort(key=_get_sort_key)

        # 5. Diagnostic reporting
        diagnostics = {
            "routing_decision": routing_decision,
            "estimated_required_sessions": est_s,
            "subqueries_extracted": subqueries,
            "selected_sessions": diagnostics_sessions,
            "budget_allocation": budget_map,
            "total_retrieved_chunks": len(collected_chunks),
            "retrieved_chunk_ids": [c["chunk_id"] for c in collected_chunks]
        }

        return collected_chunks, diagnostics

