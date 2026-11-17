"""Episodic Retrieval Engine for IVARA.

Provides query-driven semantic and hybrid search over verbatim conversational turns
stored in ConversationStore or in-memory turn collections.
Supports speaker attribution, conversation/session filtering, temporal bounds,
and chronological context-window expansion around relevant dialogue turns.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Optional
import faiss
import numpy as np

from ..conversations.store import ConversationStore, EpisodicTurn
from ..embedding import get_embedding


@dataclass
class EpisodicEvidenceItem:
    """Represents a retrieved episodic conversation turn enriched with context and scoring."""
    turn_id: str
    conversation_id: str
    session_id: Optional[str]
    speaker: str
    speaker_role: str
    content: str
    timestamp: str
    sequence_num: int
    score: float
    channel: str = "episodic"
    context_window: list[dict[str, Any]] = field(default_factory=list)
    formatted_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "session_id": self.session_id,
            "speaker": self.speaker,
            "speaker_role": self.speaker_role,
            "content": self.content,
            "timestamp": self.timestamp,
            "sequence_num": self.sequence_num,
            "score": round(self.score, 4),
            "channel": self.channel,
            "context_window": self.context_window,
            "formatted_text": self.formatted_text,
            "metadata": self.metadata
        }


class BM25TurnRetriever:
    """In-memory BM25 index over conversational turns for exact keywords, names, and codes."""

    def __init__(self, corpus: list[dict[str, Any]]):
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.avg_doc_len = 0.0
        self.doc_tokens = []
        self.doc_freqs: dict[str, int] = {}

        if self.doc_count > 0:
            total_len = 0
            for doc in corpus:
                tokens = self._tokenize(doc.get("searchable_text", ""))
                self.doc_tokens.append(tokens)
                total_len += len(tokens)
                for t in set(tokens):
                    self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
            self.avg_doc_len = total_len / self.doc_count

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower() for t in re.findall(r"\b[a-zA-Z0-9_\-+#]{2,}\b", text)]

    def search(self, query: str, top_k: int = 20) -> list[tuple[float, int]]:
        q_tokens = self._tokenize(query)
        if not q_tokens or self.doc_count == 0:
            return []

        scores = []
        k1 = 1.5
        b = 0.75

        for idx, doc_toks in enumerate(self.doc_tokens):
            doc_len = len(doc_toks)
            score = 0.0
            tok_counts: dict[str, int] = {}
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


class EpisodicRetriever:
    """
    Query-driven retriever over episodic conversational turns.
    Combines dense FAISS cosine similarity and lexical matching with speaker context
    and bounded chronological context-window expansion.
    """

    def __init__(
        self,
        store: Optional[ConversationStore] = None,
        turns: Optional[list[EpisodicTurn]] = None,
        k_rrf: int = 60,
        weight_vector: float = 1.0,
        weight_lexical: float = 0.8,
        user_name: Optional[str] = None
    ):
        self.store = store
        self.turns: list[EpisodicTurn] = []
        self.k_rrf = k_rrf
        self.weight_vector = weight_vector
        self.weight_lexical = weight_lexical
        self.user_name = user_name

        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bm25: Optional[BM25TurnRetriever] = None
        self.turn_records: list[dict[str, Any]] = []

        if turns is not None:
            self.turns = list(turns)
        elif self.store is not None:
            # Load all episodic turns from the persistent store
            # Order by conversation and sequence
            conn = self.store._get_connection()
            cur = conn.execute("SELECT * FROM episodic_turns ORDER BY sequence_num ASC")
            rows = cur.fetchall()
            import json
            for r in rows:
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
                self.turns.append(EpisodicTurn(
                    turn_id=r["turn_id"],
                    conversation_id=r["conversation_id"],
                    speaker=r["speaker"],
                    speaker_role=r["speaker_role"],
                    content=r["content"],
                    timestamp=r["timestamp"],
                    sequence_num=r["sequence_num"],
                    prev_turn_id=r["prev_turn_id"],
                    next_turn_id=r["next_turn_id"],
                    session_id=r["session_id"],
                    metadata=meta
                ))

        self._build_indices()

    def _format_turn_text(self, turn: EpisodicTurn) -> str:
        """Format turn with explicit speaker attribution and timestamp."""
        ts_part = f"[{turn.timestamp}] " if turn.timestamp else ""
        speaker_name = turn.speaker
        if not speaker_name or speaker_name.lower() in ("user", "unknown"):
            default_user = self.user_name if self.user_name else "the User"
            speaker_name = default_user if turn.speaker_role == "user" else "IVARA"
        return f"{ts_part}Speaker: {speaker_name} ({turn.speaker_role}): {turn.content}"

    def _build_indices(self):
        """Construct FAISS dense vector index and BM25 lexical index."""
        if not self.turns:
            return

        self.turn_records = []
        texts_to_embed = []

        for t in self.turns:
            formatted = self._format_turn_text(t)
            searchable = f"{formatted} {t.content}"
            self.turn_records.append({
                "turn": t,
                "formatted_text": formatted,
                "searchable_text": searchable
            })
            texts_to_embed.append(formatted)

        # 1. Vector index
        embeddings = [get_embedding(txt) for txt in texts_to_embed]
        arr = np.array(embeddings, dtype="float32")
        if arr.ndim == 2 and arr.shape[0] > 0:
            faiss.normalize_L2(arr)
            dim = arr.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(arr)

        # 2. Lexical index
        self.bm25 = BM25TurnRetriever(self.turn_records)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.30,
        conversation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        speaker: Optional[str] = None,
        speaker_role: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        window_size: int = 1
    ) -> list[EpisodicEvidenceItem]:
        """
        Execute filtered hybrid retrieval over episodic conversation turns.
        Expands chronological context windows around top qualifying hits.
        """
        clean_q = query.strip()
        if not clean_q or not self.turns or self.faiss_index is None:
            return []

        # 1. Vector Search
        q_emb = np.array([get_embedding(clean_q)], dtype="float32")
        faiss.normalize_L2(q_emb)

        search_k = min(len(self.turns), max(k * 4, 20))
        sims, indices = self.faiss_index.search(q_emb, search_k)

        vec_ranks: dict[int, int] = {}
        vec_scores: dict[int, float] = {}
        for rank_idx, (sim, doc_idx) in enumerate(zip(sims[0], indices[0]), 1):
            if 0 <= doc_idx < len(self.turns):
                vec_ranks[doc_idx] = rank_idx
                vec_scores[doc_idx] = float(sim)

        # 2. Lexical Search
        lex_ranks: dict[int, int] = {}
        lex_scores: dict[int, float] = {}
        if self.bm25 is not None:
            lex_hits = self.bm25.search(clean_q, top_k=search_k)
            for rank_idx, (b_score, doc_idx) in enumerate(lex_hits, 1):
                lex_ranks[doc_idx] = rank_idx
                lex_scores[doc_idx] = float(b_score)

        # 3. Candidate Scoring with RRF & Metadata Filtering
        all_candidate_indices = set(vec_ranks.keys()) | set(lex_ranks.keys())
        scored_candidates: list[tuple[float, int, float]] = []

        for doc_idx in all_candidate_indices:
            turn = self.turns[doc_idx]

            # Apply metadata filters
            if conversation_id and turn.conversation_id != conversation_id:
                continue
            if session_id and turn.session_id != session_id:
                continue
            if speaker and turn.speaker.lower() != speaker.lower():
                continue
            if speaker_role and turn.speaker_role.lower() != speaker_role.lower():
                continue
            if start_time and turn.timestamp < start_time:
                continue
            if end_time and turn.timestamp > end_time:
                continue

            # Calculate Reciprocal Rank Fusion score
            rrf_vec = (self.weight_vector / (self.k_rrf + vec_ranks[doc_idx])) if doc_idx in vec_ranks else 0.0
            rrf_lex = (self.weight_lexical / (self.k_rrf + lex_ranks[doc_idx])) if doc_idx in lex_ranks else 0.0
            fused_score = rrf_vec + rrf_lex

            # Cosine similarity for semantic threshold gating
            sim_score = vec_scores.get(doc_idx, 0.0)

            # Minimum score gate: require either sufficient semantic similarity or strong lexical match
            if sim_score < min_score and doc_idx not in lex_ranks:
                continue

            scored_candidates.append((fused_score, doc_idx, sim_score))

        # Sort by fused score descending
        scored_candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)

        # 4. Context Window Expansion and Packaging
        results: list[EpisodicEvidenceItem] = []
        selected_turn_ids: set[str] = set()

        for fused_score, doc_idx, sim_score in scored_candidates:
            if len(results) >= k:
                break

            turn = self.turns[doc_idx]
            if turn.turn_id in selected_turn_ids:
                continue
            selected_turn_ids.add(turn.turn_id)

            # Expand context window
            context_window_items = []
            if window_size > 0:
                if self.store is not None:
                    # Use store's transactional window query
                    window_turns = self.store.get_context_window(turn.turn_id, window_size=window_size)
                    for wt in window_turns:
                        rel_tag = "target" if wt.turn_id == turn.turn_id else (
                            "preceding" if wt.sequence_num < turn.sequence_num else "succeeding"
                        )
                        context_window_items.append({
                            "turn_id": wt.turn_id,
                            "speaker": wt.speaker,
                            "speaker_role": wt.speaker_role,
                            "content": wt.content,
                            "timestamp": wt.timestamp,
                            "sequence_num": wt.sequence_num,
                            "relation": rel_tag,
                            "formatted": self._format_turn_text(wt)
                        })
                else:
                    # In-memory window calculation across self.turns in same conversation
                    conv_turns = [
                        t for t in self.turns
                        if t.conversation_id == turn.conversation_id
                    ]
                    conv_turns.sort(key=lambda t: t.sequence_num)
                    curr_pos = next((i for i, t in enumerate(conv_turns) if t.turn_id == turn.turn_id), -1)
                    if curr_pos != -1:
                        start_pos = max(0, curr_pos - window_size)
                        end_pos = min(len(conv_turns), curr_pos + window_size + 1)
                        for wt in conv_turns[start_pos:end_pos]:
                            rel_tag = "target" if wt.turn_id == turn.turn_id else (
                                "preceding" if wt.sequence_num < turn.sequence_num else "succeeding"
                            )
                            context_window_items.append({
                                "turn_id": wt.turn_id,
                                "speaker": wt.speaker,
                                "speaker_role": wt.speaker_role,
                                "content": wt.content,
                                "timestamp": wt.timestamp,
                                "sequence_num": wt.sequence_num,
                                "relation": rel_tag,
                                "formatted": self._format_turn_text(wt)
                            })

            record = self.turn_records[doc_idx]
            # Primary score exposed is the cosine similarity (or fused score if pure lexical)
            display_score = sim_score if sim_score > 0.0 else fused_score

            results.append(EpisodicEvidenceItem(
                turn_id=turn.turn_id,
                conversation_id=turn.conversation_id,
                session_id=turn.session_id,
                speaker=turn.speaker,
                speaker_role=turn.speaker_role,
                content=turn.content,
                timestamp=turn.timestamp,
                sequence_num=turn.sequence_num,
                score=display_score,
                channel="episodic",
                context_window=context_window_items,
                formatted_text=record["formatted_text"],
                metadata=turn.metadata
            ))

        return results
