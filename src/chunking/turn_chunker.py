"""
Turn-Aware Conversational Chunker (TACC) for IVARA.

Implements principled, turn-level sliding window partitioning for multi-turn
conversational sessions. Preserves turn boundaries, speaker roles, temporal
timestamps, and session provenance while eliminating vector truncation.
"""

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any


@dataclass
class TurnChunk:
    """
    A bounded window of conversational turns within a single session.
    """
    chunk_id: str                      # e.g. "sess_280352e9_c001"
    session_id: str                    # Original session identifier
    memory_id: str                     # e.g. "sess_280352e9"
    turn_start_idx: int                # 0-indexed start turn in session
    turn_end_idx: int                  # 0-indexed inclusive end turn in session
    total_session_turns: int           # Total turns in parent session
    timestamp: str                     # Session date string
    parsed_datetime: datetime | None   # Parsed ISO datetime if available
    turns: list[dict[str, str]]        # [{"role": "user", "content": "..."}, ...]
    text: str                          # Formatted dialogue text
    word_count: int                    # Word count of formatted text
    token_estimate: int                # Estimated token count (~words / 0.75)
    is_long_turn: bool = False         # True if any turn exceeds threshold
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "memory_id": self.memory_id,
            "turn_start_idx": self.turn_start_idx,
            "turn_end_idx": self.turn_end_idx,
            "total_session_turns": self.total_session_turns,
            "timestamp": self.timestamp,
            "text": self.text,
            "word_count": self.word_count,
            "token_estimate": self.token_estimate,
            "is_long_turn": self.is_long_turn,
            "metadata": self.metadata
        }


class TurnAwareChunker:
    """
    Partitions conversational sessions into turn-aware windows.
    Default: 3-turn window with 1-turn overlap (step size 2).
    """

    def __init__(
        self,
        window_size: int = 3,
        overlap: int = 1,
        long_turn_word_threshold: int = 250
    ):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if overlap < 0 or overlap >= window_size:
            raise ValueError(f"overlap must be in [0, window_size-1], got {overlap}")

        self.window_size = window_size
        self.overlap = overlap
        self.step_size = window_size - overlap
        self.long_turn_word_threshold = long_turn_word_threshold

    def chunk_session(
        self,
        session_id: str,
        turns: list[dict[str, str] | Any],
        timestamp: str = "",
        parsed_datetime: datetime | None = None,
        memory_id: str | None = None,
        session_metadata: dict[str, Any] | None = None
    ) -> list[TurnChunk]:
        """
        Partition a single session's turns into overlapping TurnChunk instances.
        Guarantees zero loss: every non-empty turn is included in at least one chunk.
        """
        mem_id = memory_id or (f"sess_{session_id}" if not session_id.startswith("sess_") else session_id)
        raw_sess_id = session_id[5:] if session_id.startswith("sess_") else session_id

        # Normalize turns into list of dicts: {"role": role, "content": content}
        normalized_turns = []
        for idx, t in enumerate(turns):
            if isinstance(t, dict):
                role = str(t.get("role", "user")).strip()
                content = str(t.get("content", ""))
            elif hasattr(t, "role") and hasattr(t, "content"):
                role = str(t.role).strip()
                content = str(t.content)
            else:
                role = "user"
                content = str(t)

            normalized_turns.append({
                "turn_index": idx,
                "role": role if role else "user",
                "content": content
            })

        total_turns = len(normalized_turns)
        if total_turns == 0:
            return []

        # If session length is <= window_size, return a single chunk covering all turns
        if total_turns <= self.window_size:
            chunk = self._create_chunk(
                session_id=raw_sess_id,
                memory_id=mem_id,
                chunk_index=0,
                turns=normalized_turns,
                start_idx=0,
                end_idx=total_turns - 1,
                total_session_turns=total_turns,
                timestamp=timestamp,
                parsed_datetime=parsed_datetime,
                session_metadata=session_metadata
            )
            return [chunk]

        # Sliding window with step_size = window_size - overlap
        chunks: list[TurnChunk] = []
        chunk_idx = 0
        start = 0

        while start < total_turns:
            end = min(start + self.window_size, total_turns)
            window_turns = normalized_turns[start:end]

            chunk = self._create_chunk(
                session_id=raw_sess_id,
                memory_id=mem_id,
                chunk_index=chunk_idx,
                turns=window_turns,
                start_idx=start,
                end_idx=end - 1,
                total_session_turns=total_turns,
                timestamp=timestamp,
                parsed_datetime=parsed_datetime,
                session_metadata=session_metadata
            )
            chunks.append(chunk)
            chunk_idx += 1

            # Advance by step_size
            if end >= total_turns:
                break
            start += self.step_size

        return chunks

    def chunk_all_sessions(
        self,
        sessions: list[dict[str, Any] | Any]
    ) -> list[TurnChunk]:
        """
        Chunk a list of session objects or dictionaries.
        """
        all_chunks: list[TurnChunk] = []
        for sess in sessions:
            if isinstance(sess, dict):
                sid = str(sess.get("session_id") or sess.get("id", "unknown"))
                turns = sess.get("turns", [])
                ts = str(sess.get("timestamp") or sess.get("date", ""))
                dt = sess.get("parsed_datetime")
                meta = {k: v for k, v in sess.items() if k not in ("turns", "text")}
            else:
                sid = str(getattr(sess, "session_id", "unknown"))
                turns = getattr(sess, "turns", [])
                ts = str(getattr(sess, "date", ""))
                dt = getattr(sess, "parsed_datetime", None)
                meta = {}

            chunks = self.chunk_session(
                session_id=sid,
                turns=turns,
                timestamp=ts,
                parsed_datetime=dt,
                session_metadata=meta
            )
            all_chunks.extend(chunks)

        return all_chunks

    def _create_chunk(
        self,
        session_id: str,
        memory_id: str,
        chunk_index: int,
        turns: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        total_session_turns: int,
        timestamp: str,
        parsed_datetime: datetime | None,
        session_metadata: dict[str, Any] | None
    ) -> TurnChunk:
        """Format chunk text and assemble TurnChunk."""
        lines = []
        has_long_turn = False

        for t in turns:
            speaker = "User" if t["role"].lower() == "user" else "Assistant"
            content = t["content"].strip()
            w_count = len(content.split())
            if w_count > self.long_turn_word_threshold:
                has_long_turn = True
            lines.append(f"{speaker}: {content}")

        formatted_text = "\n".join(lines)
        words = len(formatted_text.split())
        token_est = max(1, int(words / 0.75))

        chunk_id = f"{memory_id}_c{chunk_index:03d}"

        meta = dict(session_metadata or {})
        meta["chunk_index"] = chunk_index
        meta["turns_in_chunk"] = len(turns)

        return TurnChunk(
            chunk_id=chunk_id,
            session_id=session_id,
            memory_id=memory_id,
            turn_start_idx=start_idx,
            turn_end_idx=end_idx,
            total_session_turns=total_session_turns,
            timestamp=timestamp,
            parsed_datetime=parsed_datetime,
            turns=[{"role": t["role"], "content": t["content"]} for t in turns],
            text=formatted_text,
            word_count=words,
            token_estimate=token_est,
            is_long_turn=has_long_turn,
            metadata=meta
        )
