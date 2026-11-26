"""SQLite Persistent Conversation Store for IVARA.

Provides transactional, ACID-compliant storage for interactive chat sessions and messages.
Includes a dedicated persistent Episodic Memory layer preserving verbatim conversational
turns independently of semantic memory extraction.
Strictly decoupled from long-term semantic memory storage and graph structures.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Literal, Optional
import uuid

DEFAULT_CONVERSATIONS_DB = Path(__file__).resolve().parent.parent.parent / "data" / "conversations.db"


@dataclass
class Conversation:
    """Represents a conversation session."""
    id: str
    title: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }


@dataclass
class Message:
    """Represents a single message turn in a conversation."""
    id: str
    conversation_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str
    sequence_num: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "sequence_num": self.sequence_num,
            "metadata": self.metadata
        }


@dataclass
class EpisodicTurn:
    """Represents a persistent episodic conversation turn in IVARA.
    
    Guarantees raw conversational context preservation with double-linked sequence chaining,
    speaker provenance, and timestamp indexing independently of semantic memory extraction.
    """
    turn_id: str
    conversation_id: str
    speaker: str
    speaker_role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str  # ISO 8601 UTC
    sequence_num: int
    prev_turn_id: Optional[str] = None
    next_turn_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "speaker": self.speaker,
            "speaker_role": self.speaker_role,
            "content": self.content,
            "timestamp": self.timestamp,
            "sequence_num": self.sequence_num,
            "prev_turn_id": self.prev_turn_id,
            "next_turn_id": self.next_turn_id,
            "session_id": self.session_id,
            "metadata": self.metadata
        }


class ConversationStore:
    """Persistent SQLite-backed store for conversations, messages, and episodic turns."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path) if db_path is not None else str(DEFAULT_CONVERSATIONS_DB)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.RLock()
        self._fts_enabled: bool = False
        self._get_connection()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
                self._conn.execute("PRAGMA synchronous = NORMAL")
                self._conn.execute("PRAGMA temp_store = MEMORY")
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ConversationStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _init_db(self):
        """Idempotently initialize tables, indexes, FTS, and migrations."""
        conn = self._get_connection()
        with self._write_lock, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    sequence_num INTEGER NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv_seq ON messages(conversation_id, sequence_num ASC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv_created ON messages(conversation_id, created_at ASC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC)")

            # Dedicated episodic memory turns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_turns (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    session_id TEXT,
                    speaker TEXT NOT NULL,
                    speaker_role TEXT NOT NULL CHECK(speaker_role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sequence_num INTEGER NOT NULL,
                    prev_turn_id TEXT,
                    next_turn_id TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (prev_turn_id) REFERENCES episodic_turns(turn_id) ON DELETE SET NULL,
                    FOREIGN KEY (next_turn_id) REFERENCES episodic_turns(turn_id) ON DELETE SET NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_conv_seq ON episodic_turns(conversation_id, sequence_num ASC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_conv_time ON episodic_turns(conversation_id, timestamp ASC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_prev ON episodic_turns(prev_turn_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_next ON episodic_turns(next_turn_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_speaker ON episodic_turns(conversation_id, speaker)")

            # FTS5 full-text indexing for episodic search
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS episodic_turns_fts USING fts5(
                        turn_id UNINDEXED,
                        conversation_id UNINDEXED,
                        content
                    )
                """)
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False

            # Migration: Backfill any legacy messages missing from episodic_turns
            cur = conn.execute("""
                SELECT m.* FROM messages m
                WHERE m.id NOT IN (SELECT turn_id FROM episodic_turns)
                ORDER BY m.conversation_id, m.sequence_num ASC
            """)
            unmigrated_rows = cur.fetchall()

            if unmigrated_rows:
                by_conv: dict[str, list[Any]] = {}
                for r in unmigrated_rows:
                    by_conv.setdefault(r["conversation_id"], []).append(r)

                for conv_id, m_rows in by_conv.items():
                    # Find last existing turn in this conversation to chain from
                    cur_last = conn.execute(
                        "SELECT turn_id FROM episodic_turns WHERE conversation_id = ? ORDER BY sequence_num DESC LIMIT 1",
                        (conv_id,)
                    )
                    last_row = cur_last.fetchone()
                    prev_id = last_row["turn_id"] if last_row else None

                    for r in m_rows:
                        t_id = r["id"]
                        role = r["role"]
                        conn.execute("""
                            INSERT OR IGNORE INTO episodic_turns 
                            (turn_id, conversation_id, session_id, speaker, speaker_role, content, timestamp, sequence_num, prev_turn_id, next_turn_id, metadata)
                            VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, NULL, ?)
                        """, (t_id, conv_id, role, role, r["content"], r["created_at"], r["sequence_num"], prev_id, r["metadata"]))

                        if prev_id is not None:
                            conn.execute(
                                "UPDATE episodic_turns SET next_turn_id = ? WHERE turn_id = ?",
                                (t_id, prev_id)
                            )

                        if self._fts_enabled:
                            try:
                                conn.execute(
                                    "INSERT OR IGNORE INTO episodic_turns_fts (turn_id, conversation_id, content) VALUES (?, ?, ?)",
                                    (t_id, conv_id, r["content"])
                                )
                            except Exception:
                                pass
                        prev_id = t_id

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_conversation(
        self,
        title: str = "New Conversation",
        metadata: Optional[dict[str, Any]] = None,
        conversation_id: Optional[str] = None
    ) -> Conversation:
        """Create and persist a new conversation session."""
        conv_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        now = self._now_utc()
        meta = metadata or {}
        meta_json = json.dumps(meta)

        conn = self._get_connection()
        with self._write_lock, conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (conv_id, title.strip(), now, now, meta_json)
            )

        return Conversation(
            id=conv_id,
            title=title.strip(),
            created_at=now,
            updated_at=now,
            metadata=meta
        )

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Fetch a conversation by ID."""
        conn = self._get_connection()
        cur = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
        row = cur.fetchone()
        if not row:
            return None
        return Conversation(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"])
        )

    def list_conversations(self, limit: int = 50, offset: int = 0) -> list[Conversation]:
        """List conversations ordered by most recently updated."""
        conn = self._get_connection()
        cur = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = cur.fetchall()
        return [
            Conversation(
                id=r["id"],
                title=r["title"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                metadata=json.loads(r["metadata"])
            )
            for r in rows
        ]

    def update_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> Optional[Conversation]:
        """Update conversation title or metadata and bump updated_at."""
        existing = self.get_conversation(conversation_id)
        if not existing:
            return None

        new_title = title.strip() if title is not None else existing.title
        new_meta = metadata if metadata is not None else existing.metadata
        now = self._now_utc()

        conn = self._get_connection()
        with self._write_lock, conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ?, metadata = ? WHERE id = ?",
                (new_title, now, json.dumps(new_meta), conversation_id)
            )

        return Conversation(
            id=conversation_id,
            title=new_title,
            created_at=existing.created_at,
            updated_at=now,
            metadata=new_meta
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and cascade-delete all its messages and episodic turns."""
        conn = self._get_connection()
        with self._write_lock, conn:
            if getattr(self, "_fts_enabled", False):
                try:
                    conn.execute("DELETE FROM episodic_turns_fts WHERE conversation_id = ?", (conversation_id,))
                except Exception:
                    pass
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            return cur.rowcount > 0

    def clear_conversations(self) -> int:
        """Clear all conversations, messages, and episodic turns."""
        conn = self._get_connection()
        with self._write_lock, conn:
            if getattr(self, "_fts_enabled", False):
                try:
                    conn.execute("DELETE FROM episodic_turns_fts")
                except Exception:
                    pass
            cur = conn.execute("DELETE FROM conversations")
            return cur.rowcount

    # -------------------------------------------------------------------------
    # Episodic Layer Persistence & Retrieval
    # -------------------------------------------------------------------------

    def add_turn(
        self,
        conversation_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        speaker: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        turn_id: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> EpisodicTurn:
        """Append an episodic conversation turn with double-linked list chaining."""
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid message role '{role}'. Must be 'user', 'assistant', or 'system'.")

        t_id = turn_id or f"turn_{uuid.uuid4().hex[:12]}"
        speaker_val = speaker or role
        now = timestamp or self._now_utc()
        meta = metadata or {}
        meta_json = json.dumps(meta)

        conn = self._get_connection()
        with self._write_lock, conn:
            # Check conversation exists
            cur = conn.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
            if not cur.fetchone():
                raise KeyError(f"Conversation '{conversation_id}' does not exist.")

            # Check duplicate turn_id
            cur = conn.execute("SELECT turn_id FROM episodic_turns WHERE turn_id = ?", (t_id,))
            if cur.fetchone():
                raise ValueError(f"Turn with ID '{t_id}' already exists.")

            # Find previous turn in this conversation
            cur = conn.execute(
                "SELECT turn_id, sequence_num FROM episodic_turns WHERE conversation_id = ? ORDER BY sequence_num DESC LIMIT 1",
                (conversation_id,)
            )
            last_row = cur.fetchone()
            if last_row:
                prev_turn_id = last_row["turn_id"]
                next_seq = last_row["sequence_num"] + 1
            else:
                prev_turn_id = None
                # Check messages sequence_num fallback if any
                cur = conn.execute(
                    "SELECT COALESCE(MAX(sequence_num), 0) + 1 FROM messages WHERE conversation_id = ?",
                    (conversation_id,)
                )
                next_seq = cur.fetchone()[0]

            # Insert episodic turn
            conn.execute(
                """
                INSERT INTO episodic_turns (
                    turn_id, conversation_id, session_id, speaker, speaker_role,
                    content, timestamp, sequence_num, prev_turn_id, next_turn_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (t_id, conversation_id, session_id, speaker_val, role, content, now, next_seq, prev_turn_id, meta_json)
            )

            # Update previous turn's next_turn_id
            if prev_turn_id is not None:
                conn.execute(
                    "UPDATE episodic_turns SET next_turn_id = ? WHERE turn_id = ?",
                    (t_id, prev_turn_id)
                )

            # Synchronize to messages table for backwards compatibility
            conn.execute(
                """
                INSERT OR REPLACE INTO messages (id, conversation_id, role, content, created_at, sequence_num, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (t_id, conversation_id, role, content, now, next_seq, meta_json)
            )

            # Synchronize to FTS5
            if getattr(self, "_fts_enabled", False):
                try:
                    conn.execute(
                        "INSERT INTO episodic_turns_fts (turn_id, conversation_id, content) VALUES (?, ?, ?)",
                        (t_id, conversation_id, content)
                    )
                except Exception:
                    pass

            # Update conversation timestamp
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id)
            )

        return EpisodicTurn(
            turn_id=t_id,
            conversation_id=conversation_id,
            speaker=speaker_val,
            speaker_role=role,
            content=content,
            timestamp=now,
            sequence_num=next_seq,
            prev_turn_id=prev_turn_id,
            next_turn_id=None,
            session_id=session_id,
            metadata=meta
        )

    def add_turns_batch(
        self,
        conversation_id: str,
        turns: list[dict[str, Any]]
    ) -> list[EpisodicTurn]:
        """Atomically append multiple episodic turns in a single transaction."""
        if not turns:
            return []

        now = self._now_utc()
        conn = self._get_connection()
        results: list[EpisodicTurn] = []

        with conn:
            # Check conversation existence
            cur = conn.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
            if not cur.fetchone():
                raise KeyError(f"Conversation '{conversation_id}' does not exist.")

            # Find current last turn
            cur = conn.execute(
                "SELECT turn_id, sequence_num FROM episodic_turns WHERE conversation_id = ? ORDER BY sequence_num DESC LIMIT 1",
                (conversation_id,)
            )
            last_row = cur.fetchone()
            curr_prev_id = last_row["turn_id"] if last_row else None
            curr_seq = last_row["sequence_num"] if last_row else 0

            for turn_data in turns:
                role = turn_data["role"]
                if role not in ("user", "assistant", "system"):
                    raise ValueError(f"Invalid message role '{role}'.")
                curr_seq += 1
                t_id = turn_data.get("turn_id") or turn_data.get("id") or f"turn_{uuid.uuid4().hex[:12]}"

                # Check duplicate turn_id
                cur = conn.execute("SELECT turn_id FROM episodic_turns WHERE turn_id = ?", (t_id,))
                if cur.fetchone():
                    raise ValueError(f"Turn with ID '{t_id}' already exists.")

                content = turn_data["content"]
                speaker = turn_data.get("speaker") or role
                session_id = turn_data.get("session_id")
                meta = turn_data.get("metadata", {})
                t_stamp = turn_data.get("timestamp") or now

                # Insert episodic turn
                conn.execute(
                    """
                    INSERT INTO episodic_turns (
                        turn_id, conversation_id, session_id, speaker, speaker_role,
                        content, timestamp, sequence_num, prev_turn_id, next_turn_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                    """,
                    (t_id, conversation_id, session_id, speaker, role, content, t_stamp, curr_seq, curr_prev_id, json.dumps(meta))
                )

                # Update previous turn's next_turn_id
                if curr_prev_id is not None:
                    conn.execute(
                        "UPDATE episodic_turns SET next_turn_id = ? WHERE turn_id = ?",
                        (t_id, curr_prev_id)
                    )

                # Insert message
                conn.execute(
                    """
                    INSERT OR REPLACE INTO messages (id, conversation_id, role, content, created_at, sequence_num, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (t_id, conversation_id, role, content, t_stamp, curr_seq, json.dumps(meta))
                )

                # Sync FTS
                if getattr(self, "_fts_enabled", False):
                    try:
                        conn.execute(
                            "INSERT INTO episodic_turns_fts (turn_id, conversation_id, content) VALUES (?, ?, ?)",
                            (t_id, conversation_id, content)
                        )
                    except Exception:
                        pass

                results.append(EpisodicTurn(
                    turn_id=t_id,
                    conversation_id=conversation_id,
                    speaker=speaker,
                    speaker_role=role,
                    content=content,
                    timestamp=t_stamp,
                    sequence_num=curr_seq,
                    prev_turn_id=curr_prev_id,
                    next_turn_id=None,
                    session_id=session_id,
                    metadata=meta
                ))
                curr_prev_id = t_id

            # Update next_turn_id in returned objects
            for i in range(len(results) - 1):
                results[i].next_turn_id = results[i + 1].turn_id

            # Bump conversation updated_at
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id)
            )

        return results

    def get_turn(self, turn_id: str) -> Optional[EpisodicTurn]:
        """Fetch a single episodic turn by ID."""
        conn = self._get_connection()
        cur = conn.execute("SELECT * FROM episodic_turns WHERE turn_id = ?", (turn_id,))
        row = cur.fetchone()
        if not row:
            return None
        return EpisodicTurn(
            turn_id=row["turn_id"],
            conversation_id=row["conversation_id"],
            speaker=row["speaker"],
            speaker_role=row["speaker_role"],
            content=row["content"],
            timestamp=row["timestamp"],
            sequence_num=row["sequence_num"],
            prev_turn_id=row["prev_turn_id"],
            next_turn_id=row["next_turn_id"],
            session_id=row["session_id"],
            metadata=json.loads(row["metadata"])
        )

    def get_turn_neighbors(self, turn_id: str) -> tuple[Optional[EpisodicTurn], Optional[EpisodicTurn]]:
        """Fetch immediate (previous, next) turns for a given turn."""
        turn = self.get_turn(turn_id)
        if not turn:
            return (None, None)
        prev_turn = self.get_turn(turn.prev_turn_id) if turn.prev_turn_id else None
        next_turn = self.get_turn(turn.next_turn_id) if turn.next_turn_id else None
        return (prev_turn, next_turn)

    def get_context_window(self, turn_id: str, window_size: int = 1) -> list[EpisodicTurn]:
        """
        Fetch bounded context window around a turn (+/- window_size turns).
        Strictly isolated to the target turn's conversation_id.
        Returned in strict chronological sequence order.
        """
        if window_size < 0:
            raise ValueError("window_size must be non-negative.")
        turn = self.get_turn(turn_id)
        if not turn:
            return []

        conn = self._get_connection()
        min_seq = max(1, turn.sequence_num - window_size)
        max_seq = turn.sequence_num + window_size

        cur = conn.execute(
            """
            SELECT * FROM episodic_turns
            WHERE conversation_id = ? AND sequence_num BETWEEN ? AND ?
            ORDER BY sequence_num ASC
            """,
            (turn.conversation_id, min_seq, max_seq)
        )
        rows = cur.fetchall()
        return [
            EpisodicTurn(
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
                metadata=json.loads(r["metadata"])
            )
            for r in rows
        ]

    def get_turns(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        chronological: bool = True
    ) -> list[EpisodicTurn]:
        """Fetch episodic turns for a conversation ordered chronologically or reverse."""
        conn = self._get_connection()
        order = "ASC" if chronological else "DESC"

        if limit is not None:
            query = f"SELECT * FROM episodic_turns WHERE conversation_id = ? ORDER BY sequence_num {order} LIMIT ? OFFSET ?"
            params = (conversation_id, limit, offset)
        elif offset > 0:
            query = f"SELECT * FROM episodic_turns WHERE conversation_id = ? ORDER BY sequence_num {order} LIMIT -1 OFFSET ?"
            params = (conversation_id, offset)
        else:
            query = f"SELECT * FROM episodic_turns WHERE conversation_id = ? ORDER BY sequence_num {order}"
            params = (conversation_id,)

        cur = conn.execute(query, params)
        rows = cur.fetchall()
        return [
            EpisodicTurn(
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
                metadata=json.loads(r["metadata"])
            )
            for r in rows
        ]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text) // 4) + 6)

    def get_recent_turns(
        self,
        conversation_id: str,
        limit: Optional[int] = 12,
        max_tokens: Optional[int] = 2000,
        exclude_turn_id: Optional[str] = None
    ) -> list[EpisodicTurn]:
        """Fetch the most recent episodic turns in chronological order, bounded by count and token budget."""
        conn = self._get_connection()
        where = "conversation_id = ?"
        params: list[Any] = [conversation_id]
        if exclude_turn_id:
            where += " AND turn_id != ?"
            params.append(exclude_turn_id)

        if limit is not None:
            if limit <= 0:
                return []
            query = f"SELECT * FROM episodic_turns WHERE {where} ORDER BY sequence_num DESC LIMIT ?"
            params.append(limit)
        else:
            query = f"SELECT * FROM episodic_turns WHERE {where} ORDER BY sequence_num DESC"

        rows = conn.execute(query, tuple(params)).fetchall()
        selected = []
        used_tokens = 0
        for r in rows:
            toks = self._estimate_tokens(r["content"])
            if max_tokens is not None and selected and (used_tokens + toks > max_tokens):
                break
            selected.append(
                EpisodicTurn(
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
                    metadata=json.loads(r["metadata"])
                )
            )
            used_tokens += toks
        selected.reverse()
        return selected

    def search_turns(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        speaker: Optional[str] = None,
        limit: int = 20
    ) -> list[EpisodicTurn]:
        """Search turns using FTS5 when available, falling back to LIKE."""
        clean_q = query.strip()
        if not clean_q:
            return []

        conn = self._get_connection()

        if getattr(self, "_fts_enabled", False):
            try:
                fts_query = f'"{clean_q.replace("\"", "\"\"")}"'
                sql = """
                    SELECT e.* FROM episodic_turns e
                    JOIN episodic_turns_fts f ON e.turn_id = f.turn_id
                    WHERE episodic_turns_fts MATCH ?
                """
                params: list[Any] = [fts_query]
                if conversation_id:
                    sql += " AND e.conversation_id = ?"
                    params.append(conversation_id)
                if speaker:
                    sql += " AND e.speaker = ?"
                    params.append(speaker)
                sql += " ORDER BY e.sequence_num ASC LIMIT ?"
                params.append(limit)

                cur = conn.execute(sql, tuple(params))
                rows = cur.fetchall()
                if rows:
                    return [
                        EpisodicTurn(
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
                            metadata=json.loads(r["metadata"])
                        )
                        for r in rows
                    ]
            except Exception:
                pass

        # Standard LIKE fallback
        sql = "SELECT * FROM episodic_turns WHERE content LIKE ?"
        params = [f"%{clean_q}%"]
        if conversation_id:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        if speaker:
            sql += " AND speaker = ?"
            params.append(speaker)
        sql += " ORDER BY sequence_num ASC LIMIT ?"
        params.append(limit)

        cur = conn.execute(sql, tuple(params))
        rows = cur.fetchall()
        return [
            EpisodicTurn(
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
                metadata=json.loads(r["metadata"])
            )
            for r in rows
        ]

    def filter_turns(
        self,
        conversation_id: Optional[str] = None,
        speaker: Optional[str] = None,
        speaker_role: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50
    ) -> list[EpisodicTurn]:
        """Filter episodic turns by metadata, speaker, role, or time range."""
        conn = self._get_connection()
        clauses = []
        params: list[Any] = []

        if conversation_id:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if speaker:
            clauses.append("speaker = ?")
            params.append(speaker)
        if speaker_role:
            clauses.append("speaker_role = ?")
            params.append(speaker_role)
        if start_time:
            clauses.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            clauses.append("timestamp <= ?")
            params.append(end_time)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM episodic_turns {where_sql} ORDER BY sequence_num ASC LIMIT ?"
        params.append(limit)

        cur = conn.execute(sql, tuple(params))
        rows = cur.fetchall()
        return [
            EpisodicTurn(
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
                metadata=json.loads(r["metadata"])
            )
            for r in rows
        ]

    def count_turns(self, conversation_id: Optional[str] = None) -> int:
        """Return total count of episodic turns."""
        conn = self._get_connection()
        if conversation_id:
            cur = conn.execute("SELECT COUNT(*) FROM episodic_turns WHERE conversation_id = ?", (conversation_id,))
        else:
            cur = conn.execute("SELECT COUNT(*) FROM episodic_turns")
        return cur.fetchone()[0]

    # -------------------------------------------------------------------------
    # Backward Compatibility Methods for Message API
    # -------------------------------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        message_id: Optional[str] = None
    ) -> Message:
        """Append a message turn to a conversation atomically via the episodic layer."""
        turn = self.add_turn(
            conversation_id=conversation_id,
            role=role,
            content=content,
            speaker=role,
            metadata=metadata,
            turn_id=message_id
        )
        return Message(
            id=turn.turn_id,
            conversation_id=turn.conversation_id,
            role=turn.speaker_role,
            content=turn.content,
            created_at=turn.timestamp,
            sequence_num=turn.sequence_num,
            metadata=turn.metadata
        )

    def add_messages_batch(
        self,
        conversation_id: str,
        turns: list[dict[str, Any]]
    ) -> list[Message]:
        """Atomically append multiple message turns in a single transaction via the episodic layer."""
        episodic_turns = self.add_turns_batch(conversation_id, turns)
        return [
            Message(
                id=t.turn_id,
                conversation_id=t.conversation_id,
                role=t.speaker_role,
                content=t.content,
                created_at=t.timestamp,
                sequence_num=t.sequence_num,
                metadata=t.metadata
            )
            for t in episodic_turns
        ]

    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        chronological: bool = True
    ) -> list[Message]:
        """Fetch messages for a conversation in chronological order."""
        conn = self._get_connection()
        order = "ASC" if chronological else "DESC"

        if limit is not None:
            query = f"SELECT * FROM messages WHERE conversation_id = ? ORDER BY sequence_num {order} LIMIT ? OFFSET ?"
            params = (conversation_id, limit, offset)
        elif offset > 0:
            query = f"SELECT * FROM messages WHERE conversation_id = ? ORDER BY sequence_num {order} LIMIT -1 OFFSET ?"
            params = (conversation_id, offset)
        else:
            query = f"SELECT * FROM messages WHERE conversation_id = ? ORDER BY sequence_num {order}"
            params = (conversation_id,)

        cur = conn.execute(query, params)
        rows = cur.fetchall()
        return [
            Message(
                id=r["id"],
                conversation_id=r["conversation_id"],
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"],
                sequence_num=r["sequence_num"],
                metadata=json.loads(r["metadata"])
            )
            for r in rows
        ]

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = 12,
        max_tokens: Optional[int] = 2000,
        exclude_message_id: Optional[str] = None
    ) -> list[Message]:
        """Fetch the most recent messages in chronological order, bounded by count and token budget."""
        conn = self._get_connection()
        where = "conversation_id = ?"
        params: list[Any] = [conversation_id]
        if exclude_message_id:
            where += " AND id != ?"
            params.append(exclude_message_id)

        if limit is not None:
            if limit <= 0:
                return []
            query = f"SELECT * FROM messages WHERE {where} ORDER BY sequence_num DESC LIMIT ?"
            params.append(limit)
        else:
            query = f"SELECT * FROM messages WHERE {where} ORDER BY sequence_num DESC"

        rows = conn.execute(query, tuple(params)).fetchall()
        selected = []
        used_tokens = 0
        for r in rows:
            toks = self._estimate_tokens(r["content"])
            if max_tokens is not None and selected and (used_tokens + toks > max_tokens):
                break
            selected.append(
                Message(
                    id=r["id"],
                    conversation_id=r["conversation_id"],
                    role=r["role"],
                    content=r["content"],
                    created_at=r["created_at"],
                    sequence_num=r["sequence_num"],
                    metadata=json.loads(r["metadata"])
                )
            )
            used_tokens += toks
        selected.reverse()
        return selected

    def get_message(self, message_id: str) -> Optional[Message]:
        """Fetch a single message by ID."""
        conn = self._get_connection()
        cur = conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
        row = cur.fetchone()
        if not row:
            return None
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            sequence_num=row["sequence_num"],
            metadata=json.loads(row["metadata"])
        )

    def count_conversations(self) -> int:
        """Return total count of conversations."""
        conn = self._get_connection()
        cur = conn.execute("SELECT COUNT(*) FROM conversations")
        return cur.fetchone()[0]

    def count_messages(self, conversation_id: Optional[str] = None) -> int:
        """Return count of messages (overall or for a specific conversation)."""
        conn = self._get_connection()
        if conversation_id:
            cur = conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,))
        else:
            cur = conn.execute("SELECT COUNT(*) FROM messages")
        return cur.fetchone()[0]
