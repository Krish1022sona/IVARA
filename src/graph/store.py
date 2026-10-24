"""
SQLite Persistent Graph Store for IVARA.

Provides ACID-compliant, persistent, and rebuildable storage for the Memory Graph.
Completely isolated from existing production data stores.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Literal
from .schema import GraphNode, GraphEdge


class SQLiteGraphStore:
    """
    Persistent SQLite-backed graph store for memory nodes, entities, episodes, and typed edges.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_db()

    def close(self):
        """Close SQLite database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def _init_db(self):
        """Create tables and performance indexes."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    properties TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT,
                    memory_id TEXT,
                    chunk_id TEXT,
                    timestamp TEXT,
                    properties TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES nodes (id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES nodes (id) ON DELETE CASCADE
                )
            """)
            # Performance indexes for graph traversals
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_memory ON edges(memory_id)")
            conn.commit()

    def upsert_node(self, node: GraphNode):
        """Idempotently insert or update a node."""
        props_json = json.dumps(node.properties)
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO nodes (id, node_type, label, properties)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    node_type = excluded.node_type,
                    label = excluded.label,
                    properties = excluded.properties
            """, (node.id, node.node_type, node.label, props_json))
            conn.commit()

    def upsert_edge(self, edge: GraphEdge):
        """Idempotently insert or update an edge."""
        props_json = json.dumps(edge.properties)
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO edges (
                    id, source_id, target_id, edge_type, origin,
                    confidence, evidence, memory_id, chunk_id, timestamp, properties
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_id = excluded.source_id,
                    target_id = excluded.target_id,
                    edge_type = excluded.edge_type,
                    origin = excluded.origin,
                    confidence = excluded.confidence,
                    evidence = excluded.evidence,
                    memory_id = excluded.memory_id,
                    chunk_id = excluded.chunk_id,
                    timestamp = excluded.timestamp,
                    properties = excluded.properties
            """, (
                edge.id, edge.source_id, edge.target_id, edge.edge_type, edge.origin,
                edge.confidence, edge.evidence, edge.memory_id, edge.chunk_id,
                edge.timestamp, props_json
            ))
            conn.commit()

    def get_node(self, node_id: str) -> GraphNode | None:
        """Fetch a single node by its ID."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
            row = cur.fetchone()
            if not row:
                return None
            return GraphNode(
                id=row["id"],
                node_type=row["node_type"],
                label=row["label"],
                properties=json.loads(row["properties"])
            )

    def get_nodes_by_type(self, node_type: str) -> list[GraphNode]:
        """Fetch all nodes of a specific logical type."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM nodes WHERE node_type = ?", (node_type,))
            rows = cur.fetchall()
            return [
                GraphNode(
                    id=r["id"],
                    node_type=r["node_type"],
                    label=r["label"],
                    properties=json.loads(r["properties"])
                )
                for r in rows
            ]

    def get_outgoing_edges(self, source_id: str, edge_type: str | None = None) -> list[GraphEdge]:
        """Fetch outgoing edges from a node."""
        with self._get_connection() as conn:
            if edge_type:
                cur = conn.execute("SELECT * FROM edges WHERE source_id = ? AND edge_type = ?", (source_id, edge_type))
            else:
                cur = conn.execute("SELECT * FROM edges WHERE source_id = ?", (source_id,))
            rows = cur.fetchall()
            return [self._row_to_edge(r) for r in rows]

    def get_incoming_edges(self, target_id: str, edge_type: str | None = None) -> list[GraphEdge]:
        """Fetch incoming edges to a node."""
        with self._get_connection() as conn:
            if edge_type:
                cur = conn.execute("SELECT * FROM edges WHERE target_id = ? AND edge_type = ?", (target_id, edge_type))
            else:
                cur = conn.execute("SELECT * FROM edges WHERE target_id = ?", (target_id,))
            rows = cur.fetchall()
            return [self._row_to_edge(r) for r in rows]

    def get_neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "both"
    ) -> list[tuple[GraphEdge, GraphNode]]:
        """
        Return connected (edge, neighbor_node) pairs in the specified direction.
        """
        results = []
        with self._get_connection() as conn:
            # 1. Outgoing neighbors
            if direction in ("outgoing", "both"):
                query = """
                    SELECT e.*, n.id as n_id, n.node_type as n_node_type, n.label as n_label, n.properties as n_properties
                    FROM edges e
                    JOIN nodes n ON e.target_id = n.id
                    WHERE e.source_id = ?
                """
                params = [node_id]
                if edge_type:
                    query += " AND e.edge_type = ?"
                    params.append(edge_type)
                cur = conn.execute(query, params)
                for r in cur.fetchall():
                    edge = self._row_to_edge(r)
                    node = GraphNode(
                        id=r["n_id"],
                        node_type=r["n_node_type"],
                        label=r["n_label"],
                        properties=json.loads(r["n_properties"])
                    )
                    results.append((edge, node))

            # 2. Incoming neighbors
            if direction in ("incoming", "both"):
                query = """
                    SELECT e.*, n.id as n_id, n.node_type as n_node_type, n.label as n_label, n.properties as n_properties
                    FROM edges e
                    JOIN nodes n ON e.source_id = n.id
                    WHERE e.target_id = ?
                """
                params = [node_id]
                if edge_type:
                    query += " AND e.edge_type = ?"
                    params.append(edge_type)
                cur = conn.execute(query, params)
                for r in cur.fetchall():
                    edge = self._row_to_edge(r)
                    node = GraphNode(
                        id=r["n_id"],
                        node_type=r["n_node_type"],
                        label=r["n_label"],
                        properties=json.loads(r["n_properties"])
                    )
                    results.append((edge, node))

        return results

    def _row_to_edge(self, r: sqlite3.Row) -> GraphEdge:
        return GraphEdge(
            id=r["id"],
            source_id=r["source_id"],
            target_id=r["target_id"],
            edge_type=r["edge_type"],
            origin=r["origin"],
            confidence=r["confidence"],
            evidence=r["evidence"],
            memory_id=r["memory_id"],
            chunk_id=r["chunk_id"],
            timestamp=r["timestamp"],
            properties=json.loads(r["properties"])
        )

    def get_all_nodes(self) -> list[GraphNode]:
        """Fetch all nodes in the graph store."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM nodes")
            return [
                GraphNode(
                    id=r["id"],
                    node_type=r["node_type"],
                    label=r["label"],
                    properties=json.loads(r["properties"])
                )
                for r in cur.fetchall()
            ]

    def get_edges_by_type(self, edge_type: str) -> list[GraphEdge]:
        """Fetch all edges of a given edge_type."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM edges WHERE edge_type = ?", (edge_type,))
            return [self._row_to_edge(r) for r in cur.fetchall()]

    def count_nodes(self) -> int:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM nodes")
            return cur.fetchone()[0]

    def count_edges(self) -> int:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM edges")
            return cur.fetchone()[0]

    def clear(self):
        """Wipe all nodes and edges from this store."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")
            conn.commit()
