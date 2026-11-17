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
        self._node_cache: dict[str, GraphNode] = {}
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA cache_size = -32000")
            self._conn.execute("PRAGMA temp_store = MEMORY")
        self._init_db()

    def close(self):
        """Close SQLite database connection."""
        self._node_cache.clear()
        if self._conn:
            self._conn.close()
            self._conn = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
                self._conn.execute("PRAGMA synchronous = NORMAL")
                self._conn.execute("PRAGMA cache_size = -32000")
                self._conn.execute("PRAGMA temp_store = MEMORY")
            self._init_db()
        return self._conn

    def _init_db(self):
        """Create tables and performance indexes."""
        conn = self._get_connection()
        with conn:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source_type ON edges(source_id, edge_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target_type ON edges(target_id, edge_type)")

    def batch_upsert(
        self,
        nodes: list[GraphNode] | None = None,
        edges: list[GraphEdge] | None = None
    ) -> None:
        """
        Atomically upsert both nodes and edges in a single transaction.
        Nodes are upserted before edges to respect foreign-key constraints.
        If any statement fails, the entire transaction is rolled back.
        """
        conn = self._get_connection()
        with conn:
            if nodes:
                node_rows = [
                    (node.id, node.node_type, node.label, json.dumps(node.properties))
                    for node in nodes
                ]
                conn.executemany(
                    """
                    INSERT INTO nodes (id, node_type, label, properties)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        node_type = excluded.node_type,
                        label = excluded.label,
                        properties = excluded.properties
                    """,
                    node_rows
                )

            if edges:
                edge_rows = [
                    (
                        edge.id, edge.source_id, edge.target_id, edge.edge_type, edge.origin,
                        edge.confidence, edge.evidence, edge.memory_id, edge.chunk_id,
                        edge.timestamp, json.dumps(edge.properties)
                    )
                    for edge in edges
                ]
                conn.executemany(
                    """
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
                    """,
                    edge_rows
                )

        # Update cache only after transaction commits successfully
        if nodes:
            for node in nodes:
                self._node_cache[node.id] = node

    def upsert_nodes_batch(self, nodes: list[GraphNode]) -> None:
        """Idempotently insert or update multiple nodes in a single transaction."""
        self.batch_upsert(nodes=nodes, edges=None)

    def upsert_edges_batch(self, edges: list[GraphEdge]) -> None:
        """Idempotently insert or update multiple edges in a single transaction."""
        self.batch_upsert(nodes=None, edges=edges)

    def upsert_node(self, node: GraphNode):
        """Idempotently insert or update a node."""
        self.batch_upsert(nodes=[node], edges=None)

    def upsert_edge(self, edge: GraphEdge):
        """Idempotently insert or update an edge."""
        self.batch_upsert(nodes=None, edges=[edge])

    def get_node(self, node_id: str) -> GraphNode | None:
        """Fetch a single node by its ID."""
        if node_id in self._node_cache:
            return self._node_cache[node_id]

        conn = self._get_connection()
        cur = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = cur.fetchone()
        if not row:
            return None
        node = GraphNode(
            id=row["id"],
            node_type=row["node_type"],
            label=row["label"],
            properties=json.loads(row["properties"])
        )
        self._node_cache[node_id] = node
        return node

    def get_nodes_by_type(self, node_type: str) -> list[GraphNode]:
        """Fetch all nodes of a specific logical type."""
        conn = self._get_connection()
        cur = conn.execute("SELECT * FROM nodes WHERE node_type = ?", (node_type,))
        rows = cur.fetchall()
        result = []
        for r in rows:
            n_id = r["id"]
            node = self._node_cache.get(n_id)
            if node is None:
                node = GraphNode(
                    id=n_id,
                    node_type=r["node_type"],
                    label=r["label"],
                    properties=json.loads(r["properties"])
                )
                self._node_cache[n_id] = node
            result.append(node)
        return result

    def get_outgoing_edges(self, source_id: str, edge_type: str | None = None) -> list[GraphEdge]:
        """Fetch outgoing edges from a node."""
        conn = self._get_connection()
        if edge_type:
            cur = conn.execute("SELECT * FROM edges WHERE source_id = ? AND edge_type = ?", (source_id, edge_type))
        else:
            cur = conn.execute("SELECT * FROM edges WHERE source_id = ?", (source_id,))
        rows = cur.fetchall()
        return [self._row_to_edge(r) for r in rows]

    def get_incoming_edges(self, target_id: str, edge_type: str | None = None) -> list[GraphEdge]:
        """Fetch incoming edges to a node."""
        conn = self._get_connection()
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
        Preserves deterministic ordering (outgoing first, then incoming).
        """
        results = []
        conn = self._get_connection()

        if direction == "both":
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

            query += """
                UNION ALL
                SELECT e.*, n.id as n_id, n.node_type as n_node_type, n.label as n_label, n.properties as n_properties
                FROM edges e
                JOIN nodes n ON e.source_id = n.id
                WHERE e.target_id = ?
            """
            params.append(node_id)
            if edge_type:
                query += " AND e.edge_type = ?"
                params.append(edge_type)

            cur = conn.execute(query, params)
            for r in cur.fetchall():
                edge = self._row_to_edge(r)
                n_id = r["n_id"]
                node = self._node_cache.get(n_id)
                if node is None:
                    node = GraphNode(
                        id=n_id,
                        node_type=r["n_node_type"],
                        label=r["n_label"],
                        properties=json.loads(r["n_properties"])
                    )
                    self._node_cache[n_id] = node
                results.append((edge, node))

        elif direction == "outgoing":
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
                n_id = r["n_id"]
                node = self._node_cache.get(n_id)
                if node is None:
                    node = GraphNode(
                        id=n_id,
                        node_type=r["n_node_type"],
                        label=r["n_label"],
                        properties=json.loads(r["n_properties"])
                    )
                    self._node_cache[n_id] = node
                results.append((edge, node))

        elif direction == "incoming":
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
                n_id = r["n_id"]
                node = self._node_cache.get(n_id)
                if node is None:
                    node = GraphNode(
                        id=n_id,
                        node_type=r["n_node_type"],
                        label=r["n_label"],
                        properties=json.loads(r["n_properties"])
                    )
                    self._node_cache[n_id] = node
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
        conn = self._get_connection()
        cur = conn.execute("SELECT * FROM nodes")
        result = []
        for r in cur.fetchall():
            n_id = r["id"]
            node = self._node_cache.get(n_id)
            if node is None:
                node = GraphNode(
                    id=n_id,
                    node_type=r["node_type"],
                    label=r["label"],
                    properties=json.loads(r["properties"])
                )
                self._node_cache[n_id] = node
            result.append(node)
        return result

    def get_edges_by_type(self, edge_type: str) -> list[GraphEdge]:
        """Fetch all edges of a given edge_type."""
        conn = self._get_connection()
        cur = conn.execute("SELECT * FROM edges WHERE edge_type = ?", (edge_type,))
        return [self._row_to_edge(r) for r in cur.fetchall()]

    def count_nodes(self) -> int:
        conn = self._get_connection()
        cur = conn.execute("SELECT COUNT(*) FROM nodes")
        return cur.fetchone()[0]

    def count_edges(self) -> int:
        conn = self._get_connection()
        cur = conn.execute("SELECT COUNT(*) FROM edges")
        return cur.fetchone()[0]

    def clear(self):
        """Wipe all nodes and edges from this store."""
        self._node_cache.clear()
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM edges")
            conn.execute("DELETE FROM nodes")

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and let foreign key constraints cascade to its connected edges."""
        self._node_cache.pop(node_id, None)
        conn = self._get_connection()
        with conn:
            cur = conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            return cur.rowcount > 0

    def delete_memory_nodes_and_edges(self, memory_id: str) -> dict[str, int]:
        """Delete all nodes and edges associated with a specific memory ID.
        
        Removes:
          1. Any edge referencing memory_id or chunk_id matching memory_id.
          2. Any claim nodes where source_chunk_id matches memory_id.
          3. The memory container node itself (id == memory_id).
        """
        conn = self._get_connection()
        deleted_nodes_count = 0
        deleted_edges_count = 0

        with conn:
            # Find claim nodes whose source_chunk_id matches memory_id
            cur = conn.execute("SELECT id, properties FROM nodes WHERE node_type = 'claim'")
            claim_node_ids = []
            for row in cur.fetchall():
                try:
                    props = json.loads(row["properties"])
                    if props.get("source_chunk_id") == memory_id:
                        claim_node_ids.append(row["id"])
                except Exception:
                    pass

            all_node_ids = list(claim_node_ids)
            cur = conn.execute("SELECT id FROM nodes WHERE id = ?", (memory_id,))
            if cur.fetchone():
                all_node_ids.append(memory_id)

            # Delete edges referencing memory_id or chunk_id
            cur = conn.execute(
                "DELETE FROM edges WHERE memory_id = ? OR chunk_id = ?",
                (memory_id, memory_id)
            )
            deleted_edges_count += cur.rowcount

            # Delete nodes
            for nid in all_node_ids:
                self._node_cache.pop(nid, None)
                cur = conn.execute("DELETE FROM nodes WHERE id = ?", (nid,))
                deleted_nodes_count += cur.rowcount

        return {"deleted_nodes": deleted_nodes_count, "deleted_edges": deleted_edges_count}
