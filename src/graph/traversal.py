"""
Relevance-guided Graph Traversal for IVARA.

Implements the candidate-exploration OR rule:
explore(edge, neighbor, query) = relevant(edge, query) OR relevant(neighbor, query)

Features:
- Cycle prevention via visited sets.
- Strict traversal depth and node budget limits.
- Separate evaluations for relationship relevance and destination relevance.
- Rich diagnostics logging exploration signals ("edge", "destination", or "both").
"""

from typing import Any
from .schema import GraphNode, GraphEdge
from .store import SQLiteGraphStore


# Canonical question-intent mapping for edge predicates
EDGE_INTENT_PATTERNS = {
    "technology": {
        "keywords": ["language", "framework", "stack", "database", "tools", "technology", "code", "ingestion", "pipeline", "rust", "python", "pytorch", "c++", "neo4j"],
        "predicates": ["uses", "uses_framework", "uses_language", "grounds_uses"]
    },
    "people_collaboration": {
        "keywords": ["who", "worked with", "collaborated", "mentored", "mentor", "colleague", "reports to", "reporting", "manager", "lead", "engineer"],
        "predicates": ["worked_with", "employed_by", "employs", "has_role", "grounds_worked_with", "grounds_employed_by", "grounds_has_role"]
    },
    "project_involvement": {
        "keywords": ["project", "chronos", "aegis", "nexus", "hyperion", "built", "spearheaded", "initiated", "launched", "contributed", "assigned"],
        "predicates": ["worked_on", "worked_on_by", "grounds_worked_on", "grounds_worked_on_by"]
    },
    "family_social": {
        "keywords": ["father", "parent", "son", "child", "friend", "undergraduate", "college", "family"],
        "predicates": ["parent_of", "child_of", "friend_of", "grounds_parent_of", "grounds_friend_of"]
    },
    "temporal_evolution": {
        "keywords": ["when", "date", "timeline", "started", "shipped", "postponed", "cancelled", "sabbatical", "rotation", "changed"],
        "predicates": ["precedes", "supersedes", "grounds_status"]
    }
}


def evaluate_edge_relevance(edge: GraphEdge, query: str) -> tuple[bool, float, str]:
    """
    Evaluate if an edge relationship is relevant to the query intent.
    """
    q_lower = query.lower()
    edge_type_clean = edge.edge_type.lower()

    # 1. Structural navigation edges: adjacent chunks or parent containers are always navigable at low baseline score
    if edge_type_clean in ("contains_chunk", "adjacent_to", "contains_memory"):
        return True, 0.40, f"Structural navigation edge ({edge.edge_type})"

    # 2. Check intent-based predicate matching
    for intent, config in EDGE_INTENT_PATTERNS.items():
        intent_match = any(kw in q_lower for kw in config["keywords"])
        if intent_match and any(p in edge_type_clean for p in config["predicates"]):
            return True, 0.85, f"Edge predicate '{edge.edge_type}' matches query intent for '{intent}'"

    # 3. Check if edge evidence text contains key query tokens
    if edge.evidence:
        ev_lower = edge.evidence.lower()
        query_tokens = [t for t in q_lower.split() if len(t) > 3]
        matched_tokens = [t for t in query_tokens if t in ev_lower]
        if matched_tokens:
            score = 0.5 + min(0.4, 0.1 * len(matched_tokens))
            return True, score, f"Edge evidence matches query tokens: {matched_tokens}"

    return False, 0.0, "Edge not relevant to query intent"


def evaluate_destination_relevance(node: GraphNode, query: str) -> tuple[bool, float, str]:
    """
    Evaluate if the destination node itself contains relevant information for the query.
    """
    q_lower = query.lower()
    query_tokens = set(t.strip(".,?!\"'") for t in q_lower.split() if len(t) > 3)

    # 1. Entity Node
    if node.node_type == "entity":
        canon_name = node.properties.get("canonical_name", "").lower()
        if canon_name and canon_name in q_lower:
            return True, 0.90, f"Destination entity '{node.label}' explicitly named in query"
        aliases = [a.lower() for a in node.properties.get("aliases", [])]
        for a in aliases:
            if a in q_lower:
                return True, 0.85, f"Destination entity alias '{a}' named in query"
        types = [t.lower() for t in node.properties.get("types", [])]
        for t in types:
            if t in q_lower:
                return True, 0.60, f"Destination entity type '{t}' matches query"

    # 2. Chunk or Memory Node
    elif node.node_type in ("chunk", "memory"):
        text = node.properties.get("text", "").lower()
        if not text:
            return False, 0.0, "Empty node text"

        # Check token containment
        matches = [tok for tok in query_tokens if tok in text]
        if len(matches) >= 2 or (len(query_tokens) == 1 and len(matches) == 1):
            score = min(0.95, 0.35 + 0.15 * len(matches))
            return True, score, f"Destination text contains query keywords: {matches}"

    # 3. Episode Node
    elif node.node_type == "episode":
        title = node.properties.get("title", "").lower()
        if any(tok in title for tok in query_tokens):
            return True, 0.65, f"Destination episode title matches query: {title}"

    return False, 0.0, "Destination node not relevant to query"


def traverse_relevance_guided(
    store: SQLiteGraphStore,
    query: str,
    seed_node_ids: list[str],
    max_depth: int = 2,
    max_nodes: int = 15
) -> dict[str, Any]:
    """
    Traverse the Memory Graph starting from seed nodes using the relevance OR rule.

    Returns:
        {
            "discovered_nodes": list[GraphNode],
            "discovered_edges": list[GraphEdge],
            "diagnostics": list[dict]
        }
    """
    visited_nodes: set[str] = set()
    discovered_nodes: dict[str, GraphNode] = {}
    discovered_edges: dict[str, GraphEdge] = {}
    diagnostics: list[dict] = []

    # Frontier holds tuples: (priority_score, current_depth, node_id, source_path)
    # Use simple list sorted by priority for predictable priority queue behavior
    frontier: list[tuple[float, int, str, list[str]]] = []

    for s_id in seed_node_ids:
        node = store.get_node(s_id)
        if node:
            frontier.append((1.0, 0, s_id, [s_id]))
            discovered_nodes[s_id] = node

    while frontier and len(discovered_nodes) < max_nodes:
        # Sort to pop highest priority candidate
        frontier.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        priority, depth, current_id, path = frontier.pop(0)

        if current_id in visited_nodes:
            continue
        visited_nodes.add(current_id)

        if depth >= max_depth:
            continue

        # Explore outgoing and incoming neighbors
        neighbors = store.get_neighbors(current_id, direction="both")
        for edge, neighbor in neighbors:
            if neighbor.id in visited_nodes:
                continue

            # Evaluate Relevance OR Rule:
            # explore(edge, neighbor, query) = relevant(edge, query) OR relevant(neighbor, query)
            edge_rel, edge_score, edge_reason = evaluate_edge_relevance(edge, query)
            dest_rel, dest_score, dest_reason = evaluate_destination_relevance(neighbor, query)

            should_explore = edge_rel or dest_rel

            if should_explore:
                # Record signal used
                if edge_rel and dest_rel:
                    signal = "both"
                elif edge_rel:
                    signal = "edge_relevance"
                else:
                    signal = "destination_relevance"

                combined_score = max(edge_score, dest_score)

                discovered_nodes[neighbor.id] = neighbor
                discovered_edges[edge.id] = edge

                diagnostics.append({
                    "from_node": current_id,
                    "to_node": neighbor.id,
                    "edge_type": edge.edge_type,
                    "depth": depth + 1,
                    "signal": signal,
                    "edge_reason": edge_reason,
                    "dest_reason": dest_reason,
                    "score": round(combined_score, 3)
                })

                frontier.append((combined_score, depth + 1, neighbor.id, path + [neighbor.id]))

                if len(discovered_nodes) >= max_nodes:
                    break

    return {
        "discovered_nodes": list(discovered_nodes.values()),
        "discovered_edges": list(discovered_edges.values()),
        "diagnostics": diagnostics,
        "nodes_expanded": len(discovered_nodes),
        "edges_traversed": len(discovered_edges)
    }
