"""
Bounded Temporal Graph Traversal and Retrieval for IVARA.

Performs targeted 1- to 2-hop traversals over EventNodes and TemporalRelationshipEdges
to resolve event ordering, cross-session relationships, and historical state supersession.
"""

from collections import defaultdict
import re
from typing import Any

from .event_graph import (
    EventNode,
    TemporalRelationshipEdge,
    TemporalRelationshipType,
    TemporalPrecision
)
from .calendar_arithmetic import calculate_date_delta, parse_date_string, DeterministicCalendarEngine
from ..ontology.vocabulary import normalize_name


class TemporalEventGraphRetriever:
    """
    Retrieves and reasons over temporal event subgraphs to answer ordering,
    supersession, and cross-session entity queries.
    """

    def __init__(self, events: list[EventNode], edges: list[TemporalRelationshipEdge]):
        self.events = events
        self.edges = edges
        self.events_by_id: dict[str, EventNode] = {e.event_id: e for e in events}
        
        # Build adjacency maps
        self.out_edges: dict[str, list[TemporalRelationshipEdge]] = defaultdict(list)
        self.in_edges: dict[str, list[TemporalRelationshipEdge]] = defaultdict(list)
        for edge in edges:
            self.out_edges[edge.source_id].append(edge)
            self.in_edges[edge.target_id].append(edge)

        self.calendar_engine = DeterministicCalendarEngine()

    def query_graph(self, query: str, top_k_events: int = 6) -> dict[str, Any]:
        """
        Query the temporal event graph.
        Returns structured graph evidence, supporting chunk IDs, and confidence.
        """
        q_lower = query.lower()

        # 1. Classify temporal query intent
        is_ordering_query = bool(re.search(
            r'\b(?:order of|order from|earliest|latest|first|second|third|which happened first|before|after|between|consecutive|in a row|days passed)\b',
            q_lower
        ))
        is_supersession_query = bool(re.search(
            r'\b(?:before moving to|before working at|previous(?:ly)?|where did i live before|what did i do before|who did i work for before|prior to)\b',
            q_lower
        ))
        is_invalidation_query = bool(re.search(
            r'\b(?:cancelled|canceled|terminated|stopped|active|still subscribe|still have)\b',
            q_lower
        ))

        # 2. Find candidate events relevant to query
        matched_events = self._match_events(query)
        if not matched_events:
            return {
                "evidence_text": "",
                "supporting_chunk_ids": [],
                "events": [],
                "edges": [],
                "confidence": 0.0
            }

        selected_events: list[EventNode] = matched_events[:top_k_events]
        selected_event_ids = {e.event_id for e in selected_events}
        traversed_edges: list[TemporalRelationshipEdge] = []
        structured_findings: list[str] = []

        # 3. Traversal based on query intent
        if is_supersession_query:
            # Look for SUPERSEDES edges involving matched events
            for e in selected_events:
                # In-edges: what supersedes e, or Out-edges: what e supersedes
                for edge in self.out_edges[e.event_id]:
                    if edge.relationship == TemporalRelationshipType.SUPERSEDES:
                        target_ev = self.events_by_id.get(edge.target_id)
                        if target_ev:
                            traversed_edges.append(edge)
                            structured_findings.append(
                                f"Historical Transition: '{e.description}' ({e.valid_from}) superseded previous state '{target_ev.description}' ({target_ev.valid_from})."
                            )
                            if target_ev.event_id not in selected_event_ids:
                                selected_events.append(target_ev)
                                selected_event_ids.add(target_ev.event_id)

        elif is_invalidation_query:
            # Look for INVALIDATES edges
            for e in selected_events:
                for edge in self.out_edges[e.event_id]:
                    if edge.relationship == TemporalRelationshipType.INVALIDATES:
                        target_ev = self.events_by_id.get(edge.target_id)
                        if target_ev:
                            traversed_edges.append(edge)
                            structured_findings.append(
                                f"Invalidation/Cancellation: '{e.description}' ({e.valid_from}) invalidated '{target_ev.description}'."
                            )

        if is_ordering_query or not structured_findings:
            # Look for OCCURRED_BEFORE / OCCURRED_AFTER edges between matched events
            for i, e1 in enumerate(selected_events):
                for e2 in selected_events[i + 1:]:
                    # Check direct edge e1 -> e2
                    for edge in self.out_edges[e1.event_id]:
                        if edge.target_id == e2.event_id and edge.relationship == TemporalRelationshipType.OCCURRED_BEFORE:
                            traversed_edges.append(edge)
                            structured_findings.append(
                                f"Verified Ordering: '{e1.description}' OCCURRED_BEFORE '{e2.description}' (Evidence: {edge.evidence})."
                            )
                    # Check direct edge e2 -> e1
                    for edge in self.out_edges[e2.event_id]:
                        if edge.target_id == e1.event_id and edge.relationship == TemporalRelationshipType.OCCURRED_BEFORE:
                            traversed_edges.append(edge)
                            structured_findings.append(
                                f"Verified Ordering: '{e2.description}' OCCURRED_BEFORE '{e1.description}' (Evidence: {edge.evidence})."
                            )

        # 4. If query asks for complete chronological ordering of several events
        if is_ordering_query and len(selected_events) >= 2:
            # Sort events by valid_from where known
            dated_events = [e for e in selected_events if e.valid_from]
            if len(dated_events) >= 2:
                dated_events.sort(key=lambda x: (x.valid_from or "", x.description))
                timeline_desc = " -> ".join([f"'{e.description}' ({e.valid_from})" for e in dated_events])
                structured_findings.append(f"Chronological Timeline: {timeline_desc}")

        # 5. Assemble supporting chunks and evidence block
        supporting_chunks = set()
        for e in selected_events:
            if e.turn_id:
                supporting_chunks.add(e.turn_id)
        for edge in traversed_edges:
            supporting_chunks.update(edge.provenance_chunk_ids)

        evidence_lines = ["[TEMPORAL EVENT GRAPH EVIDENCE]"]
        for e in selected_events:
            date_display = e.valid_from if e.valid_from else "Unknown date"
            prec_display = e.precision.value if hasattr(e.precision, "value") else str(e.precision)
            evidence_lines.append(
                f"- Event: \"{e.description}\" | Date: {date_display} (Precision: {prec_display}) | Source Session: {e.session_id}"
            )
            if e.provenance:
                evidence_lines.append(f"  Provenance: \"{e.provenance.strip()}\"")

        if structured_findings:
            evidence_lines.append("Relationships & Ordering:")
            for finding in structured_findings:
                evidence_lines.append(f"  * {finding}")

        evidence_text = "\n".join(evidence_lines)

        overall_conf = 0.85 if structured_findings else (0.6 if selected_events else 0.0)

        return {
            "evidence_text": evidence_text,
            "supporting_chunk_ids": list(supporting_chunks),
            "events": selected_events,
            "edges": traversed_edges,
            "confidence": overall_conf
        }

    def _match_events(self, query: str) -> list[EventNode]:
        """
        Rank events by lexical and entity overlap with query.
        """
        q_tokens = set(re.findall(r'[a-zA-Z0-9#]+', query.lower()))
        # Remove common query stop words
        stop_words = {
            "what", "when", "where", "which", "did", "the", "and", "from", "first",
            "second", "third", "order", "between", "happened", "how", "many", "days",
            "passed", "were", "was", "have", "you", "user", "i", "my", "me"
        }
        query_keywords = q_tokens - stop_words

        scored_events: list[tuple[float, EventNode]] = []
        for e in self.events:
            score = 0.0
            desc_tokens = set(re.findall(r'[a-zA-Z0-9#]+', e.description.lower()))
            overlap = query_keywords.intersection(desc_tokens)
            score += len(overlap) * 2.0

            # Entity mention overlap
            for ent_id in e.entity_ids:
                ent_clean = ent_id.replace("ent_", "").lower()
                if ent_clean in query.lower():
                    score += 3.0

            # Target overlap
            target = e.properties.get("target", "").lower()
            if target and target in query.lower():
                score += 4.0

            # Action overlap
            action = e.properties.get("action_type", "").lower()
            if action and action in query.lower():
                score += 1.5

            if score > 0.0:
                # Tie-breaker: prefer events with exact dates
                if e.precision == TemporalPrecision.EXACT_DAY:
                    score += 0.5
                scored_events.append((score, e))

        scored_events.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored_events]

