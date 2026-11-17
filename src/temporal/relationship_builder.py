"""
Temporal Relationship Construction Engine for IVARA.

Builds OCCURRED_BEFORE, OCCURRED_AFTER, SUPERSEDES, and INVALIDATES edges
between EventNodes with strict provenance and confidence scoring.

CRITICAL INVARIANT:
Never infers temporal relationships solely from ingestion sequence or conversation recording time.
"""

from datetime import date
import re
from typing import Literal

from .event_graph import (
    EventNode,
    TemporalRelationshipEdge,
    TemporalRelationshipType,
    TemporalPrecision,
    generate_temporal_edge_id
)
from .calendar_arithmetic import parse_date_string, calculate_date_delta


class TemporalRelationshipBuilder:
    """
    Constructs directed temporal edges between EventNodes.
    Enforces non-ingestion ordering safety.
    """

    def build_relationships(self, events: list[EventNode]) -> list[TemporalRelationshipEdge]:
        """
        Build all verifiable temporal relationships across a collection of EventNodes.
        """
        edges: list[TemporalRelationshipEdge] = []
        n = len(events)
        if n < 2:
            return edges

        # Pairwise comparison
        for i in range(n):
            for j in range(i + 1, n):
                e1 = events[i]
                e2 = events[j]

                # 1. Grounded interval comparison (OCCURRED_BEFORE / OCCURRED_AFTER)
                order_edges = self._compare_intervals(e1, e2)
                edges.extend(order_edges)

                # 2. State supersession check (SUPERSEDES)
                sup_edge = self._check_supersession(e1, e2)
                if sup_edge:
                    edges.append(sup_edge)

                # 3. State invalidation check (INVALIDATES)
                inv_edge = self._check_invalidation(e1, e2)
                if inv_edge:
                    edges.append(inv_edge)

        return edges

    def _compare_intervals(self, e1: EventNode, e2: EventNode) -> list[TemporalRelationshipEdge]:
        """
        Compare temporal bounds [valid_from, valid_to] of two events.
        NEVER uses recorded_at to infer ordering!
        """
        edges: list[TemporalRelationshipEdge] = []

        # If either event lacks valid date bounds, do NOT infer ordering from ingestion time!
        if not e1.valid_to or not e2.valid_from or not e1.valid_from or not e2.valid_to:
            return edges

        # Parse boundary dates
        d1_end = parse_date_string(e1.valid_to)
        d2_start = parse_date_string(e2.valid_from)
        d2_end = parse_date_string(e2.valid_to)
        d1_start = parse_date_string(e1.valid_from)

        if not d1_end or not d2_start or not d2_end or not d1_start:
            return edges

        # Confidence based on precision
        prec_penalty = 1.0
        if e1.precision == TemporalPrecision.YEAR or e2.precision == TemporalPrecision.YEAR:
            prec_penalty = 0.85
        elif e1.precision == TemporalPrecision.MONTH or e2.precision == TemporalPrecision.MONTH:
            prec_penalty = 0.92
        elif e1.precision == TemporalPrecision.RELATIVE or e2.precision == TemporalPrecision.RELATIVE:
            prec_penalty = 0.90

        # Case 1: e1 strictly before e2
        if d1_end < d2_start:
            delta = calculate_date_delta(d1_end, d2_start)
            evidence = f"{e1.description} ({e1.valid_to}) ended {delta['days']} days before {e2.description} ({e2.valid_from})"

            # e1 OCCURRED_BEFORE e2
            edge_fwd = TemporalRelationshipEdge(
                id=generate_temporal_edge_id(e1.event_id, e2.event_id, TemporalRelationshipType.OCCURRED_BEFORE),
                source_id=e1.event_id,
                target_id=e2.event_id,
                relationship=TemporalRelationshipType.OCCURRED_BEFORE,
                confidence=round(min(e1.confidence, e2.confidence) * prec_penalty, 2),
                evidence=evidence,
                provenance_chunk_ids=[c for c in [e1.turn_id, e2.turn_id] if c],
                derived_by="grounded_interval_comparison",
                properties={"days_between": delta["days"]}
            )
            edges.append(edge_fwd)

            # e2 OCCURRED_AFTER e1
            edge_bwd = TemporalRelationshipEdge(
                id=generate_temporal_edge_id(e2.event_id, e1.event_id, TemporalRelationshipType.OCCURRED_AFTER),
                source_id=e2.event_id,
                target_id=e1.event_id,
                relationship=TemporalRelationshipType.OCCURRED_AFTER,
                confidence=round(min(e1.confidence, e2.confidence) * prec_penalty, 2),
                evidence=evidence,
                provenance_chunk_ids=[c for c in [e1.turn_id, e2.turn_id] if c],
                derived_by="grounded_interval_comparison",
                properties={"days_between": delta["days"]}
            )
            edges.append(edge_bwd)

        # Case 2: e2 strictly before e1
        elif d2_end < d1_start:
            delta = calculate_date_delta(d2_end, d1_start)
            evidence = f"{e2.description} ({e2.valid_to}) ended {delta['days']} days before {e1.description} ({e1.valid_from})"

            # e2 OCCURRED_BEFORE e1
            edge_fwd = TemporalRelationshipEdge(
                id=generate_temporal_edge_id(e2.event_id, e1.event_id, TemporalRelationshipType.OCCURRED_BEFORE),
                source_id=e2.event_id,
                target_id=e1.event_id,
                relationship=TemporalRelationshipType.OCCURRED_BEFORE,
                confidence=round(min(e1.confidence, e2.confidence) * prec_penalty, 2),
                evidence=evidence,
                provenance_chunk_ids=[c for c in [e1.turn_id, e2.turn_id] if c],
                derived_by="grounded_interval_comparison",
                properties={"days_between": delta["days"]}
            )
            edges.append(edge_fwd)

            # e1 OCCURRED_AFTER e2
            edge_bwd = TemporalRelationshipEdge(
                id=generate_temporal_edge_id(e1.event_id, e2.event_id, TemporalRelationshipType.OCCURRED_AFTER),
                source_id=e1.event_id,
                target_id=e2.event_id,
                relationship=TemporalRelationshipType.OCCURRED_AFTER,
                confidence=round(min(e1.confidence, e2.confidence) * prec_penalty, 2),
                evidence=evidence,
                provenance_chunk_ids=[c for c in [e1.turn_id, e2.turn_id] if c],
                derived_by="grounded_interval_comparison",
                properties={"days_between": delta["days"]}
            )
            edges.append(edge_bwd)

        return edges

    def _check_supersession(self, e1: EventNode, e2: EventNode) -> TemporalRelationshipEdge | None:
        """
        Check if one event supersedes another on the same slot (e.g. residence, employment).
        Requires later event date or explicit transition marker.
        """
        act1 = e1.properties.get("action_type")
        act2 = e2.properties.get("action_type")

        # Mutually exclusive state slots
        state_slots = {"lived in", "worked at"}
        if act1 in state_slots and act2 in state_slots and act1 == act2:
            # Different targets (e.g. lived in Boston vs lived in Seattle)
            tgt1 = e1.properties.get("target", "").lower()
            tgt2 = e2.properties.get("target", "").lower()

            if tgt1 and tgt2 and tgt1 != tgt2:
                # Compare valid_from dates
                d1 = parse_date_string(e1.valid_from)
                d2 = parse_date_string(e2.valid_from)

                if d1 and d2:
                    if d2 > d1:
                        # e2 SUPERSEDES e1
                        return TemporalRelationshipEdge(
                            id=generate_temporal_edge_id(e2.event_id, e1.event_id, TemporalRelationshipType.SUPERSEDES),
                            source_id=e2.event_id,
                            target_id=e1.event_id,
                            relationship=TemporalRelationshipType.SUPERSEDES,
                            confidence=0.9,
                            evidence=f"New state '{e2.description}' ({e2.valid_from}) supersedes '{e1.description}' ({e1.valid_from})",
                            provenance_chunk_ids=[c for c in [e1.turn_id, e2.turn_id] if c],
                            derived_by="state_supersession_rule"
                        )
                    elif d1 > d2:
                        # e1 SUPERSEDES e2
                        return TemporalRelationshipEdge(
                            id=generate_temporal_edge_id(e1.event_id, e2.event_id, TemporalRelationshipType.SUPERSEDES),
                            source_id=e1.event_id,
                            target_id=e2.event_id,
                            relationship=TemporalRelationshipType.SUPERSEDES,
                            confidence=0.9,
                            evidence=f"New state '{e1.description}' ({e1.valid_from}) supersedes '{e2.description}' ({e2.valid_from})",
                            provenance_chunk_ids=[c for c in [e1.turn_id, e2.turn_id] if c],
                            derived_by="state_supersession_rule"
                        )

        return None

    def _check_invalidation(self, e1: EventNode, e2: EventNode) -> TemporalRelationshipEdge | None:
        """
        Check if a termination/cancellation event invalidates an active subscription/membership.
        """
        act1 = e1.properties.get("action_type")
        act2 = e2.properties.get("action_type")
        tgt1 = e1.properties.get("target", "").lower()
        tgt2 = e2.properties.get("target", "").lower()

        # Shared entity / target
        shared_target = (tgt1 and tgt2 and (tgt1 in tgt2 or tgt2 in tgt1))

        if shared_target:
            if act1 == "cancelled" and act2 in ("subscribed to", "participated in"):
                return TemporalRelationshipEdge(
                    id=generate_temporal_edge_id(e1.event_id, e2.event_id, TemporalRelationshipType.INVALIDATES),
                    source_id=e1.event_id,
                    target_id=e2.event_id,
                    relationship=TemporalRelationshipType.INVALIDATES,
                    confidence=0.95,
                    evidence=f"Cancellation event '{e1.description}' invalidates '{e2.description}'",
                    provenance_chunk_ids=[c for c in [e1.turn_id, e2.turn_id] if c],
                    derived_by="invalidation_rule"
                )
            elif act2 == "cancelled" and act1 in ("subscribed to", "participated in"):
                return TemporalRelationshipEdge(
                    id=generate_temporal_edge_id(e2.event_id, e1.event_id, TemporalRelationshipType.INVALIDATES),
                    source_id=e2.event_id,
                    target_id=e1.event_id,
                    relationship=TemporalRelationshipType.INVALIDATES,
                    confidence=0.95,
                    evidence=f"Cancellation event '{e2.description}' invalidates '{e1.description}'",
                    provenance_chunk_ids=[c for c in [e1.turn_id, e2.turn_id] if c],
                    derived_by="invalidation_rule"
                )

        return None

