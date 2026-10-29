"""
Conservative Knowledge Supersession Engine for IVARA.

Enforces:
1. Distinction between valid/event time and recording/session time.
2. Preservation of historical claims (no destructive deletes).
3. Directed supersession links (C_new -> C_old).
4. Functional vs non-functional predicate distinction (hobbies & visits coexist).
5. Explicit date interval comparisons (2026 > 2025).
6. Unresolved conflict preservation when state updates are ambiguous.
7. Zero fabricated valid_from dates from session ingestion timestamps.
"""

from dataclasses import dataclass
import re
from typing import Any

from ..graph.schema import GraphNode, GraphEdge, TemporalClaim
from ..graph.store import SQLiteGraphStore
from .claim import extract_year, overlaps_year


FUNCTIONAL_PREDICATES: set[str] = {
    "lives_in", "resides_in", "located_in", "current_city", "home_city", "city",
    "works_at", "employed_by", "employer", "company",
    "role", "job_title", "title", "position",
    "marital_status", "relationship_status",
    "phone_number", "email", "primary_email",
    "prefers", "favorite"
}

NON_FUNCTIONAL_PREDICATES: set[str] = {
    "has_hobby", "hobby", "interested_in", "likes", "enjoys",
    "visited", "visited_place", "traveled_to", "travelled_to",
    "knows", "friend_of", "collaborated_with", "worked_with",
    "owns", "bought", "purchased",
    "speaks", "language"
}

TRANSITION_CUES: list[str] = [
    "moved to", "relocated to", "transitioned to", "switched to",
    "changed job to", "started working at", "left", "no longer",
    "became", "transferred to", "now living in", "moved from",
    "now prefer", "used to prefer", "switched preference"
]


def normalize_token(s: str | None) -> str:
    """Normalize names and predicates for comparison."""
    if not s:
        return ""
    clean = re.sub(r'[^a-zA-Z0-9]', '', str(s).lower())
    if clean in ("user", "self", "me", "i"):
        return "user"
    return clean


def get_effective_event_year(claim: TemporalClaim) -> tuple[int | None, bool]:
    """
    Returns (year, is_explicit).
    Explicit valid_from event date is given top priority over recording timestamp.
    If valid_from is not set, fall back to year in recorded_at.
    """
    y_explicit = extract_year(claim.valid_from)
    if y_explicit is not None:
        return y_explicit, True
    y_rec = extract_year(claim.recorded_at)
    if y_rec is not None:
        return y_rec, False
    return None, False


class TemporalSupersessionEngine:
    """
    Evaluates pairs and streams of TemporalClaim objects to establish directed
    supersession relationships, maintain historical validity, and isolate conflicts.
    """

    def __init__(
        self,
        functional_predicates: set[str] | None = None,
        non_functional_predicates: set[str] | None = None
    ):
        self.functional_predicates = functional_predicates or set(FUNCTIONAL_PREDICATES)
        self.non_functional_predicates = non_functional_predicates or set(NON_FUNCTIONAL_PREDICATES)

    def is_functional(self, predicate: str) -> bool:
        """Check if predicate denotes single-valued state at any time t."""
        norm = normalize_token(predicate)
        return any(normalize_token(fp) == norm for fp in self.functional_predicates)

    def evaluate_pair(
        self,
        claim_old: TemporalClaim,
        claim_new: TemporalClaim
    ) -> tuple[bool, str]:
        """
        Evaluate whether claim_new genuinely supersedes claim_old.
        Returns:
            (does_supersede: bool, reason: str)
        """
        # 1. Subject match
        sub_old = normalize_token(claim_old.subject)
        sub_new = normalize_token(claim_new.subject)
        if sub_old != sub_new:
            return False, f"Subject mismatch: '{claim_old.subject}' vs '{claim_new.subject}'"

        # 2. Predicate match
        pred_old = normalize_token(claim_old.predicate)
        pred_new = normalize_token(claim_new.predicate)
        if pred_old != pred_new:
            return False, f"Predicate mismatch: '{claim_old.predicate}' vs '{claim_new.predicate}'"

        # 3. Context scope match: e.g. "at work" vs "for personal projects"
        ctx_old = claim_old.properties.get("context_scope", "general")
        ctx_new = claim_new.properties.get("context_scope", "general")
        if ctx_old != ctx_new and (ctx_old != "general" and ctx_new != "general"):
            return False, f"Context distinction: '{ctx_old}' vs '{ctx_new}' allows distinct coexisting values"

        # 4. Functional vs Non-functional check
        if not self.is_functional(claim_new.predicate):
            return False, f"Non-functional predicate '{claim_new.predicate}' allows coexisting facts"

        # 5. Object equivalence
        obj_old = normalize_token(claim_old.object)
        obj_new = normalize_token(claim_new.object)
        if obj_old == obj_new:
            return False, f"Identical object '{claim_old.object}' reaffirmation; no state change"

        # 6. Temporal Ordering Evaluation (Event time vs Record time)
        y_old_ev, is_old_explicit = get_effective_event_year(claim_old)
        y_new_ev, is_new_explicit = get_effective_event_year(claim_new)

        # Case A: Both have determined event years
        if y_old_ev is not None and y_new_ev is not None:
            if y_new_ev > y_old_ev:
                return True, f"Chronological state update: {y_new_ev} > {y_old_ev}"
            elif y_new_ev < y_old_ev:
                return False, f"Historical prior fact: event time {y_new_ev} < {y_old_ev} (late-recorded prior fact does not supersede later state)"
            elif is_old_explicit and is_new_explicit and y_new_ev == y_old_ev:
                return False, f"Ambiguous contradiction: conflicting claims in identical event year {y_new_ev}"

        # Case B: Event transition cues
        evidence_text = f"{claim_new.evidence or ''} {claim_new.properties.get('text', '')}".lower()
        has_transition = any(cue in evidence_text for cue in TRANSITION_CUES)

        rec_old = claim_old.recorded_at or ""
        rec_new = claim_new.recorded_at or ""

        if has_transition:
            if rec_old and rec_new and rec_new > rec_old:
                return True, f"State transition cue confirmed by recording order ({rec_new} > {rec_old})"
            elif y_new_ev is not None and y_old_ev is None:
                return True, f"Explicit date {y_new_ev} provides newer anchor over unanchored claim"
            elif claim_old.valid_to is not None:
                return True, "State transition updates terminated prior claim"

        # If one is explicit preference update linked to old claim
        if claim_new.supersedes_claim_id == claim_old.claim_id:
            return True, claim_new.properties.get("supersession_reason", "Explicit linked supersession")

        # Case C: Ambiguous contradiction
        return False, "Ambiguous contradiction: contradictory functional claims without temporal resolution or distinct context"

    def process_claims(self, claims: list[TemporalClaim]) -> list[TemporalClaim]:
        """
        Process a list of claims in chronological order, updating validity intervals,
        setting directed supersession links, and tagging active vs historical states.
        Preserves all original claims (no deletion).
        """
        if not claims:
            return []

        # Sort claims by (effective event year, recorded_at, claim_id)
        def _sort_key(c: TemporalClaim):
            y, is_exp = get_effective_event_year(c)
            return (
                y if y is not None else 9999,
                c.recorded_at or "",
                c.claim_id
            )

        processed = [c.model_copy(deep=True) for c in claims]
        processed.sort(key=_sort_key)

        # Set default active state
        for c in processed:
            if "is_current" not in c.properties:
                c.properties["is_current"] = True

        n = len(processed)
        for i in range(n):
            for j in range(i + 1, n):
                old_claim = processed[i]
                new_claim = processed[j]

                does_supersede, reason = self.evaluate_pair(old_claim, new_claim)
                if does_supersede:
                    new_claim.supersedes_claim_id = old_claim.claim_id
                    new_claim.properties["is_current"] = True
                    new_claim.properties["supersession_reason"] = reason

                    # Mark old claim as historical/superseded
                    old_claim.properties["is_current"] = False
                    old_claim.properties["superseded_by"] = new_claim.claim_id

                    # Close valid_to interval if new_claim has explicit start date
                    if new_claim.valid_from and not old_claim.valid_to:
                        y_new = extract_year(new_claim.valid_from)
                        y_old = extract_year(old_claim.valid_from)
                        if y_new is not None and y_old is not None and y_new > y_old:
                            old_claim.valid_to = str(y_new - 1)
                        else:
                            old_claim.valid_to = new_claim.valid_from

                elif "Ambiguous contradiction" in reason:
                    # Mark ambiguous conflict without destroying either claim
                    old_claim.properties["conflict_status"] = "unresolved"
                    new_claim.properties["conflict_status"] = "unresolved"
                    old_claim.properties["conflict_with"] = new_claim.claim_id
                    new_claim.properties["conflict_with"] = old_claim.claim_id

        return processed

    def apply_to_graph_store(
        self,
        store: SQLiteGraphStore,
        claims: list[TemporalClaim]
    ) -> dict[str, int]:
        """
        Persist processed claims into SQLiteGraphStore as ClaimNodes and directed 'supersedes' edges.
        """
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        for claim in claims:
            nodes.append(claim.to_graph_node())

            # If this claim supersedes an earlier claim, create directed supersedes edge
            if claim.supersedes_claim_id:
                edge = GraphEdge(
                    id=f"edge_sup_{claim.claim_id}_{claim.supersedes_claim_id}",
                    source_id=claim.claim_id,
                    target_id=claim.supersedes_claim_id,
                    edge_type="supersedes",
                    origin="explicit" if claim.is_explicit else "inferred",
                    confidence=claim.confidence,
                    evidence=claim.evidence,
                    timestamp=claim.valid_from or claim.recorded_at,
                    properties={
                        "valid_from": claim.valid_from,
                        "valid_to": claim.valid_to,
                        "recorded_at": claim.recorded_at
                    }
                )
                edges.append(edge)

        if hasattr(store, "batch_upsert"):
            store.batch_upsert(nodes=nodes, edges=edges)

        return {"claim_nodes": len(nodes), "supersedes_edges": len(edges)}
