"""
Explicit Contradiction Analysis and Resolution Engine for IVARA.

Distinguishes:
1. LEGITIMATE_UPDATE: Sequential state change over time on a functional predicate (e.g. "lived in Delhi in 2023; moved to Bangalore in 2025").
2. TRUE_CONFLICT: Incompatible claims about the same subject, predicate, and time period with no resolution.
3. CONTEXT_DISTINCTION: Compatible claims qualified by distinct contexts (e.g. "prefers Python at work vs Rust for personal projects").
4. UNCERTAINTY: Conditional, speculative, or tentative statements vs factual assertions (e.g. "might move next month").
5. NEGATION: Explicit denial/negation of a state or ownership (e.g. "I do not own a car" vs "owns a car").
6. COMPATIBLE: Consistent facts or coexisting non-functional claims (e.g. multiple hobbies, colleagues, visited places).
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from ..graph.schema import TemporalClaim
from ..temporal.supersession import (
    FUNCTIONAL_PREDICATES,
    NON_FUNCTIONAL_PREDICATES,
    TRANSITION_CUES,
    normalize_token,
    get_effective_event_year
)


class ContradictionType(str, Enum):
    LEGITIMATE_UPDATE = "legitimate_update"
    TRUE_CONFLICT = "true_conflict"
    CONTEXT_DISTINCTION = "context_distinction"
    UNCERTAINTY = "uncertainty"
    NEGATION = "negation"
    COMPATIBLE = "compatible"


@dataclass
class ContradictionResult:
    """Detailed evaluation result for a pair of claims."""
    type: ContradictionType
    is_conflict: bool
    is_resolved: bool
    explanation: str
    active_claim_id: str | None = None
    historical_claim_id: str | None = None
    conflicting_claim_ids: list[str] = field(default_factory=list)
    recommended_action: str = "use_active"  # "use_active" | "surface_conflict" | "preserve_both" | "ignore_uncertain"


class ContradictionAnalyzer:
    """
    Analyzes pairs and sets of claims to classify semantic and temporal relationships,
    detecting genuine contradictions while respecting updates, contexts, negations, and uncertainty.
    """

    def __init__(
        self,
        functional_predicates: set[str] | None = None,
        non_functional_predicates: set[str] | None = None
    ):
        self.functional_predicates = functional_predicates or set(FUNCTIONAL_PREDICATES)
        self.non_functional_predicates = non_functional_predicates or set(NON_FUNCTIONAL_PREDICATES)

    def is_functional(self, predicate: str) -> bool:
        norm = normalize_token(predicate)
        return any(normalize_token(fp) == norm for fp in self.functional_predicates)

    def analyze_pair(self, claim_a: TemporalClaim, claim_b: TemporalClaim) -> ContradictionResult:
        """
        Analyze two claims to determine whether they represent an update, conflict,
        context distinction, uncertainty, negation, or compatible facts.
        """
        # 1. Subject check
        sub_a = normalize_token(claim_a.subject)
        sub_b = normalize_token(claim_b.subject)
        if sub_a != sub_b:
            return ContradictionResult(
                type=ContradictionType.COMPATIBLE,
                is_conflict=False,
                is_resolved=True,
                explanation=f"Distinct subjects ('{claim_a.subject}' vs '{claim_b.subject}') coexist."
            )

        # 2. Predicate check
        pred_a = normalize_token(claim_a.predicate)
        pred_b = normalize_token(claim_b.predicate)
        if pred_a != pred_b:
            return ContradictionResult(
                type=ContradictionType.COMPATIBLE,
                is_conflict=False,
                is_resolved=True,
                explanation=f"Distinct predicates ('{claim_a.predicate}' vs '{claim_b.predicate}') coexist."
            )

        # 3. Negation check
        neg_a = claim_a.properties.get("is_negated", False)
        neg_b = claim_b.properties.get("is_negated", False)
        obj_a = normalize_token(claim_a.object)
        obj_b = normalize_token(claim_b.object)

        if (neg_a != neg_b) and (obj_a == obj_b or not obj_a or not obj_b):
            # One asserts X, one negates X
            rec_a = claim_a.recorded_at or ""
            rec_b = claim_b.recorded_at or ""
            if rec_a and rec_b and rec_a != rec_b:
                newer_c = claim_b if rec_b > rec_a else claim_a
                older_c = claim_a if rec_b > rec_a else claim_b
                status_str = "negated" if newer_c.properties.get("is_negated") else "affirmed"
                return ContradictionResult(
                    type=ContradictionType.NEGATION,
                    is_conflict=True,
                    is_resolved=True,
                    active_claim_id=newer_c.claim_id,
                    historical_claim_id=older_c.claim_id,
                    explanation=f"Explicit state update by negation: later claim at {newer_c.recorded_at} {status_str} earlier state.",
                    recommended_action="use_active"
                )
            else:
                return ContradictionResult(
                    type=ContradictionType.NEGATION,
                    is_conflict=True,
                    is_resolved=False,
                    conflicting_claim_ids=[claim_a.claim_id, claim_b.claim_id],
                    explanation=f"Unresolved direct negation: one memory asserts '{claim_a.object}' while another denies it without chronological resolution.",
                    recommended_action="surface_conflict"
                )

        # 4. Uncertainty check
        unc_a = claim_a.properties.get("is_uncertain", False) or claim_a.confidence < 0.70
        unc_b = claim_b.properties.get("is_uncertain", False) or claim_b.confidence < 0.70
        if unc_a or unc_b:
            factual_c = claim_b if (unc_a and not unc_b) else (claim_a if (unc_b and not unc_a) else None)
            uncertain_c = claim_a if unc_a else claim_b
            if factual_c:
                return ContradictionResult(
                    type=ContradictionType.UNCERTAINTY,
                    is_conflict=False,
                    is_resolved=True,
                    active_claim_id=factual_c.claim_id,
                    explanation=f"Factual assertion '{factual_c.object}' takes precedence over speculative statement '{uncertain_c.object}'.",
                    recommended_action="use_active"
                )
            else:
                return ContradictionResult(
                    type=ContradictionType.UNCERTAINTY,
                    is_conflict=False,
                    is_resolved=False,
                    conflicting_claim_ids=[claim_a.claim_id, claim_b.claim_id],
                    explanation=f"Both statements are tentative or speculative ('{claim_a.object}' vs '{claim_b.object}').",
                    recommended_action="surface_conflict"
                )

        # 5. Context Distinction check (e.g., at work vs for personal projects)
        ctx_a = claim_a.properties.get("context_scope", "general")
        ctx_b = claim_b.properties.get("context_scope", "general")
        if ctx_a != ctx_b and (ctx_a != "general" and ctx_b != "general"):
            return ContradictionResult(
                type=ContradictionType.CONTEXT_DISTINCTION,
                is_conflict=False,
                is_resolved=True,
                explanation=f"Context-dependent preferences/values ('{claim_a.object}' {ctx_a} vs '{claim_b.object}' {ctx_b}) are complementary and compatible.",
                recommended_action="preserve_both"
            )

        # 6. Non-functional predicate check
        if not self.is_functional(claim_a.predicate):
            return ContradictionResult(
                type=ContradictionType.COMPATIBLE,
                is_conflict=False,
                is_resolved=True,
                explanation=f"Non-functional predicate '{claim_a.predicate}' allows multiple coexisting values ('{claim_a.object}', '{claim_b.object}').",
                recommended_action="preserve_both"
            )

        # 7. Identical value check
        if obj_a == obj_b:
            return ContradictionResult(
                type=ContradictionType.COMPATIBLE,
                is_conflict=False,
                is_resolved=True,
                active_claim_id=claim_a.claim_id,
                explanation=f"Identical values ('{claim_a.object}') reaffirm each other.",
                recommended_action="preserve_both"
            )

        # 8. Functional Predicate with Different Values -> Evaluate Temporal Resolution
        y_a, is_a_exp = get_effective_event_year(claim_a)
        y_b, is_b_exp = get_effective_event_year(claim_b)

        # Explicit preference transition
        if claim_b.supersedes_claim_id == claim_a.claim_id or claim_b.properties.get("superseded_by") == claim_a.claim_id:
            active_c = claim_b if claim_b.properties.get("is_current", True) else claim_a
            hist_c = claim_a if active_c == claim_b else claim_b
            return ContradictionResult(
                type=ContradictionType.LEGITIMATE_UPDATE,
                is_conflict=False,
                is_resolved=True,
                active_claim_id=active_c.claim_id,
                historical_claim_id=hist_c.claim_id,
                explanation=f"Legitimate state update: '{hist_c.object}' updated to '{active_c.object}'.",
                recommended_action="use_active"
            )

        # Explicit event year comparison
        if y_a is not None and y_b is not None:
            if y_b > y_a:
                return ContradictionResult(
                    type=ContradictionType.LEGITIMATE_UPDATE,
                    is_conflict=False,
                    is_resolved=True,
                    active_claim_id=claim_b.claim_id,
                    historical_claim_id=claim_a.claim_id,
                    explanation=f"Legitimate chronological update: event date {y_b} updates prior {y_a} state ('{claim_a.object}' -> '{claim_b.object}').",
                    recommended_action="use_active"
                )
            elif y_a > y_b:
                return ContradictionResult(
                    type=ContradictionType.LEGITIMATE_UPDATE,
                    is_conflict=False,
                    is_resolved=True,
                    active_claim_id=claim_a.claim_id,
                    historical_claim_id=claim_b.claim_id,
                    explanation=f"Legitimate chronological update: event date {y_a} updates prior {y_b} state ('{claim_b.object}' -> '{claim_a.object}').",
                    recommended_action="use_active"
                )
            elif is_a_exp and is_b_exp and y_a == y_b:
                return ContradictionResult(
                    type=ContradictionType.TRUE_CONFLICT,
                    is_conflict=True,
                    is_resolved=False,
                    conflicting_claim_ids=[claim_a.claim_id, claim_b.claim_id],
                    explanation=f"Unresolved conflict: contradictory values ('{claim_a.object}' vs '{claim_b.object}') asserted for the identical period ({y_a}).",
                    recommended_action="surface_conflict"
                )

        # Transition cues with recording timestamps
        ev_a = f"{claim_a.evidence or ''} {claim_a.properties.get('text', '')}".lower()
        ev_b = f"{claim_b.evidence or ''} {claim_b.properties.get('text', '')}".lower()
        has_trans_a = any(cue in ev_a for cue in TRANSITION_CUES)
        has_trans_b = any(cue in ev_b for cue in TRANSITION_CUES)

        rec_a = claim_a.recorded_at or ""
        rec_b = claim_b.recorded_at or ""

        if has_trans_b and rec_b and rec_a and rec_b > rec_a:
            return ContradictionResult(
                type=ContradictionType.LEGITIMATE_UPDATE,
                is_conflict=False,
                is_resolved=True,
                active_claim_id=claim_b.claim_id,
                historical_claim_id=claim_a.claim_id,
                explanation=f"Legitimate transition cue ('{claim_b.object}') supersedes prior state ('{claim_a.object}') based on session chronology.",
                recommended_action="use_active"
            )
        elif has_trans_a and rec_a and rec_b and rec_a > rec_b:
            return ContradictionResult(
                type=ContradictionType.LEGITIMATE_UPDATE,
                is_conflict=False,
                is_resolved=True,
                active_claim_id=claim_a.claim_id,
                historical_claim_id=claim_b.claim_id,
                explanation=f"Legitimate transition cue ('{claim_a.object}') supersedes prior state ('{claim_b.object}') based on session chronology.",
                recommended_action="use_active"
            )

        # Unresolved Incompatible Claims
        return ContradictionResult(
            type=ContradictionType.TRUE_CONFLICT,
            is_conflict=True,
            is_resolved=False,
            conflicting_claim_ids=[claim_a.claim_id, claim_b.claim_id],
            explanation=f"Unresolved conflict: incompatible values ('{claim_a.object}' vs '{claim_b.object}') on functional predicate '{claim_a.predicate}' without temporal resolution.",
            recommended_action="surface_conflict"
        )

    def analyze_claims(self, claims: list[TemporalClaim]) -> dict[str, Any]:
        """
        Analyze an entire set of claims, identifying all pairwise relationships,
        unresolved contradictions, legitimate updates, and context distinctions.
        """
        results: list[ContradictionResult] = []
        unresolved_conflicts: list[dict[str, Any]] = []
        legitimate_updates: list[dict[str, Any]] = []
        context_distinctions: list[dict[str, Any]] = []
        active_claim_ids: set[str] = {c.claim_id for c in claims}
        superseded_claim_ids: set[str] = set()

        n = len(claims)
        for i in range(n):
            for j in range(i + 1, n):
                res = self.analyze_pair(claims[i], claims[j])
                results.append(res)

                if res.type == ContradictionType.TRUE_CONFLICT and not res.is_resolved:
                    unresolved_conflicts.append({
                        "claims": [claims[i].claim_id, claims[j].claim_id],
                        "subject": claims[i].subject,
                        "predicate": claims[i].predicate,
                        "values": [claims[i].object, claims[j].object],
                        "explanation": res.explanation
                    })
                elif res.type == ContradictionType.LEGITIMATE_UPDATE and res.is_resolved:
                    if res.historical_claim_id:
                        superseded_claim_ids.add(res.historical_claim_id)
                        active_claim_ids.discard(res.historical_claim_id)
                    legitimate_updates.append({
                        "active": res.active_claim_id,
                        "historical": res.historical_claim_id,
                        "explanation": res.explanation
                    })
                elif res.type == ContradictionType.CONTEXT_DISTINCTION:
                    context_distinctions.append({
                        "claims": [claims[i].claim_id, claims[j].claim_id],
                        "contexts": [
                            claims[i].properties.get("context_scope"),
                            claims[j].properties.get("context_scope")
                        ],
                        "values": [claims[i].object, claims[j].object],
                        "explanation": res.explanation
                    })

        return {
            "has_unresolved_conflicts": len(unresolved_conflicts) > 0,
            "unresolved_conflicts": unresolved_conflicts,
            "legitimate_updates": legitimate_updates,
            "context_distinctions": context_distinctions,
            "active_claim_ids": list(active_claim_ids),
            "superseded_claim_ids": list(superseded_claim_ids),
            "pair_results": results
        }
