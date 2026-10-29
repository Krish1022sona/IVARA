"""
Query-Time Temporal Resolution for IVARA.

Distinguishes:
1. Current-state questions ("Where do I live now?").
2. Historical interval questions ("Where did I live in 2025?").
3. Relative ordering questions ("What happened before I moved?").
4. Ambiguous / unanchored questions.

Aligns candidate claims with query temporal intent, labels historical vs current states,
and formats temporal annotations for reader context.
"""

from enum import Enum
import re
from typing import Any

from ..graph.schema import TemporalClaim
from .claim import extract_year, overlaps_year


class TemporalQueryIntent(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    RELATIVE = "relative"
    AMBIGUOUS = "ambiguous"


CURRENT_CUES: list[str] = [
    "now", "currently", "at present", "presently", "today",
    "current", "right now", "latest", "recent"
]

RELATIVE_CUES: list[str] = [
    "before", "prior to", "after", "subsequently", "first",
    "what happened next", "what happened before", "earlier", "previously"
]


class QueryTemporalResolver:
    """
    Analyzes user queries for temporal intent and filters/ranks claims accordingly.
    """

    def detect_intent(self, query: str) -> tuple[TemporalQueryIntent, dict[str, Any]]:
        """
        Detect temporal intent and extract temporal anchors (e.g. target year).
        """
        q_lower = query.strip().lower()

        # 1. Check for explicit 4-digit calendar year (e.g., "in 2025", "during 2024")
        year_match = re.search(r'\b(19\d\d|20\d\d)\b', q_lower)
        if year_match:
            target_year = int(year_match.group(1))
            return TemporalQueryIntent.HISTORICAL, {"target_year": target_year, "raw_anchor": str(target_year)}

        # 2. Check for current-state cues
        for cue in CURRENT_CUES:
            if re.search(rf'\b{re.escape(cue)}\b', q_lower):
                return TemporalQueryIntent.CURRENT, {"cue": cue}

        # 3. Check for relative ordering cues
        for cue in RELATIVE_CUES:
            if re.search(rf'\b{re.escape(cue)}\b', q_lower):
                return TemporalQueryIntent.RELATIVE, {"cue": cue}

        # 4. Check for past tense indicators without specific year
        if any(w in q_lower.split() for w in ["did", "was", "were", "used", "past"]):
            return TemporalQueryIntent.HISTORICAL, {"target_year": None, "raw_anchor": "past_general"}

        return TemporalQueryIntent.AMBIGUOUS, {}

    def resolve(
        self,
        query: str,
        claims: list[TemporalClaim]
    ) -> dict[str, Any]:
        """
        Resolve candidate claims against query temporal intent.
        Returns:
            {
                "intent": TemporalQueryIntent,
                "anchor": dict,
                "selected_claims": list[TemporalClaim],
                "historical_claims": list[TemporalClaim],
                "unresolved_conflicts": list[TemporalClaim],
                "temporal_summary": str
            }
        """
        intent, anchor = self.detect_intent(query)

        if not claims:
            return {
                "intent": intent,
                "anchor": anchor,
                "selected_claims": [],
                "historical_claims": [],
                "unresolved_conflicts": [],
                "temporal_summary": "No claims available"
            }

        selected: list[TemporalClaim] = []
        historical: list[TemporalClaim] = []
        conflicts: list[TemporalClaim] = []

        # Find superseded claim IDs
        superseded_ids = {
            c.supersedes_claim_id
            for c in claims
            if c.supersedes_claim_id
        }

        # Check for unresolved conflicts
        for c in claims:
            if c.properties.get("conflict_status") == "unresolved":
                conflicts.append(c)

        if intent == TemporalQueryIntent.CURRENT:
            for c in claims:
                is_curr = c.properties.get("is_current", True)
                if is_curr and c.claim_id not in superseded_ids:
                    selected.append(c)
                else:
                    historical.append(c)

            summary = f"Current state requested: Selected {len(selected)} active claims; {len(historical)} historical claims superseded."

        elif intent == TemporalQueryIntent.HISTORICAL:
            target_year = anchor.get("target_year")
            if target_year is not None:
                for c in claims:
                    if overlaps_year(c.valid_from, c.valid_to, target_year):
                        selected.append(c)
                    else:
                        historical.append(c)
                summary = f"Historical state requested for year {target_year}: {len(selected)} claims valid during {target_year}."
            else:
                # Past general: include all historical and active claims, sorted by date
                selected = list(claims)
                summary = f"Historical query without explicit year: Returning {len(selected)} claims."

        elif intent == TemporalQueryIntent.RELATIVE:
            # Sort claims chronologically to show state sequence
            def _chrono_key(c: TemporalClaim):
                y = extract_year(c.valid_from)
                return (
                    y if y is not None else 9999,
                    c.recorded_at or "",
                    c.claim_id
                )
            selected = sorted(claims, key=_chrono_key)
            summary = f"Relative ordering query: Returning {len(selected)} claims sorted chronologically."

        else:
            # Ambiguous: retain all claims
            selected = list(claims)
            summary = f"Ambiguous/unanchored query: Retaining {len(selected)} claims."

        return {
            "intent": intent,
            "anchor": anchor,
            "selected_claims": selected,
            "historical_claims": historical,
            "unresolved_conflicts": conflicts,
            "temporal_summary": summary
        }

    def format_temporal_context(self, resolved: dict[str, Any]) -> str:
        """
        Format resolved claims into clear context lines for answer generation.
        """
        lines = []
        intent = resolved.get("intent")
        selected = resolved.get("selected_claims", [])
        historical = resolved.get("historical_claims", [])
        conflicts = resolved.get("unresolved_conflicts", [])

        if conflicts:
            lines.append("--- UNRESOLVED TEMPORAL CONFLICTS ---")
            for c in conflicts:
                lines.append(f"- Note conflicting claim: {c.subject} {c.predicate} {c.object} (Recorded: {c.recorded_at or 'unknown'})")

        if selected:
            header = "--- CURRENT VALID CLAIMS ---" if intent == TemporalQueryIntent.CURRENT else "--- TEMPORALLY VALID CLAIMS ---"
            lines.append(header)
            for c in selected:
                date_str = f"valid: {c.valid_from}" if c.valid_from else f"recorded: {c.recorded_at or 'unknown'}"
                lines.append(f"- [{c.claim_id}] {c.subject} {c.predicate} {c.object} ({date_str})")

        if historical and intent == TemporalQueryIntent.CURRENT:
            lines.append("--- HISTORICAL (SUPERSEDED) CLAIMS ---")
            for c in historical:
                date_str = f"valid: {c.valid_from} to {c.valid_to or 'end'}" if c.valid_from else "past"
                lines.append(f"- [HISTORICAL - Superseded] {c.subject} {c.predicate} {c.object} ({date_str})")

        return "\n".join(lines)
