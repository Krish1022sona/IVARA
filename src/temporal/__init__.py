"""
IVARA Temporal Memory and Knowledge Supersession Package.

Provides:
- TemporalClaim: Backward-compatible claim representation with valid and recording times.
- TemporalSupersessionEngine: Conservative state-update rules and directed supersession edges.
- QueryTemporalResolver: Distinguishes current-state, historical interval, relative ordering, and ambiguous queries.
"""

from ..graph.schema import TemporalClaim
from .claim import build_claims_from_text, extract_year, overlaps_year, generate_claim_id
from .supersession import (
    TemporalSupersessionEngine,
    FUNCTIONAL_PREDICATES,
    NON_FUNCTIONAL_PREDICATES
)
from .resolution import QueryTemporalResolver, TemporalQueryIntent

__all__ = [
    "TemporalClaim",
    "TemporalSupersessionEngine",
    "QueryTemporalResolver",
    "TemporalQueryIntent",
    "build_claims_from_text",
    "extract_year",
    "overlaps_year",
    "generate_claim_id",
    "FUNCTIONAL_PREDICATES",
    "NON_FUNCTIONAL_PREDICATES"
]
