"""
IVARA Temporal Memory, Event Graph, and Knowledge Supersession Package.

Provides:
- TemporalPrecision & TemporalRelationshipType: Typed temporal enums.
- EventNode & TemporalRelationshipEdge: Explicit episodic event representations and directed edges.
- TemporalEventExtractor: Decoupled event extraction with uncertainty preservation.
- TemporalRelationshipBuilder: Bounded temporal relationship construction (no ingestion-order inference).
- TemporalEventGraphRetriever: Bounded graph traversal and evidence formulation.
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
from .event_graph import (
    TemporalPrecision,
    TemporalRelationshipType,
    EventNode,
    TemporalRelationshipEdge,
    generate_event_id,
    generate_temporal_edge_id
)
from .event_extractor import TemporalEventExtractor
from .relationship_builder import TemporalRelationshipBuilder
from .event_graph_traversal import TemporalEventGraphRetriever

__all__ = [
    "TemporalPrecision",
    "TemporalRelationshipType",
    "EventNode",
    "TemporalRelationshipEdge",
    "generate_event_id",
    "generate_temporal_edge_id",
    "TemporalEventExtractor",
    "TemporalRelationshipBuilder",
    "TemporalEventGraphRetriever",
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
