"""IVARA Unified Multi-Source Retrieval System.

Coordinates evidence retrieval and fusion across:
1. Episodic Memory (raw conversational turns)
2. Semantic Memory (extracted facts and claims)
3. Knowledge Graph (entities and relationships)
"""

from .episodic import EpisodicRetriever, EpisodicEvidenceItem
from .semantic import SemanticRetriever, SemanticEvidenceItem
from .graph import GraphRetriever, GraphEvidenceItem
from .fusion import UnifiedEvidenceFuser, UnifiedEvidenceItem, RRFConfig
from .pipeline import UnifiedRetriever, UnifiedRetrievalResult
from .router import AdaptiveRouter, RouteDecision
from .graph_safeguards import GraphSafeguards, GraphSafeguardDecision, GraphSafeguardReport
from .evidence_selector import AdaptiveEvidenceSelector, SelectedEvidenceItem, SelectionResult

__all__ = [
    "EpisodicRetriever",
    "EpisodicEvidenceItem",
    "SemanticRetriever",
    "SemanticEvidenceItem",
    "GraphRetriever",
    "GraphEvidenceItem",
    "UnifiedEvidenceFuser",
    "UnifiedEvidenceItem",
    "RRFConfig",
    "UnifiedRetriever",
    "UnifiedRetrievalResult",
    "AdaptiveRouter",
    "RouteDecision",
    "GraphSafeguards",
    "GraphSafeguardDecision",
    "GraphSafeguardReport",
    "AdaptiveEvidenceSelector",
    "SelectedEvidenceItem",
    "SelectionResult"
]

