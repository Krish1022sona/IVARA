"""
Turn-Aware Conversational Chunking (TACC) for IVARA.

Provides turn-level sliding window chunking over multi-turn dialogues to eliminate
whole-session vector dilution and context bloat during long-term memory retrieval.
"""

from .turn_chunker import TurnChunk, TurnAwareChunker
from .retrieval import ChunkRetriever, assemble_bounded_context
from .ontology_retrieval import OntologyGuidedChunkRetriever
from .graph_retrieval import TurnGraphBuilder, TurnGraphRetriever
from .hybrid_pipeline import DeterministicQueryRouter, UnifiedCandidateRetriever, RoutingDecision

__all__ = [
    "TurnChunk",
    "TurnAwareChunker",
    "ChunkRetriever",
    "assemble_bounded_context",
    "OntologyGuidedChunkRetriever",
    "TurnGraphBuilder",
    "TurnGraphRetriever",
    "DeterministicQueryRouter",
    "UnifiedCandidateRetriever",
    "RoutingDecision"
]

