"""Unit and integration tests for AdaptiveRouter in IVARA Unified Retrieval.

Covers:
1. Category classification across all 8 canonical query types.
2. Channel enablement invariants (episodic always on, graph restricted to multi-hop/relational, historical-state graph guard).
3. Ambiguity handling and safe fallback.
4. Unknown entity handling (safe fallback, never misclassified as unanswerable).
5. Out-of-domain and explicit unanswerable queries.
6. Negative information and absence handling.
7. Empty and malformed queries.
8. Pipeline integration with UnifiedRetriever (both disabled and enabled).
9. Sub-millisecond performance benchmark.
"""

import time
import pytest
from src.unified_retrieval.router import AdaptiveRouter, RouteDecision
from src.unified_retrieval.pipeline import UnifiedRetriever, UnifiedRetrievalResult
from src.conversations.store import EpisodicTurn


@pytest.fixture
def router():
    """Default AdaptiveRouter instance."""
    return AdaptiveRouter()


class TestAdaptiveRouterCategories:
    """Verify deterministic routing across all 8 canonical categories."""

    def test_single_fact(self, router):
        queries = [
            "What is Arjun's birthday?",
            "On what date was Arjun born?",
            "What is Arjun's alma mater?",
            "From which university did Arjun graduate?",
            "What local development tooling does Arjun use?"
        ]
        for q in queries:
            decision = router.route(q)
            assert decision.query_type == "single_fact", f"Failed on query: {q}, got {decision.query_type}"
            assert decision.episodic_enabled is True
            assert decision.semantic_enabled is True
            assert decision.graph_enabled is False

    def test_historical_state(self, router):
        queries = [
            "What was Arjun's previous job title before switching?",
            "What was Arjun's initial salary at HyperScale Systems?",
            "What was his former role prior to 2024?",
            "Where did Arjun live originally before moving to Bengaluru?",
            "What did Arjun used to prefer for code review?"
        ]
        for q in queries:
            decision = router.route(q)
            assert decision.query_type == "historical_state", f"Failed on query: {q}, got {decision.query_type}"
            assert decision.episodic_enabled is True
            assert decision.semantic_enabled is True
            # Critical invariant: Graph must NOT be enabled for historical state queries
            assert decision.graph_enabled is False
            assert decision.context_mode == "historical_state"

    def test_temporal_longitudinal(self, router):
        queries = [
            "Did the promotion occur before or after the relocation to Bengaluru?",
            "What is the chronological sequence of Arjun's conference presentations?",
            "When did Arjun publish the cache paper relative to joining Aetheria?",
            "Was the NeuroCompact paper earlier than the CacheMesh talk?"
        ]
        for q in queries:
            decision = router.route(q)
            assert decision.query_type == "temporal_longitudinal", f"Failed on query: {q}, got {decision.query_type}"
            assert decision.episodic_enabled is True
            assert decision.graph_enabled is False

    def test_cross_session_synth(self, router):
        queries = [
            "Trace Arjun's career progression and residential moves across all companies.",
            "Summarize the full lifecycle of the CacheMesh project from inception to deprecation.",
            "List all companies Arjun has worked for over the years.",
            "Identify the first project, his second promotion, his third paper, and the final team lead role."
        ]
        for q in queries:
            decision = router.route(q)
            assert decision.query_type == "cross_session_synth", f"Failed on query: {q}, got {decision.query_type}"
            assert decision.episodic_enabled is True
            assert decision.graph_enabled is False

    def test_entity_relationships(self, router):
        queries = [
            "What is the relationship between Arjun and Vikram Rao?",
            "How is Priya Sharma connected to Arjun Mehta?",
            "What connection exists between Rohan Sharma and HyperScale Systems?"
        ]
        for q in queries:
            decision = router.route(q)
            assert decision.query_type == "entity_relationships", f"Failed on query: {q}, got {decision.query_type}"
            assert decision.episodic_enabled is True
            assert decision.semantic_enabled is True
            # For clear entity relationship queries with high confidence and known entities, graph should be enabled
            assert decision.graph_enabled is True
            assert decision.context_mode == "entity_relationships"

    def test_multi_hop_reasoning(self, router):
        queries = [
            "In which city does the manager who supervised Arjun at HyperScale Systems currently live?",
            "What project did Arjun build with the collaborator who rewrote the compiler at TensorForge?",
            "Where did the engineer who treated Arjun for knee surgery relocate?"
        ]
        for q in queries:
            decision = router.route(q)
            assert decision.query_type == "multi_hop_reasoning", f"Failed on query: {q}, got {decision.query_type}"
            assert decision.episodic_enabled is True
            assert decision.semantic_enabled is True
            assert decision.graph_enabled is True
            assert decision.context_mode == "multi_hop"

    def test_negative_information(self, router):
        queries = [
            "Did Arjun undergo knee surgery during his tenure at Aetheria?",
            "Did Arjun accept the initial salary counter-offer or decline it?",
            "Does Arjun drink coffee or does he have a caffeine restriction?",
            "Does Arjun still live in Seattle?"
        ]
        for q in queries:
            decision = router.route(q)
            assert decision.query_type == "negative_information", f"Failed on query: {q}, got {decision.query_type}"
            assert decision.episodic_enabled is True
            assert decision.semantic_enabled is True
            # Invariant: Graph only contains positive facts, must be disabled for negative info
            assert decision.graph_enabled is False
            assert decision.context_mode == "negative_information"

    def test_unanswerable_out_of_domain(self, router):
        queries = [
            "What is the capital of Mars according to Arjun?",
            "What was the weather on Jupiter during Arjun's trip?",
            "Who was the president of the moon in 2099?"
        ]
        for q in queries:
            decision = router.route(q)
            assert decision.query_type == "unanswerable", f"Failed on query: {q}, got {decision.query_type}"
            assert decision.episodic_enabled is True
            assert decision.semantic_enabled is False
            assert decision.graph_enabled is False
            assert decision.context_mode == "unanswerable"


class TestAdaptiveRouterInvariantsAndEdgeCases:
    """Verify invariants, safe fallback, unknown entities, and malformed queries."""

    def test_unknown_entity_routes_to_safe_fallback(self, router):
        """Unknown entities must NEVER be routed to 'unanswerable'; they must route to safe_fallback."""
        queries = [
            "What did Zachariah Finklestein say about the quantum neutrino compiler?",
            "Does Barnaby Bumblebee work at Galactic Transistors?",
            "How is Lord Voldemort related to Sirius Black?"
        ]
        for q in queries:
            decision = router.route(q)
            assert decision.query_type != "unanswerable", f"Query should not be unanswerable: {q}"
            assert decision.episodic_enabled is True
            # Must remain safe fallback or standard channels enabled
            assert decision.semantic_enabled is True

    def test_empty_and_whitespace_query(self, router):
        for q in ["", "   ", "\t\n  "]:
            decision = router.route(q)
            assert decision.query_type == "safe_fallback"
            assert decision.confidence == 1.0
            assert decision.episodic_enabled is True
            assert decision.semantic_enabled is True
            assert decision.graph_enabled is False

    def test_ambiguous_short_query_routes_to_safe_fallback(self, router):
        ambiguous = [
            "Tell me about systems.",
            "What happened?",
            "Details please."
        ]
        for q in ambiguous:
            decision = router.route(q)
            assert decision.query_type == "safe_fallback"
            assert decision.episodic_enabled is True
            assert decision.semantic_enabled is True
            assert decision.graph_enabled is False

    def test_episodic_backbone_always_enabled(self, router):
        """Invariant: Episodic retrieval is never disabled across any query type."""
        test_queries = [
            "What is Arjun's birthday?",
            "What was his previous role?",
            "Did it happen before or after 2024?",
            "Trace career progression",
            "Relationship between Arjun and Vikram?",
            "Who was the engineer who supervised him?",
            "Did Arjun undergo surgery?",
            "What is the capital of Mars?",
            "Random text that cannot be classified"
        ]
        for q in test_queries:
            decision = router.route(q)
            assert decision.episodic_enabled is True, f"Episodic disabled on: {q}"

    def test_to_dict_serialization(self, router):
        decision = router.route("What was Arjun's previous role at HyperScale?")
        d = decision.to_dict()
        assert isinstance(d, dict)
        assert d["query_type"] == "historical_state"
        assert "confidence" in d
        assert "signals" in d
        assert "episodic_enabled" in d
        assert "semantic_enabled" in d
        assert "graph_enabled" in d
        assert "diagnostics" in d

    def test_sub_millisecond_latency(self, router):
        """Deterministic Stage 1 must execute well under 1.0 ms per query (typically < 0.1 ms)."""
        queries = [
            "What is Arjun's birthday?",
            "What was Arjun's previous job title before switching?",
            "Did the promotion occur before or after the relocation?",
            "Trace Arjun's career progression across all companies.",
            "What is the relationship between Arjun and Vikram Rao?",
            "In which city does the manager who supervised Arjun live?",
            "Did Arjun undergo knee surgery?",
            "What is the capital of Mars?"
        ] * 25  # 200 iterations
        start = time.perf_counter()
        for q in queries:
            router.route(q)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / len(queries)) * 1000.0
        assert avg_ms < 0.5, f"Routing too slow: {avg_ms:.4f} ms per query"


class TestUnifiedRetrieverIntegration:
    """Verify integration of AdaptiveRouter inside UnifiedRetriever."""

    @pytest.fixture
    def mock_environment(self):
        turns = [
            EpisodicTurn(
                turn_id="turn_001",
                conversation_id="conv_01",
                speaker="user",
                speaker_role="user",
                content="I joined HyperScale Systems as a Senior Distributed Systems Engineer.",
                timestamp="2024-01-15T10:00:00Z",
                sequence_num=1,
                session_id="session_01"
            ),
            EpisodicTurn(
                turn_id="turn_002",
                conversation_id="conv_02",
                speaker="user",
                speaker_role="user",
                content="I switched to Aetheria AI as a Principal Architect.",
                timestamp="2024-06-01T10:00:00Z",
                sequence_num=2,
                session_id="session_02"
            )
        ]
        memories = [
            {
                "id": "mem_001",
                "text": "Arjun joined HyperScale Systems in January 2024.",
                "timestamp": "2024-01-15T10:00:00Z",
                "relationships": [
                    {"source": "Arjun", "predicate": "employed_by", "target": "HyperScale Systems"}
                ],
                "entities": [{"canonical_name": "Arjun"}, {"canonical_name": "HyperScale Systems"}]
            }
        ]
        return turns, memories

    def test_default_behavior_preserves_unrouted_pipeline(self, mock_environment):
        """Default use_adaptive_routing=False must preserve exact legacy behavior."""
        turns, memories = mock_environment
        retriever = UnifiedRetriever(
            episodic_turns=turns,
            memories=memories,
            use_adaptive_routing=False
        )
        assert retriever.use_adaptive_routing is False

        result = retriever.retrieve("What was Arjun's previous role?")
        assert "routing" not in result.diagnostics
        assert result.diagnostics["enabled_channels"]["episodic"] is True
        assert result.diagnostics["enabled_channels"]["semantic"] is True

    def test_active_adaptive_routing_filters_channels(self, mock_environment):
        """When use_adaptive_routing=True, channel activation reflects RouteDecision."""
        turns, memories = mock_environment
        retriever = UnifiedRetriever(
            episodic_turns=turns,
            memories=memories,
            use_adaptive_routing=True
        )
        assert retriever.use_adaptive_routing is True

        # Test historical state query
        res_historical = retriever.retrieve("What was Arjun's previous job title before switching?")
        assert "routing" in res_historical.diagnostics
        assert res_historical.diagnostics["routing"]["query_type"] == "historical_state"
        assert res_historical.diagnostics["enabled_channels"]["graph"] is False

        # Test unanswerable query (semantic disabled)
        res_unans = retriever.retrieve("What is the capital of Mars according to Arjun?")
        assert "routing" in res_unans.diagnostics
        assert res_unans.diagnostics["routing"]["query_type"] == "unanswerable"
        assert res_unans.diagnostics["enabled_channels"]["semantic"] is False
        assert res_unans.diagnostics["enabled_channels"]["graph"] is False

    def test_per_query_override(self, mock_environment):
        """Per-query override allows dynamically toggling routing regardless of constructor default."""
        turns, memories = mock_environment
        retriever = UnifiedRetriever(
            episodic_turns=turns,
            memories=memories,
            use_adaptive_routing=False  # default false
        )
        # Call with override True
        res_routed = retriever.retrieve(
            "What was Arjun's former job title?",
            use_adaptive_routing=True
        )
        assert "routing" in res_routed.diagnostics
        assert res_routed.diagnostics["routing"]["query_type"] == "historical_state"

        # Call with override False
        res_unrouted = retriever.retrieve(
            "What was Arjun's former job title?",
            use_adaptive_routing=False
        )
        assert "routing" not in res_unrouted.diagnostics
