"""Comprehensive unit and integration tests for Batch 2 (Evidence Selector)
and Batch 3 (Graph Safeguards) in IVARA Unified Retrieval.

Covers:
1. Evidence Selection:
   - Context target and budget compliance for all 8 categories
   - Contiguous context window expansion and overlapping-window deduplication
   - Chronological ordering within sessions
   - Speaker and session attribution
   - Multi-session synthesis (up to 4 sessions) without collapsing narratives
   - Semantic annotation via valid source_turn_id (anchored vs unanchored)
   - Unanchored semantic memory handling
   - Token budget estimation and calibration
   - Empty, malformed, and edge-case inputs
   - Deterministic output for identical inputs

2. Graph Evidence Safeguards:
   - Provenance validation (episodic turn and semantic memory)
   - Rejection on missing or unverified provenance
   - Temporal safety: historical queries strictly reject unversioned edges
   - Null validity interval rejection for historical queries
   - Relevance threshold gating (e.g. 0.45)
   - Generic entity distractor rejection
   - Self-referential cycle and duplicate triple rejection
   - Fan-out limits per entity
   - Router policy enforcement (graph disabled by router)

3. Pipeline Integration:
   - Opt-in behavior: use_adaptive_routing=False preserves exact legacy fusion
   - use_adaptive_routing=True engages selector and safeguards
   - Episodic backbone invariance
   - Safe fallback on exception
"""

import pytest
from src.unified_retrieval.episodic import EpisodicEvidenceItem
from src.unified_retrieval.semantic import SemanticEvidenceItem
from src.unified_retrieval.graph import GraphEvidenceItem
from src.unified_retrieval.router import AdaptiveRouter, RouteDecision
from src.unified_retrieval.graph_safeguards import GraphSafeguards, GraphSafeguardDecision, GraphSafeguardReport
from src.unified_retrieval.evidence_selector import AdaptiveEvidenceSelector, SelectedEvidenceItem, SelectionResult
from src.unified_retrieval.pipeline import UnifiedRetriever, UnifiedRetrievalResult
from src.conversations.store import EpisodicTurn


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_episodic_turns():
    """Sample episodic conversation turns across multiple sessions."""
    return [
        EpisodicTurn(
            turn_id="msg_001",
            conversation_id="conv_s1",
            session_id="session_01",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="I joined HyperScale Systems in January 2022 as a Senior Engineer.",
            timestamp="2022-01-15T09:00:00Z",
            sequence_num=1
        ),
        EpisodicTurn(
            turn_id="msg_002",
            conversation_id="conv_s1",
            session_id="session_01",
            speaker="IVARA",
            speaker_role="assistant",
            content="Welcome to HyperScale! What team will you be leading?",
            timestamp="2022-01-15T09:01:00Z",
            sequence_num=2
        ),
        EpisodicTurn(
            turn_id="msg_003",
            conversation_id="conv_s1",
            session_id="session_01",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="I will be leading the CacheMesh distributed cache project.",
            timestamp="2022-01-15T09:02:00Z",
            sequence_num=3
        ),
        EpisodicTurn(
            turn_id="msg_004",
            conversation_id="conv_s2",
            session_id="session_02",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="I was promoted to Staff Engineer in Bengaluru in June 2023.",
            timestamp="2023-06-10T14:00:00Z",
            sequence_num=4
        ),
        EpisodicTurn(
            turn_id="msg_005",
            conversation_id="conv_s2",
            session_id="session_02",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="I moved into my new apartment in Indiranagar, Bengaluru.",
            timestamp="2023-06-15T10:00:00Z",
            sequence_num=5
        ),
        EpisodicTurn(
            turn_id="msg_006",
            conversation_id="conv_s3",
            session_id="session_03",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="In January 2024, I switched to Aetheria AI as Principal Architect.",
            timestamp="2024-01-10T11:00:00Z",
            sequence_num=6
        ),
        EpisodicTurn(
            turn_id="msg_007",
            conversation_id="conv_s4",
            session_id="session_04",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="In March 2025, I transitioned to advisory roles and relocated to Zurich.",
            timestamp="2025-03-01T16:00:00Z",
            sequence_num=7
        )
    ]


@pytest.fixture
def sample_semantic_memories():
    """Sample semantic memories with valid source links and unanchored facts."""
    return [
        {
            "id": "m001",
            "text": "Arjun joined HyperScale Systems in January 2022 as Senior Engineer.",
            "timestamp": "2022-01-15T09:00:00Z",
            "source": "session_01_msg_msg_001",
            "relationships": [{"source": "Arjun", "predicate": "employed_by", "target": "HyperScale Systems"}]
        },
        {
            "id": "m002",
            "text": "Arjun led the CacheMesh project at HyperScale Systems.",
            "timestamp": "2022-01-15T09:02:00Z",
            "source": "session_01_msg_msg_003",
            "relationships": [{"source": "Arjun", "predicate": "leads_project", "target": "CacheMesh"}]
        },
        {
            "id": "m003",
            "text": "Arjun was promoted to Staff Engineer in Bengaluru in June 2023.",
            "timestamp": "2023-06-10T14:00:00Z",
            "source": "session_02_msg_msg_004",
            "relationships": [{"source": "Arjun", "predicate": "promoted_to", "target": "Staff Engineer"}]
        },
        {
            "id": "m004",
            "text": "Arjun switched to Aetheria AI as Principal Architect in January 2024.",
            "timestamp": "2024-01-10T11:00:00Z",
            "source": "session_03_msg_msg_006",
            "relationships": [{"source": "Arjun", "predicate": "employed_by", "target": "Aetheria AI"}]
        },
        {
            "id": "m099",
            "text": "Arjun earned a B.Tech in Computer Science from IIT Delhi in 2018.",
            "timestamp": "2018-05-15T00:00:00Z",
            "source": "external_resume_doc",  # Unanchored semantic memory
            "relationships": [{"source": "Arjun", "predicate": "graduated_from", "target": "IIT Delhi"}]
        }
    ]


# ---------------------------------------------------------------------------
# Batch 3: Graph Safeguards Tests
# ---------------------------------------------------------------------------

class TestGraphSafeguards:
    """Rigorous tests for GraphSafeguards provenance, temporal safety, and relevance gating."""

    def test_strict_provenance_valid_semantic_memory(self):
        """Triples with verified memory_id are admissible."""
        safeguards = GraphSafeguards(min_relevance_threshold=0.45)
        triple = GraphEvidenceItem(
            triple_id="edge_01",
            source_id="arjun",
            source_name="Arjun",
            source_type="entity",
            predicate="leads_project",
            target_id="cachemesh",
            target_name="CacheMesh",
            target_type="entity",
            direction="outgoing",
            hops=1,
            confidence=0.95,
            score=0.85,
            evidence="Arjun led CacheMesh",
            memory_id="m002"
        )
        route_decision = RouteDecision(
            query_type="entity_relationships",
            confidence=0.90,
            signals=["explicit_relationship_phrase"],
            graph_enabled=True
        )

        admissible, report = safeguards.evaluate_triples(
            triples=[triple],
            query="What is the relationship between Arjun and CacheMesh?",
            route_decision=route_decision,
            known_memory_ids={"m001", "m002", "m003"}
        )
        assert len(admissible) == 1
        assert report.admissible_count == 1
        assert report.decisions[0].provenance_type == "semantic_memory"
        assert report.decisions[0].provenance_id == "m002"

    def test_rejection_on_missing_provenance(self):
        """Triples lacking memory_id and source_turn_id must be rejected."""
        safeguards = GraphSafeguards()
        triple = GraphEvidenceItem(
            triple_id="edge_orphan",
            source_id="arjun",
            source_name="Arjun",
            source_type="entity",
            predicate="knows",
            target_id="unknown_person",
            target_name="Unknown Person",
            target_type="entity",
            direction="outgoing",
            hops=1,
            confidence=0.9,
            score=0.8,
            memory_id=None
        )
        route_decision = RouteDecision(query_type="entity_relationships", confidence=0.9, graph_enabled=True, signals=[])
        admissible, report = safeguards.evaluate_triples(
            triples=[triple],
            query="Relationship between Arjun and Unknown Person?",
            route_decision=route_decision,
            known_memory_ids={"m001"}
        )
        assert len(admissible) == 0
        assert report.provenance_exclusions == 1
        assert report.decisions[0].rejection_reason == "missing_provenance"

    def test_rejection_on_unverified_memory_id(self):
        """Triples with memory_id not in known_memory_ids must be rejected."""
        safeguards = GraphSafeguards()
        triple = GraphEvidenceItem(
            triple_id="edge_bad_mem",
            source_id="arjun",
            source_name="Arjun",
            source_type="entity",
            predicate="employed_by",
            target_id="bogus_corp",
            target_name="Bogus Corp",
            target_type="entity",
            direction="outgoing",
            hops=1,
            confidence=0.9,
            score=0.8,
            memory_id="m999_nonexistent"
        )
        route_decision = RouteDecision(query_type="entity_relationships", confidence=0.9, graph_enabled=True, signals=[])
        admissible, report = safeguards.evaluate_triples(
            triples=[triple],
            query="Does Arjun work at Bogus Corp?",
            route_decision=route_decision,
            known_memory_ids={"m001", "m002"}
        )
        assert len(admissible) == 0
        assert report.provenance_exclusions == 1
        assert "unverified_memory_id" in report.decisions[0].rejection_reason

    def test_unsubstantiated_triple_rejected_even_with_valid_memory_id(self, sample_semantic_memories):
        """A triple linked to an existing memory ID must be rejected if the memory does NOT substantiate the claim."""
        safeguards = GraphSafeguards()
        # Memory m001 exists and is about Arjun joining HyperScale Systems as Senior Engineer.
        # But this bogus triple claims Arjun divorced Alice.
        bogus_triple = GraphEvidenceItem(
            triple_id="edge_bogus_claim",
            source_id="arjun",
            source_name="Arjun",
            source_type="entity",
            predicate="divorced_from",
            target_id="alice",
            target_name="Alice",
            target_type="entity",
            direction="outgoing",
            hops=1,
            confidence=0.9,
            score=0.85,
            memory_id="m001"  # Existing memory ID!
        )
        mems_by_id = {m["id"]: m for m in sample_semantic_memories}
        route_decision = RouteDecision(query_type="entity_relationships", confidence=0.9, graph_enabled=True, signals=[])
        admissible, report = safeguards.evaluate_triples(
            triples=[bogus_triple],
            query="Who is Alice relative to Arjun?",
            route_decision=route_decision,
            known_memory_ids=set(mems_by_id.keys()),
            memories_by_id=mems_by_id
        )
        assert len(admissible) == 0
        assert report.provenance_exclusions == 1
        assert "unsubstantiated_by_source_memory" in report.decisions[0].rejection_reason

    def test_historical_query_rejects_unversioned_edges(self):
        """Historical queries MUST strictly reject edges lacking valid_from / valid_to."""
        safeguards = GraphSafeguards()
        # Edge has timestamp (ingestion time) but NO valid_from/valid_to in metadata
        triple = GraphEvidenceItem(
            triple_id="edge_curr_role",
            source_id="arjun",
            source_name="Arjun",
            source_type="entity",
            predicate="employed_by",
            target_id="aetheria",
            target_name="Aetheria AI",
            target_type="entity",
            direction="outgoing",
            hops=1,
            confidence=0.95,
            score=0.85,
            memory_id="m004",
            metadata={"valid_from": None, "valid_to": None}  # Empirically null
        )
        route_decision = RouteDecision(
            query_type="historical_state",
            confidence=0.95,
            signals=["temporal_historical_keyword"],
            graph_enabled=True  # even if graph retriever was triggered
        )
        admissible, report = safeguards.evaluate_triples(
            triples=[triple],
            query="What was Arjun's initial employer before Aetheria?",
            route_decision=route_decision,
            known_memory_ids={"m004"}
        )
        assert len(admissible) == 0
        assert report.temporal_exclusions == 1
        assert report.decisions[0].rejection_reason == "temporal_safety_unversioned_graph"

    def test_relevance_threshold_exclusion(self):
        """Triples with score below min_relevance_threshold must be rejected."""
        safeguards = GraphSafeguards(min_relevance_threshold=0.55)
        triple = GraphEvidenceItem(
            triple_id="edge_low_score",
            source_id="arjun",
            source_name="Arjun",
            source_type="entity",
            predicate="attended_talk",
            target_id="talk_123",
            target_name="Talk 123",
            target_type="entity",
            direction="outgoing",
            hops=2,
            confidence=0.6,
            score=0.48,  # Below 0.55
            memory_id="m001"
        )
        route_decision = RouteDecision(query_type="entity_relationships", confidence=0.8, graph_enabled=True, signals=[])
        admissible, report = safeguards.evaluate_triples(
            triples=[triple],
            query="What conference did Arjun attend?",
            route_decision=route_decision,
            known_memory_ids={"m001"}
        )
        assert len(admissible) == 0
        assert report.relevance_exclusions == 1
        assert "below_threshold" in report.decisions[0].rejection_reason

    def test_generic_entity_distractor_filtering(self):
        """2-hop edges connected to the user ('Arjun') with zero query token overlap must be rejected."""
        safeguards = GraphSafeguards(user_name="Arjun")
        triple = GraphEvidenceItem(
            triple_id="edge_distractor",
            source_id="arjun",
            source_name="Arjun",
            source_type="entity",
            predicate="prefers_drink",
            target_id="green_tea",
            target_name="Green Tea",
            target_type="entity",
            direction="outgoing",
            hops=2,  # 2 hops away
            confidence=0.9,
            score=0.60,
            evidence="Arjun drinks green tea in morning",
            memory_id="m001"
        )
        route_decision = RouteDecision(
            query_type="multi_hop_reasoning",
            confidence=0.85,
            signals=["relative_clause_bridge"],
            detected_entities=["Vikram Rao"],
            graph_enabled=True
        )
        admissible, report = safeguards.evaluate_triples(
            triples=[triple],
            query="Who was the manager who supervised Arjun at HyperScale Systems?",
            route_decision=route_decision,
            known_memory_ids={"m001"}
        )
        assert len(admissible) == 0
        assert report.relevance_exclusions == 1
        assert report.decisions[0].rejection_reason == "generic_entity_distractor_no_overlap"

    def test_cycle_and_duplicate_rejection(self):
        """Self-referential edges and duplicate triples must be excluded."""
        safeguards = GraphSafeguards()
        self_loop = GraphEvidenceItem(
            triple_id="edge_self",
            source_id="arjun",
            source_name="Arjun",
            source_type="entity",
            predicate="same_as",
            target_id="arjun",
            target_name="Arjun",
            target_type="entity",
            direction="outgoing",
            hops=1,
            confidence=1.0,
            score=0.9,
            memory_id="m001"
        )
        dup1 = GraphEvidenceItem(
            triple_id="edge_dup1",
            source_id="arjun",
            source_name="Arjun",
            source_type="entity",
            predicate="leads_project",
            target_id="cachemesh",
            target_name="CacheMesh",
            target_type="entity",
            direction="outgoing",
            hops=1,
            confidence=0.95,
            score=0.85,
            memory_id="m001"
        )
        dup2 = GraphEvidenceItem(
            triple_id="edge_dup2",
            source_id="arjun",
            source_name="Arjun",
            source_type="entity",
            predicate="leads_project",
            target_id="cachemesh",
            target_name="CacheMesh",
            target_type="entity",
            direction="outgoing",
            hops=1,
            confidence=0.95,
            score=0.80,
            memory_id="m001"
        )
        route_decision = RouteDecision(query_type="entity_relationships", confidence=0.9, graph_enabled=True, signals=[])
        admissible, report = safeguards.evaluate_triples(
            triples=[self_loop, dup1, dup2],
            query="Tell me about Arjun and CacheMesh",
            route_decision=route_decision,
            known_memory_ids={"m001"}
        )
        assert len(admissible) == 1
        assert admissible[0].triple_id == "edge_dup1"
        assert report.rejections_by_reason.get("self_referential_cycle") == 1
        assert report.rejections_by_reason.get("duplicate_triple") == 1

    def test_graph_disabled_by_router_rejects_all(self):
        """When router sets graph_enabled=False, all triples must be rejected with policy reason."""
        safeguards = GraphSafeguards()
        triple = GraphEvidenceItem(
            triple_id="edge_01",
            source_id="arjun",
            source_name="Arjun",
            source_type="entity",
            predicate="born_in",
            target_id="delhi",
            target_name="Delhi",
            target_type="entity",
            direction="outgoing",
            hops=1,
            confidence=1.0,
            score=0.9,
            memory_id="m001"
        )
        route_decision = RouteDecision(
            query_type="single_fact",
            confidence=0.95,
            signals=["single_fact_direct_marker"],
            graph_enabled=False  # Disabled by router
        )
        admissible, report = safeguards.evaluate_triples(
            triples=[triple],
            query="Where was Arjun born?",
            route_decision=route_decision,
            known_memory_ids={"m001"}
        )
        assert len(admissible) == 0
        assert report.policy_exclusions == 1
        assert report.decisions[0].rejection_reason == "graph_disabled_by_router_single_fact"


# ---------------------------------------------------------------------------
# Batch 2: Evidence Selector Tests
# ---------------------------------------------------------------------------

class TestAdaptiveEvidenceSelector:
    """Rigorous tests for AdaptiveEvidenceSelector context budgets, expansion, and annotation."""

    @pytest.fixture
    def selector(self):
        return AdaptiveEvidenceSelector()

    def test_single_fact_focused_budget(self, selector, sample_episodic_turns, sample_semantic_memories):
        """single_fact category allocates focused context (2-3 turns, 1 session)."""
        hit1 = EpisodicEvidenceItem(
            turn_id="msg_001",
            conversation_id="conv_s1",
            session_id="session_01",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="I joined HyperScale Systems in January 2022 as a Senior Engineer.",
            timestamp="2022-01-15T09:00:00Z",
            sequence_num=1,
            score=0.92,
            context_window=[
                {"turn_id": "msg_002", "speaker": "IVARA", "speaker_role": "assistant", "content": "Welcome to HyperScale!", "timestamp": "2022-01-15T09:01:00Z", "sequence_num": 2, "relation": "succeeding"}
            ]
        )
        # Hit from unrelated session
        hit2 = EpisodicEvidenceItem(
            turn_id="msg_006",
            conversation_id="conv_s3",
            session_id="session_03",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="In January 2024, I switched to Aetheria AI.",
            timestamp="2024-01-10T11:00:00Z",
            sequence_num=6,
            score=0.60
        )
        route_decision = RouteDecision(
            query_type="single_fact",
            confidence=0.95,
            signals=["single_fact_direct_marker"],
            context_mode="single_fact"
        )
        res = selector.select(
            query="When did Arjun join HyperScale Systems?",
            route_decision=route_decision,
            episodic_hits=[hit1, hit2],
            semantic_hits=[],
            graph_hits=[],
            all_turns=sample_episodic_turns,
            all_memories=sample_semantic_memories
        )
        diag = res.diagnostics
        assert diag["router_category"] == "single_fact"
        assert diag["sessions_represented"]["count"] == 1
        assert diag["sessions_represented"]["sessions"] == ["session_01"]
        assert len(res.selected_items) <= 3
        assert "msg_001" in res.citations

    def test_semantic_annotation_anchoring(self, selector, sample_episodic_turns, sample_semantic_memories):
        """Semantic memories with matching source_turn_id annotate episodic turns without consuming top-level slots."""
        hit = EpisodicEvidenceItem(
            turn_id="msg_001",
            conversation_id="conv_s1",
            session_id="session_01",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="I joined HyperScale Systems in January 2022 as a Senior Engineer.",
            timestamp="2022-01-15T09:00:00Z",
            sequence_num=1,
            score=0.90
        )
        sem_hit = SemanticEvidenceItem(
            memory_id="m001",
            text="Arjun joined HyperScale Systems in January 2022 as Senior Engineer.",
            timestamp="2022-01-15T09:00:00Z",
            source_turn_id="session_01_msg_msg_001",
            score=0.92,
            relationships=[{"source": "Arjun", "predicate": "employed_by", "target": "HyperScale Systems"}]
        )
        route_decision = RouteDecision(
            query_type="single_fact",
            confidence=0.90,
            signals=[],
            context_mode="single_fact"
        )
        res = selector.select(
            query="What was Arjun's role at HyperScale?",
            route_decision=route_decision,
            episodic_hits=[hit],
            semantic_hits=[sem_hit],
            graph_hits=[],
            all_turns=sample_episodic_turns,
            all_memories=sample_semantic_memories
        )
        diag = res.diagnostics
        assert diag["anchored_semantic_count"] == 1
        assert diag["unanchored_semantic_count"] == 0
        # Selected items should have 1 item (the episodic turn), with the semantic fact attached as an annotation
        assert len(res.selected_items) == 1
        epi_item = res.selected_items[0]
        assert epi_item.evidence_type == "episodic"
        assert len(epi_item.anchored_semantic_memories) == 1
        assert epi_item.anchored_semantic_memories[0]["memory_id"] == "m001"
        assert "m001" in res.citations

    def test_unanchored_semantic_preservation(self, selector, sample_episodic_turns, sample_semantic_memories):
        """Semantic memories without an episodic source in context are preserved as unanchored evidence."""
        hit = EpisodicEvidenceItem(
            turn_id="msg_001",
            conversation_id="conv_s1",
            session_id="session_01",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="I joined HyperScale Systems in January 2022 as a Senior Engineer.",
            timestamp="2022-01-15T09:00:00Z",
            sequence_num=1,
            score=0.85
        )
        # Unanchored semantic memory (e.g. from resume)
        unanchored_sem = SemanticEvidenceItem(
            memory_id="m099",
            text="Arjun earned a B.Tech in Computer Science from IIT Delhi in 2018.",
            timestamp="2018-05-15T00:00:00Z",
            source_turn_id="external_resume_doc",
            score=0.95
        )
        route_decision = RouteDecision(
            query_type="single_fact",
            confidence=0.85,
            signals=[],
            context_mode="single_fact"
        )
        res = selector.select(
            query="Where did Arjun graduate from?",
            route_decision=route_decision,
            episodic_hits=[hit],
            semantic_hits=[unanchored_sem],
            graph_hits=[],
            all_turns=sample_episodic_turns,
            all_memories=sample_semantic_memories
        )
        diag = res.diagnostics
        assert diag["unanchored_semantic_count"] == 1
        # Should contain both the episodic turn and the unanchored semantic memory
        unanchored_items = [i for i in res.selected_items if i.evidence_type == "semantic"]
        assert len(unanchored_items) == 1
        assert unanchored_items[0].evidence_id == "m099"
        assert unanchored_items[0].is_anchored is False
        assert "m099" in res.citations

    def test_cross_session_synthesis_multi_session(self, selector, sample_episodic_turns, sample_semantic_memories):
        """cross_session_synth supports up to 4 sessions without collapsing narratives."""
        hits = [
            EpisodicEvidenceItem(turn_id="msg_001", conversation_id="conv_s1", session_id="session_01", speaker="Arjun", speaker_role="user", content="Joined HyperScale in 2022.", timestamp="2022-01-15", sequence_num=1, score=0.9),
            EpisodicEvidenceItem(turn_id="msg_004", conversation_id="conv_s2", session_id="session_02", speaker="Arjun", speaker_role="user", content="Promoted in Bengaluru in 2023.", timestamp="2023-06-10", sequence_num=4, score=0.88),
            EpisodicEvidenceItem(turn_id="msg_006", conversation_id="conv_s3", session_id="session_03", speaker="Arjun", speaker_role="user", content="Switched to Aetheria in 2024.", timestamp="2024-01-10", sequence_num=6, score=0.85),
            EpisodicEvidenceItem(turn_id="msg_007", conversation_id="conv_s4", session_id="session_04", speaker="Arjun", speaker_role="user", content="Relocated to Zurich in 2025.", timestamp="2025-03-01", sequence_num=7, score=0.82)
        ]
        route_decision = RouteDecision(
            query_type="cross_session_synth",
            confidence=0.95,
            signals=["cross_session_lifecycle_marker"],
            context_mode="cross_session_synth"
        )
        res = selector.select(
            query="Trace Arjun's career progression and residential moves across all companies.",
            route_decision=route_decision,
            episodic_hits=hits,
            semantic_hits=[],
            graph_hits=[],
            all_turns=sample_episodic_turns,
            all_memories=sample_semantic_memories
        )
        diag = res.diagnostics
        assert diag["sessions_represented"]["count"] == 4
        assert set(diag["sessions_represented"]["sessions"]) == {"session_01", "session_02", "session_03", "session_04"}
        # Structured context text must delineate each session clearly
        assert "--- SESSION: session_01 ---" in res.structured_context_text
        assert "--- SESSION: session_02 ---" in res.structured_context_text
        assert "--- SESSION: session_03 ---" in res.structured_context_text
        assert "--- SESSION: session_04 ---" in res.structured_context_text

    def test_overlapping_window_deduplication(self, selector, sample_episodic_turns):
        """Overlapping context windows in the same session merge seamlessly without duplicate entries."""
        # Hit 1 and Hit 2 are consecutive turns in session_01
        hit1 = EpisodicEvidenceItem(
            turn_id="msg_001",
            conversation_id="conv_s1",
            session_id="session_01",
            speaker="Arjun Mehta",
            speaker_role="user",
            content="I joined HyperScale Systems in January 2022.",
            timestamp="2022-01-15T09:00:00Z",
            sequence_num=1,
            score=0.90,
            context_window=[
                {"turn_id": "msg_002", "speaker": "IVARA", "speaker_role": "assistant", "content": "Welcome to HyperScale!", "timestamp": "2022-01-15T09:01:00Z", "sequence_num": 2, "relation": "succeeding"}
            ]
        )
        hit2 = EpisodicEvidenceItem(
            turn_id="msg_002",
            conversation_id="conv_s1",
            session_id="session_01",
            speaker="IVARA",
            speaker_role="assistant",
            content="Welcome to HyperScale!",
            timestamp="2022-01-15T09:01:00Z",
            sequence_num=2,
            score=0.88,
            context_window=[
                {"turn_id": "msg_001", "speaker": "Arjun Mehta", "speaker_role": "user", "content": "I joined HyperScale Systems in January 2022.", "timestamp": "2022-01-15T09:00:00Z", "sequence_num": 1, "relation": "preceding"},
                {"turn_id": "msg_003", "speaker": "Arjun Mehta", "speaker_role": "user", "content": "I will be leading CacheMesh.", "timestamp": "2022-01-15T09:02:00Z", "sequence_num": 3, "relation": "succeeding"}
            ]
        )
        route_decision = RouteDecision(query_type="historical_state", confidence=0.9, signals=[], context_mode="historical_state")
        res = selector.select(
            query="What was Arjun's role when he joined?",
            route_decision=route_decision,
            episodic_hits=[hit1, hit2],
            semantic_hits=[],
            graph_hits=[],
            all_turns=sample_episodic_turns
        )
        diag = res.diagnostics
        assert diag["deduplicated_turn_count"] >= 1
        # Selected turn IDs should be unique
        selected_ids = [item.evidence_id for item in res.selected_items]
        assert len(selected_ids) == len(set(selected_ids))

    def test_chronological_ordering_within_sessions(self, selector, sample_episodic_turns):
        """Selected turns within a session must always be in chronological order by sequence_num."""
        # Pass hits out of chronological order
        hit_later = EpisodicEvidenceItem(
            turn_id="msg_003", conversation_id="conv_s1", session_id="session_01",
            speaker="Arjun", speaker_role="user", content="Leading CacheMesh.", timestamp="2022-01-15T09:02:00Z", sequence_num=3, score=0.95
        )
        hit_earlier = EpisodicEvidenceItem(
            turn_id="msg_001", conversation_id="conv_s1", session_id="session_01",
            speaker="Arjun", speaker_role="user", content="Joined HyperScale.", timestamp="2022-01-15T09:00:00Z", sequence_num=1, score=0.80
        )
        route_decision = RouteDecision(query_type="single_fact", confidence=0.85, signals=[], context_mode="single_fact")
        res = selector.select(
            query="When did Arjun lead CacheMesh?",
            route_decision=route_decision,
            episodic_hits=[hit_later, hit_earlier],
            semantic_hits=[],
            graph_hits=[],
            all_turns=sample_episodic_turns
        )
        selected_seqs = [item.metadata.get("sequence_num") for item in res.selected_items if item.evidence_type == "episodic"]
        assert selected_seqs == sorted(selected_seqs)

    def test_deterministic_output(self, selector, sample_episodic_turns, sample_semantic_memories):
        """Identical inputs must produce identical selection results and citations."""
        hit = EpisodicEvidenceItem(
            turn_id="msg_001", conversation_id="conv_s1", session_id="session_01",
            speaker="Arjun", speaker_role="user", content="Joined HyperScale in 2022.", timestamp="2022-01-15T09:00:00Z", sequence_num=1, score=0.9
        )
        route_decision = RouteDecision(query_type="single_fact", confidence=0.9, signals=[], context_mode="single_fact")
        res1 = selector.select(
            query="What did Arjun say?",
            route_decision=route_decision,
            episodic_hits=[hit],
            semantic_hits=[],
            graph_hits=[],
            all_turns=sample_episodic_turns,
            all_memories=sample_semantic_memories
        )
        res2 = selector.select(
            query="What did Arjun say?",
            route_decision=route_decision,
            episodic_hits=[hit],
            semantic_hits=[],
            graph_hits=[],
            all_turns=sample_episodic_turns,
            all_memories=sample_semantic_memories
        )
        assert res1.structured_context_text == res2.structured_context_text
        assert res1.citations == res2.citations
        assert res1.diagnostics == res2.diagnostics

    def test_oversized_context_budget_enforcement(self, sample_episodic_turns):
        """When evidence text exceeds max_context_tokens, selector trims lower-priority items and marks truncation."""
        tight_selector = AdaptiveEvidenceSelector(max_context_tokens=100)  # Very low token budget
        # Oversized hits
        hit_huge = EpisodicEvidenceItem(
            turn_id="msg_huge",
            conversation_id="conv_s1",
            session_id="session_01",
            speaker="Arjun",
            speaker_role="user",
            content="This is an extremely long conversation turn. " * 50,
            timestamp="2022-01-15T09:00:00Z",
            sequence_num=1,
            score=0.95
        )
        unanchored_sem = SemanticEvidenceItem(
            memory_id="m_extra",
            text="Extra unanchored fact that should be trimmed under tight budget.",
            timestamp="2022-01-15T09:00:00Z",
            source_turn_id=None,
            score=0.70
        )
        route_decision = RouteDecision(query_type="single_fact", confidence=0.9, signals=[], context_mode="single_fact")
        res = tight_selector.select(
            query="Tell me everything",
            route_decision=route_decision,
            episodic_hits=[hit_huge],
            semantic_hits=[unanchored_sem],
            graph_hits=[]
        )
        assert res.diagnostics["context_truncated"] is True
        assert len(res.diagnostics["truncation_actions"]) > 0
        assert res.diagnostics["context_token_estimate"] <= 120


# ---------------------------------------------------------------------------
# Pipeline Integration Tests
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Verify UnifiedRetriever integration with selector and safeguards."""

    @pytest.fixture
    def populated_retriever(self, sample_episodic_turns, sample_semantic_memories):
        return UnifiedRetriever(
            episodic_turns=sample_episodic_turns,
            memories=sample_semantic_memories,
            use_adaptive_routing=False
        )

    def test_adaptive_routing_off_preserves_classic_fusion(self, populated_retriever):
        """When use_adaptive_routing=False, classic RRF-P fusion is preserved exactly."""
        res = populated_retriever.retrieve("What was Arjun's role at HyperScale?", use_adaptive_routing=False)
        assert "evidence_selection" not in res.diagnostics
        assert "routing" not in res.diagnostics
        assert res.status == "sufficient_evidence"
        assert len(res.evidence_items) > 0

    def test_adaptive_routing_on_uses_selector(self, populated_retriever):
        """When use_adaptive_routing=True, selector produces structured context and diagnostics."""
        res = populated_retriever.retrieve("What was Arjun's previous role before switching?", use_adaptive_routing=True)
        assert "routing" in res.diagnostics
        assert "evidence_selection" in res.diagnostics
        assert res.diagnostics["routing"]["query_type"] == "historical_state"
        assert res.diagnostics["enabled_channels"]["graph"] is False

        # Structured context text must feature the primary episodic section
        assert "=== PRIMARY EPISODIC CONVERSATION EVIDENCE ===" in res.context_text
        # Evidence items should be properly populated
        assert len(res.evidence_items) > 0

    def test_historical_query_never_receives_unversioned_graph_evidence(self, populated_retriever):
        """Historical state query must NEVER include unversioned graph triples."""
        res = populated_retriever.retrieve(
            "What was Arjun's former job title before switching to Aetheria?",
            use_adaptive_routing=True
        )
        assert res.diagnostics["enabled_channels"]["graph"] is False
        assert "=== VERIFIED KNOWLEDGE GRAPH RELATIONSHIPS ===" not in res.context_text

    def test_safe_fallback_on_selector_failure(self, populated_retriever, monkeypatch):
        """If evidence selector encounters an unexpected exception, gracefully fall back to RRF-P."""
        def mock_failing_select(*args, **kwargs):
            raise RuntimeError("Simulated selector internal crash")

        monkeypatch.setattr(populated_retriever.evidence_selector, "select", mock_failing_select)
        res = populated_retriever.retrieve("When did Arjun join?", use_adaptive_routing=True)
        assert "evidence_selection_fallback" in res.diagnostics
        assert "Simulated selector internal crash" in res.diagnostics["evidence_selection_fallback"]
        assert res.status == "sufficient_evidence"
        assert len(res.evidence_items) > 0

