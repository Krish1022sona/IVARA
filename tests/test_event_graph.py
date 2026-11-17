"""
Unit and Integration Tests for IVARA Phase 2B.1 Temporal Event Graph.

Tests:
1. Event extraction (exact dates, months, years, relative expressions, uncertainty preservation).
2. Decoupling of event time from conversation recording/ingestion time.
3. Temporal relationship construction (OCCURRED_BEFORE, OCCURRED_AFTER, SUPERSEDES, INVALIDATES).
4. Strict safety check: No temporal ordering inferred solely from ingestion/recording order.
5. Bounded graph traversal and cross-session relationship reasoning.
6. Provenance validity (verbatim quotes, source chunk/session traceability).
7. Clean fallback when graph confidence is insufficient.
"""

from datetime import date
import pytest

from src.temporal.event_graph import (
    EventNode,
    TemporalRelationshipEdge,
    TemporalPrecision,
    TemporalRelationshipType,
    generate_event_id,
    generate_temporal_edge_id
)
from src.temporal.event_extractor import TemporalEventExtractor
from src.temporal.relationship_builder import TemporalRelationshipBuilder
from src.temporal.event_graph_traversal import TemporalEventGraphRetriever
from src.graph.store import SQLiteGraphStore
from src.chunking.turn_chunker import TurnChunk
from src.chunking.graph_retrieval import TurnGraphBuilder


class TestEventExtraction:
    """Test EventNode extraction, decoupling, and uncertainty preservation."""

    def setup_method(self):
        self.extractor = TemporalEventExtractor()

    def test_exact_date_extraction(self):
        text = "On April 6, 2023, I attended the PyCon conference in Salt Lake City."
        events = self.extractor.extract_events_from_chunk(
            text=text,
            session_id="sess_001",
            chunk_id="chunk_001",
            recorded_at="2023-05-01"
        )
        assert len(events) >= 1
        ev = events[0]
        assert ev.precision == TemporalPrecision.EXACT_DAY
        assert ev.valid_from == "2023-04-06"
        assert ev.valid_to == "2023-04-06"
        assert ev.recorded_at == "2023-05-01"
        # Event time strictly decoupled from recorded time
        assert ev.valid_from != ev.recorded_at
        assert "PyCon" in ev.description or "conference" in ev.description
        assert "April 6, 2023" in ev.provenance

    def test_month_precision_preservation(self):
        text = "I visited Paris in July 2022 and had a wonderful time."
        events = self.extractor.extract_events_from_chunk(
            text=text,
            session_id="sess_002",
            chunk_id="chunk_002",
            recorded_at="2023-01-10"
        )
        assert len(events) >= 1
        ev = events[0]
        assert ev.precision == TemporalPrecision.MONTH
        assert ev.valid_from == "2022-07-01"
        assert ev.valid_to == "2022-07-31"
        assert "Paris" in ev.description

    def test_year_precision_preservation(self):
        text = "I graduated from MIT in 2019."
        events = self.extractor.extract_events_from_chunk(
            text=text,
            session_id="sess_003",
            chunk_id="chunk_003",
            recorded_at="2023-01-10"
        )
        assert len(events) >= 1
        ev = events[0]
        assert ev.precision == TemporalPrecision.YEAR
        assert ev.valid_from == "2019-01-01"
        assert ev.valid_to == "2019-12-31"

    def test_relative_date_anchoring(self):
        text = "Yesterday I cancelled my FarmFresh subscription."
        events = self.extractor.extract_events_from_chunk(
            text=text,
            session_id="sess_004",
            chunk_id="chunk_004",
            recorded_at="2023-08-15"
        )
        assert len(events) >= 1
        ev = events[0]
        assert ev.precision == TemporalPrecision.EXACT_DAY
        assert ev.valid_from == "2023-08-14"
        assert ev.valid_to == "2023-08-14"
        assert ev.recorded_at == "2023-08-15"
        assert "Cancelled" in ev.description or "FarmFresh" in ev.description

    def test_unknown_date_preserves_uncertainty(self):
        text = "Sometime in the past I joined the chess club."
        events = self.extractor.extract_events_from_chunk(
            text=text,
            session_id="sess_005",
            chunk_id="chunk_005",
            recorded_at=None
        )
        if events:
            ev = events[0]
            assert ev.precision == TemporalPrecision.UNKNOWN
            assert ev.valid_from is None
            assert ev.valid_to is None


class TestTemporalRelationships:
    """Test minimal relationship vocabulary and non-ingestion ordering safety."""

    def setup_method(self):
        self.builder = TemporalRelationshipBuilder()

    def test_occurred_before_and_after_grounded(self):
        e1 = EventNode(
            event_id="evt_01",
            description="Attended TechCon 2022",
            session_id="sess_1",
            turn_id="c1",
            valid_from="2022-05-10",
            valid_to="2022-05-12",
            recorded_at="2022-06-01",
            precision=TemporalPrecision.EXACT_DAY,
            confidence=1.0,
            provenance="Attended TechCon 2022 on May 10-12, 2022"
        )
        e2 = EventNode(
            event_id="evt_02",
            description="Visited Tokyo",
            session_id="sess_2",
            turn_id="c2",
            valid_from="2022-10-01",
            valid_to="2022-10-10",
            recorded_at="2022-11-01",
            precision=TemporalPrecision.EXACT_DAY,
            confidence=1.0,
            provenance="Visited Tokyo October 1-10, 2022"
        )

        edges = self.builder.build_relationships([e1, e2])
        rel_types = {e.relationship for e in edges}
        assert TemporalRelationshipType.OCCURRED_BEFORE in rel_types
        assert TemporalRelationshipType.OCCURRED_AFTER in rel_types

        # Verify direction
        before_edge = next(e for e in edges if e.relationship == TemporalRelationshipType.OCCURRED_BEFORE)
        assert before_edge.source_id == "evt_01"
        assert before_edge.target_id == "evt_02"
        assert before_edge.derived_by == "grounded_interval_comparison"
        assert before_edge.provenance_chunk_ids == ["c1", "c2"]

    def test_strict_safety_no_ingestion_order_inference(self):
        """
        CRITICAL SAFETY TEST:
        If e1 is ingested in session 1 (earlier) and e2 in session 2 (later),
        but neither has dates, NO OCCURRED_BEFORE edge should be created.
        """
        e1 = EventNode(
            event_id="evt_undated_1",
            description="Read book A",
            session_id="sess_1",
            recorded_at="2023-01-01",
            valid_from=None,
            valid_to=None,
            precision=TemporalPrecision.UNKNOWN
        )
        e2 = EventNode(
            event_id="evt_undated_2",
            description="Read book B",
            session_id="sess_2",
            recorded_at="2023-06-01",
            valid_from=None,
            valid_to=None,
            precision=TemporalPrecision.UNKNOWN
        )

        edges = self.builder.build_relationships([e1, e2])
        # Must be empty! Never infer ordering from ingestion time!
        assert len(edges) == 0

    def test_event_order_opposite_to_ingestion_order(self):
        """
        If e2 is ingested in Session 2 (2023-06-01) but describes an event in 2021,
        and e1 is ingested in Session 1 (2023-01-01) describing an event in 2022:
        e2 OCCURRED_BEFORE e1, proving event time overrides ingestion order!
        """
        e1 = EventNode(
            event_id="evt_2022",
            description="Bought bicycle",
            session_id="sess_1",
            recorded_at="2023-01-01",
            valid_from="2022-03-15",
            valid_to="2022-03-15",
            precision=TemporalPrecision.EXACT_DAY
        )
        e2 = EventNode(
            event_id="evt_2021",
            description="Bought skateboard",
            session_id="sess_2",
            recorded_at="2023-06-01",
            valid_from="2021-05-10",
            valid_to="2021-05-10",
            precision=TemporalPrecision.EXACT_DAY
        )

        edges = self.builder.build_relationships([e1, e2])
        before_edge = next(e for e in edges if e.relationship == TemporalRelationshipType.OCCURRED_BEFORE)
        # e2 (2021) occurred before e1 (2022)
        assert before_edge.source_id == "evt_2021"
        assert before_edge.target_id == "evt_2022"

    def test_state_supersedes_relationship(self):
        e1 = EventNode(
            event_id="evt_res_1",
            description="Lived in Boston",
            session_id="sess_1",
            valid_from="2020-01-01",
            valid_to="2022-05-01",
            precision=TemporalPrecision.EXACT_DAY,
            properties={"action_type": "lived in", "target": "Boston"}
        )
        e2 = EventNode(
            event_id="evt_res_2",
            description="Lived in Seattle",
            session_id="sess_2",
            valid_from="2022-06-01",
            valid_to="2024-01-01",
            precision=TemporalPrecision.EXACT_DAY,
            properties={"action_type": "lived in", "target": "Seattle"}
        )

        edges = self.builder.build_relationships([e1, e2])
        sup_edges = [e for e in edges if e.relationship == TemporalRelationshipType.SUPERSEDES]
        assert len(sup_edges) == 1
        assert sup_edges[0].source_id == "evt_res_2"  # New state supersedes old
        assert sup_edges[0].target_id == "evt_res_1"

    def test_invalidation_relationship(self):
        e_sub = EventNode(
            event_id="evt_sub",
            description="Subscribed to FarmFresh",
            session_id="sess_1",
            valid_from="2023-01-01",
            valid_to="2023-01-01",
            precision=TemporalPrecision.EXACT_DAY,
            properties={"action_type": "subscribed to", "target": "FarmFresh"}
        )
        e_cancel = EventNode(
            event_id="evt_cancel",
            description="Cancelled FarmFresh",
            session_id="sess_2",
            valid_from="2023-08-14",
            valid_to="2023-08-14",
            precision=TemporalPrecision.EXACT_DAY,
            properties={"action_type": "cancelled", "target": "FarmFresh"}
        )

        edges = self.builder.build_relationships([e_sub, e_cancel])
        inv_edges = [e for e in edges if e.relationship == TemporalRelationshipType.INVALIDATES]
        assert len(inv_edges) == 1
        assert inv_edges[0].source_id == "evt_cancel"
        assert inv_edges[0].target_id == "evt_sub"


class TestBoundedGraphTraversalAndRetrieval:
    """Test bounded graph traversal, query answering, and fallback."""

    def test_ordering_traversal_between_two_events(self):
        e1 = EventNode(
            event_id="evt_berlin",
            description="Visited Berlin",
            entity_ids=["ent_berlin"],
            session_id="sess_1",
            turn_id="c1",
            valid_from="2023-04-10",
            valid_to="2023-04-12",
            precision=TemporalPrecision.EXACT_DAY,
            provenance="I visited Berlin from April 10 to April 12, 2023."
        )
        e2 = EventNode(
            event_id="evt_madrid",
            description="Visited Madrid",
            entity_ids=["ent_madrid"],
            session_id="sess_2",
            turn_id="c2",
            valid_from="2023-07-15",
            valid_to="2023-07-20",
            precision=TemporalPrecision.EXACT_DAY,
            provenance="I flew to Madrid on July 15, 2023."
        )

        builder = TemporalRelationshipBuilder()
        edges = builder.build_relationships([e1, e2])

        retriever = TemporalEventGraphRetriever([e1, e2], edges)
        res = retriever.query_graph("Which event happened first: Berlin or Madrid?")

        assert res["confidence"] >= 0.8
        assert "OCCURRED_BEFORE" in res["evidence_text"]
        assert "Berlin" in res["evidence_text"]
        assert "Madrid" in res["evidence_text"]
        assert "c1" in res["supporting_chunk_ids"]
        assert "c2" in res["supporting_chunk_ids"]

    def test_supersession_traversal(self):
        e1 = EventNode(
            event_id="evt_job_1",
            description="Worked at Acme Corp",
            session_id="sess_1",
            turn_id="c1",
            valid_from="2020-01-01",
            valid_to="2022-08-31",
            precision=TemporalPrecision.EXACT_DAY,
            properties={"action_type": "worked at", "target": "Acme Corp"}
        )
        e2 = EventNode(
            event_id="evt_job_2",
            description="Worked at BetaTech",
            session_id="sess_2",
            turn_id="c2",
            valid_from="2022-09-01",
            valid_to="2024-01-01",
            precision=TemporalPrecision.EXACT_DAY,
            properties={"action_type": "worked at", "target": "BetaTech"}
        )

        builder = TemporalRelationshipBuilder()
        edges = builder.build_relationships([e1, e2])

        retriever = TemporalEventGraphRetriever([e1, e2], edges)
        res = retriever.query_graph("Who did I work for before BetaTech?")

        assert res["confidence"] >= 0.8
        assert "superseded" in res["evidence_text"].lower() or "worked at acme corp" in res["evidence_text"].lower()

    def test_fallback_on_insufficient_graph_confidence(self):
        retriever = TemporalEventGraphRetriever([], [])
        res = retriever.query_graph("What is quantum computing?")
        assert res["confidence"] == 0.0
        assert res["evidence_text"] == ""
        assert res["supporting_chunk_ids"] == []


class TestStoreIntegration:
    """Test full integration with SQLiteGraphStore and TurnChunk instances."""

    def test_store_population_and_retrieval(self):
        store = SQLiteGraphStore(":memory:")
        chunks = [
            TurnChunk(
                chunk_id="chk_01",
                session_id="sess_10",
                memory_id="m10",
                turn_start_idx=0,
                turn_end_idx=1,
                total_session_turns=2,
                timestamp="2023-05-18",
                parsed_datetime=None,
                turns=[{"role": "user", "content": "On May 18, 2023, I attended the marathon in Chicago."}],
                text="On May 18, 2023, I attended the marathon in Chicago.",
                word_count=10,
                token_estimate=15
            ),
            TurnChunk(
                chunk_id="chk_02",
                session_id="sess_11",
                memory_id="m11",
                turn_start_idx=0,
                turn_end_idx=1,
                total_session_turns=2,
                timestamp="2023-05-24",
                parsed_datetime=None,
                turns=[{"role": "user", "content": "On May 24, 2023, I attended the AI Summit in New York."}],
                text="On May 24, 2023, I attended the AI Summit in New York.",
                word_count=12,
                token_estimate=15
            )
        ]

        graph_builder = TurnGraphBuilder(store)
        stats = graph_builder.build_from_chunks(chunks)

        # Verify event nodes were created in SQLiteGraphStore
        event_nodes = store.get_nodes_by_type("event")
        assert len(event_nodes) >= 2

        # Verify temporal edges were created
        before_edges = store.get_edges_by_type("OCCURRED_BEFORE")
        assert len(before_edges) >= 1

        store.close()
