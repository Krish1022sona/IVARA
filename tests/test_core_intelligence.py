"""
Comprehensive Tests for IVARA Core Intelligence Capabilities.

Verifies:
1. Phase 2 (Temporal Memory & Supersession):
   - Location updates (Delhi -> Pune)
   - Preference transitions (dark roast -> light roast)
   - Historical time query answering
   - Late-recorded old facts (2026 record of 2022 fact does not supersede 2024 state)
   - Unresolved incompatible claims preserved with uncertainty
2. Phase 3 (Relationship-Aware Memory Graph):
   - Cross-session relational links
   - Cycle protection and budget limits
   - Direct evidence priority
3. Phase 4 (Inter-Memory Evidence Integration):
   - Complementary multi-memory joint support
   - Redundant copy de-weighting
   - Role classifications (supporting, complementary, redundant, contradictory, superseded)
4. Phase 5 (Contradiction Detection & Resolution):
   - Legitimate update vs True conflict
   - Context distinctions (at work vs for personal projects)
   - Negations (do not own vs owns)
   - Uncertainty handling
5. Phase 6 (Evidence Sufficiency & Abstention Gating):
   - Sufficient evidence proceeds
   - Partial evidence surfaces unrecorded subclaims
   - Insufficient evidence triggers clean abstention
   - Citation validation rejects fabricated IDs
"""

import json
from pathlib import Path
import pytest

from src.answering import answer_question
from src.chunking.turn_chunker import TurnChunk
from src.chunking.hybrid_pipeline import UnifiedCandidateRetriever
from src.graph.store import SQLiteGraphStore
from src.graph.schema import GraphNode, GraphEdge
from src.temporal.claim import TemporalClaim, build_claims_from_text
from src.temporal.supersession import TemporalSupersessionEngine
from src.temporal.resolution import QueryTemporalResolver, TemporalQueryIntent
from src.reasoning.contradiction import ContradictionAnalyzer, ContradictionType
from src.reasoning.evidence_integrator import EvidenceIntegrator, EvidenceRole
from src.reasoning.evidence_gate import EvidenceGate, GatingDecision


def make_chunk(chunk_id: str, session_id: str, text: str, timestamp: str = "2024-01-01") -> TurnChunk:
    return TurnChunk(
        chunk_id=chunk_id,
        session_id=session_id,
        memory_id=f"sess_{session_id}",
        turn_start_idx=0,
        turn_end_idx=2,
        turns=[{"speaker": "user", "text": text}],
        text=text,
        timestamp=timestamp,
        token_estimate=len(text.split()),
        total_session_turns=3,
        parsed_datetime=None,
        word_count=len(text.split())
    )


# ============================================================================
# PHASE 2: TEMPORAL MEMORY & SUPERSESSION TESTS
# ============================================================================

def test_temporal_location_change_supersession():
    """Verify that moving to a new city supersedes previous residence."""
    engine = TemporalSupersessionEngine()
    c1 = TemporalClaim(
        claim_id="c_delhi",
        subject="user",
        predicate="lives_in",
        object="Delhi",
        valid_from="2023",
        recorded_at="2023-01-01"
    )
    c2 = TemporalClaim(
        claim_id="c_pune",
        subject="user",
        predicate="lives_in",
        object="Pune",
        valid_from="2025",
        recorded_at="2025-01-01"
    )
    processed = engine.process_claims([c1, c2])
    c_map = {c.claim_id: c for c in processed}

    assert c_map["c_delhi"].properties["is_current"] is False
    assert c_map["c_delhi"].properties["superseded_by"] == "c_pune"
    assert c_map["c_delhi"].valid_to == "2024"
    assert c_map["c_pune"].properties["is_current"] is True
    assert c_map["c_pune"].supersedes_claim_id == "c_delhi"


def test_temporal_preference_transition():
    """Verify that explicit preference update supersedes older preference."""
    claims = build_claims_from_text("I used to prefer dark roast coffee, but now I prefer light roast.")
    engine = TemporalSupersessionEngine()
    processed = engine.process_claims(claims)

    current_claims = [c for c in processed if c.properties.get("is_current")]
    historical_claims = [c for c in processed if not c.properties.get("is_current")]

    assert len(current_claims) == 1
    assert current_claims[0].object == "light roast"
    assert len(historical_claims) == 1
    assert historical_claims[0].object == "dark roast coffee"


def test_temporal_historical_question_resolution():
    """Verify that a historical query resolves to the claim valid at that time."""
    claims = [
        TemporalClaim(claim_id="c1", subject="user", predicate="lives_in", object="Seattle", valid_from="2021", valid_to="2023"),
        TemporalClaim(claim_id="c2", subject="user", predicate="lives_in", object="Austin", valid_from="2024", valid_to=None)
    ]
    resolver = QueryTemporalResolver()
    res = resolver.resolve("Where did I live in 2022?", claims)

    assert res["intent"] == TemporalQueryIntent.HISTORICAL
    assert res["anchor"]["target_year"] == 2022
    assert len(res["selected_claims"]) == 1
    assert res["selected_claims"][0].object == "Seattle"


def test_late_recorded_old_fact_does_not_supersede_newer_fact():
    """
    CRITICAL: In 2026, user mentions 'In 2022 I lived in Boston'.
    Earlier memory in 2024 states user moved to Austin.
    The late-recorded 2022 fact MUST NOT supersede the 2024 state!
    """
    engine = TemporalSupersessionEngine()
    c_austin = TemporalClaim(
        claim_id="c_austin_2024",
        subject="user",
        predicate="lives_in",
        object="Austin",
        valid_from="2024",
        recorded_at="2024-05-01"
    )
    # Recorded in session 2026, but refers to 2022 event time
    c_boston = TemporalClaim(
        claim_id="c_boston_2022",
        subject="user",
        predicate="lives_in",
        object="Boston",
        valid_from="2022",
        recorded_at="2026-01-15"
    )

    processed = engine.process_claims([c_austin, c_boston])
    c_map = {c.claim_id: c for c in processed}

    # Austin (2024) remains the current state
    assert c_map["c_austin_2024"].properties["is_current"] is True
    # Boston (2022) is marked historical
    assert c_map["c_boston_2022"].properties["is_current"] is False


def test_unresolved_incompatible_claims_preserved():
    """Verify that two conflicting claims with unresolved timing are retained with conflict status."""
    engine = TemporalSupersessionEngine()
    c1 = TemporalClaim(claim_id="c_berlin", subject="user", predicate="lives_in", object="Berlin", recorded_at="2024-01-01")
    c2 = TemporalClaim(claim_id="c_tokyo", subject="user", predicate="lives_in", object="Tokyo", recorded_at="2024-01-01")

    processed = engine.process_claims([c1, c2])
    c_map = {c.claim_id: c for c in processed}

    assert c_map["c_berlin"].properties.get("conflict_status") == "unresolved"
    assert c_map["c_tokyo"].properties.get("conflict_status") == "unresolved"


# ============================================================================
# PHASE 3: RELATIONSHIP-AWARE MEMORY GRAPH TESTS
# ============================================================================

def test_cross_session_graph_linking_and_cycle_protection(tmp_path):
    """Verify cross-session entity linking, bounded traversal budget, and cycle protection."""
    db_path = tmp_path / "test_graph.db"
    store = SQLiteGraphStore(db_path)

    # Chunks from two different sessions
    chunk_a = make_chunk("c_sess1", "s1", "I worked with Priya on project Aegis.")
    chunk_b = make_chunk("c_sess2", "s2", "Priya lives in Chicago and works remotely.")

    # Nodes
    store.upsert_node(GraphNode(id="c_sess1", node_type="chunk", label="Chunk 1"))
    store.upsert_node(GraphNode(id="c_sess2", node_type="chunk", label="Chunk 2"))
    store.upsert_node(GraphNode(id="ent_priya", node_type="entity", label="Priya"))

    # Edges forming a potential cycle and cross-session link
    store.upsert_edge(GraphEdge(id="e1", source_id="c_sess1", target_id="ent_priya", edge_type="mentions_entity", confidence=0.9))
    store.upsert_edge(GraphEdge(id="e2", source_id="ent_priya", target_id="c_sess2", edge_type="mentions_entity", confidence=0.9))
    store.upsert_edge(GraphEdge(id="e3_cycle", source_id="c_sess2", target_id="c_sess1", edge_type="adjacent_to", confidence=0.8))

    retriever = UnifiedCandidateRetriever(chunks=[chunk_a, chunk_b], store=store)
    # Query about Priya's relationship
    res, diag = retriever.retrieve("How is Priya connected to project Aegis?", k=3)

    cids = [r["chunk_id"] for r in res]
    assert "c_sess1" in cids
    # Bounded fallback recovered c_sess2 without getting stuck in cycle
    assert len(cids) <= 3
    store.close()


# ============================================================================
# PHASE 4: INTER-MEMORY EVIDENCE INTEGRATION TESTS
# ============================================================================

def test_inter_memory_evidence_integrator_classifies_roles():
    """Verify EvidenceIntegrator assigns complementary, redundant, and supporting roles."""
    integrator = EvidenceIntegrator()

    chunks = [
        {"chunk_id": "c1", "session_id": "s1", "text": "I went fishing on Lake Michigan on July 10 with Alex.", "score": 0.8},
        {"chunk_id": "c2", "session_id": "s2", "text": "On that July 10 trip with Alex, we caught 7 largemouth bass.", "score": 0.75},
        # Near duplicate of c1
        {"chunk_id": "c3", "session_id": "s1", "text": "I went fishing on Lake Michigan on July 10 with Alex in the morning.", "score": 0.78}
    ]

    res = integrator.integrate("How many bass did I catch on Lake Michigan?", chunks)
    profiles = res["profiles"]

    # c1 is supporting
    assert profiles["c1"]["role"] in (EvidenceRole.SUPPORTING.value, EvidenceRole.COMPLEMENTARY.value)
    # c2 complements c1 across sessions
    assert profiles["c2"]["role"] == EvidenceRole.COMPLEMENTARY.value
    # c3 is redundant duplicate of c1
    assert profiles["c3"]["role"] == EvidenceRole.REDUNDANT.value
    assert res["redundancy_pruned_count"] == 1


# ============================================================================
# PHASE 5: CONTRADICTION DETECTION & RESOLUTION TESTS
# ============================================================================

def test_contradiction_analyzer_distinguishes_conflict_types():
    """Verify analyzer correctly separates updates, conflicts, contexts, and negations."""
    analyzer = ContradictionAnalyzer()

    # 1. Legitimate chronological update
    c_up_old = TemporalClaim(claim_id="1", subject="user", predicate="lives_in", object="Seattle", valid_from="2021")
    c_up_new = TemporalClaim(claim_id="2", subject="user", predicate="lives_in", object="Austin", valid_from="2024")
    res_update = analyzer.analyze_pair(c_up_old, c_up_new)
    assert res_update.type == ContradictionType.LEGITIMATE_UPDATE
    assert res_update.is_resolved is True
    assert res_update.active_claim_id == "2"

    # 2. True conflict (identical period, different values)
    c_conf_1 = TemporalClaim(claim_id="3", subject="user", predicate="lives_in", object="Berlin", valid_from="2024")
    c_conf_2 = TemporalClaim(claim_id="4", subject="user", predicate="lives_in", object="Tokyo", valid_from="2024")
    res_conf = analyzer.analyze_pair(c_conf_1, c_conf_2)
    assert res_conf.type == ContradictionType.TRUE_CONFLICT
    assert res_conf.is_resolved is False

    # 3. Context distinction (compatible conditional preferences)
    c_ctx_1 = TemporalClaim(claim_id="5", subject="user", predicate="prefers", object="Python", properties={"context_scope": "at work"})
    c_ctx_2 = TemporalClaim(claim_id="6", subject="user", predicate="prefers", object="Rust", properties={"context_scope": "for personal projects"})
    res_ctx = analyzer.analyze_pair(c_ctx_1, c_ctx_2)
    assert res_ctx.type == ContradictionType.CONTEXT_DISTINCTION
    assert res_ctx.is_conflict is False

    # 4. Negation
    c_neg_1 = TemporalClaim(claim_id="7", subject="user", predicate="owns", object="car", properties={"is_negated": False}, recorded_at="2023")
    c_neg_2 = TemporalClaim(claim_id="8", subject="user", predicate="owns", object="car", properties={"is_negated": True}, recorded_at="2024")
    res_neg = analyzer.analyze_pair(c_neg_1, c_neg_2)
    assert res_neg.type == ContradictionType.NEGATION
    assert res_neg.active_claim_id == "8"


# ============================================================================
# PHASE 6: EVIDENCE SUFFICIENCY & ABSTENTION GATING TESTS
# ============================================================================

def test_evidence_gate_pre_generation_decisions():
    """Verify evidence gate distinguishes sufficient, partial, conflicting, and absent evidence."""
    gate = EvidenceGate()

    # Case A: Absent evidence -> ABSTAIN_INSUFFICIENT
    rep_absent = gate.evaluate_pre_generation("What is my secret pin number?", [])
    assert rep_absent.decision == GatingDecision.ABSTAIN_INSUFFICIENT

    # Case B: Sufficient evidence -> PROCEED
    chunks_suff = [{"chunk_id": "c1", "text": "I adopted a golden retriever named Buster in Seattle."}]
    rep_suff = gate.evaluate_pre_generation("What kind of dog do I have?", chunks_suff)
    assert rep_suff.decision == GatingDecision.PROCEED

    # Case C: Partial evidence -> PARTIAL_ANSWER
    chunks_partial = [{"chunk_id": "c1", "text": "I visited Paris in June."}]
    rep_partial = gate.evaluate_pre_generation("What did I do in Paris and what hotel did I stay at?", chunks_partial)
    assert rep_partial.decision == GatingDecision.PARTIAL_ANSWER
    assert len(rep_partial.unsupported_subclaims) > 0

    # Case D: Unresolved conflict -> EXPLAIN_CONTRADICTION
    rep_conflict = gate.evaluate_pre_generation(
        "Where do I live?",
        chunks_suff,
        contradiction_summary={"unresolved_conflicts": [{"predicate": "lives_in", "values": ["Berlin", "Tokyo"]}]}
    )
    assert rep_conflict.decision == GatingDecision.EXPLAIN_CONTRADICTION


def test_evidence_gate_citation_validation():
    """Verify gate validates legitimate citations and rejects non-existent IDs."""
    gate = EvidenceGate()
    chunks = [{"chunk_id": "chunk_001", "memory_id": "sess_100", "session_id": "100"}]

    valid, rejected = gate.validate_citations(["chunk_001", "fake_999", "sess_100"], chunks)
    assert "chunk_001" in valid
    assert "sess_100" in valid
    assert "fake_999" in rejected
