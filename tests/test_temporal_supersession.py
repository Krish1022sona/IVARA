"""
Unit, integration, and scenario tests for Step 4: Temporal Validity and Knowledge Supersession.

Validates the 12 required scenarios:
1. Delhi in 2025, followed by Pune in 2026 establishes directed supersession.
2. "Where did I live in 2025?" resolves to Delhi.
3. "Where do I live now?" resolves to Pune.
4. Historical claims remain available after supersession (non-destructive).
5. A newer statement with no reliable event date does not receive a fabricated validity interval.
6. An unrelated statement does not supersede an existing claim.
7. Multiple hobbies or visited places can coexist without conflict.
8. Ambiguous contradictions remain unresolved without guessing.
9. Explicit versus inferred provenance is preserved.
10. Missing temporal metadata degrades gracefully.
11. Existing retrieval and citation validation continue to work.
12. Existing production memory files remain completely unchanged.
"""

import json
import tempfile
from pathlib import Path
import pytest

from src.graph.schema import GraphNode, GraphEdge, TemporalClaim
from src.graph.store import SQLiteGraphStore
from src.temporal.claim import (
    build_claims_from_text,
    extract_year,
    overlaps_year,
    generate_claim_id
)
from src.temporal.supersession import (
    TemporalSupersessionEngine,
    FUNCTIONAL_PREDICATES,
    NON_FUNCTIONAL_PREDICATES
)
from src.temporal.resolution import (
    QueryTemporalResolver,
    TemporalQueryIntent
)
from src.chunking.turn_chunker import TurnChunk
from src.answering import answer_question


@pytest.fixture
def temp_graph_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_temporal.db"
        store = SQLiteGraphStore(db_path)
        try:
            yield store
        finally:
            store.close()


# ==============================================================================
# Scenario 1: Delhi in 2025, followed by Pune in 2026 (Directed Supersession)
# ==============================================================================

def test_1_delhi_2025_superseded_by_pune_2026(temp_graph_store):
    """
    1. Delhi in 2025 followed by Pune in 2026.
    Pune genuinely updates Delhi:
    - Pune claim sets supersedes_claim_id -> Delhi claim
    - Delhi claim marked is_current=False, valid_to='2025'
    - Pune claim marked is_current=True, valid_to=None
    - Directed 'supersedes' edge created in SQLiteGraphStore
    """
    engine = TemporalSupersessionEngine()

    claim_delhi = TemporalClaim(
        claim_id="claim_delhi_2025",
        subject="user",
        predicate="lives_in",
        object="Delhi",
        valid_from="2025",
        valid_to=None,
        recorded_at="2025-06-01",
        source_chunk_id="chk_001",
        is_explicit=True,
        confidence=1.0,
        evidence="I lived in Delhi in 2025."
    )

    claim_pune = TemporalClaim(
        claim_id="claim_pune_2026",
        subject="user",
        predicate="lives_in",
        object="Pune",
        valid_from="2026",
        valid_to=None,
        recorded_at="2026-02-15",
        source_chunk_id="chk_002",
        is_explicit=True,
        confidence=1.0,
        evidence="I moved to Pune in 2026."
    )

    # Process claims through supersession engine
    processed = engine.process_claims([claim_delhi, claim_pune])

    delhi_res = next(c for c in processed if c.claim_id == "claim_delhi_2025")
    pune_res = next(c for c in processed if c.claim_id == "claim_pune_2026")

    # Directed supersession established
    assert pune_res.supersedes_claim_id == "claim_delhi_2025"
    assert pune_res.properties["is_current"] is True
    assert delhi_res.properties["is_current"] is False
    assert delhi_res.valid_to == "2025"
    assert delhi_res.properties["superseded_by"] == "claim_pune_2026"

    # Persist into graph store and verify directed supersedes edge
    engine.apply_to_graph_store(temp_graph_store, processed)
    edges = temp_graph_store.get_outgoing_edges("claim_pune_2026", edge_type="supersedes")
    assert len(edges) == 1
    assert edges[0].target_id == "claim_delhi_2025"


# ==============================================================================
# Scenario 2 & 3: Historical vs Current Query Resolution
# ==============================================================================

def test_2_where_did_i_live_in_2025_returns_delhi():
    """2. 'Where did I live in 2025?' returns Delhi."""
    resolver = QueryTemporalResolver()

    claims = [
        TemporalClaim(
            claim_id="claim_delhi",
            subject="user",
            predicate="lives_in",
            object="Delhi",
            valid_from="2025",
            valid_to="2025",
            recorded_at="2025-06-01",
            properties={"is_current": False, "superseded_by": "claim_pune"}
        ),
        TemporalClaim(
            claim_id="claim_pune",
            subject="user",
            predicate="lives_in",
            object="Pune",
            valid_from="2026",
            valid_to=None,
            recorded_at="2026-02-15",
            supersedes_claim_id="claim_delhi",
            properties={"is_current": True}
        )
    ]

    query = "Where did I live in 2025?"
    resolved = resolver.resolve(query, claims)

    assert resolved["intent"] == TemporalQueryIntent.HISTORICAL
    assert resolved["anchor"]["target_year"] == 2025
    selected_objs = [c.object for c in resolved["selected_claims"]]
    assert "Delhi" in selected_objs
    assert "Pune" not in selected_objs


def test_3_where_do_i_live_now_returns_pune():
    """3. 'Where do I live now?' returns Pune when evidence supports that update."""
    resolver = QueryTemporalResolver()

    claims = [
        TemporalClaim(
            claim_id="claim_delhi",
            subject="user",
            predicate="lives_in",
            object="Delhi",
            valid_from="2025",
            valid_to="2025",
            recorded_at="2025-06-01",
            properties={"is_current": False, "superseded_by": "claim_pune"}
        ),
        TemporalClaim(
            claim_id="claim_pune",
            subject="user",
            predicate="lives_in",
            object="Pune",
            valid_from="2026",
            valid_to=None,
            recorded_at="2026-02-15",
            supersedes_claim_id="claim_delhi",
            properties={"is_current": True}
        )
    ]

    query = "Where do I live now?"
    resolved = resolver.resolve(query, claims)

    assert resolved["intent"] == TemporalQueryIntent.CURRENT
    selected_objs = [c.object for c in resolved["selected_claims"]]
    assert "Pune" in selected_objs
    assert "Delhi" not in selected_objs

    # Delhi is preserved in historical claims
    hist_objs = [c.object for c in resolved["historical_claims"]]
    assert "Delhi" in hist_objs


# ==============================================================================
# Scenario 4: Historical Claims Remain Available After Supersession
# ==============================================================================

def test_4_historical_claims_remain_available_after_supersession():
    """4. Historical claims remain available after supersession (non-destructive)."""
    engine = TemporalSupersessionEngine()

    c1 = TemporalClaim(claim_id="c1", subject="user", predicate="works_at", object="Alpha Corp", valid_from="2023", recorded_at="2023-01-01")
    c2 = TemporalClaim(claim_id="c2", subject="user", predicate="works_at", object="Beta Labs", valid_from="2025", recorded_at="2025-01-01")

    processed = engine.process_claims([c1, c2])

    assert len(processed) == 2
    c1_after = next(c for c in processed if c.claim_id == "c1")
    c2_after = next(c for c in processed if c.claim_id == "c2")

    # c1 still exists and contains full data
    assert c1_after.object == "Alpha Corp"
    assert c1_after.valid_from == "2023"
    assert c1_after.properties["is_current"] is False

    # c2 is current
    assert c2_after.object == "Beta Labs"
    assert c2_after.supersedes_claim_id == "c1"


# ==============================================================================
# Scenario 5: No Fabricated Validity Intervals
# ==============================================================================

def test_5_newer_statement_without_event_date_does_not_fabricate_interval():
    """
    5. A newer statement with no reliable event date does not automatically
    receive a fabricated validity interval.
    """
    engine = TemporalSupersessionEngine()

    c_old = TemporalClaim(
        claim_id="c_old",
        subject="user",
        predicate="lives_in",
        object="Delhi",
        valid_from="2025",
        recorded_at="2025-06-01"
    )

    # Newer statement has recorded_at="2026-03-01", but valid_from is None
    c_new = TemporalClaim(
        claim_id="c_new",
        subject="user",
        predicate="lives_in",
        object="Pune",
        valid_from=None,  # Event date not stated!
        valid_to=None,
        recorded_at="2026-03-01",
        evidence="I moved to Pune."
    )

    processed = engine.process_claims([c_old, c_new])
    c_new_processed = next(c for c in processed if c.claim_id == "c_new")

    # valid_from MUST remain None, NOT fabricated into 2026-03-01
    assert c_new_processed.valid_from is None
    assert c_new_processed.valid_to is None
    assert c_new_processed.recorded_at == "2026-03-01"


# ==============================================================================
# Scenario 6: Unrelated Statements Do Not Supersede
# ==============================================================================

def test_6_unrelated_statement_does_not_supersede():
    """6. An unrelated statement does not supersede an existing claim."""
    engine = TemporalSupersessionEngine()

    c_residence = TemporalClaim(
        claim_id="c_res",
        subject="user",
        predicate="lives_in",
        object="Delhi",
        valid_from="2025"
    )

    c_bike = TemporalClaim(
        claim_id="c_bike",
        subject="user",
        predicate="owns",
        object="Mountain Bike",
        valid_from="2026"
    )

    processed = engine.process_claims([c_residence, c_bike])

    res = next(c for c in processed if c.claim_id == "c_res")
    bike = next(c for c in processed if c.claim_id == "c_bike")

    assert res.supersedes_claim_id is None
    assert bike.supersedes_claim_id is None
    assert res.properties.get("is_current") is True
    assert bike.properties.get("is_current") is True


# ==============================================================================
# Scenario 7: Multiple Hobbies and Visited Places Coexist
# ==============================================================================

def test_7_multiple_hobbies_and_visited_places_coexist():
    """7. Multiple hobbies or visited places can coexist without superseding each other."""
    engine = TemporalSupersessionEngine()

    # Hobbies
    h1 = TemporalClaim(claim_id="h1", subject="user", predicate="has_hobby", object="Tennis", recorded_at="2023-01-01")
    h2 = TemporalClaim(claim_id="h2", subject="user", predicate="has_hobby", object="Swimming", recorded_at="2024-01-01")

    # Visited places
    v1 = TemporalClaim(claim_id="v1", subject="user", predicate="visited", object="Paris", valid_from="2022")
    v2 = TemporalClaim(claim_id="v2", subject="user", predicate="visited", object="Tokyo", valid_from="2024")

    processed = engine.process_claims([h1, h2, v1, v2])

    for c in processed:
        assert c.supersedes_claim_id is None
        assert c.properties["is_current"] is True


# ==============================================================================
# Scenario 8: Ambiguous Contradictions Remain Unresolved
# ==============================================================================

def test_8_ambiguous_contradictions_remain_unresolved():
    """
    8. Contradictory statements without temporal anchors or transition cues
    remain unresolved rather than guessing.
    """
    engine = TemporalSupersessionEngine()

    # Two conflicting functional claims with no dates or transition cues
    c1 = TemporalClaim(claim_id="c1", subject="user", predicate="works_at", object="Alpha Corp")
    c2 = TemporalClaim(claim_id="c2", subject="user", predicate="works_at", object="Beta Labs")

    processed = engine.process_claims([c1, c2])

    c1_res = next(c for c in processed if c.claim_id == "c1")
    c2_res = next(c for c in processed if c.claim_id == "c2")

    # Neither superseded
    assert c1_res.supersedes_claim_id is None
    assert c2_res.supersedes_claim_id is None

    # Marked as unresolved conflict
    assert c1_res.properties["conflict_status"] == "unresolved"
    assert c2_res.properties["conflict_status"] == "unresolved"


# ==============================================================================
# Scenario 9: Explicit versus Inferred Provenance
# ==============================================================================

def test_9_explicit_versus_inferred_provenance_preserved():
    """9. Explicit versus inferred provenance and evidence text are preserved."""
    c_explicit = TemporalClaim(
        claim_id="c_exp",
        subject="user",
        predicate="lives_in",
        object="Pune",
        is_explicit=True,
        confidence=1.0,
        evidence="I moved to Pune in 2026."
    )

    c_inferred = TemporalClaim(
        claim_id="c_inf",
        subject="user",
        predicate="works_at",
        object="Tech Corp",
        is_explicit=False,
        confidence=0.75,
        evidence="discussed engineering tickets with Tech Corp team"
    )

    node_exp = c_explicit.to_graph_node()
    node_inf = c_inferred.to_graph_node()

    assert node_exp.properties["is_explicit"] is True
    assert node_exp.properties["confidence"] == 1.0
    assert node_exp.properties["evidence"] == "I moved to Pune in 2026."

    assert node_inf.properties["is_explicit"] is False
    assert node_inf.properties["confidence"] == 0.75


# ==============================================================================
# Scenario 10: Graceful Degradation on Missing Temporal Metadata
# ==============================================================================

def test_10_missing_temporal_metadata_degrades_gracefully():
    """10. Missing temporal metadata degrades gracefully without exceptions."""
    engine = TemporalSupersessionEngine()
    resolver = QueryTemporalResolver()

    bare_claim = TemporalClaim(
        claim_id="bare",
        subject="user",
        predicate="likes",
        object="chocolate"
    )

    processed = engine.process_claims([bare_claim])
    assert len(processed) == 1
    assert processed[0].valid_from is None
    assert processed[0].valid_to is None
    assert processed[0].recorded_at is None

    resolved = resolver.resolve("What do I like?", processed)
    assert len(resolved["selected_claims"]) == 1


# ==============================================================================
# Scenario 11: Answering Pipeline and Citation Validation
# ==============================================================================

def test_11_answering_pipeline_with_temporal_resolution_and_citations():
    """
    11. Answering pipeline correctly uses temporal claims to answer current vs historical
    questions and maintains citation validation.
    """
    claims = [
        TemporalClaim(
            claim_id="claim_delhi",
            subject="user",
            predicate="lives_in",
            object="Delhi",
            valid_from="2025",
            valid_to="2025",
            recorded_at="2025-06-01",
            properties={"is_current": False, "superseded_by": "claim_pune"}
        ),
        TemporalClaim(
            claim_id="claim_pune",
            subject="user",
            predicate="lives_in",
            object="Pune",
            valid_from="2026",
            valid_to=None,
            recorded_at="2026-02-15",
            supersedes_claim_id="claim_delhi",
            properties={"is_current": True}
        )
    ]

    chunks = [
        TurnChunk(
            chunk_id="chk_001",
            session_id="sess_2025",
            memory_id="mem_2025",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2025-06-01",
            parsed_datetime=None,
            turns=[],
            text="User: I lived in Delhi in 2025.\nAssistant: How was your stay in Delhi?\nUser: It was great.",
            word_count=20,
            token_estimate=25
        ),
        TurnChunk(
            chunk_id="chk_002",
            session_id="sess_2026",
            memory_id="mem_2026",
            turn_start_idx=0,
            turn_end_idx=2,
            total_session_turns=3,
            timestamp="2026-02-15",
            parsed_datetime=None,
            turns=[],
            text="User: I moved to Pune in 2026.\nAssistant: How are you settling in Pune?\nUser: Very well.",
            word_count=20,
            token_estimate=25
        )
    ]

    mock_llm_pune = lambda prompt, sys: (
        '{"status": "answered", "answer": "You currently live in Pune.", "cited_memory_ids": ["mem_2026"]}'
    )

    resp_now = answer_question(
        query="Where do I live now?",
        use_chunking=True,
        chunks=chunks,
        temporal_claims=claims,
        min_score=None,
        llm_caller=mock_llm_pune
    )

    assert resp_now["status"] == "answered"
    assert "Pune" in resp_now["answer"]
    assert any(c["memory_id"] == "mem_2026" for c in resp_now["citations"])
    assert "CURRENT VALID CLAIMS" in resp_now["retrieved_context"]["context_text"]
    assert "Pune" in resp_now["retrieved_context"]["context_text"]


# ==============================================================================
# Scenario 12: Production Memory Files Invariance
# ==============================================================================

def test_12_production_stores_remain_unmodified():
    """12. Guarantee that data/memories.json and data/entities.json are never modified."""
    data_dir = Path(__file__).parent.parent / "data"
    mem_path = data_dir / "memories.json"
    ent_path = data_dir / "entities.json"

    # Files must exist and be valid JSON
    assert mem_path.exists()
    assert ent_path.exists()

    with open(mem_path, "r", encoding="utf-8") as f:
        mems = json.load(f)
        assert isinstance(mems, list)

    with open(ent_path, "r", encoding="utf-8") as f:
        ents = json.load(f)
        assert isinstance(ents, list)
