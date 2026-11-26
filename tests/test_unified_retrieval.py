"""Comprehensive Test Suite for IVARA Milestone 3: Unified Retrieval System.

Validates:
1. Episodic Retrieval (semantic search, speaker awareness, session filtering, context windows).
2. Semantic Retrieval (memory IDs, entities, relationships, source turn IDs).
3. Knowledge Graph Retrieval (seed entities, directed triples, cycle prevention, hop decay).
4. Evidence Fusion (RRF-P, provenance deduplication, diversity capping).
5. Answer Generation Integration (sufficient evidence answering, citation validation, abstention).
"""

import json
from pathlib import Path
import pytest
import sqlite3

from src.conversations.store import ConversationStore, EpisodicTurn
from src.graph.store import SQLiteGraphStore
from src.graph.schema import GraphNode, GraphEdge
from src.unified_retrieval.episodic import EpisodicRetriever, EpisodicEvidenceItem
from src.unified_retrieval.semantic import SemanticRetriever, SemanticEvidenceItem
from src.unified_retrieval.graph import GraphRetriever, GraphEvidenceItem
from src.unified_retrieval.fusion import UnifiedEvidenceFuser, UnifiedEvidenceItem, RRFConfig
from src.unified_retrieval.pipeline import UnifiedRetriever, UnifiedRetrievalResult
from src.answering import answer_question


@pytest.fixture
def sample_conversation_store(tmp_path):
    """Provides an isolated ConversationStore populated with multi-session dialogue turns."""
    db_path = tmp_path / "test_conv.db"
    store = ConversationStore(db_path)
    store.create_conversation(conversation_id="conv_001", title="Arjun Background")
    store.create_conversation(conversation_id="conv_002", title="Arjun HyperScale")

    turns_conv_1 = [
        {"id": "turn_001", "role": "user", "speaker": "Arjun Mehta", "content": "I was born in Chandigarh on August 14, 1999.", "session_id": "session_01", "timestamp": "2024-03-10T10:00:00Z"},
        {"id": "turn_002", "role": "assistant", "speaker": "IVARA", "content": "Recorded: born in Chandigarh on August 14, 1999.", "session_id": "session_01", "timestamp": "2024-03-10T10:00:05Z"},
        {"id": "turn_003", "role": "user", "speaker": "Arjun Mehta", "content": "I completed my B.Tech in Computer Science from ABV IIITM Gwalior in June 2022.", "session_id": "session_01", "timestamp": "2024-03-10T10:01:00Z"},
        {"id": "turn_004", "role": "assistant", "speaker": "IVARA", "content": "Understood, graduated from ABV IIITM Gwalior.", "session_id": "session_01", "timestamp": "2024-03-10T10:01:05Z"},
        {"id": "turn_005", "role": "user", "speaker": "Arjun Mehta", "content": "My college roommate was Rohan Sharma. We lived in Hostel 3.", "session_id": "session_01", "timestamp": "2024-03-10T10:02:00Z"},
        {"id": "turn_006", "role": "assistant", "speaker": "IVARA", "content": "Recorded: Rohan Sharma was your college roommate.", "session_id": "session_01", "timestamp": "2024-03-10T10:02:05Z"},
        {"id": "turn_007", "role": "user", "speaker": "Arjun Mehta", "content": "I live in Indiranagar, Bengaluru with my rescue dog Kaalu.", "session_id": "session_01", "timestamp": "2024-03-10T10:03:00Z"}
    ]
    store.add_turns_batch("conv_001", turns_conv_1)

    turns_conv_2 = [
        {"id": "turn_008", "role": "user", "speaker": "Arjun Mehta", "content": "I joined HyperScale Systems in Bengaluru as an Associate Software Engineer.", "session_id": "session_02", "timestamp": "2024-06-15T09:00:00Z"},
        {"id": "turn_009", "role": "assistant", "speaker": "IVARA", "content": "Noted your role at HyperScale Systems.", "session_id": "session_02", "timestamp": "2024-06-15T09:00:05Z"},
        {"id": "turn_010", "role": "user", "speaker": "Arjun Mehta", "content": "My engineering manager at HyperScale is Vikram Rao.", "session_id": "session_02", "timestamp": "2024-06-15T09:01:00Z"}
    ]
    store.add_turns_batch("conv_002", turns_conv_2)
    return store


@pytest.fixture
def sample_semantic_memories():
    """Provides sample extracted memories corresponding to the dialogue turns."""
    return [
        {
            "id": "m001",
            "text": "Arjun Mehta was born in Chandigarh on August 14, 1999.",
            "timestamp": "2024-03-10",
            "source": "turn_001",
            "type": "biographical",
            "confidence": 1.0,
            "entities": [
                {"canonical_name": "Arjun Mehta", "mention": "Arjun", "types": ["Person"], "entity_id": "self"},
                {"canonical_name": "Chandigarh", "mention": "Chandigarh", "types": ["Location"], "entity_id": "ent_chandigarh"}
            ],
            "relationships": [
                {"source": {"id": "self", "canonical_name": "Arjun Mehta"}, "predicate": "born_in", "target": {"id": "ent_chandigarh", "canonical_name": "Chandigarh"}, "evidence": "born in Chandigarh", "origin": "explicit"}
            ]
        },
        {
            "id": "m002",
            "text": "Rohan Sharma was Arjun's college roommate in Hostel 3.",
            "timestamp": "2024-03-10",
            "source": "turn_005",
            "type": "social",
            "confidence": 1.0,
            "entities": [
                {"canonical_name": "Rohan Sharma", "mention": "Rohan Sharma", "types": ["Person"], "entity_id": "ent_rohan"},
                {"canonical_name": "Arjun Mehta", "mention": "Arjun", "types": ["Person"], "entity_id": "self"}
            ],
            "relationships": [
                {"source": {"id": "self", "canonical_name": "Arjun Mehta"}, "predicate": "roommate_of", "target": {"id": "ent_rohan", "canonical_name": "Rohan Sharma"}, "evidence": "college roommate was Rohan Sharma", "origin": "explicit"}
            ]
        },
        {
            "id": "m003",
            "text": "Arjun Mehta lives in Indiranagar, Bengaluru with his dog Kaalu.",
            "timestamp": "2024-03-10",
            "source": "turn_007",
            "type": "residence",
            "confidence": 1.0,
            "entities": [
                {"canonical_name": "Arjun Mehta", "mention": "Arjun", "types": ["Person"], "entity_id": "self"},
                {"canonical_name": "Indiranagar", "mention": "Indiranagar", "types": ["Location"], "entity_id": "ent_indiranagar"},
                {"canonical_name": "Kaalu", "mention": "Kaalu", "types": ["Animal"], "entity_id": "ent_kaalu"}
            ],
            "relationships": [
                {"source": {"id": "self", "canonical_name": "Arjun Mehta"}, "predicate": "lives_in", "target": {"id": "ent_indiranagar", "canonical_name": "Indiranagar"}, "evidence": "live in Indiranagar", "origin": "explicit"},
                {"source": {"id": "self", "canonical_name": "Arjun Mehta"}, "predicate": "has_pet", "target": {"id": "ent_kaalu", "canonical_name": "Kaalu"}, "evidence": "rescue dog named Kaalu", "origin": "explicit"}
            ]
        }
    ]


@pytest.fixture
def sample_graph_store(tmp_path):
    """Provides an isolated SQLiteGraphStore populated with entity nodes and domain edges."""
    db_path = tmp_path / "test_graph.db"
    store = SQLiteGraphStore(db_path)

    # Add entity nodes
    store.upsert_node(GraphNode(id="self", node_type="entity", label="Arjun Mehta", properties={"canonical_name": "Arjun Mehta", "types": ["Person"], "aliases": ["Arjun", "Mehta"]}))
    store.upsert_node(GraphNode(id="ent_chandigarh", node_type="entity", label="Chandigarh", properties={"canonical_name": "Chandigarh", "types": ["Location"], "aliases": []}))
    store.upsert_node(GraphNode(id="ent_iiitm", node_type="entity", label="ABV IIITM Gwalior", properties={"canonical_name": "ABV IIITM Gwalior", "types": ["Institution"], "aliases": ["IIITM Gwalior"]}))
    store.upsert_node(GraphNode(id="ent_rohan", node_type="entity", label="Rohan Sharma", properties={"canonical_name": "Rohan Sharma", "types": ["Person"], "aliases": ["Rohan"]}))
    store.upsert_node(GraphNode(id="ent_kaalu", node_type="entity", label="Kaalu", properties={"canonical_name": "Kaalu", "types": ["Animal"], "aliases": []}))
    store.upsert_node(GraphNode(id="ent_indiranagar", node_type="entity", label="Indiranagar", properties={"canonical_name": "Indiranagar", "types": ["Location"], "aliases": []}))

    # Add relationship edges
    store.upsert_edge(GraphEdge(id="e_001", source_id="self", target_id="ent_chandigarh", edge_type="born_in", origin="explicit", confidence=1.0, evidence="born in Chandigarh", memory_id="m001"))
    store.upsert_edge(GraphEdge(id="e_002", source_id="self", target_id="ent_iiitm", edge_type="graduated_from", origin="explicit", confidence=0.95, evidence="completed B.Tech from ABV IIITM Gwalior", memory_id="m004"))
    store.upsert_edge(GraphEdge(id="e_003", source_id="self", target_id="ent_rohan", edge_type="roommate_of", origin="explicit", confidence=1.0, evidence="roommate was Rohan Sharma", memory_id="m002"))
    store.upsert_edge(GraphEdge(id="e_004", source_id="self", target_id="ent_kaalu", edge_type="has_pet", origin="explicit", confidence=1.0, evidence="dog named Kaalu", memory_id="m003"))
    store.upsert_edge(GraphEdge(id="e_005", source_id="self", target_id="ent_indiranagar", edge_type="lives_in", origin="explicit", confidence=1.0, evidence="live in Indiranagar", memory_id="m003"))

    return store


# -----------------------------------------------------------------------------
# Phase 1: Episodic Retrieval Tests
# -----------------------------------------------------------------------------

def test_01_episodic_semantic_retrieval(sample_conversation_store):
    """Episodic retriever accurately retrieves relevant turns using semantic similarity."""
    retriever = EpisodicRetriever(store=sample_conversation_store)
    results = retriever.retrieve("Where was Arjun born?", k=3, min_score=0.30)
    assert len(results) > 0
    top = results[0]
    assert top.turn_id == "turn_001"
    assert "Chandigarh" in top.content
    assert top.speaker == "Arjun Mehta"
    assert top.channel == "episodic"


def test_02_episodic_speaker_aware_formatting(sample_conversation_store):
    """Episodic turns include speaker attribution eliminating perspective confusion."""
    retriever = EpisodicRetriever(store=sample_conversation_store)
    results = retriever.retrieve("Who was Arjun's college roommate?", k=3)
    assert len(results) > 0
    top = results[0]
    assert "Speaker: Arjun Mehta" in top.formatted_text
    assert "Rohan Sharma" in top.content


def test_03_episodic_session_and_conversation_filtering(sample_conversation_store):
    """Episodic retrieval respects session_id and conversation_id boundaries."""
    retriever = EpisodicRetriever(store=sample_conversation_store)

    # Filter to session_02 only
    results_s2 = retriever.retrieve("engineering manager", session_id="session_02")
    assert len(results_s2) > 0
    assert all(r.session_id == "session_02" for r in results_s2)
    assert any("Vikram Rao" in r.content for r in results_s2)

    # Filter to session_01 should not return session_02 items
    results_s1 = retriever.retrieve("engineering manager", session_id="session_01")
    assert not any("Vikram Rao" in r.content for r in results_s1)


def test_04_episodic_context_window_expansion(sample_conversation_store):
    """Context window expansion fetches adjacent turns with correct relational tags."""
    retriever = EpisodicRetriever(store=sample_conversation_store)
    results = retriever.retrieve("ABV IIITM Gwalior degree", k=1, window_size=1)
    assert len(results) == 1
    top = results[0]
    assert top.turn_id == "turn_003"
    assert len(top.context_window) >= 2

    rels = [w["relation"] for w in top.context_window]
    assert "target" in rels
    assert "preceding" in rels or "succeeding" in rels


# -----------------------------------------------------------------------------
# Phase 2: Semantic Retrieval Tests
# -----------------------------------------------------------------------------

def test_05_semantic_retrieval_and_metadata(sample_semantic_memories):
    """Semantic retriever exposes structured memory ID, content, entities, and turn provenance."""
    retriever = SemanticRetriever(memories=sample_semantic_memories)
    results = retriever.retrieve("Where does Arjun live with his dog?", k=3)
    assert len(results) > 0
    m3 = next((r for r in results if r.memory_id == "m003"), None)
    assert m3 is not None
    assert "Indiranagar" in m3.text
    assert m3.source_turn_id == "turn_007"
    assert len(m3.relationships) == 2
    preds = [r["predicate"] for r in m3.relationships]
    assert "lives_in" in preds
    assert "has_pet" in preds


# -----------------------------------------------------------------------------
# Phase 3: Graph Retrieval Tests
# -----------------------------------------------------------------------------

def test_06_graph_retrieval_seed_and_directed_triples(sample_graph_store):
    """Graph retriever identifies seeds, traverses relationships, and returns directed triples."""
    retriever = GraphRetriever(store=sample_graph_store)
    seeds = retriever.identify_seed_entities("Tell me about Arjun Mehta's pets")
    assert "self" in seeds

    triples = retriever.retrieve("Arjun Mehta pet dog", k=5, seed_entity_ids=seeds)
    assert len(triples) > 0
    pet_triple = next((t for t in triples if t.predicate == "has_pet"), None)
    assert pet_triple is not None
    assert pet_triple.source_name == "Arjun Mehta"
    assert pet_triple.target_name == "Kaalu"
    assert pet_triple.direction == "outgoing"
    assert pet_triple.hops == 1


def test_07_graph_retrieval_cycle_prevention(sample_graph_store):
    """Graph retriever prevents infinite loops on symmetric relationships."""
    # Add symmetric edges between Rohan and Arjun
    sample_graph_store.upsert_edge(GraphEdge(
        id="e_sym_01", source_id="ent_rohan", target_id="self", edge_type="roommate_of", origin="explicit", confidence=1.0
    ))
    retriever = GraphRetriever(store=sample_graph_store, max_depth=2, max_nodes=10)
    triples = retriever.retrieve("Rohan Sharma roommate", seed_entity_ids=["ent_rohan"])
    # Triples must not contain duplicate reverse edges in the results
    keys = [(t.source_id, t.predicate, t.target_id) for t in triples]
    assert len(keys) == len(set(keys))


# -----------------------------------------------------------------------------
# Phase 4: Evidence Fusion Tests (RRF-P)
# -----------------------------------------------------------------------------

def test_08_evidence_fusion_rrf_and_deduplication(sample_conversation_store, sample_semantic_memories, sample_graph_store):
    """UnifiedEvidenceFuser merges multi-channel hits without duplicate counting."""
    epi_ret = EpisodicRetriever(store=sample_conversation_store)
    sem_ret = SemanticRetriever(memories=sample_semantic_memories)
    graph_ret = GraphRetriever(store=sample_graph_store)

    q = "Where was Arjun born?"
    epi_hits = epi_ret.retrieve(q, k=3)
    sem_hits = sem_ret.retrieve(q, k=3)
    graph_hits = graph_ret.retrieve(q, k=3)

    fuser = UnifiedEvidenceFuser(config=RRFConfig())
    fused = fuser.fuse(epi_hits, sem_hits, graph_hits, top_k=3)

    assert len(fused) > 0
    top = fused[0]
    # m001 links to turn_001, so top cluster combines semantic and episodic channels
    assert "semantic" in top.channels
    assert "episodic" in top.channels or "graph" in top.channels
    assert top.episodic_turn is not None
    assert top.semantic_memory is not None
    assert "m001" in top.citations
    assert "turn_001" in top.citations


def test_09_evidence_fusion_diversity_capping(sample_conversation_store, sample_semantic_memories):
    """RRF-P caps dominant sessions to ensure diverse coverage across multi-session topics."""
    epi_ret = EpisodicRetriever(store=sample_conversation_store)
    sem_ret = SemanticRetriever(memories=sample_semantic_memories)

    q = "Arjun background and employment"
    epi_hits = epi_ret.retrieve(q, k=10)
    sem_hits = sem_ret.retrieve(q, k=10)

    # Set max_items_per_session = 2
    fuser = UnifiedEvidenceFuser(config=RRFConfig(max_items_per_session=2))
    fused = fuser.fuse(epi_hits, sem_hits, [], top_k=5)

    sess_counts = {}
    for item in fused:
        if item.session_id:
            sess_counts[item.session_id] = sess_counts.get(item.session_id, 0) + 1
    for sess, count in sess_counts.items():
        assert count <= 2


# -----------------------------------------------------------------------------
# Phase 5: Pipeline & Answering Integration Tests
# -----------------------------------------------------------------------------

def test_10_unified_retriever_pipeline_end_to_end(sample_conversation_store, sample_semantic_memories, sample_graph_store):
    """UnifiedRetriever pipeline executes across all 3 channels and formats context text."""
    pipeline = UnifiedRetriever(
        conversation_store=sample_conversation_store,
        memories=sample_semantic_memories,
        graph_store=sample_graph_store
    )
    res = pipeline.retrieve("Where was Arjun born and what university did he attend?", k=5)
    assert res.has_sufficient_evidence is True
    assert res.status == "sufficient_evidence"
    assert len(res.evidence_items) > 0
    assert "--- EPISODIC CONVERSATION TURNS ---" in res.context_text
    assert "--- SEMANTIC MEMORIES & EXTRACTED FACTS ---" in res.context_text
    assert "--- KNOWLEDGE GRAPH RELATIONSHIPS ---" in res.context_text


def test_11_answer_question_with_unified_retrieval(sample_conversation_store, sample_semantic_memories, sample_graph_store):
    """answer_question integrates UnifiedRetriever and produces grounded citations."""
    def mock_llm(prompt, sys_prompt):
        return {
            "status": "answered",
            "answer": "Arjun Mehta was born in Chandigarh on August 14, 1999 [m001] and graduated from ABV IIITM Gwalior [turn_003].",
            "cited_memory_ids": ["m001", "turn_003"]
        }

    ans = answer_question(
        query="From which university did Arjun graduate and where was he born?",
        k=5,
        use_unified=True,
        conversation_store=sample_conversation_store,
        memories=sample_semantic_memories,
        graph_store=sample_graph_store,
        rrf_config=RRFConfig(max_items_per_session=5),
        llm_caller=mock_llm
    )
    assert ans["status"] == "answered"
    assert "Chandigarh" in ans["answer"]
    assert "ABV IIITM Gwalior" in ans["answer"]
    cited_ids = [c["memory_id"] for c in ans["citations"]]
    assert "m001" in cited_ids
    assert "turn_003" in cited_ids


def test_12_answer_question_abstention_on_unrecorded_facts(sample_conversation_store, sample_semantic_memories, sample_graph_store):
    """answer_question strictly abstains when unified retrieval discovers no evidence."""
    def mock_abstaining_llm(prompt, system_prompt):
        import json
        evidence_section = prompt.split("</evidence_context>")[0]
        body_after_query = evidence_section.split("\n", 2)[-1].lower()
        assert "star wars" not in body_after_query
        return json.dumps({
            "status": "insufficient_evidence",
            "answer": "I do not have enough information in my recorded memories to answer that.",
            "cited_memory_ids": [],
            "limitations": "No Star Wars spaceship preference is recorded.",
        })

    ans = answer_question(
        query="What is Arjun's favorite spaceship in Star Wars?",
        use_unified=True,
        conversation_store=sample_conversation_store,
        memories=sample_semantic_memories,
        graph_store=sample_graph_store,
        min_score=0.35,
        llm_caller=mock_abstaining_llm,
    )
    assert ans["status"] == "insufficient_evidence"
    ans_lower = ans["answer"].lower()
    assert any(phrase in ans_lower for phrase in ("sufficient", "not have", "don't have", "enough information"))
    assert ans["citations"] == []
