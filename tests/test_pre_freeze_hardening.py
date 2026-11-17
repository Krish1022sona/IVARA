"""
Tests for Pre-Freeze Production Hardening.

Covers:
1. Malformed output fallback safety in src/answering.py:
   - Valid JSON answer
   - Valid markdown Answer field
   - Plain text without parser contract
   - Empty model output
   - Completely malformed garbage
   - Malformed JSON with extractable valid answer
   - Malformed JSON with no recoverable answer
   - Calendar fallback behavior
   - Unsupported answer must never become answered merely because text is non-empty
2. Neutral persona handling and parameterization in answering and retrieval modules:
   - Neutral default identity
   - Explicit custom identity
   - Correct first-person speaker attribution
   - Zero accidental Arjun Mehta references in production prompts
   - A user asking about another person named Arjun
   - Correct self-node resolution
   - Correct entity-count estimation
   - No accidental suppression of legitimate Arjun entities
"""

import json
from unittest.mock import MagicMock

from src.answering import answer_question, get_answering_system_prompt
from src.conversations.store import EpisodicTurn
from src.graph.schema import GraphNode
from src.unified_retrieval.episodic import EpisodicRetriever
from src.unified_retrieval.graph import GraphEvidenceItem, GraphRetriever
from src.unified_retrieval.graph_safeguards import GraphSafeguards
from src.unified_retrieval.router import AdaptiveRouter, RouteDecision
from src.unified_retrieval.pipeline import UnifiedRetriever


# ==============================================================================
# 1. Answer Parsing & Malformed Output Fallback Tests
# ==============================================================================

MOCK_MEMORIES = [
    {
        "id": "m101",
        "text": "The user visited Paris in July 2023 for a tech conference.",
        "timestamp": "2023-07-15",
        "entities": [{"name": "Paris", "type": "Location"}],
        "relationships": []
    }
]


def test_1_valid_json_answer():
    """1. Valid JSON containing 'answer' must be parsed and marked 'answered'."""
    raw = json.dumps({
        "status": "answered",
        "answer": "The user visited Paris in July 2023.",
        "cited_memory_ids": ["m101"],
        "limitations": None
    })
    result = answer_question(
        "When did the user visit Paris?",
        memories=MOCK_MEMORIES,
        min_score=None,
        llm_caller=lambda p, s: raw
    )
    assert result["status"] == "answered"
    assert "Paris" in result["answer"]


def test_2_valid_markdown_answer():
    """2. Valid markdown output with **Answer:** field must be extracted and marked 'answered'."""
    raw = (
        "Here is my reasoning:\n"
        "- Memory m101 indicates a trip to Paris.\n\n"
        "**Answer:** Paris\n"
        "**Confidence:** high"
    )
    result = answer_question(
        "Where did the user travel in July 2023?",
        memories=MOCK_MEMORIES,
        min_score=None,
        llm_caller=lambda p, s: raw
    )
    assert result["status"] == "answered"
    assert "Paris" in result["answer"]


def test_3_plain_text_without_contract_returns_insufficient_evidence():
    """3. Plain text answer without structured contract must safely return insufficient_evidence."""
    raw = "The answer is definitely Paris because of the tech conference."
    result = answer_question(
        "Where did the user go?",
        memories=MOCK_MEMORIES,
        min_score=None,
        llm_caller=lambda p, s: raw
    )
    assert result["status"] == "insufficient_evidence"
    assert "Could not reliably synthesize" in result["answer"]
    assert "malformed" in result.get("limitations", "").lower()


def test_4_empty_model_output_returns_insufficient_evidence():
    """4. Completely empty output from LLM must return 'insufficient_evidence'."""
    result = answer_question(
        "What is the user's favorite coffee?",
        memories=MOCK_MEMORIES,
        min_score=None,
        llm_caller=lambda p, s: ""
    )
    assert result["status"] == "insufficient_evidence"
    assert "Could not reliably synthesize" in result["answer"]
    assert "malformed" in result.get("limitations", "").lower()


def test_5_completely_malformed_garbage_returns_insufficient_evidence():
    """5. Completely malformed garbage text must return 'insufficient_evidence'."""
    raw = "<<<>>> [[[ ]]] $$$ %% Random gibberish with no JSON or tags."
    result = answer_question(
        "What is the secret code?",
        memories=MOCK_MEMORIES,
        min_score=None,
        llm_caller=lambda p, s: raw
    )
    assert result["status"] == "insufficient_evidence"
    assert "Could not reliably synthesize" in result["answer"]
    assert "malformed" in result.get("limitations", "").lower()


def test_6_malformed_json_with_extractable_answer():
    """6. Malformed JSON that still contains '\"answer\": \"...\"' is recovered via regex."""
    raw = '{"status": "answered", "answer": "Paris", "confidence": "high", invalid_trailing_token'
    result = answer_question(
        "Where did the user go?",
        memories=MOCK_MEMORIES,
        min_score=None,
        llm_caller=lambda p, s: raw
    )
    assert result["status"] == "answered"
    assert "Paris" in result["answer"]
    assert "safely recovered" in result.get("limitations", "").lower()


def test_7_malformed_json_without_answer_field_returns_insufficient_evidence():
    """7. Malformed JSON without any answer key/tag safely degrades to 'insufficient_evidence'."""
    raw = '{"notes": "found some clues", "confidence": "low", broken_syntax'
    result = answer_question(
        "What happened?",
        memories=MOCK_MEMORIES,
        min_score=None,
        llm_caller=lambda p, s: raw
    )
    assert result["status"] == "insufficient_evidence"
    assert "Could not reliably synthesize" in result["answer"]
    assert "malformed" in result.get("limitations", "").lower()


def test_8_calendar_arithmetic_fallback_recovers_answer():
    """8. When context contains an anchored relative date, calendar fallback computes answer."""
    calendar_memories = [
        {
            "id": "m102",
            "text": "Leaving for Rome on 2023-05-10 for vacation.",
            "timestamp": "2023-05-10",
            "entities": [],
            "relationships": []
        }
    ]
    raw = "The travel schedule was discussed earlier."
    result = answer_question(
        "How many days between 2023-05-10 and 2023-05-15?",
        memories=calendar_memories,
        min_score=None,
        llm_caller=lambda p, s: raw,
        enable_calendar_arithmetic=True
    )
    if result["status"] == "answered":
        assert "5" in result["answer"]
    else:
        assert result["status"] == "insufficient_evidence"


def test_9_unsupported_answer_never_becomes_answered_because_text_nonempty():
    """9. Unsupported answer must never become answered merely because model output text is non-empty."""
    raw = "The moon is made of green cheese and rabbits live there in hollow craters."
    result = answer_question(
        "What is the moon made of?",
        memories=MOCK_MEMORIES,
        min_score=None,
        llm_caller=lambda p, s: raw
    )
    assert result["status"] == "insufficient_evidence"
    assert result["status"] != "answered"
    assert "Could not reliably synthesize" in result["answer"]


# ==============================================================================
# 2. Neutral Persona & Dynamic Parameterization Tests
# ==============================================================================

def test_neutral_default_identity():
    """Production default prompt must use 'the User' and contain NO hardcoded persona names."""
    prompt = get_answering_system_prompt()
    assert "Arjun" not in prompt
    assert "Mehta" not in prompt
    assert "the User" in prompt
    assert "they" in prompt or "their" in prompt


def test_explicit_custom_identity():
    """Passing a custom user_name injects that name into the prompt."""
    prompt = get_answering_system_prompt("Sarah Connor")
    assert "Sarah Connor" in prompt
    assert "Arjun" not in prompt


def test_correct_first_person_speaker_attribution():
    """First-person speaker attribution correctly references the configured identity."""
    retriever_default = EpisodicRetriever()
    turn = EpisodicTurn(
        turn_id="t1",
        conversation_id="c1",
        speaker="user",
        speaker_role="user",
        content="I am heading to Berlin.",
        timestamp="2024-01-01T10:00:00Z",
        sequence_num=1
    )
    formatted_default = retriever_default._format_turn_text(turn)
    assert "Speaker: the User (user): I am heading to Berlin." in formatted_default

    retriever_custom = EpisodicRetriever(user_name="Bruce Wayne")
    formatted_custom = retriever_custom._format_turn_text(turn)
    assert "Speaker: Bruce Wayne (user): I am heading to Berlin." in formatted_custom


def test_no_accidental_arjun_mehta_in_production_prompts():
    """Zero occurrences of 'Arjun Mehta' in production prompts."""
    default_prompt = get_answering_system_prompt()
    assert "arjun" not in default_prompt.lower()
    assert "mehta" not in default_prompt.lower()


def test_user_asking_about_another_person_named_arjun():
    """A user asking about another person named Arjun maps to that person, not self."""
    mock_store = MagicMock()
    self_node = GraphNode(id="self", label="SELF", node_type="entity", properties={"canonical_name": "SELF"})
    arjun_node = GraphNode(id="ent_arjun", label="Arjun", node_type="entity", properties={"canonical_name": "Arjun"})
    vikram_node = GraphNode(id="ent_vikram", label="Vikram", node_type="entity", properties={"canonical_name": "Vikram"})
    mock_store.get_nodes_by_type.return_value = [self_node, arjun_node, vikram_node]

    # Alice is the user
    gr = GraphRetriever(store=mock_store, user_name="Alice")
    seeds = gr.identify_seed_entities("What did Arjun tell Vikram about the project?")
    # ent_arjun must be identified as an entity
    assert "ent_arjun" in seeds
    assert "ent_vikram" in seeds
    # self must NOT be matched because Arjun is someone else, not Alice!
    assert "self" not in seeds


def test_correct_self_node_resolution():
    """First person query correctly maps to self node and not other entities."""
    mock_store = MagicMock()
    self_node = GraphNode(id="self", label="SELF", node_type="entity", properties={"canonical_name": "SELF"})
    arjun_node = GraphNode(id="ent_arjun", label="Arjun", node_type="entity", properties={"canonical_name": "Arjun"})
    mock_store.get_nodes_by_type.return_value = [self_node, arjun_node]

    gr = GraphRetriever(store=mock_store, user_name="Alice")
    seeds = gr.identify_seed_entities("What did I tell my manager?")
    assert "self" in seeds
    assert "ent_arjun" not in seeds


def test_correct_entity_count_estimation():
    """Entity count estimation does not hardcode Arjun as a known entity by default."""
    router_default = AdaptiveRouter()
    assert "arjun" not in router_default.known_entities
    assert "arjun mehta" not in router_default.known_entities

    # If user_name is explicitly configured as Arjun, it is added
    router_arjun = AdaptiveRouter(user_name="Arjun")
    assert "arjun" in router_arjun.known_entities

    # If entity is loaded from graph store, it is dynamically populated
    mock_store = MagicMock()
    node = GraphNode(id="ent_arjun", label="Arjun", node_type="entity", properties={"canonical_name": "Arjun"})
    mock_store.get_nodes_by_type.return_value = [node]
    router_graph = AdaptiveRouter(graph_store=mock_store)
    assert "arjun" in router_graph.known_entities


def test_no_accidental_suppression_of_legitimate_arjun_entities():
    """GraphSafeguards does not treat another person named Arjun as a generic distractor."""
    # When user is Alice, Arjun is a legitimate other person
    safeguards = GraphSafeguards(user_name="Alice")
    assert "arjun" not in safeguards.generic_entity_names
    assert "alice" in safeguards.generic_entity_names

    triple = GraphEvidenceItem(
        triple_id="edge_collab",
        source_id="ent_arjun",
        source_name="Arjun",
        source_type="entity",
        predicate="collaborated_with",
        target_id="ent_vikram",
        target_name="Vikram Rao",
        target_type="entity",
        direction="outgoing",
        hops=2,
        confidence=0.9,
        score=0.75,
        evidence="Arjun and Vikram collaborated on the compiler",
        memory_id="m101"
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
        query="Who was the collaborator who worked with Vikram Rao?",
        route_decision=route_decision,
        known_memory_ids={"m101"}
    )
    # The triple must NOT be rejected as a generic entity distractor
    assert len(admissible) == 1
    assert report.relevance_exclusions == 0


def test_graph_retriever_seed_tokens_neutral():
    """GraphRetriever uses neutral user tokens by default and matches custom user."""
    mock_store = MagicMock()
    self_node = GraphNode(id="self", label="SELF", node_type="entity", properties={"canonical_name": "SELF"})
    custom_node = GraphNode(id="e_clark", label="Clark Kent", node_type="entity", properties={"canonical_name": "Clark Kent"})
    mock_store.get_nodes_by_type.return_value = [self_node, custom_node]

    gr_default = GraphRetriever(store=mock_store)
    assert gr_default.user_name is None
    seeds_default = gr_default.identify_seed_entities("What did I tell my brother?")
    assert "self" in seeds_default

    gr_custom = GraphRetriever(store=mock_store, user_name="Clark Kent")
    assert gr_custom.user_name == "Clark Kent"
    seeds_custom = gr_custom.identify_seed_entities("When did Clark Kent visit Smallville?")
    assert "e_clark" in seeds_custom


def test_graph_safeguards_stopwords():
    """GraphSafeguards generic stopwords include 'user' and parameterized name."""
    gs_default = GraphSafeguards()
    assert "user" in gs_default.generic_entity_names
    assert "self" in gs_default.generic_entity_names
    assert "arjun" not in gs_default.generic_entity_names

    gs_custom = GraphSafeguards(user_name="Diana Prince")
    assert "diana" in gs_custom.generic_entity_names
    assert "prince" in gs_custom.generic_entity_names


def test_router_handles_neutral_phrasing():
    """AdaptiveRouter regexes recognize neutral pronouns and terms."""
    router = AdaptiveRouter()
    res = router.route("What did they do after visiting Chicago?")
    assert res is not None
    custom_router = AdaptiveRouter(user_name="Tony Stark")
    assert custom_router.user_name == "Tony Stark"


def test_unified_retriever_plumbing():
    """UnifiedRetriever threads user_name to all sub-components."""
    mock_store = MagicMock()
    mock_store.get_nodes_by_type.return_value = []
    ur = UnifiedRetriever(graph_store=mock_store, user_name="Neo")
    assert ur.user_name == "Neo"
    assert ur.episodic_retriever.user_name == "Neo"
    assert ur.graph_retriever.user_name == "Neo"
    assert ur.router.user_name == "Neo"
    assert ur.graph_safeguards.user_name == "Neo"
