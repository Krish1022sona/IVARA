"""
Unit, integration, and regression tests for Query Decomposition and Config E.

Covers:
- Simple questions bypassing decomposition.
- Complex questions producing valid, bounded subquestions.
- Malformed JSON and model failures triggering safe fallback.
- Duplicate and irrelevant subquestions being removed.
- Retrieval executed independently for each accepted subquestion.
- Evidence aggregation preserving source and chunk IDs.
- Duplicate evidence deduplicated without losing provenance.
- Graph traversal obeying hop limits.
- Unsupported subquestions not answered with fabricated facts.
- Citation validation and partial-answer behavior.
- Gold-answer isolation.
- Existing A-D behavior remaining unchanged.
"""

import json
import pytest
from pathlib import Path

from src.query.decomposition import QueryDecomposer, DecomposedQuery
from src.graph.store import SQLiteGraphStore
from src.graph.retrieval import GraphAssistedRetriever
from src.answering import answer_question

BASE_DIR = Path(__file__).resolve().parent.parent
GRAPH_PATH = BASE_DIR / "evaluation" / "long_context" / "multi_episode" / "graph.db"
MEMORIES_PATH = BASE_DIR / "evaluation" / "long_context" / "multi_episode" / "memories.json"


@pytest.fixture
def graph_store():
    assert GRAPH_PATH.exists(), f"Graph DB not found at {GRAPH_PATH}"
    return SQLiteGraphStore(GRAPH_PATH)


@pytest.fixture
def retriever(graph_store):
    return GraphAssistedRetriever(graph_store)


# 1. Simple questions bypass decomposition
def test_simple_questions_bypass_decomposition():
    decomposer = QueryDecomposer()
    simple_queries = [
        "What is the nickname of Dr. Elena Rostova?",
        "On what date did Sarah Lin join Aetheria Labs?",
        "Who founded Vanguard Robotics?"
    ]
    for q in simple_queries:
        res = decomposer.decompose(q)
        assert not res.is_decomposed
        assert res.strategy == "bypassed_simple"
        assert res.subquestions == [q]
        assert res.original_query == q


# 2. Complex questions produce valid, bounded subquestions
def test_complex_questions_produce_valid_bounded_subquestions():
    mock_payload = {
        "should_decompose": True,
        "reason": "Two distinct timeline events queried",
        "subquestions": [
            "1. When was Project Chronos started?",
            "2. When was the final production deployment of Project Chronos completed?",
            "3. Extra subquestion that should be kept if under max"
        ]
    }
    decomposer = QueryDecomposer(max_subquestions=2, llm_caller=lambda p, s: json.dumps(mock_payload))
    q = "When was Project Chronos started, and when was its final production deployment completed?"
    res = decomposer.decompose(q)

    assert res.is_decomposed
    assert res.strategy == "llm_decomposed"
    assert len(res.subquestions) <= 2
    assert res.subquestions[0] == "When was Project Chronos started?"
    assert res.subquestions[1] == "When was the final production deployment of Project Chronos completed?"


# 3. Malformed JSON and model failures trigger safe fallback
def test_malformed_json_and_model_failure_trigger_safe_fallback():
    # Model throws exception
    def failing_caller(p, s):
        raise ConnectionError("Ollama daemon unavailable")

    decomposer_fail = QueryDecomposer(llm_caller=failing_caller)
    q = "When was Chronos started and when did it finish?"
    res1 = decomposer_fail.decompose(q)
    assert not res1.is_decomposed
    assert res1.strategy == "fallback_original"
    assert res1.subquestions == [q]

    # Model returns invalid / garbage JSON
    decomposer_garbage = QueryDecomposer(llm_caller=lambda p, s: "MALFORMED_NON_JSON")
    res2 = decomposer_garbage.decompose(q)
    assert not res2.is_decomposed
    assert res2.strategy == "fallback_original"
    assert res2.subquestions == [q]


# 4. Duplicate and irrelevant subquestions removed
def test_duplicate_and_irrelevant_subquestions_removed():
    # Duplicate subquestions
    dupe_payload = {
        "should_decompose": True,
        "reason": "Duplicates",
        "subquestions": [
            "When was Project Chronos started?",
            "when was project chronos started?",
            "When was Project Chronos started?"
        ]
    }
    d_dupe = QueryDecomposer(llm_caller=lambda p, s: json.dumps(dupe_payload))
    res_dupe = d_dupe.decompose("When was Chronos started and when did it finish?")
    # Only 1 unique subquestion remained -> rejected -> fallback
    assert not res_dupe.is_decomposed
    assert res_dupe.strategy == "validation_rejected"

    # Irrelevant subquestions lacking content words
    irrelevant_payload = {
        "should_decompose": True,
        "reason": "Off-topic",
        "subquestions": [
            "What is the weather in Tokyo?",
            "Who won the 2026 World Cup championship?"
        ]
    }
    d_irrel = QueryDecomposer(llm_caller=lambda p, s: json.dumps(irrelevant_payload))
    res_irrel = d_irrel.decompose("When was Chronos started and when did it finish?")
    assert not res_irrel.is_decomposed
    assert res_irrel.strategy == "validation_rejected"


# 5. Retrieval executed independently for each accepted subquestion
def test_retrieval_executed_independently_per_subquestion(retriever):
    subqs = [
        "When was Project Chronos started?",
        "When was Project Chronos delivered?"
    ]
    decomposed = DecomposedQuery(
        original_query="When was Chronos started and delivered?",
        is_decomposed=True,
        subquestions=subqs,
        reason="Test multi-part",
        strategy="llm_decomposed"
    )

    res = retriever.retrieve_decomposed("When was Chronos started and delivered?", decomposed, k=3)
    diag = res.get("diagnostics", {})
    assert diag.get("retrieval_calls") == 2
    assert len(diag.get("subquery_results", [])) == 2
    assert diag["subquery_results"][0]["subquestion"] == subqs[0]
    assert diag["subquery_results"][1]["subquestion"] == subqs[1]


# 6. Evidence aggregation preserves source and chunk IDs
def test_evidence_aggregation_preserves_source_and_chunk_ids(retriever):
    subqs = [
        "What is Elena Rostova's role?",
        "What is Elena Ramos's role?"
    ]
    decomposed = DecomposedQuery(
        original_query="Compare Elena Rostova and Elena Ramos",
        is_decomposed=True,
        subquestions=subqs,
        reason="Disambiguation",
        strategy="llm_decomposed"
    )

    res = retriever.retrieve_decomposed("Compare Elena Rostova and Elena Ramos", decomposed, k=3)
    assert res["has_sufficient_evidence"]
    for ev in res["direct_evidence"]:
        assert "memory_id" in ev
        assert "chunk_id" in ev
        assert ev["chunk_id"].startswith("chunk_")
        assert ev["memory_id"].startswith("m")
        assert "matched_subqueries" in ev
        assert len(ev["matched_subqueries"]) >= 1


# 7. Duplicate evidence deduplicated without losing provenance
def test_duplicate_evidence_deduplicated_without_losing_provenance(retriever):
    # Both subquestions target Episode 2 where Vanguard and Elena Ramos are discussed
    subqs = [
        "Who is Elena Ramos at Vanguard Robotics?",
        "What sensors did Elena Ramos design for Vanguard Robotics?"
    ]
    decomposed = DecomposedQuery(
        original_query="What was Elena Ramos's role and design at Vanguard?",
        is_decomposed=True,
        subquestions=subqs,
        reason="Entity role and product",
        strategy="llm_decomposed"
    )

    res = retriever.retrieve_decomposed("What was Elena Ramos's role and design at Vanguard?", decomposed, k=3)
    chunk_ids = [c["chunk_id"] for c in res["retrieved_chunks"]]
    # Every chunk must be unique (no duplicate chunk IDs)
    assert len(chunk_ids) == len(set(chunk_ids))

    # Any chunk retrieved by both subquestions must track both in matched_subqueries
    co_occurred = [c for c in res["retrieved_chunks"] if len(c.get("matched_subqueries", [])) > 1]
    if co_occurred:
        assert set(subqs).issubset(set(co_occurred[0]["matched_subqueries"]))


# 8. Graph traversal obeys hop limits
def test_graph_traversal_obeys_hop_limits(retriever):
    # Subquestion with relational keyword triggers selective graph traversal
    subqs = [
        "Who mentored Sarah Lin at Aetheria Labs?",
        "What programming language was used by Sarah Lin's mentor?"
    ]
    decomposed = DecomposedQuery(
        original_query="What language did Sarah Lin's mentor use?",
        is_decomposed=True,
        subquestions=subqs,
        reason="Relational chain",
        strategy="llm_decomposed"
    )

    res = retriever.retrieve_decomposed("What language did Sarah Lin's mentor use?", decomposed, k=3)
    diag = res.get("diagnostics", {})
    # Selective 1-hop traversal should record bounded hops
    assert diag.get("graph_hops_total", 0) <= 20
    assert len(res.get("inferred_evidence", [])) <= 5


# 9. Unsupported subquestions not answered with fabricated facts
def test_unsupported_subquestions_not_answered_with_fabrication(graph_store):
    # Completely unsupported unanswerable query
    fake_subqs = [
        "What was the budget allocated to Project Chronos in euros?",
        "Who was the chief financial officer of Aetheria Labs?"
    ]
    def mock_caller(p, s):
        if "decomposition" in s.lower() or "decompose" in p.lower():
            return json.dumps({
                "should_decompose": True,
                "reason": "Two finance questions",
                "subquestions": fake_subqs
            })
        return json.dumps({
            "status": "insufficient_evidence",
            "answer": "I don't have sufficient recorded memory to answer this question.",
            "cited_memory_ids": [],
            "limitations": "No budget or CFO information recorded."
        })

    ans = answer_question(
        query="What was Chronos's budget and who was the CFO?",
        k=3,
        use_graph=True,
        graph_store=graph_store,
        use_decomposition=True,
        llm_caller=mock_caller
    )
    # The system safely abstains with insufficient_evidence and zero citations
    assert ans["status"] == "insufficient_evidence"
    assert ans["citations"] == []


# 10. Citation validation and partial-answer behavior
def test_citation_validation_and_partial_answer(graph_store):
    # LLM caller that provides one valid citation ('m002') and one fake citation ('m999')
    def mock_answer_caller(p, s):
        if "Decompose" in p or "decomposition" in s.lower():
            return json.dumps({
                "should_decompose": True,
                "reason": "Multi-part",
                "subquestions": [
                    "What is Elena Rostova's nickname?",
                    "What company did Julian Mercer represent?"
                ]
            })
        return json.dumps({
            "status": "answered",
            "answer": "Dr. Elena Rostova is nicknamed The Oracle, and Julian Mercer represented Vanguard Robotics.",
            "cited_memory_ids": ["m002", "m999"],
            "limitations": None
        })

    ans = answer_question(
        query="What is Elena Rostova's nickname, and what company did Julian Mercer represent?",
        k=3,
        use_graph=True,
        graph_store=graph_store,
        use_decomposition=True,
        llm_caller=mock_answer_caller
    )

    # Valid citation m002 accepted
    cited_ids = [c["memory_id"] for c in ans["citations"]]
    assert "m002" in cited_ids
    # Fabricated citation m999 rejected
    assert "m999" not in cited_ids
    assert "m999" in ans.get("rejected_citation_ids", [])


# 11. Gold-answer isolation
def test_gold_answer_isolation():
    # Verify signature and code: decomposer and retriever accept ONLY user query
    decomposer = QueryDecomposer()
    # decomposer.decompose only accepts query
    import inspect
    sig_dec = inspect.signature(decomposer.decompose)
    assert list(sig_dec.parameters.keys()) == ["query"]

    # GraphAssistedRetriever.retrieve_decomposed only accepts query, decomposed, and retrieval hyperparams
    retriever_methods = inspect.getmembers(GraphAssistedRetriever, predicate=inspect.isfunction)
    method_dict = dict(retriever_methods)
    sig_ret = inspect.signature(method_dict["retrieve_decomposed"])
    param_names = list(sig_ret.parameters.keys())
    assert "gold_answer" not in param_names
    assert "rubric" not in param_names
    assert "required_episodes" not in param_names


# 12. Existing A-D behavior remains unchanged
def test_existing_ad_behavior_unchanged(graph_store):
    # Calling answer_question with default use_decomposition=False must use standard retrieval
    ans = answer_question(
        query="What is the nickname of Dr. Elena Rostova?",
        k=3,
        use_graph=True,
        graph_store=graph_store,
        use_decomposition=False,
        llm_caller=lambda p, s: json.dumps({
            "status": "answered",
            "answer": "Dr. Elena Rostova is nicknamed The Oracle.",
            "cited_memory_ids": ["m002"],
            "limitations": None
        })
    )
    assert ans["status"] == "answered"
    assert "The Oracle" in ans["answer"]
    assert ans["retrieved_context"]["diagnostics"]["mode"] != "decomposed_hybrid_graph"
