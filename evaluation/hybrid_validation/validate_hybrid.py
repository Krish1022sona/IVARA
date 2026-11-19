"""
Final Hybrid Integration Validation Script
Tests:
- Hybrid Pipeline: Qwen 2.5 14B for Extraction -> Deterministic Resolution & Validation -> Unified Retrieval -> GPT-OSS for Answering
- Baseline Pipeline: GPT-OSS for Extraction -> Deterministic Resolution & Validation -> Unified Retrieval -> GPT-OSS for Answering

Evaluates 6 core representative scenarios:
1. New entity and relationship ingestion
2. Duplicate entities and aliases
3. Temporal updates
4. Contradictory information
5. Multi-hop reasoning
6. Unanswerable questions

Runs on completely isolated temporary databases, preserving production state.
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.conversations.store import ConversationStore
from src.graph.store import SQLiteGraphStore
from src.extraction.extractor import extract_knowledge
from src.extraction.pipeline import process_and_store_memory
from src.unified_retrieval.pipeline import UnifiedRetriever
from src.answering import answer_question

VALIDATION_DIR = WORKSPACE_ROOT / "evaluation" / "hybrid_validation"


def setup_isolated_environment(env_name: str) -> dict[str, Any]:
    """Create a completely isolated set of databases and files for a test arm."""
    env_dir = VALIDATION_DIR / f"isolated_env_{env_name}"
    if env_dir.exists():
        shutil.rmtree(env_dir)
    env_dir.mkdir(parents=True, exist_ok=True)

    conv_db = env_dir / "conversations.db"
    graph_db = env_dir / "memory_graph.db"
    memories_json = env_dir / "memories.json"
    entities_json = env_dir / "entities.json"

    # Initialize empty memories and entities files
    with open(memories_json, "w", encoding="utf-8") as f:
        json.dump([], f)
    with open(entities_json, "w", encoding="utf-8") as f:
        json.dump({}, f)

    conv_store = ConversationStore(conv_db)
    graph_store = SQLiteGraphStore(graph_db)

    return {
        "dir": env_dir,
        "conv_db": conv_db,
        "graph_db": graph_db,
        "memories_json": memories_json,
        "entities_json": entities_json,
        "conv_store": conv_store,
        "graph_store": graph_store
    }


def ingest_dialogue_turn(
    env: dict[str, Any],
    session_title: str,
    speaker: str,
    content: str,
    timestamp: str,
    extraction_model: str
) -> dict[str, Any]:
    """
    Ingest a dialogue turn:
    1. Append to episodic ConversationStore.
    2. Extract knowledge using specified extraction model.
    3. Run deterministic entity resolution and ontology validation.
    4. Store valid entities and relations in isolated SQLiteGraphStore and memories.json.
    """
    conv_store: ConversationStore = env["conv_store"]
    graph_store: SQLiteGraphStore = env["graph_store"]
    memories_json = env["memories_json"]
    entities_json = env["entities_json"]

    conv = conv_store.create_conversation(title=session_title)
    turn = conv_store.add_turn(
        conversation_id=conv.id,
        role="user" if speaker.lower() in ("arjun", "user") else "assistant",
        content=content,
        speaker=speaker
    )

    t0 = time.perf_counter()
    # Ingest through production extraction & storage pipeline
    mem_result = process_and_store_memory(
        text=content,
        timestamp=timestamp,
        source=f"conv_{conv.id}_{turn.turn_id}",
        memory_type="semantic",
        importance=0.8,
        entity_store_path=entities_json,
        memory_store_path=memories_json,
        graph_store=graph_store,
        model=extraction_model
    )
    t1 = time.perf_counter()
    extract_lat_ms = round((t1 - t0) * 1000.0, 1)

    return {
        "turn_id": turn.turn_id,
        "content": content,
        "extraction_latency_ms": extract_lat_ms,
        "entities_count": len(mem_result.get("entities", [])),
        "relationships_count": len(mem_result.get("relationships", [])),
        "extracted_entities": mem_result.get("entities", []),
        "extracted_relationships": mem_result.get("relationships", []),
        "stored_memory_id": mem_result.get("memory_id")
    }


def run_scenarios_for_pipeline(pipeline_name: str, extraction_model: str, answering_model: str) -> dict[str, Any]:
    """Execute all 6 scenarios end-to-end on an isolated environment."""
    print(f"\n==================================================================")
    print(f"RUNNING PIPELINE: {pipeline_name}")
    print(f"Extraction Model: {extraction_model} | Answering Model: {answering_model}")
    print(f"==================================================================")

    env = setup_isolated_environment(pipeline_name)
    results = {}

    # -------------------------------------------------------------
    # Scenario 1: New Entity and Relationship Ingestion
    # -------------------------------------------------------------
    print("\n--- Scenario 1: New Entity & Relationship Ingestion ---")
    s1_text = "Dr. Elena Rostova joined Aetheria Labs in March 2024 as Chief Scientist to direct Project Chronos."
    s1_ingest = ingest_dialogue_turn(
        env, session_title="Aetheria Team Kickoff",
        speaker="Arjun", content=s1_text, timestamp="2024-03-15",
        extraction_model=extraction_model
    )
    print(f"Ingested turn: {s1_ingest['entities_count']} entities, {s1_ingest['relationships_count']} rels in {s1_ingest['extraction_latency_ms']}ms")

    # Inspect graph store
    nodes_s1 = env["graph_store"].get_all_nodes()
    conn_s1 = env["graph_store"]._get_connection()
    edges_s1 = conn_s1.execute("SELECT source_id, edge_type, target_id FROM edges").fetchall()
    print(f"Graph Store: {len(nodes_s1)} nodes, {len(edges_s1)} edges")

    results["scenario_1_ingestion"] = {
        "text": s1_text,
        "ingest_result": s1_ingest,
        "graph_nodes": [{"id": n.id, "type": n.node_type, "label": n.properties.get("name") or n.label} for n in nodes_s1],
        "graph_edges": [{"source": e[0], "relation": e[1], "target": e[2]} for e in edges_s1]
    }

    # -------------------------------------------------------------
    # Scenario 2: Duplicate Entities and Aliases
    # -------------------------------------------------------------
    print("\n--- Scenario 2: Duplicate Entities and Aliases ---")
    s2_text = "Dr. Rostova announced that Aetheria is collaborating directly with Vanguard Robotics on hardware drivers."
    s2_ingest = ingest_dialogue_turn(
        env, session_title="Partnership Sync",
        speaker="Arjun", content=s2_text, timestamp="2024-04-10",
        extraction_model=extraction_model
    )
    nodes_s2 = env["graph_store"].get_all_nodes()
    conn_s2 = env["graph_store"]._get_connection()
    edges_s2 = conn_s2.execute("SELECT source_id, edge_type, target_id FROM edges").fetchall()
    print(f"Graph Store after alias ingestion: {len(nodes_s2)} nodes (+{len(nodes_s2) - len(nodes_s1)}), {len(edges_s2)} edges")

    results["scenario_2_alias_dedup"] = {
        "text": s2_text,
        "ingest_result": s2_ingest,
        "total_nodes": len(nodes_s2),
        "total_edges": len(edges_s2),
        "nodes": [{"id": n.id, "type": n.node_type, "label": n.properties.get("name") or n.label} for n in nodes_s2]
    }

    # -------------------------------------------------------------
    # Scenario 3: Temporal Updates
    # -------------------------------------------------------------
    print("\n--- Scenario 3: Temporal Updates ---")
    s3_text_1 = "In 2023, Arjun used VS Code as his primary code editor for all backend projects."
    s3_ingest_1 = ingest_dialogue_turn(
        env, session_title="Dev Tools Setup 2023",
        speaker="Arjun", content=s3_text_1, timestamp="2023-05-10",
        extraction_model=extraction_model
    )
    s3_text_2 = "In November 2025, Arjun switched completely to Neovim for all software development and abandoned VS Code."
    s3_ingest_2 = ingest_dialogue_turn(
        env, session_title="Dev Tools Migration 2025",
        speaker="Arjun", content=s3_text_2, timestamp="2025-11-20",
        extraction_model=extraction_model
    )

    # Answering query on temporal update
    q3 = "What code editor does Arjun currently use for software development, and what did he use previously in 2023?"
    retriever = UnifiedRetriever(
        conversation_store=env["conv_store"],
        memories_path=env["memories_json"],
        graph_store=env["graph_store"],
        use_adaptive_routing=True
    )
    t0 = time.perf_counter()
    ans3 = answer_question(
        query=q3, k=5, min_score=0.30,
        unified_retriever=retriever,
        enable_episodic=True, enable_semantic=True, enable_graph_channel=True,
        enable_temporal=True, use_adaptive_routing=True,
        model=answering_model
    )
    t1 = time.perf_counter()
    ans3_lat_ms = round((t1 - t0) * 1000.0, 1)

    print(f"Query 3: '{q3}'")
    print(f"Status: {ans3.get('status')} | Latency: {ans3_lat_ms}ms")
    print(f"Answer: {ans3.get('answer')}")
    print(f"Citations: {[c['memory_id'] for c in ans3.get('citations', [])]}")

    results["scenario_3_temporal"] = {
        "query": q3,
        "answer_status": ans3.get("status"),
        "answer": ans3.get("answer"),
        "citations": ans3.get("citations", []),
        "latency_ms": ans3_lat_ms
    }

    # -------------------------------------------------------------
    # Scenario 4: Contradictory Information
    # -------------------------------------------------------------
    print("\n--- Scenario 4: Contradictory Information ---")
    s4_text_1 = "Marcus Vance stated that Project Hyperion was officially approved for production deployment in July 2025."
    s4_ingest_1 = ingest_dialogue_turn(
        env, session_title="Project Hyperion Meeting",
        speaker="Arjun", content=s4_text_1, timestamp="2025-06-01",
        extraction_model=extraction_model
    )
    s4_text_2 = "Executive leadership officially cancelled Project Hyperion on June 12, 2025 due to regulatory concerns and no code was deployed."
    s4_ingest_2 = ingest_dialogue_turn(
        env, session_title="Leadership Announcement",
        speaker="Arjun", content=s4_text_2, timestamp="2025-06-13",
        extraction_model=extraction_model
    )

    q4 = "Was Project Hyperion deployed to production or cancelled?"
    t0 = time.perf_counter()
    ans4 = answer_question(
        query=q4, k=5, min_score=0.30,
        unified_retriever=retriever,
        enable_episodic=True, enable_semantic=True, enable_graph_channel=True,
        enable_temporal=True, use_adaptive_routing=True,
        model=answering_model
    )
    t1 = time.perf_counter()
    ans4_lat_ms = round((t1 - t0) * 1000.0, 1)

    print(f"Query 4: '{q4}'")
    print(f"Status: {ans4.get('status')} | Latency: {ans4_lat_ms}ms")
    print(f"Answer: {ans4.get('answer')}")
    print(f"Citations: {[c['memory_id'] for c in ans4.get('citations', [])]}")

    results["scenario_4_contradiction"] = {
        "query": q4,
        "answer_status": ans4.get("status"),
        "answer": ans4.get("answer"),
        "citations": ans4.get("citations", []),
        "latency_ms": ans4_lat_ms
    }

    # -------------------------------------------------------------
    # Scenario 5: Multi-Hop Reasoning
    # -------------------------------------------------------------
    print("\n--- Scenario 5: Multi-Hop Reasoning ---")
    s5_text_1 = "David Thorne is the engineering manager leading the VectorFlow project at CloudMatrix Inc."
    s5_ingest_1 = ingest_dialogue_turn(
        env, session_title="CloudMatrix Architecture Sync",
        speaker="Arjun", content=s5_text_1, timestamp="2024-05-15",
        extraction_model=extraction_model
    )
    s5_text_2 = "The VectorFlow project utilizes Neo4j as its core graph persistence database."
    s5_ingest_2 = ingest_dialogue_turn(
        env, session_title="VectorFlow Database Selection",
        speaker="Arjun", content=s5_text_2, timestamp="2024-05-20",
        extraction_model=extraction_model
    )

    q5 = "Which database system is utilized by the project managed by David Thorne at CloudMatrix?"
    t0 = time.perf_counter()
    ans5 = answer_question(
        query=q5, k=5, min_score=0.30,
        unified_retriever=retriever,
        enable_episodic=True, enable_semantic=True, enable_graph_channel=True,
        enable_temporal=True, use_adaptive_routing=True,
        model=answering_model
    )
    t1 = time.perf_counter()
    ans5_lat_ms = round((t1 - t0) * 1000.0, 1)

    print(f"Query 5: '{q5}'")
    print(f"Status: {ans5.get('status')} | Latency: {ans5_lat_ms}ms")
    print(f"Answer: {ans5.get('answer')}")
    print(f"Citations: {[c['memory_id'] for c in ans5.get('citations', [])]}")

    results["scenario_5_multihop"] = {
        "query": q5,
        "answer_status": ans5.get("status"),
        "answer": ans5.get("answer"),
        "citations": ans5.get("citations", []),
        "latency_ms": ans5_lat_ms
    }

    # -------------------------------------------------------------
    # Scenario 6: Unanswerable Question
    # -------------------------------------------------------------
    print("\n--- Scenario 6: Unanswerable Question ---")
    q6 = "What is Arjun's mother's maiden name, and what was his SAT math score?"
    t0 = time.perf_counter()
    ans6 = answer_question(
        query=q6, k=5, min_score=0.30,
        unified_retriever=retriever,
        enable_episodic=True, enable_semantic=True, enable_graph_channel=True,
        enable_temporal=True, use_adaptive_routing=True,
        model=answering_model
    )
    t1 = time.perf_counter()
    ans6_lat_ms = round((t1 - t0) * 1000.0, 1)

    print(f"Query 6: '{q6}'")
    print(f"Status: {ans6.get('status')} | Latency: {ans6_lat_ms}ms")
    print(f"Answer: {ans6.get('answer')}")
    print(f"Citations: {[c['memory_id'] for c in ans6.get('citations', [])]}")

    results["scenario_6_unanswerable"] = {
        "query": q6,
        "answer_status": ans6.get("status"),
        "answer": ans6.get("answer"),
        "citations": ans6.get("citations", []),
        "latency_ms": ans6_lat_ms
    }

    return results


def main():
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("STARTING HYBRID INTEGRATION VALIDATION CAMPAIGN")
    print("==================================================================")

    # 1. Run Baseline: GPT-OSS Extraction + GPT-OSS Answering
    baseline_results = run_scenarios_for_pipeline(
        pipeline_name="baseline",
        extraction_model="gpt-oss:latest",
        answering_model="gpt-oss:latest"
    )

    # 2. Run Hybrid: Qwen 2.5 Extraction + GPT-OSS Answering
    hybrid_results = run_scenarios_for_pipeline(
        pipeline_name="hybrid",
        extraction_model="qwen2.5:14b",
        answering_model="gpt-oss:latest"
    )

    final_report = {
        "baseline_pipeline": {
            "extraction_model": "gpt-oss:latest",
            "answering_model": "gpt-oss:latest",
            "scenarios": baseline_results
        },
        "hybrid_pipeline": {
            "extraction_model": "qwen2.5:14b",
            "answering_model": "gpt-oss:latest",
            "scenarios": hybrid_results
        }
    }

    report_path = VALIDATION_DIR / "hybrid_validation_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nValidation campaign complete! Saved results to {report_path}")


if __name__ == "__main__":
    main()
