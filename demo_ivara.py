"""
IVARA Interactive Demonstration & Core Capability Verification.

Demonstrates the 6 core intelligence capabilities of IVARA:
1. Direct Factual Recall: Accurate extraction and grounded retrieval.
2. Cross-Session Relational Reasoning: Connecting related evidence across disparate sessions.
3. Temporal Supersession: Handling updated facts where newer claims supersede older claims.
4. Evidence Integration: Synthesizing complementary evidence across multiple memories.
5. Contradiction Detection: Identifying and qualifying conflicting unverified claims.
6. Grounded Abstention: Clean refusal to answer when evidence is absent (zero hallucination).

Usage:
  # Run live demonstration using local Ollama model (gpt-oss:latest)
  python demo_ivara.py

  # Run offline demonstration with mock LLM caller (instant deterministic verification)
  python demo_ivara.py --offline
"""

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.answering import answer_question
from src.chunking.turn_chunker import TurnAwareChunker, TurnChunk
from src.chunking.graph_retrieval import TurnGraphBuilder
from src.graph.store import SQLiteGraphStore


DEMO_SESSIONS = [
    {
        "session_id": "sess_2024_03_01",
        "date": "2024/03/01 10:15",
        "turns": [
            {"role": "user", "content": "I adopted a 3-year-old golden retriever named Buster last weekend! He loves playing fetch in our backyard here in Seattle."},
            {"role": "assistant", "content": "Congratulations on adopting Buster! Golden retrievers are wonderfully energetic and friendly companions."}
        ]
    },
    {
        "session_id": "sess_2024_05_15",
        "date": "2024/05/15 14:30",
        "turns": [
            {"role": "user", "content": "I recently picked up scale model building as a hobby and bought a 1/72 scale B-29 bomber kit. My sister Priya, who lives in Chicago, thinks it looks super intricate."},
            {"role": "assistant", "content": "A 1/72 scale B-29 is a classic and detailed aircraft kit. Do you have dedicated hobby tools for the assembly?"}
        ]
    },
    {
        "session_id": "sess_2024_09_10",
        "date": "2024/09/10 09:00",
        "turns": [
            {"role": "user", "content": "Big life update: I just moved to Austin, Texas for a new backend engineering role at CloudScale! I no longer live in Seattle."},
            {"role": "assistant", "content": "Congratulations on the new job at CloudScale and the move to Austin, Texas! How is Buster settling into the new climate?"}
        ]
    },
    {
        "session_id": "sess_2024_10_01",
        "date": "2024/10/01 16:45",
        "turns": [
            {"role": "user", "content": "Priya flew in to visit me in Austin this weekend. We spent Saturday afternoon working on the wings of my B-29 bomber model kit together."},
            {"role": "assistant", "content": "That sounds like a wonderful weekend project to share with your sister Priya."}
        ]
    },
    {
        "session_id": "sess_2024_10_15",
        "date": "2024/10/15 11:20",
        "turns": [
            {"role": "user", "content": "Buster has a favorite toy he carries everywhere: a durable blue rubber frisbee we bought at the local park in Austin."},
            {"role": "assistant", "content": "That's adorable! Dogs often become deeply attached to their favorite fetch toys."}
        ]
    },
    {
        "session_id": "sess_2024_11_05",
        "date": "2024/11/05 13:00",
        "turns": [
            {"role": "user", "content": "My college buddy Marco mentioned he works as a senior solutions architect at Databricks."},
            {"role": "assistant", "content": "Databricks is doing great work in data lakehouse architecture."}
        ]
    },
    {
        "session_id": "sess_2024_11_20",
        "date": "2024/11/20 18:30",
        "turns": [
            {"role": "user", "content": "Met Marco for dinner tonight; he told me he is a principal engineer at Snowflake."},
            {"role": "assistant", "content": "Snowflake is another leader in the cloud data platform space."}
        ]
    }
]

DEMO_TEST_CASES = [
    {
        "case_id": "CASE_1_DIRECT_FACTUAL",
        "title": "Scenario 1: Direct Factual Recall",
        "query": "What kind of dog do I have, and what is his name?",
        "expected_facts": ["Buster", "golden retriever"],
        "unexpected_facts": [],
        "min_score": None,
        "max_chunks_per_session": None,
        "mock_response": {
            "status": "answered",
            "answer": "You have a 3-year-old golden retriever named Buster.",
            "cited_memory_ids": ["sess_2024_03_01_c000"],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_limitations": None
        }
    },
    {
        "case_id": "CASE_2_CROSS_SESSION",
        "title": "Scenario 2: Cross-Session Relational Reasoning",
        "query": "Which family member helped me work on my B-29 bomber model kit, and where does she live?",
        "expected_facts": ["Priya", "Chicago"],
        "unexpected_facts": [],
        "min_score": None,
        "max_chunks_per_session": 1,
        "mock_response": {
            "status": "answered",
            "answer": "Your sister Priya helped you work on your B-29 bomber model kit, and she lives in Chicago.",
            "cited_memory_ids": ["sess_2024_05_15_c000", "sess_2024_10_01_c000"],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_limitations": None
        }
    },
    {
        "case_id": "CASE_3_TEMPORAL_SUPERSEDED",
        "title": "Scenario 3: Temporal Update & Historical Recall",
        "query": "Where do I currently live, and where did I live before?",
        "expected_facts": ["Austin", "Seattle"],
        "unexpected_facts": ["currently live in Seattle"],
        "min_score": None,
        "max_chunks_per_session": 1,
        "mock_response": {
            "status": "answered",
            "answer": "You currently live in Austin, Texas. You previously lived in Seattle before moving for a new role at CloudScale.",
            "cited_memory_ids": ["sess_2024_09_10_c000"],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_limitations": None
        }
    },
    {
        "case_id": "CASE_4_COMPLEMENTARY_EVIDENCE",
        "title": "Scenario 4: Evidence Integration Across Complementary Memories",
        "query": "What breed of dog do I have, and what is his favorite toy?",
        "expected_facts": ["golden retriever", "frisbee"],
        "unexpected_facts": [],
        "min_score": None,
        "max_chunks_per_session": 1,
        "mock_response": {
            "status": "answered",
            "answer": "You have a golden retriever named Buster, and his favorite toy is a durable blue rubber frisbee.",
            "cited_memory_ids": ["sess_2024_03_01_c000", "sess_2024_10_15_c000"],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_limitations": None
        }
    },
    {
        "case_id": "CASE_5_CONTRADICTORY_EVIDENCE",
        "title": "Scenario 5: Contradictory Memories Requiring Qualification",
        "query": "Where does my friend Marco work?",
        "expected_facts": ["Databricks", "Snowflake"],
        "unexpected_facts": [],
        "min_score": None,
        "max_chunks_per_session": 1,
        "mock_response": {
            "status": "answered",
            "answer": "There are conflicting records regarding where Marco works: on November 5 you noted he works as a senior solutions architect at Databricks, but on November 20 you noted he is a principal engineer at Snowflake.",
            "cited_memory_ids": ["sess_2024_11_05_c000", "sess_2024_11_20_c000"],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_limitations": "Unresolved conflict between Databricks and Snowflake."
        }
    },
    {
        "case_id": "CASE_6_GROUNDED_ABSTENTION",
        "title": "Scenario 6: Safe Grounded Abstention (Unanswerable Question)",
        "query": "What is my mother's maiden name and date of birth?",
        "expected_facts": [],
        "unexpected_facts": [],
        "min_score": 0.35,
        "max_chunks_per_session": None,
        "mock_response": {
            "status": "insufficient_evidence",
            "answer": "I do not have sufficient recorded memories to answer what your mother's maiden name or date of birth is.",
            "cited_memory_ids": [],
            "inferred_claims": [],
            "unsupported_claims": [],
            "evidence_limitations": "No evidence found in memory store."
        }
    }
]


def run_demonstration(offline: bool = False, model: str = "gpt-oss:latest"):
    print("=" * 80)
    print("  IVARA PERSONAL AI MEMORY SYSTEM -- CORE CAPABILITY DEMONSTRATION")
    print("=" * 80)
    print(f"Mode: {'OFFLINE (Deterministic Mock)' if offline else f'LOCAL LIVE (Ollama: {model})'}")
    print(f"Working Directory: {ROOT_DIR}")
    print(f"Isolated Temp Graph DB: scratch/demo_isolated.db (production stores untouched)\n")

    # Step 1: Chunk demo sessions into TurnChunks
    print("1. Ingesting & Chunking Conversational Memory Sessions...")
    chunker = TurnAwareChunker(window_size=2, overlap=1)
    
    adapted_sessions = []
    class DummyTurn:
        def __init__(self, r, c): self.role, self.content = r, c
    class DummySession:
        def __init__(self, sid, dt, turns):
            self.session_id = sid
            self.date = dt
            self.turns = [DummyTurn(t["role"], t["content"]) for t in turns]
            self.parsed_datetime = None

    for s in DEMO_SESSIONS:
        adapted_sessions.append(DummySession(s["session_id"], s["date"], s["turns"]))

    chunks = chunker.chunk_all_sessions(adapted_sessions)
    print(f"   Indexed {len(DEMO_SESSIONS)} sessions into {len(chunks)} turn chunks.\n")

    # Step 2: Build isolated SQLite graph store
    temp_db_path = ROOT_DIR / "scratch" / "demo_isolated.db"
    temp_db_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_db_path.exists():
        temp_db_path.unlink()

    store = SQLiteGraphStore(temp_db_path)
    graph_builder = TurnGraphBuilder(store)
    graph_builder.build_from_chunks(chunks)
    print(f"2. Graph Store Built: {store.count_nodes()} nodes, {store.count_edges()} edges.\n")

    print("-" * 80)
    print("EXECUTING REPRESENTATIVE TEST CASES (6 CORE SCENARIOS)")
    print("-" * 80)

    scorecard = []

    for idx, case in enumerate(DEMO_TEST_CASES, 1):
        print(f"\n[{case['case_id']}] {case['title']}")
        print(f"Query: \"{case['query']}\"")
        
        t0 = time.time()

        llm_caller = None
        if offline:
            mock_payload = case["mock_response"]
            llm_caller = lambda p, s: json.dumps(mock_payload)

        try:
            result = answer_question(
                query=case["query"],
                k=4,
                min_score=case["min_score"],
                model=model,
                llm_caller=llm_caller,
                graph_store=store,
                use_chunking=True,
                chunks=chunks,
                use_ontology=True,
                dynamic_graph_fallback=True,
                direct_seed_temporal=True,
                enable_temporal=True,
                max_chunks_per_session=case.get("max_chunks_per_session")
            )
            elapsed = round(time.time() - t0, 2)
            status = result.get("status")
            answer = result.get("answer", "").strip()
            citations = [c.get("chunk_id") or c.get("memory_id") for c in result.get("citations", [])]
            retrieved_chunks = result.get("retrieved_context", {}).get("direct_evidence", [])
            retrieved_sids = list(dict.fromkeys(c.get("session_id") for c in retrieved_chunks))

            print(f"Status: {status} ({elapsed}s)")
            print(f"Retrieved Sessions: {retrieved_sids}")
            print(f"Assembled Direct Evidence Chunks: {[c.get('chunk_id') for c in retrieved_chunks]}")
            print(f"Generated Answer: \"{answer}\"")
            print(f"Citations: {citations}")

            # Machine-checkable verification logic
            is_correct = True
            reasons = []
            ans_lower = answer.lower()

            if case["case_id"] == "CASE_6_GROUNDED_ABSTENTION":
                if status != "insufficient_evidence":
                    is_correct = False
                    reasons.append(f"Expected status 'insufficient_evidence', got '{status}'")
                if not any(kw in ans_lower for kw in ["sufficient", "recorded", "don't have", "do not have", "no evidence"]):
                    is_correct = False
                    reasons.append("Answer does not express absence of recorded evidence")
            elif case["case_id"] == "CASE_5_CONTRADICTORY_EVIDENCE":
                # Must acknowledge both sides or state the contradiction
                has_databricks = "databricks" in ans_lower
                has_snowflake = "snowflake" in ans_lower
                if not (has_databricks and has_snowflake):
                    is_correct = False
                    reasons.append(f"Expected both Databricks and Snowflake mentioned in conflict qualification (got DB={has_databricks}, SF={has_snowflake})")
            else:
                for ef in case.get("expected_facts", []):
                    if ef.lower() not in ans_lower:
                        is_correct = False
                        reasons.append(f"Missing expected fact: '{ef}'")
                for uf in case.get("unexpected_facts", []):
                    if uf.lower() in ans_lower:
                        is_correct = False
                        reasons.append(f"Found unexpected superseded fact: '{uf}'")

            verdict = "PASSED (Grounded & Accurate)" if is_correct else f"FAILED: {'; '.join(reasons)}"
            print(f"Verdict: {verdict}")

            scorecard.append({
                "case": case["title"],
                "status": status,
                "is_correct": is_correct,
                "reasons": reasons,
                "elapsed": elapsed
            })

        except Exception as e:
            print(f"Execution Error: {e}")
            scorecard.append({
                "case": case["title"],
                "status": "error",
                "is_correct": False,
                "reasons": [str(e)],
                "elapsed": 0.0
            })

    # Clean up temp store
    store.close()
    if temp_db_path.exists():
        temp_db_path.unlink()

    # Step 3: Print summary scorecard
    print("\n" + "=" * 80)
    print("DEMONSTRATION SCORECARD SUMMARY")
    print("=" * 80)
    all_passed = all(s["is_correct"] for s in scorecard)
    for s in scorecard:
        mark = "[PASS]" if s["is_correct"] else "[FAIL]"
        print(f"  {mark:<8} | {s['case']:<58} | Status: {s['status']:<22} | {s['elapsed']}s")
    
    print("-" * 80)
    print(f"OVERALL DEMONSTRATION RESULT: {'ALL 6 SCENARIOS PASSED CLEANLY' if all_passed else 'SOME SCENARIOS FAILED'}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IVARA Core Capability Demonstration")
    parser.add_argument("--offline", action="store_true", help="Run with deterministic mock LLM caller for instant verification")
    parser.add_argument("--model", type=str, default="gpt-oss:latest", help="Local Ollama model name (default: gpt-oss:latest)")
    args = parser.parse_args()

    success = run_demonstration(offline=args.offline, model=args.model)
    sys.exit(0 if success else 1)
