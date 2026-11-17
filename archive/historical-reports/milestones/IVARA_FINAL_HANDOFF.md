# IVARA — Final Engineering Handoff & Developer Guide
## Architecture Tour, Operational Workflows & Next-Generation Roadmap
**Milestone:** 3.2 — Batch 5 Final Engineering Milestone  
**Branch:** `research/evaluation-baseline`  
**Date:** November 16, 2026

---

### Welcome to IVARA

IVARA is an evidence-grounded personal memory system designed to preserve, organize, and retrieve human autobiographical interactions across long time horizons.

This document serves as the complete technical handoff for engineering teams maintaining, extending, or deploying IVARA.

---

### 1. Codebase Directory Map

```
IVARA/
├── data/                                 # Production Memory Stores
│   ├── conversations.db                  # SQLite episodic conversation store + FTS5
│   ├── memory_graph.db                   # SQLite knowledge graph (nodes & edges)
│   ├── memories.json                     # Extracted semantic memory records
│   └── entities.json                     # Canonical entity registry
├── src/                                  # Core Application Source Code
│   ├── answering.py                      # Calibrated answer generation & evidence gating
│   ├── memory.py                         # Memory manager entry point
│   ├── retrieval.py                      # FAISS semantic vector search
│   ├── embedding.py                      # SentenceTransformer embeddings
│   ├── chat/
│   │   └── engine.py                     # Multi-turn ChatEngine with episodic wiring
│   ├── conversations/
│   │   └── store.py                      # Persistent episodic storage & FTS5 indexing
│   ├── extraction/
│   │   ├── extractor.py                  # Semantic fact and entity extraction
│   │   ├── pipeline.py                   # Extraction pipeline coordinator
│   │   └── policy.py                     # Memory extraction eligibility heuristics
│   ├── graph/
│   │   ├── schema.py                     # GraphNode, GraphEdge, EntityType definitions
│   │   ├── store.py                      # SQLiteGraphStore implementation
│   │   └── traversal.py                  # Bounded relevance-guided graph traversal
│   ├── ontology/
│   │   ├── vocabulary.py                 # Canonical predicates & aliases
│   │   ├── vocabulary.json               # Schema definitions
│   │   ├── validator.py                  # Open-world relationship validation
│   │   └── relationship_resolver.py      # Entity-relationship disambiguation
│   └── unified_retrieval/                # Unified Multi-Source Retrieval Subsystem
│       ├── router.py                     # 8-category deterministic AdaptiveRouter
│       ├── evidence_selector.py          # Session budgeting & contiguous window expansion
│       ├── graph_safeguards.py           # 3-rule graph validation & temporal gating
│       ├── episodic.py                   # Episodic retrieval engine
│       ├── semantic.py                   # Semantic memory retrieval engine
│       ├── graph.py                      # Graph retrieval engine
│       ├── fusion.py                     # Reciprocal Rank Fusion (legacy fallback)
│       └── pipeline.py                   # UnifiedRetriever coordinator
├── tests/                                # Test Suite (429 passing tests)
│   ├── test_realistic_scenarios.py       # 12 complex personal memory user situations
│   ├── test_calibrated_answering.py      # Answering calibration & evidence gating
│   ├── test_adaptive_router.py           # Deterministic routing category tests
│   ├── test_adaptive_evidence.py         # Evidence selection & window expansion
│   ├── test_episodic_memory.py           # SQLite FTS5 persistence & turn chaining
│   ├── test_open_world_ontology.py       # Open-world ontology validation
│   └── ...                               # Component unit and integration tests
├── evaluation/                           # Benchmark & Statistical Analysis Tools
│   ├── run_final_benchmark.py            # 6-arm benchmark runner on held-out test set
│   ├── rerun_retrieval_250.py            # Fast 250-question retrieval benchmark
│   ├── compile_and_verify_all.py         # Statistical compiler (McNemar, Bootstrap CI)
│   ├── final_benchmark/                  # Output directory with predictions & CSVs
│   └── validation_sprint/                # Held-out dataset & isolated evaluation store
├── scripts/
│   └── audit_performance_resources.py    # Latency, RAM, and scaling projection audit
└── main.py                               # CLI entry point (ask, chat, extract, stats)
```

---

### 2. Quickstart & Operational Commands

#### Running the Test Suite
```bash
# Run full test suite (429 tests)
.venv\Scripts\python -m pytest -q

# Run the 12 realistic personal memory scenarios
.venv\Scripts\python -m pytest tests/test_realistic_scenarios.py -v
```

#### Running the CLI
```bash
# Ask a question using the adaptive multi-source pipeline
.venv\Scripts\python main.py ask "What company does Arjun work for?" --adaptive

# Test out-of-domain unrecorded fact (verifies honest abstention)
.venv\Scripts\python main.py ask "In which city was Arjun born?" --adaptive

# Interactive terminal chat with persistent episodic memory
.venv\Scripts\python main.py chat --adaptive
```

#### Running the Benchmark Suite
```bash
# Execute fast 250-question retrieval evaluation
.venv\Scripts\python evaluation/rerun_retrieval_250.py

# Recompile statistical reports and significance tests
.venv\Scripts\python evaluation/compile_and_verify_all.py
```

---

### 3. Core Architectural Invariants

Developers extending IVARA must uphold the following 5 invariants:

1. **Episodic Primacy:** Every user and assistant conversation turn must be appended to `conversations.db` immediately upon receipt, regardless of whether semantic memories are extracted.
2. **Episodic Backbone Principle:** When constructing evidence context, original conversational turns form the foundational ground truth. Semantic memories and graph edges serve as anchored annotations (`source_turn_id`), never displacing conversational context.
3. **Strict Graph Safeguards:** Knowledge graph edges must NEVER be traversed or included without verified provenance linking to a supporting memory or conversational turn. Unversioned graph edges must never be retrieved for historical-state queries.
4. **Honest Abstention:** When retrieved evidence is insufficient, contradictory, or absent, the system must clearly state that information is unrecorded. It must never invent facts, extrapolate biographical details, or hallucinate citations.
5. **Zero Data Tampering:** Evaluation benchmarks must never touch or alter production database files in `data/`. All evaluation runs must use isolated test fixtures.

---

### 4. Roadmap for Future Milestones

1. **Cross-Session Co-occurrence Expansion:** Enhance the `cross_session_synth` routing policy to enforce broad single-turn sampling across distinct sessions when multi-meeting synthesis is detected.
2. **Automated Offline Graph Construction:** Implement background extraction of entity-relationship triples directly from raw episodic conversation turns to densify `memory_graph.db`.
3. **Coreference Resolution Layer:** Add lightweight coreference resolution to connect pronouns across session boundaries before query embedding generation.
4. **Hierarchical FAISS Indexing (IVF / HNSW):** For large user deployments (>100K turns), transition `IndexFlatIP` to `IndexIVFFlat` or `IndexHNSWFlat` to maintain sub-millisecond search latencies at scale.

