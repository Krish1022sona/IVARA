# IVARA — Intelligent Verifiable Agentic Retrieval Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Compute: 100% Local](https://img.shields.io/badge/Compute-100%25%20Local-orange.svg)]()

IVARA is a local-first AI personal memory assistant and associative-memory research project built with Python, SQLite, FAISS, `sentence-transformers` (`all-MiniLM-L6-v2`), and Ollama.

---

## 1. Foundational Vision & Engineering Philosophy

- **Long-Term Research Vision (Associative Memory):** Beyond basic vector lookup, IVARA aims to implement human-inspired associative memory—discovering relationships across independently stored experiences, facts, entities, and events, traversing those relationships to surface indirectly relevant context, and deriving evidence-grounded conclusions. The knowledge graph and ontology layer remain central long-term research directions.
- **Product-First Engineering Philosophy:** IVARA is primarily a practical personal AI assistant. We prioritize reliable, natural conversation and accurate memory over academic novelty, adopt proven open-source solutions where they demonstrably outperform custom code (evaluating systems such as HippoRAG, GraphRAG, and Mem0 in future phases), and treat benchmarks as engineering verification tools.

---

## 2. Current Capabilities & Known Limitations

### Implemented Today
- **Local CLI & Persistence (`main.py`):** Interactive chat (`chat`), direct memory query (`ask`), conversation management (`chats`), and memory CRUD (`memories`).
- **Episodic & Semantic Storage:** SQLite conversation/episodic turn store with FTS5 (`data/conversations.db`), local JSON memory store (`data/memories.json`, untracked at runtime), and SQLite entity/relation graph (`data/memory_graph.db`).
- **Benchmark-Verified Batch Retrieval (`Config D`):** Sliding-window turn chunking ($W=3, O=1$), hybrid dense + BM25 search, Flat MMR ($\lambda=0.70$, $\le 2$ chunks/session), deterministic calendar arithmetic, and ID-based citation validation (`58.40%` on LongMemEval-S).

### Important Current Limitations (Active Remediation in Progress)
- **Interactive Chat vs. Benchmark Split:** `main.py chat` currently uses unchunked `UnifiedRetriever` rather than Candidate B's `ChunkRetriever`, unconditionally triggers memory QA before general intent routing, has a `get_messages()` pagination bug returning the oldest 6 messages after turn 3, and can retrieve the active user turn or assistant replies (`0.627` self-echo).
- **Rule-Based Extraction Gate:** `src/extraction/policy.py` uses regex heuristics only (ignoring `llm_caller`), missing cross-turn coreference and flagging some first-person questions/hypotheticals; `src/memory.py` is append-only without `UPDATE`/`DELETE` supersession.
- **Offline Test Status:** `523` tests pass offline; `2` tests (`test_query_aware_answering_flow`, `test_12_answer_question_abstention_on_unrecorded_facts`) fail with `ConnectionError` when the local Ollama server is stopped.
- **Citation Validation:** Checks context-key ID presence, not semantic entailment (NLI). InterMemory Attention (IMA) remains an unimplemented theoretical proposal.

---

## 3. Prerequisites & Installation

- **Python:** 3.10 – 3.13
- **Local LLM (Live Mode):** [Ollama](https://ollama.ai) running locally (`ollama serve`) with `gpt-oss` (default generator) and optional `qwen2.5:14b` (judge).

```bash
git clone https://github.com/Krish1022sona/IVARA.git
cd IVARA
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1 | Linux/macOS: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional dev/eval dependencies:
pip install -r requirements-dev.txt
```

Clean installations initialize an empty `data/memories.json` automatically on first write (or you can copy `data/memories.example.json` to `data/memories.json` as a starter fixture). Runtime databases and `data/memories.json` are excluded from Git tracking.

---

## 4. Basic CLI Usage

```bash
# Interactive chat session
python main.py chat

# Direct memory query
python main.py ask "What projects have I worked on?"

# Manage conversations and stored memories
python main.py chats list
python main.py memories stats

# Deterministic offline demo (no Ollama required)
python demo.py
```

---

## 5. Verified LongMemEval-S Benchmark Summary (`Config D` Batch Pipeline)

Evaluated on all 500 questions of `longmemeval_s_cleaned.json` using `gpt-oss` (`temperature=0.0`) and the official `qwen2.5:14b` judge prompt (see [`evaluation/FINAL_RESULTS.md`](evaluation/FINAL_RESULTS.md) for full breakdown and qualifications):

| Configuration | Overall ($N=500$) | Macro Avg | Answerable ($N=470$) | Negative Controls ($N=30$) | Mean Latency |
|---|---:|---:|---:|---:|---:|
| **Candidate E (Baseline RAG)** | 40.80% (204/500) | 45.10% | 37.87% (178/470) | 86.67% (26/30) | 9.92 s |
| **Candidate C (Query-Aware)** | 56.80% (284/500) | 60.54% | 54.47% (256/470) | **93.33% (28/30)** | **7.25 s** |
| **Candidate B (Arm B)** | **58.40% (292/500)** | **61.34%** | **56.60% (266/470)** | 86.67% (26/30) | 7.46 s |

*Note:* Temporal synthesis (deterministic calendar arithmetic + chronological context) drove a `+5.26 pp` gain in the 133-question temporal ablation (`46.62% -> 51.88%`), whereas Temporal Event Graph traversal alone added `0.00 pp` in that experiment. These historical batch results do not reflect current interactive CLI chat behavior.

---

## 6. Documentation Map

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — Actual directory structure, runtime flows, storage, interfaces, and limitations
- [`ROADMAP.md`](ROADMAP.md) — Active development phase, known bugs, acceptance criteria, and long-term research goals
- [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) — Chronological engineering & experimental history
- [`evaluation/FINAL_RESULTS.md`](evaluation/FINAL_RESULTS.md) — Verified LongMemEval-S metrics, McNemar tests, and failure analysis
- [`CONTRIBUTING.md`](CONTRIBUTING.md) & [`SECURITY.md`](SECURITY.md) — Contribution guidelines and local privacy policy
