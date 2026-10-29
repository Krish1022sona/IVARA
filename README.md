# IVARA — Personal AI Memory System

IVARA is a local, privacy-first, evidence-grounded personal memory engine designed to overcome the fundamental limitations of naive RAG in long-term conversational AI.

Unlike traditional vector search that merely retrieves top-k similarity matches, IVARA treats memory as an evolving, structured knowledge network that actively:
1. **Tracks facts as they change over time**: Distinguishes event time from recording time and handles temporal supersession without discarding historical facts.
2. **Connects related memories across sessions**: Navigates entity and turn relationship graphs with cycle protection and bounded budgets.
3. **Integrates evidence across multiple memories**: Synthesizes complementary fragments while penalizing redundant copies and preserving citation provenance.
4. **Detects contradictions vs. updates**: Disambiguates legitimate updates, context distinctions, uncertainty, negations, and true conflicts.
5. **Enforces evidence sufficiency**: Gates answer generation with deterministic pre-flight checks and refuses to hallucinate when evidence is absent.
6. **Produces verifiable citations**: Validates every claimed memory citation against the retrieved evidence and rejects hallucinated IDs.

---

## Architecture Overview

```
                                +---------------------------+
                                |  User Conversational Q&A  |
                                +-------------+-------------+
                                              |
                                              v
                              +-------------------------------+
                              |    Domain Ontology Matching    |
                              |   Concepts & Entity Aliases   |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |    Hybrid Dense + BM25 Search |
                              |  Turn-Aware Chunk Retrieval   |
                              |  (Session Diversity Control)  |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |   Graph Traversal & Linking   |
                              |   Relevance-Filtered Fallback |
                              +---------------+---------------+
                                              |
                                              v
                   +--------------------------+--------------------------+
                   |                          |                          |
                   v                          v                          v
       +-----------------------+  +-----------------------+  +-----------------------+
       |   Temporal Engine     |  | Contradiction Engine  |  |  Evidence Integrator  |
       |  - Event-Year Priority|  |  - Legitimate Update  |  |  - Supporting Claims  |
       |  - Supersession Chains|  |  - Context Scope      |  |  - Complementary Info |
       |  - Historical States  |  |  - Unresolved Conflict|  |  - Redundancy Penalty |
       +-----------+-----------+  +-----------+-----------+  +-----------+-----------+
                   |                          |                          |
                   +--------------------------+--------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |     Pre-Generation Gate       |
                              |  (Sufficiency & Contradiction)|
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              | Local LLM Answer Synthesis    |
                              | (Ollama: gpt-oss:latest)      |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              | Post-Gen Citation Validator   |
                              | Grounded Answer with Provenance|
                              +-------------------------------+
```

---

## Core Intelligence Capabilities

| Capability | Module | Behavior |
| :--- | :--- | :--- |
| **Temporal Supersession** | `src/temporal/` | Event timestamps supersede older states (e.g. Austin 2024 supersedes Seattle 2023), while preserving historical facts when queried. Late-recorded historical facts do not overwrite newer states. |
| **Session Diversity Retrieval** | `src/chunking/retrieval.py` | Configurable `max_chunks_per_session` prevents a single recent session from crowding out crucial historical or relational evidence across multiple sessions. |
| **Contradiction Analysis** | `src/reasoning/contradiction.py` | Distinguishes `LEGITIMATE_UPDATE`, `CONTEXT_DISTINCTION` (e.g. Python at work vs Rust for hobby), `UNCERTAINTY`, `NEGATION`, and `TRUE_CONFLICT`. Surfaced to answering with qualification rather than silent choice. |
| **Evidence Integration** | `src/reasoning/evidence_integrator.py` | Categorizes candidates into supporting, complementary, redundant, and contradictory evidence. Penalizes duplicate claims to prevent false corroboration. |
| **Evidence Sufficiency Gate** | `src/reasoning/evidence_gate.py` | Pre-generation gate determines whether evidence is sufficient, partial, conflicting, or absent. Safely abstains (`insufficient_evidence`) when facts are missing. |
| **Citation Verification** | `src/answering.py` | Rejects hallucinated citation IDs, demotes ungrounded answers to partial status, and tracks memory-to-turn lineage. |

---

## Environment & Local Execution

IVARA is **100% LOCAL ONLY**:
- **No external LLM APIs** (no OpenAI, Anthropic, or remote endpoints).
- **No external data egress**.
- Uses local Ollama instance running CUDA/CPU.

### Prerequisites

1. **Python**: 3.12+ (tested on Python 3.14.4 virtual environment `.venv`)
2. **Ollama**: Running locally at `http://127.0.0.1:11434`
3. **Local Models**:
   - Generator: `gpt-oss:latest` (or `llama3.1:latest`)
   - Diagnostic Judge: `VladimirGav/gemma4-26b-16GB-VRAM:latest` (for evaluation harness)

---

## Quickstart & Verification Commands

All commands run from the repository root using the local environment:

### 1. Interactive 6-Scenario Capability Demonstration

Demonstrates the 6 core scenarios (Direct recall, Cross-session reasoning, Temporal update, Complementary evidence integration, Contradiction qualification, Safe abstention):

```powershell
# Offline deterministic verification (instant, zero LLM dependencies):
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe demo_ivara.py --offline

# Live verification with local Ollama generator:
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe demo_ivara.py
```

### 2. Comprehensive Test Suite

Run the full automated test suite (245 passing tests):

```powershell
$env:PYTHONPATH="."; .\.venv\Scripts\pytest.exe tests/
```

### 3. Targeted Evaluation on Held-Out Benchmark (N=40)

Runs preflight-gated controlled evaluation on the fresh held-out 40-question cohort:

```powershell
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe evaluation/autonomous_sprint/run_core_intelligence_eval.py
```

---

## Empirical Evaluation Summary

Results on the fresh held-out 40-question evaluation cohort (0% overlap with training or pilot sets):

| Metric | ARM 1 (Baseline Dense+BM25) | ARM 4 (Prior Best Candidate) | ARM 5 (Core Intelligence) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 52.5% (21/40) | 57.5% (23/40) | **60.0% (24/40)** |
| **Answerable Accuracy** | 50.0% (19/38) | 55.26% (21/38) | **57.89% (22/38)** |
| **Knowledge-Update Accuracy** | 37.5% (3/8) | 50.0% (4/8) | **75.0% (6/8)** *(+25% net gain)* |
| **Abstention Accuracy** | 100.0% (2/2) | 100.0% (2/2) | **100.0% (2/2)** *(0 hallucinations)* |
| **Average Query Latency** | 6.56s | 6.86s | **6.43s** *(fastest candidate)* |

*Note: Judgments rendered using local Gemma 27B under diagnostic rubric labeling (`diagnostic_local_gemma_not_official`).*

---

## Known Limitations & Boundaries

1. **Multi-Hop Relational Questions**: When entity relationships span 3+ conversational sessions without explicit shared keywords or ontology links, graph traversal may require deeper iterative query reformulation.
2. **Implicit Negative Temporal Transitions**: If a user stopped engaging in an activity without ever stating a cessation event (e.g. gradually ceased playing guitar), IVARA preserves the historical preference until an explicit update is presented.
3. **Diagnostic Evaluator Calibration**: Local Gemma 27B evaluation provides comparative diagnostics across arms; official benchmark comparisons require standardized external evaluator suites.
