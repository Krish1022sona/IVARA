# IVARA — Phase 2B.1: Temporal Event Graph Prototype Design

**Date:** 2026-11-17  
**Author:** Antigravity (DeepMind Team)  
**Branch:** `experiment/temporal-kg-phase2b`  
**Baseline Commit:** `55a273bff20fa0eec1b2d361e8a03b217d147abe`  
**Status:** Experimental Prototype Design (Isolated Research Sandbox)  

---

## 1. Architectural Context & Motivation

### 1.1 Empirical Background
In Phase 1 and Phase 2A/2A.1:
1. **Flat Retrieval (Phase 1):** Achieved 46.62% (62/133) overall temporal accuracy, but failed completely on Category H multi-session temporal reasoning (0/41 correct). Dense retrieval suffers severe session starvation when events are distributed across multiple disconnected sessions.
2. **Adaptive Hierarchical Retrieval (Phase 2A.1):** Recovered 6 / 41 Category H questions (+14.63%) by forcing session diversity. However, 35 Category H questions remained unresolved because the retriever surfaced individual chunks without explicit chronological linkage or event-level ordering.
3. **Forensic Findings (`PHASE2A1_FAILURE_FORENSICS.md`):**
   - 35 unrecovered Category H questions require multi-session event ordering and date comparison across disjoint episodes.
   - Regressions in Phase 2A.1 were heavily driven by prompt formatting and context inflation (6 / 11 cases), while 2 cases were semantic drift in dense chunk retrieval.
   - The forensic analysis confirmed that an explicit **Temporal Event Graph (TEG)** is technically justified for cross-session event ordering, entity-anchored temporal discovery, and historical state supersession.

### 1.2 Design Philosophy: Minimal, Isolated, Augmented
- **Not a heavy Knowledge Graph Platform:** We do NOT introduce heavy triple stores, external graph engines (Neo4j), or complex ontology expansion.
- **Episodic Supplementation:** The event graph supplements the existing episodic retriever; it does NOT replace it.
- **Traceable Provenance:** Every event node and relationship edge retains verbatim text provenance, session ID, turn ID, and extraction confidence.
- **Distinction Between Event Time and Ingestion Time:** The time an event happened in the real world (`event_time`, `valid_from`, `valid_to`) is strictly decoupled from the time the conversation occurred (`recorded_at`).
- **Preservation of Uncertainty:** When a date is vague or relative, uncertainty is preserved via `TemporalPrecision` rather than fabricating precise timestamps.
- **No Ordering Inferred from Ingestion Time Alone:** Two events mentioned in successive sessions are never assumed to have occurred in that order without explicit temporal intervals or linguistic temporal markers.

---

## 2. Temporal Event Representation & Schema

The event graph introduces `EventNode` and `TemporalRelationshipEdge` within IVARA's SQLite graph subsystem (`src/graph/store.py`).

### 2.1 Temporal Precision Specification
```python
class TemporalPrecision(str, Enum):
    EXACT_DAY = "EXACT_DAY"    # e.g. "2023-04-12"
    MONTH = "MONTH"            # e.g. "2023-07"
    YEAR = "YEAR"              # e.g. "2021"
    RELATIVE = "RELATIVE"      # e.g. "three days ago", anchored to recorded_at
    UNKNOWN = "UNKNOWN"        # e.g. "recently", "in the past", no anchor
```

### 2.2 Event Node Schema
```python
class EventNode(BaseModel):
    event_id: str                      # Unique identifier, e.g. "evt_7f8a9b2c"
    description: str                   # Clean summary of event (e.g. "Attended TechCon in Austin")
    entity_ids: list[str]              # Associated canonical entity IDs or candidate mentions
    session_id: str                    # Source conversation/session ID
    turn_id: str | None                # Source turn ID (e.g. "turn_003") or chunk ID
    event_time_text: str | None        # Verbatim temporal phrase (e.g. "on May 14th, 2023")
    valid_from: str | None             # Standardized ISO date: "YYYY-MM-DD", "YYYY-MM", or "YYYY"
    valid_to: str | None               # Standardized ISO date: "YYYY-MM-DD", "YYYY-MM", or "YYYY"
    recorded_at: str | None            # Timestamp of conversation ingestion (session.date)
    precision: TemporalPrecision       # Precision classification
    confidence: float                  # Extraction confidence [0.0, 1.0]
    provenance: str                    # Verbatim snippet from turn containing event and date
```

Key invariants:
- `event_time != recorded_at`: If a user says on 2023-11-10 "I graduated high school in 2018", `valid_from = "2018-01-01"`, `valid_to = "2018-12-31"`, while `recorded_at = "2023-11-10"`.
- Uncertainty bounding:
  - If `precision == MONTH` ("July 2023"), `valid_from = "2023-07-01"`, `valid_to = "2023-07-31"`.
  - If `precision == YEAR` ("2021"), `valid_from = "2021-01-01"`, `valid_to = "2021-12-31"`.
  - If `precision == UNKNOWN`, `valid_from = None`, `valid_to = None`.

---

## 3. Temporal Relationship Vocabulary

We implement a strictly bounded 4-relationship vocabulary:

| Relationship | Direction | Meaning | Ingestion-Order Safe? |
|---|---|---|---|
| `OCCURRED_BEFORE` | $E_1 \to E_2$ | Event $E_1$ concluded before Event $E_2$ began ($E_1.\text{valid\_to} < E_2.\text{valid\_from}$). | **Yes.** Inferred only from verified date intervals or explicit linguistic order ("before X"). |
| `OCCURRED_AFTER` | $E_1 \to E_2$ | Event $E_1$ began after Event $E_2$ concluded ($E_1.\text{valid\_from} > E_2.\text{valid\_to}$). | **Yes.** Dual edge to `OCCURRED_BEFORE`. |
| `SUPERSEDES` | $E_{\text{new}} \to E_{\text{old}}$ | State or affiliation in $E_{\text{new}}$ replaces the previous state in $E_{\text{old}}$ (e.g. job change, move). | **Yes.** Inferred only on shared slot / attribute with strictly later event date or explicit update claim. |
| `INVALIDATES` | $E_1 \to E_2$ | Event $E_1$ terminates the validity of state/event $E_2$ (e.g. subscription cancellation, lease termination). | **Yes.** Requires explicit termination semantics ("cancelled", "terminated", "ended"). |

### 3.1 Temporal Edge Schema
```python
class TemporalEdge(BaseModel):
    id: str
    source_id: str                     # Source EventNode ID
    target_id: str                     # Target EventNode ID
    relationship: TemporalRelationshipType
    confidence: float
    evidence: str | None               # Supporting textual reason
    provenance_chunk_ids: list[str]    # Chunks supporting this relationship
    derived_by: Literal[
        "grounded_interval_comparison",
        "explicit_claim",
        "state_supersession_rule",
        "invalidation_rule"
    ]
```

---

## 4. Graph Construction Pipeline

```
 ┌────────────────────────────────────────────────────────┐
 │            Episodic Turn Chunks (per question)         │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │             TemporalEventExtractor                     │
 │  1. Scans text for event mentions & temporal markers   │
 │  2. Anchors relative dates using session.date          │
 │  3. Extracts entities & resolves against entity store  │
 │  4. Emits EventNodes with verbatim provenance spans    │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │           TemporalRelationshipBuilder                  │
 │  1. Evaluates all EventNode pairs                      │
 │  2. Compares [valid_from, valid_to] bounds             │
 │  3. Checks explicit transition/ordering markers        │
 │  4. Emits OCCURRED_BEFORE, OCCURRED_AFTER,             │
 │     SUPERSEDES, and INVALIDATES edges                  │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │       SQLiteGraphStore (Isolated per question)         │
 │  - node_type="event", node_type="chunk", "entity"      │
 │  - edges: contains_event, mentions_entity,             │
 │    OCCURRED_BEFORE, OCCURRED_AFTER, SUPERSEDES         │
 └────────────────────────────────────────────────────────┘
```

### 4.1 Event Extraction Rules
1. **Explicit Absolute Dates:** Matches ISO patterns (`YYYY-MM-DD`, `YYYY/MM/DD`) and English written dates (`March 15th, 2023`, `15 March 2023`).
2. **Relative Expressions:** Matches `yesterday`, `last week`, `two months ago`, `next Tuesday`, anchoring them deterministically against `session.date`.
3. **Month / Year Anchors:** Matches `in July 2023`, `in 2021`, setting bounds to month or year start/end.
4. **Action / State Predicates:**
   - Activities: `visited`, `travelled to`, `attended`, `ran`, `read`, `bought`, `joined`.
   - Transitions: `started working at`, `left`, `cancelled`, `graduated from`, `moved to`.
5. **Entity Association:**
   - Proper nouns and known ontology entities extracted from the turn.
   - Unresolved entities preserved as candidate mentions without forced hallucinated mapping.

### 4.2 Non-Ingestion Rule Enforcement
- If Event A is extracted from Session 1 (Date: 2023-01-01) and Event B is extracted from Session 2 (Date: 2023-02-01):
  - If Event A's event date is 2023-05-01 and Event B's event date is 2022-12-01:
    - Event B `OCCURRED_BEFORE` Event A, despite Session 2 being ingested *after* Session 1!
  - If Event A has no valid date and Event B has no valid date, **NO temporal relationship is created** based on session ingestion order.

---

## 5. Bounded Temporal Graph Traversal & Retrieval

When a user query arrives:
1. **Query Analysis:**
   - Detect if query involves ordering (`first`, `earlier`, `order of`, `between`, `before`, `after`, `how many days passed`).
   - Extract candidate entity mentions from query (`Paris`, `London`, `FarmFresh`).
2. **Graph Matching:**
   - Find matching `EventNode`s connected to queried entities or matching event descriptions.
3. **Bounded Traversal (Max Hops = 2):**
   - For ordering queries:
     - Check for directed path $E_A \xrightarrow{\text{OCCURRED\_BEFORE}} E_B$ or $E_B \xrightarrow{\text{OCCURRED\_BEFORE}} E_A$.
     - Compute exact date difference using `DeterministicCalendarEngine` if exact dates are present.
   - For supersession / state queries:
     - Check for $E_{\text{new}} \xrightarrow{\text{SUPERSEDES}} E_{\text{old}}$ or $E \xrightarrow{\text{INVALIDATES}} E'$.
4. **Evidence Formulation:**
   - Retrieve the underlying `TurnChunk`s supporting the traversed event nodes and edges.
   - Synthesize a structured `[TEMPORAL EVENT GRAPH EVIDENCE]` block:
     ```
     [TEMPORAL EVENT GRAPH EVIDENCE]
     - Event: "Graduated with Computer Science degree" (Date: 2022-06-15, Precision: EXACT_DAY)
       Source: Session sess_1_c002 | Provenance: "I officially graduated with my CS degree on June 15, 2022"
     - Event: "Joined Datatech as Software Engineer" (Date: 2022-09-01, Precision: EXACT_DAY)
       Source: Session sess_2_c001 | Provenance: "Started my new role at Datatech on September 1, 2022"
     - Verified Ordering: Event 1 OCCURRED_BEFORE Event 2 (Delta: 78 days / ~2.5 months)
     ```
5. **Fallback:**
   - If no relevant event nodes or temporal paths are found, or if confidence is $< 0.4$, graph evidence is omitted and the system relies entirely on episodic retrieval.

---

## 6. Answer Synthesis & Formatting Protection

Phase 2A.1 forensics revealed that 6 of 11 regressions were caused by LLM output formatting fragility and prompt ordering inversion.

To prevent these regressions in Phase 2B.1:
1. **Relevance-First Prompt Layout:**
   - Maintain relevance-descending order for standard episodic evidence.
   - Place `[TEMPORAL EVENT GRAPH EVIDENCE]` as a clear, concise annotation block preceding the snippets.
2. **Robust Multi-Pattern Regex Fallback:**
   - In `src/answering.py`, if `json.loads` fails on the model response, use robust multi-pattern regex extraction for:
     - `{"answer": "...", "status": "..."}`
     - Markdown code fences containing JSON
     - Plain text answers following `Answer:` or direct answers.
   - This eliminates synthetic fallbacks (`Could not reliably synthesize an answer`) when the model had the right answer in text.

---

## 7. Experimental Protocol & Evaluation Metrics

We evaluate on the frozen 133-question LongMemEval temporal benchmark with the official `qwen2.5:14b` judge at temperature 0.0.

### 7.1 Evaluation Cohorts
1. **Category H (41 questions total):**
   - Focus on the 35 remaining failures from Phase 2A.1.
   - Verify whether the 6 recovered questions remain recovered.
2. **Non-Category H (92 questions):**
   - Verify whether the 11 Phase 2A.1 regressions are resolved.
3. **Full 133 Temporal Benchmark:**
   - Direct comparison against Phase 1 Flat Baseline (46.62%) and Phase 2A.1 Adaptive (44.36%).

### 7.2 Core Metrics
- **End-to-End Accuracy (%):** Official judge score across all 133 questions.
- **Category H Accuracy (%):** Accuracy on multi-session ordering.
- **Event Extraction Precision & Recall:** Audited on a sample of 20 representative episodes.
- **Temporal Relationship Accuracy:** Percentage of generated edges that are factually valid.
- **Abstention Accuracy (%):** Correct abstention on unanswerable temporal questions.
- **Latency & Context Tokens:** Overhead of event graph traversal vs flat retrieval.

