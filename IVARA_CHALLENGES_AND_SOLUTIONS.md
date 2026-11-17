# IVARA — Challenge & Solution Register

**Project Title:** IVARA (Intelligent Verifiable Augmented Reasoning Agent)  
**Document Purpose:** Comprehensive register of significant technical failures, investigations, architectural trade-offs, and verified resolutions throughout IVARA's development.  
**Audit Standard:** Grounded in empirical benchmark artifacts, commit logs, failure forensic reports, and automated test suites.

---

## Challenge 1: Category H Multi-Session Temporal Collapse (0/41 Baseline)

### Challenge
On the LongMemEval temporal reasoning benchmark (133 questions), the released baseline (`55a273b`) achieved an overall accuracy of 46.62% (62/133), but scored **0 / 41 (0.0%)** on Category H queries requiring evidence aggregation across multiple disparate conversation sessions.

### How It Was Discovered
Discovered during the Phase 1 Temporal Reasoning Experiment (`PHASE1_TEMPORAL_EXPERIMENT_REPORT.md`, `TEMPORAL_REASONING_ARCHITECTURE_AUDIT.md`), which segmented the 133 questions into single-session, temporal arithmetic, and cross-session categories.

### Why It Happened
Standard dense MMR retrieval ranks individual conversational chunks based on cosine similarity to the query. For comparative temporal queries (e.g., *"How many days passed between my visit to MoMA and my visit to the Met?"*), chunks from the more semantically dominant session occupied all top-$k$ context slots ($k=5$), completely crowding out chunks from the second required session. Multi-session session hit rate was only 35.77%.

### Approaches Attempted
1. *Global Hierarchical Retrieval (Phase 2A):* Grouped all conversation chunks by session ID, ranked sessions first, and allocated chunks evenly. **Result: Failed.** While Category H accuracy rose to 6/41, single-session accuracy collapsed from 67.39% to 48.91% due to loss of intra-session depth.
2. *Adaptive Syntax Routing (Phase 2A.1):* Used regexes to route comparative queries to hierarchical retrieval and factual queries to flat retrieval. **Result: Failed.** Recovered overall score to 59/133, still below the 62/133 baseline.
3. *Query Decomposition & Subquery Allocation (Phase 3 & 3.1):* Extracted event clauses and retrieved sessions independently. **Result: Partial.** Reached 64/133 and 65/133, but introduced latency and subquery extraction failures.

### Final Solution
**Flat Dense MMR Retrieval with Temporal-Aware Context Budgeting & Improved Synthesis (Phase 2B.2 Arm B):**
Rather than forcing brittle session-level clustering or syntax routing, retain flat dense MMR chunk retrieval with an expanded candidate pool (`effective_k = max(k, k_temporal)`, `pool = k * 2 + 5`), coupled with a calendar arithmetic post-processor (`CalendarArithmeticEngine`) and lenient structured JSON response extraction.

### Verification
- `PHASE2B2_ABLATION_REPORT.md`: Arm B achieved **69 / 133 (51.88%)** overall temporal accuracy (+7 correct answers over baseline).
- Category H accuracy improved to verified recoveries without causing non-Category H regressions.
- `FINAL_BAKEOFF_REPORT.md`: Candidate B won the 500-question final bakeoff with **58.40% (292/500)** overall accuracy.

### Remaining Limitations
True cross-session reasoning across sessions separated by dozens of distractor dialogues remains bounded by the context token limit ($2500$ tokens) and candidate pool depth.

---

## Challenge 2: The Hierarchical Retrieval Depth vs. Diversity Dilemma

### Challenge
Attempting to fix multi-session recall using hierarchical session clustering caused a massive regression on non-Category H questions (falling from 67.39% to 48.91%, net loss of 17 previously correct answers).

### How It Was Discovered
Identified during Phase 2A evaluation (`PHASE2A_HIERARCHICAL_RETRIEVAL_REPORT.md`) and forensically audited in `PHASE2A1_FAILURE_FORENSICS.md`.

### Why It Happened
Single-session queries require deep conversational context (consecutive turns) to resolve conversational anaphora, speaker shifts, and contextual nuances. Hierarchical retrieval capped chunks per session at 2 to maximize session diversity, fragmenting conversational continuity and withholding critical adjacent turns from the LLM.

### Approaches Attempted
- *Dynamic per-session budgeting based on session relevance scores.* Produced inconsistent context boundaries and exceeded token budgets on long sessions.

### Final Solution
**Rejection of Global Hierarchical Retrieval:** Formal rejection of hierarchical retrieval as a drop-in replacement. Preservation of the flat MMR retrieval engine with soft session-capping (`max_chunks_per_session=3`) only when explicitly requested, while defaulting to relevance-ordered MMR.

### Verification
Reverting from Phase 2A (51/133) to Phase 2B.2 Arm B (69/133) completely restored the 17 regressed questions while preserving multi-session recoveries.

### Remaining Limitations
Balancing intra-session coherence and inter-session diversity without query-specific human tuning remains an open information-retrieval challenge.

---

## Challenge 3: Temporal Event Graph Overconnectivity & Spurious Traversal

### Challenge
A prototype Temporal Event Graph (TEG) designed to explicitly model chronological event sequences produced a heavily overconnected graph that degraded retrieval recall and introduced irrelevant temporal distractors.

### How It Was Discovered
Exposed during Phase 2B.1 evaluation and audited in `PHASE2B2_GRAPH_QUALITY_REPORT.md` and `PHASE2B2_FAILURE_ANALYSIS.md`.

### Why It Happened
Extracting temporal relationships across 20 sample dialogue sessions generated 1,092 event nodes and **15,774 temporal edges**—an average of **14.4 edges per event**. LLM event extraction created transitive edges between virtually all timestamped mentions, turning the graph into a dense near-clique. Traversal along `OCCURRED_BEFORE` and `OCCURRED_AFTER` edges flooded the answering context with dozens of unrelated events occurring on the same day.

### Approaches Attempted
- *Confidence thresholding ($conf \ge 0.7$):* Reduced edges by 35%, but still left >10,000 spurious edges.
- *Strict Transitive Reduction:* Computationally expensive on large episodic databases ($O(V \cdot E)$) and pruned legitimate parallel events.

### Final Solution
**Controlled 4-Way Ablation & TEG Decommissioning:**
Phase 2B.2 evaluated:
- Arm A (Baseline): 62/133
- Arm B (Synthesis improvements only): 69/133
- Arm C (Graph traversal only): 55/133 (-7 drop!)
- Arm D (Graph + Synthesis): 69/133
The ablation proved that the Temporal Knowledge Graph contributed **0% incremental accuracy** over Arm B, reduced Recall@5 from 94.26% to 91.73%, and added 15k noisy edges. The TEG was formally decommissioned and excluded from the production freeze.

### Verification
`PHASE2B2_ABLATION_REPORT.md` and `FINAL_CORE_ENGINEERING_REPORT.md`.

### Remaining Limitations
Extracting sparse, semantically meaningful temporal event graphs from conversational transcripts without heuristic overconnection remains unsolved with lightweight models.

---

## Challenge 4: Unsafe Malformed-Output Fallback in Answer Generation

### Challenge
When the synthesizer LLM generated unparseable raw text (non-JSON gibberish or unformatted text), the answering engine's fallback routine promoted the raw text to an authoritative answer and marked `status = "answered"`, causing false positives and failing unit tests.

### How It Was Discovered
Exposed during the Final Bakeoff Integrity Audit (`FINAL_BAKEOFF_INTEGRITY_AUDIT.md`) and verified by two failing unit tests:
- `tests/test_answering.py::test_malformed_model_output_handled_safely`
- `tests/test_calibrated_answering.py::test_malformed_model_output_fallback`

### Why It Happened
In `src/answering.py` (legacy lines 935–968), the recovery logic executed:
```python
# Legacy vulnerable code
if text_content:
    extracted_answer = text_content.strip()
    status = "answered"
```
If regex extraction failed to find JSON or answer tags, `text_content` retained the raw unparsed string, and the ternary condition `status = "answered" if text_content else "insufficient_evidence"` defaulted to `"answered"`.

### Approaches Attempted
- Simply failing on any JSON error. Rejected because minor markdown formatting or conversational preambles from local models would cause premature abstentions on valid answers.

### Final Solution
**Multi-Stage Bounded Recovery Pipeline with Strict Degradation:**
1. Direct Pydantic validation (`AnswerPayload.model_validate_json`).
2. Lenient JSON parse (`json.loads(raw, strict=False)`).
3. Explicit regex extraction targeting `"answer"\s*:\s*"([^"\\]*)"`.
4. Explicit Markdown header extraction targeting `(?:\*\*Answer:\*\*|Answer:)\s*(.+)`.
5. Legitimate calendar computation injection (`CalendarArithmeticEngine`).
6. If none of these five structured extractions succeeds, `extracted_answer` remains empty, and the pipeline strictly returns `status = "insufficient_evidence"` with limitation note `"Model output was malformed and could not be parsed into the structured answer schema."`

### Verification
- `PRE_FREEZE_HARDENING_REPORT.md`: Both failing unit tests passed immediately.
- 9 dedicated regression tests in `tests/test_pre_freeze_hardening.py` verify that plain-text garbage, empty responses, and malformed JSON safely degrade to `insufficient_evidence`.

### Remaining Limitations
Local open-weights models (e.g. `qwen2.5:14b`) occasionally omit closing JSON delimiters under constrained context lengths, requiring regex recovery.

---

## Challenge 5: Closed-World vs. Open-World Ontology Breakdown

### Challenge
Strict ontology validation rejected 42% of legitimate conversational relationships extracted from user dialogues because predicates did not match a fixed, predefined vocabulary.

### How It Was Discovered
Discovered during Milestone 2 evaluation (`ONTOLOGY_MILESTONE_2_REPORT.md`, `tests/test_open_world_ontology.py`).

### Why It Happened
Early IVARA used a closed-world schema with 18 predefined predicates (`works_at`, `lives_in`, `manages`, etc.). When users stated *"I consult for Aetheria"* or *"I adopted a rescue terrier"*, the validator dropped the facts because `consults_for` and `adopted` were not in `vocabulary.json`.

### Approaches Attempted
- Expanding the static vocabulary. Impractical due to the unbounded linguistic diversity of personal conversation.

### Final Solution
**Two-Tier Hybrid Relationship Resolution (`src/ontology/relationship_resolver.py`):**
- *Tier 1 (Canonical Mapping):* Synonyms and aliases dynamically mapped to canonical predicates (e.g., `"employed by"` $\to$ `"works_at"`).
- *Tier 2 (Open-World Fallback):* Unmapped but grammatically well-formed predicates preserved with `predicate_status = "open_world"` and confidence penalty, rather than being discarded.

### Verification
Relationship extraction recall rose from 58% to 91% without increasing relational hallucinations (`tests/test_open_world_ontology.py`, 20 passed).

### Remaining Limitations
Open-world predicates lack formal inference axioms (e.g., transitivity or inverse relations) available to canonical predicates.

---

## Challenge 6: Knowledge Graph Fan-Out & Generic Entity Distractor Flooding

### Challenge
Knowledge graph retrieval from root entity nodes (e.g., `self` or `user`) returned dozens of irrelevant edges, flooding context and crowding out direct episodic evidence.

### How It Was Discovered
Exposed during Milestone 3 multi-hop reasoning evaluations (`GRAPH_ABLATION_REPORT.md`, `UNIFIED_RETRIEVAL_MILESTONE_3_REPORT.md`).

### Why It Happened
In personal memory graphs, the user node is a massive hub connected to hundreds of edges. 2-hop traversal from `self` pulled in almost the entire graph, exceeding token limits and causing the answering LLM to hallucinate or select incorrect facts.

### Approaches Attempted
- Global depth reduction to $d=1$. Prevented multi-hop reasoning entirely.

### Final Solution
**Knowledge Graph Evidence Safeguards (`src/unified_retrieval/graph_safeguards.py`):**
1. *Generic Entity Distractor Filtering:* Generic nodes (`self`, `user`, parameterized `user_name`) are blocked from expanding unless the query explicitly shares tokens with the edge predicate or target entity.
2. *Per-Entity Fan-Out Caps:* Limits outgoing edges per entity node to 4.
3. *Strict Provenance Verification:* Triples lacking a verifiable pointer to an underlying episodic turn ID or semantic memory ID are excluded.

### Verification
`tests/test_adaptive_evidence.py::TestGraphSafeguards` (21 passed). Distractor triples dropped by 84% while multi-hop hit rate was preserved.

### Remaining Limitations
If an edge is factually correct but described using completely different vocabulary than the query, relevance filtering may occasionally exclude it.

---

## Challenge 7: Hardcoded Synthetic Persona Assumptions ("Arjun Mehta")

### Challenge
The synthetic test persona `"Arjun Mehta"` was hardcoded into production heuristics across `router.py`, `graph.py`, `graph_safeguards.py`, and `answering.py`, creating severe production bugs:
- Queries mentioning another person named Arjun aliased that person to the current user's `self` node.
- Triples involving an external person named Arjun were suppressed as generic user distractors.
- Prompts and speaker turn formatting leaked the name Arjun Mehta.

### How It Was Discovered
Discovered during the Final Bakeoff Integrity Audit (`FINAL_BAKEOFF_INTEGRITY_AUDIT.md`) and flagged as a logical contradiction during pre-freeze verification.

### Why It Happened
During Milestones 1–3, synthetic unit and integration tests used "Arjun Mehta" as the test subject. To make heuristic regexes and seed entity matchers pass offline tests, developers hardcoded `"arjun"` directly into production heuristics.

### Approaches Attempted
- Initially claiming the references were inert documentation. Refuted upon code inspection showing active execution in regexes, seed tokens, and stopword sets.

### Final Solution
**Complete Production Neutralization & Dynamic Parameterization:**
1. `src/answering.py`: Neutral system prompt generated via `get_answering_system_prompt()` defaulting to `"the User"`, `"their"`, `"they"`.
2. `src/unified_retrieval/episodic.py`: Turns format as `Speaker: the User (user): ...`.
3. `src/unified_retrieval/graph.py`: Removed "arjun" from `user_tokens`. Appends `self.user_name` only when explicitly passed.
4. `src/unified_retrieval/graph_safeguards.py`: Removed "arjun" from `GENERIC_ENTITY_NAMES`. Appends `self.user_name` only when explicitly passed.
5. `src/unified_retrieval/router.py`: Generalized regexes to bounded multi-word patterns `[a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}`; removed "arjun" from `known_entities`.
6. Synthetic test fixtures updated to pass explicit `user_name="Arjun"`.

### Verification
- `FINAL_FREEZE_GO_NO_GO.md` and `FINAL_ROUTER_VERIFICATION.md`.
- Exactly **0 occurrences** of `"arjun"` or `"mehta"` in `src/`.
- 6 new regression tests in `tests/test_pre_freeze_hardening.py` confirm:
  - Third-party queries about Arjun do NOT map to `self`.
  - First-person queries map to `self`.
  - External entities named Arjun are NOT suppressed as distractors.
- All 525 repository tests pass.

### Remaining Limitations
Zero remaining in codebase. Identity is 100% parameterizable.

---

## Challenge 8: LLM-as-a-Judge Stochastic Boundary Flip

### Challenge
Candidate C (Phase 3) scored 64/133 in historical Phase 3 evaluation, but scored 65/133 in the 500-question final bakeoff on identical generated predictions.

### How It Was Discovered
Discovered during the provenance audit in `FINAL_BAKEOFF_INTEGRITY_AUDIT.md`.

### Why It Happened
Identified the single discordant question: `gpt4_6ed717ea` (*"Which item did I purchase first, the dog bed for Max or the training pads for Luna?"*).
Inspection of the raw hypothesis file proved the generated text was **byte-for-byte identical** in both runs. The candidate gave a hedged, ambiguous answer (*"The training pads were ordered on May 3, while the dog bed was mentioned around May 3 as well..."*). In Phase 3, the judge LLM (`qwen2.5:14b` at temperature 0) parsed it as `"No"`; in the final bakeoff, it parsed it as `"Yes"`.

### Approaches Attempted
- Modifying the prediction to force consistency. Rejected to maintain strict benchmark provenance.

### Final Solution
**Formal Documentation of Stochastic Variance:**
Recorded the exact question ID, hypothesis text, and judge explanation. Noted that correcting Candidate C back to 64/133 widens Candidate B's winning margin from +8 to +9 questions on the temporal subset, confirming that Candidate B's victory was independent of judge noise.

### Verification
Documented in `FINAL_BAKEOFF_INTEGRITY_AUDIT.md` (Section 2) and `FINAL_BAKEOFF_REPORT.md`.

### Remaining Limitations
LLM-as-a-judge evaluations carry inherent boundary noise on hedged outputs (~0.2% variance), reinforcing the necessity of strict deterministic extraction and calendar arithmetic.

---

## Challenge 9: Multi-Word & Possessive Name Failure in Generalized Router

### Challenge
When hardcoded `"arjun"` was initially replaced with single-word token pattern `[\w\-]+` in `router.py`, queries containing multi-word names (e.g. *"What is Sarah Connor's birthday?"*) or formal titles (*"Does Dr. Shalini Verma still work at Aetheria?"*) dropped into `safe_fallback` (0.50 confidence).

### How It Was Discovered
Discovered during the deep router verification audit (`FINAL_ROUTER_VERIFICATION.md`) using an automated multi-word test script.

### Why It Happened
`[\w\-]+` matched only the first word (`"Sarah"`), leaving the second word (`"Connor's"`) to fail matching against the property keyword `(birthday|birth date...)`.

### Approaches Attempted
- Using open unconstrained wildcards `.+?`. Rejected because greedy matching caused over-matching into adjacent sentence clauses.

### Final Solution
**Bounded Multi-Word Expression Pattern:**
Implemented `[a-zA-Z0-9_\.\-]+(?:\s+[a-zA-Z0-9_\.\-]+){0,2}` across all routing regexes. Enforces a strict bound of 1 to 3 words, allowing single names (`"Sarah"`), two-word names (`"Sarah Connor"`), and titled names (`"Dr. Shalini Verma"`), while allowing period punctuation (`\.`) for abbreviations.

### Verification
- 28-query compact routing regression matrix achieved **28 / 28 PASS (100.0%)** (`FINAL_ROUTER_VERIFICATION.md`).
- Both single-word and multi-word queries route deterministically with high confidence (0.85–1.00).

### Remaining Limitations
Names exceeding 3 words (e.g., highly formal noble titles) require explicit user name configuration or fallback to semantic routing.
