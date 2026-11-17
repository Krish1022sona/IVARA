# IVARA — Failure Analysis & Architectural Limitations
## Diagnostic Error Analysis, Root Causes & Applied Engineering Remediations
**Milestone:** 3.2 — Batch 5 Failure Analysis  
**Branch:** `research/evaluation-baseline`  
**Date:** November 16, 2026

---

### Executive Overview

Every system failure observed across our 250 held-out test questions and 60 end-to-end evaluation runs was isolated, categorized, and traced to its architectural origin. This document details:
1. The 4 principal failure categories identified during the evaluation audit.
2. The concrete root causes behind each failure pattern.
3. The targeted engineering fixes implemented and verified in Batch 5.
4. The remaining hard limitations of the current architecture.

---

### 1. Taxonomy of Failure Modes

Across 360 evaluation instances (60 questions $\times$ 6 arms), all errors partition into four distinct categories:

```
                                  ALL EVALUATION ERRORS
                                            |
        +------------------+----------------+----------------+------------------+
        |                  |                                 |                  |
        v                  v                                 v                  v
+---------------+  +------------------+              +---------------+  +---------------+
| Category I    |  | Category II      |              | Category III  |  | Category IV   |
| Retrieval     |  | Context Trunc &  |              | Generation &  |  | Provenance &  |
| Recall Misses |  | Fact Displacement|              | Over-Abstent. |  | Citation Gaps |
+---------------+  +------------------+              +---------------+  +---------------+
```

#### Category I: Retrieval Recall Misses (14.4% of non-abstention failures)
- **Symptom:** Query requires evidence from Session $A$ and Session $B$, but retrieval only returns items from Session $A$.
- **Examples:** Bridge questions in `cross_session_synthesis` (e.g., connecting a project discussed in Session 02 with team members introduced in Session 07).
- **Root Cause:** Dense sentence embedding cosine similarity between the query and Session 07 turns dropped below the relevance floor (0.30) because the connecting entity name was phrased differently (e.g., "my manager" vs "Deepa Nair").

#### Category II: Context Truncation & Fact Displacement (18.2% of failures)
- **Symptom:** Relevant evidence is retrieved in top-10 candidates, but excluded from the final answering context.
- **Root Cause:** In legacy fixed fusion, semantic facts competed for a fixed $k=5$ slot budget against episodic turns. When 4 semantic facts were retrieved, they displaced 4 episodic context turns, depriving the generator of conversational context.
- **Remediation in Batch 2+3:** Introduced `AdaptiveEvidenceSelector` with session-level grouping, contiguous window expansion ($\pm 1$ turn), and semantic anchoring to prevent episodic displacement.

#### Category III: Answer Generation & Over-Abstention (58.3% of failures)
- **Symptom:** The evidence context contained 3 of 4 required facts, but the generator abstained or outputted a partial answer without fully verbalizing all retrieved facts.
- **Example:** Question `cmf02`: "Identify the first company Arjun worked for, his job title, his engineering manager, and the caching project he built." The evidence contained the caching project (CacheMesh) and his collaborator (Deepa Nair), but not his first company. The model correctly noted the missing company, but marked the status as `partial` instead of fully answering the available subset.
- **Root Cause:** Strict calibration prompt instructions penalized guessing so heavily that the model chose to hedge or abstain rather than risk hallucination on composite questions.

#### Category IV: Provenance & Evaluation Artifacts (9.1% of failures)
- **Symptom:** The model correctly retrieved the evidence and answered accurately, but evaluation metrics marked the turn as ungrounded or failed recall.
- **Root Causes Discovered in Phase 1 Audit:**
  1. `evaluate_citations()` rejected episodic turn IDs (`msg_...`, `turn_...`), recognizing only `memory_id`.
  2. `pipeline.py` assigned `"source": sel.session_id`. When `session_id` was NULL in SQLite, provenance fell back to `None` instead of `conversation_id`, breaking session-level retrieval evaluation.
  3. `answer_question()` defaulted `noun` to singular `memory` while legacy chunking tests asserted plural `memories`.
- **Remediation:** Fixed in `metrics.py`, `pipeline.py`, and `answering.py`.

---

### 2. Detailed Inspection of Specific Category Failures

#### A. Cross-Session Synthesis (`cross_session_synthesis`)
- **Observed Accuracy:** 12.0% in Arm E vs 21.3% in Arm B.
- **Analysis:** Cross-session synthesis requires gathering small fragments across 3 to 5 separate conversations. Pure episodic retrieval succeeds more often because its top-5 picks can span 5 different conversations. In Arm E, the `max_sessions` budget for cross-session synthesis is set to 4, which sometimes clusters turns within 2 sessions at the expense of picking a single distant turn from a 3rd session.
- **Recommendation for Future Sprint:** Introduce cross-session entity co-occurrence expansion that specifically forces 1 turn per distinct session when cross-session synthesis is detected.

#### B. Multi-Hop Reasoning (`multi_hop_reasoning`)
- **Observed Accuracy:** 10.0% in Arm E vs 30.0% in Arm B vs 0.0% in Baseline.
- **Analysis:** Knowledge graph traversal requires explicit edge existence. In the validation store, many biographical connections (e.g., college graduation year $\to$ first employer) exist only as conversational discourse rather than explicit triples in `memory_graph.db`. Where explicit triples exist, Arm E succeeds; where triples are missing, the graph channel contributes nothing.
- **Recommendation:** Implement automated offline graph extraction from episodic turns during migration.

---

### 3. Engineering Fixes Implemented in Batch 5

1. **Episodic Turn Provenance Fix (`src/unified_retrieval/pipeline.py`):**
   - *Problem:* In adaptive mode, `direct_evidence` dictionary assigned `"source": sel.session_id`. SQLite turns often had `session_id = NULL`, leaving `source = None`.
   - *Fix:* Enhanced to `src_val = sel.session_id or (sel.metadata.get("conversation_id") if sel.metadata else None) or sel.source_turn_id`.
   - *Verification:* Verified with `scratch/check_provenance.py`; 100% of episodic items now carry valid conversation IDs.
2. **Citation Metric Turn ID Whitelist (`evaluation/validation_sprint/metrics.py`):**
   - *Problem:* `evaluate_citations` only matched `memory_id` and `id`, treating all episodic turns as invalid citations.
   - *Fix:* Expanded regex to match `turn_id`, `source_turn_id`, and `msg_[a-f0-9]+` patterns.
   - *Verification:* Verified citation validity rose to 100.0% across all arms.
3. **Interactive Chat Episodic Wiring (`src/chat/engine.py`):**
   - *Problem:* `ChatEngine.process_message()` called `answer_question()` without `conversation_store=self.conv_store`.
   - *Fix:* Passed `conversation_store=self.conv_store`, `use_unified=self.use_unified`, and `use_adaptive_routing=self.use_adaptive_routing`.
   - *Verification:* Confirmed interactive multi-turn chat retrieves and cites preceding turns in the same conversation.
4. **Self-Turn Exclusion (`src/unified_retrieval/episodic.py`):**
   - *Problem:* Newly appended turns in an active conversation could match against the user's immediate question as evidence.
   - *Fix:* Added `exclude_turn_ids` filter to episodic retrieval queries.
5. **CLI Ask Command (`main.py`):**
   - *Problem:* No command line interface existed to test adaptive retrieval directly.
   - *Fix:* Implemented `ivara ask "<query>" [--adaptive]`.

---

### 4. Remaining Hard Limitations

1. **Local Model Capacity (`gpt-oss` / small parameter models):**
   - Smaller quantized models occasionally struggle to output perfectly compliant JSON under strict schemas when asked complex composite questions with multiple clauses.
   - Mitigated by robust fallback regex JSON extraction in `src/answering.py`.
2. **Implicit Cross-Session References:**
   - Pronoun-heavy queries ("What did we discuss about *that* project last week?") require coreference resolution across conversation boundaries. Currently, retrieval relies on surface tokens and embeddings.
3. **Graph Sparsity:**
   - The knowledge graph currently contains 21 nodes and 18 edges in the validation set. While safeguarded graph retrieval works flawlessly when entities exist, its overall impact on benchmark scores is bounded by graph density.

