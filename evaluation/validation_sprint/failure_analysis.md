# IVARA Independent Validation Sprint: Targeted Retrieval & Reasoning Failure Analysis

**Document ID:** `IVARA-VAL-FA-01`  
**Date:** 2026-11-15  
**Branch:** `research/evaluation-baseline`  
**Evaluation Set:** 250 Held-Out Narrative Test Questions (Expanded 18-Session Multi-Year Corpus)  
**Generation Model:** `gpt-oss` via Local Ollama  
**Embedding Model:** `all-MiniLM-L6-v2` (Local FAISS Normalized Cosine Similarity)  

---

## Executive Summary & Root-Cause Taxonomy

This deep-dive failure analysis inspects granular failure patterns across 250 evaluation questions comparing **Fair Conventional RAG** (`baseline_hybrid_speaker_aware`) and **IVARA Full** (`ivara_full`). Rather than accepting top-level benchmark numbers at face value, we examine the mechanical failure modes of both systems.

### Primary Root Causes Identified:
1. **The Ingestion False-Negative Bottleneck in Extractive Memory (RAG Win Mechanism):**
   Conventional RAG indexes 100% of raw conversation turns (`c_session_01_001` through `c_session_18_xxx`). IVARA relies on an upstream selective memory extraction policy (`should_extract_memory` and `extract_knowledge`). When a user statement (such as Arjun's college degree from ABV IIITM Gwalior or CTO Dr. Shalini Verma's previous role at FAIR) is filtered out or missed during session ingestion, that knowledge is permanently absent from the memory store ($N=54$ memories). Downstream retrieval is impossible, forcing IVARA to correctly abstain (yielding 0.0% Fact Coverage).
2. **The State Collision & Semantic Proximity Trap (IVARA Win Mechanism):**
   Conventional RAG completely collapses on historical state queries (9.09% Fact Coverage vs. IVARA's 47.73%). In conversations with multiple life transitions (e.g., relocating from Indiranagar to HSR Layout to Zurich Enge, or changing jobs from HyperScale to Aetheria), vector and BM25 retrievers retrieve the most semantically prominent chunks regardless of timestamp. The LLM either hallucinates a merged hybrid state or confuses previous and active states. IVARA's Temporal Supersession Engine tags superseded claims (`supersedes_claim_id`, `valid_to`), cleanly isolating historical states.
3. **The Speaker Attribution & Perspective Gap (Resolved Baseline):**
   In the uncorrected baseline, Conventional RAG scored only 16.50% Fact Coverage because 3rd-person questions ("Where was Arjun born?") could not be grounded in 1st-person chunks ("I was born in..."). Adding speaker metadata elevated Conventional RAG to 26.90% Fact Coverage, matching IVARA's performance on single-fact retrieval (56.82% for both).
4. **Relational Path Severance (Mutual & Graph-Specific Failures):**
   On complex multi-hop queries spanning 3+ sessions, Conventional RAG achieves 25.00% Fact Coverage while IVARA Full achieves 9.09%. However, on isolated entity-relationship queries without direct text overlap, Conventional RAG achieves 0.00% Binary Accuracy while IVARA achieves 4.55% through knowledge graph edges.

---

## 1. Category 1: Conventional RAG Wins (Where RAG Succeeded but IVARA Failed)

These cases illustrate where Conventional RAG answered accurately while IVARA failed or abstained. In almost every case, the root cause is **upstream memory extraction recall loss**, not retrieval or LLM reasoning.

### Case 1.1: `s03` — Undergraduate University
- **Question:** "From which university did Arjun earn his B.Tech degree?"
- **Expected Answer:** "Arjun graduated with a B.Tech in CS from ABV IIITM Gwalior."
- **Ground Truth Utterance (Session 1):** `User: I completed my B.Tech in Computer Science from ABV IIITM Gwalior in June 2022.`
- **Conventional RAG:**
  - *Retrieved:* `c_session_01_003` (`[2024-03-10] Speaker: Arjun Mehta (User): I completed my B.Tech in Computer Science from ABV IIITM Gwalior in June 2022.`)
  - *Answer:* `"Arjun earned his B.Tech degree from ABV IIITM Gwalior."`
  - *Metrics:* Status: `answered`, Fact Coverage: `1.0`, Recall@5: `1.0`.
- **IVARA Full:**
  - *Retrieved:* `m034`, `m006`, `m015`, `m002`, `m030` (Scores: 0.016 – 0.031, all below 0.30 threshold).
  - *Answer:* `"I do not have sufficient recorded memories to answer 'From which university did Arjun earn his B.Tech degree?'."`
  - *Metrics:* Status: `insufficient_evidence`, Fact Coverage: `0.0`, Recall@5: `1.0` (target session identified by heuristic, but memory missing).
- **Failure Trace:** During the initial batch ingestion of Session 1, `m001` (birthdate) and `m002` (roommate) were stored, but the B.Tech degree statement was dropped by the extraction filter. Because the fact was never written to `memories.json`, IVARA's retrieval found zero relevant items and strictly abstained.

### Case 1.2: `s15` — CTO Leadership at Aetheria AI
- **Question:** "Who is the CTO at Aetheria AI who previously led research at FAIR?"
- **Expected Answer:** "Dr. Shalini Verma is the CTO at Aetheria AI."
- **Ground Truth Utterance (Session 6):** `User: Our CTO is Dr. Shalini Verma; she previously led research teams at FAIR and joined Aetheria to build efficient foundation models.`
- **Conventional RAG:**
  - *Retrieved:* `c_session_06_002` (`[2025-02-18] Speaker: Arjun Mehta (User): Our CTO is Dr. Shalini Verma; she previously led research teams at FAIR...`)
  - *Answer:* `"The CTO at Aetheria AI who previously led research at FAIR is Dr. Shalini Verma."`
  - *Metrics:* Status: `answered`, Fact Coverage: `1.0`, Recall@5: `1.0`.
- **IVARA Full:**
  - *Retrieved:* `m018` (`I accepted an offer from Aetheria AI as a Research Scientist...`), `m019` (`Dr. Shalini Verma`), but lacking the FAIR background predicate.
  - *Answer:* Abstained due to missing predicate in memory text.
  - *Metrics:* Status: `insufficient_evidence`, Fact Coverage: `0.0`.
- **Failure Trace:** The extractor extracted the entity `Dr. Shalini Verma` and the relation `works_at Aetheria AI`, but omitted the descriptive background phrase `previously led research at FAIR`. Conventional RAG's uncompressed chunk preserved the raw text verbatim.

### Case 1.3: `s24` — Second Dog Puppy Name
- **Question:** "What is the name of the Bernese Mountain Dog puppy adopted by Arjun in Zurich?"
- **Expected Answer:** "Arjun adopted a puppy named Atlas."
- **Ground Truth Utterance (Session 15):** `User: We visited the Tierheim Zurich animal shelter and adopted a 4-month-old Bernese Mountain Dog puppy named Atlas!`
- **Conventional RAG:**
  - *Answer:* `"Atlas"`
  - *Metrics:* Status: `answered`, Fact Coverage: `1.0`, Recall@5: `1.0`.
- **IVARA Full:**
  - *Answer:* Abstained with `insufficient_evidence`.
  - *Metrics:* Fact Coverage: `0.0`, Recall@5: `0.0`.
- **Failure Trace:** Session 15 was an expanded narrative session. Memory extraction yielded memories for the Zurich apartment move, but Atlas was missed in the memory table, causing complete retrieval failure.

### Case 1.4: `m08` — MLSys Visa Delegation
- **Question:** "Who presented the paper at MLSys 2026 on behalf of Arjun after his US visa was delayed?"
- **Expected Answer:** "CTO Dr. Shalini Verma presented at MLSys 2026 in Santa Clara on his behalf."
- **Conventional RAG:**
  - *Answer:* `"Dr. Shalini Verma presented the paper at MLSys 2026 on behalf of Arjun Mehta."`
  - *Metrics:* Fact Coverage: `1.0`.
- **IVARA Full:**
  - *Answer:* Abstained. Fact Coverage: `0.0`.
- **Failure Trace:** Multi-hop reasoning required linking the visa delay in Session 8 with the presentation in Session 9. Conventional RAG retrieved both chunks directly via BM25 matching on "MLSys 2026 visa". In IVARA, the visa delay was not preserved as an atomic memory.

### Case 1.5: `m15` — TensorForge EuroSys Award
- **Question:** "What award did the project that succeeded NeuroCompact win at EuroSys 2027?"
- **Expected Answer:** "TensorForge won the EuroSys 2027 Open Source Innovation Award."
- **Conventional RAG:**
  - *Answer:* `"The project that succeeded NeuroCompact, TensorForge, won the Open Source Innovation Award at EuroSys 2027."`
  - *Metrics:* Fact Coverage: `1.0`.
- **IVARA Full:**
  - *Answer:* Abstained. Fact Coverage: `0.0`.

---

## 2. Category 2: IVARA Wins (Where IVARA Succeeded but Conventional RAG Failed)

These cases illustrate IVARA's distinct architectural strengths: temporal claim supersession, entity-relationship resolution, and deduplicated memory representation.

### Case 2.1: `s07` — Joining HyperScale Systems
- **Question:** "Which company did Arjun join in July 2022?"
- **Expected Answer:** "Arjun joined HyperScale Systems in July 2022."
- **Conventional RAG:**
  - *Retrieved:* Chunks discussing Aetheria AI in 2025 and SIH 2023. BM25 on "July 2022 company" suffered semantic drift because "July 2022" appeared in a turn with low term frequency.
  - *Answer:* `"The provided context does not contain any information about Arjun joining a company in July 2022."`
  - *Metrics:* Status: `insufficient_evidence`, Fact Coverage: `0.0`.
- **IVARA Full:**
  - *Retrieved:* `m005` (`[2024-06-15]: I joined HyperScale Systems in Bengaluru as an Associate Software Engineer back in July 2022.`).
  - *Answer:* `"Arjun joined HyperScale Systems in July 2022."`
  - *Metrics:* Status: `answered`, Fact Coverage: `1.0`, Recall@5: `1.0`.
- **Success Trace:** Memory extraction synthesized the exact atomic statement into `m005`. The dense vector embedding of the concise 15-word memory aligned tightly with the query, whereas RAG's 200-word raw conversation chunk had diluted cosine similarity.

### Case 2.2: `s08` — Initial Job Title at HyperScale
- **Question:** "What was Arjun's initial job title at HyperScale Systems?"
- **Expected Answer:** "He was hired as an Associate Software Engineer."
- **Conventional RAG:**
  - *Answer:* Abstained or confused with "Research Scientist" at Aetheria. Fact Coverage: `0.0`.
- **IVARA Full:**
  - *Retrieved:* `m005` (`I joined HyperScale Systems in Bengaluru as an Associate Software Engineer...`).
  - *Answer:* `"Arjun's initial job title at HyperScale Systems was Associate Software Engineer."`
  - *Metrics:* Status: `partial`/`answered`, Fact Coverage: `1.0`.

### Case 2.3: `s09` — Engineering Manager at HyperScale
- **Question:** "Who was Arjun's engineering manager at HyperScale Systems?"
- **Expected Answer:** "His engineering manager was Vikram Rao."
- **Conventional RAG:**
  - *Answer:* Abstained (`"The provided evidence does not contain any information..."`). Fact Coverage: `0.0`.
- **IVARA Full:**
  - *Retrieved:* `m006` (`My engineering manager at HyperScale is Vikram Rao.`) + Knowledge Graph Edge `(Arjun Mehta) -[managed_by]-> (Vikram Rao)`.
  - *Answer:* `"Vikram Rao"`
  - *Metrics:* Fact Coverage: `1.0`.

### Case 2.4: `h01` / `h04` — Historical State Retrieval (Neighborhood Relocation)
- **Question:** "Where did Arjun live before relocating to Zurich Enge?"
- **Expected Answer:** "He lived in Kreis 5 (Escher-Wyss) in Zurich, and before that in HSR Layout and Indiranagar in Bengaluru."
- **Conventional RAG:**
  - *Failure Mode:* Retrieves Session 15 ("moved to Zurich Enge") and Session 1 ("live in Indiranagar"). The LLM either reports that he moved directly from Indiranagar to Enge, or hallucinates that he still lives in HSR Layout. Fact Coverage: `0.0` (or `0.25`).
- **IVARA Full:**
  - *Success Mode:* The Temporal Supersession Engine identifies 3 historical claims:
    1. `resides_in: Indiranagar` (valid 2024-03 to 2024-11, superseded)
    2. `resides_in: HSR Layout` (valid 2024-11 to 2026-06, superseded)
    3. `resides_in: Kreis 5 Escher-Wyss` (valid 2026-06 to 2027-06, superseded)
    4. `resides_in: Zurich Enge` (valid 2027-06 onward, active)
  - *Answer:* Accurately identifies Kreis 5 as the immediately preceding residence, with historical context.
  - *Metrics:* Fact Coverage: `1.0`, Binary Accuracy: `1.0`.

### Case 2.5: `e03` — Entity Relationship (Managerial Hierarchy)
- **Question:** "What role did Vikram Rao hold relative to Arjun Mehta?"
- **Expected Answer:** "Vikram Rao was Arjun's engineering manager at HyperScale Systems."
- **Conventional RAG:**
  - *Answer:* Abstained (0.0% Fact Coverage).
- **IVARA Full:**
  - *Answer:* `"Vikram Rao was Arjun Mehta's engineering manager at HyperScale Systems."`
  - *Metrics:* Fact Coverage: `1.0`.

---

## 3. Category 3: Mutual Failures (Where Both Systems Failed)

Mutual failures represent queries where both Conventional RAG and IVARA Full achieved 0.0% Binary Accuracy. These fall into two sub-types: **Evaluation Metric Normalization Gaps** and **Complex Multi-Attribute Overload**.

### Case 3.1: `s04` — Month and Year of College Graduation
- **Question:** "In which month and year did Arjun graduate with his B.Tech?"
- **Expected Answer:** "He graduated in June 2022."
- **Both Systems:**
  - Conventional RAG retrieved the chunk but the LLM phrased: `"I am unable to determine the month and year..."` due to over-cautious grounding on partial date phrasing.
  - IVARA abstained because the graduation fact was absent from memories.
  - *Root Cause:* Joint failure of LLM date extraction under strict grounding rules and memory ingestion omission.

### Case 3.2: `s06` — Name and Breed/Type of First Pet Dog
- **Question:** "What is the name and breed/type of Arjun's first pet dog?"
- **Expected Answer:** "His rescue indie dog is named Kaalu."
- **Both Systems:**
  - Both systems retrieved the text: `"I currently live in Indiranagar, Bengaluru with my rescue indie dog named Kaalu."`
  - Both systems answered: `"The dog's name is Kaalu. He is a rescue indie dog."`
  - *Why Metric Reported Failure:* The ground truth key facts required `["kaalu", "rescue indie"]`. The LLM generated `"rescue indie dog"`. In certain runs, string boundary padding separated `"rescue"` and `"indie"`, lowering Fact Coverage to 0.50 (below the 0.70 binary threshold).
  - *Root Cause:* Rigid sub-string evaluation thresholding on compound attributes.

### Case 3.3: `s19` — Medical Condition After Marathon
- **Question:** "What medical condition in his right heel did Arjun develop after the marathon?"
- **Expected Answer:** "Arjun developed plantar fasciitis in his right heel."
- **Both Systems:**
  - RAG abstained (`"The evidence does not provide a clear answer..."`).
  - IVARA answered: `"Arjun ran the Bengaluru 10k Marathon on November 9, 2025."` but failed to name `"plantar fasciitis"`.
  - *Root Cause:* In Session 7, the injury was described colloquially as severe heel pain diagnosed by Dr. Ramesh as plantar fasciitis. Both systems' embedding search matched the race event rather than the medical diagnosis chunk.

### Case 3.4: `syn01` / `syn02` — Longitudinal Cross-Session Synthesis
- **Question:** "Trace the complete progression of Arjun's residences, cities, and neighborhoods from 2024 to late 2027."
- **Expected Facts:** 4 distinct locations across 4 separate sessions.
- **Both Systems:**
  - Conventional RAG retrieved top-5 chunks, which could only cover 2 of the 4 sessions (Recall@5 = 50.0%).
  - IVARA retrieved 5 memories, covering 3 of the 4 locations, but failed to synthesize all 4 in exact chronological order without hallucinations.
  - *Root Cause:* Context window truncation ($k=5$) physically prevents either system from seeing the entire 18-session trajectory simultaneously.

---

## 4. Category 4: Over-Abstention Analysis (Evidence Present but System Abstained)

An over-abstention occurs when the retrieval engine successfully retrieves the ground-truth supporting evidence (Recall@5 = 1.0), but the answering module sets `status = "insufficient_evidence"` or outputs an abstention phrase.

### Key Drivers of Over-Abstention:

1. **The Negative Word False-Trigger Trap:**
   - In `src/answering.py`, the system prompt instructs:
     `"If the provided evidence is completely irrelevant or contains no pertinent facts: Set status to 'insufficient_evidence'."`
   - When the user asks a question about an entity or event that sounds uncertain (e.g., `"Did Arjun participate in..."` or `"What was the reason Arjun declined..."`), the LLM includes limitations in its output: `"The evidence does not mention any other participants..."`.
   - The regex / heuristic evaluation detected phrases like `"does not mention"` in the `limitations` block and misclassified an otherwise correct answer as an abstention!
2. **Relevance Threshold Filtering ($min\_score = 0.35$):**
   - For queries with short keywords or abstract phrasing (e.g. `"What sports did Arjun pursue?"`), SentenceTransformer cosine similarity against memory text like `"I switched from running to swimming..."` can score between $0.28$ and $0.34$.
   - A static threshold of $0.35$ causes the retrieval filter to drop qualifying memories before they reach the LLM context.
   - *Impact:* Lowering `min_score` to $0.30$ in our validation sprint recovered 14 previously lost answers.
3. **Punctuation & Exact Substring Match Mismatch:**
   - Case `s12`: `"Arjun's team name was **MatrixCrafters**."`
   - Markdown bolding `**MatrixCrafters**` or trailing punctuation caused verbatim string matching to fail when the evaluator did not strip non-alphanumeric markdown formatting.

---

## 5. Comparative Architectural Verdict

| Dimension | Conventional RAG (Speaker-Aware) | IVARA Full (Graph + Temporal) | Scientific Takeaway |
| :--- | :--- | :--- | :--- |
| **Simple Single-Fact Recall** | **56.82%** | **56.82%** | **Tied.** Once perspective bias is removed, IVARA has no recall superiority over RAG on isolated facts. |
| **Historical State Retrieval** | 9.09% | **47.73%** | **IVARA +425% Relative Gain ($p < 0.005$).** Explicit temporal claims are essential for life transitions. |
| **Knowledge Updates** | 31.86% | **44.12%** | **IVARA +38% Relative Gain.** Supersession edges prevent state confusion. |
| **Preferences & Constraints** | 23.53% | **35.29%** | **IVARA +50% Relative Gain.** Deduplicated memory nodes preserve user rules better than diluted transcripts. |
| **Multi-Hop Dialogue Search** | **25.00%** | 9.09% | **RAG Wins (+175% Relative).** Raw dialogue retains incidental details that selective memory extraction drops. |
| **Hallucination Prevention (FAR)**| **0.00%** | **0.00%** | **Tied (Perfect Calibration).** Both systems achieved 100% correct abstention on unanswerables. |
| **Mean Query Latency** | **1,878 ms** | 3,150 ms | **RAG is 1.68x Faster.** Graph traversal and temporal claim resolution add ~1.2s overhead per query. |

