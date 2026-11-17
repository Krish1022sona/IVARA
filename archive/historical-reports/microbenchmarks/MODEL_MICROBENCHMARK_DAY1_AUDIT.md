# IVARA — Day 1 Benchmark Integrity Audit
## Independent Forensic Audit of Model Capability Micro-Benchmark (`gpt-oss:latest` vs. `qwen2.5:14b`)

**Audit Date:** November 16, 2026  
**Auditor:** IVARA Independent Evaluation & Quality Assurance  
**Target Report Audited:** `MODEL_MICROBENCHMARK_DAY1.md`  
**Raw Benchmark Artifacts Audited:**
- `evaluation/microbenchmark_day1/raw_outputs_gpt_oss.json`
- `evaluation/microbenchmark_day1/raw_outputs_qwen2_5.json`
- `evaluation/microbenchmark_day1/benchmark_summary.json`
- `evaluation/microbenchmark_day1/metrics_comparison.json`
- `evaluation/microbenchmark_day1/evaluate_metrics.py`
- `evaluation/microbenchmark_day1/run_microbenchmark.py`

---

## 1. Executive Summary: The "Inverted Precision" Root Cause

The apparent contradiction flagged in the Day 1 report:
```
Reported:
- GPT-OSS: TP=28, FP=16, reported precision=38.36%  (Formula yields: 28/44 = 63.64%)
- Qwen:    TP=28, FP=45, reported precision=63.64%  (Formula yields: 28/73 = 38.36%)
```

### The Forensic Finding:
**The underlying mathematical computation in `metrics_comparison.json` was correct, but the explanatory note in the markdown table was inverted during human report drafting.**

Specifically:
- In `metrics_comparison.json`:
  - **`gpt-oss:latest`**: $\text{TP} = 28$, $\text{FP} = 45$, $\text{Total Extracted} = 73$.
    $$\text{Precision} = \frac{28}{28 + 45} = \frac{28}{73} = 38.36\%$$
  - **`qwen2.5:14b`**: $\text{TP} = 28$, $\text{FP} = 16$, $\text{Total Extracted} = 44$.
    $$\text{Precision} = \frac{28}{28 + 16} = \frac{28}{44} = 63.64\%$$
- In the markdown table of `MODEL_MICROBENCHMARK_DAY1.md`:
  - The column headers were arranged as: **Column 2 = Baseline (`gpt-oss`)**, **Column 3 = Candidate (`qwen2.5`)**.
  - In Column 5 ("Statistical Base & Denominator"), the author transcribed the parenthetical note as:
    `TP: 28 vs 28, FP: 16 vs 45 (Denom: 44 vs 73 extracted)`
  - The author wrote **Candidate vs. Baseline (16 vs 45)** instead of **Baseline vs. Candidate (45 vs 16)**!
  - When readers matched the parenthetical `16 vs 45` against the column headers (GPT-OSS vs Qwen), it falsely appeared that GPT-OSS had 16 FP and Qwen had 45 FP, creating an apparent mathematical inversion.
  - The exact same label inversion occurred in the Entity Extraction row: the note wrote `FP: 14 vs 33 (Denom: 55 vs 76)` (Qwen vs GPT) instead of `33 vs 14` (GPT vs Qwen).

**Conclusion on Inversion:** There was no calculation flaw in the metric script's math; it was a transcription label order reversal in the summary table prose.

However, our deeper audit of `evaluate_metrics.py` revealed **two critical evaluation script bugs** and an unhandled duplicate-prediction issue that distorted the reported numbers.

---

## 2. Evaluation Script Bugs Discovered in `evaluate_metrics.py`

### Bug 1: Predicate Ignored in Relationship Matching (Lenient Matching)
In `evaluate_metrics.py` (lines 100–106):
```python
psrc = p.get("source", "").lower().strip()
ptgt = p.get("target", "").lower().strip()
# ...
if (psrc == gsrc or psrc in gsrc or gsrc in psrc) and (ptgt == gtgt or ptgt in gtgt or gtgt in ptgt):
    match_found = True
```
**Impact:** The evaluation script completely ignored the predicate relation! Any predicted edge connecting the same subject and object was counted as a True Positive, even if the predicate was completely wrong or hallucinated.

*Example:* If gold had `(Arjun, employed_by, Datacore)` and a model predicted `(Arjun, visited, Datacore)`, the script counted it as a True Positive!

### Bug 2: Natural Language Date Mismatch in Temporal Accuracy
In `evaluate_metrics.py` (lines 118–124):
The script searched for raw ISO dates (`"2025-01-14"`, `"2024-03-01"`) in the raw JSON output dump. However:
- Models naturally output dates as formatted text (`"January 14, 2025"`, `"March 2024"`).
- Because string matching checked for the ISO string, the script reported `0.0% (0/13)` temporal accuracy for both models.
- Furthermore, in IVARA's production architecture, temporal claims are extracted deterministically by regex (`src/temporal/claim.py`), not by the LLM. GPT-OSS had polluted entity stores by converting dates into entity nodes (`January 14, 2025`), whereas Qwen correctly stored dates as temporal attributes in the `evidence` field.

### Bug 3: Duplicate Entity Predictions Penalized as False Positives
In passage `ext_06`, Qwen predicted `"Kenji Sato"` twice in its entity list. Because the first occurrence consumed the gold entity index, the second identical occurrence was counted as an ungrounded False Positive. Predictions should be deduplicated per passage prior to scoring.

---

## 3. Independent Recomputation of Metrics

We implemented an independent, rigorous recomputation script (`scratch/detailed_recomputation.py`) operating directly on `raw_outputs_gpt_oss.json` and `raw_outputs_qwen2_5.json`.

We evaluated relationships under two definitions:
1. **Lenient Matching:** Matches subject and object substrings (as originally implemented in `evaluate_metrics.py`).
2. **Strict Semantic Matching:** Requires subject substring match, object substring match, **AND** semantic predicate alignment (checking exact match, normalized synonym match, or inverse match).

### 3.1 Entity Extraction (Gold Total = 46 Entities across 10 Passages)

| Metric | GPT-OSS (Baseline) | Qwen 2.5 14B (Raw) | Qwen 2.5 14B (Deduplicated) | Qwen Net Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Total Predicted** | 76 | 55 | 54 (1 duplicate filtered) | -22 entities |
| **True Positives (TP)** | 43 | 41 | 41 | -2 |
| **False Positives (FP)** | 33 | 14 | 13 | **-20 FP (60.6% reduction)** |
| **False Negatives (FN)** | 3 | 5 | 5 | +2 |
| **Precision** | **56.58%** (43/76) | 74.55% (41/55) | **75.93%** (41/54) | **+19.35%** |
| **Recall** | **93.48%** (43/46) | 89.13% (41/46) | **89.13%** (41/46) | -4.35% |
| **F1 Score** | **70.49%** | 81.19% | **82.00%** | **+11.51%** |

*Root Cause of GPT-OSS False Positives:* GPT-OSS extracts non-entity dates (`"January 14, 2025"`, `"June 2021"`), job titles (`"Director of Operations"`, `"Lead Distributed Systems Architect"`), and common nouns (`"Apartment"`, `"Duplex"`, `"Daily coding"`) as top-level entity nodes. Qwen correctly adheres to named entity boundaries.

---

### 3.2 Relationship Extraction (Gold Total = 40 Relationships across 10 Passages)

#### A. Lenient Matching (Subject + Object Only, Script Baseline)
| Metric | GPT-OSS (Baseline) | Qwen 2.5 14B (Candidate) | Delta |
| :--- | :---: | :---: | :---: |
| **Total Predicted** | 73 | 44 | -29 |
| **Lenient TP** | 28 | 28 | 0 |
| **Lenient FP** | 45 | 16 | **-29 FP (64.4% reduction)** |
| **Lenient FN** | 12 | 12 | 0 |
| **Lenient Precision** | **38.36%** (28/73) | **63.64%** (28/44) | **+25.28%** |
| **Lenient Recall** | **70.00%** (28/40) | **70.00%** (28/40) | 0.00% |
| **Lenient F1 Score** | **49.56%** | **66.67%** | **+17.11%** |

#### B. Strict Semantic Matching (Subject + Object + Semantic Predicate)
| Metric | GPT-OSS (Baseline) | Qwen 2.5 14B (Candidate) | Delta |
| :--- | :---: | :---: | :---: |
| **Total Predicted** | 73 | 44 | -29 |
| **Strict TP** | 25 | 25 | 0 |
| **Strict FP** | 48 | 19 | **-29 FP (60.4% reduction)** |
| **Strict FN** | 15 | 15 | 0 |
| **Strict Precision** | **34.25%** (25/73) | **56.82%** (25/44) | **+22.57%** |
| **Strict Recall** | **62.50%** (25/40) | **62.50%** (25/40) | 0.00% |
| **Strict F1 Score** | **44.25%** | **59.52%** | **+15.27%** |
| **Explicit Hallucinations** | **1 instance** (`Elena Ramos works for Aetheria Labs`) | **0 instances** | **-1 hallucination** |

**Audit Verdict on Extraction Quality:**
Regardless of whether Lenient or Strict matching is used:
- Qwen extracts the **exact same number of valid relationships** (Lenient: 28, Strict: 25).
- Qwen reduces false positive relationships from **48 down to 19** (a **60.4% reduction** in spurious graph noise).
- Strict precision increases by **+22.57%** (56.82% vs 34.25%).
- Qwen suffered **zero explicit factual hallucinations**, whereas GPT-OSS fabricated a false corporate affiliation in `ext_02`.

---

## 4. Manual Passage-by-Passage Disagreement Analysis

We manually inspected all 10 extraction passages and all 10 answering queries using `scratch/inspect_disagreements.py`.

### 4.1 Knowledge Extraction Passages

#### Passage `ext_01` (Team Kickoff)
- **GPT-OSS:** Extracted the calendar date `"January 14, 2025"` as a person/agent node and created the triple `(January 14, 2025) -[kicked off]-> (Project Chronos)`. Also created generic triple `(Project Chronos) -[at]-> (Aetheria Labs)`.
- **Qwen:** Correctly assigned the leadership edge `(SELF) -[leads]-> (Project Chronos)` and omitted dates from the entity registry.

#### Passage `ext_02` (Dual-Company Partnership & Entity Disambiguation)
- **Text:** *"He introduced Elena Ramos, their [Vanguard Robotics'] lead hardware engineer. She is completely distinct from Dr. Elena Rostova, who is our Chief Scientist at Aetheria Labs."*
- **GPT-OSS Hallucination:** Created `(Elena Ramos) -[works for]-> (Aetheria Labs)` (**CRITICAL FACTUAL HALLUCINATION**). Also converted job titles `"Chief Scientist"`, `"Lead Hardware Engineer"`, `"Director of Operations"` into top-level entity nodes.
- **Qwen:** Correctly extracted `(Elena Ramos) -[worksFor]-> (Vanguard Robotics)` and `(Dr. Elena Rostova) -[worksFor]-> (Aetheria Labs)`. Zero hallucinations.

#### Passage `ext_03` (Career Timeline)
- **GPT-OSS:** Extracted dates as entities (`"March 2024"`, `"January 2024"`, `"June 2021"`) and created fragmented intermediary triples: `(SELF) -[worked_as]-> (Senior Data Engineer)` and `(Senior Data Engineer) -[at]-> (Datacore)`.
- **Qwen:** Normalized direct SPO triples: `(SELF) -[employed_by]-> (Datacore)` and `(SELF) -[job_title]-> (Senior Data Engineer)`.

#### Passage `ext_04` (Hardware Preference Shift)
- **GPT-OSS:** Extracted `"Daily coding"` as an entity node and asserted `(Apple Magic Keyboard) -[used_for]-> (daily coding)`.
- **Qwen:** Extracted atomic properties: `(Apple Magic Keyboard) -[has]-> (scissor switches)` and `(SELF) -[use]-> (Apple Magic Keyboard)`.

#### Passage `ext_05` (Multi-Hop Technology Stack)
- **GPT-OSS:** Named entity `"VectorFlow engine"`.
- **Qwen:** Named entity `"VectorFlow"`. Both correctly extracted `deployed_on AWS Kubernetes` and `implemented_in Rust`.

#### Passage `ext_06` (Biographical Attributes)
- **GPT-OSS:** Converted `"June 2022"` and `"Architect"` into entities.
- **Qwen:** Extracted normalized triples: `(Priya Mehta) -[graduate_of]-> (Stanford University)` and `(Kenji Sato) -[works_as]-> (architect)`. (Duplicated `"Kenji Sato"` in raw entity list).

#### Passage `ext_07` (Relocation Chronology)
- **GPT-OSS:** Extracted `"Apartment"` and `"Duplex"` as entities and asserted `(Apartment) -[located_in]-> (Indiranagar)` and `(Duplex) -[located_in]-> (HSR Layout)`.
- **Qwen:** Asserted direct person relations: `(SELF) -[lived_in]-> (Indiranagar / apartment)` and `(SELF) -[moved_to]-> (HSR Layout / duplex)`.

#### Passage `ext_08` (Project Cancellation & Absence of Action)
- Both models correctly identified the negative assertion: Marcus Vance and Priya Sharma did not write production code for Project Hyperion.

#### Passage `ext_09` (Vehicle Purchase & Half Marathon)
- **GPT-OSS:** Extracted `"October 20, 2024"`, `"May 10, 2024"`, and `"1 hour 52 minutes"` as separate entity nodes.
- **Qwen:** Stored the completion time and purchase date in the evidence attribute fields, preserving graph cleanliness.

#### Passage `ext_10` (Healthcare Chronology)
- **GPT-OSS:** Created entity nodes for `"Routine teeth cleaning"`, `"Annual physical exam"`, `"April 4, 2025"`, `"March 12, 2025"`.
- **Qwen:** Correctly captured provider visits: `(SELF) -[visited]-> (Dr. Robert Chen / Dr. Alvarez)`.

---

## 5. Grounded Answering Evaluation Audit

### 5.1 Factual Correctness & Absence of Evidence
- **Answerable Queries (8/8):** Both models achieved 100% factual accuracy (8/8) on direct recall and multi-hop queries.
- **Negative Information (`ans_08`):** *"Has Arjun ever worked at Google or lived in London?"*
  - Both models correctly applied absence-of-evidence framing without hallucination:
    - **Qwen:** *"Based on the recorded conversations, there is no record of Arjun working at Google or living in London."*
    - **GPT-OSS:** *"Based on the recorded conversations, there is no record of Arjun having worked at Google, and there is no record of him having lived in London."*

### 5.2 Multi-Hop Provenance Citation (`ans_05`)
- **Query:** *"What company did the current Lead Architect of CloudMatrix work for between 2021 and early 2024?"*
- **Required Evidence:**
  - `m401`: Arjun joined CloudMatrix in March 2024 as Lead Architect.
  - `m402`: Arjun worked as Senior Data Engineer at Datacore from June 2021 to January 2024.
- **Results:**
  - **GPT-OSS:** Outputted *"He worked for Datacore."* but cited **only `['m402']`**. It omitted `m401`, failing to cite the bridge premise connecting the Lead Architect role to Arjun.
  - **Qwen 2.5:** Outputted a complete synthesis citing **both `['m401', 'm402']`**, maintaining 100% provenance fidelity.

### 5.3 Honest Abstention & Schema Nuance (`ans_09`, `ans_10`)
- **Query `ans_09` (Unanswerable):** Mother's maiden name & high school GPA.
  - Both models returned `status: insufficient_evidence` with clean abstentions.
- **Query `ans_10` (Unanswerable):** Dental surgery cost on April 4, 2025.
  - **GPT-OSS:** In run 1, produced an empty string causing an EOF crash. In run 2, correctly abstained with `status: insufficient_evidence`.
  - **Qwen 2.5:** Accurately analyzed the evidence:
    ```json
    {
      "answer": "insufficient_evidence",
      "limitations": "The provided evidence only mentions that Arjun visited Dr. Alvarez at Downtown Dental for a routine teeth cleaning on April 4, 2025. There is no information about dental surgery or the amount paid out-of-pocket..."
    }
    ```
    *Nuance:* Qwen placed `"insufficient_evidence"` inside the `"answer"` field and omitted the optional `"status"` key. Pydantic defaulted `"status"` to `"answered"`.
    *Remediation for Day 2:* A lightweight post-processor in `src/answering.py`: if `answer.strip().lower() in ("insufficient_evidence", "unanswerable")`, set `status = answer.strip().lower()`.

---

## 6. Hardware, Latency & VRAM Profile Verification

All latency and throughput metrics were verified directly against the recorded timestamps in `raw_outputs_gpt_oss.json` and `raw_outputs_qwen2_5.json`.

| Metric | `gpt-oss:latest` (20.9B MoE) | `qwen2.5:14b` (14.8B Dense) | Delta / Assessment |
| :--- | :---: | :---: | :---: |
| **Model Size on Disk** | 13.0 GB | 9.0 GB | -4.0 GB (30.8% smaller) |
| **GPU VRAM Consumption** | 12,736 MB (~12.4 GB) | 9,555 MB (~9.3 GB) | **-3,181 MB (3.1 GB freed)** |
| **VRAM Headroom (16 GB RTX 5080)** | 3.6 GB headroom | **6.7 GB headroom** | Safe for large 16k KV context |
| **Generation Throughput** | **136.5 tokens/sec** | 63.9 tokens/sec | -72.6 tok/s (MoE speed advantage) |
| **Mean Extraction Latency** | **8,786.7 ms** | 8,976.0 ms | +189.3 ms (+2.1%) |
| **Mean Answering Latency** | 3,051.2 ms | **1,744.4 ms** | **-1,306.8 ms (42.8% faster)** |
| **Overall Mean Turn Latency** | 5,918.9 ms | **5,360.2 ms** | **-558.7 ms (9.4% faster overall)** |
| **JSON Parse Failures** | 1 (EOF crash on `ans_10`, run 1) | **0 failures** | 100% valid JSON generation |

**Explanation of Latency Dynamics:**
While `gpt-oss` has a higher raw token generation rate (136.5 vs 63.9 tok/s) due to sparse MoE activation, it generates **excessive verbosity and 60% more false positive triples**. Because `qwen2.5:14b` produces concise, structured JSON without filler tokens, its **answering latency is 42.8% lower** and its **overall latency is 9.4% faster**.

---

## 7. Sample Size & Statistical Limitations

- **Micro-Benchmark Scope:** $N = 10$ extraction passages (evaluating 46 gold entities and 40 gold relationships) and $N = 10$ answering queries.
- **Diagnostic Adequacy:** The sample size is sufficient to identify consistent structural behaviors:
  - Date-as-entity pollution in GPT-OSS.
  - Job-title and common-noun entity inflation in GPT-OSS.
  - Intermediary node hop proliferation in GPT-OSS.
  - Multi-hop citation omission in GPT-OSS (`ans_05`).
  - Qwen's schema omission nuance (`ans_10`).
- **Statistical Significance Limitation:** Because $N=20$ turns total, formal p-value hypothesis tests on accuracy or macro-F1 have wide confidence intervals. This sample is a **preliminary gating micro-benchmark**, not the final validation.
- **Academic Rigor Requirement:** A full held-out benchmark ($N = 60$ questions across all 8 IVARA query classes) is strictly required on Day 2 before any conclusion is accepted for the final submission.

---

## 8. Summary of Findings & Day 2 Recommendations

### Core Findings:
1. **The apparent inconsistency was a reporting prose error:** The author accidentally reversed the model labels in the table notes (`16 vs 45` instead of `45 vs 16`), whereas the underlying data was correct.
2. **Evaluation script bugs were uncovered:** `evaluate_metrics.py` was leniently ignoring predicates and failing on ISO date string matches.
3. **Qwen's real advantage is verified:** Under strict semantic matching, Qwen outperforms GPT-OSS by **+22.57% in Relationship Precision (56.82% vs 34.25%)**, **+15.27% in Relationship F1 (59.52% vs 44.25%)**, **+19.35% in Entity Precision (75.93% vs 56.58%)**, and eliminates the corporate hallucination observed in GPT-OSS.
4. **VRAM efficiency is verified:** Qwen frees **3.1 GB of VRAM**, ensuring safe 16k context window handling on our 16 GB RTX 5080.

### Recommended Next Steps for Day 2:
1. **Fix `evaluate_metrics.py`:** Update the script to enforce strict semantic predicate matching, deduplicate entity predictions per passage, and correctly evaluate temporal evidence attributes.
2. **Apply Answering Normalizer:** Add the 2-line check in `src/answering.py` to ensure `"insufficient_evidence"` answers automatically map to `status="insufficient_evidence"`.
3. **Proceed to Day 2 Full Evaluation:** Run the full $N=60$ held-out benchmark on the corrected evaluation harness across both models to establish rigorous statistical significance.

