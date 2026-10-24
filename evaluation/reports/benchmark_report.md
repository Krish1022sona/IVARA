# IVARA Milestone 4: Memory QA Benchmark Report

## 1. Executive Summary & Baseline Comparison

Deterministic offline evaluation comparing **Baseline A** (Unfiltered FAISS raw top-k vector retrieval) against **Baseline B** (IVARA Evidence-Filtered Retrieval and Answer Generation Pipeline).

- **Total Synthetic Cases:** 36 (26 answerable, 10 unanswerable/distractor)
- **Evaluation Mode:** Offline deterministic scoring with isolated synthetic memories.

| Metric | Baseline A (Raw Vector top-k) | Baseline B (IVARA Pipeline) | Improvement / Difference |
| :--- | :---: | :---: | :---: |
| **Retrieval Recall@3** | 69.9% | 88.0% | +18.1% |
| **Retrieval Precision@3** | 27.8% | 88.9% | +61.1% |
| **Unanswerable False Positive Rate** | 100.0% (10/10) | 30.0% (3/10) | **-70.0% (Rejection)** |
| **Fact Accuracy** | N/A (Retrieval only) | 87.5% | Evaluated via IVARA |
| **Citation Precision** | N/A | 88.9% | Verifiable Provenance |
| **Unsupported-Claim Rate** | N/A | 0.0% | Zero hallucinated IDs |
| **Abstention F1 Score** | N/A | 82.3% | Prec: 100.0%, Rec: 70.0% |

## 2. Category-Level Performance Breakdown

| Evaluation Category | Cases | IVARA Recall@3 | IVARA Precision@3 | Fact Accuracy | Citation Precision | Failures |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `direct_factual_recall` | 6 | 100.0% | 91.7% | 100.0% | 91.7% | 1 |
| `entity_alias_resolution` | 5 | 100.0% | 100.0% | 100.0% | 100.0% | 0 |
| `multi_memory_evidence` | 5 | 73.3% | 100.0% | 76.7% | 100.0% | 3 |
| `partial_supporting_evidence` | 5 | 100.0% | 100.0% | 93.3% | 100.0% | 0 |
| `relationship_based` | 5 | 100.0% | 90.0% | 100.0% | 90.0% | 1 |
| `similar_irrelevant_distractor` | 5 | 40.0% | 40.0% | 40.0% | 40.0% | 3 |
| `unanswerable_abstention` | 5 | 100.0% | 100.0% | 100.0% | 100.0% | 0 |

## 3. Key Observations & Failure Analysis

1. **Unfiltered Vector Baseline Flaw:** Baseline A has a **100% False Positive Retrieval Rate** on unanswerable and distractor queries because FAISS unconditionally returns top-k nearest vectors. In contrast, IVARA's relevance thresholding achieves an **effective rejection rate**, preventing spurious memories from entering the answer prompt.
2. **Multi-Memory Recall:** Queries requiring 3 disparate memories (`eval_17`) achieve partial recall at k=3 when multiple episodes compete for top-k slots, accurately reflecting the capacity limits of single-round top-k retrieval.
3. **Zero Fabricated Citations:** The deterministic citation post-processor ensures that unsupported or hallucinated IDs are eliminated, resulting in an unsupported claim rate of 0.0%.

## 4. Methodology Notes
- **Dataset:** 36 version-controlled synthetic cases covering direct recall, aliases, relations, multi-memory, partial evidence, unanswerable questions, and distractors.
- **Safety:** Real user memories in `data/memories.json` were strictly isolated and never accessed during benchmark execution.
- **Empty Denominators:** Handled via explicit conventions documented in `evaluation/metrics.py`.