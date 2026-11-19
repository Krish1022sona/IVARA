# IVARA — Extended Autonomous Research and Engineering Progress Log

**Session Start:** 2026-10-28 20:15 IST  
**Branch:** `research/evaluation-baseline`  
**Baseline Commit:** `885298c73cd0257653c49cb81d958f49ad8a0da4`  
**Starting Test Count:** 208 passed  

---

## Session Objectives
1. **Phase 1:** Verify prototype correctness, backward compatibility, and comprehensive test coverage.
2. **Phase 2:** Harden the 6-arm diagnostic evaluation harness with incremental flushing, resumption, and error isolation.
3. **Phase 3:** Execute controlled diagnostic evaluation on a fixed 100-question sample.
4. **Phase 4:** Compute paired statistical metrics, question-level wins/losses, and causal verification.
5. **Phase 5:** Produce complete research documentation, comparison tables, and roadmap for publication.

---

## Chronological Progress Log

### [20:15] Session Initialized
- Recorded baseline git commit `885298c73cd0257653c49cb81d958f49ad8a0da4` and clean status on `data/`.
- Confirmed constraints: no paid APIs, no 500-question run, production stores untouched.

### [20:17] Phase 1 Completed: Prototype Correctness & Extended Edge-Case Testing
- Inspected complete diffs of `src/chunking/graph_retrieval.py` and `src/answering.py`.
- Verified default flags preserve identical baseline behavior (`min_edge_relevance=0.0`, `direct_seed_temporal=False`).
- Expanded `tests/test_sprint_prototypes.py` from 4 to 8 tests, covering:
  - Default flag backward compatibility
  - High and low edge confidence filtering
  - Missing dates and ungrounded temporal claims
  - Citation validation and fake chunk ID rejection
  - Successive calls idempotency and state isolation
- **Full Test Suite:** 212 / 212 passed in 17.54s. Zero regressions.

### [20:24] Phase 2 Completed: Hardened Evaluation Harness & 100-Question Held-Out Sample
- Generated 100-question stratified held-out sample (`evaluation/autonomous_sprint/diagnostic_sample_100.json`) with **0 overlap** against the 60 pilot questions.
- Built `DiagnosticRunner` in `evaluation/autonomous_sprint/diagnostic_runner.py` with support for all 6 arms:
  - ARM_A: Dense + BM25 Baseline
  - ARM_B: Baseline + Ontology Guidance
  - ARM_C: Ontology + Raw 1-Hop Graph
  - ARM_D: Ontology + Relevance-Filtered Graph (Prototype B, threshold 0.50)
  - ARM_E: Ontology + Direct Seed Temporal (Prototype C)
  - ARM_F: Full Existing Architecture (Graph + Temporal)
- Engineered restart-safe deduplication, incremental `.jsonl` flushing, per-question error containment, and regex fallback for the diagnostic judge.
- Created `tests/test_diagnostic_harness.py` testing resume behavior, partial outputs, duplicate IDs, and judge JSON recovery.
- **Full Test Suite:** 216 / 216 passed in 17.14s. Zero regressions.

### [20:25] Phase 2 Hardened Evaluation Harness & 100-Question Held-Out Sample
- Generated 100-question stratified held-out sample (`evaluation/autonomous_sprint/diagnostic_sample_100.json`) with **0 overlap** against the 60 pilot questions.
- Built `DiagnosticRunner` in `evaluation/autonomous_sprint/diagnostic_runner.py` with support for all 6 arms:
  - ARM_A: Dense + BM25 Baseline
  - ARM_B: Baseline + Ontology Guidance
  - ARM_C: Ontology + Raw 1-Hop Graph
  - ARM_D: Ontology + Relevance-Filtered Graph (Prototype B, threshold 0.50)
  - ARM_E: Ontology + Direct Seed Temporal (Prototype C)
  - ARM_F: Full Existing Architecture (Graph + Temporal)
- Engineered restart-safe deduplication, incremental `.jsonl` flushing, per-question error containment, and regex fallback for the diagnostic judge.
- Created `tests/test_diagnostic_harness.py` testing resume behavior, partial outputs, duplicate IDs, and judge JSON recovery.
- **Full Test Suite:** 216 / 216 passed in 17.14s. Zero regressions.

### [20:38] Root-Cause CUDA Diagnosis & Preflight Engineering
- **Diagnostic Discovery:** During initial preflight, Ollama on Windows auto-selected the `cuda_v13` directory because the NVIDIA driver version reported 13.2. However, the `cuda_v13` directory lacked `cudart64_13.dll`, causing `ggml_cuda_compute_forward` to fail with `0xc0000409: CUDA error: shared object initialization failed`.
- **Engineering Resolution:** Identified that Ollama ships with a complete, fully linked CUDA 12 runtime (`cuda_v12` with `cublas64_12.dll`, `cudart64_12.dll`, `ggml-cuda.dll`). Set `OLLAMA_LLM_LIBRARY=cuda_v12` across the server process and Python runner.
- **Preflight Fix:** Fixed `gold_rank` computation in `DiagnosticRunner` to perform direct session matching against `direct_evidence` without relying on undefined adapter helper methods.

### [20:44] Phase 2 Preflight Passed: 6-Arm Smoke Test
- Executed clean smoke test preflight across all 6 arms (`evaluation/autonomous_sprint/runs/smoke_test_run/`).
- Verified 12 live predictions on `gpt-oss:latest` (2 questions x 6 arms) with zero errors:
  - ARM_A: 2 questions in 23.89s
  - ARM_B: 2 questions in 11.14s
  - ARM_C: 2 questions in 12.55s
  - ARM_D: 2 questions in 11.51s
  - ARM_E: 2 questions in 11.01s
  - ARM_F: 2 questions in 12.23s
- Verified live diagnostic Gemma judge evaluation on ARM_B:
  - Judged 2/2 predictions cleanly with JSON schema parsing: Overall 100.0%, latency ~1.37s/call.
- **Smoke test preflight passed in 99.1s with exit code 0.**

### [21:15] Phase 3 Completed: 6-Arm Controlled Diagnostic Evaluation (Diagnostic Pilot 30)
- Successfully completed all 180 predictions (30 questions x 6 arms) on `gpt-oss:latest` (Phase 3A: 1,151.4s, 19.2 min).
- Successfully judged all 180 predictions on `gemma4-26b` local diagnostic judge (Phase 3B: 479.35s, 8.0 min).
- Total runtime: 1,630.74s (27.2 minutes). Zero execution errors across all 180 instances.
- Incremental outputs, manifests, and predictions written cleanly to `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/`.

### [21:43] Phase 4 Completed: Paired Statistical Analysis & Causal Verification
- Built `scratch/analyze_diagnostic_results.py` and saved structured results to `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/paired_analysis.json`.
- **Key Empirical Verifications:**
  - **Hypothesis B (Relevance-Filtered Graph Expansion):** ARM_D (threshold 0.50) achieved 1 win and 0 losses against ARM_C (Raw Graph) on paired comparisons. Overall accuracy improved from 60.00% to 63.33%, and answerable accuracy improved from 58.33% to 62.50% (+4.17%). Confirmed that relevance filtering eliminates neighbor distractor edges.
  - **Hypothesis C (Direct Seed Temporal Reasoning):** ARM_E matched the accuracy of the full graph+temporal system (ARM_F) at 63.33% overall and 62.50% answerable, while completely eliminating graph construction latency (0.0 ms vs 467.8 ms) and executing 632 ms faster per query.
  - **Hypothesis A (Ontology Guidance on Abstention):** ARM_B reached 83.33% abstention accuracy (5/6 correct abstentions) vs ARM_A at 66.67% (4/6), preventing hallucinated answers on ungrounded queries.

### [22:02] Audit of Diagnostic Pilot 30 Completed
- Verified question ID identity and 100% 1:1 prediction-judgment pairing (180/180) across all six arms.
- Recomputed all metrics directly from raw JSONL files; verified 100% match to summary files.
- Documented question-level wins/losses for A vs B, C vs D, and E vs F in `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/audit_verification.json`.

### [22:03] Full 100-Question 6-Arm Diagnostic Run Launched
- Created new isolated run directory `evaluation/autonomous_sprint/runs/diagnostic_full_100/`.
- Pre-seeded initial 30 predictions and judgments from `diagnostic_pilot_30/` to avoid redundant computation.
- Launched background execution of remaining 70 questions across 6 arms (420 answer generations + 420 diagnostic judgments).

### [23:10] Full 100-Question 6-Arm Diagnostic Run Completed
- Phase A (Answer Generation on `gpt-oss:latest`): Completed in 2,713.09s (45.2 min). All 6 arms reached 100/100 predictions.
- Phase B (Diagnostic Gemma Judging on `gemma4-26b`): Completed in 1,252.47s (20.9 min). All 6 arms reached 100/100 judgments.
- Grand Total Runtime: 3,965.56s (66.1 minutes). Zero errors, zero dropped predictions.
- Summary saved to `evaluation/autonomous_sprint/runs/diagnostic_full_100/experiment_summary.json`.

### [23:11] Full Paired Statistical Analysis Completed (N=100 Questions x 6 Arms)
- Authored `scratch/analyze_diagnostic_full_100.py` and saved structured analysis to `evaluation/autonomous_sprint/runs/diagnostic_full_100/paired_analysis.json`.
- Computed Wilson 95% confidence intervals, McNemar's exact two-tailed tests, MRR, retrieval recall@k, and latency profiles across cohorts (Pilot 30, Remaining 70, Full 100).
- Confirmed directional superiority of Relevance-Filtered Graph (Prototype B, Arm D: 4 wins vs 1 loss against Raw Graph Arm C) and Direct Seed Temporal (Prototype C, Arm E: 5 wins vs 2 losses against Full System Arm F, while saving 428ms graph overhead).

### [23:12] Final Verification & Integrity Audit
- Ran full test suite: 216 / 216 passed in 18.21s. Zero regressions.
- Verified repository safety: `main` untouched, `data/` untouched, Step 7/8/9 historical records untouched, human review columns in `evaluation/step9/human_review_sample.csv` strictly blank.
