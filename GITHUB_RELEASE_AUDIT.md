# IVARA — GitHub Release Audit & Repository Inventory

## 1. Executive Summary

- **Total Tracked Files:** 260
- **Total Untracked Files:** 468
- **Current Git Branch:** `refactor/repo-cleanup`
- **Frozen Release Baseline:** `v1.0.0-freeze` (`12cba9b99e5f65b61a293c960d5d8a4145572b32`)
- **Audit Date:** 2026-11-19

### Category Breakdown

| Category | Description | Tracked | Untracked | Total Files | Total Size |
|:---|:---|:---:|:---:|:---:|:---:|
| **A** | Essential Application Source | 70 | 0 | 70 | 0.70 MB |
| **B** | Tests and Test Harnesses | 50 | 0 | 50 | 0.54 MB |
| **C** | Living Documentation | 26 | 2 | 28 | 0.54 MB |
| **D** | Research, Evaluation & Reproducibility Artifacts | 111 | 441 | 552 | 426.59 MB |
| **E** | Demonstration Assets | 3 | 0 | 3 | 0.03 MB |
| **F** | Generated Caches and Build Files | 0 | 0 | 0 | 0.00 MB |
| **G** | Sensitive / Private Data | 0 | 16 | 16 | 0.93 MB |
| **H** | Disposable Scratch and Duplicate Files | 0 | 9 | 9 | 4.28 MB |
| **Total** | | **260** | **468** | **728** | **433.61 MB** |

## 2. Category Details and Recommendations

### Category A: Essential Application Source

- **File Count:** 70
- **Total Size:** 0.70 MB

**Release Action:** Retain and track all core source files. Ensure modularity and clean imports.

<details><summary>Click to view file list</summary>

| Path | Tracked | Size (KB) |
|:---|:---:|:---:|
| `.gitignore` | Yes | 0.3 |
| `data/entities.json` | Yes | 0.5 |
| `data/memories.json` | Yes | 3.2 |
| `main.py` | Yes | 16.0 |
| `pytest.ini` | Yes | 0.2 |
| `seed.py` | Yes | 2.3 |
| `src/answering.py` | Yes | 53.2 |
| `src/chat/__init__.py` | Yes | 0.2 |
| `src/chat/engine.py` | Yes | 12.7 |
| `src/chat/terminal.py` | Yes | 4.4 |
| `src/chunking/__init__.py` | Yes | 0.8 |
| `src/chunking/evidence_validated_retrieval.py` | Yes | 24.9 |
| `src/chunking/graph_retrieval.py` | Yes | 31.6 |
| `src/chunking/hierarchical_retrieval.py` | Yes | 18.9 |
| `src/chunking/hybrid_pipeline.py` | Yes | 14.3 |
| `src/chunking/ontology_retrieval.py` | Yes | 4.4 |
| `src/chunking/query_aware_retrieval.py` | Yes | 23.3 |
| `src/chunking/retrieval.py` | Yes | 16.2 |
| `src/chunking/turn_chunker.py` | Yes | 8.6 |
| `src/conversations/__init__.py` | Yes | 0.2 |
| `src/conversations/store.py` | Yes | 37.2 |
| `src/embedding.py` | Yes | 3.5 |
| `src/extraction/__init__.py` | Yes | 0.8 |
| `src/extraction/extractor.py` | Yes | 3.5 |
| `src/extraction/pipeline.py` | Yes | 9.6 |
| `src/extraction/policy.py` | Yes | 18.8 |
| `src/graph/__init__.py` | Yes | 0.7 |
| `src/graph/builder.py` | Yes | 16.1 |
| `src/graph/chunker.py` | Yes | 5.3 |
| `src/graph/retrieval.py` | Yes | 29.3 |
| `src/graph/schema.py` | Yes | 7.6 |
| `src/graph/store.py` | Yes | 17.5 |
| `src/graph/traversal.py` | Yes | 8.4 |
| `src/memory.py` | Yes | 5.6 |
| `src/ontology/__init__.py` | Yes | 0.0 |
| `src/ontology/entities.json` | Yes | 0.4 |
| `src/ontology/entity_resolver.py` | Yes | 3.5 |
| `src/ontology/entity_store.py` | Yes | 1.0 |
| `src/ontology/loader.py` | Yes | 1.6 |
| `src/ontology/relationship_resolver.py` | Yes | 4.5 |
| `src/ontology/relationships.json` | Yes | 0.3 |
| `src/ontology/rules.json` | Yes | 0.3 |
| `src/ontology/validator.py` | Yes | 19.0 |
| `src/ontology/vocabulary.json` | Yes | 6.3 |
| `src/ontology/vocabulary.py` | Yes | 14.6 |
| `src/query/__init__.py` | Yes | 0.2 |
| `src/query/decomposition.py` | Yes | 13.1 |
| `src/reasoning/__init__.py` | Yes | 0.7 |
| `src/reasoning/contradiction.py` | Yes | 15.6 |
| `src/reasoning/evidence_gate.py` | Yes | 6.9 |
| `src/reasoning/evidence_integrator.py` | Yes | 11.5 |
| `src/retrieval.py` | Yes | 9.5 |
| `src/temporal/__init__.py` | Yes | 2.0 |
| `src/temporal/calendar_arithmetic.py` | Yes | 11.2 |
| `src/temporal/claim.py` | Yes | 15.1 |
| `src/temporal/event_extractor.py` | Yes | 12.4 |
| `src/temporal/event_graph.py` | Yes | 7.5 |
| `src/temporal/event_graph_traversal.py` | Yes | 9.5 |
| `src/temporal/relationship_builder.py` | Yes | 10.9 |
| `src/temporal/resolution.py` | Yes | 7.3 |
| `src/temporal/supersession.py` | Yes | 10.9 |
| `src/unified_retrieval/__init__.py` | Yes | 1.3 |
| `src/unified_retrieval/episodic.py` | Yes | 14.2 |
| `src/unified_retrieval/evidence_selector.py` | Yes | 29.6 |
| `src/unified_retrieval/fusion.py` | Yes | 9.5 |
| `src/unified_retrieval/graph.py` | Yes | 10.0 |
| `src/unified_retrieval/graph_safeguards.py` | Yes | 16.8 |
| `src/unified_retrieval/pipeline.py` | Yes | 21.0 |
| `src/unified_retrieval/router.py` | Yes | 18.8 |
| `src/unified_retrieval/semantic.py` | Yes | 4.4 |

</details>

### Category B: Tests and Test Harnesses

- **File Count:** 50
- **Total Size:** 0.54 MB

**Release Action:** Retain and track all 50 test files. Ensure 525/525 tests pass in CI/local runs.

<details><summary>Click to view file list</summary>

| Path | Tracked | Size (KB) |
|:---|:---:|:---:|
| `tests/test_adaptive_evidence.py` | Yes | 37.9 |
| `tests/test_adaptive_router.py` | Yes | 15.1 |
| `tests/test_answering.py` | Yes | 15.3 |
| `tests/test_benchmark.py` | Yes | 9.1 |
| `tests/test_benchmark_metrics.py` | Yes | 8.5 |
| `tests/test_bounded_graph_expansion.py` | Yes | 16.2 |
| `tests/test_calibrated_answering.py` | Yes | 16.1 |
| `tests/test_candidate_pipeline.py` | Yes | 14.7 |
| `tests/test_conversation_store.py` | Yes | 7.4 |
| `tests/test_core_intelligence.py` | Yes | 14.1 |
| `tests/test_diagnostic_harness.py` | Yes | 7.1 |
| `tests/test_embedding_offline.py` | Yes | 1.5 |
| `tests/test_entity_resolver.py` | Yes | 0.8 |
| `tests/test_episodic_memory.py` | Yes | 19.2 |
| `tests/test_evaluator_audit.py` | Yes | 4.7 |
| `tests/test_event_graph.py` | Yes | 15.0 |
| `tests/test_evidence_validated_retrieval.py` | Yes | 10.7 |
| `tests/test_extraction_regression.py` | Yes | 4.3 |
| `tests/test_extractor.py` | Yes | 0.8 |
| `tests/test_graph_ablation.py` | Yes | 7.3 |
| `tests/test_graph_store.py` | Yes | 12.3 |
| `tests/test_hierarchical_retrieval.py` | Yes | 8.8 |
| `tests/test_human_review_util.py` | Yes | 2.0 |
| `tests/test_hybrid_retrieval.py` | Yes | 9.9 |
| `tests/test_improvement_sprint_integration.py` | Yes | 6.2 |
| `tests/test_local_abstention_heuristic.py` | Yes | 4.9 |
| `tests/test_longmemeval_adapter.py` | Yes | 14.2 |
| `tests/test_longmemeval_runner.py` | Yes | 7.8 |
| `tests/test_memory_graph.py` | Yes | 23.0 |
| `tests/test_memory_policy.py` | Yes | 10.5 |
| `tests/test_ontology.py` | Yes | 21.2 |
| `tests/test_open_world_ontology.py` | Yes | 15.1 |
| `tests/test_pipeline.py` | Yes | 2.3 |
| `tests/test_pipeline_offline.py` | Yes | 13.6 |
| `tests/test_pre_freeze_hardening.py` | Yes | 15.0 |
| `tests/test_preflight_gate.py` | Yes | 1.9 |
| `tests/test_query_aware_retrieval.py` | Yes | 11.5 |
| `tests/test_query_decomposition.py` | Yes | 13.0 |
| `tests/test_real_memories.py` | Yes | 0.5 |
| `tests/test_realistic_scenarios.py` | Yes | 16.9 |
| `tests/test_retrieval.py` | Yes | 0.7 |
| `tests/test_retrieval_grounded.py` | Yes | 13.6 |
| `tests/test_sprint_prototypes.py` | Yes | 16.0 |
| `tests/test_step9_validation.py` | Yes | 3.4 |
| `tests/test_temporal_integration.py` | Yes | 15.6 |
| `tests/test_temporal_phase1_interventions.py` | Yes | 4.5 |
| `tests/test_temporal_supersession.py` | Yes | 18.2 |
| `tests/test_turn_chunking.py` | Yes | 8.0 |
| `tests/test_unified_retrieval.py` | Yes | 18.8 |
| `tests/test_user_facing_mvp.py` | Yes | 17.7 |

</details>

### Category C: Living Documentation

- **File Count:** 28
- **Total Size:** 0.54 MB

**Release Action:** Maintain living documentation in `docs/` and root. Replace absolute paths with relative links.

<details><summary>Click to view file list</summary>

| Path | Tracked | Size (KB) |
|:---|:---:|:---:|
| `INSTALLATION_GUIDE.md` | Yes | 3.7 |
| `IVARA_FINAL_PRODUCT_AUDIT.md` | No | 22.6 |
| `KNOWN_LIMITATIONS.md` | Yes | 13.9 |
| `README.md` | Yes | 11.2 |
| `REPOSITORY_CLEANUP_REPORT.md` | Yes | 17.3 |
| `audits/IVARA_Comprehensive_Audit_and_Handover_Report.md` | Yes | 48.9 |
| `docs/architecture/EPISODIC_MEMORY_DESIGN.md` | Yes | 9.6 |
| `docs/architecture/IVARA_ARCHITECTURE_EVOLUTION.md` | Yes | 14.7 |
| `docs/architecture/MILESTONE_3_2_DESIGN.md` | Yes | 21.5 |
| `docs/architecture/PHASE2B1_DESIGN.md` | Yes | 14.0 |
| `docs/architecture/PHASE3_1_DESIGN.md` | Yes | 7.0 |
| `docs/architecture/PHASE3_DESIGN.md` | Yes | 11.0 |
| `docs/architecture/UNIFIED_RETRIEVAL_DESIGN.md` | Yes | 12.6 |
| `docs/development/IVARA_CHALLENGES_AND_SOLUTIONS.md` | Yes | 18.2 |
| `docs/development/IVARA_DOCUMENTATION_AUDIT.md` | Yes | 57.5 |
| `docs/development/IVARA_DOCUMENTATION_STRUCTURE.md` | Yes | 14.6 |
| `docs/development/IVARA_HISTORY_RECONSTRUCTION_REPORT.md` | Yes | 21.0 |
| `docs/development/IVARA_PROJECT_TIMELINE.md` | Yes | 16.5 |
| `docs/evaluation/FINAL_BAKEOFF_INTEGRITY_AUDIT.md` | Yes | 18.8 |
| `docs/evaluation/FINAL_BAKEOFF_REPORT.md` | Yes | 16.9 |
| `docs/evaluation/FINAL_ROUTER_VERIFICATION.md` | Yes | 15.4 |
| `docs/evaluation/IVARA_BENCHMARK_AUDIT.md` | Yes | 20.8 |
| `docs/presentation/IVARA_PRESENTATION_STORY.md` | Yes | 34.0 |
| `docs/release/FINALIZATION_AUDIT.md` | Yes | 24.5 |
| `docs/release/FINAL_FREEZE_GO_NO_GO.md` | Yes | 11.7 |
| `docs/release/IVARA_RELEASE_CHECKLIST.md` | Yes | 4.4 |
| `docs/release/PRE_FREEZE_HARDENING_REPORT.md` | Yes | 13.3 |
| `docs/research/IVARA_IEEE_RESEARCH_PAPER_BRIEF.md` | No | 52.9 |

</details>

### Category D: Research, Evaluation & Reproducibility Artifacts

- **File Count:** 552
- **Total Size:** 426.59 MB

**Release Action:** Preserve historical research and evaluation scripts. Exclude files > 100 MB (`longmemeval_s_cleaned.json` is 264.5 MB) via `.gitignore` and document external hosting.

<details><summary>Click to view file list</summary>

| Path | Tracked | Size (KB) |
|:---|:---:|:---:|
| `archive/experiments/IVARA_FINAL_EVALUATION_REPORT.md` | Yes | 9.7 |
| `archive/experiments/IVARA_LONG_CONTEXT_EVALUATION_REPORT.md` | Yes | 23.6 |
| `archive/experiments/LONGMEMEVAL_FULL_BASELINE.md` | Yes | 7.8 |
| `archive/experiments/LONGMEMEVAL_INTEGRATION_REPORT.md` | Yes | 14.5 |
| `archive/experiments/LONGMEMEVAL_PHASE1_EXPERIMENTS.md` | Yes | 14.1 |
| `archive/experiments/NEXT_EXPERIMENT_PLAN.md` | Yes | 9.4 |
| `archive/experiments/PHASE1_TEMPORAL_EXPERIMENT_REPORT.md` | Yes | 10.2 |
| `archive/experiments/PHASE2A1_ADAPTIVE_RETRIEVAL_REPORT.md` | Yes | 17.9 |
| `archive/experiments/PHASE2A_HIERARCHICAL_RETRIEVAL_REPORT.md` | Yes | 15.2 |
| `archive/experiments/PHASE2B1_EVALUATION_REPORT.md` | Yes | 8.9 |
| `archive/experiments/PHASE2B1_IMPLEMENTATION_REPORT.md` | Yes | 9.2 |
| `archive/experiments/PHASE2B2_ABLATION_REPORT.md` | Yes | 8.1 |
| `archive/experiments/PHASE2B2_GRAPH_QUALITY_REPORT.md` | Yes | 9.0 |
| `archive/experiments/PHASE3_1_EVALUATION_REPORT.md` | Yes | 7.3 |
| `archive/experiments/PHASE3_EVALUATION_REPORT.md` | Yes | 9.0 |
| `archive/historical-reports/microbenchmarks/MODEL_BENCHMARK_DAY2.md` | Yes | 20.5 |
| `archive/historical-reports/microbenchmarks/MODEL_CAPABILITY_AUDIT.md` | Yes | 19.6 |
| `archive/historical-reports/microbenchmarks/MODEL_MICROBENCHMARK_DAY1.md` | Yes | 12.0 |
| `archive/historical-reports/microbenchmarks/MODEL_MICROBENCHMARK_DAY1_5.md` | Yes | 15.2 |
| `archive/historical-reports/microbenchmarks/MODEL_MICROBENCHMARK_DAY1_AUDIT.md` | Yes | 17.9 |
| `archive/historical-reports/microbenchmarks/PRESENTATION_OUTLINE.md` | Yes | 10.3 |
| `archive/historical-reports/milestones/EPISODIC_MILESTONE_1_REPORT.md` | Yes | 13.8 |
| `archive/historical-reports/milestones/EPISODIC_MILESTONE_1_REVIEW.md` | Yes | 7.7 |
| `archive/historical-reports/milestones/HYBRID_INTEGRATION_VALIDATION.md` | Yes | 16.9 |
| `archive/historical-reports/milestones/IVARA_FINAL_ENGINEERING_REPORT.md` | Yes | 13.4 |
| `archive/historical-reports/milestones/IVARA_FINAL_HANDOFF.md` | Yes | 7.7 |
| `archive/historical-reports/milestones/IVARA_FINAL_RELEASE_READINESS_REPORT.md` | Yes | 6.2 |
| `archive/historical-reports/milestones/MILESTONE_3_1_AUDIT_REPORT.md` | Yes | 20.9 |
| `archive/historical-reports/milestones/MILESTONE_3_2_BATCH_1_REPORT.md` | Yes | 8.2 |
| `archive/historical-reports/milestones/MILESTONE_3_2_BATCH_2_3_REPORT.md` | Yes | 14.7 |
| `archive/historical-reports/milestones/MILESTONE_3_2_BATCH_4_REPORT.md` | Yes | 12.1 |
| `archive/historical-reports/milestones/ONTOLOGY_MILESTONE_2_REPORT.md` | Yes | 12.4 |
| `archive/historical-reports/milestones/PROJECT_REPORT.md` | Yes | 19.6 |
| `archive/historical-reports/milestones/STEP6_TEMPORAL_INTEGRATION_REPORT.md` | Yes | 13.0 |
| `archive/historical-reports/milestones/UNIFIED_RETRIEVAL_MILESTONE_3_REPORT.md` | Yes | 23.5 |
| `archive/research-notes/ARCHITECTURE_FAILURE_ANALYSIS.md` | Yes | 20.1 |
| `archive/research-notes/FINAL_COMPLETENESS_AUDIT.md` | Yes | 8.8 |
| `archive/research-notes/IVARA_FAILURE_ANALYSIS.md` | Yes | 8.5 |
| `archive/research-notes/LONGMEMEVAL_FAILURE_ANALYSIS.md` | Yes | 14.0 |
| `archive/research-notes/LONGMEMEVAL_FORENSIC_ANALYSIS.md` | Yes | 23.0 |
| `archive/research-notes/PHASE2A1_FAILURE_FORENSICS.md` | Yes | 29.0 |
| `archive/research-notes/PHASE2B2_FAILURE_ANALYSIS.md` | Yes | 8.1 |
| `archive/research-notes/PHASE3_1_FAILURE_ANALYSIS.md` | Yes | 8.7 |
| `archive/research-notes/PHASE3_FAILURE_ANALYSIS.md` | Yes | 7.0 |
| `archive/research-notes/TEMPORAL_REASONING_ARCHITECTURE_AUDIT.md` | Yes | 44.3 |
| `evaluation/EVALUATION_CAMPAIGN_V1_REPORT.md` | No | 15.6 |
| `evaluation/FINAL_CORE_ENGINEERING_REPORT.md` | Yes | 12.4 |
| `evaluation/GRAPH_ABLATION_PROTOCOL.md` | No | 12.2 |
| `evaluation/GRAPH_ABLATION_REPORT.md` | No | 8.9 |
| `evaluation/INTEGRATED_ONTOLOGY_GRAPH_TEMPORAL_AUDIT.md` | No | 33.0 |
| `evaluation/LONGMEMEVAL_BASELINE_REPORT.md` | No | 7.6 |
| `evaluation/LONGMEMEVAL_CHUNKING_ABLATION.md` | No | 11.8 |
| `evaluation/LONGMEMEVAL_FAILURE_ANALYSIS.md` | No | 15.7 |
| `evaluation/LONGMEMEVAL_PROTOCOL_AUDIT.md` | No | 13.7 |
| `evaluation/LONGMEMEVAL_README.md` | No | 4.2 |
| `evaluation/LONGMEMEVAL_SOURCES.md` | No | 7.7 |
| `evaluation/MEM0_BENCHMARK_AND_ARCHITECTURE_AUDIT.md` | Yes | 22.4 |
| `evaluation/MEM0_COMPARISON_PROTOCOL.md` | No | 4.3 |
| `evaluation/README_longmemeval_official.md` | No | 15.6 |
| `evaluation/STEP5_INTEGRATION_AND_EVALUATION_READINESS.md` | No | 18.9 |
| `evaluation/STEP6_TEMPORAL_INTEGRATION_REPORT.md` | No | 14.0 |
| `evaluation/TRUSTED_EVALUATION_BASELINE_REPORT.md` | Yes | 13.2 |
| `evaluation/__init__.py` | Yes | 0.0 |
| `evaluation/adapters/longmemeval_adapter.py` | No | 14.8 |
| `evaluation/autonomous_sprint/FINALIZATION_SPRINT_REPORT.md` | No | 18.8 |
| `evaluation/autonomous_sprint/SPRINT_FINAL_REPORT.md` | No | 15.3 |
| `evaluation/autonomous_sprint/SPRINT_PROGRESS.md` | No | 9.0 |
| `evaluation/autonomous_sprint/SPRINT_REPORT.md` | No | 11.8 |
| `evaluation/autonomous_sprint/analyze_experiment_40.py` | No | 6.9 |
| `evaluation/autonomous_sprint/analyze_failures_deep.py` | No | 2.5 |
| `evaluation/autonomous_sprint/audit_17_failures.py` | No | 2.4 |
| `evaluation/autonomous_sprint/check_session_diversity.py` | No | 1.0 |
| `evaluation/autonomous_sprint/detailed_failure_inspection.py` | No | 2.5 |
| `evaluation/autonomous_sprint/diagnostic_runner.py` | Yes | 22.5 |
| `evaluation/autonomous_sprint/diagnostic_sample_100.json` | No | 52584.0 |
| `evaluation/autonomous_sprint/diagnostic_sample_100_manifest.json` | No | 5.0 |
| `evaluation/autonomous_sprint/experiment_plan.md` | No | 6.0 |
| `evaluation/autonomous_sprint/failure_patterns.json` | No | 242.7 |
| `evaluation/autonomous_sprint/human_review_util.py` | No | 3.2 |
| `evaluation/autonomous_sprint/inspect_discordant.py` | No | 1.2 |
| `evaluation/autonomous_sprint/inspect_overabstention.py` | No | 0.7 |
| `evaluation/autonomous_sprint/inspect_q_sessions.py` | No | 0.7 |
| `evaluation/autonomous_sprint/performance_audit.md` | No | 5.7 |
| `evaluation/autonomous_sprint/preflight.py` | No | 5.3 |
| `evaluation/autonomous_sprint/print_17_audit.py` | No | 1.1 |
| `evaluation/autonomous_sprint/run_core_intelligence_eval.py` | Yes | 11.1 |
| `evaluation/autonomous_sprint/run_diagnostic_100.py` | No | 4.4 |
| `evaluation/autonomous_sprint/run_diagnostic_experiment.py` | No | 4.2 |
| `evaluation/autonomous_sprint/run_finalization_experiment.py` | No | 4.9 |
| `evaluation/autonomous_sprint/run_smoke_test.py` | No | 1.7 |
| `evaluation/autonomous_sprint/runs/core_intelligence_eval/core_intelligence_evaluation_summary.json` | Yes | 4.1 |
| `evaluation/autonomous_sprint/runs/core_intelligence_eval/diagnostic_run_manifest.json` | Yes | 1.2 |
| `evaluation/autonomous_sprint/runs/core_intelligence_eval/judge_results_ARM_5_CORE_INTELLIGENCE.jsonl` | Yes | 19.7 |
| `evaluation/autonomous_sprint/runs/core_intelligence_eval/predictions_ARM_5_CORE_INTELLIGENCE.jsonl` | Yes | 40.0 |
| `evaluation/autonomous_sprint/runs/core_intelligence_eval/summary_ARM_5_CORE_INTELLIGENCE.json` | Yes | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/diagnostic_run_manifest.json` | No | 2.7 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/experiment_summary.json` | No | 2.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/judge_results_ARM_A.jsonl` | No | 49.5 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/judge_results_ARM_B.jsonl` | No | 48.8 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/judge_results_ARM_C.jsonl` | No | 49.4 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/judge_results_ARM_D.jsonl` | No | 49.1 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/judge_results_ARM_E.jsonl` | No | 48.8 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/judge_results_ARM_F.jsonl` | No | 48.1 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/paired_analysis.json` | No | 14.2 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/predictions_ARM_A.jsonl` | No | 99.9 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/predictions_ARM_B.jsonl` | No | 97.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/predictions_ARM_C.jsonl` | No | 112.0 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/predictions_ARM_D.jsonl` | No | 104.6 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/predictions_ARM_E.jsonl` | No | 90.0 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/predictions_ARM_F.jsonl` | No | 88.7 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/summary_ARM_A.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/summary_ARM_B.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/summary_ARM_C.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/summary_ARM_D.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/summary_ARM_E.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_full_100/summary_ARM_F.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/audit_verification.json` | No | 2.2 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/diagnostic_run_manifest.json` | No | 2.7 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/experiment_summary.json` | No | 2.4 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/judge_results_ARM_A.jsonl` | No | 15.0 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/judge_results_ARM_B.jsonl` | No | 14.6 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/judge_results_ARM_C.jsonl` | No | 14.8 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/judge_results_ARM_D.jsonl` | No | 14.9 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/judge_results_ARM_E.jsonl` | No | 14.9 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/judge_results_ARM_F.jsonl` | No | 14.5 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/paired_analysis.json` | No | 6.0 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/predictions_ARM_A.jsonl` | No | 29.8 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/predictions_ARM_B.jsonl` | No | 28.2 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/predictions_ARM_C.jsonl` | No | 29.4 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/predictions_ARM_D.jsonl` | No | 31.2 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/predictions_ARM_E.jsonl` | No | 24.6 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/predictions_ARM_F.jsonl` | No | 29.5 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/summary_ARM_A.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/summary_ARM_B.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/summary_ARM_C.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/summary_ARM_D.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/summary_ARM_E.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/diagnostic_pilot_30/summary_ARM_F.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/audited_17_failures.json` | No | 22.3 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/detailed_17_audit.json` | No | 45.8 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/diagnostic_run_manifest.json` | No | 2.3 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/experiment_summary.json` | No | 2.2 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/judge_results_ARM_1_BASELINE.jsonl` | No | 20.0 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/judge_results_ARM_2_ONTOLOGY.jsonl` | No | 20.0 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/judge_results_ARM_3_DIRECT_TEMPORAL.jsonl` | No | 19.6 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/judge_results_ARM_4_UNIFIED_HYBRID.jsonl` | No | 19.9 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/predictions_ARM_1_BASELINE.jsonl` | No | 38.1 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/predictions_ARM_2_ONTOLOGY.jsonl` | No | 37.3 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/predictions_ARM_3_DIRECT_TEMPORAL.jsonl` | No | 53.4 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/predictions_ARM_4_UNIFIED_HYBRID.jsonl` | No | 42.1 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/recomputed_metrics_and_paired_analysis.json` | No | 7.8 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/summary_ARM_1_BASELINE.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/summary_ARM_2_ONTOLOGY.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/summary_ARM_3_DIRECT_TEMPORAL.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/summary_ARM_4_UNIFIED_HYBRID.json` | No | 0.3 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json` | No | 21056.0 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40_manifest.json` | No | 2.8 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/human_review_diagnostic_sample_35.csv` | No | 26.9 |
| `evaluation/autonomous_sprint/runs/finalization_sprint/sprint_manifest.json` | No | 1.5 |
| `evaluation/autonomous_sprint/runs/post_evaluation_audit/POST_EVALUATION_AUDIT_REPORT.md` | No | 19.5 |
| `evaluation/autonomous_sprint/runs/post_evaluation_audit/audit_provenance.json` | No | 2.5 |
| `evaluation/autonomous_sprint/runs/post_evaluation_audit/disagreement_inspection.json` | No | 30.1 |
| `evaluation/autonomous_sprint/runs/post_evaluation_audit/implementation_flags_audit.json` | No | 1.2 |
| `evaluation/autonomous_sprint/runs/post_evaluation_audit/independent_metrics_recomputation.json` | No | 42.8 |
| `evaluation/autonomous_sprint/runs/smoke_test_run/diagnostic_run_manifest.json` | No | 2.7 |
| `evaluation/autonomous_sprint/runs/smoke_test_run/judge_results_ARM_B.jsonl` | No | 0.9 |
| `evaluation/autonomous_sprint/runs/smoke_test_run/predictions_ARM_A.jsonl` | No | 1.3 |
| `evaluation/autonomous_sprint/runs/smoke_test_run/predictions_ARM_B.jsonl` | No | 1.3 |
| `evaluation/autonomous_sprint/runs/smoke_test_run/predictions_ARM_C.jsonl` | No | 1.3 |
| `evaluation/autonomous_sprint/runs/smoke_test_run/predictions_ARM_D.jsonl` | No | 1.3 |
| `evaluation/autonomous_sprint/runs/smoke_test_run/predictions_ARM_E.jsonl` | No | 1.3 |
| `evaluation/autonomous_sprint/runs/smoke_test_run/predictions_ARM_F.jsonl` | No | 1.3 |
| `evaluation/autonomous_sprint/runs/smoke_test_run/summary_ARM_B.json` | No | 0.3 |
| `evaluation/autonomous_sprint/test_diverse_generation.py` | No | 1.4 |
| `evaluation/autonomous_sprint/test_retrieval_ranking.py` | No | 1.0 |
| `evaluation/baseline_reranker_experiment/baseline_cv_results.json` | No | 10.3 |
| `evaluation/baseline_reranker_experiment/cached_candidates.pkl` | No | 34846.8 |
| `evaluation/baseline_reranker_experiment/run_baseline_experiment.py` | No | 18.6 |
| `evaluation/benchmark_day2/category_breakdown_day2.csv` | No | 0.8 |
| `evaluation/benchmark_day2/compile_day2_results.py` | No | 9.7 |
| `evaluation/benchmark_day2/day2_benchmark_summary.json` | No | 6.0 |
| `evaluation/benchmark_day2/per_question_comparison.csv` | No | 41.4 |
| `evaluation/benchmark_day2/predictions_gpt_oss.json` | No | 109.6 |
| `evaluation/benchmark_day2/predictions_qwen2_5.json` | No | 102.1 |
| `evaluation/benchmark_day2/run_day2_benchmark.py` | No | 18.1 |
| `evaluation/benchmark_day2/statistical_comparison.json` | No | 0.5 |
| `evaluation/compile_and_verify_all.py` | No | 12.6 |
| `evaluation/data/longmemeval/longmemeval_oracle.json` | No | 15027.8 |
| `evaluation/data/longmemeval/longmemeval_s_cleaned.json` | No | 270882.3 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked/ablation_manifest.json` | No | 0.9 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked/results_config_B-CHUNKED.jsonl` | No | 39.6 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked/summary_config_B-CHUNKED.json` | No | 57.0 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked_graph/ablation_manifest.json` | No | 0.8 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked_graph/results_config_B-CHUNKED-GRAPH.jsonl` | No | 40.1 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked_graph/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked_graph/summary_config_B-CHUNKED-GRAPH.json` | No | 57.5 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked_ontology/ablation_manifest.json` | No | 0.8 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked_ontology/results_config_B-CHUNKED-ONTOLOGY.jsonl` | No | 44.8 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked_ontology/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked_ontology/summary_config_B-CHUNKED-ONTOLOGY.json` | No | 62.2 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_a/results_config_CONFIG-A.jsonl` | No | 89.1 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_a/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_a/summary_config_CONFIG-A.json` | No | 128.2 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_b/results_config_CONFIG-B.jsonl` | No | 96.3 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_b/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_b/summary_config_CONFIG-B.json` | No | 135.3 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_c/results_config_CONFIG-C.jsonl` | No | 85.9 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_c/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_c/summary_config_CONFIG-C.json` | No | 124.9 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_d/results_config_CONFIG-D.jsonl` | No | 80.7 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_d/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/config_d/summary_config_CONFIG-D.json` | No | 119.7 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/pilot_sample_manifest.json` | No | 3.4 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/pilot_summary.csv` | No | 1.0 |
| `evaluation/data/longmemeval/runs/campaign_v1_pilot/pilot_summary.json` | No | 4.8 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_a/results_config_CONFIG-A.jsonl` | No | 7.0 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_a/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_a/summary_config_CONFIG-A.json` | No | 13.7 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_b/results_config_CONFIG-B.jsonl` | No | 7.3 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_b/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_b/summary_config_CONFIG-B.json` | No | 14.0 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_c/results_config_CONFIG-C.jsonl` | No | 11.9 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_c/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_c/summary_config_CONFIG-C.json` | No | 18.7 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_d/results_config_CONFIG-D.jsonl` | No | 12.8 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_d/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/config_d/summary_config_CONFIG-D.json` | No | 19.5 |
| `evaluation/data/longmemeval/runs/campaign_v1_smoke/smoke_sample_manifest.json` | No | 0.1 |
| `evaluation/data/longmemeval/runs/diagnostic_sample_config_b/results_config_B.jsonl` | No | 29.3 |
| `evaluation/data/longmemeval/runs/diagnostic_sample_config_b/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/diagnostic_sample_config_b/sampling_methodology.json` | No | 1.6 |
| `evaluation/data/longmemeval/runs/diagnostic_sample_config_b/summary_config_B.json` | No | 45.6 |
| `evaluation/data/longmemeval/runs/experiment_a_context_budget/results_experiment_a.jsonl` | No | 199.1 |
| `evaluation/data/longmemeval/runs/experiment_a_context_budget/summary_experiment_a.json` | No | 292.7 |
| `evaluation/data/longmemeval/runs/experiment_b_retrieval/results_B0_baseline.jsonl` | No | 264.5 |
| `evaluation/data/longmemeval/runs/experiment_b_retrieval/results_B1_max2.jsonl` | No | 258.6 |
| `evaluation/data/longmemeval/runs/experiment_b_retrieval/results_B2_adaptive.jsonl` | No | 255.7 |
| `evaluation/data/longmemeval/runs/experiment_b_retrieval/results_B3_session_mmr.jsonl` | No | 257.8 |
| `evaluation/data/longmemeval/runs/experiment_b_retrieval/summary_experiment_b.json` | No | 6.6 |
| `evaluation/data/longmemeval/runs/experiment_c_combined/results_experiment_c.jsonl` | No | 706.4 |
| `evaluation/data/longmemeval/runs/experiment_c_combined/summary_experiment_c.json` | No | 1023.7 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_b_arm_b/official_judge_decisions_qwen2.5_14b.jsonl` | No | 277.0 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_b_arm_b/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_b_arm_b/results_config_D.jsonl` | No | 817.4 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_b_arm_b/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_b_arm_b/summary_config_D.json` | No | 1193.5 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_c_phase3/official_judge_decisions_qwen2.5_14b.jsonl` | No | 278.2 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_c_phase3/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_c_phase3/results_config_D.jsonl` | No | 825.2 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_c_phase3/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_c_phase3/summary_config_D.json` | No | 1203.8 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_e_baseline/corrected_heuristic_summary.json` | No | 1.0 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_e_baseline/official_judge_decisions_qwen2.5_14b.jsonl` | No | 289.0 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_e_baseline/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_e_baseline/results_config_D.jsonl` | No | 720.8 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_e_baseline/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/final_bakeoff/candidate_e_baseline/summary_config_D.json` | No | 1047.6 |
| `evaluation/data/longmemeval/runs/full_baseline_config_d/official_judge_decisions_qwen2.5_14b.jsonl` | No | 269.8 |
| `evaluation/data/longmemeval/runs/full_baseline_config_d/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/full_baseline_config_d/results_config_D.jsonl` | No | 693.3 |
| `evaluation/data/longmemeval/runs/full_baseline_config_d/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/full_baseline_config_d/summary_config_D.json` | No | 1012.2 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked/held_out_manifest.json` | No | 0.8 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked/results_config_B-CHUNKED.jsonl` | No | 37.5 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked/summary_config_B-CHUNKED.json` | No | 54.9 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked_graph/held_out_manifest.json` | No | 0.8 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked_graph/results_config_B-CHUNKED-GRAPH.jsonl` | No | 38.4 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked_graph/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked_graph/summary_config_B-CHUNKED-GRAPH.json` | No | 55.8 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked_ontology/held_out_manifest.json` | No | 0.8 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked_ontology/results_config_B-CHUNKED-ONTOLOGY.jsonl` | No | 32.6 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked_ontology/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/held_out_sample_config_b_chunked_ontology/summary_config_B-CHUNKED-ONTOLOGY.json` | No | 50.0 |
| `evaluation/data/longmemeval/runs/held_out_sample_manifest.json` | No | 1.4 |
| `evaluation/data/longmemeval/runs/integrated_production_run/corrected_heuristic_summary.json` | No | 1.0 |
| `evaluation/data/longmemeval/runs/integrated_production_run/official_judge_decisions_qwen2.5_14b.jsonl` | No | 289.0 |
| `evaluation/data/longmemeval/runs/integrated_production_run/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/integrated_production_run/results_config_D.jsonl` | No | 720.8 |
| `evaluation/data/longmemeval/runs/integrated_production_run/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/integrated_production_run/summary_config_D.json` | No | 1047.6 |
| `evaluation/data/longmemeval/runs/pilot_run/results_config_B.jsonl` | No | 3.8 |
| `evaluation/data/longmemeval/runs/pilot_run/summary_config_B.json` | No | 5.2 |
| `evaluation/data/longmemeval/runs/reevaluated_heuristic_comparison.json` | No | 2.2 |
| `evaluation/data/longmemeval/runs/smoke_test_longmemeval_s_config_b/results_config_B.jsonl` | No | 2.5 |
| `evaluation/data/longmemeval/runs/smoke_test_longmemeval_s_config_b/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/smoke_test_longmemeval_s_config_b/summary_config_B.json` | No | 7.7 |
| `evaluation/data/longmemeval/runs/temporal_phase1_experiments/phase1_combined/official_judge_decisions_qwen2.5_14b.jsonl` | No | 72.0 |
| `evaluation/data/longmemeval/runs/temporal_phase1_experiments/phase1_combined/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/temporal_phase1_experiments/phase1_combined/results_config_D.jsonl` | No | 186.3 |
| `evaluation/data/longmemeval/runs/temporal_phase1_experiments/phase1_combined/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/temporal_phase1_experiments/phase1_combined/summary_config_D.json` | No | 276.0 |
| `evaluation/data/longmemeval/runs/temporal_phase1_experiments/temporal_133_qids.json` | No | 2.4 |
| `evaluation/data/longmemeval/runs/temporal_phase2a1_experiments/phase2a1_adaptive_133/official_judge_decisions_qwen2.5_14b.jsonl` | No | 71.8 |
| `evaluation/data/longmemeval/runs/temporal_phase2a1_experiments/phase2a1_adaptive_133/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/temporal_phase2a1_experiments/phase2a1_adaptive_133/results_config_D.jsonl` | No | 188.9 |
| `evaluation/data/longmemeval/runs/temporal_phase2a1_experiments/phase2a1_adaptive_133/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/temporal_phase2a1_experiments/phase2a1_adaptive_133/summary_config_D.json` | No | 279.6 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/cat_h_41_qids.json` | No | 0.8 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/cat_h_forensics.json` | No | 30.3 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/phase2a_hierarchical_cat_h/official_judge_decisions_qwen2.5_14b.jsonl` | No | 23.9 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/phase2a_hierarchical_cat_h/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/phase2a_hierarchical_cat_h/results_config_D.jsonl` | No | 57.2 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/phase2a_hierarchical_cat_h/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/phase2a_hierarchical_cat_h/summary_config_D.json` | No | 87.5 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/phase2a_hierarchical_diversity_133/official_judge_decisions_qwen2.5_14b.jsonl` | No | 73.8 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/phase2a_hierarchical_diversity_133/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/phase2a_hierarchical_diversity_133/results_config_D.jsonl` | No | 197.6 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/phase2a_hierarchical_diversity_133/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/temporal_phase2a_experiments/phase2a_hierarchical_diversity_133/summary_config_D.json` | No | 291.4 |
| `evaluation/data/longmemeval/runs/temporal_phase2b1_experiments/phase2b1_event_graph_133/official_judge_decisions_qwen2.5_14b.jsonl` | No | 71.1 |
| `evaluation/data/longmemeval/runs/temporal_phase2b1_experiments/phase2b1_event_graph_133/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/temporal_phase2b1_experiments/phase2b1_event_graph_133/results_config_D.jsonl` | No | 211.7 |
| `evaluation/data/longmemeval/runs/temporal_phase2b1_experiments/phase2b1_event_graph_133/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/temporal_phase2b1_experiments/phase2b1_event_graph_133/summary_config_D.json` | No | 313.4 |
| `evaluation/data/longmemeval/runs/temporal_phase2b1_experiments/smoke_test_2/results_config_D.jsonl` | No | 3.5 |
| `evaluation/data/longmemeval/runs/temporal_phase2b1_experiments/smoke_test_2/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/temporal_phase2b1_experiments/smoke_test_2/summary_config_D.json` | No | 9.5 |
| `evaluation/data/longmemeval/runs/temporal_phase2b2_experiments/arm_b_flat_improved_synthesis/official_judge_decisions_qwen2.5_14b.jsonl` | No | 70.6 |
| `evaluation/data/longmemeval/runs/temporal_phase2b2_experiments/arm_b_flat_improved_synthesis/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/temporal_phase2b2_experiments/arm_b_flat_improved_synthesis/results_config_D.jsonl` | No | 211.2 |
| `evaluation/data/longmemeval/runs/temporal_phase2b2_experiments/arm_b_flat_improved_synthesis/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/temporal_phase2b2_experiments/arm_b_flat_improved_synthesis/summary_config_D.json` | No | 312.8 |
| `evaluation/data/longmemeval/runs/temporal_phase3_1_experiments/evidence_validated_retrieval/official_judge_decisions_qwen2.5_14b.jsonl` | No | 68.1 |
| `evaluation/data/longmemeval/runs/temporal_phase3_1_experiments/evidence_validated_retrieval/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/temporal_phase3_1_experiments/evidence_validated_retrieval/results_config_D.jsonl` | No | 212.9 |
| `evaluation/data/longmemeval/runs/temporal_phase3_1_experiments/evidence_validated_retrieval/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/temporal_phase3_1_experiments/evidence_validated_retrieval/summary_config_D.json` | No | 315.7 |
| `evaluation/data/longmemeval/runs/temporal_phase3_1_experiments/smoke_test/results_config_D.jsonl` | No | 1.5 |
| `evaluation/data/longmemeval/runs/temporal_phase3_1_experiments/smoke_test/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/temporal_phase3_1_experiments/smoke_test/summary_config_D.json` | No | 7.0 |
| `evaluation/data/longmemeval/runs/temporal_phase3_1_experiments/smoke_test_2/results_config_D.jsonl` | No | 1.5 |
| `evaluation/data/longmemeval/runs/temporal_phase3_1_experiments/smoke_test_2/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/temporal_phase3_1_experiments/smoke_test_2/summary_config_D.json` | No | 6.9 |
| `evaluation/data/longmemeval/runs/temporal_phase3_experiments/query_aware_retrieval/official_judge_decisions_qwen2.5_14b.jsonl` | No | 72.6 |
| `evaluation/data/longmemeval/runs/temporal_phase3_experiments/query_aware_retrieval/official_judge_summary_qwen2.5_14b.json` | No | 1.2 |
| `evaluation/data/longmemeval/runs/temporal_phase3_experiments/query_aware_retrieval/results_config_D.jsonl` | No | 217.7 |
| `evaluation/data/longmemeval/runs/temporal_phase3_experiments/query_aware_retrieval/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/temporal_phase3_experiments/query_aware_retrieval/summary_config_D.json` | No | 320.9 |
| `evaluation/data/longmemeval/runs/test_metrics_dry/results_config_D.jsonl` | No | 1.2 |
| `evaluation/data/longmemeval/runs/test_metrics_dry/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/test_metrics_dry/summary_config_D.json` | No | 6.5 |
| `evaluation/data/longmemeval/runs/test_speed/results_config_D.jsonl` | No | 1.1 |
| `evaluation/data/longmemeval/runs/test_speed/run_manifest.json` | No | 0.4 |
| `evaluation/data/longmemeval/runs/test_speed/summary_config_D.json` | No | 4.9 |
| `evaluation/dataset.py` | Yes | 25.8 |
| `evaluation/demo_episodic_milestone_1.py` | No | 6.9 |
| `evaluation/diagnose_ontology_reeval.py` | No | 7.7 |
| `evaluation/evaluate_qa_official.py` | No | 7.3 |
| `evaluation/evaluators/longmemeval_evaluator.py` | No | 15.2 |
| `evaluation/experiment_a/calibrated_answering_budgeted.py` | No | 13.6 |
| `evaluation/final_bakeoff_30_sample_review.json` | No | 33.0 |
| `evaluation/final_benchmark/category_breakdown.csv` | No | 1.8 |
| `evaluation/final_benchmark/final_summary.json` | No | 14.5 |
| `evaluation/final_benchmark/latency_breakdown.csv` | No | 0.5 |
| `evaluation/final_benchmark/predictions_arm_a_baseline_fixed_fusion.json` | No | 73.5 |
| `evaluation/final_benchmark/predictions_arm_b_episodic_only.json` | No | 84.1 |
| `evaluation/final_benchmark/predictions_arm_c_semantic_only.json` | No | 69.4 |
| `evaluation/final_benchmark/predictions_arm_d_adaptive_episodic_semantic.json` | No | 73.3 |
| `evaluation/final_benchmark/predictions_arm_e_full_adaptive_safeguarded_graph.json` | No | 73.4 |
| `evaluation/final_benchmark/predictions_arm_e_uncalibrated_generation.json` | No | 82.5 |
| `evaluation/final_benchmark/retrieval_250_summary.json` | No | 1.2 |
| `evaluation/final_benchmark/statistical_comparison.json` | No | 2.2 |
| `evaluation/finalization_audit/FINALIZATION_SYSTEM_AUDIT_REPORT.md` | No | 23.1 |
| `evaluation/finalization_audit/USER_FACING_MVP_PLAN.md` | No | 8.6 |
| `evaluation/hybrid_validation/hybrid_validation_results.json` | No | 25.5 |
| `evaluation/hybrid_validation/isolated_env_baseline/entities.json` | No | 0.0 |
| `evaluation/hybrid_validation/isolated_env_baseline/memories.json` | No | 25.9 |
| `evaluation/hybrid_validation/isolated_env_hybrid/entities.json` | No | 0.0 |
| `evaluation/hybrid_validation/isolated_env_hybrid/memories.json` | No | 21.3 |
| `evaluation/hybrid_validation/validate_hybrid.py` | No | 15.3 |
| `evaluation/long_context/MEMORY_GRAPH_EVALUATION_REPORT.md` | Yes | 20.4 |
| `evaluation/long_context/audit_baseline_vs_graph.py` | Yes | 4.5 |
| `evaluation/long_context/audit_metrics.py` | Yes | 2.2 |
| `evaluation/long_context/audit_pairs.py` | Yes | 1.9 |
| `evaluation/long_context/baseline_metrics_summary.json` | Yes | 1.9 |
| `evaluation/long_context/baseline_raw_extraction_multi.json` | Yes | 263.9 |
| `evaluation/long_context/baseline_raw_extraction_single.json` | Yes | 62.9 |
| `evaluation/long_context/baseline_results_multi_baseline.json` | Yes | 51.2 |
| `evaluation/long_context/baseline_results_multi_ivara.json` | Yes | 50.9 |
| `evaluation/long_context/baseline_results_single_baseline.json` | Yes | 50.9 |
| `evaluation/long_context/baseline_results_single_ivara.json` | Yes | 49.6 |
| `evaluation/long_context/compare_cats.py` | Yes | 0.6 |
| `evaluation/long_context/comprehensive_ablation_a_to_e_results.json` | Yes | 145.3 |
| `evaluation/long_context/config.json` | Yes | 1.1 |
| `evaluation/long_context/controlled_ablation_results.json` | Yes | 66.5 |
| `evaluation/long_context/corrected_baseline_results_multi_baseline.json` | Yes | 51.7 |
| `evaluation/long_context/corrected_baseline_results_multi_ivara.json` | Yes | 51.4 |
| `evaluation/long_context/corrected_baseline_results_single_baseline.json` | Yes | 51.3 |
| `evaluation/long_context/corrected_baseline_results_single_ivara.json` | Yes | 50.3 |
| `evaluation/long_context/corrected_metrics.py` | Yes | 16.9 |
| `evaluation/long_context/corrected_metrics_summary.json` | Yes | 7.1 |
| `evaluation/long_context/corrected_results_multi_baseline.json` | Yes | 74.7 |
| `evaluation/long_context/corrected_results_multi_ivara.json` | Yes | 66.8 |
| `evaluation/long_context/corrected_results_single_baseline.json` | Yes | 50.5 |
| `evaluation/long_context/corrected_results_single_ivara.json` | Yes | 50.1 |
| `evaluation/long_context/display_summary.py` | Yes | 1.9 |
| `evaluation/long_context/gold_annotations.json` | Yes | 16.0 |
| `evaluation/long_context/graph_benchmark_results.json` | Yes | 297.6 |
| `evaluation/long_context/harness.py` | Yes | 20.5 |
| `evaluation/long_context/inspect_failures.py` | Yes | 0.8 |
| `evaluation/long_context/metrics_summary.json` | Yes | 1.9 |
| `evaluation/long_context/multi_episode/entities.json` | Yes | 4.5 |
| `evaluation/long_context/multi_episode/memories.json` | Yes | 160.5 |
| `evaluation/long_context/questions.json` | Yes | 36.2 |
| `evaluation/long_context/raw_extraction_multi.json` | Yes | 254.7 |
| `evaluation/long_context/raw_extraction_single.json` | Yes | 60.5 |
| `evaluation/long_context/recalculate_all_metrics.py` | Yes | 6.4 |
| `evaluation/long_context/results_multi_baseline.json` | Yes | 74.2 |
| `evaluation/long_context/results_multi_ivara.json` | Yes | 66.3 |
| `evaluation/long_context/results_single_baseline.json` | Yes | 50.0 |
| `evaluation/long_context/results_single_ivara.json` | Yes | 49.4 |
| `evaluation/long_context/run_ablation_experiment.py` | Yes | 7.4 |
| `evaluation/long_context/run_benchmark.py` | Yes | 8.0 |
| `evaluation/long_context/run_comprehensive_ablation_benchmark.py` | Yes | 14.0 |
| `evaluation/long_context/run_graph_benchmark.py` | Yes | 16.8 |
| `evaluation/long_context/single_episode/entities.json` | Yes | 4.5 |
| `evaluation/long_context/single_episode/memories.json` | Yes | 43.4 |
| `evaluation/long_context/story.md` | Yes | 11.0 |
| `evaluation/metrics.py` | Yes | 6.5 |
| `evaluation/microbenchmark_day1/benchmark_dataset.py` | No | 18.8 |
| `evaluation/microbenchmark_day1/benchmark_summary.json` | No | 135.1 |
| `evaluation/microbenchmark_day1/evaluate_metrics.py` | No | 11.9 |
| `evaluation/microbenchmark_day1/metrics_comparison.json` | No | 15.4 |
| `evaluation/microbenchmark_day1/raw_outputs_gpt_oss.json` | No | 73.8 |
| `evaluation/microbenchmark_day1/raw_outputs_qwen2_5.json` | No | 56.4 |
| `evaluation/microbenchmark_day1/run_microbenchmark.py` | No | 11.4 |
| `evaluation/microbenchmark_day1_5/benchmark_summary.json` | No | 255.2 |
| `evaluation/microbenchmark_day1_5/evaluate_metrics_day1_5.py` | No | 16.1 |
| `evaluation/microbenchmark_day1_5/metrics_comparison.json` | No | 21.7 |
| `evaluation/microbenchmark_day1_5/raw_outputs_gemma4.json` | No | 118.9 |
| `evaluation/microbenchmark_day1_5/raw_outputs_gpt_oss.json` | No | 73.8 |
| `evaluation/microbenchmark_day1_5/raw_outputs_qwen2_5.json` | No | 56.4 |
| `evaluation/microbenchmark_day1_5/run_microbenchmark_day1_5.py` | No | 13.4 |
| `evaluation/milestone_3_1_audit/STATISTICAL_AUDIT_CORRECTION.md` | No | 4.5 |
| `evaluation/milestone_3_1_audit/analyze_failures.py` | No | 8.3 |
| `evaluation/milestone_3_1_audit/audit_phase1_integrity.py` | No | 10.1 |
| `evaluation/milestone_3_1_audit/investigate_graph_contribution.py` | No | 8.5 |
| `evaluation/milestone_3_1_audit/phase1_integrity_audit.json` | No | 13.7 |
| `evaluation/milestone_3_1_audit/phase2_statistical_analysis.json` | No | 17.1 |
| `evaluation/milestone_3_1_audit/phase3_graph_investigation.json` | No | 166.6 |
| `evaluation/milestone_3_1_audit/phase4_retrieval_ablation.json` | No | 1.2 |
| `evaluation/milestone_3_1_audit/phase5_failure_analysis.json` | No | 23.8 |
| `evaluation/milestone_3_1_audit/phase5_failure_analysis_refined.json` | No | 23.7 |
| `evaluation/milestone_3_1_audit/refine_failures.py` | No | 4.8 |
| `evaluation/milestone_3_1_audit/run_retrieval_ablation.py` | No | 5.4 |
| `evaluation/milestone_3_1_audit/run_statistical_analysis.py` | No | 10.4 |
| `evaluation/milestone_3_results/category_breakdown.csv` | No | 1.4 |
| `evaluation/milestone_3_results/latency_breakdown.csv` | No | 0.4 |
| `evaluation/milestone_3_results/milestone_3_summary.json` | No | 11.7 |
| `evaluation/milestone_3_results/predictions_arm_a_semantic_hybrid.json` | No | 2685.3 |
| `evaluation/milestone_3_results/predictions_arm_b_episodic_only.json` | No | 79.8 |
| `evaluation/milestone_3_results/predictions_arm_c_semantic_only.json` | No | 82.4 |
| `evaluation/milestone_3_results/predictions_arm_d_episodic_semantic.json` | No | 79.2 |
| `evaluation/milestone_3_results/predictions_arm_e_unified_tri_source.json` | No | 88.7 |
| `evaluation/performance_audit_results.json` | No | 1.8 |
| `evaluation/print_qa_metrics_official.py` | No | 1.3 |
| `evaluation/print_retrieval_metrics_official.py` | No | 1.4 |
| `evaluation/reports/benchmark_report.md` | Yes | 3.0 |
| `evaluation/reports/benchmark_results.json` | Yes | 37.1 |
| `evaluation/rerun_retrieval_250.py` | No | 5.4 |
| `evaluation/retrieval_failure_analysis/RETRIEVAL_FAILURE_ANALYSIS.md` | No | 4.5 |
| `evaluation/run_benchmark.py` | Yes | 4.8 |
| `evaluation/run_final_benchmark.py` | No | 23.9 |
| `evaluation/run_milestone_3_benchmark.py` | No | 16.2 |
| `evaluation/runner.py` | Yes | 14.9 |
| `evaluation/runners/run_chunked_ablation.py` | No | 2.6 |
| `evaluation/runners/run_diagnostic_sample.py` | No | 3.2 |
| `evaluation/runners/run_graph_ablation.py` | No | 2.9 |
| `evaluation/runners/run_held_out_ablation.py` | No | 2.8 |
| `evaluation/runners/run_longmemeval.py` | No | 27.1 |
| `evaluation/runners/run_official_llm_judge.py` | No | 8.2 |
| `evaluation/scripts/analyze_pilot_results.py` | No | 3.1 |
| `evaluation/scripts/generate_pilot_sample.py` | No | 2.8 |
| `evaluation/scripts/run_campaign_pilot.py` | No | 12.4 |
| `evaluation/scripts/run_experiment_a.py` | No | 9.6 |
| `evaluation/scripts/run_experiment_b_retrieval.py` | No | 10.7 |
| `evaluation/scripts/run_experiment_c_combined.py` | No | 15.5 |
| `evaluation/scripts/run_smoke_test.py` | No | 4.0 |
| `evaluation/step8/STEP8_EVALUATION_RELIABILITY_REPORT.md` | No | 20.6 |
| `evaluation/step8/build_disagreement_analysis.py` | No | 8.4 |
| `evaluation/step8/disagreement_analysis.json` | No | 496.1 |
| `evaluation/step8/local_judge_results.json` | No | 144.1 |
| `evaluation/step8/recompute_metrics.py` | No | 9.9 |
| `evaluation/step8/recomputed_metrics_audit.csv` | No | 0.6 |
| `evaluation/step8/recomputed_metrics_audit.json` | No | 4.1 |
| `evaluation/step8/repeatability/repeatability_summary.json` | No | 8.0 |
| `evaluation/step8/repeatability/run_1/results_config_CONFIG-D.jsonl` | No | 7.7 |
| `evaluation/step8/repeatability/run_1/run_manifest.json` | No | 0.4 |
| `evaluation/step8/repeatability/run_1/summary_config_CONFIG-D.json` | No | 14.4 |
| `evaluation/step8/repeatability/run_2/results_config_CONFIG-D.jsonl` | No | 7.7 |
| `evaluation/step8/repeatability/run_2/run_manifest.json` | No | 0.4 |
| `evaluation/step8/repeatability/run_2/summary_config_CONFIG-D.json` | No | 14.4 |
| `evaluation/step8/run_local_model_judge.py` | No | 9.8 |
| `evaluation/step8/run_repeatability_experiment.py` | No | 4.1 |
| `evaluation/step8/test_local_judge.py` | No | 1.3 |
| `evaluation/step9/STEP9_HUMAN_VALIDATION_PLAN.md` | No | 14.0 |
| `evaluation/step9/audit_integrity.py` | No | 10.0 |
| `evaluation/step9/failure_analysis.json` | No | 39.5 |
| `evaluation/step9/generate_human_review_sample.py` | No | 12.3 |
| `evaluation/step9/human_review_sample.csv` | No | 64.0 |
| `evaluation/step9/metric_integrity_audit.json` | No | 3.6 |
| `evaluation/story_benchmark/__init__.py` | No | 0.0 |
| `evaluation/story_benchmark/baseline_rag.py` | No | 11.3 |
| `evaluation/story_benchmark/dataset.py` | No | 46.1 |
| `evaluation/story_benchmark/ground_truth.json` | No | 27.5 |
| `evaluation/story_benchmark/ivara_store/entities.json` | No | 0.2 |
| `evaluation/story_benchmark/ivara_store/memories.json` | No | 135.1 |
| `evaluation/story_benchmark/metrics.py` | No | 6.9 |
| `evaluation/story_benchmark/results/per_question_comparison.csv` | No | 8.1 |
| `evaluation/story_benchmark/results/predictions_baseline_hybrid.json` | No | 55.9 |
| `evaluation/story_benchmark/results/predictions_baseline_vector.json` | No | 53.8 |
| `evaluation/story_benchmark/results/predictions_ivara_full.json` | No | 132.1 |
| `evaluation/story_benchmark/results/predictions_ivara_hybrid_no_graph.json` | No | 609.6 |
| `evaluation/story_benchmark/results/results_summary.json` | No | 6.2 |
| `evaluation/story_benchmark/runner.py` | No | 17.8 |
| `evaluation/story_benchmark/story_data.json` | No | 20.5 |
| `evaluation/validation_sprint/analyze_early_categories.py` | No | 1.4 |
| `evaluation/validation_sprint/baseline_rag_fair.py` | No | 13.9 |
| `evaluation/validation_sprint/build_dataset.py` | No | 27.8 |
| `evaluation/validation_sprint/dataset_expanded.py` | No | 0.5 |
| `evaluation/validation_sprint/extract_case_studies.py` | No | 2.1 |
| `evaluation/validation_sprint/failure_analysis.md` | No | 18.0 |
| `evaluation/validation_sprint/generate_all_questions.py` | No | 76.5 |
| `evaluation/validation_sprint/ground_truth_expanded.json` | No | 130.3 |
| `evaluation/validation_sprint/ivara_store_validation/entities.json` | No | 0.2 |
| `evaluation/validation_sprint/ivara_store_validation/memories.json` | No | 173.6 |
| `evaluation/validation_sprint/methodology_audit.md` | No | 10.1 |
| `evaluation/validation_sprint/metrics.py` | No | 7.0 |
| `evaluation/validation_sprint/reproduction_and_config.md` | No | 5.5 |
| `evaluation/validation_sprint/results/category_metrics.csv` | No | 5.4 |
| `evaluation/validation_sprint/results/latency_metrics.csv` | No | 0.3 |
| `evaluation/validation_sprint/results/per_question_comparison.csv` | No | 77.6 |
| `evaluation/validation_sprint/results/predictions_baseline_hybrid.json` | No | 905.4 |
| `evaluation/validation_sprint/results/predictions_baseline_hybrid_speaker_aware.json` | No | 918.1 |
| `evaluation/validation_sprint/results/predictions_baseline_vector.json` | No | 899.5 |
| `evaluation/validation_sprint/results/predictions_ivara_full.json` | No | 664.6 |
| `evaluation/validation_sprint/results/predictions_ivara_full_no_temporal.json` | No | 691.9 |
| `evaluation/validation_sprint/results/predictions_ivara_hybrid_no_graph.json` | No | 2684.7 |
| `evaluation/validation_sprint/results/predictions_ivara_no_ontology_no_graph.json` | No | 2145.1 |
| `evaluation/validation_sprint/results/results_summary.json` | No | 20.4 |
| `evaluation/validation_sprint/results/statistical_tests.json` | No | 3.9 |
| `evaluation/validation_sprint/run_all_evaluations.py` | No | 1.4 |
| `evaluation/validation_sprint/run_all_remaining.py` | No | 2.0 |
| `evaluation/validation_sprint/runner.py` | No | 21.5 |
| `evaluation/validation_sprint/statistical_analysis.py` | No | 6.9 |
| `evaluation/validation_sprint/story_data_expanded.json` | No | 30.9 |
| `evaluation/validation_sprint/validation_report.md` | No | 16.3 |
| `research/faiss_demo.py` | Yes | 2.3 |
| `scripts/audit_performance_resources.py` | No | 9.1 |
| `scripts/test_user_journey.py` | No | 8.6 |

</details>

### Category E: Demonstration Assets

- **File Count:** 3
- **Total Size:** 0.03 MB

**Release Action:** Retain deterministic demo assets for offline verification and showcase.

<details><summary>Click to view file list</summary>

| Path | Tracked | Size (KB) |
|:---|:---:|:---:|
| `DEMO_GUIDE.md` | Yes | 6.3 |
| `demo.py` | Yes | 11.4 |
| `demo_ivara.py` | Yes | 15.6 |

</details>

### Category F: Generated Caches and Build Files

- **File Count:** 0
- **Total Size:** 0.00 MB

**Release Action:** Ensure all build/cache artifacts are strictly ignored by `.gitignore`.

<details><summary>Click to view file list</summary>

| Path | Tracked | Size (KB) |
|:---|:---:|:---:|

</details>

### Category G: Sensitive / Private Data

- **File Count:** 16
- **Total Size:** 0.93 MB

**Release Action:** STRICTLY EXCLUDE from release. Ensure `.gitignore` prevents tracking local databases (`*.db`) and conversation backups (`data/backups/`).

<details><summary>Click to view file list</summary>

| Path | Tracked | Size (KB) |
|:---|:---:|:---:|
| `data/backups/benchmark_validation_phase0/entities.json` | No | 0.5 |
| `data/backups/benchmark_validation_phase0/ground_truth.json` | No | 27.5 |
| `data/backups/benchmark_validation_phase0/manifest.json` | No | 1.9 |
| `data/backups/benchmark_validation_phase0/memories.json` | No | 12.0 |
| `data/backups/benchmark_validation_phase0/per_question_comparison.csv` | No | 8.1 |
| `data/backups/benchmark_validation_phase0/predictions_baseline_hybrid.json` | No | 55.9 |
| `data/backups/benchmark_validation_phase0/predictions_baseline_vector.json` | No | 53.8 |
| `data/backups/benchmark_validation_phase0/predictions_ivara_full.json` | No | 132.1 |
| `data/backups/benchmark_validation_phase0/predictions_ivara_hybrid_no_graph.json` | No | 609.6 |
| `data/backups/benchmark_validation_phase0/results_summary.json` | No | 6.2 |
| `data/backups/benchmark_validation_phase0/story_data.json` | No | 20.5 |
| `data/backups/milestone1_episodic_baseline/entities.json` | No | 0.5 |
| `data/backups/milestone1_episodic_baseline/manifest.json` | No | 0.5 |
| `data/backups/milestone1_episodic_baseline/memories.json` | No | 12.0 |
| `data/backups/phase0_audit/entities.json` | No | 0.5 |
| `data/backups/phase0_audit/memories.json` | No | 12.0 |

</details>

### Category H: Disposable Scratch and Duplicate Files

- **File Count:** 9
- **Total Size:** 4.28 MB

**Release Action:** Delete disposable files (`IVARA_FILE_TREE.txt`, temporary inspection scripts in `scratch/`).

<details><summary>Click to view file list</summary>

| Path | Tracked | Size (KB) |
|:---|:---:|:---:|
| `IVARA_FILE_TREE.txt` | No | 4370.8 |
| `scratch/.gitkeep` | No | 0.0 |
| `scratch/check_messages.py` | No | 0.2 |
| `scratch/detailed_inventory.py` | No | 2.9 |
| `scratch/generate_audit.py` | No | 5.6 |
| `scratch/inspect_repo.py` | No | 0.7 |
| `scratch/scan_eval_imports.py` | No | 1.8 |
| `scratch/scan_imports.py` | No | 2.6 |
| `scratch/security_audit.py` | No | 1.9 |

</details>

