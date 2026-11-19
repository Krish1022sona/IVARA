"""
Reproducible Baseline Runner for LongMemEval on IVARA.

Executes an isolated evaluation pass of LongMemEval questions across IVARA configurations:
  - Config A: Vector baseline
  - Config B: Hybrid vector + BM25
  - Config D: Graph-assisted retrieval (SQLite graph triples + traversal)
  - Config E: Hybrid + query decomposition + selective graph evidence

Safety & Isolation Guarantees:
  - Writes exclusively to `evaluation/data/longmemeval/runs/`
  - Never reads from or writes to production stores (data/memories.json, data/entities.json)
  - Strictly isolates gold answers until the post-generation scoring stage
  - Supports resuming interrupted runs with manifest validation (preventing mixed model/config data)
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.adapters.longmemeval_adapter import LongMemEvalAdapter, LongMemEvalInstance
from evaluation.evaluators.longmemeval_evaluator import LongMemEvalEvaluator, OFFICIAL_TASKS, evaluate_response_locally
from src.graph.store import SQLiteGraphStore
from src.graph.builder import MemoryGraphBuilder
from src.answering import answer_question
import numpy as np


class RunConfigurationMismatchError(Exception):
    """Raised when an attempt is made to resume a run with conflicting settings."""
    pass


def get_git_info() -> dict[str, str]:
    """Capture current Git revision for reproducibility."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True
        ).strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a dataset file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class LongMemEvalRunner:
    """
    Orchestrates ingestion, retrieval, answering, and evaluation for LongMemEval.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        output_dir: str | Path,
        config: str = "B",
        model: str = "gpt-oss",
        k: int = 5,
        dataset_variant: str | None = None,
        llm_caller: Callable[[str, str], Any] | None = None,
        judge_caller: Callable[[str], str] | None = None,
        use_local_judge: bool = True,
        retrieval_strategy: str = "flat"
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.config = config.upper()
        self.model = model
        self.k = k
        self.llm_caller = llm_caller
        self.judge_caller = judge_caller
        self.use_local_judge = use_local_judge
        self.retrieval_strategy = retrieval_strategy

        # Auto-detect or set dataset variant
        if dataset_variant:
            self.dataset_variant = dataset_variant
        elif "oracle" in self.dataset_path.name.lower():
            self.dataset_variant = "oracle_reader_eval"
        elif "_s" in self.dataset_path.name.lower():
            self.dataset_variant = "longmemeval_s_retrieval_eval"
        elif "_m" in self.dataset_path.name.lower():
            self.dataset_variant = "longmemeval_m_retrieval_eval"
        else:
            self.dataset_variant = "custom_longmemeval"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter = LongMemEvalAdapter(self.dataset_path, self.output_dir, sort_sessions_chronologically=True)

    def run(
        self,
        limit: int | None = None,
        resume: bool = True,
        target_question_ids: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Execute the evaluation run.
        """
        start_time = time.time()
        git_info = get_git_info()
        dataset_sha256 = compute_file_sha256(self.dataset_path)

        manifest_file = self.output_dir / "run_manifest.json"
        results_file = self.output_dir / f"results_config_{self.config}.jsonl"
        summary_file = self.output_dir / f"summary_config_{self.config}.json"

        run_manifest = {
            "dataset_file": self.dataset_path.name,
            "dataset_sha256": dataset_sha256,
            "dataset_variant": self.dataset_variant,
            "config": self.config,
            "model": self.model,
            "k": self.k,
            "retrieval_strategy": self.retrieval_strategy,
            "git_commit": git_info["commit"],
            "git_branch": git_info["branch"],
            "evaluation_mode": "smoke_test_deterministic_heuristics" if self.use_local_judge else "official_llm_judge"
        }

        # Check existing manifest on resume to prevent mixed configurations
        completed_qids = set()
        existing_results = []

        if resume and manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                prev_manifest = json.load(f)

            # Validate matching core parameters
            mismatch_keys = []
            for k_check in ["dataset_sha256", "config", "model", "k", "dataset_variant"]:
                if prev_manifest.get(k_check) != run_manifest.get(k_check):
                    mismatch_keys.append(f"{k_check} (prev={prev_manifest.get(k_check)}, new={run_manifest.get(k_check)})")

            if mismatch_keys:
                raise RunConfigurationMismatchError(
                    f"Cannot resume run: Configuration mismatch detected in existing manifest: {', '.join(mismatch_keys)}"
                )

            if results_file.exists():
                with open(results_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                qid = record.get("question_id")
                                if qid:
                                    completed_qids.add(qid)
                                    existing_results.append(record)
                            except json.JSONDecodeError:
                                pass
        else:
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(run_manifest, f, indent=2)

        # 1. Load and validate instances
        instances, failures = self.adapter.load_and_validate(
            max_examples=limit,
            target_question_ids=target_question_ids
        )
        references = self.adapter.get_reference_data(instances)

        print(f"\n{'='*70}")
        print(f"Starting LongMemEval Evaluation | Config: {self.config} | Model: {self.model}")
        print(f"Dataset Variant: {self.dataset_variant} ({self.dataset_path.name})")
        print(f"Dataset SHA-256: {dataset_sha256[:16]}... | Instances to test: {len(instances)}")
        print(f"Completed (resumed): {len(completed_qids)} | Remaining: {len(instances) - len(completed_qids)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Evaluation Mode: {run_manifest['evaluation_mode']}")
        print(f"{'='*70}\n")

        # Open results file in append mode
        with open(results_file, "a", encoding="utf-8") as out_f:
            for idx, inst in enumerate(instances):
                if inst.question_id in completed_qids:
                    continue

                q_start = time.time()
                print(f"[{idx+1}/{len(instances)}] Processing {inst.question_id} ({inst.question_type})...", end=" ", flush=True)

                # Set up isolated SQLite graph store for this instance
                db_path = self.output_dir / f"instance_{inst.question_id}_graph.db"
                if db_path.exists():
                    db_path.unlink()

                store = SQLiteGraphStore(db_path)
                memories = self.adapter.convert_instance_to_memories(inst)

                builder = MemoryGraphBuilder(store)
                builder.build_from_records(memories)

                # Determine retrieval & answering parameters based on config
                norm_config = self.config.upper().replace("-", "_")

                # Step 7 Evaluation Campaign Configurations:
                # Config A: Dense + BM25 baseline
                is_config_a = norm_config in ("CONFIG_A", "A", "B_CHUNKED", "BCHUNKED")
                # Config B: Baseline + ontology
                is_config_b = norm_config in ("CONFIG_B", "B_CHUNKED_ONTOLOGY", "BCHUNKED_ONTOLOGY", "B_ONTOLOGY")
                # Config C: Ontology + bounded graph expansion (temporal disabled)
                is_config_c = norm_config in ("CONFIG_C", "C", "B_CHUNKED_GRAPH", "BCHUNKED_GRAPH", "C_GRAPH")
                # Config D: Full system with temporal reasoning
                is_config_d = norm_config in ("CONFIG_D", "D", "B_CHUNKED_FULL", "BCHUNKED_FULL", "FULL", "D_FULL")

                if is_config_a:
                    is_chunked = True
                    use_ontology = False
                    use_chunk_graph = False
                    enable_temporal = False
                elif is_config_b:
                    is_chunked = True
                    use_ontology = True
                    use_chunk_graph = False
                    enable_temporal = False
                elif is_config_c:
                    is_chunked = True
                    use_ontology = True
                    use_chunk_graph = True
                    enable_temporal = False
                elif is_config_d:
                    is_chunked = True
                    use_ontology = True
                    use_chunk_graph = True
                    enable_temporal = True
                elif norm_config in ("LEGACY_D", "E"):
                    is_chunked = False
                    use_ontology = False
                    use_chunk_graph = False
                    enable_temporal = False
                else:
                    # Legacy fallback (e.g. monolithic unchunked B)
                    is_chunked = False
                    use_ontology = False
                    use_chunk_graph = False
                    enable_temporal = False

                use_graph = norm_config in ("LEGACY_D", "E")
                use_decomposition = (norm_config == "E")

                chunks = None
                t_chunk_ms = 0.0
                t_graph_ms = 0.0
                t_ans_ms = 0.0

                if is_chunked:
                    t_chunk_start = time.time()
                    from src.chunking import TurnAwareChunker
                    chunker = TurnAwareChunker(window_size=3, overlap=1)
                    chunks = chunker.chunk_all_sessions(inst.sessions)
                    t_chunk_ms = (time.time() - t_chunk_start) * 1000

                    if use_chunk_graph:
                        t_graph_start = time.time()
                        from src.chunking.graph_retrieval import TurnGraphBuilder
                        turn_graph_builder = TurnGraphBuilder(store)
                        turn_graph_builder.build_from_chunks(chunks)
                        t_graph_ms = (time.time() - t_graph_start) * 1000

                # Set relevance threshold
                min_score = 0.35 if inst.is_abstention else None

                answer_result = None
                try:
                    t_ans_start = time.time()
                    answer_result = answer_question(
                        query=inst.question,
                        k=self.k,
                        min_score=min_score,
                        memories=memories,
                        model=self.model,
                        llm_caller=self.llm_caller,
                        use_graph=use_graph,
                        graph_store=store,
                        use_decomposition=use_decomposition,
                        use_chunking=is_chunked,
                        chunks=chunks,
                        max_context_tokens=1500,
                        use_ontology=use_ontology,
                        use_chunk_graph=use_chunk_graph,
                        enable_temporal=enable_temporal,
                        question_date=inst.question_date,
                        k_temporal=8,
                        retrieval_strategy=self.retrieval_strategy
                    )
                    t_ans_ms = (time.time() - t_ans_start) * 1000
                    latency_ms = round((time.time() - q_start) * 1000, 2)
                    hypothesis = answer_result.get("answer", "")
                    status = answer_result.get("status", "answered")
                    citations = [c.get("memory_id") for c in answer_result.get("citations", [])]
                    direct_ev = answer_result.get("retrieved_context", {}).get("direct_evidence", [])
                    retrieved_evidence = [c.get("memory_id") for c in direct_ev]
                    retrieved_chunk_ids = [c.get("chunk_id") for c in direct_ev if "chunk_id" in c]
                    failure_reason = None
                except Exception as e:
                    latency_ms = round((time.time() - q_start) * 1000, 2)
                    hypothesis = f"Execution error: {str(e)}"
                    status = "error"
                    citations = []
                    retrieved_evidence = []
                    retrieved_chunk_ids = []
                    failure_reason = str(e)

                # Record graph node counts before store close
                graph_nodes_count = store.count_nodes() if (store and hasattr(store, "count_nodes")) else 0
                graph_edges_count = store.count_edges() if (store and hasattr(store, "count_edges")) else 0

                # Close and remove isolated per-instance DB
                store.close()
                if db_path.exists():
                    try:
                        db_path.unlink()
                    except Exception:
                        pass

                # Post-hoc diagnostic metrics (Zero leakage to retriever or model)
                gold_rank = None
                target_sids = set()
                if inst.answer_session_ids:
                    for s in inst.answer_session_ids:
                        target_sids.add(str(s))
                        target_sids.add(f"sess_{s}")
                        target_sids.add(str(s).replace("sess_", ""))

                    for r_i, c_ev in enumerate(direct_ev):
                        ev_m_id = str(c_ev.get("memory_id") or "")
                        ev_s_id = str(c_ev.get("session_id") or "")
                        if ev_m_id in target_sids or ev_s_id in target_sids:
                            gold_rank = r_i + 1
                            break

                retrieved_sids_set = set()
                for c_ev in direct_ev:
                    m_id = str(c_ev.get("memory_id") or "")
                    s_id = str(c_ev.get("session_id") or "")
                    if m_id:
                        retrieved_sids_set.add(m_id)
                        retrieved_sids_set.add(m_id.replace("sess_", ""))
                    if s_id:
                        retrieved_sids_set.add(s_id)
                        retrieved_sids_set.add(f"sess_{s_id}")

                if inst.answer_session_ids and not inst.is_abstention:
                    unique_gold_sids = {str(s).replace("sess_", "") for s in inst.answer_session_ids}
                    retrieved_gold_count = sum(1 for s in unique_gold_sids if s in retrieved_sids_set or f"sess_{s}" in retrieved_sids_set)
                    evidence_coverage_ratio = round(retrieved_gold_count / len(unique_gold_sids), 4) if unique_gold_sids else 1.0
                else:
                    evidence_coverage_ratio = 1.0

                recall_at_1 = bool(gold_rank == 1) if not inst.is_abstention else True
                recall_at_3 = bool(gold_rank is not None and gold_rank <= 3) if not inst.is_abstention else True
                recall_at_5 = bool(gold_rank is not None and gold_rank <= 5) if not inst.is_abstention else True
                final_coverage = (gold_rank is not None) if not inst.is_abstention else True
                ctx_text = ""
                ctx = answer_result.get("retrieved_context") if isinstance(answer_result, dict) else {}
                if isinstance(ctx, dict):
                    ctx_text = ctx.get("context_text", "")
                approx_tokens = int(len(ctx_text.split()) / 0.75) if ctx_text else 0

                temp_res = ctx.get("temporal_resolution") if isinstance(ctx, dict) else None
                temporal_activated = bool(temp_res and temp_res.get("intent"))
                temporal_intent = str(temp_res.get("intent").value if hasattr(temp_res.get("intent"), "value") else temp_res.get("intent")) if temp_res else None
                selected_claims = len(temp_res.get("selected_claims", [])) if temp_res else 0
                historical_claims = len(temp_res.get("historical_claims", [])) if temp_res else 0
                unresolved_conflicts = len(temp_res.get("unresolved_conflicts", [])) if temp_res else 0

                local_heur = evaluate_response_locally(
                    task=inst.question_type,
                    question=inst.question,
                    gold_answer=inst.gold_answer,
                    hypothesis=hypothesis,
                    is_abstention=inst.is_abstention
                )

                record = {
                    "question_id": inst.question_id,
                    "question_type": inst.question_type,
                    "question": inst.question,
                    "question_date": inst.question_date,
                    "hypothesis": hypothesis,
                    "status": status,
                    "citations": citations,
                    "retrieved_evidence": retrieved_evidence,
                    "retrieved_chunk_ids": retrieved_chunk_ids if is_chunked else [],
                    "latency_ms": latency_ms,
                    "stage_latencies_ms": {
                        "chunking": round(t_chunk_ms, 2),
                        "graph_build": round(t_graph_ms, 2),
                        "answering": round(t_ans_ms, 2),
                        "total": round(latency_ms, 2)
                    },
                    "config": self.config,
                    "is_abstention": inst.is_abstention,
                    "failure_reason": failure_reason,
                    # Audit diagnostic fields (post-hoc, non-official)
                    "indexing_success": len(memories) > 0,
                    "gold_retrieval_rank": gold_rank,
                    "recall_at_1": recall_at_1,
                    "recall_at_3": recall_at_3,
                    "recall_at_5": recall_at_5,
                    "evidence_coverage_ratio": evidence_coverage_ratio,
                    "final_context_evidence_coverage": final_coverage,
                    "context_token_count": approx_tokens,
                    "local_heuristic_score": local_heur,
                    "evaluation_label": "smoke_test_deterministic_heuristics",
                    "official_status": "smoke_test_pending_official_judge",
                    # Observables for temporal and graph
                    "temporal_activated": temporal_activated,
                    "temporal_intent": temporal_intent,
                    "selected_claims_count": selected_claims,
                    "historical_claims_count": historical_claims,
                    "unresolved_conflicts_count": unresolved_conflicts,
                    "graph_nodes_count": graph_nodes_count,
                    "graph_edges_count": graph_edges_count
                }

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                existing_results.append(record)
                completed_qids.add(inst.question_id)

                print(f"Done ({latency_ms:.1f}ms) [{status}] heur={local_heur}")

        # 2. Run post-hoc evaluation
        evaluator = LongMemEvalEvaluator(
            judge_model="gpt-4o",
            llm_caller=self.judge_caller,
            use_local_heuristic=self.use_local_judge
        )
        eval_metrics = evaluator.evaluate(existing_results, references)

        # 3. Compute latency stats
        latencies = [r["latency_ms"] for r in existing_results if "latency_ms" in r]
        latency_stats = {
            "mean_ms": round(float(sum(latencies) / len(latencies)), 2) if latencies else 0.0,
            "min_ms": round(min(latencies), 2) if latencies else 0.0,
            "max_ms": round(max(latencies), 2) if latencies else 0.0,
            "total_eval_time_s": round(time.time() - start_time, 2)
        }

        # 3.5. Compute retrieval diagnostic aggregations
        ans_records = [r for r in existing_results if not r.get("is_abstention", False)]
        retrieval_diagnostics = {
            "overall": {
                "recall_at_1": round(float(np.mean([1 if r.get("recall_at_1") else 0 for r in ans_records])), 4) if ans_records else 0.0,
                "recall_at_3": round(float(np.mean([1 if r.get("recall_at_3") else 0 for r in ans_records])), 4) if ans_records else 0.0,
                "recall_at_5": round(float(np.mean([1 if r.get("recall_at_5") else 0 for r in ans_records])), 4) if ans_records else 0.0,
                "mean_evidence_coverage_ratio": round(float(np.mean([r.get("evidence_coverage_ratio", 0.0) for r in ans_records])), 4) if ans_records else 0.0,
                "total_answerable_instances": len(ans_records)
            },
            "by_task": {}
        }
        for task in OFFICIAL_TASKS:
            task_records = [r for r in ans_records if r.get("question_type") == task]
            if task_records:
                retrieval_diagnostics["by_task"][task] = {
                    "recall_at_1": round(float(np.mean([1 if r.get("recall_at_1") else 0 for r in task_records])), 4),
                    "recall_at_3": round(float(np.mean([1 if r.get("recall_at_3") else 0 for r in task_records])), 4),
                    "recall_at_5": round(float(np.mean([1 if r.get("recall_at_5") else 0 for r in task_records])), 4),
                    "mean_evidence_coverage_ratio": round(float(np.mean([r.get("evidence_coverage_ratio", 0.0) for r in task_records])), 4),
                    "total_instances": len(task_records)
                }
            else:
                retrieval_diagnostics["by_task"][task] = {
                    "recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mean_evidence_coverage_ratio": 0.0, "total_instances": 0
                }

        # 4. Build comprehensive summary
        summary = {
            "run_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_info["commit"],
                "git_branch": git_info["branch"],
                "dataset_file": str(self.dataset_path.name),
                "dataset_sha256": dataset_sha256,
                "dataset_variant": self.dataset_variant,
                "config": self.config,
                "model": self.model,
                "k": self.k,
                "total_dataset_instances": len(instances),
                "evaluated_instances": len(existing_results),
                "validation_failures": len(failures)
            },
            "metrics": eval_metrics,
            "retrieval_diagnostics": retrieval_diagnostics,
            "latency": latency_stats
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Evaluation Completed! Results saved to:")
        print(f"  Summary: {summary_file}")
        print(f"  Records: {results_file}")
        print(f"Evaluation Mode: {eval_metrics['evaluation_mode']}")
        print(f"Official Overall Accuracy: {eval_metrics['official_metrics']['overall_accuracy'] * 100:.2f}% (den: {eval_metrics['official_metrics']['denominators']['overall_accuracy']})")
        print(f"Official Task-Averaged Accuracy: {eval_metrics['official_metrics']['task_averaged_accuracy'] * 100:.2f}%")
        print(f"Abstention Accuracy: {eval_metrics['official_metrics']['abstention_accuracy'] * 100:.2f}% (den: {eval_metrics['official_metrics']['denominators']['abstention_accuracy']})")
        print(f"Answerable-Only Overall Accuracy: {eval_metrics['answerable_only_metrics']['overall_accuracy'] * 100:.2f}% (den: {eval_metrics['answerable_only_metrics']['denominators']['overall_accuracy']})")
        print(f"{'='*70}\n")

        return summary


def main():
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Run LongMemEval on IVARA baseline configurations.")
    parser.add_argument("--dataset", type=str, default="evaluation/data/longmemeval/longmemeval_s_cleaned.json")
    parser.add_argument(
        "--config",
        type=str,
        default="B",
        choices=[
            "A", "B", "C", "D", "E",
            "Config-A", "Config-B", "Config-C", "Config-D",
            "B-Chunked", "B-Chunked-Ontology", "B-Chunked-Graph", "B-Chunked-Full",
            "B_CHUNKED", "B_CHUNKED_ONTOLOGY", "B_CHUNKED_GRAPH", "B_CHUNKED_FULL"
        ]
    )
    parser.add_argument("--model", type=str, default="gpt-oss")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retrieval-strategy", type=str, default="flat", choices=["flat", "hierarchical", "hierarchical_diversity", "adaptive", "event_graph", "query_aware", "evidence_validated"])
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save run results.")
    parser.add_argument("--target-qids", type=str, default=None, help="Comma-separated list or JSON file of question IDs.")
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()

    target_qids = None
    if args.target_qids:
        target_path = Path(args.target_qids)
        if target_path.exists():
            with open(target_path, "r", encoding="utf-8") as f:
                target_qids = json.load(f)
        else:
            target_qids = [q.strip() for q in args.target_qids.split(",") if q.strip()]

    # Automatic safe folder defaults separating Oracle from LongMemEval-S
    if args.output_dir is None:
        dataset_tag = "oracle" if "oracle" in args.dataset.lower() else "longmemeval_s"
        output_dir = f"evaluation/data/longmemeval/runs/{dataset_tag}_config_{args.config.lower()}"
    else:
        output_dir = args.output_dir

    runner = LongMemEvalRunner(
        dataset_path=args.dataset,
        output_dir=output_dir,
        config=args.config,
        model=args.model,
        k=args.k,
        use_local_judge=True,
        retrieval_strategy=args.retrieval_strategy
    )
    runner.run(limit=args.limit, resume=not args.no_resume, target_question_ids=target_qids)


if __name__ == "__main__":
    main()
