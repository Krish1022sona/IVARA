"""
Hardened 6-Arm Diagnostic Evaluation Runner for IVARA Research Sprint.

Evaluates 6 isolated memory architectures over a fixed 100-question sample:
- ARM_A: Dense + BM25 Baseline
- ARM_B: Baseline + Ontology Guidance
- ARM_C: Ontology + Raw 1-Hop Graph Expansion (min_edge_relevance=0.0)
- ARM_D: Ontology + Relevance-Filtered Graph Expansion (min_edge_relevance=0.50) [Prototype B]
- ARM_E: Ontology + Direct Seed Temporal Extraction (direct_seed_temporal=True) [Prototype C]
- ARM_F: Full Existing Architecture (Graph + Temporal)

Features:
- Incremental JSONL flushing per question.
- Strict deduplication and restart-safe resumption.
- Per-question exception isolation.
- Detailed stage latencies, citations, and retrieval rank logging.
- Diagnostic local judge execution with robust JSON parsing and fallback.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, ".")
from typing import Any, Callable

from src.answering import answer_question
from src.chunking.turn_chunker import TurnAwareChunker
from src.chunking.graph_retrieval import TurnGraphBuilder
from src.graph.store import SQLiteGraphStore
from evaluation.adapters.longmemeval_adapter import LongMemEvalAdapter, LongMemEvalInstance

try:
    from ollama import chat
except ImportError:
    chat = None


ARM_DEFINITIONS: dict[str, dict[str, Any]] = {
    "ARM_A": {
        "name": "Baseline (Dense + BM25)",
        "use_ontology": False,
        "use_chunk_graph": False,
        "min_edge_relevance": 0.0,
        "direct_seed_temporal": False,
        "enable_temporal": False,
        "description": "Baseline hybrid retriever without ontology, graph, or temporal modules."
    },
    "ARM_B": {
        "name": "Ontology Guidance Only",
        "use_ontology": True,
        "use_chunk_graph": False,
        "min_edge_relevance": 0.0,
        "direct_seed_temporal": False,
        "enable_temporal": False,
        "description": "Baseline plus domain ontology query concept and entity matching."
    },
    "ARM_C": {
        "name": "Ontology + Raw Graph Expansion",
        "use_ontology": True,
        "use_chunk_graph": True,
        "min_edge_relevance": 0.0,
        "direct_seed_temporal": False,
        "enable_temporal": False,
        "description": "Ontology plus 1-hop bounded graph expansion without edge relevance filtering."
    },
    "ARM_D": {
        "name": "Ontology + Relevance-Filtered Graph (Prototype B)",
        "use_ontology": True,
        "use_chunk_graph": True,
        "min_edge_relevance": 0.50,
        "direct_seed_temporal": False,
        "enable_temporal": False,
        "description": "Ontology plus 1-hop graph expansion pruned by min_edge_relevance=0.50."
    },
    "ARM_E": {
        "name": "Ontology + Direct Seed Temporal (Prototype C)",
        "use_ontology": True,
        "use_chunk_graph": False,
        "min_edge_relevance": 0.0,
        "direct_seed_temporal": True,
        "enable_temporal": True,
        "description": "Ontology guidance plus temporal claim extraction directly from top seed chunks."
    },
    "ARM_F": {
        "name": "Full Architecture (Graph + Temporal)",
        "use_ontology": True,
        "use_chunk_graph": True,
        "min_edge_relevance": 0.0,
        "direct_seed_temporal": False,
        "enable_temporal": True,
        "description": "Full IVARA architecture with raw 1-hop graph expansion and graph temporal claims."
    },
    "ARM_1_BASELINE": {
        "name": "Baseline (Dense + BM25)",
        "use_ontology": False,
        "use_chunk_graph": False,
        "min_edge_relevance": 0.0,
        "direct_seed_temporal": False,
        "enable_temporal": False,
        "dynamic_graph_fallback": False,
        "description": "Baseline hybrid retriever without ontology, graph, or temporal modules."
    },
    "ARM_2_ONTOLOGY": {
        "name": "Ontology-Guided Hybrid",
        "use_ontology": True,
        "use_chunk_graph": False,
        "min_edge_relevance": 0.0,
        "direct_seed_temporal": False,
        "enable_temporal": False,
        "dynamic_graph_fallback": False,
        "description": "Ontology guidance for query concept and entity matching without graph or temporal modules."
    },
    "ARM_3_DIRECT_TEMPORAL": {
        "name": "Ontology + Direct-Seed Temporal",
        "use_ontology": True,
        "use_chunk_graph": False,
        "min_edge_relevance": 0.0,
        "direct_seed_temporal": True,
        "enable_temporal": True,
        "dynamic_graph_fallback": False,
        "description": "Ontology guidance plus temporal claim extraction directly from seed chunks."
    },
    "ARM_4_UNIFIED_HYBRID": {
        "name": "Unified Research Candidate (IVARA-Hybrid)",
        "use_ontology": True,
        "use_chunk_graph": False,
        "min_edge_relevance": 0.50,
        "direct_seed_temporal": True,
        "enable_temporal": True,
        "dynamic_graph_fallback": True,
        "description": "Unified research candidate: Ontology guidance + direct-seed temporal with relevance-filtered graph fallback."
    },
    "ARM_5_CORE_INTELLIGENCE": {
        "name": "IVARA Core Intelligence (Integrated Reasoning)",
        "use_ontology": True,
        "use_chunk_graph": False,
        "min_edge_relevance": 0.50,
        "direct_seed_temporal": True,
        "enable_temporal": True,
        "dynamic_graph_fallback": True,
        "max_chunks_per_session": 1,
        "description": "Integrated core intelligence: Ontology + direct-seed temporal with event priority, session diversity retrieval, evidence gate, and contradiction analysis."
    }
}


def get_git_info() -> dict[str, str]:
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class DiagnosticRunner:
    """
    Executes controlled diagnostic evaluations across specified arms.
    """

    def __init__(
        self,
        sample_path: str | Path,
        output_dir: str | Path,
        model: str = "gpt-oss:latest",
        judge_model: str = "VladimirGav/gemma4-26b-16GB-VRAM:latest",
        k: int = 5,
        llm_caller: Callable[[str, str], Any] | None = None,
        judge_caller: Callable[[str], str] | None = None
    ):
        self.sample_path = Path(sample_path)
        self.output_dir = Path(output_dir)
        self.model = model
        self.judge_model = judge_model
        self.k = k
        self.llm_caller = llm_caller
        self.judge_caller = judge_caller

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter = LongMemEvalAdapter(self.sample_path, self.output_dir, sort_sessions_chronologically=True)

    def write_manifest(self, selected_arms: list[str]) -> Path:
        manifest_path = self.output_dir / "diagnostic_run_manifest.json"
        git_info = get_git_info()
        data = {
            "evaluation_title": "IVARA 6-Arm Controlled Diagnostic Evaluation",
            "evaluation_tier": "diagnostic_sample",
            "benchmark_status": "diagnostic_only_not_official_longmemeval",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_branch": git_info["branch"],
            "git_commit": git_info["commit"],
            "sample_path": str(self.sample_path),
            "sample_sha256": compute_file_sha256(self.sample_path),
            "answering_model": self.model,
            "judge_model": self.judge_model,
            "retrieval_k": self.k,
            "max_context_tokens": 1500,
            "arms_evaluated": {arm: ARM_DEFINITIONS[arm] for arm in selected_arms}
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return manifest_path

    def run_arm(
        self,
        arm_key: str,
        limit: int | None = None,
        resume: bool = True
    ) -> dict[str, Any]:
        """
        Run answer generation for a single experimental arm.
        """
        if arm_key not in ARM_DEFINITIONS:
            raise ValueError(f"Unknown arm '{arm_key}'. Available: {list(ARM_DEFINITIONS.keys())}")

        arm_cfg = ARM_DEFINITIONS[arm_key]
        instances, _ = self.adapter.load_and_validate(max_examples=limit)

        results_file = self.output_dir / f"predictions_{arm_key}.jsonl"
        
        # Resumption / Deduplication check
        completed_ids = set()
        if resume and results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            completed_ids.add(record["question_id"])
                        except Exception:
                            pass

        print(f"[{arm_key}] Running '{arm_cfg['name']}' ({len(instances)} instances, {len(completed_ids)} already completed)...")
        
        chunker = TurnAwareChunker(window_size=3, overlap=1)
        total_time_start = time.time()
        processed_count = 0

        # Open in append mode for incremental flushing
        with open(results_file, "a", encoding="utf-8") as out_f:
            for idx, inst in enumerate(instances):
                if inst.question_id in completed_ids:
                    continue

                q_start = time.time()
                memories = self.adapter.convert_instance_to_memories(inst)
                chunks = chunker.chunk_all_sessions(inst.sessions)

                # SQLite graph store setup (if graph expansion or temporal graph needed)
                db_path = self.output_dir / f"temp_{arm_key}_{inst.question_id}.db"
                if db_path.exists():
                    db_path.unlink()

                store = SQLiteGraphStore(db_path)
                t_graph_ms = 0.0

                if arm_cfg["use_chunk_graph"] or arm_cfg.get("dynamic_graph_fallback", False):
                    t_g_start = time.time()
                    turn_graph_builder = TurnGraphBuilder(store)
                    turn_graph_builder.build_from_chunks(chunks)
                    t_graph_ms = (time.time() - t_g_start) * 1000

                # Relevance threshold: 0.35 on abstention questions to filter distractors
                min_score = 0.35 if inst.is_abstention else None

                t_ans_start = time.time()
                try:
                    ans_res = answer_question(
                        query=inst.question,
                        k=self.k,
                        min_score=min_score,
                        memories=memories,
                        model=self.model,
                        llm_caller=self.llm_caller,
                        use_graph=False,
                        graph_store=store,
                        use_chunking=True,
                        chunks=chunks,
                        max_context_tokens=1500,
                        use_ontology=arm_cfg["use_ontology"],
                        use_chunk_graph=arm_cfg["use_chunk_graph"],
                        min_edge_relevance=arm_cfg["min_edge_relevance"],
                        enable_temporal=arm_cfg["enable_temporal"],
                        direct_seed_temporal=arm_cfg["direct_seed_temporal"],
                        dynamic_graph_fallback=arm_cfg.get("dynamic_graph_fallback", False),
                        max_chunks_per_session=arm_cfg.get("max_chunks_per_session", None)
                    )
                    t_ans_ms = (time.time() - t_ans_start) * 1000
                    total_q_ms = (time.time() - q_start) * 1000

                    hypothesis = ans_res.get("answer", "")
                    status = ans_res.get("status", "answered")
                    citations = [c.get("chunk_id") or c.get("memory_id") for c in ans_res.get("citations", [])]
                    direct_ev = ans_res.get("retrieved_context", {}).get("direct_evidence", [])
                    retrieved_chunk_ids = [c.get("chunk_id") for c in direct_ev if "chunk_id" in c]
                    gold_rank = None
                    if inst.answer_session_ids:
                        target_sids = {str(s) for s in inst.answer_session_ids} | {f"sess_{s}" for s in inst.answer_session_ids}
                        for r_i, c in enumerate(direct_ev):
                            sid = str(c.get("session_id") or c.get("memory_id") or "")
                            cid = str(c.get("chunk_id") or "")
                            if sid in target_sids or any(ts in cid for ts in target_sids):
                                gold_rank = r_i + 1
                                break
                    failure_reason = None
                except Exception as e:
                    t_ans_ms = (time.time() - t_ans_start) * 1000
                    total_q_ms = (time.time() - q_start) * 1000
                    hypothesis = f"Execution error: {str(e)}"
                    status = "error"
                    citations = []
                    retrieved_chunk_ids = []
                    gold_rank = None
                    failure_reason = str(e)
                finally:
                    store.close()
                    if db_path.exists():
                        try:
                            db_path.unlink()
                        except Exception:
                            pass

                result_record = {
                    "question_id": inst.question_id,
                    "task_type": inst.question_type,
                    "question": inst.question,
                    "gold_answer": inst.gold_answer,
                    "is_abstention": inst.is_abstention,
                    "arm": arm_key,
                    "arm_name": arm_cfg["name"],
                    "hypothesis": hypothesis,
                    "status": status,
                    "citations": citations,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                    "gold_retrieval_rank": gold_rank,
                    "latency_ms": round(total_q_ms, 2),
                    "stage_latencies_ms": {
                        "graph_build": round(t_graph_ms, 2),
                        "answering": round(t_ans_ms, 2)
                    },
                    "failure_reason": failure_reason
                }

                out_f.write(json.dumps(result_record) + "\n")
                out_f.flush()
                completed_ids.add(inst.question_id)
                processed_count += 1

                if (idx + 1) % 10 == 0 or (idx + 1) == len(instances):
                    print(f"  [{arm_key}] Processed {idx + 1}/{len(instances)} questions...")

        total_runtime_s = round(time.time() - total_time_start, 2)
        print(f"[{arm_key}] Finished {processed_count} new predictions in {total_runtime_s}s.")
        return {
            "arm": arm_key,
            "total_instances": len(completed_ids),
            "newly_processed": processed_count,
            "runtime_s": total_runtime_s,
            "output_file": str(results_file)
        }

    def evaluate_with_judge(
        self,
        arm_key: str,
        resume: bool = True
    ) -> dict[str, Any]:
        """
        Evaluate generated predictions using the local diagnostic judge.
        """
        preds_file = self.output_dir / f"predictions_{arm_key}.jsonl"
        judge_out_file = self.output_dir / f"judge_results_{arm_key}.jsonl"

        if not preds_file.exists():
            raise FileNotFoundError(f"No predictions file found for {arm_key} at {preds_file}")

        completed_judge_ids = set()
        if resume and judge_out_file.exists():
            with open(judge_out_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            completed_judge_ids.add(item["question_id"])
                        except Exception:
                            pass

        records = []
        with open(preds_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        print(f"[{arm_key}] Running Diagnostic Gemma Judge on {len(records)} predictions ({len(completed_judge_ids)} already judged)...")
        
        judge_rubric_template = """You are a strict, objective evaluation judge assessing whether an AI assistant's response accurately answers a user question according to a reference gold answer.

User Question: {question}
Reference Gold Answer: {gold_answer}
Candidate Prediction: {hypothesis}
Is Abstention Required: {is_abstention}

Evaluation Rules:
1. For answerable questions (Is Abstention Required: False):
   - Return is_correct: true if the prediction accurately captures the factual information stated in the gold answer, including semantic paraphrases, equivalent dates/durations, and spelled-out numbers vs digits.
   - Return is_correct: false if the prediction is factually wrong, contradicts the evidence, fails to provide the requested information, or improperly abstains.
2. For unanswerable questions (Is Abstention Required: True):
   - Return is_correct: true if the candidate correctly identifies that the information is missing, unrecorded, or unanswerable from memory.
   - Return is_correct: false if the candidate invents an answer or provides unsupported factual claims.

Respond ONLY with a valid JSON object matching this schema:
{{
  "is_correct": true,
  "reasoning": "A concise one-sentence factual justification."
}}
"""
        correct_count = 0
        total_judged = 0
        ans_correct = 0
        ans_total = 0
        abs_correct = 0
        abs_total = 0

        with open(judge_out_file, "a", encoding="utf-8") as out_f:
            for idx, r in enumerate(records):
                qid = r["question_id"]
                if qid in completed_judge_ids:
                    continue

                prompt = judge_rubric_template.format(
                    question=r["question"],
                    gold_answer=r["gold_answer"],
                    hypothesis=r["hypothesis"],
                    is_abstention=str(r["is_abstention"])
                )

                t0 = time.time()
                is_correct = False
                reasoning = "Judge execution failure"
                judge_failed = False
                raw_judge_text = ""

                try:
                    if self.judge_caller is not None:
                        raw_judge_text = self.judge_caller(prompt)
                    else:
                        if chat is None:
                            raise RuntimeError("Ollama chat not available")
                        resp = chat(
                            model=self.judge_model,
                            messages=[{"role": "user", "content": prompt}],
                            format="json",
                            options={"temperature": 0.0, "num_ctx": 4096}
                        )
                        raw_judge_text = resp.message.content

                    # Parse JSON safely
                    try:
                        parsed = json.loads(raw_judge_text)
                        is_correct = bool(parsed.get("is_correct", False))
                        reasoning = parsed.get("reasoning", "")
                    except Exception:
                        # Fallback parsing
                        import re
                        m = re.search(r'"is_correct"\s*:\s*(true|false)', raw_judge_text, re.IGNORECASE)
                        if m:
                            is_correct = (m.group(1).lower() == "true")
                            reasoning = "Regex fallback parsed."
                        else:
                            judge_failed = True
                            is_correct = False
                            reasoning = f"Malformed judge JSON: {raw_judge_text[:100]}"
                except Exception as e:
                    judge_failed = True
                    is_correct = False
                    reasoning = f"Judge exception: {str(e)}"

                judge_latency_ms = round((time.time() - t0) * 1000, 2)

                judge_record = {
                    "question_id": qid,
                    "task_type": r["task_type"],
                    "is_abstention": r["is_abstention"],
                    "arm": arm_key,
                    "is_correct": is_correct,
                    "reasoning": reasoning,
                    "judge_failed": judge_failed,
                    "latency_ms": judge_latency_ms,
                    "raw_response": raw_judge_text
                }

                out_f.write(json.dumps(judge_record) + "\n")
                out_f.flush()
                completed_judge_ids.add(qid)

                if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
                    print(f"  [{arm_key}] Judged {idx + 1}/{len(records)} predictions...")

        # Compute summary
        with open(judge_out_file, "r", encoding="utf-8") as f:
            all_judged = [json.loads(line) for line in f if line.strip()]

        for j in all_judged:
            total_judged += 1
            if j["is_correct"]:
                correct_count += 1
            if j["is_abstention"]:
                abs_total += 1
                if j["is_correct"]:
                    abs_correct += 1
            else:
                ans_total += 1
                if j["is_correct"]:
                    ans_correct += 1

        overall_acc = round((correct_count / total_judged * 100), 2) if total_judged > 0 else 0.0
        ans_acc = round((ans_correct / ans_total * 100), 2) if ans_total > 0 else 0.0
        abs_acc = round((abs_correct / abs_total * 100), 2) if abs_total > 0 else 0.0

        summary = {
            "arm": arm_key,
            "arm_name": ARM_DEFINITIONS[arm_key]["name"],
            "total_instances": total_judged,
            "overall_accuracy": overall_acc,
            "answerable_accuracy": ans_acc,
            "abstention_accuracy": abs_acc,
            "abstentions_correct": abs_correct,
            "total_abstentions": abs_total,
            "answerable_correct": ans_correct,
            "total_answerable": ans_total
        }

        summary_file = self.output_dir / f"summary_{arm_key}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[{arm_key}] Evaluation complete: Overall {overall_acc}%, Answerable {ans_acc}%, Abstention {abs_acc}%")
        return summary
