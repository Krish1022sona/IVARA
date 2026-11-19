"""
Day 2: Held-Out End-to-End Evaluation Runner
Compares:
1. gpt-oss:latest (production baseline)
2. qwen2.5:14b (candidate)

Strict requirements:
- Exactly identical 60 held-out test questions (stratified 5/cat across 12 categories).
- Exactly identical adaptive retrieval pipeline (Arm E: Adaptive Episodic + Semantic + Safeguarded Graph).
- Preserves all production configurations, prompts, and databases.
- Runs sequentially.
- Captures token counts, latency, VRAM, and full answer & citation metrics.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

import ollama

from evaluation.validation_sprint.dataset_expanded import QUESTIONS
from evaluation.validation_sprint.metrics import (
    evaluate_retrieval,
    evaluate_answer,
    evaluate_citations,
    _get_conversation_session_map
)
from src.conversations.store import ConversationStore
from src.graph.store import SQLiteGraphStore
from src.answering import answer_question, AnswerPayload, ANSWERING_SYSTEM_PROMPT
from src.unified_retrieval.pipeline import UnifiedRetriever

BENCHMARK_DIR = Path(__file__).resolve().parent
IVARA_STORE_DIR = WORKSPACE_ROOT / "evaluation" / "validation_sprint" / "ivara_store_validation"


def get_vram_usage_mb() -> float | None:
    """Query current GPU VRAM usage in MiB using nvidia-smi."""
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return float(res.stdout.strip())
    except Exception:
        return None


def detect_repetition_loop(text: str) -> bool:
    """Detect pathological sentence or phrase repetition loops in generation."""
    if not text or len(text) < 100:
        return False
    phrases = re.findall(r'[^.!?\n]{15,}[.!?\n]', text)
    if not phrases:
        return False
    from collections import Counter
    counts = Counter(p.strip().lower() for p in phrases)
    most_common = counts.most_common(1)
    if most_common and most_common[0][1] >= 4:
        return True
    return False


def check_factual_support(answer: str, cited_items: list[dict[str, Any]], key_facts: list[str]) -> dict[str, Any]:
    """
    Verify whether the text of cited evidence items actually contains
    the stated facts / key facts (distinguishing citation validity from factual support).
    """
    if not cited_items or not key_facts:
        return {"supported_facts_count": 0, "factual_support_rate": 1.0 if not key_facts else 0.0}

    combined_cited_text = " ".join(
        (str(item.get("text", "")) + " " + " ".join(item.get("direct_facts", [])))
        for item in cited_items
    ).lower()

    supported = 0
    for kf in key_facts:
        kf_clean = " ".join(re.sub(r"[^\w\s]", " ", kf.lower()).split())
        if kf_clean in combined_cited_text:
            supported += 1

    rate = supported / len(key_facts) if key_facts else 1.0
    return {
        "supported_facts_count": supported,
        "factual_support_rate": round(rate, 4)
    }


def check_multihop_citation_completeness(cited_items: list[dict[str, Any]], supporting_sessions: list[str], session_map: dict[str, str]) -> dict[str, Any]:
    """
    Check if cited evidence spans all required supporting sessions in multi-hop queries.
    """
    if not supporting_sessions or len(supporting_sessions) <= 1:
        return {"is_multihop": False, "covered_sessions_ratio": 1.0, "multihop_complete": True}

    cited_sessions = set()
    for item in cited_items:
        s_id = item.get("session_id")
        src = str(item.get("source", ""))
        text = str(item.get("text", ""))
        if not s_id and src:
            m = re.search(r"session_\d{2}", src)
            if m:
                s_id = m.group(0)
            else:
                c_match = re.search(r"conv_[a-f0-9]+", src)
                if c_match and c_match.group(0) in session_map:
                    s_id = session_map[c_match.group(0)]
        if s_id:
            cited_sessions.add(s_id)
        for supp in supporting_sessions:
            if supp in src or supp in text:
                cited_sessions.add(supp)

    covered = len(cited_sessions.intersection(supporting_sessions))
    ratio = covered / len(supporting_sessions)
    return {
        "is_multihop": True,
        "required_sessions": supporting_sessions,
        "covered_sessions": list(cited_sessions.intersection(supporting_sessions)),
        "covered_sessions_ratio": round(ratio, 4),
        "multihop_complete": ratio >= 0.99
    }


def run_evaluation_for_model(
    model_tag: str,
    clean_name: str,
    questions: list[dict[str, Any]],
    retriever: UnifiedRetriever,
    session_map: dict[str, str],
    out_dir: Path,
    resume: bool = True
) -> dict[str, Any]:
    """Run sequential evaluation for a model on the 60 held-out questions."""
    out_file = out_dir / f"predictions_{clean_name}.json"
    predictions = {}

    if resume and out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                predictions = json.load(f)
            print(f"[{clean_name}] Resuming from {len(predictions)} cached predictions.")
        except Exception:
            predictions = {}

    print(f"\n=================================================================")
    print(f"  RUNNING HELD-OUT EVALUATION: {model_tag} ({len(questions)} questions)")
    print(f"=================================================================")

    last_call_stats = {}

    def custom_llm_caller(user_prompt: str, system_prompt: str) -> str:
        """Call Ollama and record token statistics."""
        t_llm_0 = time.perf_counter()
        resp = ollama.chat(
            model=model_tag,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            format=AnswerPayload.model_json_schema(),
            options={"temperature": 0, "num_ctx": 4096, "num_predict": 1024}
        )
        t_llm_1 = time.perf_counter()
        duration_ms = (t_llm_1 - t_llm_0) * 1000.0

        eval_count = getattr(resp, "eval_count", 0) or 0
        prompt_eval_count = getattr(resp, "prompt_eval_count", 0) or 0
        eval_duration_ms = getattr(resp, "eval_duration", 0) / 1e6 if hasattr(resp, "eval_duration") else 0.0

        tps = round((eval_count / (eval_duration_ms / 1000.0)), 2) if eval_duration_ms > 0 else (
            round(eval_count / (duration_ms / 1000.0), 2) if duration_ms > 0 else None
        )

        last_call_stats["eval_count"] = eval_count
        last_call_stats["prompt_eval_count"] = prompt_eval_count
        last_call_stats["tokens_per_sec"] = tps
        last_call_stats["llm_latency_ms"] = round(duration_ms, 1)

        return resp.message.content

    for idx, q_spec in enumerate(questions, 1):
        q_id = q_spec["id"]
        q_text = q_spec["question"]
        q_cat = q_spec["category"]

        if q_id in predictions:
            continue

        vram_start = get_vram_usage_mb()
        last_call_stats.clear()

        t0 = time.perf_counter()
        try:
            ans = answer_question(
                query=q_text,
                k=5,
                min_score=0.30,
                unified_retriever=retriever,
                enable_episodic=True,
                enable_semantic=True,
                enable_graph_channel=True,
                model=model_tag,
                llm_caller=custom_llm_caller,
                enable_temporal=True,
                use_adaptive_routing=True
            )
            t1 = time.perf_counter()
            total_elapsed_ms = round((t1 - t0) * 1000.0, 1)

            vram_end = get_vram_usage_mb()

            raw_ctx = ans.get("retrieved_context") or {}
            retrieved_items = raw_ctx.get("direct_evidence") or raw_ctx.get("retrieved_memories", [])
            valid_citations = ans.get("citations", [])
            cited_ids = [c["memory_id"] for c in valid_citations if "memory_id" in c]

            supp_sessions = q_spec.get("supporting_sessions", [])
            ret_metrics = evaluate_retrieval(
                retrieved_items=retrieved_items,
                supporting_sessions=supp_sessions,
                is_answerable=q_spec.get("is_answerable", True),
                session_map=session_map
            )

            ans_metrics = evaluate_answer(
                prediction=ans,
                question_spec=q_spec
            )

            cite_metrics = evaluate_citations(
                cited_ids=cited_ids,
                retrieved_items=retrieved_items
            )

            # Factual support check
            fact_support = check_factual_support(
                answer=ans.get("answer", ""),
                cited_items=valid_citations,
                key_facts=q_spec.get("key_facts", [])
            )

            # Multi-hop completeness check
            multihop_check = check_multihop_citation_completeness(
                cited_items=valid_citations,
                supporting_sessions=supp_sessions,
                session_map=session_map
            )

            # Repetition detection
            has_rep = detect_repetition_loop(ans.get("answer", ""))

            predictions[q_id] = {
                "question_id": q_id,
                "category": q_cat,
                "query": q_text,
                "status": ans.get("status"),
                "answer": ans.get("answer"),
                "limitations": ans.get("limitations"),
                "citations": cited_ids,
                "rejected_citations": ans.get("rejected_citation_ids", []),
                "retrieval_metrics": ret_metrics,
                "answer_metrics": ans_metrics,
                "citation_metrics": cite_metrics,
                "factual_support": fact_support,
                "multihop_citation": multihop_check,
                "repetition_loop_detected": has_rep,
                "latency_ms": total_elapsed_ms,
                "llm_latency_ms": last_call_stats.get("llm_latency_ms"),
                "eval_count": last_call_stats.get("eval_count", 0),
                "prompt_eval_count": last_call_stats.get("prompt_eval_count", 0),
                "tokens_per_sec": last_call_stats.get("tokens_per_sec"),
                "vram_start_mb": vram_start,
                "vram_end_mb": vram_end
            }

            print(f"[{clean_name}] {idx}/{len(questions)}: {q_id} [{q_cat}] -> status={ans.get('status')} | acc={ans_metrics['binary_accuracy']} | {total_elapsed_ms}ms")

        except Exception as e:
            t1 = time.perf_counter()
            total_elapsed_ms = round((t1 - t0) * 1000.0, 1)
            vram_end = get_vram_usage_mb()
            print(f"[{clean_name}] {idx}/{len(questions)}: {q_id} FAILED ({e})")
            predictions[q_id] = {
                "question_id": q_id,
                "category": q_cat,
                "query": q_text,
                "status": "error",
                "answer": f"Error: {e}",
                "citations": [],
                "rejected_citations": [],
                "retrieval_metrics": {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "evidence_coverage": 0.0},
                "answer_metrics": {"is_answerable": q_spec.get("is_answerable", True), "abstained": False, "correct_abstention": False, "false_rejection": True, "false_acceptance": False, "key_facts_found": [], "fact_coverage": 0.0, "binary_accuracy": 0.0, "exact_match": 0.0},
                "citation_metrics": {"citation_validity": 1.0, "hallucinated_citations": 0},
                "factual_support": {"supported_facts_count": 0, "factual_support_rate": 0.0},
                "multihop_citation": {"is_multihop": False, "covered_sessions_ratio": 0.0, "multihop_complete": False},
                "repetition_loop_detected": False,
                "latency_ms": total_elapsed_ms,
                "eval_count": 0,
                "prompt_eval_count": 0,
                "tokens_per_sec": None,
                "vram_start_mb": vram_start,
                "vram_end_mb": vram_end
            }

        # Checkpoint every question
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)

    return predictions


def compute_paired_statistics(preds_base: dict[str, Any], preds_cand: dict[str, Any]) -> dict[str, Any]:
    """
    Compute McNemar exact test for binary accuracy and Bootstrap 95% CI for fact coverage.
    """
    common_ids = sorted(list(set(preds_base.keys()).intersection(preds_cand.keys())))
    N = len(common_ids)

    # 1. McNemar Contingency Table
    # n00: both incorrect, n11: both correct
    # n10: Base correct, Cand incorrect (Base win)
    # n01: Cand correct, Base incorrect (Cand win)
    n00 = 0
    n11 = 0
    n10 = 0
    n01 = 0

    base_acc_list = []
    cand_acc_list = []
    base_cov_list = []
    cand_cov_list = []

    for qid in common_ids:
        b_acc = preds_base[qid]["answer_metrics"]["binary_accuracy"]
        c_acc = preds_cand[qid]["answer_metrics"]["binary_accuracy"]
        b_cov = preds_base[qid]["answer_metrics"]["fact_coverage"]
        c_cov = preds_cand[qid]["answer_metrics"]["fact_coverage"]

        base_acc_list.append(b_acc)
        cand_acc_list.append(c_acc)
        base_cov_list.append(b_cov)
        cand_cov_list.append(c_cov)

        if b_acc == 1.0 and c_acc == 1.0:
            n11 += 1
        elif b_acc == 0.0 and c_acc == 0.0:
            n00 += 1
        elif b_acc == 1.0 and c_acc == 0.0:
            n10 += 1
        elif b_acc == 0.0 and c_acc == 1.0:
            n01 += 1

    # Exact binomial two-sided test for McNemar:
    # Under H0: b ~ Binomial(n10 + n01, 0.5)
    discordant = n10 + n01
    if discordant == 0:
        p_mcnemar = 1.0
    else:
        k = min(n10, n01)
        # Two-sided exact binomial p-value: sum_{i=0}^k (n choose i) * 0.5^n * 2
        p_val = 0.0
        for i in range(k + 1):
            p_val += math.comb(discordant, i) * (0.5 ** discordant)
        p_mcnemar = min(1.0, 2.0 * p_val)

    # 2. Bootstrap 95% Confidence Interval for Fact Coverage Difference
    random.seed(42)
    diffs = [c - b for c, b in zip(cand_cov_list, base_cov_list)]
    observed_mean_diff = sum(diffs) / len(diffs)

    B = 2000
    boot_means = []
    for _ in range(B):
        sample = random.choices(diffs, k=len(diffs))
        boot_means.append(sum(sample) / len(sample))

    boot_means.sort()
    ci_lower = round(boot_means[int(0.025 * B)], 4)
    ci_upper = round(boot_means[int(0.975 * B)], 4)

    return {
        "sample_size": N,
        "contingency_table": {
            "candidate_win_base_loss (n01)": n01,
            "base_win_candidate_loss (n10)": n10,
            "both_correct (n11)": n11,
            "both_incorrect (n00)": n00,
            "discordant_pairs": discordant,
            "concordant_pairs": n00 + n11
        },
        "mcnemar_exact_two_sided_p_value": round(p_mcnemar, 5),
        "statistically_significant_p05": p_mcnemar < 0.05,
        "fact_coverage_observed_mean_diff": round(observed_mean_diff, 4),
        "fact_coverage_bootstrap_95_ci": [ci_lower, ci_upper]
    }


def main():
    parser = argparse.ArgumentParser(description="Day 2 Held-Out Benchmark Runner")
    parser.add_argument("--stratified", type=int, default=5, help="Questions per category (default 5 for 60 questions)")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume cached predictions")
    args = parser.parse_args()

    out_dir = BENCHMARK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load 60 held-out test questions
    test_questions = [q for q in QUESTIONS if q.get("split") == "test"]
    by_cat = defaultdict(list)
    for q in test_questions:
        by_cat[q["category"]].append(q)

    stratified_q = []
    for cat in sorted(by_cat.keys()):
        stratified_q.extend(by_cat[cat][:args.stratified])

    print(f"Loaded {len(stratified_q)} stratified held-out test questions across {len(by_cat)} categories.")

    # 2. Setup stores and retriever
    conv_db = IVARA_STORE_DIR / "conversations.db"
    graph_db = IVARA_STORE_DIR / "memory_graph.db"
    memories_json = IVARA_STORE_DIR / "memories.json"

    conv_store = ConversationStore(conv_db)
    graph_store = SQLiteGraphStore(graph_db)
    retriever = UnifiedRetriever(
        conversation_store=conv_store,
        memories_path=memories_json,
        graph_store=graph_store,
        use_adaptive_routing=True
    )

    session_map = _get_conversation_session_map(conv_db)

    # 3. Run sequentially
    print("\n>>> PHASE 1: Running GPT-OSS Baseline sequentially...")
    preds_gpt = run_evaluation_for_model(
        model_tag="gpt-oss:latest",
        clean_name="gpt_oss",
        questions=stratified_q,
        retriever=retriever,
        session_map=session_map,
        out_dir=out_dir,
        resume=not args.no_resume
    )

    print("\n>>> PHASE 2: Running Qwen 2.5 Candidate sequentially...")
    preds_qwen = run_evaluation_for_model(
        model_tag="qwen2.5:14b",
        clean_name="qwen2_5",
        questions=stratified_q,
        retriever=retriever,
        session_map=session_map,
        out_dir=out_dir,
        resume=not args.no_resume
    )

    # 4. Compute Statistical Comparison
    stats = compute_paired_statistics(preds_gpt, preds_qwen)
    stats_file = out_dir / "statistical_comparison.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n=================================================================")
    print("  DAY 2 STATISTICAL COMPARISON COMPLETE")
    print(f"  Candidate Wins (n01): {stats['contingency_table']['candidate_win_base_loss (n01)']}")
    print(f"  Baseline Wins (n10):  {stats['contingency_table']['base_win_candidate_loss (n10)']}")
    print(f"  Ties:                 {stats['contingency_table']['concordant_pairs']}")
    print(f"  Exact McNemar p-value:{stats['mcnemar_exact_two_sided_p_value']}")
    print(f"  Fact Coverage 95% CI: {stats['fact_coverage_bootstrap_95_ci']}")
    print("=================================================================")


if __name__ == "__main__":
    main()

