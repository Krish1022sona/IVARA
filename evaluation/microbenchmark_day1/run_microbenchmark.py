"""
Day 1 Controlled Micro-Benchmark Runner
Evaluates gpt-oss:latest vs qwen2.5:14b under strictly identical prompts,
schemas, and inference parameters without modifying production systems.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any
from ollama import chat

# Import production schemas and prompts (read-only)
from src.extraction.extractor import Extraction, SYSTEM_PROMPT as EXTRACTION_PROMPT
from src.answering import AnswerPayload, ANSWERING_SYSTEM_PROMPT, format_user_prompt
from evaluation.microbenchmark_day1.benchmark_dataset import EXTRACTION_DATASET, ANSWERING_DATASET


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
    # Check for repeated sentences or clauses
    phrases = re.findall(r'[^.!?\n]{15,}[.!?\n]', text)
    if not phrases:
        return False
    from collections import Counter
    counts = Counter(p.strip().lower() for p in phrases)
    most_common = counts.most_common(1)
    if most_common and most_common[0][1] >= 4:
        return True
    return False


def run_extraction_for_model(model_name: str) -> list[dict[str, Any]]:
    """Execute Part A: Knowledge Extraction across 10 benchmark turns."""
    results = []
    print(f"\n==================================================")
    print(f"Running Part A (Extraction) for Model: {model_name}")
    print(f"==================================================")

    for item in EXTRACTION_DATASET:
        tid = item["id"]
        text = item["text"]
        print(f"-> Ingesting {tid} ({item['description'][:40]}...)...", end=" ", flush=True)

        user_content = f"Extract knowledge from this memory:\n\n{text}"
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_content}
        ]

        vram_start = get_vram_usage_mb()
        t0 = time.perf_counter()

        raw_content = ""
        schema_valid = False
        parsed_extraction = None
        error_msg = None
        prompt_eval_count = 0
        eval_count = 0
        prompt_eval_duration_ms = 0.0
        eval_duration_ms = 0.0

        try:
            resp = chat(
                model=model_name,
                messages=messages,
                format=Extraction.model_json_schema(),
                options={
                    "temperature": 0,
                    "num_ctx": 16384,
                    "num_predict": 4096
                }
            )
            t1 = time.perf_counter()
            total_time_ms = (t1 - t0) * 1000.0

            raw_content = resp.message.content
            # Check token statistics if provided by ollama
            prompt_eval_count = getattr(resp, "prompt_eval_count", 0) or 0
            eval_count = getattr(resp, "eval_count", 0) or 0
            prompt_eval_duration_ms = getattr(resp, "prompt_eval_duration", 0) / 1e6 if hasattr(resp, "prompt_eval_duration") else 0.0
            eval_duration_ms = getattr(resp, "eval_duration", 0) / 1e6 if hasattr(resp, "eval_duration") else 0.0

            # Validate against Pydantic schema
            parsed_extraction = Extraction.model_validate_json(raw_content)
            schema_valid = True
            print(f"OK ({total_time_ms:.1f}ms, {eval_count} tokens)")

        except Exception as e:
            t1 = time.perf_counter()
            total_time_ms = (t1 - t0) * 1000.0
            error_msg = str(e)
            print(f"FAILED ({total_time_ms:.1f}ms: {e})")

        vram_end = get_vram_usage_mb()

        # Check for compound entity traps
        extracted_entities = []
        compound_entities_found = []
        if parsed_extraction:
            for ent in parsed_extraction.entities:
                extracted_entities.append({"name": ent.name, "type": ent.type, "confidence": ent.confidence})
                # Check if entity contains conjunctions ' and ', ' & ', ' nor ' or multiple names
                ename = ent.name.strip().lower()
                if " and " in ename or " & " in ename or " nor " in ename:
                    compound_entities_found.append(ent.name)

        extracted_relationships = []
        if parsed_extraction:
            for rel in parsed_extraction.relationships:
                extracted_relationships.append({
                    "source": rel.source,
                    "relation": rel.relation,
                    "target": rel.target,
                    "confidence": rel.confidence,
                    "origin": rel.origin,
                    "evidence": rel.evidence
                })
                # Also check if relationship target contains compound entity
                rtarget = rel.target.strip().lower()
                if (" and " in rtarget or " & " in rtarget or " nor " in rtarget) and len(rtarget.split()) >= 3:
                    compound_entities_found.append(rel.target)

        results.append({
            "id": tid,
            "description": item["description"],
            "model": model_name,
            "latency_ms": round(total_time_ms, 2),
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "tokens_per_sec": round((eval_count / (eval_duration_ms / 1000.0)), 2) if eval_duration_ms > 0 else None,
            "schema_valid": schema_valid,
            "error": error_msg,
            "vram_start_mb": vram_start,
            "vram_end_mb": vram_end,
            "extracted_entities": extracted_entities,
            "extracted_relationships": extracted_relationships,
            "compound_entities_found": list(set(compound_entities_found)),
            "raw_output": raw_content
        })

    return results


def run_answering_for_model(model_name: str) -> list[dict[str, Any]]:
    """Execute Part B: Grounded Answering across 10 benchmark queries."""
    results = []
    print(f"\n==================================================")
    print(f"Running Part B (Answering) for Model: {model_name}")
    print(f"==================================================")

    for item in ANSWERING_DATASET:
        qid = item["id"]
        category = item["category"]
        query = item["query"]
        context = item["context"]
        print(f"-> Answering {qid} [{category}]...", end=" ", flush=True)

        # Format prompt with exact production formatter
        # Mock evidence_gate report for format_user_prompt
        user_prompt = format_user_prompt(query, context)
        messages = [
            {"role": "system", "content": ANSWERING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        vram_start = get_vram_usage_mb()
        t0 = time.perf_counter()

        raw_content = ""
        schema_valid = False
        parsed_payload = None
        error_msg = None
        prompt_eval_count = 0
        eval_count = 0
        prompt_eval_duration_ms = 0.0
        eval_duration_ms = 0.0

        try:
            resp = chat(
                model=model_name,
                messages=messages,
                format=AnswerPayload.model_json_schema(),
                options={
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": 1024
                }
            )
            t1 = time.perf_counter()
            total_time_ms = (t1 - t0) * 1000.0

            raw_content = resp.message.content
            prompt_eval_count = getattr(resp, "prompt_eval_count", 0) or 0
            eval_count = getattr(resp, "eval_count", 0) or 0
            prompt_eval_duration_ms = getattr(resp, "prompt_eval_duration", 0) / 1e6 if hasattr(resp, "prompt_eval_duration") else 0.0
            eval_duration_ms = getattr(resp, "eval_duration", 0) / 1e6 if hasattr(resp, "eval_duration") else 0.0

            # Validate against Pydantic schema
            parsed_payload = AnswerPayload.model_validate_json(raw_content)
            schema_valid = True
            print(f"OK ({total_time_ms:.1f}ms, {eval_count} tokens)")

        except Exception as e:
            t1 = time.perf_counter()
            total_time_ms = (t1 - t0) * 1000.0
            error_msg = str(e)
            print(f"FAILED ({total_time_ms:.1f}ms: {e})")

        vram_end = get_vram_usage_mb()

        # Check repetition loop
        has_repetition = detect_repetition_loop(raw_content)

        answer_status = parsed_payload.status if parsed_payload else None
        answer_text = parsed_payload.answer if parsed_payload else raw_content
        cited_ids = parsed_payload.cited_memory_ids if parsed_payload else []

        results.append({
            "id": qid,
            "category": category,
            "query": query,
            "model": model_name,
            "latency_ms": round(total_time_ms, 2),
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "tokens_per_sec": round((eval_count / (eval_duration_ms / 1000.0)), 2) if eval_duration_ms > 0 else None,
            "schema_valid": schema_valid,
            "repetition_loop_detected": has_repetition,
            "error": error_msg,
            "vram_start_mb": vram_start,
            "vram_end_mb": vram_end,
            "status": answer_status,
            "answer": answer_text,
            "cited_memory_ids": cited_ids,
            "gold_status": item["gold_status"],
            "gold_key_facts": item["gold_key_facts"],
            "expected_citations": item["expected_citations"],
            "raw_output": raw_content
        })

    return results


def main():
    out_dir = Path("evaluation/microbenchmark_day1")
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["gpt-oss:latest", "qwen2.5:14b"]
    all_runs = {}

    for model in models:
        print(f"\n==================================================================")
        print(f"STARTING EVALUATION RUN FOR: {model}")
        print(f"==================================================================")
        
        ext_results = run_extraction_for_model(model)
        ans_results = run_answering_for_model(model)

        clean_name = "gpt_oss" if "gpt-oss" in model else "qwen2_5"
        raw_file = out_dir / f"raw_outputs_{clean_name}.json"
        
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump({
                "model": model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "extraction_results": ext_results,
                "answering_results": ans_results
            }, f, indent=2)
        
        print(f"Saved raw outputs to: {raw_file}")
        all_runs[clean_name] = {
            "extraction": ext_results,
            "answering": ans_results
        }

    # Consolidated summary
    summary_file = out_dir / "benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\nConsolidated micro-benchmark completed successfully! Saved to: {summary_file}")


if __name__ == "__main__":
    main()
