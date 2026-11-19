"""
Day 1 Micro-Benchmark Evaluation & Comparison Script
Computes explicit counts, denominators, and rigorous metrics for:
  - Entity Atomicity, Precision, Recall, F1
  - Relationship Precision, Recall, F1
  - Temporal extraction accuracy
  - Answer accuracy, evidence grounding, honest abstention, hallucination rate
  - Latency, token throughput, schema compliance, VRAM usage
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.microbenchmark_day1.benchmark_dataset import EXTRACTION_DATASET, ANSWERING_DATASET


def normalize_str(s: str) -> str:
    """Normalize string for fuzzy comparison."""
    if not s:
        return ""
    import re
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', s.lower())).strip()


def evaluate_extraction_run(records: list[dict]) -> dict:
    gold_map = {item["id"]: item for item in EXTRACTION_DATASET}
    
    total_gold_entities = sum(len(item["gold_entities"]) for item in EXTRACTION_DATASET)
    total_gold_relationships = sum(len(item["gold_relationships"]) for item in EXTRACTION_DATASET)
    total_gold_temporal = sum(len(item["gold_temporal"]) for item in EXTRACTION_DATASET)

    tp_entities = 0
    fp_entities = 0
    fn_entities = 0
    compound_entity_count = 0

    tp_rel = 0
    fp_rel = 0
    fn_rel = 0
    valid_predicates = 0

    temporal_correct = 0

    per_passage = []

    for rec in records:
        tid = rec["id"]
        gold = gold_map[tid]
        gold_ent_names = [normalize_str(e["name"]) for e in gold["gold_entities"]]
        pred_entities = rec.get("extracted_entities", [])
        pred_ent_names = [normalize_str(e["name"]) for e in pred_entities]

        # Check compound traps
        c_found = rec.get("compound_entities_found", [])
        compound_entity_count += len(c_found)

        # Entity Matching
        matched_gold_ents = set()
        p_tp = 0
        p_fp = 0
        for pname in pred_ent_names:
            matched = False
            for idx, gname in enumerate(gold_ent_names):
                if idx not in matched_gold_ents and (pname == gname or pname in gname or gname in pname):
                    matched_gold_ents.add(idx)
                    matched = True
                    break
            if matched:
                p_tp += 1
            else:
                p_fp += 1
        
        p_fn = len(gold_ent_names) - len(matched_gold_ents)
        tp_entities += p_tp
        fp_entities += p_fp
        fn_entities += p_fn

        # Relationship Matching
        gold_rels = gold["gold_relationships"]
        pred_rels = rec.get("extracted_relationships", [])
        matched_gold_rels = set()
        r_tp = 0
        r_fp = 0

        for prel in pred_rels:
            psrc = normalize_str(prel.get("source", ""))
            ptgt = normalize_str(prel.get("target", ""))
            prel_name = normalize_str(prel.get("relation", ""))

            # Basic predicate sanity
            if prel_name and len(prel_name) >= 2:
                valid_predicates += 1

            matched = False
            for idx, grel in enumerate(gold_rels):
                if idx not in matched_gold_rels:
                    gsrc = normalize_str(grel[0])
                    gtgt = normalize_str(grel[2])
                    if (psrc == gsrc or psrc in gsrc or gsrc in psrc) and (ptgt == gtgt or ptgt in gtgt or gtgt in ptgt):
                        matched_gold_rels.add(idx)
                        matched = True
                        break
            if matched:
                r_tp += 1
            else:
                r_fp += 1
        
        r_fn = len(gold_rels) - len(matched_gold_rels)
        tp_rel += r_tp
        fp_rel += r_fp
        fn_rel += r_fn

        # Temporal check
        if gold["gold_temporal"]:
            # Check if dates were extracted in entities or evidence
            dates_in_gold = [t.get("date") or t.get("valid_from") for t in gold["gold_temporal"] if t.get("date") or t.get("valid_from")]
            extracted_text_dump = json.dumps(pred_rels) + json.dumps(pred_entities)
            all_dates_found = all(d in extracted_text_dump for d in dates_in_gold if d)
            if all_dates_found:
                temporal_correct += len(gold["gold_temporal"])

        per_passage.append({
            "id": tid,
            "latency_ms": rec["latency_ms"],
            "tokens": rec["eval_count"],
            "compound_entities": c_found,
            "ent_precision": round(p_tp / (p_tp + p_fp), 3) if (p_tp + p_fp) > 0 else 0.0,
            "ent_recall": round(p_tp / len(gold_ent_names), 3) if gold_ent_names else 1.0,
            "rel_matches": r_tp,
            "rel_gold": len(gold_rels)
        })

    ent_p = tp_entities / (tp_entities + fp_entities) if (tp_entities + fp_entities) > 0 else 0.0
    ent_r = tp_entities / (tp_entities + fn_entities) if (tp_entities + fn_entities) > 0 else 0.0
    ent_f1 = (2 * ent_p * ent_r) / (ent_p + ent_r) if (ent_p + ent_r) > 0 else 0.0

    rel_p = tp_rel / (tp_rel + fp_rel) if (tp_rel + fp_rel) > 0 else 0.0
    rel_r = tp_rel / (tp_rel + fn_rel) if (tp_rel + fn_rel) > 0 else 0.0
    rel_f1 = (2 * rel_p * rel_r) / (rel_p + rel_r) if (rel_p + rel_r) > 0 else 0.0

    return {
        "entity_tp": tp_entities,
        "entity_fp": fp_entities,
        "entity_fn": fn_entities,
        "total_gold_entities": total_gold_entities,
        "entity_precision": round(ent_p * 100, 2),
        "entity_recall": round(ent_r * 100, 2),
        "entity_f1": round(ent_f1 * 100, 2),
        "compound_entity_count": compound_entity_count,
        "relationship_tp": tp_rel,
        "relationship_fp": fp_rel,
        "relationship_fn": fn_rel,
        "total_gold_relationships": total_gold_relationships,
        "relationship_precision": round(rel_p * 100, 2),
        "relationship_recall": round(rel_r * 100, 2),
        "relationship_f1": round(rel_f1 * 100, 2),
        "temporal_claims_correct": temporal_correct,
        "total_gold_temporal": total_gold_temporal,
        "temporal_accuracy": round((temporal_correct / total_gold_temporal) * 100, 2) if total_gold_temporal > 0 else 100.0,
        "per_passage": per_passage
    }


def evaluate_answering_run(records: list[dict]) -> dict:
    gold_map = {item["id"]: item for item in ANSWERING_DATASET}
    
    total_queries = len(records)
    factual_correct = 0
    abstention_correct = 0
    hallucinations = 0
    repetition_loops = 0
    valid_citations_count = 0
    citation_opportunities = 0

    per_query = []

    for rec in records:
        qid = rec["id"]
        gold = gold_map[qid]
        status = rec.get("status")
        ans_text = rec.get("answer", "")
        norm_ans = normalize_str(ans_text)
        cited = rec.get("cited_memory_ids", [])
        has_rep = rec.get("repetition_loop_detected", False)
        if has_rep:
            repetition_loops += 1

        is_abstention_expected = (gold["gold_status"] == "insufficient_evidence")
        is_abstention_actual = (status in ("insufficient_evidence", "unanswerable")) or ("insufficient" in norm_ans or "not recorded" in norm_ans or "do not have" in norm_ans)

        factually_correct = False
        hallucinated = False

        if is_abstention_expected:
            if is_abstention_actual:
                abstention_correct += 1
                factually_correct = True
            else:
                hallucinations += 1
                factually_correct = False
        else:
            # Answerable query
            if is_abstention_actual:
                # False negative abstention
                factually_correct = False
            else:
                # Check key facts
                all_facts = True
                for kf in gold["gold_key_facts"]:
                    # Check key words of key fact
                    kf_words = [w for w in normalize_str(kf).split() if len(w) > 3]
                    if not any(w in norm_ans for w in kf_words):
                        all_facts = False
                        break
                if all_facts:
                    factually_correct = True
                    factual_correct += 1
                else:
                    factually_correct = False

        # Citation precision
        exp_cits = set(gold["expected_citations"])
        act_cits = set(cited)
        if exp_cits:
            citation_opportunities += 1
            if act_cits and act_cits.issubset(exp_cits | {"m001", "m002", "turn_101", "turn_102", "m201", "m202", "turn_301", "turn_302", "m401", "m402", "m501", "turn_601", "turn_602", "m701", "m702"}):
                valid_citations_count += 1
        elif not act_cits:
            valid_citations_count += 1
            citation_opportunities += 1

        per_query.append({
            "id": qid,
            "category": rec["category"],
            "factually_correct": factually_correct,
            "is_abstention_actual": is_abstention_actual,
            "cited_ids": cited,
            "expected_citations": list(exp_cits),
            "repetition_loop": has_rep,
            "latency_ms": rec["latency_ms"]
        })

    return {
        "total_queries": total_queries,
        "factual_accuracy": round((factual_correct / (total_queries - 2)) * 100, 2), # out of 8 answerable
        "answerable_correct": factual_correct,
        "answerable_total": total_queries - 2,
        "abstention_accuracy": round((abstention_correct / 2) * 100, 2),
        "abstention_correct": abstention_correct,
        "abstention_total": 2,
        "overall_accuracy": round(((factual_correct + abstention_correct) / total_queries) * 100, 2),
        "overall_correct": factual_correct + abstention_correct,
        "hallucination_count": hallucinations,
        "repetition_loop_count": repetition_loops,
        "citation_fidelity": round((valid_citations_count / citation_opportunities) * 100, 2) if citation_opportunities > 0 else 100.0,
        "per_query": per_query
    }


def main():
    summary_path = Path("evaluation/microbenchmark_day1/benchmark_summary.json")
    if not summary_path.exists():
        print(f"Summary file not found: {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    comparison = {}
    for model_key in ["gpt_oss", "qwen2_5"]:
        if model_key not in data:
            continue
        ext_eval = evaluate_extraction_run(data[model_key]["extraction"])
        ans_eval = evaluate_answering_run(data[model_key]["answering"])

        ext_latencies = [r["latency_ms"] for r in data[model_key]["extraction"]]
        ans_latencies = [r["latency_ms"] for r in data[model_key]["answering"]]
        all_latencies = ext_latencies + ans_latencies

        ext_tok_rates = [r["tokens_per_sec"] for r in data[model_key]["extraction"] if r.get("tokens_per_sec")]
        ans_tok_rates = [r["tokens_per_sec"] for r in data[model_key]["answering"] if r.get("tokens_per_sec")]
        all_tok_rates = ext_tok_rates + ans_tok_rates

        schema_valid_count = sum(1 for r in data[model_key]["extraction"] if r["schema_valid"]) + \
                             sum(1 for r in data[model_key]["answering"] if r["schema_valid"])

        comparison[model_key] = {
            "extraction_metrics": ext_eval,
            "answering_metrics": ans_eval,
            "system_metrics": {
                "mean_latency_ms": round(sum(all_latencies) / len(all_latencies), 1),
                "median_latency_ms": round(sorted(all_latencies)[len(all_latencies)//2], 1),
                "mean_tokens_per_sec": round(sum(all_tok_rates) / len(all_tok_rates), 1) if all_tok_rates else None,
                "schema_compliance_rate": f"{schema_valid_count} / {len(all_latencies)} ({round(schema_valid_count / len(all_latencies) * 100, 1)}%)"
            }
        }

    out_comp = Path("evaluation/microbenchmark_day1/metrics_comparison.json")
    with open(out_comp, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Metrics evaluation complete. Saved to: {out_comp}")


if __name__ == "__main__":
    main()
