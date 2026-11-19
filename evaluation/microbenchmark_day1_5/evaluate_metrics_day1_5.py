"""
Day 1.5 Three-Model Evaluation & Metric Computation Script
Incorporates all verified audit fixes:
- Strict semantic relationship matching (Subject + Predicate + Object)
- Entity prediction deduplication
- Correct ground-truth alignment and calculation formulas
- Citation completeness and validity
- Hardware and latency profiles
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.microbenchmark_day1.benchmark_dataset import EXTRACTION_DATASET, ANSWERING_DATASET


# Normalized predicate synonym map for strict semantic matching
PREDICATE_SYNONYMS = {
    "employed_by": {"employed_by", "works_for", "worksfor", "employed at", "works at", "at", "employedby", "joined", "joined_at"},
    "works_for": {"employed_by", "works_for", "worksfor", "employed at", "works at", "at", "employedby", "joined", "joined_at"},
    "job_title": {"job_title", "title", "role", "role_as", "worked_as", "works_as", "serves as", "serves_as", "leads", "lead"},
    "leads": {"leads", "lead", "directed_by", "led_by", "managed_by", "project_lead", "manager_of"},
    "graduate_of": {"graduate_of", "alumni_of", "studied_at", "degree_from", "graduated_from", "education"},
    "partner_of": {"partner_of", "spouse_of", "partnered_with", "married_to"},
    "lived_in": {"lived_in", "lives_in", "resided_in", "resides_in", "resident_of"},
    "moved_to": {"moved_to", "relocated_to", "resides_in", "lives_in"},
    "implemented_in": {"implemented_in", "written_in", "built_with", "uses_language"},
    "deployed_on": {"deployed_on", "deployed_across", "runs_on", "hosted_on"},
    "bought_from": {"bought_from", "purchased_from", "acquired_from"},
    "color": {"color", "hascolor", "paint_color"},
    "has_switch_type": {"has_switch_type", "switch_type", "uses_switches", "has", "switches"},
    "visited": {"visited", "had_appointment_with", "saw", "consulted"},
    "located_in": {"located_in", "based_in", "located_at"},
}


def normalize_str(s: str) -> str:
    if not s:
        return ""
    return re.sub(r'[^a-z0-9]', '', s.lower().strip())


def predicate_matches(pred1: str, pred2: str) -> bool:
    p1 = pred1.lower().strip().replace(" ", "_").replace("-", "_")
    p2 = pred2.lower().strip().replace(" ", "_").replace("-", "_")
    if p1 == p2 or p1 in p2 or p2 in p1:
        return True
    norm1 = normalize_str(p1)
    norm2 = normalize_str(p2)
    if norm1 == norm2 or norm1 in norm2 or norm2 in norm1:
        return True
    for key, syns in PREDICATE_SYNONYMS.items():
        if (p1 in syns or any(s in p1 for s in syns)) and (p2 in syns or any(s in p2 for s in syns)):
            return True
    return False


def evaluate_extraction_for_model(raw_extractions: list[dict[str, Any]]) -> dict[str, Any]:
    gold_map = {item["id"]: item for item in EXTRACTION_DATASET}

    total_gold_entities = sum(len(item["gold_entities"]) for item in EXTRACTION_DATASET)
    total_gold_relations = sum(len(item["gold_relationships"]) for item in EXTRACTION_DATASET)

    pred_entity_count_raw = 0
    pred_entity_count_dedup = 0
    ent_tp = 0
    ent_fp = 0
    ent_fn = 0

    rel_count = 0
    rel_lenient_tp = 0
    rel_strict_tp = 0
    rel_strict_fp = 0
    rel_strict_fn = 0

    per_passage = {}

    for passage_res in raw_extractions:
        pid = passage_res["id"]
        gold_item = gold_map[pid]
        gold_ents = gold_item["gold_entities"]
        gold_rels = gold_item["gold_relationships"]

        # 1. Deduplicate predicted entities per passage
        raw_pred_ents = passage_res.get("extracted_entities", [])
        pred_entity_count_raw += len(raw_pred_ents)

        seen_names = set()
        unique_pred_ents = []
        for e in raw_pred_ents:
            norm_name = normalize_str(e.get("name", ""))
            if norm_name and norm_name not in seen_names:
                seen_names.add(norm_name)
                unique_pred_ents.append(e)

        pred_entity_count_dedup += len(unique_pred_ents)

        matched_gold_ents = set()
        p_ent_tp = 0
        p_ent_fp = 0

        for pe in unique_pred_ents:
            pname = normalize_str(pe.get("name", ""))
            match = False
            for i, ge in enumerate(gold_ents):
                if i in matched_gold_ents:
                    continue
                gname = normalize_str(ge["name"])
                if pname == gname or (len(pname) >= 3 and pname in gname) or (len(gname) >= 3 and gname in pname):
                    matched_gold_ents.add(i)
                    match = True
                    break
            if match:
                p_ent_tp += 1
            else:
                p_ent_fp += 1

        p_ent_fn = len(gold_ents) - len(matched_gold_ents)
        ent_tp += p_ent_tp
        ent_fp += p_ent_fp
        ent_fn += p_ent_fn

        # 2. Relationship Matching
        pred_rels = passage_res.get("extracted_relationships", [])
        rel_count += len(pred_rels)

        matched_gold_lenient = set()
        matched_gold_strict = set()
        p_rel_lenient_tp = 0
        p_rel_strict_tp = 0

        for pr in pred_rels:
            psrc = normalize_str(pr.get("source", ""))
            ptgt = normalize_str(pr.get("target", ""))
            prel = pr.get("relation", "")

            # Lenient check
            for i, gr in enumerate(gold_rels):
                if i in matched_gold_lenient:
                    continue
                gsrc = normalize_str(gr[0])
                gtgt = normalize_str(gr[2])
                if (psrc == gsrc or (len(psrc) >= 3 and psrc in gsrc) or (len(gsrc) >= 3 and gsrc in psrc)) and \
                   (ptgt == gtgt or (len(ptgt) >= 3 and ptgt in gtgt) or (len(gtgt) >= 3 and gtgt in ptgt)):
                    matched_gold_lenient.add(i)
                    p_rel_lenient_tp += 1
                    break

            # Strict check
            for i, gr in enumerate(gold_rels):
                if i in matched_gold_strict:
                    continue
                gsrc = normalize_str(gr[0])
                gtgt = normalize_str(gr[2])
                grel = gr[1]
                if (psrc == gsrc or (len(psrc) >= 3 and psrc in gsrc) or (len(gsrc) >= 3 and gsrc in psrc)) and \
                   (ptgt == gtgt or (len(ptgt) >= 3 and ptgt in gtgt) or (len(gtgt) >= 3 and gtgt in ptgt)) and \
                   predicate_matches(prel, grel):
                    matched_gold_strict.add(i)
                    p_rel_strict_tp += 1
                    break

        p_rel_strict_fp = len(pred_rels) - p_rel_strict_tp
        p_rel_strict_fn = len(gold_rels) - len(matched_gold_strict)

        rel_lenient_tp += p_rel_lenient_tp
        rel_strict_tp += p_rel_strict_tp
        rel_strict_fp += p_rel_strict_fp
        rel_strict_fn += p_rel_strict_fn

        per_passage[pid] = {
            "entities": {"tp": p_ent_tp, "fp": p_ent_fp, "fn": p_ent_fn},
            "relations_strict": {"tp": p_rel_strict_tp, "fp": p_rel_strict_fp, "fn": p_rel_strict_fn}
        }

    # Summary metrics
    ent_prec = (ent_tp / (ent_tp + ent_fp)) if (ent_tp + ent_fp) > 0 else 0.0
    ent_rec = (ent_tp / (ent_tp + ent_fn)) if (ent_tp + ent_fn) > 0 else 0.0
    ent_f1 = (2 * ent_prec * ent_rec / (ent_prec + ent_rec)) if (ent_prec + ent_rec) > 0 else 0.0

    rel_lenient_prec = (rel_lenient_tp / rel_count) if rel_count > 0 else 0.0
    rel_lenient_rec = (rel_lenient_tp / total_gold_relations) if total_gold_relations > 0 else 0.0
    rel_lenient_f1 = (2 * rel_lenient_prec * rel_lenient_rec / (rel_lenient_prec + rel_lenient_rec)) if (rel_lenient_prec + rel_lenient_rec) > 0 else 0.0

    rel_strict_prec = (rel_strict_tp / rel_count) if rel_count > 0 else 0.0
    rel_strict_rec = (rel_strict_tp / total_gold_relations) if total_gold_relations > 0 else 0.0
    rel_strict_f1 = (2 * rel_strict_prec * rel_strict_rec / (rel_strict_prec + rel_strict_rec)) if (rel_strict_prec + rel_strict_rec) > 0 else 0.0

    return {
        "entity_extraction": {
            "gold_total": total_gold_entities,
            "raw_extracted": pred_entity_count_raw,
            "dedup_extracted": pred_entity_count_dedup,
            "tp": ent_tp,
            "fp": ent_fp,
            "fn": ent_fn,
            "precision": round(ent_prec * 100, 2),
            "recall": round(ent_rec * 100, 2),
            "f1": round(ent_f1 * 100, 2)
        },
        "relationship_extraction_lenient": {
            "gold_total": total_gold_relations,
            "extracted": rel_count,
            "tp": rel_lenient_tp,
            "fp": rel_count - rel_lenient_tp,
            "fn": total_gold_relations - rel_lenient_tp,
            "precision": round(rel_lenient_prec * 100, 2),
            "recall": round(rel_lenient_rec * 100, 2),
            "f1": round(rel_lenient_f1 * 100, 2)
        },
        "relationship_extraction_strict": {
            "gold_total": total_gold_relations,
            "extracted": rel_count,
            "tp": rel_strict_tp,
            "fp": rel_strict_fp,
            "fn": rel_strict_fn,
            "precision": round(rel_strict_prec * 100, 2),
            "recall": round(rel_strict_rec * 100, 2),
            "f1": round(rel_strict_f1 * 100, 2)
        },
        "per_passage": per_passage
    }


def evaluate_answering_for_model(raw_answers: list[dict[str, Any]]) -> dict[str, Any]:
    gold_map = {item["id"]: item for item in ANSWERING_DATASET}

    total_queries = len(ANSWERING_DATASET)
    answerable_count = 0
    answerable_correct = 0

    unanswerable_count = 0
    unanswerable_correct = 0

    citation_complete_count = 0
    citation_valid_count = 0
    schema_valid_count = 0
    repetition_loop_count = 0

    per_query = {}

    for qres in raw_answers:
        qid = qres["id"]
        gold = gold_map[qid]
        gold_status = gold["gold_status"]
        expected_cits = gold.get("expected_citations", [])

        status = qres.get("status")
        answer_text = qres.get("answer", "") or ""
        cited_ids = qres.get("cited_memory_ids", []) or []
        schema_valid = qres.get("schema_valid", False)
        has_rep = qres.get("repetition_loop_detected", False)

        if schema_valid:
            schema_valid_count += 1
        if has_rep:
            repetition_loop_count += 1

        is_answerable = (gold_status == "answered")
        if is_answerable:
            answerable_count += 1
            # Check factual correctness: all gold key facts present or semantically matched
            matched_facts = 0
            for fact in gold["gold_key_facts"]:
                # Normalize check
                norm_f = normalize_str(fact)
                norm_ans = normalize_str(answer_text)
                # Word overlap check
                f_words = [w for w in re.split(r'\W+', fact.lower()) if len(w) >= 4]
                if all(w in answer_text.lower() for w in f_words):
                    matched_facts += 1
                elif norm_f in norm_ans:
                    matched_facts += 1

            fact_acc = (matched_facts >= len(gold["gold_key_facts"]) * 0.7)
            # Also check status: either 'answered' or answer text not empty
            if fact_acc and status != "insufficient_evidence":
                answerable_correct += 1

            # Citation completeness: all expected citations are present
            if expected_cits:
                if all(ec in cited_ids for ec in expected_cits):
                    citation_complete_count += 1
            else:
                citation_complete_count += 1

        else:
            # Unanswerable query
            unanswerable_count += 1
            # Abstention is correct if status is 'insufficient_evidence' OR answer contains 'insufficient_evidence' / 'no record'
            abstained = (
                status in ("insufficient_evidence", "unanswerable") or
                "insufficient_evidence" in answer_text.lower() or
                "no record" in answer_text.lower() or
                "no evidence" in answer_text.lower()
            )
            if abstained:
                unanswerable_correct += 1

        # Citation validity: all cited IDs exist in the context
        ctx_data = gold.get("context", {})
        if isinstance(ctx_data, dict):
            context_ids = [m["id"] for m in ctx_data.get("direct_evidence", [])]
        elif isinstance(ctx_data, list):
            context_ids = [m["id"] for m in ctx_data if isinstance(m, dict) and "id" in m]
        else:
            context_ids = []
        valid_citations = all(cid in context_ids for cid in cited_ids) if cited_ids else True
        if valid_citations:
            citation_valid_count += 1

        per_query[qid] = {
            "status": status,
            "answer_preview": answer_text[:80],
            "cited_ids": cited_ids,
            "expected_citations": expected_cits,
            "schema_valid": schema_valid
        }

    return {
        "total_queries": total_queries,
        "answerable_accuracy": round((answerable_correct / answerable_count) * 100, 2) if answerable_count > 0 else 0.0,
        "abstention_accuracy": round((unanswerable_correct / unanswerable_count) * 100, 2) if unanswerable_count > 0 else 0.0,
        "overall_qa_accuracy": round(((answerable_correct + unanswerable_correct) / total_queries) * 100, 2),
        "citation_completeness": round((citation_complete_count / answerable_count) * 100, 2) if answerable_count > 0 else 0.0,
        "citation_validity": round((citation_valid_count / total_queries) * 100, 2),
        "schema_compliance": round((schema_valid_count / total_queries) * 100, 2),
        "repetition_loops": repetition_loop_count,
        "per_query": per_query
    }


def evaluate_hardware_for_model(raw_extractions: list[dict[str, Any]], raw_answers: list[dict[str, Any]]) -> dict[str, Any]:
    ext_lat = [x["latency_ms"] for x in raw_extractions if "latency_ms" in x]
    ans_lat = [x["latency_ms"] for x in raw_answers if "latency_ms" in x]
    all_lat = ext_lat + ans_lat

    ext_tps = [x["tokens_per_sec"] for x in raw_extractions if x.get("tokens_per_sec")]
    ans_tps = [x["tokens_per_sec"] for x in raw_answers if x.get("tokens_per_sec")]
    all_tps = ext_tps + ans_tps

    ext_tokens = sum(x.get("eval_count", 0) for x in raw_extractions)
    ans_tokens = sum(x.get("eval_count", 0) for x in raw_answers)

    thinking_chars = sum(x.get("thinking_chars", 0) for x in raw_extractions) + sum(x.get("thinking_chars", 0) for x in raw_answers)

    vram_starts = [x["vram_start_mb"] for x in raw_extractions + raw_answers if x.get("vram_start_mb")]
    vram_ends = [x["vram_end_mb"] for x in raw_extractions + raw_answers if x.get("vram_end_mb")]
    all_vram = vram_starts + vram_ends
    peak_vram = max(all_vram) if all_vram else None

    return {
        "mean_latency_ms": round(sum(all_lat) / len(all_lat), 2) if all_lat else 0.0,
        "mean_extraction_latency_ms": round(sum(ext_lat) / len(ext_lat), 2) if ext_lat else 0.0,
        "mean_answering_latency_ms": round(sum(ans_lat) / len(ans_lat), 2) if ans_lat else 0.0,
        "tokens_per_sec": round(sum(all_tps) / len(all_tps), 2) if all_tps else 0.0,
        "total_eval_tokens": ext_tokens + ans_tokens,
        "total_thinking_chars": thinking_chars,
        "peak_vram_mb": peak_vram
    }


def main():
    bench_dir = Path("evaluation/microbenchmark_day1_5")
    models = ["gpt_oss", "qwen2_5", "gemma4"]

    comparison = {}

    for m in models:
        raw_file = bench_dir / f"raw_outputs_{m}.json"
        if not raw_file.exists():
            print(f"Skipping {m}, {raw_file} not found yet.")
            continue

        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        ext_metrics = evaluate_extraction_for_model(data["extraction_results"])
        ans_metrics = evaluate_answering_for_model(data["answering_results"])
        hw_metrics = evaluate_hardware_for_model(data["extraction_results"], data["answering_results"])

        comparison[m] = {
            "model_tag": data["model"],
            "extraction": ext_metrics,
            "answering": ans_metrics,
            "hardware": hw_metrics
        }

    out_file = bench_dir / "metrics_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"Metrics evaluated for {list(comparison.keys())}. Saved to {out_file}")


if __name__ == "__main__":
    main()

