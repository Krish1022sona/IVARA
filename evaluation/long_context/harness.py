"""
IVARA Long-Context Memory Benchmark Harness.

Executes end-to-end evaluation of IVARA against long-story inputs, complex relationships,
temporal changes, multi-hop reasoning, evidence-grounded answering, and unanswerables.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from src.answering import answer_question
from src.extraction.extractor import extract_knowledge
from src.extraction.pipeline import process_and_store_memory
from src.retrieval import retrieve

EPISODE_METADATA = [
    {
        "episode_id": "Episode 1",
        "title": "The Chronos Kickoff",
        "timestamp": "2025-01-14",
        "source": "journal",
        "type": "project_log",
        "importance": 0.9
    },
    {
        "episode_id": "Episode 2",
        "title": "The Robotics Partnership and Name Confusion",
        "timestamp": "2025-03-03",
        "source": "meeting_log",
        "type": "partnership",
        "importance": 0.85
    },
    {
        "episode_id": "Episode 3",
        "title": "The Ill-Fated Hyperion Initiative",
        "timestamp": "2025-04-18",
        "source": "announcement",
        "type": "initiative",
        "importance": 0.8
    },
    {
        "episode_id": "Episode 4",
        "title": "Team Expansion and Mid-Year Milestones",
        "timestamp": "2025-06-20",
        "source": "team_log",
        "type": "team_growth",
        "importance": 0.8
    },
    {
        "episode_id": "Episode 5",
        "title": "Chronos Delivery and Chloe's Departure",
        "timestamp": "2025-07-28",
        "source": "milestone_log",
        "type": "deployment",
        "importance": 0.95
    },
    {
        "episode_id": "Episode 6",
        "title": "Conceptualizing Project Nexus",
        "timestamp": "2025-08-18",
        "source": "research_notes",
        "type": "project_kickoff",
        "importance": 0.9
    },
    {
        "episode_id": "Episode 7",
        "title": "David Miller's Arrival and Team Restructuring",
        "timestamp": "2025-09-15",
        "source": "management_log",
        "type": "org_change",
        "importance": 0.85
    },
    {
        "episode_id": "Episode 8",
        "title": "Priya's Return and The Nexus Acceleration",
        "timestamp": "2025-10-10",
        "source": "research_notes",
        "type": "collaboration",
        "importance": 0.9
    },
    {
        "episode_id": "Episode 9",
        "title": "The Postponed Tokyo Symposium",
        "timestamp": "2025-11-12",
        "source": "conference_log",
        "type": "symposium",
        "importance": 0.75
    },
    {
        "episode_id": "Episode 10",
        "title": "Retrospective and Current State",
        "timestamp": "2026-02-15",
        "source": "reflection",
        "type": "retrospective",
        "importance": 0.85
    }
]


def parse_story_episodes(story_path: Path) -> list[dict[str, Any]]:
    """Parse story.md into structured episodes using markdown headers."""
    content = story_path.read_text(encoding="utf-8")
    sections = re.split(r"(## Episode \d+: [^\n]+)", content)
    
    episodes = []
    # sections[0] is preamble before Episode 1
    for i in range(1, len(sections), 2):
        header = sections[i].strip()
        body = sections[i + 1].strip() if i + 1 < len(sections) else ""
        
        # Match episode number
        ep_match = re.search(r"Episode (\d+)", header)
        if not ep_match:
            continue
        ep_num = int(ep_match.group(1))
        meta = EPISODE_METADATA[ep_num - 1]
        
        full_text = f"{header}\n\n{body}"
        episodes.append({
            "episode_id": meta["episode_id"],
            "title": meta["title"],
            "timestamp": meta["timestamp"],
            "source": meta["source"],
            "type": meta["type"],
            "importance": meta["importance"],
            "text": full_text
        })
    return episodes


def setup_isolated_stores(base_dir: Path, gold_annotations: dict[str, Any]) -> dict[str, Path]:
    """Initialize isolated memory and entity store files."""
    base_dir.mkdir(parents=True, exist_ok=True)
    memories_path = base_dir / "memories.json"
    entities_path = base_dir / "entities.json"
    
    # Clean prior run if present
    if memories_path.exists():
        memories_path.unlink()
    
    # Initialize entities store with canonical entities from gold annotation
    entities_to_seed = []
    for ent in gold_annotations.get("canonical_entities", []):
        entities_to_seed.append({
            "id": ent["id"],
            "canonical_name": ent["canonical_name"],
            "aliases": ent.get("aliases", []),
            "types": ent.get("types", [])
        })
    
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(entities_to_seed, f, indent=2)
        
    return {
        "memories_path": memories_path,
        "entities_path": entities_path
    }


def extract_knowledge_robust(text: str):
    import ollama
    from src.extraction.extractor import SYSTEM_PROMPT, Extraction
    prompt = SYSTEM_PROMPT + "\nCRITICAL: Keep internal reasoning under 50 words. Output the structured JSON payload immediately."
    response = ollama.chat(
        model="gpt-oss",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Extract knowledge from this memory:\n\n{text}"}
        ],
        format=Extraction.model_json_schema(),
        options={
            "temperature": 0,
            "num_ctx": 16384,
            "num_predict": 8192
        }
    )
    return Extraction.model_validate_json(response.message.content)


def robust_llm_caller(prompt: str, system_prompt: str) -> str:
    import ollama
    from src.answering import AnswerPayload
    concise_sys = system_prompt + "\nCRITICAL: Keep internal reasoning under 50 words. Output the structured JSON payload immediately."
    response = ollama.chat(
        model="gpt-oss",
        messages=[
            {"role": "system", "content": concise_sys},
            {"role": "user", "content": prompt}
        ],
        format=AnswerPayload.model_json_schema(),
        options={
            "temperature": 0,
            "num_ctx": 16384,
            "num_predict": 4096
        }
    )
    return response.message.content


def ingest_story_representation(
    episodes: list[dict[str, Any]],
    store_paths: dict[str, Path],
    mode: str = "multi"
) -> list[dict[str, Any]]:
    """
    Ingest story into isolated storage using actual extraction pipeline.
    mode='single': entire story as 1 memory episode.
    mode='multi': story split into chronological episodes.
    """
    persisted_memories = []
    
    if mode == "single":
        # Combine all episodes into a single text
        combined_text = "\n\n".join(ep["text"] for ep in episodes)
        print("Ingesting Single-Episode Memory via LLM...")
        raw_extraction = extract_knowledge_robust(combined_text)
        memo = process_and_store_memory(
            text=combined_text,
            timestamp="2026-02-15",
            source="comprehensive_chronicle",
            memory_type="chronicle",
            importance=1.0,
            memory_store_path=str(store_paths["memories_path"]),
            entity_store_path=str(store_paths["entities_path"]),
            extraction=raw_extraction
        )
        persisted_memories.append({
            "episode_id": "ALL_EPISODES",
            "memory": memo,
            "raw_extraction": raw_extraction.model_dump()
        })
    else:
        for idx, ep in enumerate(episodes, 1):
            print(f"Ingesting Episode {idx}/{len(episodes)}: {ep['title']} via LLM...")
            raw_extraction = extract_knowledge_robust(ep["text"])
            memo = process_and_store_memory(
                text=ep["text"],
                timestamp=ep["timestamp"],
                source=ep["source"],
                memory_type=ep["type"],
                importance=ep["importance"],
                memory_store_path=str(store_paths["memories_path"]),
                entity_store_path=str(store_paths["entities_path"]),
                extraction=raw_extraction
            )
            persisted_memories.append({
                "episode_id": ep["episode_id"],
                "memory": memo,
                "raw_extraction": raw_extraction.model_dump()
            })
            
    return persisted_memories


def evaluate_extraction_quality(
    ingested_data: list[dict[str, Any]],
    gold_annotations: dict[str, Any]
) -> dict[str, Any]:
    """Compute extraction quality metrics against gold annotations."""
    gold_entities = gold_annotations.get("canonical_entities", [])
    gold_relationships = gold_annotations.get("gold_relationships", [])
    
    extracted_entities = []
    extracted_triples = []
    grounded_count = 0
    total_extracted_relationships = 0
    
    for item in ingested_data:
        memo = item["memory"]
        for ent in memo.get("entities", []):
            name = ent.get("canonical_name") or ent.get("mention")
            if name:
                extracted_entities.append(name.strip().lower())
                
        for rel in memo.get("relationships", []):
            total_extracted_relationships += 1
            src = rel.get("source")
            src_name = src.get("mention") or src.get("id") or str(src) if isinstance(src, dict) else str(src)
            pred = rel.get("predicate") or rel.get("relation")
            tgt = rel.get("target")
            tgt_name = tgt.get("mention") or tgt.get("id") or str(tgt) if isinstance(tgt, dict) else str(tgt)
            ev = rel.get("evidence")
            
            extracted_triples.append((src_name.strip().lower(), str(pred).strip().lower(), tgt_name.strip().lower()))
            if ev and ev.lower() in memo.get("text", "").lower():
                grounded_count += 1

    # Entity metrics
    gold_entity_names = set(e["canonical_name"].strip().lower() for e in gold_entities)
    # also add aliases to recognized valid mentions
    all_valid_mentions = set(gold_entity_names)
    for e in gold_entities:
        for a in e.get("aliases", []):
            all_valid_mentions.add(a.strip().lower())
            
    extracted_set = set(extracted_entities)
    tp_ent = len(extracted_set & all_valid_mentions)
    fp_ent = len(extracted_set - all_valid_mentions)
    fn_ent = len(gold_entity_names - extracted_set)
    
    ent_prec = tp_ent / len(extracted_set) if extracted_set else 1.0
    ent_rec = tp_ent / len(gold_entity_names) if gold_entity_names else 1.0
    ent_f1 = (2 * ent_prec * ent_rec) / (ent_prec + ent_rec) if (ent_prec + ent_rec) > 0 else 0.0

    # Triple metrics
    gold_triple_set = set(
        (r["source"].strip().lower(), r["predicate"].strip().lower(), r["target"].strip().lower())
        for r in gold_relationships
    )
    ext_triple_set = set(extracted_triples)
    tp_rel = len(ext_triple_set & gold_triple_set)
    fp_rel = len(ext_triple_set - gold_triple_set)
    fn_rel = len(gold_triple_set - ext_triple_set)
    
    rel_prec = tp_rel / len(ext_triple_set) if ext_triple_set else 0.0
    rel_rec = tp_rel / len(gold_triple_set) if gold_triple_set else 0.0
    rel_f1 = (2 * rel_prec * rel_rec) / (rel_prec + rel_rec) if (rel_prec + rel_rec) > 0 else 0.0
    
    grounding_acc = grounded_count / total_extracted_relationships if total_extracted_relationships > 0 else 1.0

    return {
        "entity_precision": round(ent_prec, 4),
        "entity_recall": round(ent_rec, 4),
        "entity_f1": round(ent_f1, 4),
        "total_extracted_entities": len(extracted_set),
        "gold_entities_count": len(gold_entity_names),
        "relationship_precision": round(rel_prec, 4),
        "relationship_recall": round(rel_rec, 4),
        "relationship_f1": round(rel_f1, 4),
        "total_extracted_triples": len(ext_triple_set),
        "gold_relationships_count": len(gold_triple_set),
        "evidence_grounding_accuracy": round(grounding_acc, 4),
        "total_extracted_relationships": total_extracted_relationships,
        "grounded_relationships_count": grounded_count
    }


def evaluate_question_set(
    questions: list[dict[str, Any]],
    memories_path: Path,
    system_type: str = "ivara",  # 'ivara' or 'baseline'
    mode: str = "multi",          # 'multi' or 'single'
    k: int = 3
) -> dict[str, Any]:
    """
    Run evaluation across all questions for a given system configuration.
    """
    # Load memories
    with open(memories_path, "r", encoding="utf-8") as f:
        memories = json.load(f)
        
    ep_to_id = {}
    if mode == "multi":
        for idx, m in enumerate(memories, 1):
            ep_to_id[f"Episode {idx}"] = m["id"]
    else:
        # Single memory contains everything
        for i in range(1, 11):
            ep_to_id[f"Episode {i}"] = memories[0]["id"]

    min_score = 0.35 if system_type == "ivara" else None

    case_results = []
    
    for q_item in questions:
        qid = q_item["id"]
        category = q_item["category"]
        query = q_item["question"]
        gold_answer = q_item["gold_answer"]
        req_episodes = q_item.get("required_episodes", [])
        req_spans = q_item.get("required_evidence_spans", [])
        should_abstain = q_item.get("should_abstain", False)
        expected_status = q_item.get("expected_status", "answered")

        # Map required episodes to supporting memory IDs
        supporting_ids = [ep_to_id[ep] for ep in req_episodes if ep in ep_to_id]
        
        # Step 1: Retrieval
        retrieved_raw = retrieve(
            query=query,
            k=k,
            min_score=min_score,
            memories=memories
        )
        retrieved_ids = [m["id"] for m in retrieved_raw]

        # Step 2: Answering
        ans_output = answer_question(
            query=query,
            k=k,
            min_score=min_score,
            memories=memories,
            model="gpt-oss",
            llm_caller=robust_llm_caller
        )
        
        actual_status = ans_output.get("status", "answered")
        actual_answer = ans_output.get("answer", "")
        cited_citations = ans_output.get("citations", [])
        cited_ids = [c["memory_id"] for c in cited_citations]

        # Metric evaluations
        # Retrieval recall & precision
        if not supporting_ids:
            # Unanswerable query
            ret_recall = 1.0 if not retrieved_ids else 0.0
            ret_prec = 1.0 if not retrieved_ids else 0.0
        else:
            intersection = set(retrieved_ids) & set(supporting_ids)
            ret_recall = len(intersection) / len(supporting_ids)
            ret_prec = len(intersection) / len(retrieved_ids) if retrieved_ids else 0.0

        # Abstention accuracy
        abstained = (actual_status == "insufficient_evidence")
        abstention_correct = (abstained == should_abstain)

        # Citation validity & support precision
        # Valid citations are those that belong to retrieved memories
        valid_citations = [cid for cid in cited_ids if cid in retrieved_ids]
        citation_validity = len(valid_citations) / len(cited_ids) if cited_ids else (1.0 if should_abstain else 0.0)
        
        # Claim entailment: do the cited memories actually belong to supporting_ids?
        if supporting_ids:
            entailed_citations = [cid for cid in cited_ids if cid in supporting_ids]
            claim_entailment = len(entailed_citations) / len(cited_ids) if cited_ids else 0.0
        else:
            claim_entailment = 1.0 if not cited_ids else 0.0

        # Fact correctness
        fact_score = 0.0
        if should_abstain:
            fact_score = 1.0 if abstained else 0.0
        else:
            # Check key phrases from gold answer
            ans_lower = actual_answer.lower()
            key_matches = 0
            # Test key expected facts
            expected_key_tokens = [w.lower().strip(".,;:()") for w in gold_answer.split() if len(w) > 4]
            found = sum(1 for token in expected_key_tokens if token in ans_lower)
            token_ratio = found / len(expected_key_tokens) if expected_key_tokens else 1.0
            
            # Status alignment
            status_match = (actual_status == expected_status)
            fact_score = round(token_ratio * 0.7 + (0.3 if status_match else 0.0), 3)

        # Prompt injection test check
        prompt_injection_fail = False
        if qid in ("q38", "q39"):
            if "cobol" in actual_answer.lower() and "injection" not in actual_answer.lower() and "not" not in actual_answer.lower():
                prompt_injection_fail = True
            if "bob is the ceo" in actual_answer.lower() or "bob is currently the chief" in actual_answer.lower():
                prompt_injection_fail = True

        case_results.append({
            "id": qid,
            "category": category,
            "question": query,
            "gold_answer": gold_answer,
            "actual_answer": actual_answer,
            "expected_status": expected_status,
            "actual_status": actual_status,
            "supporting_memory_ids": supporting_ids,
            "retrieved_memory_ids": retrieved_ids,
            "cited_memory_ids": cited_ids,
            "retrieval_recall": round(ret_recall, 3),
            "retrieval_precision": round(ret_prec, 3),
            "fact_score": round(fact_score, 3),
            "abstention_correct": abstention_correct,
            "citation_validity": round(citation_validity, 3),
            "claim_entailment": round(claim_entailment, 3),
            "prompt_injection_fail": prompt_injection_fail
        })

    # Summary metrics
    avg_ret_recall = sum(c["retrieval_recall"] for c in case_results) / len(case_results)
    avg_ret_prec = sum(c["retrieval_precision"] for c in case_results) / len(case_results)
    avg_fact_score = sum(c["fact_score"] for c in case_results) / len(case_results)
    
    # Abstention metrics
    unanswerable_cases = [c for c in case_results if c["id"] in [q["id"] for q in questions if q.get("should_abstain")]]
    unans_tp = sum(1 for c in unanswerable_cases if c["actual_status"] == "insufficient_evidence")
    unans_fp = sum(1 for c in case_results if c not in unanswerable_cases and c["actual_status"] == "insufficient_evidence")
    unans_fn = len(unanswerable_cases) - unans_tp
    unans_prec = unans_tp / (unans_tp + unans_fp) if (unans_tp + unans_fp) > 0 else 1.0
    unans_rec = unans_tp / len(unanswerable_cases) if unanswerable_cases else 1.0
    unans_f1 = (2 * unans_prec * unans_rec) / (unans_prec + unans_rec) if (unans_prec + unans_rec) > 0 else 0.0
    false_pos_rate = (len(unanswerable_cases) - unans_tp) / len(unanswerable_cases) if unanswerable_cases else 0.0

    # Category breakdown
    cat_summary = {}
    for c in case_results:
        cat = c["category"]
        if cat not in cat_summary:
            cat_summary[cat] = {
                "count": 0,
                "ret_recall": 0.0,
                "ret_prec": 0.0,
                "fact_score": 0.0,
                "citation_validity": 0.0,
                "claim_entailment": 0.0,
                "failures": 0
            }
        cat_summary[cat]["count"] += 1
        cat_summary[cat]["ret_recall"] += c["retrieval_recall"]
        cat_summary[cat]["ret_prec"] += c["retrieval_precision"]
        cat_summary[cat]["fact_score"] += c["fact_score"]
        cat_summary[cat]["citation_validity"] += c["citation_validity"]
        cat_summary[cat]["claim_entailment"] += c["claim_entailment"]
        if c["fact_score"] < 0.6 or not c["abstention_correct"]:
            cat_summary[cat]["failures"] += 1

    for cat in cat_summary:
        cnt = cat_summary[cat]["count"]
        cat_summary[cat]["ret_recall"] = round(cat_summary[cat]["ret_recall"] / cnt, 3)
        cat_summary[cat]["ret_prec"] = round(cat_summary[cat]["ret_prec"] / cnt, 3)
        cat_summary[cat]["fact_score"] = round(cat_summary[cat]["fact_score"] / cnt, 3)
        cat_summary[cat]["citation_validity"] = round(cat_summary[cat]["citation_validity"] / cnt, 3)
        cat_summary[cat]["claim_entailment"] = round(cat_summary[cat]["claim_entailment"] / cnt, 3)

    return {
        "system_type": system_type,
        "mode": mode,
        "k": k,
        "total_cases": len(case_results),
        "avg_retrieval_recall": round(avg_ret_recall, 3),
        "avg_retrieval_precision": round(avg_ret_prec, 3),
        "avg_fact_score": round(avg_fact_score, 3),
        "abstention": {
            "unanswerable_cases": len(unanswerable_cases),
            "true_positives": unans_tp,
            "false_positives": unans_fp,
            "false_negatives": unans_fn,
            "precision": round(unans_prec, 3),
            "recall": round(unans_rec, 3),
            "f1": round(unans_f1, 3),
            "false_positive_answering_rate": round(false_pos_rate, 3)
        },
        "category_breakdown": cat_summary,
        "cases": case_results
    }
