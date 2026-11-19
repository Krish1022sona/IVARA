"""Comprehensive Evaluation Metrics for Benchmark Validation Sprint.

Calculates:
- Fact-level coverage (continuous score in [0.0, 1.0])
- Binary answer accuracy (1.0 if fact_coverage >= 0.70 or correct abstention, else 0.0)
- Exact Match rate (1.0 if fact_coverage >= 0.99, else 0.0)
- Abstention calibration:
    * correct_abstention (True Negative)
    * false_rejection (False Negative / Over-abstention)
    * false_acceptance (False Positive / Hallucination on unanswerable)
- Retrieval quality: Recall@1, Recall@3, Recall@5, MRR, Evidence Coverage
- Grounding: Citation validity and hallucinated citations count
"""

import re
from pathlib import Path
from typing import Any


def _get_conversation_session_map(db_path: Path | None = None) -> dict[str, str]:
    """Map conversation IDs or titles to session IDs using the evaluation SQLite DB."""
    conv_map = {}
    target_db = db_path or (Path(__file__).parent / "ivara_store_validation" / "conversations.db")
    if target_db.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True)
            for row in conn.execute("SELECT id, title FROM conversations"):
                c_id, title = row[0], row[1]
                m = re.search(r"session_\d{2}", title)
                if m:
                    conv_map[c_id] = m.group(0)
            conn.close()
        except Exception:
            pass
    return conv_map


def evaluate_retrieval(
    retrieved_items: list[dict[str, Any]],
    supporting_sessions: list[str],
    is_answerable: bool,
    session_map: dict[str, str] | None = None
) -> dict[str, float]:
    """
    Evaluate retrieval metrics against ground-truth supporting sessions.
    """
    if not is_answerable or not supporting_sessions:
        return {
            "recall_at_1": 1.0,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "mrr": 1.0,
            "evidence_coverage": 1.0
        }

    conv_map = session_map if session_map is not None else _get_conversation_session_map()
    target_sessions = set(supporting_sessions)
    found_sessions = set()
    first_rank = None

    for rank, item in enumerate(retrieved_items[:5], 1):
        s_id = item.get("session_id")
        src = str(item.get("source", ""))
        text = str(item.get("text", ""))

        if not s_id and src:
            m = re.search(r"session_\d{2}", src)
            if m:
                s_id = m.group(0)
            else:
                c_match = re.search(r"conv_[a-f0-9]+", src)
                if c_match and c_match.group(0) in conv_map:
                    s_id = conv_map[c_match.group(0)]

        if s_id in target_sessions:
            found_sessions.add(s_id)
            if first_rank is None:
                first_rank = rank
        else:
            for ts in target_sessions:
                if ts in src or ts in text:
                    found_sessions.add(ts)
                    if first_rank is None:
                        first_rank = rank

    r1 = 1.0 if first_rank == 1 else 0.0
    r3 = 1.0 if first_rank is not None and first_rank <= 3 else 0.0
    r5 = 1.0 if first_rank is not None and first_rank <= 5 else 0.0
    mrr = (1.0 / first_rank) if first_rank is not None else 0.0
    coverage = len(found_sessions) / len(target_sessions) if target_sessions else 1.0

    return {
        "recall_at_1": r1,
        "recall_at_3": r3,
        "recall_at_5": r5,
        "mrr": round(mrr, 4),
        "evidence_coverage": round(coverage, 4)
    }


def evaluate_answer(
    prediction: dict[str, Any],
    question_spec: dict[str, Any]
) -> dict[str, Any]:
    """
    Evaluate predicted answer against expected facts, exact match, and abstention criteria.
    """
    is_answerable = question_spec.get("is_answerable", True)
    key_facts = question_spec.get("key_facts", [])
    pred_status = prediction.get("status", "answered")
    pred_answer = prediction.get("answer", "").strip()
    if pred_answer.startswith("{"):
        m = re.search(r'"answer":\s*"([^"\\]*(?:\\.[^"\\]*)*)"', pred_answer)
        if m:
            pred_answer = m.group(1)
    ans_lower = pred_answer.lower()

    # Abstention detection heuristics
    abstention_indicators = [
        "insufficient", "don't have sufficient", "do not have sufficient",
        "not mentioned", "not recorded", "unknown", "no information",
        "cannot answer", "not provided", "no record", "does not contain"
    ]
    pred_abstained = (
        pred_status in ("insufficient_evidence", "unanswerable") or
        any(ind in ans_lower for ind in abstention_indicators)
    )

    if not is_answerable:
        correct_abstention = pred_abstained
        return {
            "is_answerable": False,
            "abstained": pred_abstained,
            "correct_abstention": correct_abstention,
            "false_rejection": False,
            "false_acceptance": not pred_abstained,  # Hallucination on unanswerable
            "key_facts_found": [],
            "fact_coverage": 1.0 if correct_abstention else 0.0,
            "binary_accuracy": 1.0 if correct_abstention else 0.0,
            "exact_match": 1.0 if correct_abstention else 0.0
        }

    # Answerable question
    if pred_abstained:
        return {
            "is_answerable": True,
            "abstained": True,
            "correct_abstention": False,
            "false_rejection": True,  # Over-abstention
            "false_acceptance": False,
            "key_facts_found": [],
            "fact_coverage": 0.0,
            "binary_accuracy": 0.0,
            "exact_match": 0.0
        }

    found_facts = []
    ans_clean = " " + " ".join(re.sub(r"[^\w\s]", " ", ans_lower).split()) + " "
    for kf in key_facts:
        kf_clean = " ".join(re.sub(r"[^\w\s]", " ", kf.lower()).split())
        if f" {kf_clean} " in ans_clean or kf_clean in ans_clean:
            found_facts.append(kf)

    fact_cov = len(found_facts) / len(key_facts) if key_facts else 1.0
    exact_m = 1.0 if fact_cov >= 0.99 else 0.0
    bin_acc = 1.0 if fact_cov >= 0.70 else 0.0

    return {
        "is_answerable": True,
        "abstained": False,
        "correct_abstention": True,
        "false_rejection": False,
        "false_acceptance": False,
        "key_facts_found": found_facts,
        "fact_coverage": round(fact_cov, 4),
        "binary_accuracy": bin_acc,
        "exact_match": exact_m
    }


def evaluate_citations(
    cited_ids: list[str],
    retrieved_items: list[dict[str, Any]]
) -> dict[str, float]:
    """Check whether cited IDs actually appear in the retrieved context."""
    if not cited_ids:
        return {"citation_validity": 1.0, "hallucinated_citations": 0}

    valid_ids = set()
    for item in retrieved_items:
        for k in ("id", "memory_id", "chunk_id", "turn_id", "source_turn_id"):
            if k in item and item[k]:
                valid_ids.add(str(item[k]))

    valid_count = sum(1 for cid in cited_ids if cid in valid_ids)
    hallucinated = len(cited_ids) - valid_count
    validity = valid_count / len(cited_ids)

    return {
        "citation_validity": round(validity, 4),
        "hallucinated_citations": hallucinated
    }
