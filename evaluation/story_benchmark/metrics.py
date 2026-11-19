"""Evaluation Metrics for Story Benchmark.

Calculates:
- Exact match / Key fact coverage
- Answer accuracy
- Abstention correctness (rejecting unanswerable, answering answerable)
- Retrieval Recall@1, Recall@3, Recall@5, MRR, Evidence Coverage
- Citation validity and unsupported claim detection
- Latency statistics
"""

from pathlib import Path
import re
from typing import Any


def _get_conversation_session_map() -> dict[str, str]:
    """Build mapping from conversation IDs to session IDs using evaluation SQLite database."""
    conv_map = {}
    db_path = Path(__file__).parent / "ivara_store" / "conversations.db"
    if db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
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
    Evaluate retrieval quality against ground truth supporting sessions.
    Each retrieved item must have 'session_id' or 'source' containing the session tag.
    """
    if not is_answerable or not supporting_sessions:
        # For unanswerable questions, retrieval coverage is not applicable
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
        # Extract session ID from item metadata or source
        s_id = item.get("session_id")
        src = str(item.get("source", ""))

        if not s_id and src:
            # Check for direct session pattern
            m = re.search(r"session_\d{2}", src)
            if m:
                s_id = m.group(0)
            else:
                # Check for conversation ID in source (e.g. conv_ef30f29ef2e3_msg_...)
                c_match = re.search(r"conv_[a-f0-9]+", src)
                if c_match and c_match.group(0) in conv_map:
                    s_id = conv_map[c_match.group(0)]

        # Check if item text or id mentions the session
        if s_id in target_sessions:
            found_sessions.add(s_id)
            if first_rank is None:
                first_rank = rank
        else:
            # Check if any target session matches text or source
            for ts in target_sessions:
                if ts in src or ts in str(item.get("text", "")):
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
    Evaluate predicted answer against expected facts and abstention criteria.
    """
    is_answerable = question_spec.get("is_answerable", True)
    key_facts = question_spec.get("key_facts", [])
    pred_status = prediction.get("status", "answered")
    pred_answer = prediction.get("answer", "").strip()
    ans_lower = pred_answer.lower()

    # Abstention detection
    abstention_indicators = [
        "insufficient", "don't have sufficient", "do not have sufficient",
        "not mentioned", "not recorded", "unknown", "no information",
        "cannot answer", "not provided"
    ]
    pred_abstained = (
        pred_status == "insufficient_evidence" or
        any(ind in ans_lower for ind in abstention_indicators)
    )

    if not is_answerable:
        correct_abstention = pred_abstained
        accuracy = 1.0 if correct_abstention else 0.0
        exact_match = 1.0 if correct_abstention else 0.0
        return {
            "is_answerable": False,
            "abstained": pred_abstained,
            "correct_abstention": correct_abstention,
            "key_facts_found": [],
            "key_facts_coverage": 1.0 if correct_abstention else 0.0,
            "accuracy": accuracy,
            "exact_match": exact_match
        }

    # Answerable question
    # If the system falsely abstained on an answerable query:
    if pred_abstained:
        return {
            "is_answerable": True,
            "abstained": True,
            "correct_abstention": False,
            "key_facts_found": [],
            "key_facts_coverage": 0.0,
            "accuracy": 0.0,
            "exact_match": 0.0
        }

    # Check key facts coverage
    found_facts = []
    for kf in key_facts:
        kf_clean = re.sub(r"[^\w\s]", "", kf.lower()).strip()
        # Word boundary or substring search
        if re.search(r"\b" + re.escape(kf_clean) + r"\b", ans_lower) or kf_clean in ans_lower:
            found_facts.append(kf)

    fact_coverage = len(found_facts) / len(key_facts) if key_facts else 1.0
    accuracy = round(fact_coverage, 4)
    exact_match = 1.0 if fact_coverage >= 0.99 else 0.0

    return {
        "is_answerable": True,
        "abstained": False,
        "correct_abstention": True,
        "key_facts_found": found_facts,
        "key_facts_coverage": round(fact_coverage, 4),
        "accuracy": accuracy,
        "exact_match": exact_match
    }


def evaluate_citations(
    cited_ids: list[str],
    retrieved_items: list[dict[str, Any]]
) -> dict[str, float]:
    """Check whether cited memory or chunk IDs actually exist in the retrieved context."""
    if not cited_ids:
        return {"citation_validity": 1.0, "hallucinated_citations": 0}

    valid_ids = set()
    for item in retrieved_items:
        if "id" in item:
            valid_ids.add(str(item["id"]))
        if "memory_id" in item:
            valid_ids.add(str(item["memory_id"]))
        if "chunk_id" in item:
            valid_ids.add(str(item["chunk_id"]))

    valid_count = sum(1 for cid in cited_ids if cid in valid_ids)
    hallucinated = len(cited_ids) - valid_count
    validity = valid_count / len(cited_ids)

    return {
        "citation_validity": round(validity, 4),
        "hallucinated_citations": hallucinated
    }

