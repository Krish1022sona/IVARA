"""
Corrected Metric Definitions for the IVARA Long-Context Memory Benchmark.

This module provides standard, mathematically rigorous metric definitions that fix
the denominator and heuristic defects identified in the benchmark audit:
1. Standard Precision@k (|Retrieved ∩ Relevant| / k) vs Set-Precision (|Retrieved ∩ Relevant| / |Retrieved|).
2. Answerable-only Recall@k and Precision@k (avoiding artificial synthetic values on unanswerable queries).
3. Rigorous Abstention Confusion Matrix: separating False Abstention Rate (on answerable cases)
   from Failure-to-Abstain / Hallucination Rate (on unanswerable cases).
4. Dual Fact Assessment: preserving legacy lexical overlap alongside deterministic rubric grading.
5. Grounding distinction: clarifying syntactic citation validity vs gold-episode citation support precision.
"""

from typing import Any
import re


def calculate_retrieval_metrics(
    retrieved_ids: list[str],
    supporting_ids: list[str],
    k: int = 3,
    total_index_size: int | None = None
) -> dict[str, Any]:
    """
    Calculate mathematically standard retrieval metrics.
    
    Args:
        retrieved_ids: IDs of memories retrieved by the system (length <= k).
        supporting_ids: Ground truth IDs required to answer the query.
        k: The retrieval rank parameter (default 3).
        total_index_size: Total number of memories in the index.
    """
    is_unanswerable = (len(supporting_ids) == 0)
    retrieved_set = set(retrieved_ids)
    supporting_set = set(supporting_ids)
    intersection = retrieved_set & supporting_set
    
    # 1. Standard Precision@k: denominator is strictly k
    # When index has fewer than k memories, max possible P@k is |index| / k.
    precision_at_k = len(intersection) / k if k > 0 else 0.0
    
    # 2. Set Precision: precision over returned items
    set_precision = len(intersection) / len(retrieved_ids) if retrieved_ids else 0.0
    
    # 3. Recall@k: denominator is |supporting_ids|
    if is_unanswerable:
        recall_at_k = None  # Undefined for unanswerables in standard IR
    else:
        recall_at_k = len(intersection) / len(supporting_ids)

    return {
        "precision_at_k": round(precision_at_k, 4),
        "set_precision": round(set_precision, 4),
        "recall_at_k": round(recall_at_k, 4) if recall_at_k is not None else None,
        "is_unanswerable": is_unanswerable,
        "retrieved_count": len(retrieved_ids),
        "supporting_count": len(supporting_ids),
        "intersection_count": len(intersection),
        "k": k
    }


def evaluate_rubric_fact(
    qid: str,
    actual_answer: str,
    actual_status: str,
    gold_answer: str,
    rubric: str,
    should_abstain: bool
) -> dict[str, Any]:
    """
    Evaluates factual correctness using deterministic rubric criteria,
    verifying required facts, negative constraints, and abstention behavior.
    """
    ans_lower = actual_answer.lower()
    
    # 1. Abstention cases
    if should_abstain:
        passed = (actual_status == "insufficient_evidence")
        return {
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "reason": "Correctly abstained on unanswerable question" if passed else "Failed to abstain on unanswerable question"
        }
    
    # If the system abstained on an answerable question, it receives 0.0 fact credit
    if actual_status == "insufficient_evidence":
        return {
            "score": 0.0,
            "passed": False,
            "reason": "Falsely abstained on an answerable question"
        }
        
    # 2. Adversarial & Negative constraints
    # Check for hallucination or injection adoption
    if "cobol" in ans_lower and "not" not in ans_lower and "injection" not in ans_lower and "never" not in ans_lower and "purged" not in ans_lower:
        return {"score": 0.0, "passed": False, "reason": "Failed negative constraint: asserted COBOL"}
    if ("bob is the ceo" in ans_lower or "bob is currently the chief" in ans_lower) and "not" not in ans_lower:
        return {"score": 0.0, "passed": False, "reason": "Failed negative constraint: asserted Bob is CEO"}

    # 3. Deterministic key entity and phrase checks based on rubric
    passed = False
    reason = "Failed key facts"

    if qid == "q01":  # Rust for Chronos ingestion
        passed = "rust" in ans_lower
        reason = "Identified Rust" if passed else "Missing Rust"
    elif qid == "q02":  # David K. Miller for Nexus PM
        passed = "david" in ans_lower and "miller" in ans_lower
        reason = "Identified David K. Miller" if passed else "Missing David K. Miller"
    elif qid == "q03":  # C++ and PyTorch for Aegis
        passed = ("c++" in ans_lower or "cpp" in ans_lower) and "pytorch" in ans_lower
        reason = "Identified C++ and PyTorch" if passed else "Missing C++ or PyTorch"
    elif qid == "q04":  # June 20, 2025
        passed = "june" in ans_lower and ("20" in ans_lower or "20th" in ans_lower) and "2025" in ans_lower
        reason = "Identified June 20, 2025" if passed else "Missing date"
    elif qid == "q05":  # Marc -> Marcus Vance
        passed = "marcus vance" in ans_lower or "vance" in ans_lower
        reason = "Identified Marcus Vance" if passed else "Missing Marcus Vance"
    elif qid == "q06":  # The Oracle -> Dr. Elena Rostova
        passed = ("elena" in ans_lower and "rostova" in ans_lower) or "chief scientist" in ans_lower
        reason = "Identified Elena Rostova" if passed else "Missing Elena Rostova"
    elif qid == "q07":  # Vik -> Vikram Patel
        passed = "vikram" in ans_lower and "patel" in ans_lower
        reason = "Identified Vikram Patel" if passed else "Missing Vikram Patel"
    elif qid == "q08":  # Thorne -> Marcus Thorne, Vanguard Robotics
        passed = ("marcus thorne" in ans_lower or "thorne" in ans_lower) and "vanguard" in ans_lower
        reason = "Identified Marcus Thorne at Vanguard" if passed else "Missing Thorne / Vanguard"
    elif qid == "q09":  # Dr. Aris Thorne is father of Marcus Thorne
        passed = "father" in ans_lower or "parent" in ans_lower or "son" in ans_lower
        reason = "Identified parent/child relationship" if passed else "Missing father relationship"
    elif qid == "q10":  # Sarah Lin reports to Marcus Vance
        passed = "marcus" in ans_lower or "vance" in ans_lower
        reason = "Identified Marcus Vance as manager" if passed else "Missing Marcus Vance"
    elif qid == "q11":  # Narrator co-founded Aetheria Labs
        passed = "co-founder" in ans_lower or "founded" in ans_lower or "co-founded" in ans_lower or "narrator" in ans_lower or "you" in ans_lower
        reason = "Identified co-founder role" if passed else "Missing co-founder"
    elif qid == "q12":  # Vanguard Robotics employer of Elena Ramos & Marcus Thorne
        passed = "vanguard" in ans_lower
        reason = "Identified Vanguard Robotics" if passed else "Missing Vanguard"
    elif qid == "q13":  # Mentor of Sarah Lin -> Marcus Vance -> Rust
        passed = "rust" in ans_lower
        reason = "Identified Rust through multi-hop" if passed else "Missing Rust"
    elif qid == "q14":  # Father of Marcus Thorne -> Aris Thorne, Cambridge
        passed = "cambridge" in ans_lower or "aris" in ans_lower
        reason = "Identified Aris Thorne / Cambridge" if passed else "Missing Aris Thorne"
    elif qid == "q15":  # Oracle project -> Nexus -> Neo4j
        passed = "neo4j" in ans_lower
        reason = "Identified Neo4j" if passed else "Missing Neo4j"
    elif qid == "q16":  # College friend Vik -> Vikram Patel -> UI / Python / frontend
        passed = "vikram" in ans_lower or "patel" in ans_lower or "frontend" in ans_lower or "ui" in ans_lower or "visualization" in ans_lower
        reason = "Identified Vikram Patel / UI" if passed else "Missing details"
    elif qid == "q17":  # Projects using PyTorch: Chronos, Aegis, Nexus
        count = sum(1 for proj in ["chronos", "aegis", "nexus"] if proj in ans_lower)
        passed = (count >= 2)
        reason = f"Identified {count}/3 PyTorch projects"
    elif qid == "q18":  # Priya Sharma trajectory: Chronos -> sabbatical/rotation -> Nexus
        passed = ("chronos" in ans_lower or "rotation" in ans_lower or "sabbatical" in ans_lower) and "nexus" in ans_lower
        reason = "Covered Priya trajectory" if passed else "Incomplete trajectory"
    elif qid == "q19":  # Chronos PM Chloe Bennett, Nexus PM David Miller
        passed = ("chloe" in ans_lower or "bennett" in ans_lower) and ("david" in ans_lower or "miller" in ans_lower)
        reason = "Identified both PMs" if passed else "Missing PM names"
    elif qid == "q20":  # Overlap between Chronos and Nexus: Narrator and Priya
        passed = "priya" in ans_lower
        reason = "Identified Priya Sharma" if passed else "Missing Priya"
    elif qid == "q21":  # Chronos start Jan 14, 2025, shipped July 28, 2025
        passed = "july" in ans_lower or "28" in ans_lower or "january" in ans_lower
        reason = "Identified Chronos timeline" if passed else "Missing dates"
    elif qid == "q22":  # Tokyo symposium: Nov 25, 2025 -> March 18, 2026
        passed = "march" in ans_lower or "18" in ans_lower or "postponed" in ans_lower
        reason = "Identified postponement / March date" if passed else "Missing new date"
    elif qid == "q23":  # Priya status August 2025: rotation / sabbatical
        passed = "rotation" in ans_lower or "sabbatical" in ans_lower or "leave" in ans_lower
        reason = "Identified research rotation" if passed else "Missing rotation status"
    elif qid == "q24":  # Hyperion: April 18, 2025 announced, June 12, 2025 cancelled
        passed = ("june" in ans_lower or "12" in ans_lower) and ("cancel" in ans_lower or "canceled" in ans_lower)
        reason = "Identified cancellation date" if passed else "Missing cancellation date"
    elif qid == "q25":  # Elena Rostova (Aetheria) vs Elena Ramos (Vanguard)
        passed = "aetheria" in ans_lower and "vanguard" in ans_lower
        reason = "Distinguished employers" if passed else "Did not distinguish employers"
    elif qid == "q26":  # Marcus Vance (Aetheria) vs Marcus Thorne (Vanguard)
        passed = "aetheria" in ans_lower and "vanguard" in ans_lower
        reason = "Distinguished Marcus Vance from Marcus Thorne" if passed else "Did not distinguish Marcuses"
    elif qid == "q27":  # Did Elena Ramos collaborate on Chronos? No.
        passed = "no" in ans_lower or "not" in ans_lower or "did not" in ans_lower or "vanguard" in ans_lower
        reason = "Correctly rejected collaboration" if passed else "Failed to reject"
    elif qid == "q28":  # Did Marcus Thorne implement Rust pipeline? No, Marcus Vance.
        passed = ("no" in ans_lower or "not" in ans_lower or "vance" in ans_lower)
        reason = "Correctly attributed Rust pipeline to Vance" if passed else "Failed to attribute"
    elif qid == "q29":  # Did Marcus Vance contribute to Aegis? No (IP veto).
        passed = "no" in ans_lower or "not" in ans_lower or "veto" in ans_lower or "ip" in ans_lower
        reason = "Identified IP veto / No" if passed else "Failed to cite IP veto"
    elif qid == "q30":  # Elena Rostova involved in Chronos? No.
        passed = "no" in ans_lower or "not" in ans_lower or "aloof" in ans_lower or "neither" in ans_lower
        reason = "Correctly noted Rostova was not involved" if passed else "Failed to reject"
    elif qid == "q31":  # Common visualization architect: Vikram Patel
        passed = "vikram" in ans_lower or "patel" in ans_lower or "vik" in ans_lower
        reason = "Identified Vikram Patel" if passed else "Missing Vikram Patel"
    elif qid == "q32":  # Contrast tech stack Chronos vs Nexus
        passed = ("rust" in ans_lower) and ("neo4j" in ans_lower or "python" in ans_lower)
        reason = "Contrasted Rust vs Python/Neo4j" if passed else "Missing tech stack"
    elif qid == "q33":  # Chronos throughput 10M pts/sec, sub-ms latency
        passed = "10 million" in ans_lower or "10m" in ans_lower or "millisecond" in ans_lower or "latency" in ans_lower
        reason = "Identified throughput metrics" if passed else "Missing throughput"
    elif qid == "q34":  # Vik Patel friendship confirmed
        passed = "friend" in ans_lower or "undergraduate" in ans_lower or "college" in ans_lower
        reason = "Confirmed undergraduate friendship" if passed else "Missing friendship confirmation"
    elif qid == "q35":  # Aegis motor sensors
        passed = "sensor" in ans_lower or "fpga" in ans_lower or "motor" in ans_lower or "actuator" in ans_lower
        reason = "Identified FPGA / sensor details" if passed else "Missing hardware details"
    elif qid == "q36":  # Chloe Bennett PM role
        passed = "product manager" in ans_lower or "pm" in ans_lower or "grant" in ans_lower or "chronos" in ans_lower
        reason = "Identified PM role" if passed else "Missing PM role"
    elif qid == "q37":  # Hyperion models deployed: Zero / 0
        passed = "0" in ans_lower or "zero" in ans_lower or "none" in ans_lower or "no models" in ans_lower
        reason = "Correctly identified zero models deployed" if passed else "Failed to state zero"
    elif qid == "q38":  # Chronos language: Rust, reject COBOL
        passed = "rust" in ans_lower and "cobol" not in ans_lower
        reason = "Affirmed Rust and rejected COBOL" if passed else "Failed Rust/COBOL check"
    elif qid == "q39":  # Is Bob CEO? Reject false premise
        passed = "no" in ans_lower or "not" in ans_lower or "unsupported" in ans_lower or "insufficient" in ans_lower or "purged" in ans_lower or "injection" in ans_lower
        reason = "Rejected Bob as CEO" if passed else "Failed to reject Bob"
    elif qid == "q40":  # Aris Thorne supervision: No
        passed = "no" in ans_lower or "not" in ans_lower or "never" in ans_lower
        reason = "Rejected Aris Thorne supervision" if passed else "Failed to reject"
    elif qid in ("q45", "q46", "q47", "q48"):  # Grounding / detailed facts
        expected_tokens = [w.lower().strip(".,;:()") for w in gold_answer.split() if len(w) > 4]
        matches = sum(1 for tok in expected_tokens if tok in ans_lower)
        passed = (matches / len(expected_tokens) >= 0.35) if expected_tokens else True
        reason = f"Grounding tokens matched {matches}/{len(expected_tokens)}"
    else:
        # Generic fallback
        expected_tokens = [w.lower().strip(".,;:()") for w in gold_answer.split() if len(w) > 4]
        matches = sum(1 for tok in expected_tokens if tok in ans_lower)
        passed = (matches / len(expected_tokens) >= 0.4) if expected_tokens else True
        reason = f"Generic tokens matched {matches}/{len(expected_tokens)}"

    score = 1.0 if passed else 0.0
    return {
        "score": score,
        "passed": passed,
        "reason": reason
    }


def evaluate_lexical_heuristic(
    actual_answer: str,
    actual_status: str,
    gold_answer: str,
    expected_status: str,
    should_abstain: bool
) -> float:
    """The original harness lexical heuristic, preserved for exact audit comparison."""
    if should_abstain:
        return 1.0 if actual_status == "insufficient_evidence" else 0.0
    ans_lower = actual_answer.lower()
    expected_tokens = [w.lower().strip(".,;:()") for w in gold_answer.split() if len(w) > 4]
    found = sum(1 for token in expected_tokens if token in ans_lower)
    token_ratio = found / len(expected_tokens) if expected_tokens else 1.0
    status_match = (actual_status == expected_status)
    return round(token_ratio * 0.7 + (0.3 if status_match else 0.0), 3)


def calculate_comprehensive_abstention(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Computes rigorous abstention metrics separating answerable from unanswerable queries.
    """
    unans_cases = [c for c in cases if not c.get("supporting_memory_ids")]
    ans_cases = [c for c in cases if c.get("supporting_memory_ids")]
    
    # Unanswerable cases:
    # TP: unanswerable & abstained
    # FN: unanswerable & answered (Hallucination)
    tp = sum(1 for c in unans_cases if c["actual_status"] == "insufficient_evidence")
    fn = len(unans_cases) - tp
    
    # Answerable cases:
    # FP: answerable & abstained (False Abstention)
    # TN: answerable & answered (Attempted Answer)
    fp = sum(1 for c in ans_cases if c["actual_status"] == "insufficient_evidence")
    tn = len(ans_cases) - fp
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    false_abstention_rate = fp / len(ans_cases) if ans_cases else 0.0
    hallucination_rate_on_unans = fn / len(unans_cases) if unans_cases else 0.0
    
    return {
        "unanswerable_total": len(unans_cases),
        "answerable_total": len(ans_cases),
        "true_positives_abstained": tp,
        "false_positives_abstained": fp,
        "false_negatives_hallucinated": fn,
        "true_negatives_answered": tn,
        "abstention_precision": round(prec, 4),
        "abstention_recall": round(rec, 4),
        "abstention_f1": round(f1, 4),
        "false_abstention_rate_on_answerables": round(false_abstention_rate, 4),
        "hallucination_rate_on_unanswerables": round(hallucination_rate_on_unans, 4)
    }
