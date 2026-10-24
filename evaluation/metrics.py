"""
Deterministic evaluation metrics for IVARA memory retrieval and answer generation.

All metrics are purely deterministic functions with explicit, documented conventions
for handling empty denominators.
"""

from typing import Sequence


def retrieval_recall_at_k(retrieved_ids: Sequence[str], supporting_ids: Sequence[str]) -> float:
    """
    Compute Recall@k against annotated supporting memory IDs.

    Formula:
        |Retrieved IDs ∩ Supporting IDs| / |Supporting IDs|

    Empty Denominator Conventions:
        - If Supporting IDs is empty (unanswerable / distractor query):
          * Returns 1.0 if Retrieved IDs is empty (successful rejection).
          * Returns 0.0 if Retrieved IDs is non-empty (false positive retrieval).
        - If Supporting IDs is non-empty and Retrieved IDs is empty:
          * Returns 0.0 (complete retrieval miss).
    """
    ret_set = set(retrieved_ids)
    sup_set = set(supporting_ids)

    if not sup_set:
        return 1.0 if not ret_set else 0.0

    if not ret_set:
        return 0.0

    intersection = ret_set & sup_set
    return len(intersection) / len(sup_set)


def retrieval_precision_at_k(retrieved_ids: Sequence[str], supporting_ids: Sequence[str]) -> float:
    """
    Compute Precision@k against annotated supporting memory IDs.

    Formula:
        |Retrieved IDs ∩ Supporting IDs| / |Retrieved IDs|

    Empty Denominator Conventions:
        - If Retrieved IDs is empty:
          * Returns 1.0 if Supporting IDs is also empty (correctly avoided retrieving anything).
          * Returns 0.0 if Supporting IDs is non-empty (failed to retrieve required memories).
    """
    ret_set = set(retrieved_ids)
    sup_set = set(supporting_ids)

    if not ret_set:
        return 1.0 if not sup_set else 0.0

    intersection = ret_set & sup_set
    return len(intersection) / len(ret_set)


def citation_precision(cited_ids: Sequence[str], supporting_ids: Sequence[str]) -> float:
    """
    Compute Citation Precision: the fraction of cited memory IDs that genuinely
    belong to the annotated supporting memories.

    Formula:
        |Cited IDs ∩ Supporting IDs| / |Cited IDs|

    Empty Denominator Conventions:
        - If Cited IDs is empty:
          * Returns 1.0 if Supporting IDs is empty (abstention was appropriate, no citations expected).
          * Returns 0.0 if Supporting IDs was non-empty (answer claimed without citations).
    """
    c_set = set(str(c).strip().strip("[]'\"") for c in cited_ids if str(c).strip())
    sup_set = set(supporting_ids)

    if not c_set:
        return 1.0 if not sup_set else 0.0

    valid_cites = c_set & sup_set
    return len(valid_cites) / len(c_set)


def unsupported_claim_rate(cited_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    r"""
    Compute Unsupported-Claim Rate (Hallucinated Citation Rate):
    Fraction of cited memory IDs that were NOT in the retrieved context.

    Formula:
        |Cited IDs \ Retrieved IDs| / |Cited IDs|

    Empty Denominator Conventions:
        - If Cited IDs is empty:
          * Returns 0.0 (no unsupported citations were made).
    """
    c_set = set(str(c).strip().strip("[]'\"") for c in cited_ids if str(c).strip())
    ret_set = set(retrieved_ids)

    if not c_set:
        return 0.0

    hallucinated = c_set - ret_set
    return len(hallucinated) / len(c_set)


def fact_accuracy(
    answer_text: str,
    expected_facts: Sequence[str],
    is_unanswerable: bool = False,
    status: str = "answered"
) -> float:
    """
    Compute deterministic fact accuracy against annotated expected facts.

    Convention:
        - For unanswerable queries:
          * Returns 1.0 if status == 'insufficient_evidence' (system accurately abstained).
          * Returns 0.0 if status != 'insufficient_evidence' (system hallucinated an answer).
        - For answerable queries:
          * Returns fraction of expected fact strings found (case-insensitively) in answer_text.
          * If expected_facts is empty: returns 1.0.
    """
    if is_unanswerable:
        return 1.0 if status == "insufficient_evidence" else 0.0

    if not expected_facts:
        return 1.0

    if not answer_text or not isinstance(answer_text, str):
        return 0.0

    ans_lower = answer_text.lower()
    matches = sum(1 for fact in expected_facts if fact.lower() in ans_lower)
    return matches / len(expected_facts)


def compute_abstention_metrics(
    predictions_abstained: Sequence[bool],
    ground_truth_unanswerable: Sequence[bool]
) -> dict[str, float]:
    """
    Compute Abstention Precision, Recall, and F1 on answerable vs unanswerable questions.

    Definitions:
        - True Positive (TP): Case was unanswerable AND system abstained.
        - False Positive (FP): Case was answerable BUT system abstained.
        - False Negative (FN): Case was unanswerable BUT system attempted an answer.
        - True Negative (TN): Case was answerable AND system attempted an answer.

    Empty Denominator Conventions:
        - Precision: TP / (TP + FP). If (TP + FP) == 0, returns 1.0 if all cases were answerable, else 0.0.
        - Recall: TP / (TP + FN). If (TP + FN) == 0 (no unanswerable cases), returns 1.0.
        - F1: 2 * P * R / (P + R). If (P + R) == 0, returns 0.0.
    """
    if len(predictions_abstained) != len(ground_truth_unanswerable):
        raise ValueError("predictions and ground_truth must have equal length.")

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for pred, actual in zip(predictions_abstained, ground_truth_unanswerable):
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1

    total_abstained = tp + fp
    total_unanswerable = tp + fn

    if total_abstained > 0:
        precision = tp / total_abstained
    else:
        precision = 1.0 if total_unanswerable == 0 else 0.0

    if total_unanswerable > 0:
        recall = tp / total_unanswerable
    else:
        recall = 1.0

    if (precision + recall) > 0.0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    total_cases = len(predictions_abstained)
    accuracy = (tp + tn) / total_cases if total_cases > 0 else 1.0

    return {
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "abstention_f1": round(f1, 4),
        "abstention_accuracy": round(accuracy, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    }
