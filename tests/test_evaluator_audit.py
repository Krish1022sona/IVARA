"""
Unit tests auditing the evaluation heuristics used in LongMemEval CI smoke testing.
Tests positive cases, negative cases, paraphrases, abstention variations,
and known edge cases/flaws in deterministic rule-based evaluation.
"""

import pytest
from evaluation.evaluators.longmemeval_evaluator import evaluate_response_locally


class TestEvaluatorAudit:
    """Audit of evaluate_response_locally behavior across question types and edge cases."""

    # 1. Exact match and direct containment
    def test_direct_containment_positive(self):
        gold = "Business Administration"
        hyp = "You graduated with a degree in Business Administration in 2021."
        assert evaluate_response_locally("single-session-user", "degree", gold, hyp, is_abstention=False) is True

    def test_direct_containment_negative(self):
        gold = "Computer Science"
        hyp = "You graduated with a degree in Business Administration."
        assert evaluate_response_locally("single-session-user", "degree", gold, hyp, is_abstention=False) is False

    # 2. Known Flaw: Spelled-out numbers vs digits in temporal reasoning
    def test_temporal_numeric_digit_matches(self):
        gold = "14 days"
        hyp = "14 days"
        assert evaluate_response_locally("temporal-reasoning", "days", gold, hyp, is_abstention=False) is True

    def test_temporal_spelled_out_number_fails_in_existing_evaluator(self):
        r"""
        Demonstrates known flaw: 'Fourteen days' contains no digits \b\d+\b,
        so it fails temporal evaluation even though semantically identical to '14 days'.
        """
        gold = "14 days"
        hyp = "Fourteen days had passed between the two events."
        # Known flaw: evaluate_response_locally returns False because it only extracts \d+
        res = evaluate_response_locally("temporal-reasoning", "days", gold, hyp, is_abstention=False)
        assert res is False, "Expected existing heuristic to fail on spelled-out number 'Fourteen'"

    def test_temporal_off_by_one_tolerance(self):
        """
        Demonstrates heuristic feature: numeric check has an intentional +/- 1 day tolerance.
        """
        gold = "14 days"
        hyp = "15 days"
        assert evaluate_response_locally("temporal-reasoning", "days", gold, hyp, is_abstention=False) is True

    def test_temporal_off_by_two_fails(self):
        gold = "14 days"
        hyp = "16 days"
        assert evaluate_response_locally("temporal-reasoning", "days", gold, hyp, is_abstention=False) is False

    # 3. Known Flaw: Abstention phrasing brittleness
    def test_abstention_matching_contraction_passes(self):
        gold = "unanswerable"
        hyp = "I don't have sufficient recorded memory to answer."
        assert evaluate_response_locally("knowledge-update", "q", gold, hyp, is_abstention=True) is True

    def test_abstention_matching_formal_phrasing_fails_in_existing_evaluator(self):
        """
        Demonstrates known flaw: 'do not have sufficient' is not in the hardcoded list
        ['don't have sufficient', 'insufficient', ...], so formal phrasing fails.
        """
        gold = "unanswerable"
        hyp = "I do not have sufficient recorded memories to determine how often you see Dr. Johnson."
        # Known flaw: returns False because 'do not have sufficient' is not 'don't have sufficient'
        res = evaluate_response_locally("knowledge-update", "q", gold, hyp, is_abstention=True)
        assert res is False, "Expected existing heuristic to fail on 'do not have sufficient' (no contraction)"

    def test_abstention_false_positive_risk(self):
        """
        Demonstrates keyword false positive risk: if model produces an ungrounded hallucination
        that happens to mention 'no information was provided, but the answer is 42',
        the substring 'no information' causes a false positive abstention score.
        """
        gold = "unanswerable"
        hyp = "There is no information in the file, but in general the answer is 42."
        # Contains 'no information'
        assert evaluate_response_locally("knowledge-update", "q", gold, hyp, is_abstention=True) is True

    # 4. Token overlap check (75%+ content words)
    def test_token_overlap_positive(self):
        gold = "red leather vintage camera bag"
        hyp = "The user bought a red leather vintage camera case from the store."
        # tokens: red, leather, vintage, camera, bag (5 tokens, 4 match = 80%)
        assert evaluate_response_locally("single-session-user", "q", gold, hyp, is_abstention=False) is True

    def test_token_overlap_negative_under_threshold(self):
        gold = "red leather vintage camera bag"
        hyp = "The user bought a red plastic modern camera case."
        # only 'red' and 'camera' match (2/5 = 40% < 75%)
        assert evaluate_response_locally("single-session-user", "q", gold, hyp, is_abstention=False) is False
