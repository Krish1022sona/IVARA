"""
Unit tests for the corrected local abstention scoring heuristic.
Tests positive, negative, and ambiguous candidate responses on unanswerable questions.
"""

import pytest
from evaluation.evaluators.longmemeval_evaluator import evaluate_response_locally


class TestLocalAbstentionHeuristic:
    """Tests evaluating unanswerable questions (is_abstention=True)."""

    @pytest.mark.parametrize(
        "positive_hyp",
        [
            "Based on the recorded conversations, there is no record of the name of your hamster.",
            "There is no record of you collecting vintage films in the provided conversations.",
            "Based on the provided evidence, there is no record of what you baked for your uncle's birthday party.",
            "I could not find any information about how long you were in Korea.",
            "The recorded conversations do not mention any violin practice.",
            "There is no evidence indicating when you started the Porsche 991 model.",
            "I do not have sufficient information to answer what your dad gave you.",
            "This information is unknown and not recorded in the chat history.",
            "The data does not contain details regarding the cost of the bus ticket.",
            "Could not reliably synthesize an answer for 'What is the name of my hamster?'.",
            "This question is unanswerable from the recorded dialogs.",
            "No mention of any Italian restaurants was found in the memory log."
        ]
    )
    def test_positive_abstention_cases(self, positive_hyp: str):
        """Positive cases: model properly recognizes absence of evidence and abstains."""
        result = evaluate_response_locally(
            task="abstention",
            question="What is the name of my hamster?",
            gold_answer="There is no record of a hamster.",
            hypothesis=positive_hyp,
            is_abstention=True
        )
        assert result is True, f"Failed to recognize valid abstention: '{positive_hyp}'"

    @pytest.mark.parametrize(
        "negative_hyp",
        [
            "Your hamster's name is Fluffy and it is very energetic.",
            "You have been collecting vintage films for 12 years since 2011.",
            "You baked a chocolate strawberry cake for your uncle's 50th birthday party.",
            "You spent 3 weeks in Korea visiting Seoul and Busan.",
            "You dedicate 45 minutes to violin practice every evening.",
            "Alex became a parent first, having adopted a baby girl from China in January.",
            "You have been working professionally for 9 years before starting your job at Google.",
            "Yes, you attended the Holiday Market 3 days before purchasing the iPad."
        ]
    )
    def test_negative_abstention_cases(self, negative_hyp: str):
        """Negative cases: model hallucinates a factual answer without abstaining."""
        result = evaluate_response_locally(
            task="abstention",
            question="What is the name of my hamster?",
            gold_answer="There is no record of a hamster.",
            hypothesis=negative_hyp,
            is_abstention=True
        )
        assert result is False, f"Erroneously approved hallucinated answer as abstention: '{negative_hyp}'"

    @pytest.mark.parametrize(
        "ambiguous_hyp,expected",
        [
            # Partial abstention: answers part 1, explicitly abstains on part 2 (should pass abstention)
            (
                "You spent 10 days traveling in Hawaii. There is no record in the provided evidence of any days spent traveling in Seattle.",
                True
            ),
            # Partial education: knows high school, abstains on master's degree
            (
                "Based on the recorded conversations, you attended Arcadia High School from 2010 to 2014. There is no record of your Master's degree.",
                True
            ),
            # Empty string should fail
            ("", False),
            ("   ", False),
            # Pure assertion with slight hedge but no statement of missing evidence
            (
                "I think your hamster might be named Cocoa or Brownie.",
                False
            ),
            (
                "Probably you have been collecting vintage films for a long time.",
                False
            )
        ]
    )
    def test_ambiguous_and_edge_abstention_cases(self, ambiguous_hyp: str, expected: bool):
        """Ambiguous cases: partial answers, hedging, and empty outputs."""
        result = evaluate_response_locally(
            task="abstention",
            question="How many days did I spend in total traveling in Hawaii and in Seattle?",
            gold_answer="Seattle travel is unrecorded.",
            hypothesis=ambiguous_hyp,
            is_abstention=True
        )
        assert result is expected, f"Unexpected result for ambiguous case '{ambiguous_hyp}': got {result}, expected {expected}"

