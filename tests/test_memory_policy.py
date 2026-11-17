"""Comprehensive Test Suite for IVARA Memory Formation Policy.

Verifies:
1. Rejection of greetings, social pleasantries, and conversational fillers.
2. Rejection of general educational queries and concept explanations.
3. Rejection of pure recall questions ("When did I start RL?", "What do you know about me?").
4. Reliable extraction of personal declarative facts.
5. Correct classification into PREFERENCE, GOAL, EXPERIENCE, DECISION, PERSONAL_FACT.
6. Proper extraction of personal statements embedded in mixed messages (statements + questions).
7. Detection of updates, corrections, and revocations ("I no longer use Python").
8. Deduplication and suppression of redundant memory creation.
9. Formal precision, recall, false-positive rate, false-negative rate measurement across a curated dataset.
"""

import pytest
from src.extraction.policy import (
    MemoryCategory,
    MemoryFormationDecision,
    evaluate_memory_formation,
    should_extract_memory,
    is_casual_or_filler,
    is_pure_question_or_recall,
    is_general_knowledge_query,
    check_redundancy
)


# Test cases for category-level and policy-level validation
CASUAL_MESSAGES = [
    "Hi IVARA!",
    "Hello there",
    "hey",
    "good morning",
    "How are you doing today?",
    "what's up",
    "thanks a lot",
    "Thank you so much!",
    "ok",
    "cool",
    "awesome",
    "bye bye",
    "see you later",
    "yes",
    "nope",
    "you remember it all thats crazy"
]

GENERAL_KNOWLEDGE_MESSAGES = [
    "What is reinforcement learning?",
    "Explain backpropagation.",
    "How does gradient descent work?",
    "Can you explain transformer attention?",
    "What does API stand for?",
    "Why is GPU faster than CPU for deep learning?",
    "Tell me about quantum computing."
]

QUESTION_AND_RECALL_MESSAGES = [
    "When did I start learning RL?",
    "What do you know about me?",
    "What projects have I worked on?",
    "Do you remember my dog's name?",
    "Did I mention where I live?",
    "Who did I work with on the SIH project?",
    "What is my favorite programming language?"
]

PERSONAL_FACT_MESSAGES = [
    "I was born in New Delhi.",
    "My sister's name is Priya.",
    "I study at ABV IIITM Gwalior.",
    "I live in Bengaluru now.",
    "My dog is a golden retriever named Bruno."
]

PREFERENCE_MESSAGES = [
    "I prefer Python over Java.",
    "My favorite editor is Neovim.",
    "I really love dark mode interfaces.",
    "I dislike working with legacy SOAP APIs."
]

GOAL_MESSAGES = [
    "I am planning to learn computer vision next month.",
    "My goal is to publish a paper at NeurIPS 2026.",
    "I want to build an autonomous drone navigation system.",
    "I will start reading the Sutton and Barto RL book this weekend."
]

EXPERIENCE_MESSAGES = [
    "I built a neural network from scratch to play tic tac toe.",
    "I graduated from Delhi Public School in 2020.",
    "I worked on our college hackathon project last semester.",
    "I attended PyCon India in 2024."
]

DECISION_MESSAGES = [
    "I decided to switch our backend from Node.js to FastAPI.",
    "I chose PostgreSQL instead of MongoDB for our primary database.",
    "I opted for PyTorch rather than TensorFlow for our research lab."
]

UPDATE_MESSAGES = [
    "I no longer use Python for this project, I switched to Rust.",
    "Actually, I moved from Seattle to Berlin last week.",
    "Update: I stopped using Vim and now use VS Code.",
    "I left my job at TechCorp and joined an AI startup."
]

MIXED_MESSAGES = [
    # Statement + question
    ("I built a neural network from scratch, what do you think?", "I built a neural network from scratch."),
    ("I live in Berlin now. What's good to eat around Mitte?", "I live in Berlin now."),
    ("I worked with Rahul on the SIH project. Do you think we can win?", "I worked with Rahul on the SIH project."),
    ("I started learning reinforcement learning in July 2025! Can you give me some quiz questions?", "I started learning reinforcement learning in July 2025.")
]


class TestMemoryFormationPolicy:

    @pytest.mark.parametrize("msg", CASUAL_MESSAGES)
    def test_casual_messages_are_not_extracted(self, msg: str):
        decision = evaluate_memory_formation(msg)
        assert not decision.should_extract, f"Expected '{msg}' to be skipped, got should_extract=True"
        assert decision.category == MemoryCategory.CASUAL_CONVERSATION

    @pytest.mark.parametrize("msg", GENERAL_KNOWLEDGE_MESSAGES)
    def test_general_knowledge_queries_are_not_extracted(self, msg: str):
        decision = evaluate_memory_formation(msg)
        assert not decision.should_extract, f"Expected general knowledge '{msg}' to be skipped"
        assert decision.category == MemoryCategory.GENERAL_KNOWLEDGE

    @pytest.mark.parametrize("msg", QUESTION_AND_RECALL_MESSAGES)
    def test_pure_questions_and_recall_are_not_extracted(self, msg: str):
        decision = evaluate_memory_formation(msg)
        assert not decision.should_extract, f"Expected recall query '{msg}' to be skipped"
        assert decision.category == MemoryCategory.QUESTION

    @pytest.mark.parametrize("msg", PERSONAL_FACT_MESSAGES)
    def test_personal_facts_are_extracted(self, msg: str):
        decision = evaluate_memory_formation(msg)
        assert decision.should_extract, f"Expected personal fact '{msg}' to be extracted"
        assert decision.category == MemoryCategory.PERSONAL_FACT

    @pytest.mark.parametrize("msg", PREFERENCE_MESSAGES)
    def test_preferences_are_extracted(self, msg: str):
        decision = evaluate_memory_formation(msg)
        assert decision.should_extract, f"Expected preference '{msg}' to be extracted"
        assert decision.category == MemoryCategory.PREFERENCE
        assert decision.memory_type == "preference"

    @pytest.mark.parametrize("msg", GOAL_MESSAGES)
    def test_goals_are_extracted_and_distinguished(self, msg: str):
        decision = evaluate_memory_formation(msg)
        assert decision.should_extract, f"Expected goal '{msg}' to be extracted"
        assert decision.category == MemoryCategory.GOAL
        assert decision.memory_type == "goal"

    @pytest.mark.parametrize("msg", EXPERIENCE_MESSAGES)
    def test_experiences_are_extracted(self, msg: str):
        decision = evaluate_memory_formation(msg)
        assert decision.should_extract, f"Expected experience '{msg}' to be extracted"
        assert decision.category == MemoryCategory.EXPERIENCE
        assert decision.memory_type == "experience"

    @pytest.mark.parametrize("msg", DECISION_MESSAGES)
    def test_decisions_are_extracted(self, msg: str):
        decision = evaluate_memory_formation(msg)
        assert decision.should_extract, f"Expected decision '{msg}' to be extracted"
        assert decision.category == MemoryCategory.DECISION
        assert decision.memory_type == "decision"

    @pytest.mark.parametrize("msg", UPDATE_MESSAGES)
    def test_updates_are_flagged(self, msg: str):
        decision = evaluate_memory_formation(msg)
        assert decision.should_extract, f"Expected update '{msg}' to be extracted"
        assert decision.is_update is True
        assert decision.memory_type == "update"

    @pytest.mark.parametrize("full_msg,expected_sub", MIXED_MESSAGES)
    def test_mixed_messages_extract_personal_statement(self, full_msg: str, expected_sub: str):
        decision = evaluate_memory_formation(full_msg)
        assert decision.should_extract, f"Expected mixed message '{full_msg}' to extract personal statement"
        assert decision.extracted_text is not None
        # Check that the inquiry portion was stripped or distilled
        assert "what do you think" not in decision.extracted_text.lower()
        assert "can you give me" not in decision.extracted_text.lower()

    def test_deduplication_and_redundancy(self):
        existing_memories = [
            {"id": "m001", "text": "I worked with Rahul on our SIH project."},
            {"id": "m002", "text": "I live in Berlin."}
        ]
        # Identical text
        dup_decision = evaluate_memory_formation("I worked with Rahul on our SIH project.", existing_memories=existing_memories)
        assert not dup_decision.should_extract
        assert dup_decision.is_duplicate is True
        assert dup_decision.existing_memory_id == "m001"

        # Slightly rephrased redundant text
        dup_decision2 = evaluate_memory_formation("I live in Berlin!", existing_memories=existing_memories)
        assert not dup_decision2.should_extract
        assert dup_decision2.is_duplicate is True
        assert dup_decision2.existing_memory_id == "m002"

        # Novel fact
        novel_decision = evaluate_memory_formation("I learned Rust in 2026.", existing_memories=existing_memories)
        assert novel_decision.should_extract
        assert not novel_decision.is_duplicate


def test_memory_formation_metrics():
    """
    Formally evaluate precision, recall, false-positive rate, and false-negative rate
    across a balanced corpus of positive (personal information) and negative (questions,
    fillers, general knowledge) utterances.
    """
    positives = (
        PERSONAL_FACT_MESSAGES +
        PREFERENCE_MESSAGES +
        GOAL_MESSAGES +
        EXPERIENCE_MESSAGES +
        DECISION_MESSAGES +
        UPDATE_MESSAGES +
        [m[0] for m in MIXED_MESSAGES]
    )
    negatives = (
        CASUAL_MESSAGES +
        GENERAL_KNOWLEDGE_MESSAGES +
        QUESTION_AND_RECALL_MESSAGES
    )

    tp = 0
    fn = 0
    for pos in positives:
        dec = evaluate_memory_formation(pos)
        if dec.should_extract:
            tp += 1
        else:
            fn += 1

    fp = 0
    tn = 0
    for neg in negatives:
        dec = evaluate_memory_formation(neg)
        if dec.should_extract:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"\n--- Memory Formation Policy Quality Metrics ---")
    print(f"Total Test Cases: {len(positives) + len(negatives)} ({len(positives)} positive, {len(negatives)} negative)")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"FNR:       {fnr:.4f}")

    # Strict performance criteria
    assert precision >= 0.95, f"Expected precision >= 0.95, got {precision:.4f}"
    assert recall >= 0.90, f"Expected recall >= 0.90, got {recall:.4f}"
    assert fpr <= 0.05, f"Expected FPR <= 0.05, got {fpr:.4f}"
    assert fnr <= 0.10, f"Expected FNR <= 0.10, got {fnr:.4f}"

