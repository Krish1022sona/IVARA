"""
LongMemEval Benchmark Evaluator for IVARA.

Implements the official scoring protocol verified from `xiaowu0162/LongMemEval/src/evaluation/evaluate_qa.py`
and `src/evaluation/print_qa_metrics.py`.

Supports:
1. Official LLM-as-a-Judge mode (using OpenAI gpt-4o-2024-08-06 or an OpenAI-compatible endpoint)
2. Offline Deterministic Smoke-Test mode (clearly labeled as smoke_test_deterministic_heuristics)

Provides separate metric accounting for:
- Official metrics exactly as defined by `print_qa_metrics.py` (pooling answerable and abstention per task)
- Answerable-only metrics (evaluating the 6 tasks strictly on non-abstention questions)
- Abstention metrics (evaluated on the 30 `_abs` questions)
- Supplementary all-500 accuracy with fixed denominator 500
"""

import re
from typing import Any, Callable
import numpy as np


OFFICIAL_TASKS = [
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update"
]


def get_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    """
    Verbatim official prompt generation from `xiaowu0162/LongMemEval/src/evaluation/evaluate_qa.py`.
    """
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. If the response only contains a "
                "subset of the information required by the answer, answer no. \n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            prompt = template.format(question, answer, response)
        elif task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. If the response only contains a "
                "subset of the information required by the answer, answer no. In addition, do not penalize "
                "off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, "
                "etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), "
                "the model's response is still correct. \n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            prompt = template.format(question, answer, response)
        elif task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, the response "
                "should be considered as correct as long as the updated answer is the required answer.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            prompt = template.format(question, answer, response)
        elif task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. The response is correct as "
                "long as it recalls and utilizes the user's personal information correctly.\n\n"
                "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            prompt = template.format(question, answer, response)
        else:
            template = (
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            prompt = template.format(question, answer, response)
    else:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )
        prompt = template.format(question, answer, response)
    return prompt


def evaluate_response_locally(
    task: str,
    question: str,
    gold_answer: str,
    hypothesis: str,
    is_abstention: bool = False
) -> bool:
    """
    Deterministic rule-based offline scoring heuristic for fast CI smoke testing.
    Explicitly labeled as heuristic; never substitutes for official LLM judging.
    """
    hyp_lower = str(hypothesis).lower().strip()
    gold_lower = str(gold_answer).lower().strip()

    if is_abstention:
        abstention_markers = [
            "insufficient",
            "cannot answer",
            "cannot be answered",
            "cannot determine",
            "cannot be determined",
            "don't have sufficient",
            "do not have sufficient",
            "not mentioned",
            "no mention of",
            "no mention",
            "does not mention",
            "do not mention",
            "did not mention",
            "no information",
            "not stated",
            "unknown",
            "unanswerable",
            "not available",
            "does not contain",
            "do not contain",
            "not recorded",
            "no record of",
            "no record",
            "there is no record",
            "not documented",
            "no documentation",
            "not found in",
            "not found",
            "could not find",
            "cannot find",
            "not find",
            "no evidence",
            "not provided",
            "does not specify",
            "no details",
            "could not reliably synthesize",
        ]
        return any(m in hyp_lower for m in abstention_markers)

    if not hyp_lower:
        return False

    # Direct containment
    if gold_lower in hyp_lower:
        return True

    # Temporal reasoning numeric check with off-by-one tolerance
    if task == "temporal-reasoning":
        gold_nums = [int(n) for n in re.findall(r'\b\d+\b', gold_lower)]
        hyp_nums = [int(n) for n in re.findall(r'\b\d+\b', hyp_lower)]
        if gold_nums:
            if not hyp_nums:
                return False
            matches = [any(abs(gn - hn) <= 1 for hn in hyp_nums) for gn in gold_nums]
            return all(matches)

    # Token overlap check (75%+ of content words)
    gold_tokens = [t for t in re.findall(r'\b\w+\b', gold_lower) if len(t) > 2]
    if gold_tokens:
        matched = sum(1 for t in gold_tokens if t in hyp_lower)
        if matched / len(gold_tokens) >= 0.75:
            return True

    return False


class LongMemEvalEvaluator:
    """
    Evaluator running verified LongMemEval scoring.
    """

    def __init__(
        self,
        judge_model: str = "gpt-4o",
        llm_caller: Callable[[str], str] | None = None,
        use_local_heuristic: bool = False
    ):
        self.judge_model = judge_model
        self.llm_caller = llm_caller
        self.use_local_heuristic = use_local_heuristic

    def score_single(
        self,
        task: str,
        question: str,
        gold_answer: str,
        hypothesis: str,
        is_abstention: bool = False
    ) -> bool:
        """Score a single prediction."""
        if self.use_local_heuristic or self.llm_caller is None:
            return evaluate_response_locally(task, question, gold_answer, hypothesis, is_abstention)

        prompt = get_anscheck_prompt(task, question, gold_answer, hypothesis, abstention=is_abstention)
        judge_raw = self.llm_caller(prompt)
        return "yes" in str(judge_raw).lower()

    def evaluate(
        self,
        hypotheses: list[dict[str, Any]],
        references: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Evaluate full predictions against references and compute official metrics.

        Faithfully implements `src/evaluation/print_qa_metrics.py`:
          all_acc, task_acc, abstention_acc = [], [], []
          type2acc = {t: [] for t in OFFICIAL_TASKS}
          for entry in in_data:
              type2acc[ref_entry['question_type']].append(...)
              if '_abs' in entry['question_id']:
                  abstention_acc.append(...)
        """
        # Official script tracking (pools answerables + abstentions in type2acc)
        type2acc: dict[str, list[int]] = {t: [] for t in OFFICIAL_TASKS}
        all_acc: list[int] = []
        task_acc: list[float] = []
        abstention_acc: list[int] = []

        # Isolated answerable-only tracking (strictly non-_abs questions)
        ans_type2acc: dict[str, list[int]] = {t: [] for t in OFFICIAL_TASKS}
        ans_all_acc: list[int] = []
        ans_task_acc: list[float] = []

        detailed_logs = []
        missing_count = 0

        for entry in hypotheses:
            qid = entry.get("question_id")
            if not qid or qid not in references:
                missing_count += 1
                continue

            ref = references[qid]
            task = ref.get("question_type", "unknown")
            question = ref.get("question", "")
            gold_ans = ref.get("answer", "")
            hypothesis = entry.get("hypothesis", "")
            is_abs = ref.get("is_abstention", False) or "_abs" in qid

            label = self.score_single(
                task=task,
                question=question,
                gold_answer=gold_ans,
                hypothesis=hypothesis,
                is_abstention=is_abs
            )

            int_label = 1 if label else 0

            # 1. Official pooling
            if task in type2acc:
                type2acc[task].append(int_label)

            if is_abs:
                abstention_acc.append(int_label)
            else:
                if task in ans_type2acc:
                    ans_type2acc[task].append(int_label)
                ans_all_acc.append(int_label)

            judge_name = (
                "smoke_test_deterministic_heuristics"
                if (self.use_local_heuristic or self.llm_caller is None)
                else self.judge_model
            )

            scored_entry = dict(entry)
            scored_entry["autoeval_label"] = {
                "judge": judge_name,
                "label": label,
                "is_abstention": is_abs,
                "task": task
            }
            detailed_logs.append(scored_entry)

        # 2. Compute official metrics exactly per print_qa_metrics.py
        official_tasks = {}
        for t, v in type2acc.items():
            if len(v) > 0:
                mean_v = float(np.mean(v))
                all_acc += v
                task_acc.append(mean_v)
                official_tasks[t] = {
                    "accuracy": round(mean_v, 4),
                    "correct": sum(v),
                    "total": len(v)
                }
            else:
                official_tasks[t] = {"accuracy": 0.0, "correct": 0, "total": 0}

        official_overall = round(float(np.mean(all_acc)), 4) if all_acc else 0.0
        official_task_avg = round(float(np.mean(task_acc)), 4) if task_acc else 0.0
        official_abstention = round(float(np.mean(abstention_acc)), 4) if abstention_acc else 0.0

        # 3. Compute answerable-only metrics
        answerable_tasks = {}
        for t, v in ans_type2acc.items():
            if len(v) > 0:
                mean_v = float(np.mean(v))
                ans_task_acc.append(mean_v)
                answerable_tasks[t] = {
                    "accuracy": round(mean_v, 4),
                    "correct": sum(v),
                    "total": len(v)
                }
            else:
                answerable_tasks[t] = {"accuracy": 0.0, "correct": 0, "total": 0}

        ans_overall = round(float(np.mean(ans_all_acc)), 4) if ans_all_acc else 0.0
        ans_task_avg = round(float(np.mean(ans_task_acc)), 4) if ans_task_acc else 0.0

        # 4. Optional all-500 accuracy (clearly labeled as supplementary)
        all_500_acc = round(float(sum(all_acc) / 500.0), 4) if len(all_acc) <= 500 else None

        eval_mode_label = (
            "smoke_test_deterministic_heuristics"
            if (self.use_local_heuristic or self.llm_caller is None)
            else "official_llm_judge"
        )

        return {
            "evaluation_mode": eval_mode_label,
            "judge_model": self.judge_model if eval_mode_label == "official_llm_judge" else None,
            "official_status": "official" if eval_mode_label == "official_llm_judge" else "smoke_test_pending_official_judge",
            "official_metrics": {
                "overall_accuracy": official_overall,
                "task_averaged_accuracy": official_task_avg,
                "abstention_accuracy": official_abstention,
                "denominators": {
                    "overall_accuracy": len(all_acc),
                    "task_averaged_accuracy": len(task_acc),
                    "abstention_accuracy": len(abstention_acc)
                },
                "tasks": official_tasks
            },
            "answerable_only_metrics": {
                "overall_accuracy": ans_overall,
                "task_averaged_accuracy": ans_task_avg,
                "denominators": {
                    "overall_accuracy": len(ans_all_acc),
                    "task_averaged_accuracy": len(ans_task_acc)
                },
                "tasks": answerable_tasks
            },
            "supplementary_all_500_metrics": {
                "all_500_accuracy": all_500_acc,
                "denominator": 500,
                "note": "Supplementary metric with fixed denominator 500. Differs from official overall_accuracy when fewer than 500 questions are evaluated."
            },
            "summary_counts": {
                "total_evaluated": len(detailed_logs),
                "total_missing": missing_count,
                "total_answerable_evaluated": len(ans_all_acc),
                "total_abstention_evaluated": len(abstention_acc)
            },
            "detailed_logs": detailed_logs
        }
