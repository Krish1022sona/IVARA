"""
Reproducible Diagnostic Sample Runner for LongMemEval-S (Config B).

Executes a balanced, deterministic 30-question diagnostic sample spanning all
6 official LongMemEval task categories (5 questions per category: 4 answerable +
1 abstention where present; 5 answerable for preference and assistant).
"""

import io
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.runners.run_longmemeval import LongMemEvalRunner

# Deterministic Stratified 30-Question Diagnostic Sample
DIAGNOSTIC_SAMPLE_QIDS = [
    # 1. single-session-user (4 ans, 1 abs)
    "e47becba", "118b2229", "51a45a95", "58bf7951", "0862e8bf_abs",
    # 2. multi-session (4 ans, 1 abs)
    "0a995998", "6d550036", "gpt4_59c863d7", "b5ef892d", "88432d0a_abs",
    # 3. single-session-preference (5 ans)
    "8a2466db", "06878be2", "75832dbd", "0edc2aef", "35a27287",
    # 4. temporal-reasoning (4 ans, 1 abs)
    "gpt4_59149c77", "gpt4_f49edff3", "71017276", "b46e15ed", "gpt4_70e84552_abs",
    # 5. knowledge-update (4 ans, 1 abs)
    "6a1eabeb", "6aeb4375", "830ce83f", "852ce960", "6aeb4375_abs",
    # 6. single-session-assistant (5 ans)
    "7161e7e2", "c4f10528", "89527b6b", "e9327a54", "4c36ccef"
]


def run_diagnostic():
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

    dataset_path = Path("evaluation/data/longmemeval/longmemeval_s_cleaned.json")
    output_dir = Path("evaluation/data/longmemeval/runs/diagnostic_sample_config_b")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Document exact sampling methodology
    sampling_doc = {
        "sampling_method": "deterministic_stratified",
        "rationale": "Evenly samples 5 questions across all 6 LongMemEval tasks (4 answerable + 1 abstention where present; 5 answerable for preference and assistant tasks).",
        "total_sample_size": len(DIAGNOSTIC_SAMPLE_QIDS),
        "tasks": {
            "single-session-user": ["e47becba", "118b2229", "51a45a95", "58bf7951", "0862e8bf_abs"],
            "multi-session": ["0a995998", "6d550036", "gpt4_59c863d7", "b5ef892d", "88432d0a_abs"],
            "single-session-preference": ["8a2466db", "06878be2", "75832dbd", "0edc2aef", "35a27287"],
            "temporal-reasoning": ["gpt4_59149c77", "gpt4_f49edff3", "71017276", "b46e15ed", "gpt4_70e84552_abs"],
            "knowledge-update": ["6a1eabeb", "6aeb4375", "830ce83f", "852ce960", "6aeb4375_abs"],
            "single-session-assistant": ["7161e7e2", "c4f10528", "89527b6b", "e9327a54", "4c36ccef"]
        },
        "question_ids": DIAGNOSTIC_SAMPLE_QIDS
    }
    with open(output_dir / "sampling_methodology.json", "w", encoding="utf-8") as f:
        json.dump(sampling_doc, f, indent=2)

    runner = LongMemEvalRunner(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config="B",
        model="gpt-oss",
        k=5,
        use_local_judge=True
    )

    summary = runner.run(
        target_question_ids=DIAGNOSTIC_SAMPLE_QIDS,
        resume=True
    )
    return summary


if __name__ == "__main__":
    run_diagnostic()
