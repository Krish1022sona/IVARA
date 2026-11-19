"""
Controlled Ablation Runner: Config B-Chunked on 30-Question Diagnostic Sample.

Executes Turn-Aware Conversational Chunking (TACC) on LongMemEval-S using the identical
30-question sample, model (gpt-oss), k=5, and judging method as Config B.
"""

import io
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.runners.run_longmemeval import LongMemEvalRunner

# Identical 30 questions evaluated on Config B
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


def run_chunked_ablation():
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

    dataset_path = Path("evaluation/data/longmemeval/longmemeval_s_cleaned.json")
    output_dir = Path("evaluation/data/longmemeval/runs/ablation_sample_config_b_chunked")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_doc = {
        "ablation_target": "Config B-Chunked (Turn-Aware Conversational Chunking)",
        "baseline_run_dir": "evaluation/data/longmemeval/runs/diagnostic_sample_config_b",
        "question_ids": DIAGNOSTIC_SAMPLE_QIDS,
        "parameters": {
            "window_size": 3,
            "overlap": 1,
            "max_context_tokens": 1500,
            "vector_weight": 0.7,
            "bm25_weight": 0.3,
            "k": 5,
            "model": "gpt-oss"
        }
    }
    with open(output_dir / "ablation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)

    runner = LongMemEvalRunner(
        dataset_path=dataset_path,
        output_dir=output_dir,
        config="B-Chunked",
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
    run_chunked_ablation()
