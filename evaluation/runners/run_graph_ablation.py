"""
Graph and Ontology Ablation Runner for LongMemEval-S.

Executes controlled comparison between:
1. Config B-Chunked (Baseline)
2. Config B-Chunked-Ontology (Ontology-guided expansion/re-ranking, no graph)
3. Config B-Chunked-Graph (Turn graph construction + relevance-guided traversal)
"""

import io
import json
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.runners.run_longmemeval import LongMemEvalRunner

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


def run_configuration(config_name: str, output_dir: Path, target_qids: list[str]) -> dict:
    dataset_path = Path("evaluation/data/longmemeval/longmemeval_s_cleaned.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_doc = {
        "ablation_target": f"Config {config_name}",
        "question_ids": target_qids,
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
        config=config_name,
        model="gpt-oss",
        k=5,
        use_local_judge=True
    )

    summary = runner.run(
        target_question_ids=target_qids,
        resume=True
    )
    return summary


def main():
    base_runs_dir = Path("evaluation/data/longmemeval/runs")

    # 1. Run B-Chunked-Ontology on diagnostic sample
    print("\n" + "="*70)
    print("RUNNING CONFIG B-CHUNKED-ONTOLOGY (DIAGNOSTIC SAMPLE)")
    print("="*70)
    ont_dir = base_runs_dir / "ablation_sample_config_b_chunked_ontology"
    run_configuration("B-Chunked-Ontology", ont_dir, DIAGNOSTIC_SAMPLE_QIDS)

    # 2. Run B-Chunked-Graph on diagnostic sample
    print("\n" + "="*70)
    print("RUNNING CONFIG B-CHUNKED-GRAPH (DIAGNOSTIC SAMPLE)")
    print("="*70)
    graph_dir = base_runs_dir / "ablation_sample_config_b_chunked_graph"
    run_configuration("B-Chunked-Graph", graph_dir, DIAGNOSTIC_SAMPLE_QIDS)


if __name__ == "__main__":
    main()
