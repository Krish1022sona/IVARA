"""
Held-Out Sample Evaluation Runner for LongMemEval-S.

Executes frozen configurations on the strictly disjoint 30-question held-out sample:
1. Config B-Chunked (Baseline)
2. Config B-Chunked-Ontology (Config B)
3. Config B-Chunked-Graph (Config C)
"""

import json
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.runners.run_longmemeval import LongMemEvalRunner

HELD_OUT_MANIFEST_PATH = Path("evaluation/data/longmemeval/runs/held_out_sample_manifest.json")


def load_held_out_qids() -> list[str]:
    with open(HELD_OUT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["question_ids"]


def run_configuration(config_name: str, output_dir: Path, target_qids: list[str]) -> dict:
    dataset_path = Path("evaluation/data/longmemeval/longmemeval_s_cleaned.json")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_doc = {
        "ablation_target": f"Config {config_name} (Held-Out Evaluation)",
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
    with open(output_dir / "held_out_manifest.json", "w", encoding="utf-8") as f:
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
    held_out_qids = load_held_out_qids()
    base_runs_dir = Path("evaluation/data/longmemeval/runs")

    # 1. Run B-Chunked Baseline on held-out sample
    print("\n" + "="*70)
    print("RUNNING CONFIG B-CHUNKED BASELINE (HELD-OUT SAMPLE)")
    print("="*70)
    base_dir = base_runs_dir / "held_out_sample_config_b_chunked"
    run_configuration("B-Chunked", base_dir, held_out_qids)

    # 2. Run B-Chunked-Ontology on held-out sample
    print("\n" + "="*70)
    print("RUNNING CONFIG B-CHUNKED-ONTOLOGY (HELD-OUT SAMPLE)")
    print("="*70)
    ont_dir = base_runs_dir / "held_out_sample_config_b_chunked_ontology"
    run_configuration("B-Chunked-Ontology", ont_dir, held_out_qids)

    # 3. Run B-Chunked-Graph on held-out sample
    print("\n" + "="*70)
    print("RUNNING CONFIG B-CHUNKED-GRAPH (HELD-OUT SAMPLE)")
    print("="*70)
    graph_dir = base_runs_dir / "held_out_sample_config_b_chunked_graph"
    run_configuration("B-Chunked-Graph", graph_dir, held_out_qids)


if __name__ == "__main__":
    main()
