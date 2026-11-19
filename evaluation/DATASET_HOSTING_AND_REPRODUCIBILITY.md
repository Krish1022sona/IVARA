# IVARA — Research Reproducibility & Benchmark Dataset Guide

## 1. Overview & Dataset Taxonomy

To ensure complete scientific transparency while adhering to GitHub file-size limits, this guide categorizes all benchmark artifacts, explains their provenance, and outlines the exact steps to verify and reproduce the 500-question **LongMemEval-S** evaluation results.

| Artifact Class | Description | Location | Handling in Git |
|:---|:---|:---|:---|
| **Raw Benchmark Dataset** | 500-question multi-session dataset (264.5 MB) | `evaluation/data/longmemeval/longmemeval_s_cleaned.json` | Excluded via `.gitignore` (Exceeds GitHub 100MB limit; use Git LFS or external mirror) |
| **Diagnostic Subsets** | 100-question and 40-question validation slices | `evaluation/autonomous_sprint/` | Large JSON files (>20 MB) excluded via `.gitignore` |
| **Sealed Model Predictions** | 500 JSONL predictions per candidate | `evaluation/data/longmemeval/runs/final_bakeoff/` | Tracked / Archived with SHA-256 verification |
| **Official Evaluation Outputs** | Accuracy, Recall@5, latency, confusion matrices | `evaluation/data/longmemeval/runs/final_bakeoff/*/summary_config_D.json` | Tracked with cryptographic hashes |
| **Statistical Analysis** | McNemar test calculations and contingency tables | `docs/evaluation/FINAL_BAKEOFF_REPORT.md` | Tracked living documentation |
| **Ablation Reports** | 4-arm graph ablation and routing evaluations | `archive/experiments/` | Tracked historical research archive |

---

## 2. Cryptographic Provenance & Hash Verification

The final 500-question bakeoff evaluated three architectures under frozen conditions at Git commit `55a273bff20fa0eec1b2d361e8a03b217d147abe`.

You can independently verify the cryptographic integrity of all prediction sets using SHA-256:

### Candidate B (Winning Architecture — Flat MMR + TACC Engine)
- **Path:** `evaluation/data/longmemeval/runs/final_bakeoff/candidate_b_arm_b/results_config_D.jsonl`
- **Expected SHA-256:** `ba9e71bec593c71f76d802fe154d21321776defd9fc808f6ade735867e90a9b9`
- **Sample Size:** 500 questions
- **Overall Accuracy:** 58.40% (292 / 500) | **Macro Task Average:** 61.34%

### Candidate C (Adaptive Router Architecture)
- **Path:** `evaluation/data/longmemeval/runs/final_bakeoff/candidate_c_phase3/results_config_D.jsonl`
- **Expected SHA-256:** `58062320b1c0424d78aa90c84a79866ccbfd33bb233ee1e028b0e4a2ee04a2ee`
- **Sample Size:** 500 questions
- **Overall Accuracy:** 56.80% (284 / 500) | **Macro Task Average:** 58.75%

### Candidate E (Baseline RAG Architecture)
- **Path:** `evaluation/data/longmemeval/runs/final_bakeoff/candidate_e_baseline/results_config_D.jsonl`
- **Expected SHA-256:** `914227ae70df1b8e00ba421492249e3b65d22fc0acf477b784a81ff8503ae3a3`
- **Sample Size:** 500 questions
- **Overall Accuracy:** 40.80% (204 / 500) | **Macro Task Average:** 44.57%

### Verification Script:
```bash
python -c "
import hashlib
from pathlib import Path

files = {
    'Candidate B': 'evaluation/data/longmemeval/runs/final_bakeoff/candidate_b_arm_b/results_config_D.jsonl',
    'Candidate C': 'evaluation/data/longmemeval/runs/final_bakeoff/candidate_c_phase3/results_config_D.jsonl',
    'Candidate E': 'evaluation/data/longmemeval/runs/final_bakeoff/candidate_e_baseline/results_config_D.jsonl',
}

for name, path in files.items():
    p = Path(path)
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f'{name:12s} SHA-256: {h}')
    else:
        print(f'{name:12s} NOT FOUND')
"
```

---

## 3. How to Re-Run Official Evaluation

To reproduce the exact accuracy scores, confusion matrices, and per-task metrics without re-running 1,500 LLM inference queries:

```bash
# Execute official evaluation parser against candidate predictions:
python evaluation/evaluate_qa_official.py
```

This reads each candidate's raw output, parses answer keys and status gates, evaluates against ground-truth labels, and outputs comparative summaries matching the bakeoff report.

---

## 4. Re-running Live Benchmark Ingestion & Generation

If you have downloaded the full raw dataset `longmemeval_s_cleaned.json` and wish to generate new predictions from scratch:

1. Ensure Ollama is active with the required generator model:
   ```bash
   ollama serve
   ollama pull gpt-oss
   ```

2. Execute the candidate benchmark runner:
   ```bash
   # Run Candidate B pipeline on a test slice:
   python evaluation/run_final_benchmark.py --config candidate_b --limit 50
   ```

3. Run the official judge model (`qwen2.5:14b`):
   ```bash
   ollama pull qwen2.5:14b
   python evaluation/evaluators/official_judge.py --input-predictions <path-to-predictions.jsonl>
   ```

---

## 5. External Benchmark References

- **LongMemEval Reference Dataset:** Wu et al., *"LongMemEval: Benchmarking Chat Assistants on Long-Horizon Conversational Memory"*, ICLR 2025.
- Baseline figures for naive dense RAG (~40.80%) and full-context window degradation (30–45% needle drop) are cited from published results in Wu et al.
