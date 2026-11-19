# IVARA — Evaluation Suite & Benchmark Index

Welcome to the IVARA benchmark evaluation suite. This directory contains the harnesses, adapters, sealed experimental outputs, and authoritative research documentation for reproducing IVARA's empirical results on **LongMemEval** (*Wu et al., ICLR 2025*).

---

## Authoritative Research Documentation Index

The evaluation findings and methodology are organized into five canonical documents:

1. [**`README.md`**](README.md)  
   Benchmark overview, directory architecture, quick-start reproduction instructions, and cryptographic verification scripts.

2. [**`FINAL_RESULTS.md`**](FINAL_RESULTS.md)  
   Complete 500-question LongMemEval-S benchmark results, task-by-task breakdown, Category H relative calendar time performance, paired McNemar statistical significance tests ($\chi^2 = 67.58, p = 2.02 \times 10^{-16}$), latency analysis, and SHA-256 cryptographic verification of sealed prediction files.

3. [**`ABLATION_STUDIES.md`**](ABLATION_STUDIES.md)  
   Controlled 4-arm temporal ablation isolating the +5.26 pp gain from deterministic calendar synthesis vs. 0.00 pp from event graph traversal, turn chunking vs. whole-session embedding dilution, ontology concept expansion, session-aware MMR diversity filtering ($\lambda = 0.70$), and Phase 3.1 session routing analysis.

4. [**`METHODOLOGY.md`**](METHODOLOGY.md)  
   Dataset provenance and attribution, local isolated evaluation environments, metric definitions (Overall Accuracy, Macro Task Average, Recall@5, Evidence Coverage), official LLM-as-a-judge protocol (`qwen2.5:14b` at temperature 0.0), and cryptographic hash verification scripts.

5. [**`LIMITATIONS_AND_FAILURES.md`**](LIMITATIONS_AND_FAILURES.md)  
   In-depth forensic failure mode taxonomy across non-passing instances (multi-session dispersion $\ge 3$ sessions, calendar arithmetic, implicit preference supersession), documented negative findings, and explicit architectural non-claims.

---

## 1. Quick Start: Reproducing Benchmarks

### Prerequisites
- Python 3.10–3.13 with dependencies installed:
  ```bash
  pip install -r requirements-dev.txt
  ```
- Local Ollama daemon running with required models:
  ```bash
  ollama serve
  ollama pull gpt-oss
  ollama pull qwen2.5:14b
  ```

### Step 1: Download Benchmark Datasets
The benchmark datasets are officially hosted on Hugging Face at [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).

To download into the benchmark data directory:
```bash
mkdir -p evaluation/data/longmemeval
cd evaluation/data/longmemeval

# 1. Oracle Evidence Variant (15.38 MB, 500 questions) - Rapid baseline:
curl -L -O https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json

# 2. Full Haystack Variant (264.53 MB, 500 questions, 40-80 sessions per question):
curl -L -O https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
cd ../../..
```

### Step 2: Run Benchmark Inference
To run the full 500-question evaluation with Candidate B (winning architecture):
```bash
python -m evaluation.runners.run_longmemeval \
    --config B \
    --dataset evaluation/data/longmemeval/longmemeval_s_cleaned.json \
    --output evaluation/data/longmemeval/runs/my_candidate_b_run
```

### Step 3: Run the Official LLM Judge
Score the generated predictions using the official evaluation judge prompt:
```bash
python -m evaluation.runners.run_official_llm_judge \
    --predictions evaluation/data/longmemeval/runs/my_candidate_b_run/results_config_B.jsonl \
    --output evaluation/data/longmemeval/runs/my_candidate_b_run/judge_results.jsonl
```

---

## 2. Directory Structure

```text
evaluation/
├── README.md                      # This evaluation index and reproduction guide
├── FINAL_RESULTS.md               # Authoritative 500-question bakeoff performance
├── ABLATION_STUDIES.md            # 4-arm temporal ablation & component isolations
├── METHODOLOGY.md                 # Dataset provenance, metrics, judge, & protocols
├── LIMITATIONS_AND_FAILURES.md    # Failure taxonomy, negative findings, & boundaries
├── adapters/                      # LongMemEval dataset parser and turn adapters
│   └── longmemeval_adapter.py     # Converts raw benchmark JSON into turn chunks
├── evaluators/                    # Evaluation logic and metrics calculations
│   └── longmemeval_evaluator.py   # Heuristic and LLM judge scoring evaluators
├── runners/                       # Executable benchmark and ablation runners
│   ├── run_longmemeval.py         # Primary LongMemEval benchmark runner
│   ├── run_chunked_ablation.py    # Chunking window ablation runner
│   ├── run_diagnostic_sample.py   # Stratified diagnostic sample runner
│   ├── run_graph_ablation.py      # Knowledge graph traversal ablation runner
│   ├── run_held_out_ablation.py   # Held-out split ablation runner
│   └── run_official_llm_judge.py  # Standalone LLM judge scoring runner
└── data/                          # Benchmark datasets and sealed bakeoff outputs
    └── longmemeval/
        └── runs/
            └── final_bakeoff/     # Sealed predictions and official judge decisions
                ├── candidate_b_arm_b/       # Candidate B (Winner, 58.40%)
                ├── candidate_c_phase3/      # Candidate C (Adaptive Router, 56.80%)
                └── candidate_e_baseline/    # Candidate E (Baseline RAG, 40.80%)
```

---

## 3. Cryptographic Verification of Sealed Results

To verify the integrity of the sealed evaluation outputs without re-running 15+ hours of GPU inference:

```bash
python -c "
import hashlib
from pathlib import Path

files = {
    'Candidate B': 'evaluation/data/longmemeval/runs/final_bakeoff/candidate_b_arm_b/results_config_D.jsonl',
    'Candidate C': 'evaluation/data/longmemeval/runs/final_bakeoff/candidate_c_phase3/results_config_D.jsonl',
    'Candidate E': 'evaluation/data/longmemeval/runs/final_bakeoff/candidate_e_baseline/results_config_D.jsonl',
}

expected = {
    'Candidate B': 'ba9e71bec593c71f76d802fe154d21321776defd9fc808f6ade735867e90a9b9',
    'Candidate C': '58062320b1c0424d78aa90c84a79866ccbfd33bb233ee1e028b0e4a2ee04a2ee',
    'Candidate E': '914227ae70df1b8e00ba421492249e3b65d22fc0acf477b784a81ff8503ae3a3',
}

for name, path in files.items():
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert digest == expected[name], f'{name} hash mismatch!'
    print(f'{name}: Verified SHA-256 [{digest}]')
"
```
