# IVARA — Evaluation Methodology, Protocols & Reproducibility Guide

**Standardized Academic Benchmarking Protocols for Long-Horizon Conversational AI**

---

## 1. Benchmark Dataset Taxonomy & Attribution

IVARA is benchmarked using **LongMemEval** (*Wu et al., ICLR 2025*), the standard empirical benchmark for evaluating long-context autobiographical memory in conversational agents.

### Dataset Provenance
- **Official Source:** Hosted on Hugging Face at [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
- **Core Dataset (LongMemEval-S):** `longmemeval_s_cleaned.json` (264.5 MB, 500 questions, ~40–80 dialogue sessions per question). Excluded from Git tracking via `.gitignore` due to GitHub's 100 MB file limit.
- **Oracle Variant:** `longmemeval_oracle.json` (15.38 MB, 500 questions with only the gold evidence sessions provided).
- **Task Distribution (500 questions; 30 `_abs` negative controls pooled across 4 tasks):**
  1. *Single-Session User:* 70 questions (64 answerable, 6 `_abs`).
  2. *Single-Session Assistant:* 56 questions (all answerable).
  3. *Single-Session Preference:* 30 questions (all answerable).
  4. *Multi-Session Reasoning:* 133 questions (121 answerable, 12 `_abs`).
  5. *Knowledge Update (Supersession):* 78 questions (72 answerable, 6 `_abs`).
  6. *Temporal Reasoning:* 133 questions (127 answerable including 41 Category H multi-session temporal questions, 6 `_abs`).

---

## 2. Experimental Environment & Isolation

All benchmarks are executed under strictly local, reproducible conditions without external network dependencies:
- **Host Platform:** Local host machine with GPU acceleration.
- **Inference Server:** Local Ollama daemon running on `http://localhost:11434`.
- **Generator Model:** `gpt-oss` (8,192 token context window, greedy decoding, `temperature=0.0`).
- **Judge Model:** `qwen2.5:14b` (greedy decoding, `temperature=0.0`).
- **Store Isolation:** The evaluation runner (`evaluation/runners/run_longmemeval.py`) constructs ephemeral in-memory stores and temporary SQLite instances (`instance_{qid}_graph.db`) for every test instance. The production database (`data/conversations.db` and `data/memories.json`) is never modified during evaluation.

---

## 3. Evaluation Metrics & Definitions

1. **Overall Accuracy ($N=500$):** Proportion of all 500 questions evaluated as correct by the official LLM judge.
2. **Macro Task-Average Accuracy:** Unweighted arithmetic mean of accuracy across the 6 task categories:
   $$\text{Macro Avg} = \frac{1}{6} \sum_{i=1}^6 \text{Acc}_i$$
   *(Prevents over-representation of larger categories like Temporal Reasoning [133 Qs] and Multi-Session [125 Qs]).*
3. **Answerable Accuracy ($N=470$):** Accuracy computed strictly over the 470 questions with ground-truth answers (excluding the 30 negative control questions).
4. **Negative Control Abstention Precision ($N=30$):** Accuracy on unrecorded autobiographical facts where the system must explicitly decline to answer or state insufficient evidence.
5. **Retrieval Recall@k ($k=5$):** Proportion of questions where all necessary gold evidence sessions are present in the top-$k$ retrieved chunks.
6. **Evidence Coverage:** Proportion of required ground-truth evidence snippets captured in the prompt context.

---

## 4. LLM-as-a-Judge Evaluation Protocol

IVARA uses the official LongMemEval judge prompt with greedy decoding (`temperature=0.0`) on `qwen2.5:14b`:
- **Judge Input:** Ground-truth question, gold answer, and generated candidate response.
- **Decision Schema:**
  ```json
  {
    "is_correct": true,
    "reasoning": "Candidate answer matches the gold answer page number exactly."
  }
  ```
- **Robust Parsing:** The evaluation harness parses judge output using strict JSON parsing with regex fallback to prevent scoring crashes on malformed fence formatting.

---

## 5. Statistical Testing Methodology (McNemar Test)

For paired binary outcomes across the exact same 500 questions, independent $t$-tests and $z$-tests are methodologically invalid due to sample correlation. We employ the **McNemar test with Edwards' continuity correction**:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

Where:
- $b$: Questions correct in Candidate B but incorrect in Candidate E (discordant pairs favoring B).
- $c$: Questions correct in Candidate E but incorrect in Candidate B (discordant pairs favoring E).
- Degrees of freedom: $df = 1$.
- Significance threshold: $\alpha = 0.05$ (two-tailed).

---

## 6. Independent Cryptographic Reproducibility

To independently verify the integrity of the sealed evaluation outputs without re-running 15+ hours of GPU inference:

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
