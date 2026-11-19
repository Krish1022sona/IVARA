# IVARA LongMemEval Evaluation Harness

This directory provides the official benchmark evaluation suite for testing IVARA on **LongMemEval** (ICLR 2025).

---

## 1. Dataset Setup

The LongMemEval datasets are officially hosted on Hugging Face at [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).

To download the datasets into the isolated benchmark directory:

```bash
mkdir -p evaluation/data/longmemeval
cd evaluation/data/longmemeval

# 1. Oracle Evidence Variant (15.38 MB, 500 questions) - Recommended for rapid baseline
curl -L -O https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json

# 2. Short Haystack Variant (264.53 MB, 500 questions, ~40-80 sessions/question)
curl -L -O https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# 3. Medium Haystack Variant (~500 sessions/question)
curl -L -O https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
```

---

## 2. Running Evaluations

The runner executes an isolated evaluation pass on IVARA without touching production data.

### 2.1 Basic Usage

```bash
# Run baseline Hybrid Retrieval (Config B) on LongMemEval Oracle
python evaluation/runners/run_longmemeval.py \
  --dataset evaluation/data/longmemeval/longmemeval_oracle.json \
  --output-dir evaluation/data/longmemeval/runs/oracle_config_b \
  --config B \
  --model gpt-oss \
  --k 5
```

### 2.2 CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Path to dataset JSON file | `evaluation/data/longmemeval/longmemeval_oracle.json` |
| `--output-dir` | Output directory for logs, summary, and isolated graph DBs | `evaluation/data/longmemeval/runs` |
| `--config` | Retrieval configuration: `A` (Vector), `B` (Hybrid), `D` (Graph-Assisted), `E` (Decomposition) | `B` |
| `--model` | Answering model in Ollama | `gpt-oss` |
| `--k` | Top-$k$ evidence chunks retrieved | `5` |
| `--limit` | Limit number of questions (useful for smoke tests or pilot batches) | `None` (all) |
| `--no-resume` | Disables resume logic; overwrites existing output results | `False` (resumes by default) |

### 2.3 Resuming Interrupted Runs
The runner automatically records each evaluated question atomically in `results_config_{CONFIG}.jsonl`. If a run is interrupted by hardware timeout or GPU restart, re-running the exact same command will automatically detect completed `question_id` entries and resume from the first uncompleted question without duplicate LLM calls.

---

## 3. Strict Safety & Production Isolation

The runner and adapter enforce strict path assertions:
1. **Production Store Immunity:** Attempts to point `--dataset` or `--output-dir` to `data/memories.json` or `data/entities.json` raise a fatal `ProductionStoreIsolationError`.
2. **Ephemeral Isolated Graph Stores:** Graph retrieval builds per-question SQLite databases (`instance_{qid}_graph.db`) under `--output-dir` and cleans them up automatically.
3. **Gold Answer Isolation:** Ground-truth `answer`, `answer_session_ids`, and `has_answer` flags are completely stripped during memory conversion. They are stored only in reference memory for post-hoc scoring.

---

## 4. Evaluation Metrics & Scoring Modes

The evaluator (`LongMemEvalEvaluator`) implements the verified official evaluation protocol:

1. **Local Deterministic Rules Mode (Default):**
   - Enables fast offline CI and local testing without external paid APIs.
   - Evaluates exact/semantic matches, numeric off-by-one tolerances on temporal queries, and negative refusal patterns on abstentions.
2. **Official LLM-as-a-Judge Mode:**
   - Evaluates predictions using the verbatim prompts from `xiaowu0162/LongMemEval/src/evaluation/evaluate_qa.py`.
   - Supports `gpt-4o` via OpenAI API or any local model via custom callable.

Metrics reported:
- **Task-averaged Accuracy:** Unweighted mean across the 6 tasks (`single-session-user`, `single-session-assistant`, `single-session-preference`, `multi-session`, `temporal-reasoning`, `knowledge-update`).
- **Overall Accuracy:** Total correct / Total evaluated.
- **Abstention Accuracy:** Accuracy on the 30 `_abs` questions.
