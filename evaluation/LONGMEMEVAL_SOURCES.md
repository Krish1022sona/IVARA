# LongMemEval Canonical Sources, Dataset Verification, and Protocol Audit

**Document ID:** `IVARA-EVAL-LME-SRC-01`  
**Date:** 2026-10-27  
**Status:** Verified Primary Sources  
**Benchmark:** LongMemEval (ICLR 2025)  

---

## 1. Canonical URLs & Primary References

| Resource | Canonical URL | Verified Version / Commit | Notes |
| :--- | :--- | :--- | :--- |
| **Official Paper** | [arXiv:2410.10813](https://arxiv.org/abs/2410.10813) | `v1` (Oct 2024), accepted at ICLR 2025 | *"LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"* |
| **Project Website** | [https://xiaowu0162.github.io/long-mem-eval/](https://xiaowu0162.github.io/long-mem-eval/) | Live | Interactive task visualizer & examples |
| **Official GitHub Repository** | [https://github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) | `main` branch | Official evaluation code, runners, generation and indexing scripts |
| **Official Cleaned Dataset (HF)** | [https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | `main` (Updated Sept 2025 to remove history session interference) | Contains `longmemeval_oracle.json`, `longmemeval_s_cleaned.json`, `longmemeval_m_cleaned.json` |
| **V2 Follow-Up Repository** | [https://github.com/xiaowu0162/LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) | `main` | Extended benchmark for enterprise & agentic settings |

---

## 2. Dataset Variants & Format Verification

Direct inspection of `longmemeval_oracle.json` (15.38 MB, 500 questions) and `longmemeval_s_cleaned.json` (264.53 MB, 500 questions) downloaded from the official Hugging Face repository confirms the following schema:

### 2.1 Released Dataset Files
1. **`longmemeval_oracle.json` (15.38 MB):**
   - Contains only the required ground-truth evidence sessions in `haystack_sessions` (no filler sessions).
   - Designed for isolating reader / answering capabilities from retrieval errors.
2. **`longmemeval_s_cleaned.json` (264.53 MB):**
   - Includes full haystacks of ~40-80 conversational sessions (~115k tokens) with filler sessions from ShareGPT and UltraChat.
   - Chronologically ordered timestamps.
3. **`longmemeval_m_cleaned.json`:**
   - Extended haystack containing ~500 history sessions per question.

### 2.2 Schema per Instance (500 instances per file)
Each JSON entry is an object with the following verified keys:
- `question_id`: Unique identifier (e.g., `gpt4_2655b836` or ending in `_abs` for unanswerable/abstention queries).
- `question_type`: Categorical label identifying the capability tested.
- `question`: Natural language question string.
- `answer`: Expected gold answer string or explanation.
- `question_date`: Temporal anchor formatted as `YYYY/MM/DD (Day) HH:MM` (e.g. `2023/04/10 (Mon) 23:07`).
- `haystack_dates`: List of timestamp strings for each session in `haystack_sessions`.
- `haystack_session_ids`: List of unique session IDs corresponding to each session.
- `haystack_sessions`: List of sessions, where each session is a list of turn objects:
  ```json
  [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "has_answer": true}
  ]
  ```
- `answer_session_ids`: Ground-truth list of session IDs containing required evidence (used for session-level recall evaluation).

### 2.3 Verified Question Breakdown (500 Total Questions)
Analysis of the official dataset reveals the exact breakdown across the 6 core tasks:
- **`temporal-reasoning`**: 133 questions (26.6%)
- **`multi-session`**: 133 questions (26.6%)
- **`knowledge-update`**: 78 questions (15.6%)
- **`single-session-user`**: 70 questions (14.0%)
- **`single-session-assistant`**: 56 questions (11.2%)
- **`single-session-preference`**: 30 questions (6.0%)
- **`abstention`**: Exactly 30 questions (identified deterministically by `'_abs' in entry['question_id']`).

---

## 3. Official Evaluation Protocol & Scoring Procedure

Inspection of `src/evaluation/evaluate_qa.py` and `src/evaluation/print_qa_metrics.py` from the official repository reveals the exact scoring mechanics:

### 3.1 LLM-as-a-Judge Architecture
- **Judge Model:** Officially uses `gpt-4o-2024-08-06` (`gpt-4o`) via the OpenAI API (or local `llama-3.1-70b-instruct` via vLLM).
- **Inference Parameters:** `temperature: 0`, `n: 1`, `max_tokens: 10`.
- **Decision Rule:** Binary check: `label = 'yes' in eval_response.lower()`.

### 3.2 Category-Specific Prompts
1. **Standard QA (`single-session-user`, `single-session-assistant`, `multi-session`):**
   > *"I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no."*
2. **Temporal Reasoning (`temporal-reasoning`):**
   > Extends standard prompt with an explicit off-by-one tolerance rule:
   > *"In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct."*
3. **Knowledge Updates (`knowledge-update`):**
   > Explicit rule for handling prior vs updated facts:
   > *"If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer."*
4. **Personalization / Preference (`single-session-preference`):**
   > Rubric satisfaction prompt:
   > *"The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly."*
5. **Abstention (`_abs` in `question_id`):**
   > Unanswerability check:
   > *"Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not."*

### 3.3 Metric Aggregation
Official companion script `print_qa_metrics.py` computes:
- **Task-averaged Accuracy:** Unweighted arithmetic mean across the 6 task accuracies.
- **Overall Accuracy:** Micro-average across all 500 questions.
- **Abstention Accuracy:** Accuracy on the 30 `_abs` questions.

---

## 4. Dependencies, Licensing & Access Restrictions

- **License:** Open research and evaluation use.
- **Access:** Direct HTTPS download from Hugging Face without gated registration or access tokens (`https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/...`).
- **Dependencies:** Standard Python libraries (`json`, `sys`, `pathlib`, `numpy`, `tqdm`), plus `openai` / `ollama` for LLM inference.

---

## 5. Implementation Decisions for IVARA

1. **Dual Scoring Capability:**
   - **Local / Deterministic Rubric Mode:** To enable offline, zero-cost continuous integration without requiring paid OpenAI keys, IVARA implements a local deterministic scoring fallback using exact substring, token overlap, and negative/abstention pattern matching.
   - **Official LLM Judge Mode:** Implements the verbatim official prompt templates (`get_anscheck_prompt`) targeting either an OpenAI client or a local judge model (e.g., Ollama `gpt-oss` / `llama3.1:70b`).
2. **Strict Isolation:**
   - Benchmark data is placed exclusively in `evaluation/data/longmemeval/`.
   - Dedicated SQLite databases and FAISS indices are created per run (`longmemeval_baseline_graph.db`).
   - Production stores (`data/memories.json`, `data/entities.json`) are protected by explicit path-assertion safeguards.
