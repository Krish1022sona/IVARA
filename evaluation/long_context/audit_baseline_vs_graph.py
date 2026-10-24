import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
BASELINE_FILE = BASE_DIR / "corrected_baseline_results_multi_ivara.json"
GRAPH_FILE = BASE_DIR / "graph_benchmark_results.json"
QUESTIONS_FILE = BASE_DIR / "questions.json"

with open(BASELINE_FILE, "r", encoding="utf-8") as f:
    base_data = json.load(f)
base_cases = {c["id"]: c for c in base_data["cases"]}

with open(GRAPH_FILE, "r", encoding="utf-8") as f:
    graph_data = json.load(f)
graph_cases = {c["id"]: c for c in graph_data["end_to_end_answering"]["cases"]}

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = {q["id"]: q for q in json.load(f)["questions"]}

print("==================================================================")
print("1. OVERALL METRIC RECONCILIATION")
print("==================================================================")
base_passed = sum(1 for c in base_cases.values() if c["rubric_passed"])
graph_passed = sum(1 for c in graph_cases.values() if c["rubric_passed"])
total = len(questions)

print(f"Total questions: {total}")
print(f"Baseline Rubric Passed: {base_passed} / {total} = {base_passed/total*100:.2f}% (reported as 79.2%)")
print(f"Graph Config E Passed:   {graph_passed} / {total} = {graph_passed/total*100:.2f}% (reported as 91.7%)")

# Denominator check: Answerable only (44) vs All (48)
ans_ids = [qid for qid, q in questions.items() if not q.get("should_abstain", False)]
unans_ids = [qid for qid, q in questions.items() if q.get("should_abstain", False)]
print(f"\nDenominator Breakdown:")
print(f"  Answerable questions:   {len(ans_ids)}")
print(f"  Unanswerable questions: {len(unans_ids)}")

base_ans_passed = sum(1 for qid in ans_ids if base_cases[qid]["rubric_passed"])
base_unans_passed = sum(1 for qid in unans_ids if base_cases[qid]["rubric_passed"])
graph_ans_passed = sum(1 for qid in ans_ids if graph_cases[qid]["rubric_passed"])
graph_unans_passed = sum(1 for qid in unans_ids if graph_cases[qid]["rubric_passed"])

print(f"  Baseline Answerable Accuracy:   {base_ans_passed}/{len(ans_ids)} = {base_ans_passed/len(ans_ids)*100:.2f}%")
print(f"  Baseline Unanswerable Accuracy: {base_unans_passed}/{len(unans_ids)} = {base_unans_passed/len(unans_ids)*100:.2f}%")
print(f"  Graph Config E Answerable Acc:  {graph_ans_passed}/{len(ans_ids)} = {graph_ans_passed/len(ans_ids)*100:.2f}%")
print(f"  Graph Config E Unanswerable Acc:{graph_unans_passed}/{len(unans_ids)} = {graph_unans_passed/len(unans_ids)*100:.2f}%")

print("\n==================================================================")
print("2. DISCREPANCY ANALYSIS: ALL 10 BASELINE FAILURES")
print("==================================================================")
base_failures = [qid for qid in sorted(questions.keys()) if not base_cases[qid]["rubric_passed"]]
print(f"Baseline failed on {len(base_failures)} questions: {base_failures}")

for qid in base_failures:
    b = base_cases[qid]
    g = graph_cases[qid]
    q = questions[qid]
    status_str = "FIXED IN GRAPH (PASS)" if g["rubric_passed"] else "STILL FAILED IN GRAPH"
    print(f"\n[{qid}] Category: {q['category']} | Status: {status_str}")
    print(f"  Question: {q['question']}")
    print(f"  Gold Answer: {q['gold_answer']}")
    print(f"  Baseline Answer: {b['actual_answer'][:120]}... [Status: {b['actual_status']}]")
    print(f"  Baseline Reason: {b['rubric_reason']}")
    print(f"  Graph Answer:    {g['actual_answer'][:120]}... [Status: {g['actual_status']}]")
    print(f"  Graph Reason:    {g['rubric_reason']}")
    print(f"  Baseline Retrieved IDs: {b.get('retrieved_memory_ids')}")
    print(f"  Graph Retrieved IDs:    {g.get('retrieved_memory_ids')}")

print("\n==================================================================")
print("3. GRAPH FAILURES (4 QUESTIONS)")
print("==================================================================")
graph_failures = [qid for qid in sorted(questions.keys()) if not graph_cases[qid]["rubric_passed"]]
print(f"Graph Config E failed on {len(graph_failures)} questions: {graph_failures}")
for qid in graph_failures:
    g = graph_cases[qid]
    q = questions[qid]
    print(f"\n[{qid}] Category: {q['category']}")
    print(f"  Question: {q['question']}")
    print(f"  Gold Answer: {q['gold_answer']}")
    print(f"  Actual Answer: {g['actual_answer']}")
    print(f"  Status: {g['actual_status']} | Reason: {g['rubric_reason']}")
    print(f"  Retrieved IDs: {g['retrieved_memory_ids']}")
