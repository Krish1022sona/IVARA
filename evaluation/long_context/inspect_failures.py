import json

data = json.load(open('evaluation/long_context/baseline_results_multi_ivara.json', encoding='utf-8'))
cases = data['cases']

print("--- INSPECTING FAILURES & EDGE CASES ---")
for c in cases:
    if c['id'] in ['q37', 'q38', 'q39', 'q40', 'q01', 'q13', 'q15', 'q25']:
        print(f"\n[{c['id']}] ({c['category']}) Q: {c['question']}")
        print(f"  EXPECTED STATUS: {c['expected_status']} | ACTUAL: {c['actual_status']} | SCORE: {c['fact_score']}")
        print(f"  RETRIEVED IDs: {c['retrieved_memory_ids']} | SUPPORTING: {c['supporting_memory_ids']}")
        print(f"  GOLD ANSWER:   {c['gold_answer']}")
        print(f"  ACTUAL ANSWER: {c['actual_answer'].encode('ascii', errors='replace').decode()}")
        print(f"  PROMPT INJ FAIL: {c['prompt_injection_fail']}")
