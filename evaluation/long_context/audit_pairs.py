import json

pairs = [
    ("multi_ivara", "evaluation/long_context/baseline_results_multi_ivara.json", "evaluation/long_context/results_multi_ivara.json"),
    ("multi_baseline", "evaluation/long_context/baseline_results_multi_baseline.json", "evaluation/long_context/results_multi_baseline.json"),
    ("single_ivara", "evaluation/long_context/baseline_results_single_ivara.json", "evaluation/long_context/results_single_ivara.json"),
    ("single_baseline", "evaluation/long_context/baseline_results_single_baseline.json", "evaluation/long_context/results_single_baseline.json"),
]

for label, base_path, curr_path in pairs:
    b_data = json.load(open(base_path, encoding='utf-8'))
    c_data = json.load(open(curr_path, encoding='utf-8'))
    
    print(f"\n==================== {label.upper()} ====================")
    print(f"BASELINE: Recall={b_data['avg_retrieval_recall']}, Prec={b_data['avg_retrieval_precision']}, Fact={b_data['avg_fact_score']}, Abstain F1={b_data['abstention']['f1']}")
    print(f"CURRENT : Recall={c_data['avg_retrieval_recall']}, Prec={c_data['avg_retrieval_precision']}, Fact={c_data['avg_fact_score']}, Abstain F1={c_data['abstention']['f1']}")
    
    # Check citation counts
    b_cites = sum(len(c['cited_memory_ids']) for c in b_data['cases'])
    c_cites = sum(len(c['cited_memory_ids']) for c in c_data['cases'])
    b_cite_cases = sum(1 for c in b_data['cases'] if c['cited_memory_ids'])
    c_cite_cases = sum(1 for c in c_data['cases'] if c['cited_memory_ids'])
    print(f"CITED CASES: Base={b_cite_cases}, Curr={c_cite_cases} | TOTAL CITED IDs: Base={b_cites}, Curr={c_cites}")
    
    # Are the answers identical or different?
    diff_answers = 0
    for bc, cc in zip(b_data['cases'], c_data['cases']):
        if bc['actual_answer'].strip() != cc['actual_answer'].strip():
            diff_answers += 1
    print(f"Questions with different answers: {diff_answers} / {len(b_data['cases'])}")
