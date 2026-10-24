import json

def show_summary(path, label):
    data = json.load(open(path, encoding='utf-8'))
    print(f"\n=== {label} (Overall Recall: {data['avg_retrieval_recall']}, Prec: {data['avg_retrieval_precision']}, Fact: {data['avg_fact_score']}, Abstain F1: {data['abstention']['f1']}) ===")
    print(f"{'Category':<32} | {'Recall':<6} | {'Prec':<6} | {'Fact':<6} | {'CiteVal':<7} | {'ClaimEnt':<8} | {'Fails':<5}")
    print("-" * 85)
    for cat, m in data['category_breakdown'].items():
        print(f"{cat:<32} | {m['ret_recall']:>6.2f} | {m['ret_prec']:>6.2f} | {m['fact_score']:>6.2f} | {m['citation_validity']:>7.2f} | {m['claim_entailment']:>8.2f} | {m['failures']:>5}")

print("*******************************************************************************")
print("                  POST-FIX BENCHMARK RESULTS (AFTER FIXES)                     ")
print("*******************************************************************************")
show_summary("evaluation/long_context/results_multi_ivara.json", "IVARA Multi-Episode (POST-FIX)")
show_summary("evaluation/long_context/results_multi_baseline.json", "Ordinary Vector Baseline Multi-Episode (POST-FIX)")
show_summary("evaluation/long_context/results_single_ivara.json", "IVARA Single-Episode (POST-FIX)")
show_summary("evaluation/long_context/results_single_baseline.json", "Ordinary Vector Baseline Single-Episode (POST-FIX)")

print("\n*******************************************************************************")
print("                 ORIGINAL BASELINE RESULTS (BEFORE FIXES)                      ")
print("*******************************************************************************")
show_summary("evaluation/long_context/baseline_results_multi_ivara.json", "IVARA Multi-Episode (BASELINE)")
show_summary("evaluation/long_context/baseline_results_multi_baseline.json", "Ordinary Vector Baseline Multi-Episode (BASELINE)")
