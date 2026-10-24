import json

base = json.load(open('evaluation/long_context/baseline_results_multi_ivara.json', encoding='utf-8'))['category_breakdown']
curr = json.load(open('evaluation/long_context/results_multi_ivara.json', encoding='utf-8'))['category_breakdown']

print("Category                  | Base Fact | Curr Fact | Base Fails | Curr Fails")
print("-" * 65)
for k in base:
    bf = base[k]['fact_score']
    cf = curr[k]['fact_score']
    b_fails = base[k]['failures']
    c_fails = curr[k]['failures']
    print(f"{k:<25} | {bf:>9.2f} | {cf:>9.2f} | {b_fails:>10} | {c_fails:>10}")
