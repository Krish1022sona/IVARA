import json

with open("evaluation/validation_sprint/results/predictions_baseline_hybrid_speaker_aware.json", "r", encoding="utf-8") as f:
    spk = json.load(f)
with open("evaluation/validation_sprint/results/predictions_ivara_full.json", "r", encoding="utf-8") as f:
    iv = json.load(f)

categories = [
    "single_fact_retrieval", "multi_hop_reasoning", "entity_relationships",
    "temporal_reasoning", "historical_state_retrieval", "knowledge_updates",
    "contradiction_handling", "preferences_and_constraints", "negative_information",
    "unanswerable_questions", "cross_session_synthesis", "complex_multi_fact"
]

print("=== Category Breakdown: Fair RAG vs IVARA Full (N=250) ===")
header = f"{'Category':<30} | {'N':<4} | {'Fair RAG Cov':<13} | {'IVARA Cov':<13} | {'Fair RAG Acc':<13} | {'IVARA Acc':<13}"
print(header)
print("-" * len(header))
for cat in categories:
    spk_cat = [p for p in spk.values() if p["category"] == cat]
    iv_cat = [p for p in iv.values() if p["category"] == cat]
    n = len(spk_cat)
    if n == 0:
        continue
    c_spk = sum(p["answer_metrics"]["fact_coverage"] for p in spk_cat) / n
    c_iv = sum(p["answer_metrics"]["fact_coverage"] for p in iv_cat) / n
    b_spk = sum(p["answer_metrics"]["binary_accuracy"] for p in spk_cat) / n
    b_iv = sum(p["answer_metrics"]["binary_accuracy"] for p in iv_cat) / n
    print(f"{cat:<30} | {n:<4} | {c_spk*100:>11.2f}% | {c_iv*100:>11.2f}% | {b_spk*100:>11.2f}% | {b_iv*100:>11.2f}%")

