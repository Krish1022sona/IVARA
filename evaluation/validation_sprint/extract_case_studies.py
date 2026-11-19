import json

with open("evaluation/validation_sprint/results/predictions_baseline_hybrid_speaker_aware.json", "r", encoding="utf-8") as f:
    spk = json.load(f)
with open("evaluation/validation_sprint/results/predictions_ivara_full.json", "r", encoding="utf-8") as f:
    iv = json.load(f)

def summarize_cases(qids, label):
    print(f"\n=======================================================")
    print(f"  {label} (Sample {len(qids)})")
    print(f"=======================================================")
    for qid in qids:
        p_iv = iv[qid]
        p_spk = spk[qid]
        print(f"ID: {qid} | Cat: {p_iv['category']}")
        print(f"Q:  {p_iv['question']}")
        print(f"Exp: {p_iv['expected_answer']}")
        print(f"RAG:   [{p_spk['predicted_status']}] {p_spk['predicted_answer'][:100]}... (Cov: {p_spk['answer_metrics']['fact_coverage']}, Rec@5: {p_spk['retrieval_metrics']['recall_at_5']})")
        print(f"IVARA: [{p_iv['predicted_status']}] {p_iv['predicted_answer'][:100]}... (Cov: {p_iv['answer_metrics']['fact_coverage']}, Rec@5: {p_iv['retrieval_metrics']['recall_at_5']})")
        print("-" * 65)

# 1. RAG wins
rag_wins = [qid for qid, p in iv.items() if spk[qid]["answer_metrics"]["binary_accuracy"] >= 0.99 and p["answer_metrics"]["fact_coverage"] < 0.40][:5]
summarize_cases(rag_wins, "1. Conventional RAG Wins")

# 2. IVARA wins
iv_wins = [qid for qid, p in iv.items() if p["answer_metrics"]["binary_accuracy"] >= 0.99 and spk[qid]["answer_metrics"]["fact_coverage"] < 0.40][:5]
summarize_cases(iv_wins, "2. IVARA Full Wins")

# 3. Mutual Fails
mutual_fails = [qid for qid, p in iv.items() if p["answer_metrics"]["is_answerable"] and p["answer_metrics"]["binary_accuracy"] == 0.0 and spk[qid]["answer_metrics"]["binary_accuracy"] == 0.0][:5]
summarize_cases(mutual_fails, "3. Mutual Failures")

# 4. Over-abstentions
over_abs = [qid for qid, p in iv.items() if p["answer_metrics"]["is_answerable"] and p["answer_metrics"]["abstained"] and p["retrieval_metrics"]["recall_at_5"] >= 0.99][:5]
summarize_cases(over_abs, "4. Over-Abstentions (Relevant Context Retrieved but Abstained)")

