import json

exp_dir = "evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40"
arms = ["ARM_1_BASELINE", "ARM_2_ONTOLOGY", "ARM_3_DIRECT_TEMPORAL", "ARM_4_UNIFIED_HYBRID"]

focus_qids = ["945e3d21", "gpt4_c27434e8", "gpt4_483dd43c", "00ca467f"]

preds = {}
judges = {}
for arm in arms:
    with open(f"{exp_dir}/predictions_{arm}.jsonl", "r", encoding="utf-8") as f:
        preds[arm] = {line["question_id"]: line for line in (json.loads(l) for l in f)}
    with open(f"{exp_dir}/judge_results_{arm}.jsonl", "r", encoding="utf-8") as f:
        judges[arm] = {line["question_id"]: line for line in (json.loads(l) for l in f)}

for qid in focus_qids:
    print("="*80)
    print(f"QUESTION ID: {qid}")
    p0 = preds["ARM_1_BASELINE"][qid]
    print(f"Task Type: {p0['task_type']}")
    print(f"Question: {p0['question']}")
    print(f"Gold Answer: {p0['gold_answer']}")
    print("-"*60)
    for arm in arms:
        p = preds[arm][qid]
        j = judges[arm][qid]
        print(f"[{arm}] Correct: {j['is_correct']} | Citations: {p['citations']}")
        print(f"  Hypothesis: {repr(p['hypothesis'][:160])}")
        print(f"  Judge Reasoning: {repr(j['reasoning'][:160])}")
