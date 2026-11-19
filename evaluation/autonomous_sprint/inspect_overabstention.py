import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

with open("evaluation/autonomous_sprint/runs/finalization_sprint/experiment_40/detailed_17_audit.json", "r", encoding="utf-8") as f:
    items = json.load(f)

over_abs_qids = ["00ca467f", "10e09553", "50635ada", "b46e15ed", "bbf86515", "gpt4_93159ced"]

for it in items:
    if it["question_id"] in over_abs_qids:
        print("=" * 80)
        print(f"QID: {it['question_id']}")
        print(f"Q: {it['question']}")
        print(f"Gold: {it['gold_answer']}")
        print("Retrieved chunks text:")
        for i, txt in enumerate(it["retrieved_chunks_text"]):
            print(f"  Chunk {i+1}: {repr(txt[:300])}")
