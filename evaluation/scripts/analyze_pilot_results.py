import json
from pathlib import Path

pilot_dir = Path("evaluation/data/longmemeval/runs/campaign_v1_pilot")
res_a = {r["question_id"]: r for r in [json.loads(line) for line in open(pilot_dir / "config_a/results_config_CONFIG-A.jsonl", encoding="utf-8")]}
res_b = {r["question_id"]: r for r in [json.loads(line) for line in open(pilot_dir / "config_b/results_config_CONFIG-B.jsonl", encoding="utf-8")]}
res_c = {r["question_id"]: r for r in [json.loads(line) for line in open(pilot_dir / "config_c/results_config_CONFIG-C.jsonl", encoding="utf-8")]}
res_d = {r["question_id"]: r for r in [json.loads(line) for line in open(pilot_dir / "config_d/results_config_CONFIG-D.jsonl", encoding="utf-8")]}

print("=== Task-level accuracy breakdown ===")
tasks = sorted(list({r["question_type"] for r in res_a.values()}))
for t in tasks:
    qids = [q for q, r in res_a.items() if r["question_type"] == t]
    acc_a = sum(1 for q in qids if res_a[q]["local_heuristic_score"] is True) / len(qids) * 100
    acc_b = sum(1 for q in qids if res_b[q]["local_heuristic_score"] is True) / len(qids) * 100
    acc_c = sum(1 for q in qids if res_c[q]["local_heuristic_score"] is True) / len(qids) * 100
    acc_d = sum(1 for q in qids if res_d[q]["local_heuristic_score"] is True) / len(qids) * 100
    print(f"{t:30s} | A: {acc_a:5.1f}% | B: {acc_b:5.1f}% | C: {acc_c:5.1f}% | D: {acc_d:5.1f}% (N={len(qids)})")

print("\n=== Paired Differences: D vs A ===")
for qid in res_a:
    sa = res_a[qid]["local_heuristic_score"]
    sd = res_d[qid]["local_heuristic_score"]
    if sa != sd:
        is_abs = res_a[qid]["is_abstention"]
        print(f"[{qid}] ({res_a[qid]['question_type']}) abs={is_abs}: A={sa} -> D={sd}")
        print(f"  Q: {res_a[qid]['question']}".encode("ascii", "replace").decode("ascii"))
        print(f"  Ans A: {res_a[qid]['hypothesis']}".encode("ascii", "replace").decode("ascii"))
        print(f"  Ans D: {res_d[qid]['hypothesis']}".encode("ascii", "replace").decode("ascii"))

print("\n=== Paired Differences: D vs C ===")
for qid in res_c:
    sc = res_c[qid]["local_heuristic_score"]
    sd = res_d[qid]["local_heuristic_score"]
    if sc != sd:
        is_abs = res_c[qid]["is_abstention"]
        print(f"[{qid}] ({res_c[qid]['question_type']}) abs={is_abs}: C={sc} -> D={sd}")
        print(f"  Q: {res_c[qid]['question']}".encode("ascii", "replace").decode("ascii"))
        print(f"  Ans C: {res_c[qid]['hypothesis']}".encode("ascii", "replace").decode("ascii"))
        print(f"  Ans D: {res_d[qid]['hypothesis']}".encode("ascii", "replace").decode("ascii"))

print("\n=== Paired Differences: C vs B ===")
for qid in res_b:
    sb = res_b[qid]["local_heuristic_score"]
    sc = res_c[qid]["local_heuristic_score"]
    if sb != sc:
        is_abs = res_b[qid]["is_abstention"]
        print(f"[{qid}] ({res_b[qid]['question_type']}) abs={is_abs}: B={sb} -> C={sc}")

print("\n=== Paired Differences: B vs A ===")
for qid in res_a:
    sa = res_a[qid]["local_heuristic_score"]
    sb = res_b[qid]["local_heuristic_score"]
    if sa != sb:
        is_abs = res_a[qid]["is_abstention"]
        print(f"[{qid}] ({res_a[qid]['question_type']}) abs={is_abs}: A={sa} -> B={sb}")
