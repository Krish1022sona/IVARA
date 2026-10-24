import json
from pathlib import Path

base_multi = json.load(open('evaluation/long_context/baseline_raw_extraction_multi.json', encoding='utf-8'))
curr_multi = json.load(open('evaluation/long_context/raw_extraction_multi.json', encoding='utf-8'))
memories_multi = json.load(open('evaluation/long_context/multi_episode/memories.json', encoding='utf-8'))

print("=== BASELINE MULTI RAW EXTRACTION ===")
print("Items count:", len(base_multi))
total_base_raw = 0
total_base_saved = 0
for i, item in enumerate(base_multi):
    raw_rels = len(item.get('raw_extraction', {}).get('relationships', []))
    saved_rels = len(item.get('memory', {}).get('relationships', []))
    total_base_raw += raw_rels
    total_base_saved += saved_rels
    print(f"Episode {i+1}: raw={raw_rels}, saved={saved_rels}")
print(f"Total Base: raw={total_base_raw}, saved={total_base_saved}")

print("\n=== CURRENT (POST-FIX) MULTI RAW EXTRACTION (raw_extraction_multi.json) ===")
print("Items count:", len(curr_multi))
total_curr_raw = 0
total_curr_saved = 0
for i, item in enumerate(curr_multi):
    raw_rels = len(item.get('raw_extraction', {}).get('relationships', []))
    saved_rels = len(item.get('memory', {}).get('relationships', []))
    total_curr_raw += raw_rels
    total_curr_saved += saved_rels
    print(f"Episode {i+1}: raw={raw_rels}, saved={saved_rels}")
print(f"Total Current: raw={total_curr_raw}, saved={total_curr_saved}")

print("\n=== STORED MEMORIES (multi_episode/memories.json) ===")
print("Memories count:", len(memories_multi))
total_stored_rels = 0
for i, m in enumerate(memories_multi):
    saved_rels = len(m.get('relationships', []))
    total_stored_rels += saved_rels
    print(f"Memory {m.get('id')}: saved={saved_rels}")
print(f"Total Stored in memories.json: {total_stored_rels}")

print("\n=== METRICS IN baseline_metrics_summary.json ===")
b_sum = json.load(open('evaluation/long_context/baseline_metrics_summary.json', encoding='utf-8'))
print(json.dumps(b_sum['multi_episode']['extraction'], indent=2))

print("\n=== METRICS IN metrics_summary.json ===")
c_sum = json.load(open('evaluation/long_context/metrics_summary.json', encoding='utf-8'))
print(json.dumps(c_sum['multi_episode']['extraction'], indent=2))
