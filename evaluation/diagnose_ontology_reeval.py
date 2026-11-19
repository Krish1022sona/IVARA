import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
from pathlib import Path
from src.ontology.validator import (
    validate_extraction,
    validate_extracted_relationship,
    validate_resolved_relationship
)
from src.ontology.vocabulary import validate_predicate, get_canonical_predicate
from src.ontology.relationship_resolver import resolve_relationship
from src.graph.store import SQLiteGraphStore
from src.graph.builder import MemoryGraphBuilder

memories_path = Path("evaluation/validation_sprint/ivara_store_validation/memories.json")
with open(memories_path, "r", encoding="utf-8") as f:
    mems = json.load(f)

print(f"Total memories loaded: {len(mems)}")

total_attempted_rels = 0
legacy_accepted_rels = 0
new_accepted_rels = 0
legacy_mems_with_rels = sum(1 for m in mems if m.get("relationships"))

# We will collect reconstructed extractions for all 54 memories
reeval_memories = []

for m in mems:
    text = m.get("text", "")
    diag = m.get("diagnostics", {})
    existing_rels = m.get("relationships", [])
    invalid_rels = diag.get("invalid_extracted_relationships", [])
    
    legacy_accepted_rels += len(existing_rels)
    
    # Collect all candidate relationships originally attempted
    attempted_rels = []
    # 1. Existing accepted rels
    for r in existing_rels:
        raw_r = {
            "source": r["source"].get("canonical_name") or r["source"].get("mention") or r["source"].get("id"),
            "relation": r.get("predicate"),
            "target": r["target"].get("canonical_name") or r["target"].get("mention") or r["target"].get("id"),
            "confidence": r.get("confidence", 1.0),
            "origin": r.get("origin", "explicit"),
            "evidence": r.get("evidence", text)
        }
        attempted_rels.append(raw_r)
        
    # 2. Invalid extracted rels recorded in diagnostics
    for r_entry in invalid_rels:
        r = r_entry[0] if isinstance(r_entry, list) else r_entry
        attempted_rels.append(r)
        
    total_attempted_rels += len(attempted_rels)
    
    # Collect all entities (from m['entities'] and extracted errors/targets)
    entities = []
    seen_ent_names = set()
    for e in m.get("entities", []):
        name = e.get("mention") or e.get("name") or e.get("canonical_name")
        if name and name not in seen_ent_names:
            seen_ent_names.add(name)
            entities.append({
                "name": name,
                "type": (e.get("types") or [e.get("type", "Entity")])[0],
                "confidence": e.get("confidence", 1.0)
            })
            
    # Add any relationship endpoints that were missing from entity list
    for r in attempted_rels:
        src = r.get("source")
        tgt = r.get("target")
        if src and src not in seen_ent_names:
            seen_ent_names.add(src)
            entities.append({"name": src, "type": "Entity", "confidence": 1.0})
        if tgt and tgt not in seen_ent_names:
            seen_ent_names.add(tgt)
            # Check if it looks like a Date
            t_type = "Date" if any(char.isdigit() for char in tgt) else "Entity"
            entities.append({"name": tgt, "type": t_type, "confidence": 1.0})
            
    # Now run validation on attempted_rels using our new Open-World Ontology
    # In open-world mode, pass rules with open_world_predicates=True
    valid_new_rels = []
    rejected_new_rels = []
    
    # Build resolved entity mock objects
    resolved_entities = []
    for e in entities:
        ent_id = "self" if e["name"] == "SELF" else f"ent_{e['name'].lower().replace(' ', '_')}"
        resolved_entities.append({
            "mention": e["name"],
            "entity_id": ent_id,
            "canonical_name": e["name"],
            "types": [e["type"]],
            "confidence": e["confidence"],
            "status": "matched"
        })
        
    for r in attempted_rels:
        # Validate extracted relationship
        val_res = validate_extracted_relationship(
            r,
            memory_text=text,
            rules={"open_world_predicates": True}
        )
        if val_res.is_valid:
            # Resolve relationship
            resolved = resolve_relationship(r, resolved_entities)
            if resolved and resolved.get("status") == "resolved":
                # Validate resolved relationship
                res_val = validate_resolved_relationship(resolved, memory_text=text)
                if res_val.is_valid:
                    clean_rel = {k: v for k, v in resolved.items() if k not in ("status", "resolved")}
                    valid_new_rels.append(clean_rel)
                else:
                    rejected_new_rels.append((r, res_val.errors))
            else:
                rejected_new_rels.append((r, ["Unresolved endpoints"]))
        else:
            rejected_new_rels.append((r, val_res.errors))
            
    new_accepted_rels += len(valid_new_rels)
    
    reeval_memories.append({
        "id": m["id"],
        "text": text,
        "timestamp": m.get("timestamp"),
        "source": m.get("source"),
        "type": m.get("type", "episodic"),
        "importance": m.get("importance", 0.5),
        "entities": resolved_entities,
        "relationships": valid_new_rels,
        "rejected_count": len(rejected_new_rels),
        "rejected_rels": rejected_new_rels
    })

new_mems_with_rels = sum(1 for m in reeval_memories if m.get("relationships"))

print("\n================== ONTOLOGY RE-EVALUATION SUMMARY ==================")
print(f"Total memories: {len(mems)}")
print(f"Total candidate relationships attempted: {total_attempted_rels}")
print(f"Baseline (Closed-World) accepted relationships: {legacy_accepted_rels}")
print(f"Baseline (Closed-World) memories with relationships: {legacy_mems_with_rels}/{len(mems)} ({legacy_mems_with_rels/len(mems)*100:.1f}%)")
print(f"Baseline (Closed-World) relationship-loss: {len(mems) - legacy_mems_with_rels}/{len(mems)} ({((len(mems) - legacy_mems_with_rels)/len(mems))*100:.1f}%)")
print("--------------------------------------------------------------------")
print(f"Open-World accepted relationships: {new_accepted_rels}")
print(f"Open-World memories with relationships: {new_mems_with_rels}/{len(mems)} ({new_mems_with_rels/len(mems)*100:.1f}%)")
print(f"Open-World relationship recovery: +{new_accepted_rels - legacy_accepted_rels} relationships (+{(new_accepted_rels - legacy_accepted_rels)/max(1, total_attempted_rels)*100:.1f}% yield)")
print("====================================================================")

# Now test building SQLiteGraphStore with re-evaluated memories in a temporary DB
test_graph_path = Path("evaluation/validation_sprint/results/reeval_graph_test.db")
test_graph_path.parent.mkdir(parents=True, exist_ok=True)
if test_graph_path.exists():
    test_graph_path.unlink()

test_store = SQLiteGraphStore(test_graph_path)
builder = MemoryGraphBuilder(test_store)
graph_stats = builder.build_from_records(reeval_memories)

print("\n================== GRAPH PERSISTENCE VERIFICATION =================")
print(f"Nodes created: {graph_stats['nodes_count']}")
print(f"Edges created: {graph_stats['edges_count']}")

# Sample graph edges
import sqlite3
conn = sqlite3.connect(test_graph_path)
c = conn.cursor()
c.execute("SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type ORDER BY COUNT(*) DESC")
print("\nEdge distribution by type:")
for edge_type, count in c.fetchall():
    print(f"  {edge_type}: {count}")

c.execute("SELECT source_id, edge_type, target_id FROM edges WHERE edge_type NOT IN ('contains_chunk', 'mentions_entity') LIMIT 20")
print("\nSample semantic relationships in graph:")
for s, e, t in c.fetchall():
    print(f"  {s} --[{e}]--> {t}")

conn.close()
del test_store
import gc
gc.collect()
if test_graph_path.exists():
    try:
        test_graph_path.unlink()
    except Exception:
        pass
