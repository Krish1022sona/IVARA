import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
from evaluation.adapters.longmemeval_adapter import LongMemEvalAdapter
from src.chunking.turn_chunker import TurnAwareChunker
from src.chunking.hybrid_pipeline import UnifiedCandidateRetriever
from src.answering import answer_question

adapter = LongMemEvalAdapter('evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json')
instances, _ = adapter.load_and_validate()
inst_map = {i.question_id: i for i in instances}
chunker = TurnAwareChunker(window_size=3, overlap=1)

inst = inst_map['10e09553']
chunks = chunker.chunk_all_sessions(inst.sessions)
retriever = UnifiedCandidateRetriever(chunks=chunks, use_ontology=True)
res, _ = retriever.retrieve(inst.question, k=10)

# Diverse selection: take chunk objects with max 1 per session
diverse_chunk_objs = []
seen_sids = set()
for r in res:
    if r['session_id'] not in seen_sids:
        diverse_chunk_objs.append(r['chunk'])
        seen_sids.add(r['session_id'])
    if len(diverse_chunk_objs) == 3:
        break

ans = answer_question(
    query=inst.question,
    chunks=diverse_chunk_objs,
    use_chunking=True,
    model='gpt-oss:latest',
    dynamic_graph_fallback=True,
    direct_seed_temporal=True
)

print('=' * 60)
print('Q:', inst.question)
print('Gold Answer:', inst.gold_answer)
print('Status:', ans['status'])
print('Answer:', ans['answer'])
print('Citations:', [c.get('chunk_id') for c in ans.get('citations', [])])
