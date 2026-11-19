import sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')
from evaluation.adapters.longmemeval_adapter import LongMemEvalAdapter
from src.chunking.turn_chunker import TurnAwareChunker
from src.chunking.hybrid_pipeline import UnifiedCandidateRetriever

adapter = LongMemEvalAdapter('evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json')
instances, _ = adapter.load_and_validate()
inst = next(i for i in instances if i.question_id == '10e09553')
chunker = TurnAwareChunker(window_size=3, overlap=1)
chunks = chunker.chunk_all_sessions(inst.sessions)

retriever = UnifiedCandidateRetriever(chunks=chunks, use_ontology=True)
res, diag = retriever.retrieve(inst.question, k=10)
print('Retrieved top-10 chunks for 10e09553:')
for rank, r in enumerate(res):
    cid = r["chunk_id"]
    sid = r["session_id"]
    sc = r["score"]
    tx = r["text"][:80].replace("\n", " ")
    print(f"Rank {rank+1}: {cid} (score={sc}, sid={sid}) text={repr(tx)}")
