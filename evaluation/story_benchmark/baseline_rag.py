"""Standalone Conventional RAG and Hybrid Baselines for Story Benchmark.

Implements:
1. Conventional Vector RAG (pure FAISS dense retrieval over narrative dialogue turns).
2. Conventional Hybrid RAG (Dense Vector + BM25 Lexical with RRF, without graph or ontology).

Uses:
- Same embedding model (SentenceTransformer all-MiniLM-L6-v2)
- Same generation model (gpt-oss via Ollama)
- Same prompt isolation & AnswerPayload structured schema
- Independent in-memory index; zero access to IVARA knowledge graph, entity stores, or temporal claims.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Optional
import faiss
import numpy as np

from src.embedding import get_embedding

try:
    from ollama import chat as ollama_chat
except ImportError:
    ollama_chat = None


RAG_SYSTEM_PROMPT = """You are an evidence-grounded answering assistant.
Your task is to answer user questions using ONLY the provided dialogue context.

STRICT GROUNDING RULES:
1. Answer using ONLY facts directly stated or logically inferred in the provided evidence.
2. NEVER invent personal facts, dates, project details, or colleague names not in the evidence.
3. If the evidence only partially answers the question, state what is known and mention what is missing.
4. If the provided evidence is completely irrelevant or contains no pertinent facts, set status to 'insufficient_evidence'.
5. In cited_chunk_ids, include all chunk IDs (e.g. ['c01_03']) from the evidence that support your answer.
6. Keep internal reasoning concise and immediately output the structured JSON.
"""


class SimpleBM25:
    """Lightweight BM25 lexical retriever for the baseline."""
    def __init__(self, corpus: list[dict[str, Any]]):
        self.corpus = corpus
        self.doc_count = len(corpus)
        self.avg_doc_len = 0.0
        self.doc_tokens = []
        self.doc_freqs = {}

        if self.doc_count > 0:
            total_len = 0
            for doc in corpus:
                tokens = self._tokenize(doc.get("text", ""))
                self.doc_tokens.append(tokens)
                total_len += len(tokens)
                for t in set(tokens):
                    self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1
            self.avg_doc_len = total_len / self.doc_count

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in re.findall(r'\b[a-zA-Z0-9_\-+#]{2,}\b', text)]

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, int]]:
        q_tokens = self._tokenize(query)
        if not q_tokens or self.doc_count == 0:
            return []
        scores = []
        k1 = 1.5
        b = 0.75
        for idx, doc_toks in enumerate(self.doc_tokens):
            doc_len = len(doc_toks)
            score = 0.0
            tok_counts = {}
            for t in doc_toks:
                tok_counts[t] = tok_counts.get(t, 0) + 1
            for qt in q_tokens:
                if qt in tok_counts:
                    df = self.doc_freqs.get(qt, 1)
                    idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
                    tf = tok_counts[qt]
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / (self.avg_doc_len or 1.0))))
            if score > 0.0:
                scores.append((score, idx))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]


class ConventionalRAGBaseline:
    """
    Standard Vector & Hybrid RAG Baseline for narrative questioning.
    """
    def __init__(
        self,
        sessions: list[dict[str, Any]],
        model: str = "gpt-oss",
        llm_caller: Optional[Callable[[str, str], Any]] = None
    ):
        self.sessions = sessions
        self.model = model
        self.llm_caller = llm_caller
        self.chunks: list[dict[str, Any]] = []
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bm25: Optional[SimpleBM25] = None

        self._build_chunks()
        self._build_indices()

    def _build_chunks(self):
        """Chunk sessions into discrete dialogue units with date and session metadata."""
        chunk_idx = 0
        for s in self.sessions:
            s_id = s.get("session_id", "session_00")
            s_date = s.get("date", "2024-01-01")
            turns = s.get("turns", [])
            # Chunk user-assistant turn pairs or individual user turns
            for i in range(len(turns)):
                t = turns[i]
                if t.get("role") == "user":
                    user_text = t.get("content", "").strip()
                    # Check if next turn is assistant response
                    asst_text = ""
                    if i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
                        asst_text = turns[i + 1].get("content", "").strip()
                    chunk_text = f"[{s_date}] User: {user_text}"
                    if asst_text:
                        chunk_text += f"\nAssistant: {asst_text}"
                    c_id = f"c_{s_id}_{chunk_idx:03d}"
                    self.chunks.append({
                        "chunk_id": c_id,
                        "session_id": s_id,
                        "date": s_date,
                        "text": chunk_text,
                        "user_text": user_text
                    })
                    chunk_idx += 1

    def _build_indices(self):
        """Build FAISS vector index and BM25 index."""
        if not self.chunks:
            return

        texts = [c["text"] for c in self.chunks]
        embeddings = [get_embedding(t) for t in texts]
        arr = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(arr)

        dim = arr.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(arr)

        self.bm25 = SimpleBM25(self.chunks)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: str = "vector"  # 'vector' or 'hybrid'
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k chunks.
        mode='vector': pure FAISS cosine similarity.
        mode='hybrid': FAISS + BM25 fused via Reciprocal Rank Fusion (k=60).
        """
        clean_q = query.strip()
        if not clean_q or not self.chunks or self.faiss_index is None:
            return []

        # Vector retrieval
        q_emb = np.array([get_embedding(clean_q)], dtype="float32")
        faiss.normalize_L2(q_emb)
        search_k = min(15, len(self.chunks))
        scores, indices = self.faiss_index.search(q_emb, search_k)

        vec_hits = {}
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx >= 0 and idx < len(self.chunks):
                vec_hits[idx] = (rank, float(score))

        if mode == "vector":
            selected = []
            for idx in list(vec_hits.keys())[:k]:
                rank, score = vec_hits[idx]
                chunk = dict(self.chunks[idx])
                chunk["score"] = round(score, 4)
                chunk["rank"] = rank
                selected.append(chunk)
            return selected

        # Hybrid retrieval (Vector + BM25 RRF)
        lex_hits = {}
        if self.bm25:
            bm_results = self.bm25.search(clean_q, top_k=15)
            for rank, (score, idx) in enumerate(bm_results, 1):
                lex_hits[idx] = (rank, score)

        rrf_scores = {}
        all_candidates = set(vec_hits.keys()) | set(lex_hits.keys())
        k_rrf = 60
        for idx in all_candidates:
            v_rrf = (1.0 / (k_rrf + vec_hits[idx][0])) if idx in vec_hits else 0.0
            l_rrf = (1.0 / (k_rrf + lex_hits[idx][0])) if idx in lex_hits else 0.0
            rrf_scores[idx] = v_rrf + l_rrf

        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        selected = []
        for rank, idx in enumerate(sorted_indices, 1):
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(rrf_scores[idx], 5)
            chunk["rank"] = rank
            selected.append(chunk)
        return selected

    def answer_question(
        self,
        query: str,
        k: int = 5,
        mode: str = "vector",
        min_relevance: float = 0.30
    ) -> dict[str, Any]:
        """Retrieve context and generate grounded answer."""
        retrieved = self.retrieve(query, k=k, mode=mode)

        # Immediate abstention if nothing retrieved or similarity too low
        if not retrieved or (mode == "vector" and retrieved[0].get("score", 0.0) < min_relevance):
            return {
                "status": "insufficient_evidence",
                "answer": f"I don't have sufficient recorded memory to answer '{query}'.",
                "cited_chunk_ids": [],
                "retrieved_chunks": retrieved
            }

        # Build context prompt
        context_lines = []
        for c in retrieved:
            context_lines.append(f"[{c['chunk_id']}] (Session: {c['session_id']}, Date: {c['date']}):\n{c['text']}\n")
        context_str = "\n".join(context_lines)

        user_prompt = f"""<evidence_context>
{context_str}
</evidence_context>

User Question: {query}

Please provide an evidence-grounded answer based strictly on the provided context."""

        if self.llm_caller is not None:
            raw = self.llm_caller(user_prompt, RAG_SYSTEM_PROMPT)
            if isinstance(raw, dict):
                return {
                    "status": raw.get("status", "answered"),
                    "answer": raw.get("answer", ""),
                    "cited_chunk_ids": raw.get("cited_memory_ids", raw.get("cited_chunk_ids", [])),
                    "retrieved_chunks": retrieved
                }
            elif isinstance(raw, str):
                return {
                    "status": "answered",
                    "answer": raw,
                    "cited_chunk_ids": [c["chunk_id"] for c in retrieved[:2]],
                    "retrieved_chunks": retrieved
                }

        if ollama_chat is not None:
            try:
                schema = {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["answered", "partial", "insufficient_evidence"]},
                        "answer": {"type": "string"},
                        "cited_chunk_ids": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["status", "answer", "cited_chunk_ids"]
                }
                resp = ollama_chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": RAG_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    format=schema,
                    options={"temperature": 0.0, "num_ctx": 4096, "num_predict": 512}
                )
                payload = json.loads(resp.message.content)
                payload["retrieved_chunks"] = retrieved
                return payload
            except Exception as e:
                return {
                    "status": "error",
                    "answer": f"LLM error: {e}",
                    "cited_chunk_ids": [],
                    "retrieved_chunks": retrieved
                }

        return {
            "status": "answered",
            "answer": f"Extracted from {len(retrieved)} retrieved chunks.",
            "cited_chunk_ids": [c["chunk_id"] for c in retrieved],
            "retrieved_chunks": retrieved
        }

