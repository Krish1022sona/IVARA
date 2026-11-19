"""
IVARA Baseline Retrieval and Reranking Experiment
Strictly local, session-grouped 5-fold cross-validation on 358 clean development questions.
Preserves all historical runs, frozen h40, and d100.
"""

import json
import math
import time
import os
import sys
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from scipy import stats
import faiss

from src.embedding import model
from src.chunking.turn_chunker import TurnAwareChunker
from src.chunking.retrieval import ChunkRetriever, BM25ChunkSearcher

OUTPUT_DIR = Path("evaluation/baseline_reranker_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = OUTPUT_DIR / "cached_candidates.pkl"

GLOBAL_EMB_CACHE = {}

class TurnObj:
    def __init__(self, r, c):
        self.role = r
        self.content = c

class SessObj:
    def __init__(self, sid, dt, turns):
        self.session_id = sid
        self.date = dt
        self.turns = [TurnObj(t.get("role", ""), t.get("content", "")) for t in turns]
        self.parsed_datetime = None

class FastChunkRetriever(ChunkRetriever):
    """
    Subclass that uses batched tensor encoding for faster local CPU evaluation.
    Mathematically identical embeddings and FAISS index as ChunkRetriever.
    """
    def _build_indexes(self):
        texts = [c.text for c in self.chunks]
        if texts:
            uncached = [t for t in texts if t not in GLOBAL_EMB_CACHE]
            if uncached:
                new_embs = model.encode(uncached, batch_size=256, show_progress_bar=False)
                for t, e in zip(uncached, new_embs):
                    GLOBAL_EMB_CACHE[t] = e
            embeddings = np.array([GLOBAL_EMB_CACHE[t] for t in texts], dtype="float32")
            faiss.normalize_L2(embeddings)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
        self.bm25 = BM25ChunkSearcher(self.chunks)

def load_data_and_audit():
    p_cleaned = Path("evaluation/data/longmemeval/longmemeval_s_cleaned.json")
    with open(p_cleaned, "r", encoding="utf-8") as f:
        s_cleaned = json.load(f)

    with open("evaluation/autonomous_sprint/runs/finalization_sprint/fresh_heldout_40.json", "r", encoding="utf-8") as f:
        h40_qids = set(q["question_id"] for q in json.load(f))
    with open("evaluation/autonomous_sprint/diagnostic_sample_100.json", "r", encoding="utf-8") as f:
        d100_qids = set(q["question_id"] for q in json.load(f))

    quarantine = {"07741c44", "71017276"}
    excluded = h40_qids | d100_qids | quarantine
    dev_pool = [q for q in s_cleaned if q["question_id"] not in excluded and len(q.get("answer_session_ids", [])) > 0]

    print(f"[AUDIT] Raw s_cleaned: {len(s_cleaned)} questions")
    print(f"[AUDIT] Preserved h40: {len(h40_qids)} questions")
    print(f"[AUDIT] Isolated d100: {len(d100_qids)} questions")
    print(f"[AUDIT] Quarantined leakage: {len(quarantine)} questions")
    print(f"[AUDIT] Clean Development Pool: {len(dev_pool)} questions")

    # Union-Find for answer session grouping
    parent = {q["question_id"]: q["question_id"] for q in dev_pool}
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    ans_to_qids = defaultdict(list)
    for q in dev_pool:
        for ans_sid in q.get("answer_session_ids", []):
            ans_to_qids[ans_sid].append(q["question_id"])

    for ans_sid, qids in ans_to_qids.items():
        for qid in qids[1:]:
            union(qids[0], qid)

    groups = defaultdict(list)
    for q in dev_pool:
        groups[find(q["question_id"])].append(q)

    sorted_groups = sorted(groups.values(), key=lambda g: len(g), reverse=True)
    folds = [[] for _ in range(5)]
    fold_counts = [0] * 5

    for g in sorted_groups:
        min_idx = fold_counts.index(min(fold_counts))
        folds[min_idx].extend(g)
        fold_counts[min_idx] += len(g)

    # Verify disjointness
    for i in range(5):
        q_i = set(q["question_id"] for q in folds[i])
        ans_i = set(s for q in folds[i] for s in q.get("answer_session_ids", []))
        for j in range(i + 1, 5):
            q_j = set(q["question_id"] for q in folds[j])
            ans_j = set(s for q in folds[j] for s in q.get("answer_session_ids", []))
            assert len(q_i & q_j) == 0, f"QID overlap Fold {i} vs {j}"
            assert len(ans_i & ans_j) == 0, f"Answer session overlap Fold {i} vs {j}"

    print(f"[AUDIT] 5-Fold Partition complete: {fold_counts}")
    print("[AUDIT] Verified 100% question-ID disjoint and answer-session disjoint across all folds.")
    return dev_pool, folds

def extract_question_candidates(q_item, chunker):
    sids = q_item["haystack_session_ids"]
    dts = q_item.get("haystack_dates", [""] * len(sids))
    sess_turns = q_item["haystack_sessions"]
    sessions = [SessObj(sid, dt, t) for sid, dt, t in zip(sids, dts, sess_turns)]
    chunks = chunker.chunk_all_sessions(sessions)
    retriever = FastChunkRetriever(chunks=chunks, vector_weight=0.55, bm25_weight=0.30, ontology_weight=0.15, use_ontology=True)
    cands = retriever.retrieve(q_item["question"], k=25, min_score=None)

    gold_sids = set(q_item["answer_session_ids"])
    all_gold_chunk_ids = set(c.chunk_id for c in chunks if c.session_id in gold_sids)

    feats = []
    labels = []
    for c in cands:
        is_gold = 1 if c["session_id"] in gold_sids else 0
        f = [
            float(c.get("vector_score", 0.0)),
            float(c.get("bm25_score", 0.0)),
            float(c.get("ontology_score", 0.0)),
            float(len(c.get("text", "").split())),
            float(c.get("turn_range", (0, 0))[0])
        ]
        feats.append(f)
        labels.append(is_gold)

    return {
        "question_id": q_item["question_id"],
        "cands": cands,
        "feats": np.array(feats, dtype="float32"),
        "labels": np.array(labels, dtype="int32"),
        "gold_sids": gold_sids,
        "all_gold_chunk_ids": all_gold_chunk_ids,
        "total_chunks": len(chunks)
    }

def evaluate_cand_ranking(ranked_cands_with_scores, gold_sids, all_gold_chunk_ids):
    sorted_chunks = sorted(ranked_cands_with_scores, key=lambda x: -x[1])
    n_gold_chunks = len(all_gold_chunk_ids)

    # Chunk metrics
    top5_c = sorted_chunks[:5]
    top10_c = sorted_chunks[:10]
    c_rec5 = sum(1 for c, s in top5_c if c["chunk_id"] in all_gold_chunk_ids) / max(1, n_gold_chunks)
    c_rec10 = sum(1 for c, s in top10_c if c["chunk_id"] in all_gold_chunk_ids) / max(1, n_gold_chunks)

    c_mrr = 0.0
    for r, (c, s) in enumerate(sorted_chunks, 1):
        if c["chunk_id"] in all_gold_chunk_ids:
            c_mrr = 1.0 / r
            break

    # Session metrics (deduplicated by max chunk score per session)
    sess_scores = {}
    for c, s in sorted_chunks:
        sid = c["session_id"]
        if sid not in sess_scores or s > sess_scores[sid]:
            sess_scores[sid] = s

    ranked_sessions = sorted(sess_scores.keys(), key=lambda sid: -sess_scores[sid])
    top5_s = set(ranked_sessions[:5])
    top10_s = set(ranked_sessions[:10])

    s_rec5 = len(top5_s & gold_sids) / len(gold_sids)
    s_rec10 = len(top10_s & gold_sids) / len(gold_sids)

    s_mrr = 0.0
    for r, sid in enumerate(ranked_sessions, 1):
        if sid in gold_sids:
            s_mrr = 1.0 / r
            break

    dcg10 = 0.0
    k_eval = min(10, len(ranked_sessions))
    for r in range(1, k_eval + 1):
        rel = 1 if ranked_sessions[r - 1] in gold_sids else 0
        dcg10 += rel / math.log2(r + 1)
    idcg10 = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold_sids), 10) + 1))
    s_ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0

    return {
        "c_rec5": c_rec5, "c_rec10": c_rec10, "c_mrr": c_mrr,
        "s_rec5": s_rec5, "s_rec10": s_rec10, "s_mrr": s_mrr, "s_ndcg10": s_ndcg10
    }

def bootstrap_ci(diffs, n_bootstrap=1000, alpha=0.05):
    means = []
    n = len(diffs)
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return lower, upper

def main():
    t_start = time.time()
    print("=== STARTING IVARA BASELINE 5-FOLD CV EXPERIMENT ===")
    dev_pool, folds = load_data_and_audit()

    chunker = TurnAwareChunker(window_size=2, overlap=1)

    print("\n[PHASE 1] Extracting Top-25 Candidates across all 358 questions...")
    t_ext0 = time.time()
    cached_data = {}
    ceiling_hits = 0
    total_gold_sessions = 0

    if CACHE_FILE.exists():
        print(f"\n[PHASE 1] Loading pre-extracted candidates from {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            cached_data = pickle.load(f)
        for q in dev_pool:
            data = cached_data[q["question_id"]]
            cands_sids = set(c["session_id"] for c in data["cands"])
            g_sids = data["gold_sids"]
            ceiling_hits += len(cands_sids & g_sids)
            total_gold_sessions += len(g_sids)
        print(f"  Loaded {len(cached_data)} questions from disk cache in {time.time() - t_ext0:.2f}s")
    else:
        for idx, q in enumerate(dev_pool):
            data = extract_question_candidates(q, chunker)
            cached_data[q["question_id"]] = data

            # Candidate ceiling check: is gold session in top-25 candidate pool?
            cands_sids = set(c["session_id"] for c in data["cands"])
            g_sids = data["gold_sids"]
            ceiling_hits += len(cands_sids & g_sids)
            total_gold_sessions += len(g_sids)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(dev_pool):
                print(f"  Processed {idx + 1}/{len(dev_pool)} questions in {time.time() - t_ext0:.1f}s")

        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cached_data, f)
        print(f"  Saved candidate cache to {CACHE_FILE}")

    cand_ceiling_recall = (ceiling_hits / total_gold_sessions) * 100.0
    print(f"\n[CEILING] Top-25 Candidate Gold-Session Recall Ceiling: {cand_ceiling_recall:.2f}% ({ceiling_hits}/{total_gold_sessions} gold sessions)")

    # Run 5-fold cross validation
    methods = [
        "Dense_Only",
        "BM25_Only",
        "Ontology_Only",
        "Method_A_Current_Fusion",
        "Method_B1_Logistic_Regression"
    ]

    oof_results = {m: [] for m in methods}
    fold_metrics = {m: [] for m in methods}

    print("\n[PHASE 2] Running 5-Fold Cross-Validation...")

    for fold_idx in range(5):
        val_fold = folds[fold_idx]
        train_fold = []
        for other_idx in range(5):
            if other_idx != fold_idx:
                train_fold.extend(folds[other_idx])

        # Train fold-specific Logistic Regression
        X_tr_list, y_tr_list = [], []
        for q in train_fold:
            d = cached_data[q["question_id"]]
            if len(d["feats"]) > 0:
                X_tr_list.append(d["feats"])
                y_tr_list.append(d["labels"])
        X_train = np.vstack(X_tr_list)
        y_train = np.concatenate(y_tr_list)

        lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=200)
        lr.fit(X_train, y_train)

        # Evaluate on validation fold
        fold_eval = {m: [] for m in methods}

        for q in val_fold:
            d = cached_data[q["question_id"]]
            cands = d["cands"]
            feats = d["feats"]
            gold_sids = d["gold_sids"]
            all_gold_chunks = d["all_gold_chunk_ids"]

            if len(cands) == 0:
                # Handle edge case of empty candidates
                zero_res = {"c_rec5": 0.0, "c_rec10": 0.0, "c_mrr": 0.0, "s_rec5": 0.0, "s_rec10": 0.0, "s_mrr": 0.0, "s_ndcg10": 0.0}
                for m in methods:
                    fold_eval[m].append(zero_res)
                    oof_results[m].append(zero_res)
                continue

            # 1. Dense Only (vector_score)
            cands_dense = [(c, c.get("vector_score", 0.0)) for c in cands]
            res_dense = evaluate_cand_ranking(cands_dense, gold_sids, all_gold_chunks)
            fold_eval["Dense_Only"].append(res_dense)
            oof_results["Dense_Only"].append(res_dense)

            # 2. BM25 Only (bm25_score)
            cands_bm25 = [(c, c.get("bm25_score", 0.0)) for c in cands]
            res_bm25 = evaluate_cand_ranking(cands_bm25, gold_sids, all_gold_chunks)
            fold_eval["BM25_Only"].append(res_bm25)
            oof_results["BM25_Only"].append(res_bm25)

            # 3. Ontology Only (ontology_score)
            cands_ont = [(c, c.get("ontology_score", 0.0)) for c in cands]
            res_ont = evaluate_cand_ranking(cands_ont, gold_sids, all_gold_chunks)
            fold_eval["Ontology_Only"].append(res_ont)
            oof_results["Ontology_Only"].append(res_ont)

            # 4. Method A (Current IVARA Fused Score)
            cands_A = [(c, c.get("score", 0.0)) for c in cands]
            res_A = evaluate_cand_ranking(cands_A, gold_sids, all_gold_chunks)
            fold_eval["Method_A_Current_Fusion"].append(res_A)
            oof_results["Method_A_Current_Fusion"].append(res_A)

            # 5. Method B1 (Logistic Regression Reranker)
            probs = lr.predict_proba(feats)[:, 1]
            cands_B1 = [(c, probs[i]) for i, c in enumerate(cands)]
            res_B1 = evaluate_cand_ranking(cands_B1, gold_sids, all_gold_chunks)
            fold_eval["Method_B1_Logistic_Regression"].append(res_B1)
            oof_results["Method_B1_Logistic_Regression"].append(res_B1)

        print(f"  Fold {fold_idx} ({len(val_fold)} Qs): LR Coefs={np.round(lr.coef_[0], 3)}, Intercept={lr.intercept_[0]:.3f}")
        for m in methods:
            m_avg = {k: np.mean([r[k] for r in fold_eval[m]]) for k in fold_eval[m][0].keys()}
            fold_metrics[m].append(m_avg)

    # Aggregate Analysis
    keys = ["s_rec5", "s_rec10", "s_mrr", "s_ndcg10", "c_rec5", "c_rec10", "c_mrr"]
    print("\n" + "="*80)
    print("AGGREGATE OUT-OF-FOLD RESULTS (N=358 DEVELOPMENT QUESTIONS)")
    print("="*80)
    summary_table = {}

    for m in methods:
        summary_table[m] = {k: float(np.mean([r[k] for r in oof_results[m]])) for k in keys}
        s = summary_table[m]
        print(f"\nModel: {m}")
        print(f"  Session Recall@5:  {s['s_rec5']*100:.2f}%")
        print(f"  Session Recall@10: {s['s_rec10']*100:.2f}%")
        print(f"  Session MRR:       {s['s_mrr']:.4f}")
        print(f"  Session nDCG@10:   {s['s_ndcg10']:.4f}")
        print(f"  Chunk Recall@5:    {s['c_rec5']*100:.2f}% (Diagnostic)")
        print(f"  Chunk Recall@10:   {s['c_rec10']*100:.2f}% (Diagnostic)")
        print(f"  Chunk MRR:         {s['c_mrr']:.4f} (Diagnostic)")

    # Statistical Significance & Bootstrap CI: Method B1 vs Method A
    diff_s_rec5 = [b["s_rec5"] - a["s_rec5"] for a, b in zip(oof_results["Method_A_Current_Fusion"], oof_results["Method_B1_Logistic_Regression"])]
    diff_s_rec10 = [b["s_rec10"] - a["s_rec10"] for a, b in zip(oof_results["Method_A_Current_Fusion"], oof_results["Method_B1_Logistic_Regression"])]
    diff_s_mrr = [b["s_mrr"] - a["s_mrr"] for a, b in zip(oof_results["Method_A_Current_Fusion"], oof_results["Method_B1_Logistic_Regression"])]
    diff_s_ndcg10 = [b["s_ndcg10"] - a["s_ndcg10"] for a, b in zip(oof_results["Method_A_Current_Fusion"], oof_results["Method_B1_Logistic_Regression"])]

    ci_rec5 = bootstrap_ci(diff_s_rec5)
    ci_rec10 = bootstrap_ci(diff_s_rec10)
    ci_mrr = bootstrap_ci(diff_s_mrr)
    ci_ndcg10 = bootstrap_ci(diff_s_ndcg10)

    # Paired Wilcoxon Signed-Rank Test
    w_stat_mrr, p_val_mrr = stats.wilcoxon(diff_s_mrr, zero_method="pratt") if any(d != 0 for d in diff_s_mrr) else (0.0, 1.0)
    w_stat_ndcg, p_val_ndcg = stats.wilcoxon(diff_s_ndcg10, zero_method="pratt") if any(d != 0 for d in diff_s_ndcg10) else (0.0, 1.0)

    print("\n" + "="*80)
    print("PAIRED COMPARISON: Method B1 (Logistic Regression) vs Method A (Current Fusion)")
    print("="*80)
    print(f"Delta Session Recall@5:  {np.mean(diff_s_rec5)*100:+.2f}% | 95% Bootstrap CI: [{ci_rec5[0]*100:+.2f}%, {ci_rec5[1]*100:+.2f}%]")
    print(f"Delta Session Recall@10: {np.mean(diff_s_rec10)*100:+.2f}% | 95% Bootstrap CI: [{ci_rec10[0]*100:+.2f}%, {ci_rec10[1]*100:+.2f}%]")
    print(f"Delta Session MRR:       {np.mean(diff_s_mrr):+.4f} | 95% Bootstrap CI: [{ci_mrr[0]:+.4f}, {ci_mrr[1]:+.4f}] | Wilcoxon p-value = {p_val_mrr:.4e}")
    print(f"Delta Session nDCG@10:   {np.mean(diff_s_ndcg10):+.4f} | 95% Bootstrap CI: [{ci_ndcg10[0]:+.4f}, {ci_ndcg10[1]:+.4f}] | Wilcoxon p-value = {p_val_ndcg:.4e}")

    # Per-question diff counts
    better_mrr = sum(1 for d in diff_s_mrr if d > 0)
    worse_mrr = sum(1 for d in diff_s_mrr if d < 0)
    tied_mrr = sum(1 for d in diff_s_mrr if d == 0)
    print(f"\nPer-Question MRR Breakdown: {better_mrr} Better, {worse_mrr} Worse, {tied_mrr} Tied (out of {len(dev_pool)})")

    # Save complete artifacts
    total_time = time.time() - t_start
    full_output = {
        "metadata": {
            "total_questions": len(dev_pool),
            "fold_sizes": [len(f) for f in folds],
            "candidate_pool_size": 25,
            "candidate_gold_session_ceiling": cand_ceiling_recall,
            "runtime_seconds": round(total_time, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "summary": summary_table,
        "paired_stats": {
            "delta_recall5": {"mean": float(np.mean(diff_s_rec5)), "ci95": [float(ci_rec5[0]), float(ci_rec5[1])]},
            "delta_recall10": {"mean": float(np.mean(diff_s_rec10)), "ci95": [float(ci_rec10[0]), float(ci_rec10[1])]},
            "delta_mrr": {"mean": float(np.mean(diff_s_mrr)), "ci95": [float(ci_mrr[0]), float(ci_mrr[1])], "wilcoxon_p": float(p_val_mrr)},
            "delta_ndcg10": {"mean": float(np.mean(diff_s_ndcg10)), "ci95": [float(ci_ndcg10[0]), float(ci_ndcg10[1])], "wilcoxon_p": float(p_val_ndcg)},
            "mrr_breakdown": {"better": better_mrr, "worse": worse_mrr, "tied": tied_mrr}
        },
        "per_fold": {
            fold_idx: {m: fold_metrics[m][fold_idx] for m in methods} for fold_idx in range(5)
        }
    }

    out_json = OUTPUT_DIR / "baseline_cv_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[ARTIFACT] Complete results written to {out_json}")
    print(f"[RUNTIME] Total experiment runtime: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()
