"""Evaluate every retrieval system on the subset and write results/results.json.

    python -m src.evaluation.evaluate --k 100
Dense / hybrid rows appear only if embeddings exist (python -m src.embeddings.encode).
"""
import argparse
import json
import time

import numpy as np

from src.config import (CORPUS_EMB_FILE, CORPUS_FILE, PROCESSED_DIR, QRELS_FILE, QUERIES_FILE,
                        QUERY_EMB_FILE, RESULTS_DIR)
from src.data_io import read_corpus, read_qrels, read_queries
from src.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.retrieval.bm25 import BM25
from src.retrieval.dense import DenseIndex
from src.retrieval.fusion import rrf_batch
from src.retrieval.tfidf import TfIdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=100, help="depth of retrieved lists")
    p.add_argument("--backend", default="auto", choices=["auto", "cpp", "numpy"])
    args = p.parse_args()

    pids, texts = read_corpus(PROCESSED_DIR / CORPUS_FILE)
    qids, qtexts = read_queries(PROCESSED_DIR / QUERIES_FILE)
    qrels = read_qrels(PROCESSED_DIR / QRELS_FILE)
    relevant = [qrels[q] for q in qids]
    n_queries = len(qids)

    results = {}

    def record(name, row_ids, seconds):
        ranked = np.where(row_ids >= 0, pids[np.clip(row_ids, 0, None)], -1)  # rows -> pids
        results[name] = {
            "MRR@10": round(mrr_at_k(ranked, relevant, 10), 4),
            "nDCG@10": round(ndcg_at_k(ranked, relevant, 10), 4),
            "Recall@10": round(recall_at_k(ranked, relevant, 10), 4),
            f"Recall@{args.k}": round(recall_at_k(ranked, relevant, args.k), 4),
            "ms/query": round(1000 * seconds / n_queries, 3),
        }

    t = time.perf_counter()
    bm25 = BM25().fit(texts)
    bm25_ids, _ = bm25.search(qtexts, args.k)
    record("BM25", bm25_ids, time.perf_counter() - t)

    t = time.perf_counter()
    tfidf = TfIdf().fit(texts)
    tfidf_ids, _ = tfidf.search(qtexts, args.k)
    record("TF-IDF", tfidf_ids, time.perf_counter() - t)

    corpus_emb_path = PROCESSED_DIR / CORPUS_EMB_FILE
    query_emb_path = PROCESSED_DIR / QUERY_EMB_FILE
    if corpus_emb_path.exists() and query_emb_path.exists():
        corpus_emb = np.load(corpus_emb_path)
        query_emb = np.load(query_emb_path)
        index = DenseIndex(args.backend).add(corpus_emb)
        t = time.perf_counter()
        dense_ids, _ = index.search(query_emb, args.k)
        record(f"Dense exact ({index.backend})", dense_ids, time.perf_counter() - t)

        hybrid_ids = rrf_batch([bm25_ids, dense_ids], top_k=args.k)
        record("Hybrid RRF (BM25 + dense)", hybrid_ids, 0.0)
    else:
        print("(no embeddings found: skipping dense and hybrid rows)")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    cols = list(next(iter(results.values())))
    print(f"{'system':32s}" + "".join(f"{c:>12s}" for c in cols))
    for name, row in results.items():
        print(f"{name:32s}" + "".join(f"{row[c]:>12}" for c in cols))
    print("\nNote: ms/query for sparse systems includes index build time; hybrid reports 0 (fusion only).")


if __name__ == "__main__":
    main()