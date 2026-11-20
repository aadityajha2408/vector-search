"""Evaluate every retrieval system on the subset and write results/results.json.

    python -m src.evaluation.evaluate --k 100
    python -m src.evaluation.evaluate --k 100 --with-hnsw        # also builds our HNSW (~1-2 min)

Dense / hybrid rows appear only if embeddings exist (python -m src.embeddings.encode).

Hybrid weighting is TUNED on the first half of the queries and REPORTED on the second half,
so the hybrid number is not inflated by tuning on the same queries it is scored on.
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


def print_table(title, table):
    print(f"\n{title}")
    cols = list(next(iter(table.values())))
    print(f"{'system':44s}" + "".join(f"{c:>12s}" for c in cols))
    for name, row in table.items():
        print(f"{name:44s}" + "".join(f"{row[c]:>12}" for c in cols))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=100, help="depth of retrieved lists")
    p.add_argument("--backend", default="auto", choices=["auto", "cpp", "numpy"])
    p.add_argument("--dense-weights", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0, 8.0],
                   help="candidate weights for the dense ranking in hybrid RRF (BM25 weight is 1)")
    p.add_argument("--with-hnsw", action="store_true", help="also evaluate our HNSW index end to end")
    p.add_argument("--hnsw-ef", type=int, nargs="+", default=[100, 200, 400], help="must be >= --k to be useful")
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    args = p.parse_args()

    pids, texts = read_corpus(PROCESSED_DIR / CORPUS_FILE)
    qids, qtexts = read_queries(PROCESSED_DIR / QUERIES_FILE)
    qrels = read_qrels(PROCESSED_DIR / QRELS_FILE)
    relevant = [qrels[q] for q in qids]
    n_queries = len(qids)
    half = n_queries // 2
    tune, test = slice(0, half), slice(half, None)

    def to_pids(row_ids):
        return np.where(row_ids >= 0, pids[np.clip(row_ids, 0, None)], -1)  # rows -> pids

    def scores(row_ids, sl=slice(None)):
        ranked, rel = to_pids(row_ids[sl]), relevant[sl]
        return {
            "MRR@10": round(mrr_at_k(ranked, rel, 10), 4),
            "nDCG@10": round(ndcg_at_k(ranked, rel, 10), 4),
            "Recall@10": round(recall_at_k(ranked, rel, 10), 4),
            f"Recall@{args.k}": round(recall_at_k(ranked, rel, args.k), 4),
        }

    full, held_out = {}, {}

    def record(name, row_ids, ms_per_query):
        full[name] = {**scores(row_ids), "ms/query": round(ms_per_query, 3)}

    t = time.perf_counter()
    bm25 = BM25().fit(texts)
    bm25_ids, _ = bm25.search(qtexts, args.k)
    record("BM25", bm25_ids, 1000 * (time.perf_counter() - t) / n_queries)

    t = time.perf_counter()
    tfidf = TfIdf().fit(texts)
    tfidf_ids, _ = tfidf.search(qtexts, args.k)
    record("TF-IDF", tfidf_ids, 1000 * (time.perf_counter() - t) / n_queries)

    corpus_emb_path, query_emb_path = PROCESSED_DIR / CORPUS_EMB_FILE, PROCESSED_DIR / QUERY_EMB_FILE
    if corpus_emb_path.exists() and query_emb_path.exists():
        corpus_emb, query_emb = np.load(corpus_emb_path), np.load(query_emb_path)

        index = DenseIndex(args.backend).add(corpus_emb)
        t = time.perf_counter()
        dense_ids, _ = index.search(query_emb, args.k)
        record(f"Dense exact ({index.backend})", dense_ids, 1000 * (time.perf_counter() - t) / n_queries)
        record("Hybrid RRF, equal weights", rrf_batch([bm25_ids, dense_ids], top_k=args.k), 0.0)

        if args.with_hnsw:
            from src.retrieval.ann import OurHNSW
            t = time.perf_counter()
            hnsw = OurHNSW(corpus_emb.shape[1], M=args.M, ef_construction=args.ef_construction).build(corpus_emb)
            print(f"HNSW build: {time.perf_counter() - t:.1f}s")
            for ef in args.hnsw_ef:
                t = time.perf_counter()
                ids, _ = hnsw.search(query_emb, args.k, ef, 1)
                record(f"HNSW ours (ef={ef}, 1 thread)", ids, 1000 * (time.perf_counter() - t) / n_queries)

        # Tune the dense weight on the first half of the queries only ...
        sweep = {}
        for w in args.dense_weights:
            fused = rrf_batch([bm25_ids, dense_ids], top_k=args.k, weights=[1.0, w])
            sweep[w] = scores(fused, tune)["MRR@10"]
        best_w = max(sweep, key=sweep.get)
        print("\nHybrid weight sweep (MRR@10 on the TUNING half, first %d queries):" % half)
        for w, m in sweep.items():
            print(f"  dense weight {w:>4}: {m}{'   <- best' if w == best_w else ''}")

        # ... and report all systems on the untouched second half.
        weighted = rrf_batch([bm25_ids, dense_ids], top_k=args.k, weights=[1.0, best_w])
        for name, ids in [("BM25", bm25_ids), ("TF-IDF", tfidf_ids), ("Dense exact", dense_ids),
                          ("Hybrid RRF, equal weights", rrf_batch([bm25_ids, dense_ids], top_k=args.k)),
                          (f"Hybrid RRF, dense weight {best_w}", weighted)]:
            held_out[name] = scores(ids, test)
    else:
        print("(no embeddings found: skipping dense and hybrid rows)")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(
        json.dumps({"all_queries": full, "held_out_half": held_out}, indent=2), encoding="utf-8")

    print_table(f"All {n_queries} queries", full)
    if held_out:
        print_table(f"Held-out half (last {n_queries - half} queries; hybrid weight tuned on the other half)", held_out)
    print("\nNote: ms/query for sparse systems includes index build time; hybrid reports 0 (fusion only).")


if __name__ == "__main__":
    main()