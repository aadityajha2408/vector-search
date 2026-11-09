"""Recall-vs-QPS sweep for HNSW: our C++ index vs reference libraries, against exact ground truth.

    python -m benchmarks.bench_ann --indexes ours hnswlib --M 16 --ef-construction 200
Writes results/ann_bench.csv (read by benchmarks/plot.py). All searches use ONE thread so the
comparison is about the algorithm and implementation, not about core counts.
"""
import argparse
import csv
import time

import numpy as np

from benchmarks.common import load_vectors
from src.config import RESULTS_DIR
from src.evaluation.metrics import recall_against_truth
from src.retrieval.ann import make_ann
from src.retrieval.dense import DenseIndex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indexes", nargs="+", default=["ours", "hnswlib"], choices=["ours", "hnswlib", "faiss"])
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--M", type=int, default=16)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    p.add_argument("--no-heuristic", action="store_true", help="ours: use simple closest-M selection (ablation)")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    corpus, queries = load_vectors(args.n, args.dim, args.n_queries, use_real=not args.synthetic)
    queries = queries[:args.n_queries]
    dim = corpus.shape[1]

    truth, _ = DenseIndex("numpy").add(corpus).search(queries, args.k)

    rows = []
    for name in args.indexes:
        index = make_ann(name, dim, M=args.M, ef_construction=args.ef_construction,
                         **({"use_heuristic": not args.no_heuristic} if name == "ours" else {}))
        t = time.perf_counter()
        index.build(corpus)
        build_s = time.perf_counter() - t
        print(f"\n[{name}] build {build_s:.1f}s")
        print(f"{'ef':>6s}{'recall@k':>10s}{'QPS':>10s}{'ms/query':>10s}")
        for ef in args.ef:
            index.search(queries[:50], args.k, ef, 1)  # warm-up
            t = time.perf_counter()
            ids, _ = index.search(queries, args.k, ef, 1)
            secs = time.perf_counter() - t
            recall = recall_against_truth(ids, truth)
            qps = len(queries) / secs
            print(f"{ef:6d}{recall:10.4f}{qps:10.0f}{1000 / qps:10.3f}")
            rows.append({"index": name, "M": args.M, "ef_construction": args.ef_construction, "ef": ef,
                         "recall": round(recall, 4), "qps": round(qps, 1), "build_s": round(build_s, 2),
                         "heuristic": not args.no_heuristic if name == "ours" else ""})

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "ann_bench.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
