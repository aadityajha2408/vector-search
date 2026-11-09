"""Latency / throughput of exact search: NumPy (BLAS) vs our C++ index at several thread counts.

    python -m benchmarks.bench_exact --k 10
Report what you measure. BLAS matrix multiply is very well tuned, so the C++ index will not
automatically win in every setting: batch size and thread count change the picture.
"""
import argparse
import time

import numpy as np

from benchmarks.common import load_vectors
from src.retrieval.dense import DenseIndex, cpp_available


def best_of(fn, repeats=3):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return min(times), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50_000, help="synthetic corpus size (if no real embeddings)")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--n-queries", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--synthetic", action="store_true", help="ignore real embeddings")
    args = p.parse_args()

    corpus, queries = load_vectors(args.n, args.dim, args.n_queries, use_real=not args.synthetic)
    queries = queries[:args.n_queries]
    nq = len(queries)

    numpy_index = DenseIndex("numpy").add(corpus)
    t_ref, (ref_ids, _) = best_of(lambda: numpy_index.search(queries, args.k))
    rows = [("numpy (BLAS, batch)", t_ref, 1.0)]

    t_single, _ = best_of(lambda: [numpy_index.search(queries[i:i + 1], args.k) for i in range(min(nq, 100))])
    single_numpy_ms = 1000 * t_single / min(nq, 100)

    if cpp_available():
        cpp_index = DenseIndex("cpp").add(corpus)
        for th in args.threads:
            t, (ids, _) = best_of(lambda th=th: cpp_index.search(queries, args.k, th))
            agree = np.mean([len(set(a) & set(b)) / args.k for a, b in zip(ids, ref_ids)])
            rows.append((f"cpp, {th} thread(s)", t, agree))
        t_single_cpp, _ = best_of(lambda: [cpp_index.search(queries[i:i + 1], args.k, 1) for i in range(min(nq, 100))])
        single_cpp_ms = 1000 * t_single_cpp / min(nq, 100)
    else:
        print("C++ extension not built (python setup.py build_ext --inplace): numpy only")
        single_cpp_ms = None

    print(f"\ncorpus={len(corpus)} dim={corpus.shape[1]} queries={nq} k={args.k}")
    print(f"{'backend':26s}{'total s':>10s}{'ms/query':>11s}{'QPS':>10s}{'top-k agreement':>18s}")
    for name, t, agree in rows:
        print(f"{name:26s}{t:10.3f}{1000 * t / nq:11.3f}{nq / t:10.0f}{agree:18.4f}")
    print(f"\nsingle-query latency (batch size 1): numpy {single_numpy_ms:.2f} ms"
          + (f", cpp(1 thread) {single_cpp_ms:.2f} ms" if single_cpp_ms is not None else ""))


if __name__ == "__main__":
    main()
