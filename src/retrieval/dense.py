"""Exact (brute-force) dense retrieval with two interchangeable backends.

  - "numpy": chunked matrix multiply + argpartition (always available, BLAS-accelerated)
  - "cpp"  : our C++ index (src/vecsearch_cpp), heap-based top-k with query tiling + OpenMP

Vectors must be L2-normalised, so inner product == cosine similarity.
"""
import numpy as np

from src.retrieval.utils import topk_from_scores

try:
    from src import vecsearch_cpp
except ImportError:  # extension not built yet
    vecsearch_cpp = None


def cpp_available() -> bool:
    return vecsearch_cpp is not None


class DenseIndex:
    def __init__(self, backend: str = "auto"):
        if backend == "auto":
            backend = "cpp" if cpp_available() else "numpy"
        if backend == "cpp" and not cpp_available():
            raise RuntimeError("C++ extension not built. Run: python setup.py build_ext --inplace")
        if backend not in ("cpp", "numpy"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self._cpp = None
        self._matrix = None

    def add(self, vectors: np.ndarray):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.backend == "cpp":
            if self._cpp is None:
                self._cpp = vecsearch_cpp.ExactIndex(vectors.shape[1])
            self._cpp.add(vectors)
        else:
            self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        return self

    def search(self, queries: np.ndarray, k: int = 10, num_threads: int = 0, chunk: int = 256):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self.backend == "cpp":
            return self._cpp.search(queries, k, num_threads)
        ids, vals = [], []
        for start in range(0, len(queries), chunk):
            scores = queries[start:start + chunk] @ self._matrix.T
            i, v = topk_from_scores(scores, k)
            ids.append(i)
            vals.append(v)
        return np.vstack(ids), np.vstack(vals)