"""Approximate nearest-neighbour indexes behind one tiny interface: build() / search().

  - OurHNSW      : our C++ implementation (src/vecsearch_cpp.HNSWIndex)
  - HnswlibHNSW  : reference implementation (pip install hnswlib)
  - FaissHNSW    : reference implementation (pip install faiss-cpu)

All use inner product on L2-normalised vectors. search() returns (ids, scores), best first.
"""
import numpy as np


class OurHNSW:
    name = "ours"

    def __init__(self, dim, M=16, ef_construction=200, seed=42, use_heuristic=True):
        from src import vecsearch_cpp
        self.index = vecsearch_cpp.HNSWIndex(dim, M, ef_construction, seed, use_heuristic)

    def build(self, vectors):
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return self

    def search(self, queries, k, ef, num_threads=1):
        return self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k, ef, num_threads)


class HnswlibHNSW:
    name = "hnswlib"

    def __init__(self, dim, M=16, ef_construction=200, seed=42, **_):
        import hnswlib
        self.dim, self.M, self.efc, self.seed = dim, M, ef_construction, seed
        self.index = hnswlib.Index(space="ip", dim=dim)

    def build(self, vectors):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index.init_index(max_elements=len(vectors), ef_construction=self.efc, M=self.M, random_seed=self.seed)
        self.index.set_num_threads(1)  # single-threaded build, like ours
        self.index.add_items(vectors, np.arange(len(vectors)))
        return self

    def search(self, queries, k, ef, num_threads=1):
        self.index.set_ef(max(ef, k))
        self.index.set_num_threads(num_threads)
        labels, dists = self.index.knn_query(np.ascontiguousarray(queries, dtype=np.float32), k=k)
        return labels.astype(np.int64), (1.0 - dists).astype(np.float32)  # hnswlib 'ip' distance = 1 - dot


class FaissHNSW:
    name = "faiss"

    def __init__(self, dim, M=16, ef_construction=200, **_):
        import faiss
        self.faiss = faiss
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction

    def build(self, vectors):
        self.faiss.omp_set_num_threads(1)
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return self

    def search(self, queries, k, ef, num_threads=1):
        self.faiss.omp_set_num_threads(num_threads)
        self.index.hnsw.efSearch = max(ef, k)
        scores, ids = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return ids.astype(np.int64), scores.astype(np.float32)


def make_ann(name, dim, **kwargs):
    classes = {"ours": OurHNSW, "hnswlib": HnswlibHNSW, "faiss": FaissHNSW}
    if name not in classes:
        raise ValueError(f"unknown ANN index {name!r}; choose from {sorted(classes)}")
    return classes[name](dim, **kwargs)
