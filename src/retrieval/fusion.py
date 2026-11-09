"""Reciprocal Rank Fusion: merge several rankings using only ranks, never raw scores.

fused_score(d) = sum over systems of 1 / (k_const + rank_of_d_in_that_system)   (rank starts at 1)
BM25 scores and cosine similarities live on different scales, so combining ranks is the
robust, parameter-light choice. k_const=60 is the value from the original RRF paper.
"""
import numpy as np


def rrf_single(rankings, k_const: int = 60, top_k: int = 100):
    """rankings: list of 1-D arrays of doc ids (best first), one per system, for ONE query."""
    fused = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            if doc < 0:  # padding from ANN indexes
                continue
            fused[int(doc)] = fused.get(int(doc), 0.0) + 1.0 / (k_const + rank)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))  # tie-break by id: deterministic
    ids = np.full(top_k, -1, dtype=np.int64)
    for i, (doc, _) in enumerate(ordered[:top_k]):
        ids[i] = doc
    return ids


def rrf_batch(systems, k_const: int = 60, top_k: int = 100):
    """systems: list of (Q, K) id matrices, one per retrieval system. Returns (Q, top_k) ids."""
    n_queries = systems[0].shape[0]
    return np.vstack([rrf_single([s[q] for s in systems], k_const, top_k) for q in range(n_queries)])