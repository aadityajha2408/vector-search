"""Retrieval metrics with binary relevance.

Inputs:
  ranked   : (Q, K) array of passage ids (pids), best first; -1 means 'no result'
  relevant : list of Q sets of relevant pids
"""
import math

import numpy as np


def mrr_at_k(ranked: np.ndarray, relevant, k: int = 10) -> float:
    total = 0.0
    for row, rel in zip(ranked, relevant):
        for rank, pid in enumerate(row[:k], start=1):
            if int(pid) in rel:
                total += 1.0 / rank
                break
    return total / len(relevant)


def recall_at_k(ranked: np.ndarray, relevant, k: int = 100) -> float:
    total = 0.0
    for row, rel in zip(ranked, relevant):
        hits = len(rel.intersection(int(p) for p in row[:k]))
        total += hits / len(rel)
    return total / len(relevant)


def ndcg_at_k(ranked: np.ndarray, relevant, k: int = 10) -> float:
    total = 0.0
    for row, rel in zip(ranked, relevant):
        dcg = sum(1.0 / math.log2(i + 2) for i, pid in enumerate(row[:k]) if int(pid) in rel)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
        total += dcg / ideal if ideal > 0 else 0.0
    return total / len(relevant)


def recall_against_truth(approx_ids: np.ndarray, true_ids: np.ndarray) -> float:
    """ANN recall@k: fraction of the exact top-k neighbours that the approximate search found."""
    hits = 0
    for a, t in zip(approx_ids, true_ids):
        hits += len(set(a.tolist()) & set(t.tolist()))
    return hits / true_ids.size