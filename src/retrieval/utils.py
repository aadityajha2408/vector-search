import numpy as np


def topk_from_scores(scores: np.ndarray, k: int):
    """Row-wise top-k of a (Q, N) score matrix, best first.

    argpartition is O(N) per row; we only fully sort the k winners.
    Returns (ids int64 (Q, k), scores float32 (Q, k)).
    """
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    ids = np.take_along_axis(part, order, axis=1).astype(np.int64)
    vals = np.take_along_axis(part_scores, order, axis=1).astype(np.float32)
    return ids, vals