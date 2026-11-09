"""Correctness of the C++ indexes against brute-force NumPy. Skipped if the extension isn't built."""
import numpy as np
import pytest

from src.evaluation.metrics import recall_against_truth
from src.retrieval.utils import topk_from_scores

vs = pytest.importorskip("src.vecsearch_cpp", reason="build with: python setup.py build_ext --inplace")


def normalized(n, d, seed):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_exact_matches_numpy():
    X, Q = normalized(2000, 48, 0), normalized(37, 48, 1)  # 37 is not a multiple of the tile size
    index = vs.ExactIndex(48)
    index.add(X)
    ids, scores = index.search(Q, 10, 1)
    ref_ids, ref_scores = topk_from_scores(Q @ X.T, 10)
    assert np.array_equal(ids, ref_ids)
    assert np.allclose(scores, ref_scores, atol=1e-5)


def test_exact_multithread_matches_single_thread():
    X, Q = normalized(1500, 32, 2), normalized(50, 32, 3)
    index = vs.ExactIndex(32)
    index.add(X)
    assert np.array_equal(index.search(Q, 5, 1)[0], index.search(Q, 5, 4)[0])


def test_exact_k_larger_than_corpus_pads_with_minus_one():
    index = vs.ExactIndex(8)
    index.add(normalized(3, 8, 4))
    ids, _ = index.search(normalized(1, 8, 5), 5, 1)
    assert (ids[0, 3:] == -1).all() and (ids[0, :3] >= 0).all()


def test_exact_rejects_bad_input():
    index = vs.ExactIndex(8)
    with pytest.raises(ValueError):
        index.add(np.zeros((2, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        index.search(np.zeros((1, 8), dtype=np.float32), 0, 1)


def test_hnsw_recall_is_high_and_improves_with_ef():
    X, Q = normalized(5000, 32, 6), normalized(100, 32, 7)
    truth, _ = topk_from_scores(Q @ X.T, 10)
    index = vs.HNSWIndex(32, 16, 100, 42, True)
    index.add(X)
    low, _ = index.search(Q, 10, 10, 1)
    high, _ = index.search(Q, 10, 200, 1)
    r_low, r_high = recall_against_truth(low, truth), recall_against_truth(high, truth)
    assert r_high >= r_low
    assert r_high > 0.9


def test_hnsw_results_sorted_and_unique():
    X, Q = normalized(1000, 16, 8), normalized(5, 16, 9)
    index = vs.HNSWIndex(16, 8, 50, 1, True)
    index.add(X)
    ids, scores = index.search(Q, 10, 50, 1)
    for row_ids, row_scores in zip(ids, scores):
        assert len(set(row_ids.tolist())) == 10
        assert np.all(np.diff(row_scores) <= 1e-6)


def test_hnsw_same_seed_same_graph():
    X, Q = normalized(800, 16, 10), normalized(10, 16, 11)
    a, b = vs.HNSWIndex(16, 8, 50, 7, True), vs.HNSWIndex(16, 8, 50, 7, True)
    a.add(X)
    b.add(X)
    assert np.array_equal(a.search(Q, 5, 30, 1)[0], b.search(Q, 5, 30, 1)[0])


def test_hnsw_empty_index_returns_padding():
    ids, _ = vs.HNSWIndex(8).search(np.zeros((1, 8), dtype=np.float32), 3, 10, 1)
    assert (ids == -1).all()