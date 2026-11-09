import numpy as np

from src.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_against_truth, recall_at_k


def test_mrr_first_hit_rank():
    ranked = np.array([[5, 7, 9], [1, 2, 3]])
    relevant = [{7}, {42}]
    assert mrr_at_k(ranked, relevant, k=3) == (1 / 2 + 0) / 2


def test_mrr_respects_cutoff():
    ranked = np.array([[1, 2, 3, 4]])
    assert mrr_at_k(ranked, [{4}], k=3) == 0.0
    assert mrr_at_k(ranked, [{4}], k=4) == 0.25


def test_recall_partial():
    ranked = np.array([[1, 2, 3]])
    assert recall_at_k(ranked, [{1, 3, 99}], k=3) == 2 / 3


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k(np.array([[1, 2]]), [{1}], k=2) == 1.0
    assert ndcg_at_k(np.array([[3, 4]]), [{1}], k=2) == 0.0


def test_ndcg_hit_at_rank_two():
    got = ndcg_at_k(np.array([[9, 1]]), [{1}], k=2)
    assert abs(got - 1 / np.log2(3)) < 1e-9


def test_recall_against_truth():
    approx = np.array([[1, 2, 3]])
    truth = np.array([[1, 2, 4]])
    assert abs(recall_against_truth(approx, truth) - 2 / 3) < 1e-9
