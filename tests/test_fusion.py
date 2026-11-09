import numpy as np

from src.retrieval.fusion import rrf_batch, rrf_single


def test_doc_ranked_well_by_both_systems_wins():
    a = np.array([10, 20, 30])
    b = np.array([20, 10, 40])
    fused = rrf_single([a, b], top_k=4)
    assert set(fused[:2]) == {10, 20}  # both appear in both lists at ranks 1-2
    assert set(fused[2:]) == {30, 40}


def test_padding_ignored_and_output_padded():
    fused = rrf_single([np.array([1, -1])], top_k=3)
    assert fused.tolist() == [1, -1, -1]


def test_deterministic_tie_break():
    a = np.array([5])
    b = np.array([3])
    assert rrf_single([a, b], top_k=2).tolist() == [3, 5]  # equal score -> lower id first


def test_batch_shape():
    s1 = np.array([[1, 2], [3, 4]])
    s2 = np.array([[2, 1], [4, 3]])
    assert rrf_batch([s1, s2], top_k=2).shape == (2, 2)