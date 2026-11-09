from src.retrieval.bm25 import BM25
from src.retrieval.tfidf import TfIdf

DOCS = [
    "the cat sat on the mat",
    "dogs are loyal animals and love to play fetch",
    "stock markets fell sharply on inflation fears",
    "a cat and a dog played together in the garden",
]


def test_bm25_ranks_matching_document_first():
    ids, scores = BM25().fit(DOCS).search(["stock market inflation"], k=2)
    assert ids[0][0] == 2
    assert scores[0][0] > scores[0][1]


def test_bm25_rare_term_beats_common_term():
    ids, _ = BM25().fit(DOCS).search(["fetch"], k=1)
    assert ids[0][0] == 1


def test_tfidf_ranks_matching_document_first():
    ids, _ = TfIdf().fit(DOCS).search(["loyal dogs"], k=1)
    assert ids[0][0] == 1