"""Invariants of the processed subset. Skipped until build_subset has been run."""
import pytest

from src.config import CORPUS_FILE, PROCESSED_DIR, QRELS_FILE, QUERIES_FILE
from src.data_io import read_corpus, read_qrels, read_queries

needs_data = pytest.mark.skipif(not (PROCESSED_DIR / CORPUS_FILE).exists(),
                                reason="run src.preprocessing.build_subset first")


@needs_data
def test_every_relevant_passage_is_in_corpus():
    pids, _ = read_corpus(PROCESSED_DIR / CORPUS_FILE)
    in_corpus = set(pids.tolist())
    qrels = read_qrels(PROCESSED_DIR / QRELS_FILE)
    missing = [(q, p) for q, rel in qrels.items() for p in rel if p not in in_corpus]
    assert not missing, f"{len(missing)} relevant passages missing, e.g. {missing[:3]}"


@needs_data
def test_corpus_sorted_unique_and_queries_have_qrels():
    pids, texts = read_corpus(PROCESSED_DIR / CORPUS_FILE)
    assert all(a < b for a, b in zip(pids[:-1], pids[1:])), "corpus must be sorted by pid, no duplicates"
    assert all(t.strip() for t in texts), "empty passage found"
    qids, qtexts = read_queries(PROCESSED_DIR / QUERIES_FILE)
    qrels = read_qrels(PROCESSED_DIR / QRELS_FILE)
    assert set(qids) == set(qrels)
    assert all(t == t.strip() and t for t in qtexts)
